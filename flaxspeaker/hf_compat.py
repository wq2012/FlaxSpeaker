"""HuggingFace `transformers` and `safetensors` integration for FlaxSpeaker v0.5.0.

All 48 pretrained models (44 controlled ablation models + 4 full-data SOTA flagship
models) are hosted on the Hugging Face Hub at `https://huggingface.co/wq2012/<model_id>`
and grouped in the FlaxSpeaker collection:
https://huggingface.co/collections/wq2012/flaxspeaker

Provides:
- `FlaxSpeakerConfig(PretrainedConfig)`: HuggingFace-compatible configuration class
  registered under `model_type = "flaxspeaker"`, serializing to `config.json`.
- `FlaxSpeakerModel`: High-level model wrapper supporting `.save_pretrained()` and
  `.from_pretrained()` (writing `config.json`, `model.safetensors`,
  `flax_model.msgpack`, and a HuggingFace Model Card, including `batch_stats` for
  BatchNorm backbones), plus one-line `.extract_embedding()` and `.verify()` methods.
"""

from collections.abc import Sequence
import dataclasses
import json
import os
from typing import Any, Optional, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.flax import load_file as load_safetensors
from safetensors.flax import save_file as save_safetensors
from transformers import AutoConfig, PretrainedConfig

from flaxspeaker import configs
from flaxspeaker import evaluation
from flaxspeaker import feature_extraction
from flaxspeaker import model_card
from flaxspeaker import neural_net
from flaxspeaker import scoring


HF_DEFAULT_NAMESPACE = "wq2012"
HF_COLLECTION_URL = (
    "https://huggingface.co/collections/"
    "wq2012/flaxspeaker"
)

CONTROLLED_ABLATION_MODEL_IDS: tuple[str, ...] = (
    # Universal Multi-Domain controlled models (19)
    "univ_conformer_tiny_ge2e",
    "univ_conformer_small_ge2e",
    "univ_conformer_base_ge2e",
    "univ_conformer_small_dr_vectors",
    "univ_conformer_small_ext_softmax",
    "univ_conformer_small_pfas",
    "univ_conformer_small_pfas_ge2e",
    "univ_conformer_small_arcface",
    "univ_conformer_small_cosface",
    "univ_conformer_small_sphereface",
    "univ_conformer_small_triplet",
    "univ_conformer_small_ge2e_contrast",
    "univ_mamba_small_ge2e",
    "univ_transformer_small_ge2e",
    "univ_lstm_small_ge2e",
    "univ_ecapa_tdnn_tiny_ge2e",
    "univ_ecapa_tdnn_small_ge2e",
    "univ_ecapa_tdnn_base_ge2e",
    "univ_resnet_small_ge2e",
    # LibriSpeech Single-Domain controlled models (19)
    "ls_conformer_tiny_ge2e",
    "ls_conformer_small_ge2e",
    "ls_conformer_base_ge2e",
    "ls_conformer_small_dr_vectors",
    "ls_conformer_small_ext_softmax",
    "ls_conformer_small_pfas",
    "ls_conformer_small_pfas_ge2e",
    "ls_conformer_small_arcface",
    "ls_conformer_small_cosface",
    "ls_conformer_small_sphereface",
    "ls_conformer_small_triplet",
    "ls_conformer_small_ge2e_contrast",
    "ls_mamba_small_ge2e",
    "ls_transformer_small_ge2e",
    "ls_lstm_small_ge2e",
    "ls_ecapa_tdnn_tiny_ge2e",
    "ls_ecapa_tdnn_small_ge2e",
    "ls_ecapa_tdnn_base_ge2e",
    "ls_resnet_small_ge2e",
    # CN-Celeb2 Single-Domain controlled models (4)
    "cn_conformer_small_dr_vectors",
    "cn_conformer_small_ge2e",
    "cn_conformer_small_pfas",
    "cn_ecapa_tdnn_small_arcface",
    # VoxCeleb1 Single-Domain controlled models (2)
    "vox_conformer_small_ge2e",
    "vox_ecapa_tdnn_small_ext_softmax",
)

FULL_DATA_FLAGSHIP_IDS: tuple[str, ...] = (
    "univ_full_conformer_large_arcface",
    "univ_full_ecapa_c1024_arcface",
    "univ_full_conformer_large_pfas",
    "univ_full_conformer_large_dr_vectors",
)

PRETRAINED_MODEL_IDS: tuple[str, ...] = (
    CONTROLLED_ABLATION_MODEL_IDS + FULL_DATA_FLAGSHIP_IDS
)

HF_MODEL_REGISTRY: tuple[str, ...] = PRETRAINED_MODEL_IDS


class FlaxSpeakerConfig(PretrainedConfig):
    """HuggingFace `PretrainedConfig` for FlaxSpeaker models."""

    model_type = "flaxspeaker"

    def __init__(
        self,
        backbone: str = "conformer",
        size_variant: str = "small",
        pooling_type: str = "asp",
        embedding_dim: Optional[int] = None,
        seq_len: int = 200,
        feature_type: str = "log_mel",
        n_mels: int = 80,
        n_mfcc: int = 40,
        sample_rate: int = 16000,
        scoring_type: str = "cosine",
        loss_type: str = "ge2e_softmax",
        experiment_dict: Optional[dict[str, Any]] = None,
        training_provenance: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.size_variant = size_variant
        self.pooling_type = pooling_type
        default_emb_dims = {
            "tiny": 128,
            "small": 192,
            "base": 256,
            "large": 256,
            "c1024": 256,
        }
        self._explicit_embedding_dim = embedding_dim is not None
        self.embedding_dim = (
            int(embedding_dim)
            if embedding_dim is not None
            else default_emb_dims.get(str(size_variant).lower(), 192)
        )
        self.seq_len = seq_len
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.scoring_type = scoring_type
        self.loss_type = loss_type
        self.experiment_dict = experiment_dict or {}
        self.training_provenance = training_provenance

    @classmethod
    def from_experiment_config(
        cls, exp_cfg: configs.ExperimentConfig
    ) -> "FlaxSpeakerConfig":
        exp_dict = exp_cfg.to_dict()
        return cls(
            backbone=exp_dict["model"]["backbone"],
            size_variant=exp_dict["model"]["size_variant"],
            pooling_type=exp_dict["model"]["pooling"]["pooling_type"],
            embedding_dim=exp_dict["model"]["embedding_dim"],
            seq_len=exp_dict["model"]["seq_len"],
            feature_type=exp_dict["frontend"]["feature_type"],
            n_mels=exp_dict["frontend"]["n_mels"],
            n_mfcc=exp_dict["frontend"]["n_mfcc"],
            sample_rate=exp_dict["frontend"]["sample_rate"],
            scoring_type=exp_dict["train"]["loss"]["scoring_type"],
            loss_type=exp_dict["train"]["loss"]["loss_type"],
            experiment_dict=exp_dict,
        )

    def to_experiment_config(self) -> configs.ExperimentConfig:
        if self.experiment_dict:
            return configs.ExperimentConfig.from_dict(self.experiment_dict)
        cfg = configs.ExperimentConfig()
        cfg.model.backbone = configs.BackboneType(self.backbone)
        cfg.model.apply_size_variant(self.size_variant)
        cfg.model.pooling.pooling_type = configs.PoolingType(self.pooling_type)
        if getattr(self, "_explicit_embedding_dim", True):
            cfg.model.embedding_dim = self.embedding_dim
        cfg.model.seq_len = self.seq_len
        cfg.model.scoring_type = configs.ScoringType(self.scoring_type)
        cfg.frontend.feature_type = configs.FeatureType(self.feature_type)
        cfg.frontend.n_mels = self.n_mels
        cfg.frontend.n_mfcc = self.n_mfcc
        cfg.frontend.sample_rate = self.sample_rate
        cfg.train.loss.scoring_type = configs.ScoringType(self.scoring_type)
        cfg.train.loss.loss_type = configs.LossType(self.loss_type)
        cfg.loss = cfg.train.loss
        return cfg


try:
    AutoConfig.register("flaxspeaker", FlaxSpeakerConfig)
except ValueError:
    pass


class FlaxSpeakerModel:
    """HuggingFace-compatible FlaxSpeaker model with `.save_pretrained` / `.from_pretrained`."""

    def __init__(
        self,
        config: FlaxSpeakerConfig,
        params: Optional[dict[str, Any]] = None,
        batch_stats: Optional[dict[str, Any]] = None,
    ):
        self.config = config
        self.exp_config = config.to_experiment_config()
        self.encoder, self.state = neural_net.get_speaker_encoder(
            self.exp_config, skip_warm_start=(params is not None)
        )
        if params is not None:
            self.state = self.state.replace(params=params)
        if batch_stats is not None:
            self.state = self.state.replace(batch_stats=batch_stats)
        self.frontend = feature_extraction.AudioFrontend(
            self.exp_config.frontend, self.exp_config.vad
        )
        self._jit_apply = jax.jit(
            lambda vars_d, x: self.state.apply_fn(vars_d, x)
        )
        self.eval_results: Optional[dict[str, Any]] = None

    @property
    def params(self) -> dict[str, Any]:
        return self.state.params

    @property
    def batch_stats(self) -> dict[str, Any]:
        return getattr(self.state, "batch_stats", None) or {}

    @property
    def num_parameters(self) -> int:
        enc_only = {
            k: v
            for k, v in self.state.params.items()
            if not k.startswith("_aux_class_weights")
        }
        return neural_net.count_parameters(enc_only)

    def eer_threshold(
        self, domain: str = "universal_pooled", num_enroll: int = 1
    ) -> Optional[float]:
        """Decision threshold at the checkpoint's own EER operating point."""
        if not self.eval_results:
            return None
        md = self.eval_results.get("multi_domain_eval", {})
        block = "domains_3utt" if int(num_enroll) >= 2 else "domains_1utt"
        rec = md.get(block, {}).get(domain)
        if rec is None and block == "domains_3utt":
            rec = md.get("domains_1utt", {}).get(domain)
        if rec is None and domain == "universal_pooled":
            rec = self.eval_results.get("in_domain_eval")
        if rec is None and "official_benchmarks" in self.eval_results:
            ob = self.eval_results["official_benchmarks"]
            rec = ob.get("voxceleb1_o") or ob.get("librispeech_test_clean")
        if isinstance(rec, dict) and "eer_threshold" in rec:
            return float(rec["eer_threshold"])
        return None

    @staticmethod
    def load_exp_config(model_dir: str) -> configs.ExperimentConfig:
        """The `ExperimentConfig` stored in a checkpoint directory's `config.json`."""
        return FlaxSpeakerConfig.from_pretrained(model_dir).to_experiment_config()

    def __call__(self, features: jax.Array) -> jax.Array:
        """Forward pass on batched features `(B, T, F) -> (B, D)`."""
        bs = self.batch_stats
        vars_d = (
            {"params": self.state.params, "batch_stats": bs}
            if bs
            else {"params": self.state.params}
        )
        return self._jit_apply(vars_d, features)

    def resolve_frontend(
        self,
        apply_vad: Optional[bool] = None,
        vad_top_db: Optional[float] = None,
        vad_snr_margin_db: Optional[float] = None,
    ) -> feature_extraction.AudioFrontend:
        """Resolve an AudioFrontend with optional VAD overrides."""
        if apply_vad is None and vad_top_db is None and vad_snr_margin_db is None:
            return self.frontend
        vad_cfg = dataclasses.replace(
            self.exp_config.vad,
            enabled=(
                bool(apply_vad)
                if apply_vad is not None
                else self.exp_config.vad.enabled
            ),
            energy_db_threshold=(
                float(vad_top_db)
                if vad_top_db is not None
                else self.exp_config.vad.energy_db_threshold
            ),
            snr_margin_db=(
                float(vad_snr_margin_db)
                if vad_snr_margin_db is not None
                else self.exp_config.vad.snr_margin_db
            ),
        )
        return feature_extraction.AudioFrontend(
            self.exp_config.frontend, vad_cfg
        )

    _resolve_frontend = resolve_frontend

    def extract_embedding(
        self,
        audio_or_path: Union[str, np.ndarray],
        *,
        full_sequence: Optional[bool] = None,
        apply_vad: Optional[bool] = None,
        vad_top_db: Optional[float] = None,
        vad_snr_margin_db: Optional[float] = None,
    ) -> np.ndarray:
        """Extract speaker embedding from an audio file path or 1D waveform."""
        active_frontend = self.resolve_frontend(
            apply_vad=apply_vad,
            vad_top_db=vad_top_db,
            vad_snr_margin_db=vad_snr_margin_db,
        )
        if isinstance(audio_or_path, str):
            feats = active_frontend.extract_from_file(audio_or_path)
        else:
            feats = active_frontend.extract_from_waveform(audio_or_path)

        use_full_seq = (
            bool(full_sequence)
            if full_sequence is not None
            else bool(
                self.exp_config.eval.full_sequence_inference
                and self.exp_config.model.full_sequence_inference
            )
        )
        if use_full_seq:
            return evaluation.encode_features(
                feats, self.state, self.exp_config
            )

        slide_cfg = dataclasses.replace(
            self.exp_config,
            model=dataclasses.replace(
                self.exp_config.model, full_sequence_inference=False
            ),
            eval=dataclasses.replace(
                self.exp_config.eval, full_sequence_inference=False
            ),
        )
        emb = evaluation.run_inference(
            jnp.asarray(feats, dtype=jnp.float32), self.state, slide_cfg
        )
        return np.asarray(emb, dtype=np.float32)

    def score_embeddings(
        self,
        enroll_emb: Union[
            np.ndarray, jax.Array, Sequence[Union[np.ndarray, jax.Array]]
        ],
        test_emb: Union[np.ndarray, jax.Array],
    ) -> float:
        """Score single or multi-utterance enrollment embedding(s) against a test embedding."""
        t_emb = jnp.asarray(test_emb, dtype=jnp.float32)
        if isinstance(enroll_emb, (list, tuple)):
            enroll_list = [jnp.asarray(e, dtype=jnp.float32) for e in enroll_emb]
        else:
            e_arr = jnp.asarray(enroll_emb, dtype=jnp.float32)
            if e_arr.ndim == 2:
                enroll_list = [e_arr[i] for i in range(e_arr.shape[0])]
            else:
                enroll_list = [e_arr]

        if not enroll_list:
            raise ValueError("At least one enrollment embedding is required.")

        scoring_type = self.config.scoring_type.lower()
        if (
            scoring_type == "pfas"
            or self.config.pooling_type.lower() == "pfas"
            or self.exp_config.model.pooling.pooling_type
            == configs.PoolingType.PFAS
        ):
            pfas_cfg = self.exp_config.model.pfas
            pfas_scale = (
                float(np.exp(float(self.state.params["_aux_pfas_log_scale"])))
                if "_aux_pfas_log_scale" in self.state.params
                else float(pfas_cfg.scale_factor)
            )
            t_k, t_v = scoring.unpack_pfas_representation(
                t_emb, pfas_cfg.num_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
            )
            if len(enroll_list) == 1:
                e_k, e_v = scoring.unpack_pfas_representation(
                    enroll_list[0],
                    pfas_cfg.num_keys,
                    pfas_cfg.key_dim,
                    pfas_cfg.value_dim,
                )
            else:
                stacked_e = scoring.stack_multi_enroll_pfas(
                    enroll_list,
                    num_keys=pfas_cfg.num_keys,
                    key_dim=pfas_cfg.key_dim,
                    value_dim=pfas_cfg.value_dim,
                )
                e_k, e_v = scoring.unpack_pfas_representation(
                    stacked_e,
                    len(enroll_list) * pfas_cfg.num_keys,
                    pfas_cfg.key_dim,
                    pfas_cfg.value_dim,
                )
            return float(
                scoring.pfas_score_pair(
                    t_k,
                    t_v,
                    e_k,
                    e_v,
                    scale_factor=pfas_scale,
                    per_trial_softmax=pfas_cfg.per_trial_softmax,
                    apply_softmax_per_test_key=getattr(
                        pfas_cfg, "apply_softmax_per_test_key", False
                    ),
                    apply_l2_norm_to_keys=getattr(
                        pfas_cfg, "apply_l2_norm_to_keys", True
                    ),
                    apply_l2_norm_to_values=getattr(
                        pfas_cfg, "apply_l2_norm_to_values", False
                    ),
                    apply_global_l2_norm_to_concat_form=getattr(
                        pfas_cfg, "apply_global_l2_norm_to_concat_form", True
                    ),
                    attention_weight_mode=getattr(
                        pfas_cfg, "attention_weight_mode", "exp"
                    ),
                )
            )

        if len(enroll_list) == 1:
            e_emb = enroll_list[0]
        else:
            e_emb = scoring.compute_speaker_centroid(
                jnp.stack(enroll_list, axis=0)
            )

        if scoring_type == "dr_vector" and "_aux_dr_net" in self.state.params:
            dr_net = scoring.build_decision_residual_network(
                self.exp_config.model.dr_vector
            )
            score = dr_net.apply(
                {"params": self.state.params["_aux_dr_net"]},
                t_emb[None, :],
                e_emb[None, :],
                method=dr_net.score_pairs,
            )
            return float(score[0])

        return float(scoring.cosine_similarity(e_emb, t_emb))

    def verify(
        self,
        enroll_audio: Union[
            str, np.ndarray, Sequence[Union[str, np.ndarray]]
        ],
        test_audio: Union[str, np.ndarray],
        *,
        full_sequence: Optional[bool] = None,
        apply_vad: Optional[bool] = None,
        vad_top_db: Optional[float] = None,
        vad_snr_margin_db: Optional[float] = None,
    ) -> float:
        """Compute verification similarity score between enrollment and test audio."""
        t_emb = self.extract_embedding(
            test_audio,
            full_sequence=full_sequence,
            apply_vad=apply_vad,
            vad_top_db=vad_top_db,
            vad_snr_margin_db=vad_snr_margin_db,
        )
        if isinstance(enroll_audio, (list, tuple)):
            e_embs = [
                self.extract_embedding(
                    item,
                    full_sequence=full_sequence,
                    apply_vad=apply_vad,
                    vad_top_db=vad_top_db,
                    vad_snr_margin_db=vad_snr_margin_db,
                )
                for item in enroll_audio
            ]
            return self.score_embeddings(e_embs, t_emb)
        e_emb = self.extract_embedding(
            enroll_audio,
            full_sequence=full_sequence,
            apply_vad=apply_vad,
            vad_top_db=vad_top_db,
            vad_snr_margin_db=vad_snr_margin_db,
        )
        return self.score_embeddings(e_emb, t_emb)

    def save_pretrained(
        self,
        save_directory: str,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save model in HuggingFace Hub format (`config.json`, `model.safetensors`,
        `flax_model.msgpack`, and `README.md`).
        """
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        export_params = {
            k: v
            for k, v in self.state.params.items()
            if k != "_aux_class_weights"
        }
        bs = self.batch_stats

        # 1. SafeTensors (params + optional _batch_stats.*)
        flat = flax.traverse_util.flatten_dict(export_params, sep=".")
        flat_jnp = {k: jnp.asarray(v) for k, v in flat.items()}
        if bs:
            flat_bs = flax.traverse_util.flatten_dict(bs, sep=".")
            for k, v in flat_bs.items():
                flat_jnp[f"_batch_stats.{k}"] = jnp.asarray(v)
        save_safetensors(flat_jnp, os.path.join(save_directory, "model.safetensors"))

        # 2. Flax Msgpack
        neural_net.save_model(
            os.path.join(save_directory, "flax_model.msgpack"),
            export_params,
            batch_stats=bs if bs else None,
        )

        # 3. Clean up any stale preprocessor_config.json
        stale_preproc = os.path.join(save_directory, "preprocessor_config.json")
        if os.path.exists(stale_preproc):
            os.remove(stale_preproc)

        # 4. Save metrics if provided
        model_id = os.path.basename(os.path.normpath(save_directory))
        if metrics is not None:
            self.eval_results = metrics
            with open(
                os.path.join(save_directory, "eval_results.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metrics, f, indent=2)

        # 5. Model card README.md
        card_text = model_card.build_model_card(
            model_id=model_id,
            exp_cfg=self.exp_config,
            num_parameters=self.num_parameters,
            metrics=metrics,
            save_directory=save_directory,
        )
        with open(
            os.path.join(save_directory, "README.md"), "w", encoding="utf-8"
        ) as f:
            f.write(card_text)

    @staticmethod
    def _has_local_weights(dir_path: str) -> bool:
        """Return True if `dir_path` contains `config.json` and model weight files."""
        return (
            os.path.isdir(dir_path)
            and os.path.isfile(os.path.join(dir_path, "config.json"))
            and (
                os.path.isfile(os.path.join(dir_path, "model.safetensors"))
                or os.path.isfile(os.path.join(dir_path, "flax_model.msgpack"))
            )
        )

    @staticmethod
    def _resolve_repo_id(pretrained_model_name_or_path: str) -> tuple[str, str]:
        """Resolve a model reference or URL into `(repo_id, short_id)`."""
        clean_ref = pretrained_model_name_or_path.strip().rstrip("/")
        for url_prefix in (
            "https://huggingface.co/",
            "http://huggingface.co/",
        ):
            if clean_ref.startswith(url_prefix):
                clean_ref = clean_ref.removeprefix(url_prefix).rstrip("/")
                break
        base_id = os.path.basename(os.path.normpath(clean_ref))
        if clean_ref.startswith(f"{HF_DEFAULT_NAMESPACE}/"):
            short_id = clean_ref.split("/", 1)[1]
            return clean_ref, short_id
        if base_id in PRETRAINED_MODEL_IDS:
            return f"{HF_DEFAULT_NAMESPACE}/{base_id}", base_id
        if (
            not clean_ref.startswith(("/", "./", "../", "~"))
            and clean_ref.count("/") == 1
            and not os.path.exists(pretrained_model_name_or_path)
        ):
            return clean_ref, base_id
        raise FileNotFoundError(
            "Neither model.safetensors nor flax_model.msgpack found in "
            f"{pretrained_model_name_or_path}"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        force_hub: bool = False,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "FlaxSpeakerModel":
        """Load a `FlaxSpeakerModel` from a local directory or Hugging Face Hub repo ID."""
        resolved_dir: Optional[str] = None
        if not force_hub and cls._has_local_weights(pretrained_model_name_or_path):
            resolved_dir = pretrained_model_name_or_path
        else:
            repo_id, short_id = cls._resolve_repo_id(
                pretrained_model_name_or_path
            )

            if not force_hub:
                repo_root = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                env_pm = os.environ.get("FLAXSPEAKER_PRETRAINED_ROOT")
                local_candidates: list[str] = []
                if env_pm:
                    local_candidates.append(os.path.join(env_pm, short_id))
                local_candidates.extend(
                    [
                        os.path.join("pretrained_models", short_id),
                        os.path.join(repo_root, "pretrained_models", short_id),
                    ]
                )
                resolved_dir = next(
                    (
                        cand
                        for cand in local_candidates
                        if cls._has_local_weights(cand)
                    ),
                    None,
                )

            if resolved_dir is None:
                from huggingface_hub import snapshot_download

                resolved_dir = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    revision=revision,
                    local_files_only=local_files_only,
                )

        config = FlaxSpeakerConfig.from_pretrained(resolved_dir)
        model = cls(config=config)

        st_path = os.path.join(resolved_dir, "model.safetensors")
        msg_path = os.path.join(resolved_dir, "flax_model.msgpack")
        if os.path.exists(st_path):
            flat_loaded = load_safetensors(st_path)
            flat_p = {
                k: v
                for k, v in flat_loaded.items()
                if not k.startswith("_batch_stats.")
            }
            flat_bs = {
                k.removeprefix("_batch_stats."): v
                for k, v in flat_loaded.items()
                if k.startswith("_batch_stats.")
            }
            params = flax.traverse_util.unflatten_dict(flat_p, sep=".")
            batch_stats = (
                flax.traverse_util.unflatten_dict(flat_bs, sep=".")
                if flat_bs
                else model.batch_stats
            )
            model.state = model.state.replace(
                params=params, batch_stats=batch_stats
            )
        elif os.path.exists(msg_path):
            template = (
                {"params": model.state.params, "batch_stats": model.batch_stats}
                if model.batch_stats
                else {"params": model.state.params}
            )
            loaded = neural_net.load_model(msg_path, template)
            if isinstance(loaded, dict) and "params" in loaded:
                params = loaded["params"]
                batch_stats = (
                    loaded.get("batch_stats", model.batch_stats)
                    or model.batch_stats
                )
            else:
                params = loaded
                batch_stats = model.batch_stats
            model.state = model.state.replace(
                params=params, batch_stats=batch_stats
            )
        else:
            raise FileNotFoundError(
                f"Neither model.safetensors nor flax_model.msgpack found in {resolved_dir}"
            )
        metrics_path = os.path.join(resolved_dir, "eval_results.json")
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    model.eval_results = json.load(f)
            except (OSError, ValueError):
                model.eval_results = None
        return model


def load_pretrained(
    model_dir_or_id: str, **kwargs: Any
) -> FlaxSpeakerModel:
    """Load a pretrained FlaxSpeakerModel from a local directory or Hugging Face Hub repo ID."""
    return FlaxSpeakerModel.from_pretrained(model_dir_or_id, **kwargs)
