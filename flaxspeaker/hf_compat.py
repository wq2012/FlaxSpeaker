"""HuggingFace `transformers` and `safetensors` integration for FlaxSpeaker.

Provides:
- `FlaxSpeakerConfig(PretrainedConfig)`: HuggingFace-compatible configuration class
  registered under `model_type = "flaxspeaker"`, serializing to `config.json`.
- `FlaxSpeakerModel`: High-level model wrapper supporting `.save_pretrained()` and
  `.from_pretrained()` (writing `config.json`, `model.safetensors`,
  `flax_model.msgpack`, `preprocessor_config.json`, and a HuggingFace Model Card),
  plus one-line `.extract_embedding()` and `.verify()` methods.
"""

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
from flaxspeaker import neural_net
from flaxspeaker import scoring


class FlaxSpeakerConfig(PretrainedConfig):
    """HuggingFace `PretrainedConfig` for FlaxSpeaker models."""

    model_type = "flaxspeaker"

    def __init__(
        self,
        backbone: str = "conformer",
        size_variant: str = "small",
        pooling_type: str = "asp",
        embedding_dim: int = 192,
        seq_len: int = 200,
        feature_type: str = "log_mel",
        n_mels: int = 80,
        n_mfcc: int = 40,
        sample_rate: int = 16000,
        scoring_type: str = "cosine",
        loss_type: str = "ge2e_softmax",
        experiment_dict: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.size_variant = size_variant
        self.pooling_type = pooling_type
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.scoring_type = scoring_type
        self.loss_type = loss_type
        self.experiment_dict = experiment_dict or {}

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
        cfg.model.size_variant = configs.SizeVariant(self.size_variant)
        cfg.model.pooling.pooling_type = configs.PoolingType(self.pooling_type)
        cfg.model.embedding_dim = self.embedding_dim
        cfg.model.seq_len = self.seq_len
        cfg.frontend.feature_type = configs.FeatureType(self.feature_type)
        cfg.frontend.n_mels = self.n_mels
        cfg.frontend.n_mfcc = self.n_mfcc
        cfg.frontend.sample_rate = self.sample_rate
        cfg.train.loss.scoring_type = configs.ScoringType(self.scoring_type)
        cfg.train.loss.loss_type = configs.LossType(self.loss_type)
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
    ):
        self.config = config
        self.exp_config = config.to_experiment_config()
        self.encoder, self.state = neural_net.get_speaker_encoder(self.exp_config)
        if params is not None:
            self.state = self.state.replace(params=params)
        self.frontend = feature_extraction.AudioFrontend(
            self.exp_config.frontend, self.exp_config.vad
        )

    @property
    def params(self) -> dict[str, Any]:
        return self.state.params

    @property
    def num_parameters(self) -> int:
        # Count encoder parameters (excluding auxiliary training-only classification weights)
        enc_only = {
            k: v
            for k, v in self.state.params.items()
            if not k.startswith("_aux_class_weights")
        }
        return neural_net.count_parameters(enc_only)

    def __call__(self, features: jax.Array) -> jax.Array:
        """Forward pass on batched features `(B, T, F) -> (B, D)`."""
        return self.state.apply_fn({"params": self.state.params}, features)

    def extract_embedding(
        self, audio_or_path: Union[str, np.ndarray]
    ) -> np.ndarray:
        """Extract speaker embedding from an audio file path or 1D waveform."""
        if isinstance(audio_or_path, str):
            feats = self.frontend.extract_from_file(audio_or_path)
        else:
            feats = self.frontend.extract_from_waveform(audio_or_path)
        emb = evaluation.run_inference(
            jnp.asarray(feats, dtype=jnp.float32), self.state, self.exp_config
        )
        return np.asarray(emb, dtype=np.float32)

    def verify(
        self,
        enroll_audio: Union[str, np.ndarray],
        test_audio: Union[str, np.ndarray],
    ) -> float:
        """Compute verification similarity score between enrollment and test audio."""
        e_emb = jnp.asarray(self.extract_embedding(enroll_audio))
        t_emb = jnp.asarray(self.extract_embedding(test_audio))
        scoring_type = self.config.scoring_type.lower()

        if scoring_type == "pfas":
            pfas_cfg = self.exp_config.model.pfas
            t_k, t_v = scoring.unpack_pfas_representation(
                t_emb, pfas_cfg.num_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
            )
            e_k, e_v = scoring.unpack_pfas_representation(
                e_emb, pfas_cfg.num_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
            )
            return float(
                scoring.pfas_score_pair(
                    t_k,
                    t_v,
                    e_k,
                    e_v,
                    scale_factor=pfas_cfg.scale_factor,
                    per_trial_softmax=pfas_cfg.per_trial_softmax,
                )
            )
        elif scoring_type == "dr_vector" and "_aux_dr_net" in self.state.params:
            dr_cfg = self.exp_config.model.dr_vector
            dr_net = scoring.DecisionResidualNetwork(
                hidden_dims=tuple(dr_cfg.hidden_dims),
                activation=dr_cfg.activation,
                use_layer_norm=dr_cfg.use_layer_norm,
                include_cosine_in_input=dr_cfg.include_cosine_in_input,
                residual_weight=dr_cfg.residual_weight,
            )
            score = dr_net.apply(
                {"params": self.state.params["_aux_dr_net"]},
                t_emb[None, :],
                e_emb[None, :],
                method=dr_net.score_pairs,
            )
            return float(score[0])
        else:
            return float(scoring.cosine_similarity(e_emb, t_emb))

    def save_pretrained(
        self,
        save_directory: str,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save model in standard HuggingFace Hub format.

        Writes:
        - `config.json`
        - `model.safetensors`
        - `flax_model.msgpack`
        - `preprocessor_config.json`
        - `README.md` (Model Card)
        """
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        # Save weights excluding training-only classification prototypes
        export_params = {
            k: v
            for k, v in self.state.params.items()
            if k != "_aux_class_weights"
        }

        # 1. SafeTensors
        flat = flax.traverse_util.flatten_dict(export_params, sep=".")
        flat_jnp = {k: jnp.asarray(v) for k, v in flat.items()}
        save_safetensors(flat_jnp, os.path.join(save_directory, "model.safetensors"))

        # 2. Flax Msgpack
        neural_net.save_model(
            os.path.join(save_directory, "flax_model.msgpack"),
            {"params": export_params},
        )

        # 3. HuggingFace Whisper/Audio FeatureExtractor compatible preprocessor_config.json
        preproc_cfg = {
            "feature_extractor_type": "WhisperFeatureExtractor",
            "feature_size": self.exp_config.frontend.feature_dim,
            "sampling_rate": self.exp_config.frontend.sample_rate,
            "hop_length": self.exp_config.frontend.hop_length,
            "n_fft": self.exp_config.frontend.n_fft,
            "padding_value": 0.0,
            "return_attention_mask": False,
        }
        with open(
            os.path.join(save_directory, "preprocessor_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(preproc_cfg, f, indent=2)

        # 4. Save metrics if provided
        if metrics is not None:
            with open(
                os.path.join(save_directory, "eval_results.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metrics, f, indent=2)

        # 5. Model card README.md
        card_lines = [
            "---",
            "library_name: flaxspeaker",
            "tags:",
            "- audio",
            "- speaker-verification",
            "- speaker-recognition",
            "- jax",
            "- flax",
            "---",
            f"# FlaxSpeaker Model (`{self.config.backbone}` - `{self.config.size_variant}`)",
            "",
            "## Model Summary",
            f"- **Backbone**: `{self.config.backbone}` (`{self.config.size_variant}`)",
            f"- **Pooling**: `{self.config.pooling_type}`",
            f"- **Scoring**: `{self.config.scoring_type}`",
            f"- **Loss Function**: `{self.config.loss_type}`",
            f"- **Output Embedding Dimension**: `{self.exp_config.model.output_embedding_dim}`",
            f"- **Parameter Count**: `{self.num_parameters:,}`",
            "",
            "## Usage",
            "```python",
            "from flaxspeaker.hf_compat import FlaxSpeakerModel",
            "",
            f'model = FlaxSpeakerModel.from_pretrained("{save_directory}")',
            'emb = model.extract_embedding("path/to/audio.flac")',
            'score = model.verify("path/to/enroll.flac", "path/to/test.flac")',
            "```",
        ]
        if metrics:
            card_lines += [
                "",
                "## Evaluation Results",
                "```json",
                json.dumps(metrics, indent=2),
                "```",
            ]
        with open(
            os.path.join(save_directory, "README.md"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(card_lines) + "\n")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "FlaxSpeakerModel":
        """Load a `FlaxSpeakerModel` from a directory saved via `.save_pretrained()`."""
        config = FlaxSpeakerConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config=config)

        st_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        msg_path = os.path.join(pretrained_model_name_or_path, "flax_model.msgpack")
        if os.path.exists(st_path):
            flat_loaded = load_safetensors(st_path)
            params = flax.traverse_util.unflatten_dict(flat_loaded, sep=".")
            model.state = model.state.replace(params=params)
        elif os.path.exists(msg_path):
            loaded = neural_net.load_model(msg_path, {"params": model.state.params})
            params = loaded["params"] if "params" in loaded else loaded
            model.state = model.state.replace(params=params)
        else:
            raise FileNotFoundError(
                f"Neither model.safetensors nor flax_model.msgpack found in {pretrained_model_name_or_path}"
            )
        return model
