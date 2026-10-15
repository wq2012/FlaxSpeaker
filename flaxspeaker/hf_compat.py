"""HuggingFace `transformers` and `safetensors` integration for FlaxSpeaker.

All 44 pretrained models are hosted on the Hugging Face Hub at
`https://huggingface.co/wq2012/<model_id>` and grouped in the FlaxSpeaker
collection:
https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595

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


HF_DEFAULT_NAMESPACE = "wq2012"
HF_COLLECTION_URL = (
    "https://huggingface.co/collections/"
    "wq2012/flaxspeaker-6ad02d999a4180c794870595"
)

PRETRAINED_MODEL_IDS: tuple[str, ...] = (
    # Universal Multi-Domain models (19)
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
    # LibriSpeech Single-Domain models (19)
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
    # CN-Celeb2 Single-Domain models (4)
    "cn_conformer_small_dr_vectors",
    "cn_conformer_small_ge2e",
    "cn_conformer_small_pfas",
    "cn_ecapa_tdnn_small_arcface",
    # VoxCeleb1 Single-Domain models (2)
    "vox_conformer_small_ge2e",
    "vox_ecapa_tdnn_small_ext_softmax",
)


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
            pfas_scale = (
                float(np.exp(float(self.state.params["_aux_pfas_log_scale"])))
                if "_aux_pfas_log_scale" in self.state.params
                else float(pfas_cfg.scale_factor)
            )
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
                    scale_factor=pfas_scale,
                    per_trial_softmax=pfas_cfg.per_trial_softmax,
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
        model_id = os.path.basename(os.path.normpath(save_directory))
        if metrics is not None:
            with open(
                os.path.join(save_directory, "eval_results.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metrics, f, indent=2)

        # 5. Model card README.md
        train_ds = (
            metrics.get("train_dataset", "universal")
            if isinstance(metrics, dict)
            else "universal"
        )
        card_lines = [
            "---",
            "license: apache-2.0",
            "library_name: flaxspeaker",
            "pipeline_tag: audio-classification",
            "tags:",
            "- audio",
            "- speaker-verification",
            "- speaker-recognition",
            "- speaker-embeddings",
            "- jax",
            "- flax",
            "- safetensors",
            f"- {self.config.backbone}",
            f"- {self.config.loss_type}",
            f"- {self.config.scoring_type}",
            "datasets:",
            "- librispeech_asr",
            "- cnceleb",
            "- voxceleb",
            "metrics:",
            "- eer",
            "- min_dcf",
            "- roc_auc",
            "---",
            "",
            f"# FlaxSpeaker: `{model_id}` (`{self.config.backbone}` - `{self.config.size_variant}`)",
            "",
            "## Disclaimer & Open-Source Reproduction Notice",
            "",
            "> **This is not an officially supported Google product.**",
            ">",
            (
                "> This model checkpoint and the accompanying `FlaxSpeaker` library "
                "are an independent, clean-room open-source reproduction in JAX/Flax "
                "of speaker recognition algorithms published in:"
            ),
            (
                "> - **GE2E**: *Generalized End-to-End Loss for Speaker Verification* "
                "([arXiv:1710.10467](https://arxiv.org/abs/1710.10467))"
            ),
            (
                "> - **Dr-Vectors**: *Decision Residual Networks and an Improved Loss "
                "for Speaker Recognition* ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989))"
            ),
            (
                "> - **Conformer-ATP**: *Attentive Temporal Pooling for Conformer-based "
                "Streaming Language Identification in Long-form Speech* "
                "([arXiv:2202.12163](https://arxiv.org/abs/2202.12163))"
            ),
            (
                "> - **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* "
                "([arXiv:2203.05642v3](https://arxiv.org/abs/2203.05642v3))"
            ),
            ">",
            (
                "> The experiments in those original papers were conducted using "
                "Google's proprietary internal infrastructure and internal datasets, "
                "not this open-source library or public dataset split."
            ),
            "",
            "## Model Summary",
            "",
            "| Property | Value |",
            "| :--- | :--- |",
            f"| **Model ID** | `{model_id}` |",
            (
                f"| **Hugging Face Repository** | "
                f"[`wq2012/{model_id}`](https://huggingface.co/wq2012/{model_id}) |"
            ),
            (
                "| **Hugging Face Collection** | "
                "[FlaxSpeaker Collection]"
                "(https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) |"
            ),
            "| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |",
            "| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |",
            f"| **Backbone Architecture** | `{self.config.backbone}` (`{self.config.size_variant}`) |",
            f"| **Temporal Pooling** | `{self.config.pooling_type}` |",
            f"| **Training Loss** | `{self.config.loss_type}` |",
            f"| **Verification Scoring** | `{self.config.scoring_type}` |",
            f"| **Output Embedding Dim** | `{self.exp_config.model.output_embedding_dim}` |",
            f"| **Encoder Parameters** | `{self.num_parameters:,}` |",
            f"| **Training Domain** | `{train_ds}` |",
            (
                f"| **Audio Frontend** | `{self.config.n_mels}`-dim `{self.config.feature_type}` "
                f"@ `{self.config.sample_rate}` Hz + Energy VAD + CMVN |"
            ),
            "",
            "## Model Files Included",
            "",
            "- `model.safetensors`: Primary model weights in SafeTensors format",
            "- `flax_model.msgpack`: Native Flax Msgpack serialized parameters",
            "- `config.json`: `FlaxSpeakerConfig` model & training architecture configuration",
            "- `preprocessor_config.json`: `WhisperFeatureExtractor`-compatible frontend configuration",
            (
                "- `eval_results.json`: Verified multi-domain evaluation metrics "
                "(1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs)"
            ),
        ]
        if os.path.exists(os.path.join(save_directory, "model_fp32.tflite")):
            card_lines.append(
                "- `model_fp32.tflite`: Exported FP32 TensorFlow Lite model for edge/mobile CPU inference"
            )
        if os.path.exists(os.path.join(save_directory, "model_int8.tflite")):
            card_lines.append(
                "- `model_int8.tflite`: Dynamic-range INT8 quantized TensorFlow Lite model (~3.5x smaller)"
            )
        card_lines += [
            "",
            "## Quickstart Usage",
            "",
            "### Python API (from Hugging Face Hub or Local Directory)",
            "",
            "```python",
            "from flaxspeaker import load_pretrained",
            "",
            f'model = load_pretrained("wq2012/{model_id}")',
            'emb = model.extract_embedding("path/to/utterance.flac")',
            'score = model.verify("path/to/enroll.flac", "path/to/test.flac")',
            'print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")',
            "```",
            "",
            "### Command-Line Inference",
            "",
            "```bash",
            f"python scripts/inference.py --model_dir wq2012/{model_id} \\",
            "    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac",
            "```",
        ]
        if isinstance(metrics, dict):
            card_lines += [
                "",
                "## Multi-Domain Evaluation Results",
                "",
            ]
            md_eval = metrics.get("multi_domain_eval", {})
            if md_eval and "domains_1utt" in md_eval:
                card_lines += [
                    "### 1-Utterance Verification Across Evaluation Domains",
                    "",
                    (
                        "| Evaluation Split | Trials | EER (%) | 95% Bootstrap CI (%) "
                        "| minDCF@0.01 | minDCF@0.10 | ROC-AUC |"
                    ),
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
                for dom_label, dom_key in [
                    ("LibriSpeech (English Read)", "librispeech"),
                    ("CN-Celeb2 (Chinese Multi-Genre)", "cnceleb2"),
                    ("VoxCeleb1 (In-the-Wild Celebrity)", "voxceleb1"),
                    ("Universal Pooled (All 3 Domains)", "universal_pooled"),
                ]:
                    d_m = md_eval["domains_1utt"].get(dom_key)
                    if d_m:
                        ci_str = (
                            f"[{d_m.get('eer_ci95_low_percent', d_m['eer_percent']):.2f}, "
                            f"{d_m.get('eer_ci95_high_percent', d_m['eer_percent']):.2f}]"
                        )
                        card_lines.append(
                            f"| **{dom_label}** | {d_m['num_trials']:,} | **{d_m['eer_percent']:.2f}%** | "
                            f"{ci_str} | {d_m['min_dcf_001']:.4f} | {d_m['min_dcf_01']:.4f} | {d_m['roc_auc']:.4f} |"
                        )
                macro_1 = md_eval.get("universal_macro_avg_1utt")
                if macro_1:
                    card_lines.append(
                        f"| **Universal Macro-Average (3 Domains)** | 8,000 | **{macro_1['eer_percent']:.2f}%** | "
                        f"— | {macro_1['min_dcf_001']:.4f} | {macro_1['min_dcf_01']:.4f} | {macro_1['roc_auc']:.4f} |"
                    )
                if "domains_3utt" in md_eval:
                    card_lines += [
                        "",
                        "### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains",
                        "",
                        "| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |",
                        "| :--- | :---: | :---: | :---: | :---: | :---: |",
                    ]
                    for dom_label, dom_key in [
                        ("LibriSpeech (3-Utt)", "librispeech"),
                        ("CN-Celeb2 (3-Utt)", "cnceleb2"),
                        ("VoxCeleb1 (3-Utt)", "voxceleb1"),
                        ("Universal Pooled (3-Utt)", "universal_pooled"),
                    ]:
                        d_m = md_eval["domains_3utt"].get(dom_key)
                        if d_m:
                            ci_str = (
                                f"[{d_m.get('eer_ci95_low_percent', d_m['eer_percent']):.2f}, "
                                f"{d_m.get('eer_ci95_high_percent', d_m['eer_percent']):.2f}]"
                            )
                            card_lines.append(
                                f"| **{dom_label}** | {d_m['num_trials']:,} | **{d_m['eer_percent']:.2f}%** | "
                                f"{ci_str} | {d_m['min_dcf_001']:.4f} | {d_m['roc_auc']:.4f} |"
                            )
                    macro_3 = md_eval.get("universal_macro_avg_3utt")
                    if macro_3:
                        card_lines.append(
                            f"| **Universal Macro-Average (3-Utt)** | 3,000 | **{macro_3['eer_percent']:.2f}%** | "
                            f"— | {macro_3['min_dcf_001']:.4f} | {macro_3['roc_auc']:.4f} |"
                        )
            elif "in_domain_eval" in metrics and metrics["in_domain_eval"]:
                ind = metrics["in_domain_eval"]
                card_lines += [
                    f"- **In-Domain EER**: `{ind['eer_percent']:.2f}%`",
                    f"- **In-Domain minDCF@0.01**: `{ind['min_dcf_001']:.4f}`",
                    f"- **In-Domain ROC-AUC**: `{ind['roc_auc']:.4f}`",
                ]
        card_lines += [
            "",
            "## Citation",
            "",
            "```bibtex",
            "@inproceedings{wan2018generalized,",
            "  title={Generalized End-to-End Loss for Speaker Verification},",
            "  author={Wan, Li and Wang, Quan and Papir, Alan and Moreno, Ignacio Lopez},",
            "  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},",
            "  pages={4879--4883},",
            "  year={2018}",
            "}",
            "",
            "@inproceedings{pelecanos2021dr,",
            "  title={Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition},",
            "  author={Pelecanos, Jason and Wang, Quan and Moreno, Ignacio Lopez},",
            "  booktitle={Interspeech},",
            "  pages={4603--4607},",
            "  year={2021}",
            "}",
            "",
            "@inproceedings{wang2022attentive,",
            (
                "  title={Attentive Temporal Pooling for Conformer-based Streaming "
                "Language Identification in Long-form Speech},"
            ),
            "  author={Wang, Quan and Yu, Yang and Pelecanos, Jason and Huang, Yiling and Moreno, Ignacio Lopez},",
            "  booktitle={The Speaker and Language Recognition Workshop (Odyssey)},",
            "  year={2022}",
            "}",
            "",
            "@inproceedings{pelecanos2022parameter,",
            "  title={Parameter-Free Attentive Scoring for Speaker Verification},",
            "  author={Pelecanos, Jason and Wang, Quan and Huang, Yiling and Moreno, Ignacio Lopez},",
            "  booktitle={The Speaker and Language Recognition Workshop (Odyssey)},",
            "  year={2022}",
            "}",
            "```",
        ]
        with open(
            os.path.join(save_directory, "README.md"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(card_lines) + "\n")

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
            norm_path = os.path.normpath(pretrained_model_name_or_path)
            base_id = os.path.basename(norm_path)
            if pretrained_model_name_or_path.startswith(f"{HF_DEFAULT_NAMESPACE}/"):
                short_id = pretrained_model_name_or_path.split("/", 1)[1]
                repo_id = pretrained_model_name_or_path
            elif base_id in PRETRAINED_MODEL_IDS:
                short_id = base_id
                repo_id = f"{HF_DEFAULT_NAMESPACE}/{short_id}"
            elif (
                not pretrained_model_name_or_path.startswith(("/", "./", "../", "~"))
                and pretrained_model_name_or_path.count("/") == 1
                and not os.path.exists(pretrained_model_name_or_path)
            ):
                short_id = base_id
                repo_id = pretrained_model_name_or_path
            else:
                raise FileNotFoundError(
                    "Neither model.safetensors nor flax_model.msgpack found in "
                    f"{pretrained_model_name_or_path}"
                )

            if not force_hub:
                repo_root = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                local_candidates = (
                    os.path.join("pretrained_models", short_id),
                    os.path.join(repo_root, "pretrained_models", short_id),
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
            params = flax.traverse_util.unflatten_dict(flat_loaded, sep=".")
            model.state = model.state.replace(params=params)
        elif os.path.exists(msg_path):
            loaded = neural_net.load_model(msg_path, {"params": model.state.params})
            params = loaded["params"] if "params" in loaded else loaded
            model.state = model.state.replace(params=params)
        else:
            raise FileNotFoundError(
                f"Neither model.safetensors nor flax_model.msgpack found in {resolved_dir}"
            )
        return model


def load_pretrained(
    model_dir_or_id: str, **kwargs: Any
) -> FlaxSpeakerModel:
    """Load a pretrained FlaxSpeakerModel from a local directory or Hugging Face Hub repo ID."""
    return FlaxSpeakerModel.from_pretrained(model_dir_or_id, **kwargs)
