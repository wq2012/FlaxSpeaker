"""Configuration dataclasses, enums, and presets for FlaxSpeaker."""

from __future__ import annotations

import copy
import dataclasses
import enum
from typing import Any, Optional, Union
import munch
import yaml


class _StrEnum(str, enum.Enum):
    """String enum whose `str(x)` always returns `x.value` on Python 3.11+."""

    def __str__(self) -> str:
        return str(self.value)


class VadMode(_StrEnum):
    """Supported Voice Activity Detection (VAD) modes."""

    NONE = "none"
    ENERGY = "energy"
    SPECTRAL = "spectral"
    HYBRID = "hybrid"


class FeatureType(_StrEnum):
    """Supported acoustic feature types."""

    MFCC = "mfcc"
    LOG_MEL = "log_mel"
    WHISPER_MEL = "whisper_mel"
    HF_COMPAT = "hf_compat"


class BackboneType(_StrEnum):
    """Supported neural network backbones for speaker embedding extraction."""

    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CONFORMER = "conformer"
    MAMBA = "mamba"
    ECAPA_TDNN = "ecapa_tdnn"
    RESNET = "resnet"


class ModelSize(_StrEnum):
    """Standardized model size variants (<100M parameters)."""

    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"


# Alias for convenience
SizeVariant = ModelSize


class PoolingType(_StrEnum):
    """Frame-level temporal aggregation / pooling strategies."""

    MEAN = "mean"
    LAST = "last"
    STATS = "stats"
    SAP = "sap"
    ASP = "asp"
    CUMULATIVE_STATS = "cumulative_stats"
    PFAS = "pfas"


class ScoringType(_StrEnum):
    """Speaker verification trial scoring mechanisms."""

    COSINE = "cosine"
    PFAS = "pfas"
    DR_VECTOR = "dr_vector"


class LossType(_StrEnum):
    """Supported speaker recognition training loss functions."""

    TRIPLET = "triplet"
    SOFTMAX = "softmax"
    PAIRWISE = "pairwise"
    GE2E_SOFTMAX = "ge2e_softmax"
    GE2E_CONTRAST = "ge2e_contrast"
    EXTENDED_SET_SOFTMAX = "extended_set_softmax"
    ARCFACE = "arcface"
    COSFACE = "cosface"
    SPHEREFACE = "sphereface"


@dataclasses.dataclass
class VadConfig:
    """Configuration for Voice Activity Detection (VAD)."""

    enabled: bool = True
    mode: str = VadMode.ENERGY.value  # "none", "energy", "spectral", or "hybrid"
    sample_rate: int = 16000
    frame_size_ms: float = 25.0
    frame_step_ms: float = 10.0
    energy_threshold_db: float = -35.0
    energy_db_threshold: float = 30.0
    adaptive_percentile: float = 15.0
    snr_margin_db: float = 6.0
    spectral_flatness_threshold: float = 0.65
    min_speech_frames: int = 3
    hangover_frames: int = 5


@dataclasses.dataclass
class FrontendConfig:
    """Configuration for audio feature extraction frontend."""

    feature_type: str = FeatureType.LOG_MEL.value
    sample_rate: int = 16000
    n_mfcc: int = 80
    n_mels: int = 80
    frame_size_ms: float = 25.0
    frame_step_ms: float = 10.0
    n_fft: int = 512
    fmin: float = 20.0
    fmax: float = 7600.0
    preemph: float = 0.97
    log_offset: float = 1e-6
    apply_cmvn: bool = True
    cmvn_norm_vars: bool = False
    stack_left_context: int = 0
    stack_right_context: int = 0
    frame_stride: int = 1
    hf_extractor_name: str = "openai/whisper-tiny"

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.frame_step_ms / 1000.0))

    @property
    def win_length(self) -> int:
        return int(round(self.sample_rate * self.frame_size_ms / 1000.0))

    @property
    def feature_dim(self) -> int:
        ftype = (
            self.feature_type.value
            if hasattr(self.feature_type, "value")
            else str(self.feature_type).lower()
        )
        base_dim = self.n_mfcc if ftype == FeatureType.MFCC.value else self.n_mels
        return base_dim * (1 + self.stack_left_context + self.stack_right_context)


@dataclasses.dataclass
class SpecAugConfig:
    """Configuration for SpecAugment data augmentation."""

    use_specaug: bool = True
    freq_mask_prob: float = 0.3
    time_mask_prob: float = 0.3
    freq_mask_max_width: int = 10
    time_mask_max_width: int = 10
    num_freq_masks: int = 1
    num_time_masks: int = 1
    gaussian_noise_std: float = 0.0


@dataclasses.dataclass
class LstmConfig:
    """Configuration for LSTM speaker encoder."""

    hidden_size: int = 256
    proj_size: int = 0
    num_layers: int = 3
    bidirectional: bool = False
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    frame_aggregation_mean: bool = True


@dataclasses.dataclass
class TransformerConfig:
    """Configuration for Transformer speaker encoder."""

    dim: int = 256
    num_encoder_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    use_positional_encoding: bool = True
    use_rope: bool = False


@dataclasses.dataclass
class ConformerConfig:
    """Configuration for Conformer speaker encoder."""

    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_expansion: int = 2
    ffn_dim: int = 512
    conv_kernel_size: int = 15
    dropout_rate: float = 0.1
    subsample_factor: int = 1


@dataclasses.dataclass
class MambaConfig:
    """Configuration for Mamba (Selective State Space Model) speaker encoder."""

    dim: int = 256
    num_layers: int = 4
    d_state: int = 16
    state_dim: int = 16
    d_conv: int = 4
    conv_kernel_size: int = 4
    expand_factor: int = 2
    dt_rank: int = 16
    dropout_rate: float = 0.1


@dataclasses.dataclass
class EcapaTdnnConfig:
    """Configuration for ECAPA-TDNN speaker encoder."""

    channels: int = 256
    scale: int = 4
    kernel_sizes: tuple[int, ...] = (5, 3, 3, 3)
    dilations: tuple[int, ...] = (1, 2, 3, 4)
    attention_channels: int = 64
    se_channels: int = 64
    se_bottleneck_dim: int = 64
    mfa_channels: int = 768
    mfa_dim: int = 512


@dataclasses.dataclass
class ResNetConfig:
    """Configuration for Thin-ResNet / ResNet-34 speaker encoder."""

    base_channels: int = 32
    stage_blocks: tuple[int, ...] = (2, 2, 2, 2)
    num_blocks: tuple[int, ...] = (2, 2, 2, 2)
    use_se: bool = True
    se_reduction: int = 8
    out_dim: int = 256


@dataclasses.dataclass
class PoolingConfig:
    """Configuration for temporal frame aggregation / pooling."""

    pooling_type: str = PoolingType.ASP.value
    attention_dim: int = 128
    use_global_context: bool = True
    share_params: bool = True
    nonlinear_attention: bool = True
    divide_layer: bool = False
    max_pooling_mode: str = "off"
    top_k: int = 10
    window_size: int = 10
    window_stride: int = 5
    use_weighted_frames: bool = False
    epsilon: float = 1e-5


@dataclasses.dataclass
class PfasConfig:
    """Configuration for Parameter-Free Attentive Scoring (arXiv:2203.05642v3)."""

    num_keys: int = 4
    key_dim: int = 32
    value_dim: int = 32
    attention_dim: int = 128
    attention_type: str = "shared_nonlinear"
    use_keys_and_queries: bool = False
    scale_factor: float = 5.0
    use_trainable_scale_factor: bool = False
    normalize_keys_and_values: bool = True
    normalize_overall_output: bool = False
    per_trial_softmax: bool = True
    apply_l2_norm_to_keys: bool = True
    apply_l2_norm_to_values: bool = True
    apply_global_l2_norm_to_concat_form: bool = True
    apply_softmax_per_test_key: bool = False

    @property
    def representation_dim(self) -> int:
        if self.use_keys_and_queries:
            return self.num_keys * (2 * self.key_dim + self.value_dim)
        return self.num_keys * (self.key_dim + self.value_dim)


@dataclasses.dataclass
class DrVectorConfig:
    """Configuration for Decision Residual Networks (arXiv:2104.01989)."""

    hidden_dims: tuple[int, ...] = (128, 64)
    activation: str = "relu"
    use_layer_norm: bool = True
    include_cosine_in_input: bool = True
    residual_weight: float = 0.25
    add_cosine_to_network_output: bool = True
    append_cosine_as_feature_to_network_input: bool = True
    num_cosine_terms: int = 128
    layer_type: str = "relu"
    num_cells: tuple[int, ...] = (128, 64)


@dataclasses.dataclass
class LossConfig:
    """Configuration for training loss functions and scoring."""

    loss_type: str = LossType.GE2E_SOFTMAX.value
    scoring_type: str = ScoringType.COSINE.value
    num_speakers_per_batch: int = 8
    num_utts_per_speaker: int = 4
    triplet_alpha: float = 0.1
    ge2e_learn_wb: bool = True
    ge2e_init_w: float = 10.0
    ge2e_init_b: float = -5.0
    ge2e_split_batch: bool = False
    ge2e_exact_leave_one_out: bool = True
    ge2e_vary_enroll_utts: bool = False
    ge2e_mask_dup_spk: bool = False
    multi_row_window: int = 0
    scale: float = 30.0
    margin: float = 0.2
    margin_scale: float = 30.0
    margin_m: float = 0.2
    sphereface_m: float = 1.35
    sphereface_lambda_min: float = 5.0
    sphereface_lambda_base: float = 1000.0
    sphereface_lambda_gamma: float = 0.12
    sphereface_lambda_power: float = 1.0
    label_smoothing: float = 0.0
    num_classes: int = 256


@dataclasses.dataclass
class ModelConfig:
    """Configuration for the complete speaker recognition model."""

    backbone: str = BackboneType.CONFORMER.value
    size_variant: str = ModelSize.SMALL.value
    embedding_dim: int = 192
    normalize_embedding: bool = True
    n_mfcc: int = 80
    seq_len: int = 200
    sliding_window_step: int = 50
    use_transformer: bool = False
    frame_aggregation_mean: bool = True
    full_sequence_inference: bool = True
    l2_normalize_embeddings: bool = True
    scoring_type: str = ScoringType.COSINE.value
    saved_model_path: str = ""
    lstm: LstmConfig = dataclasses.field(default_factory=LstmConfig)
    transformer: TransformerConfig = dataclasses.field(
        default_factory=TransformerConfig
    )
    conformer: ConformerConfig = dataclasses.field(default_factory=ConformerConfig)
    mamba: MambaConfig = dataclasses.field(default_factory=MambaConfig)
    ecapa_tdnn: EcapaTdnnConfig = dataclasses.field(default_factory=EcapaTdnnConfig)
    resnet: ResNetConfig = dataclasses.field(default_factory=ResNetConfig)
    pooling: PoolingConfig = dataclasses.field(default_factory=PoolingConfig)
    pfas: PfasConfig = dataclasses.field(default_factory=PfasConfig)
    dr_vector: DrVectorConfig = dataclasses.field(default_factory=DrVectorConfig)

    @property
    def output_embedding_dim(self) -> int:
        p_type = (
            self.pooling.pooling_type.value
            if hasattr(self.pooling.pooling_type, "value")
            else str(self.pooling.pooling_type).lower()
        )
        s_type = (
            self.scoring_type.value
            if hasattr(self.scoring_type, "value")
            else str(self.scoring_type).lower()
        )
        if p_type == "pfas" or s_type == "pfas":
            return self.pfas.representation_dim
        return self.embedding_dim

    def apply_size_variant(self, size: Union[str, ModelSize]) -> None:
        """Applies standard architectural dimensions (<100M params) for `size`."""
        size_val = size.value if isinstance(size, ModelSize) else str(size).lower()
        self.size_variant = size_val
        if size_val == ModelSize.TINY.value:
            self.embedding_dim = 128
            self.lstm.hidden_size = 128
            self.lstm.num_layers = 2
            self.transformer.dim = 128
            self.transformer.num_encoder_layers = 2
            self.transformer.num_heads = 4
            self.transformer.mlp_dim = 256
            self.conformer.dim = 128
            self.conformer.num_layers = 2
            self.conformer.num_heads = 4
            self.conformer.ffn_dim = 256
            self.mamba.dim = 128
            self.mamba.num_layers = 2
            self.ecapa_tdnn.channels = 128
            self.ecapa_tdnn.mfa_dim = 256
            self.ecapa_tdnn.mfa_channels = 256
            self.resnet.base_channels = 16
            self.resnet.num_blocks = (1, 1, 1, 1)
            self.resnet.stage_blocks = (1, 1, 1, 1)
            self.resnet.out_dim = 128
        elif size_val == ModelSize.SMALL.value:
            self.embedding_dim = 192
            self.lstm.hidden_size = 256
            self.lstm.num_layers = 3
            self.transformer.dim = 192
            self.transformer.num_encoder_layers = 3
            self.transformer.num_heads = 4
            self.transformer.mlp_dim = 384
            self.conformer.dim = 192
            self.conformer.num_layers = 3
            self.conformer.num_heads = 4
            self.conformer.ffn_dim = 384
            self.mamba.dim = 192
            self.mamba.num_layers = 3
            self.ecapa_tdnn.channels = 256
            self.ecapa_tdnn.mfa_dim = 512
            self.ecapa_tdnn.mfa_channels = 512
            self.resnet.base_channels = 24
            self.resnet.num_blocks = (2, 2, 2, 2)
            self.resnet.stage_blocks = (2, 2, 2, 2)
            self.resnet.out_dim = 192
        elif size_val == ModelSize.BASE.value:
            self.embedding_dim = 256
            self.lstm.hidden_size = 384
            self.lstm.num_layers = 4
            self.transformer.dim = 256
            self.transformer.num_encoder_layers = 4
            self.transformer.num_heads = 8
            self.transformer.mlp_dim = 512
            self.conformer.dim = 256
            self.conformer.num_layers = 4
            self.conformer.num_heads = 8
            self.conformer.ffn_dim = 512
            self.mamba.dim = 256
            self.mamba.num_layers = 4
            self.ecapa_tdnn.channels = 384
            self.ecapa_tdnn.mfa_dim = 768
            self.ecapa_tdnn.mfa_channels = 768
            self.resnet.base_channels = 32
            self.resnet.num_blocks = (2, 2, 2, 2)
            self.resnet.stage_blocks = (2, 2, 2, 2)
            self.resnet.out_dim = 256
        elif size_val == ModelSize.LARGE.value:
            self.embedding_dim = 512
            self.lstm.hidden_size = 512
            self.lstm.num_layers = 5
            self.transformer.dim = 384
            self.transformer.num_encoder_layers = 6
            self.transformer.num_heads = 8
            self.transformer.mlp_dim = 1024
            self.conformer.dim = 384
            self.conformer.num_layers = 6
            self.conformer.num_heads = 8
            self.conformer.ffn_dim = 1024
            self.mamba.dim = 384
            self.mamba.num_layers = 6
            self.ecapa_tdnn.channels = 512
            self.ecapa_tdnn.mfa_dim = 1024
            self.ecapa_tdnn.mfa_channels = 1024
            self.resnet.base_channels = 48
            self.resnet.num_blocks = (3, 4, 6, 3)
            self.resnet.stage_blocks = (3, 4, 6, 3)
            self.resnet.out_dim = 384
        else:
            raise ValueError(f"Unknown model size variant: {size_val}")


@dataclasses.dataclass
class DataConfig:
    """Configuration for dataset paths and sampling."""

    dataset_type: str = "librispeech"
    dataset_name: str = "librispeech"
    train_dir: str = ""
    test_dir: str = ""
    train_librispeech_dir: str = ""
    test_librispeech_dir: str = ""
    train_csv: str = ""
    test_csv: str = ""
    trials_file: str = ""
    audio_format: str = ".flac"
    speaker_label_index: int = -2


@dataclasses.dataclass
class TrainConfig:
    """Configuration for model training."""

    batch_size: int = 32
    num_spks_per_batch: int = 8
    num_utts_per_spk: int = 4
    triplet_alpha: float = 0.1
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    warmup_steps: int = 50
    lr_schedule_cosine: bool = True
    lr_schedule: str = "cosine"
    grad_clip_norm: float = 3.0
    ge2e_grad_scale: float = 0.01
    save_model_frequency: int = 500
    eval_frequency: int = 200
    num_steps: int = 500
    num_processes: int = 4
    seed: int = 42
    specaug: SpecAugConfig = dataclasses.field(default_factory=SpecAugConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)


@dataclasses.dataclass
class EvalConfig:
    """Configuration for speaker verification evaluation."""

    full_sequence_inference: bool = True
    num_triplets: int = 1000
    threshold_step: float = 0.001
    num_enroll_utts: int = 1
    use_as_norm: bool = False
    apply_as_norm: bool = False
    as_norm_top_k: int = 100
    as_norm_cohort_size: int = 100
    dcf_p_target: float = 0.01
    dcf_c_miss: float = 1.0
    dcf_c_fa: float = 1.0


@dataclasses.dataclass
class ExportConfig:
    """Configuration for model export (TFLite, SafeTensors, HuggingFace)."""

    output_dir: str = "exported_model"
    export_tflite: bool = True
    quantize_tflite: bool = True
    export_safetensors: bool = True
    export_hf_format: bool = True
    sequence_length: int = 100


@dataclasses.dataclass
class ExperimentConfig:
    """Top-level configuration for a FlaxSpeaker experiment."""

    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    vad: VadConfig = dataclasses.field(default_factory=VadConfig)
    frontend: FrontendConfig = dataclasses.field(default_factory=FrontendConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    export: ExportConfig = dataclasses.field(default_factory=ExportConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_munch(self) -> munch.Munch:
        return munch.munchify(self.to_dict())

    def to_yaml(self, path: Optional[str] = None) -> str:
        yaml_str = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Constructs an ExperimentConfig from a nested dictionary or legacy config."""
        cfg = cls()
        d = copy.deepcopy(dict(d))

        if "data" in d and isinstance(d["data"], dict):
            _update_dataclass(cfg.data, d["data"])
        if "vad" in d and isinstance(d["vad"], dict):
            _update_dataclass(cfg.vad, d["vad"])
        if "frontend" in d and isinstance(d["frontend"], dict):
            _update_dataclass(cfg.frontend, d["frontend"])
        if "loss" in d and isinstance(d["loss"], dict):
            _update_dataclass(cfg.loss, d["loss"])
            _update_dataclass(cfg.train.loss, d["loss"])
        if "eval" in d and isinstance(d["eval"], dict):
            _update_dataclass(cfg.eval, d["eval"])
        if "export" in d and isinstance(d["export"], dict):
            _update_dataclass(cfg.export, d["export"])

        if "train" in d and isinstance(d["train"], dict):
            train_d = dict(d["train"])
            if "specaug" in train_d and isinstance(train_d["specaug"], dict):
                _update_dataclass(cfg.train.specaug, train_d.pop("specaug"))
            if "loss" in train_d and isinstance(train_d["loss"], dict):
                _update_dataclass(cfg.train.loss, train_d.pop("loss"))
                _update_dataclass(cfg.loss, cfg.train.loss.__dict__)
            _update_dataclass(cfg.train, train_d)

        if "model" in d and isinstance(d["model"], dict):
            model_d = dict(d["model"])
            for sub_name in (
                "lstm",
                "transformer",
                "conformer",
                "mamba",
                "ecapa_tdnn",
                "resnet",
                "pooling",
                "pfas",
                "dr_vector",
            ):
                if sub_name in model_d and isinstance(model_d[sub_name], dict):
                    _update_dataclass(
                        getattr(cfg.model, sub_name), model_d.pop(sub_name)
                    )
            _update_dataclass(cfg.model, model_d)

            if "backbone" not in d["model"]:
                if cfg.model.use_transformer:
                    cfg.model.backbone = BackboneType.TRANSFORMER.value
                else:
                    cfg.model.backbone = BackboneType.LSTM.value
            if "frontend" not in d and "n_mfcc" in d["model"]:
                cfg.frontend.n_mfcc = cfg.model.n_mfcc
                cfg.frontend.n_mels = cfg.model.n_mfcc

        return cfg

    @classmethod
    def from_munch(cls, m: Any) -> ExperimentConfig:
        if isinstance(m, ExperimentConfig):
            return m
        return cls.from_dict(dict(m))

    @classmethod
    def from_yaml(cls, yaml_path_or_str: str) -> ExperimentConfig:
        if "\n" in yaml_path_or_str or not yaml_path_or_str.endswith(
            (".yml", ".yaml")
        ):
            try:
                with open(yaml_path_or_str, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
            except OSError:
                raw = yaml.safe_load(yaml_path_or_str)
        else:
            with open(yaml_path_or_str, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        return cls.from_dict(raw or {})

    @classmethod
    def from_any(
        cls, config: Union[ExperimentConfig, munch.Munch, dict[str, Any], str]
    ) -> ExperimentConfig:
        if isinstance(config, ExperimentConfig):
            return config
        if isinstance(config, str):
            return cls.from_yaml(config)
        if isinstance(config, (munch.Munch, dict)):
            return cls.from_dict(dict(config))
        raise TypeError(f"Unsupported config type: {type(config)}")

    def apply_size_variant(self, size: Union[str, ModelSize]) -> None:
        self.model.apply_size_variant(size)


def _update_dataclass(instance: Any, updates: dict[str, Any]) -> None:
    valid_fields = {f.name for f in dataclasses.fields(instance)}
    for k, v in updates.items():
        if k in valid_fields:
            current = getattr(instance, k)
            if isinstance(current, tuple) and isinstance(v, list):
                setattr(instance, k, tuple(v))
            else:
                setattr(instance, k, v)
