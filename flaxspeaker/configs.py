"""Configuration dataclasses, enums, and presets for FlaxSpeaker."""

from __future__ import annotations

import copy
import dataclasses
import enum
from typing import Any, Optional, Union
import warnings
import yaml


class _StrEnum(str, enum.Enum):
    """String enum whose `str(x)` always returns `x.value` on Python 3.11+."""

    def __str__(self) -> str:
        return str(self.value)


yaml.SafeDumper.add_multi_representer(
    _StrEnum, yaml.representer.SafeRepresenter.represent_str
)


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
    WAVLM = "wavlm"


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
    ATP = "atp"
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
    SUBCENTER_ARCFACE = "subcenter_arcface"
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
    apply_agc: bool = True
    energy_threshold_db: float = -42.0
    energy_db_threshold: float = 26.0
    adaptive_percentile: float = 10.0
    snr_margin_db: float = 4.5
    spectral_flatness_threshold: float = 0.65
    min_speech_frames: int = 3
    hangover_frames: int = 8


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
    kaldi_fbank: bool = False
    hf_extractor_name: str = "openai/whisper-tiny"

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.frame_step_ms / 1000.0))

    @property
    def win_length(self) -> int:
        return int(round(self.sample_rate * self.frame_size_ms / 1000.0))

    @property
    def feature_dim(self) -> int:
        ftype = str(getattr(self.feature_type, "value", self.feature_type)).lower()
        base_dim = self.n_mfcc if ftype == FeatureType.MFCC.value else self.n_mels
        return base_dim * (1 + self.stack_left_context + self.stack_right_context)


@dataclasses.dataclass
class SpecAugConfig:
    """Configuration for SpecAugment data augmentation."""

    use_specaug: bool = True
    apply_prob: float = 0.5
    freq_mask_prob: float = 0.5
    time_mask_prob: float = 0.5
    freq_mask_max_width: int = 8
    time_mask_max_width: int = 12
    num_freq_masks: int = 2
    num_time_masks: int = 2
    gaussian_noise_std: float = 0.0


@dataclasses.dataclass
class WaveformAugConfig:
    """Configuration for Multi-Style Training (MTR) augmentation.

    Supports both feature-domain MTR (`specaug.apply_feature_mtr_augmentation`)
    and real corpus waveform augmentation (`waveform_aug.MusanRirAugmenter` with
    MUSAN noise/music/babble and RIRS_NOISES room impulse responses).
    """

    enabled: bool = True
    use_waveform_aug: bool = True
    aug_prob: float = 0.35
    snr_min_db: float = 12.0
    snr_max_db: float = 20.0
    reverb_prob: float = 0.4
    reverb_rt60_min: float = 0.05
    reverb_rt60_max: float = 0.20
    musan_dir: str = ""
    rirs_dir: str = ""
    noise_snr_range: tuple[float, float] = (0.0, 15.0)
    music_snr_range: tuple[float, float] = (5.0, 15.0)
    babble_snr_range: tuple[float, float] = (13.0, 20.0)
    babble_spk_range: tuple[int, int] = (3, 7)


@dataclasses.dataclass
class LstmConfig:
    """Configuration for LSTM speaker encoder (no dropout is implemented)."""

    hidden_size: int = 256
    proj_size: int = 0
    num_layers: int = 3
    bidirectional: bool = False
    use_layer_norm: bool = True
    frame_aggregation_mean: bool = True


@dataclasses.dataclass
class TransformerConfig:
    """Configuration for Transformer speaker encoder (no dropout is implemented)."""

    dim: int = 256
    num_encoder_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 512
    use_positional_encoding: bool = False
    use_conv_stem: bool = True


@dataclasses.dataclass
class ConformerConfig:
    """Configuration for the Conformer speaker encoder.

    `layer_order` selects the block ordering inside each Conformer block:
    `"mhsa_before_conv"` (Gulati et al., FlaxSpeaker default) or
    `"conv_before_mhsa"`. `subsample_factor > 1` applies stack-by-`subsample_factor`
    time reduction (concatenate adjacent frames + linear projection back to `dim`)
    after `subsample_after_layer` Conformer blocks. No dropout is implemented.
    """

    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 512
    conv_kernel_size: int = 15
    subsample_factor: int = 1
    subsample_after_layer: int = 3
    layer_order: str = "mhsa_before_conv"
    use_group_norm: bool = True
    use_conv_stem: bool = True
    use_positional_encoding: bool = False
    normalize_output: bool = False
    resnet_stem_channels: int = 0


@dataclasses.dataclass
class MambaConfig:
    """Configuration for Mamba (Selective State Space Model) speaker encoder."""

    dim: int = 256
    num_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: int = 16
    bidirectional: bool = False


@dataclasses.dataclass
class EcapaTdnnConfig:
    """Configuration for ECAPA-TDNN speaker encoder."""

    channels: int = 256
    scale: int = 4
    kernel_sizes: tuple[int, ...] = (5, 3, 3, 3)
    dilations: tuple[int, ...] = (1, 2, 3, 4)
    se_bottleneck_dim: int = 64
    mfa_dim: int = 512
    use_batch_norm: bool = True
    norm_after_activation: bool = True
    freq_resnet_channels: int = 0


@dataclasses.dataclass
class ResNetConfig:
    """Configuration for Thin-ResNet / ResNet-18 / ResNet-34 speaker encoder."""

    base_channels: int = 32
    num_blocks: tuple[int, ...] = (2, 2, 2, 2)
    use_se: bool = True
    out_dim: int = 256
    use_batch_norm: bool = True
    wespeaker_mode: bool = False


@dataclasses.dataclass
class WavLMConfig:
    """Configuration for WavLM self-supervised speaker encoder (Chen et al., JSTSP 2022)."""

    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    num_buckets: int = 320
    max_distance: int = 800
    pos_conv_kernel: int = 128
    pos_conv_groups: int = 16
    subsample_factor: int = 2
    tdnn_dim: int = 512
    tdnn_out_dim: int = 1500
    acoustic_resnet_channels: int = 0


@dataclasses.dataclass
class PoolingConfig:
    """Configuration for temporal frame aggregation / pooling.

    For `ATP`: `attention_dim > 0` inserts a tanh hidden layer before the sigmoid
    weight projection (FlaxSpeaker extension); `attention_dim = 0` and
    `num_heads = 1` recover the single-head linear-sigmoid ATP of Wang et al. 2022 /
    Pelecanos et al. 2022. `subspace_heads = True` splits the D channels across heads
    (FlaxSpeaker extension); `False` pools all D channels with every head.
    """

    pooling_type: str = PoolingType.ATP.value
    attention_dim: int = 128
    use_global_context: bool = True
    stats_type: str = "MEAN_STD"
    num_heads: int = 4
    subspace_heads: bool = True
    epsilon: float = 1e-4
    bottleneck_dim: int = 256
    use_nonlinear_bottleneck: bool = True
    use_bottleneck_ln: bool = True
    asp_relu_bn: bool = False
    unbiased_std: bool = False


@dataclasses.dataclass
class PfasConfig:
    """Configuration for Parameter-Free Attentive Scoring (Pelecanos et al., Odyssey 2022)."""

    num_keys: int = 4
    key_dim: int = 32
    value_dim: int = 128
    attention_dim: int = 128
    attention_type: str = "shared_nonlinear"
    attention_weight_mode: str = "exp"
    atp_ffn_use_layer_norm: bool = True
    scale_factor: float = 16.0
    temperature: float = 0.0
    use_trainable_scale_factor: bool = True
    normalize_keys_and_values: bool = False
    normalize_overall_output: bool = False
    per_trial_softmax: bool = True
    apply_l2_norm_to_keys: bool = True
    apply_l2_norm_to_values: bool = False
    apply_global_l2_norm_to_concat_form: bool = True
    apply_softmax_per_test_key: bool = False
    use_global_anchor: bool = True
    use_stats_global_anchor: bool = False
    local_value_scale: float = 0.25

    @property
    def representation_dim(self) -> int:
        return self.num_keys * (self.key_dim + self.value_dim)

    @property
    def effective_scale_factor(self) -> float:
        if self.temperature > 0.0:
            return 1.0 / float(self.temperature)
        return float(self.scale_factor)


@dataclasses.dataclass
class DrVectorConfig:
    """Configuration for Decision Residual Networks (Pelecanos et al., Interspeech 2021)."""

    hidden_dims: tuple[int, ...] = (128, 64)
    activation: str = "relu"
    leaky_relu_slope: float = 0.2
    use_layer_norm: bool = True
    include_cosine_in_input: bool = True
    residual_weight: float = 0.25
    add_cosine_to_network_output: bool = True
    append_cosine_as_feature_to_network_input: bool = True
    append_cosine_to_final_hidden: bool = False
    num_cosine_terms: int = 1
    symmetric_input: bool = False
    cosine_dims: int = 0
    readout: str = "linear"
    trainable_affine: bool = False

    @property
    def skip_connect_cosine_to_readout(self) -> bool:
        return self.append_cosine_to_final_hidden

    @skip_connect_cosine_to_readout.setter
    def skip_connect_cosine_to_readout(self, value: bool) -> None:
        self.append_cosine_to_final_hidden = bool(value)


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
    ge2e_contrast_surrogate: bool = False
    ge2e_contrast_b_grad_scale: float = 0.0
    multi_row_window: int = 0
    pure_extended_set: bool = False
    ext_softmax_normalize_by_num_speakers: bool = True
    ext_softmax_row_softmax_weight: float = 0.5
    scale: float = 30.0
    margin: float = 0.2
    subcenter_k: int = 3
    sphereface_m: float = 1.35
    sphereface_lambda_min: float = 5.0
    sphereface_lambda_base: float = 1000.0
    sphereface_lambda_gamma: float = 0.12
    sphereface_lambda_power: float = 1.0
    margin_warmup_steps: int = 0


@dataclasses.dataclass
class ModelConfig:
    """Configuration for the complete speaker recognition model."""

    backbone: str = BackboneType.CONFORMER.value
    size_variant: str = ModelSize.SMALL.value
    embedding_dim: int = 192
    normalize_embedding: bool = True
    n_mfcc: int = 80
    seq_len: int = 200
    min_seq_len: int = 80
    max_seq_len: int = 200
    max_eval_seq_len: int = 0
    variable_length_training: bool = True
    sliding_window_step: int = 50
    full_sequence_inference: bool = True
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
    wavlm: WavLMConfig = dataclasses.field(default_factory=WavLMConfig)
    pooling: PoolingConfig = dataclasses.field(default_factory=PoolingConfig)
    pfas: PfasConfig = dataclasses.field(default_factory=PfasConfig)
    dr_vector: DrVectorConfig = dataclasses.field(default_factory=DrVectorConfig)

    @property
    def output_embedding_dim(self) -> int:
        p_type = str(
            getattr(self.pooling.pooling_type, "value", self.pooling.pooling_type)
        ).lower()
        s_type = str(
            getattr(self.scoring_type, "value", self.scoring_type)
        ).lower()
        if p_type == "pfas" or s_type == "pfas":
            return self.pfas.representation_dim
        return self.embedding_dim

    def apply_size_variant(self, size: Union[str, ModelSize]) -> None:
        """Applies standard architectural dimensions (<100M params) for `size`."""
        size_val = size.value if isinstance(size, ModelSize) else str(size).lower()
        if size_val == "c1024":
            size_val = ModelSize.LARGE.value
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
            self.resnet.base_channels = 16
            self.resnet.num_blocks = (1, 1, 1, 1)
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
            self.resnet.base_channels = 24
            self.resnet.num_blocks = (2, 2, 2, 2)
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
            self.resnet.base_channels = 32
            self.resnet.num_blocks = (2, 2, 2, 2)
            self.resnet.out_dim = 256
        elif size_val == ModelSize.LARGE.value:
            self.embedding_dim = 256
            self.lstm.hidden_size = 512
            self.lstm.num_layers = 5
            self.transformer.dim = 384
            self.transformer.num_encoder_layers = 6
            self.transformer.num_heads = 6
            self.transformer.mlp_dim = 1536
            self.conformer.dim = 320
            self.conformer.num_layers = 4
            self.conformer.num_heads = 8
            self.conformer.ffn_dim = 640
            self.conformer.conv_kernel_size = 15
            self.conformer.subsample_factor = 1
            self.conformer.subsample_after_layer = 4
            self.mamba.dim = 384
            self.mamba.num_layers = 6
            self.ecapa_tdnn.channels = 1024
            self.ecapa_tdnn.scale = 8
            self.ecapa_tdnn.kernel_sizes = (5, 3, 3, 3)
            self.ecapa_tdnn.dilations = (1, 2, 3, 4)
            self.ecapa_tdnn.mfa_dim = 1536
            self.ecapa_tdnn.se_bottleneck_dim = 128
            self.ecapa_tdnn.use_batch_norm = True
            self.ecapa_tdnn.norm_after_activation = True
            self.resnet.base_channels = 48
            self.resnet.num_blocks = (3, 4, 6, 3)
            self.resnet.out_dim = 384
        else:
            raise ValueError(f"Unknown model size variant: {size_val}")


@dataclasses.dataclass
class DataConfig:
    """Configuration for dataset paths, sharded caches, and sampling."""

    dataset_type: str = "librispeech"
    dataset_name: str = "librispeech"
    train_dir: str = ""
    test_dir: str = ""
    train_librispeech_dir: str = ""
    test_librispeech_dir: str = ""
    train_csv: str = ""
    test_csv: str = ""
    trials_file: str = ""
    sharded_cache_dir: str = ""
    musan_dir: str = ""
    rirs_dir: str = ""
    speed_perturb: bool = False
    speed_factors: tuple[float, ...] = (1.0, 0.9, 1.1)
    audio_format: str = ".flac"
    speaker_label_index: int = -2
    within_domain_prob: float = 0.75
    domain_temp_alpha: float = 0.35


@dataclasses.dataclass
class TrainConfig:
    """Configuration for model training, warm-starting, and LMFT."""

    batch_size: int = 32
    num_spks_per_batch: int = 8
    num_utts_per_spk: int = 4
    triplet_alpha: float = 0.1
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    optimizer: str = "adamw"
    sgd_momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 1e-4
    warmup_steps: int = 50
    lr_schedule_cosine: bool = True
    grad_clip_norm: float = 3.0
    ge2e_grad_scale: float = 0.01
    init_from_checkpoint: str = ""
    freeze_trunk: bool = False
    trunk_lr_mult: float = 1.0
    save_model_frequency: int = 500
    num_steps: int = 500
    num_processes: int = 4
    seed: int = 42
    saved_model_dir: str = ""
    specaug: SpecAugConfig = dataclasses.field(default_factory=SpecAugConfig)
    waveform_aug: WaveformAugConfig = dataclasses.field(
        default_factory=WaveformAugConfig
    )
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)


@dataclasses.dataclass
class EvalConfig:
    """Configuration for speaker verification evaluation."""

    full_sequence_inference: bool = True
    max_eval_frames_cap: int = 3000
    num_triplets: int = 1000
    threshold_step: float = 0.001
    num_enroll_utts: int = 1
    use_as_norm: bool = False
    apply_as_norm: bool = False
    as_norm_top_k: int = 300
    as_norm_cohort_size: int = 300
    use_qmf: bool = False
    dcf_c_miss: float = 1.0
    dcf_c_fa: float = 1.0


@dataclasses.dataclass
class ExportConfig:
    """Configuration for fixed-window TFLite / SavedModel export."""

    output_dir: str = "exported_model"
    export_tflite: bool = True
    sequence_length: int = 0


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

    def __post_init__(self) -> None:
        default_loss = LossConfig()
        if self.loss != default_loss and self.train.loss == default_loss:
            self.train.loss = self.loss
        else:
            self.loss = self.train.loss

    def to_dict(self) -> dict[str, Any]:
        self.loss = self.train.loss
        return dataclasses.asdict(self)

    def to_yaml(self, path: Optional[str] = None) -> str:
        yaml_str = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Constructs an ExperimentConfig from a nested dictionary."""
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
            if "waveform_aug" in train_d and isinstance(
                train_d["waveform_aug"], dict
            ):
                _update_dataclass(
                    cfg.train.waveform_aug, train_d.pop("waveform_aug")
                )
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
                "wavlm",
                "pooling",
                "pfas",
                "dr_vector",
            ):
                if sub_name in model_d and isinstance(model_d[sub_name], dict):
                    _update_dataclass(
                        getattr(cfg.model, sub_name), model_d.pop(sub_name)
                    )
            _update_dataclass(cfg.model, model_d)
            if "frontend" not in d and "n_mfcc" in d["model"]:
                cfg.frontend.n_mfcc = cfg.model.n_mfcc
                cfg.frontend.n_mels = cfg.model.n_mfcc

        cfg.loss = cfg.train.loss
        return cfg

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
        cls, config: Union[ExperimentConfig, dict[str, Any], str]
    ) -> ExperimentConfig:
        if isinstance(config, ExperimentConfig):
            return config
        if isinstance(config, str):
            return cls.from_yaml(config)
        if isinstance(config, dict):
            return cls.from_dict(dict(config))
        raise TypeError(f"Unsupported config type: {type(config)}")

    def apply_size_variant(self, size: Union[str, ModelSize]) -> None:
        self.model.apply_size_variant(size)


_DEPRECATED_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "WaveformAugConfig": frozenset({"noise_dir"}),
    "LstmConfig": frozenset({"dropout_rate"}),
    "TransformerConfig": frozenset({"dropout_rate", "use_rope"}),
    "ConformerConfig": frozenset({"ffn_expansion", "dropout_rate", "num_groups"}),
    "MambaConfig": frozenset({"state_dim", "conv_kernel_size", "dropout_rate"}),
    "EcapaTdnnConfig": frozenset({"attention_channels", "se_channels", "mfa_channels"}),
    "ResNetConfig": frozenset({"stage_blocks", "se_reduction"}),
    "PoolingConfig": frozenset(
        {
            "share_params",
            "nonlinear_attention",
            "divide_layer",
            "max_pooling_mode",
            "top_k",
            "window_size",
            "window_stride",
            "use_weighted_frames",
        }
    ),
    "PfasConfig": frozenset({"use_keys_and_queries"}),
    "DrVectorConfig": frozenset({"layer_type", "num_cells"}),
    "LossConfig": frozenset(
        {
            "ge2e_vary_enroll_utts",
            "ge2e_mask_dup_spk",
            "margin_scale",
            "margin_m",
            "label_smoothing",
            "num_classes",
        }
    ),
    "ModelConfig": frozenset(
        {"l2_normalize_embeddings", "use_transformer", "frame_aggregation_mean"}
    ),
    "TrainConfig": frozenset({"lr_schedule", "eval_frequency"}),
    "EvalConfig": frozenset({"dcf_p_target"}),
    "ExportConfig": frozenset(
        {"quantize_tflite", "export_safetensors", "export_hf_format"}
    ),
}


def _update_dataclass(instance: Any, updates: dict[str, Any]) -> None:
    """Applies `updates` to a config dataclass; raises on unknown keys."""
    valid_fields = {f.name for f in dataclasses.fields(instance)}
    cls_name = type(instance).__name__
    deprecated = _DEPRECATED_CONFIG_KEYS.get(cls_name, frozenset())
    for k, v in updates.items():
        if k in valid_fields:
            current = getattr(instance, k)
            if isinstance(current, tuple) and isinstance(v, list):
                setattr(instance, k, tuple(v))
            else:
                setattr(instance, k, v)
        elif hasattr(type(instance), k) and isinstance(
            getattr(type(instance), k), property
        ):
            setattr(instance, k, v)
        elif k in deprecated:
            warnings.warn(
                f"Config key '{cls_name}.{k}' is deprecated and has no effect; "
                "it is ignored. Remove it from your config.",
                DeprecationWarning,
                stacklevel=3,
            )
        else:
            raise ValueError(
                f"Unknown config key '{k}' for {cls_name}. "
                f"Valid keys: {sorted(valid_fields)}"
            )


def icassp2018_ge2e_config(
    size: Union[str, ModelSize] = ModelSize.SMALL,
    pure_paper_mode: bool = True,
    paper_faithful: Optional[bool] = None,
) -> ExperimentConfig:
    """Preset for Generalized End-to-End (GE2E) loss (Wan et al., ICASSP 2018)."""
    if paper_faithful is not None:
        pure_paper_mode = bool(paper_faithful)
    cfg = ExperimentConfig()
    cfg.model.backbone = BackboneType.LSTM.value
    cfg.model.scoring_type = ScoringType.COSINE.value
    cfg.model.apply_size_variant(size)
    if pure_paper_mode:
        cfg.model.pooling.pooling_type = PoolingType.LAST.value
        cfg.model.lstm.hidden_size = 768
        cfg.model.lstm.proj_size = 256
        cfg.model.lstm.num_layers = 3
        cfg.model.lstm.bidirectional = False
        cfg.model.embedding_dim = 256
        cfg.model.pooling.use_nonlinear_bottleneck = False
    else:
        cfg.model.pooling.pooling_type = PoolingType.ASP.value
    cfg.train.loss.loss_type = LossType.GE2E_SOFTMAX.value
    cfg.train.loss.scoring_type = ScoringType.COSINE.value
    cfg.train.loss.ge2e_exact_leave_one_out = True
    cfg.train.loss.ge2e_init_w = 10.0
    cfg.train.loss.ge2e_init_b = -5.0
    cfg.train.ge2e_grad_scale = 0.01
    cfg.loss = cfg.train.loss
    return cfg


def interspeech2021_dr_vectors_config(
    size: Union[str, ModelSize] = ModelSize.SMALL,
    backbone: str = BackboneType.CONFORMER.value,
    pure_paper_mode: bool = True,
    paper_faithful: Optional[bool] = None,
) -> ExperimentConfig:
    """Preset for Dr-Vectors & GE2E extended-set softmax (Pelecanos et al., Interspeech 2021)."""
    if paper_faithful is not None:
        pure_paper_mode = bool(paper_faithful)
    cfg = ExperimentConfig()
    cfg.model.backbone = backbone
    cfg.model.pooling.pooling_type = (
        PoolingType.ATP.value
        if backbone in ("conformer", "transformer")
        else PoolingType.ASP.value
    )
    cfg.model.scoring_type = ScoringType.DR_VECTOR.value
    cfg.model.apply_size_variant(size)
    if pure_paper_mode:
        cfg.model.embedding_dim = 256
        dr = cfg.model.dr_vector
        dr.cosine_dims = 200
        dr.hidden_dims = (256, 256, 256)
        dr.activation = "leaky_relu"
        dr.leaky_relu_slope = 0.2
        dr.use_layer_norm = False
        dr.readout = "weighted_sum"
        dr.trainable_affine = True
        dr.residual_weight = 1.0
        dr.num_cosine_terms = 1
        dr.append_cosine_to_final_hidden = False
        dr.append_cosine_as_feature_to_network_input = True
        dr.add_cosine_to_network_output = True
        cfg.train.loss.pure_extended_set = True
        cfg.train.loss.ext_softmax_row_softmax_weight = 0.0
        cfg.train.loss.ext_softmax_normalize_by_num_speakers = False
    cfg.train.loss.loss_type = LossType.EXTENDED_SET_SOFTMAX.value
    cfg.train.loss.scoring_type = ScoringType.DR_VECTOR.value
    cfg.loss = cfg.train.loss
    return cfg


def odyssey2022_conformer_atp_config(
    size: Union[str, ModelSize] = ModelSize.SMALL,
    pure_paper_mode: bool = True,
    paper_faithful: Optional[bool] = None,
) -> ExperimentConfig:
    """Preset for Conformer + Attentive Temporal Pooling (Wang et al., Odyssey 2022)."""
    if paper_faithful is not None:
        pure_paper_mode = bool(paper_faithful)
    cfg = ExperimentConfig()
    cfg.model.backbone = BackboneType.CONFORMER.value
    cfg.model.pooling.pooling_type = PoolingType.ATP.value
    cfg.model.scoring_type = ScoringType.COSINE.value
    cfg.model.apply_size_variant(size)
    if pure_paper_mode:
        cfg.model.conformer.layer_order = "conv_before_mhsa"
        cfg.model.conformer.use_conv_stem = False
        cfg.model.conformer.use_positional_encoding = True
        cfg.model.conformer.subsample_factor = 2
        cfg.model.conformer.subsample_after_layer = 3
        cfg.model.pooling.num_heads = 1
        cfg.model.pooling.attention_dim = 0
        cfg.model.pooling.subspace_heads = False
    cfg.train.loss.loss_type = LossType.GE2E_SOFTMAX.value
    cfg.train.loss.scoring_type = ScoringType.COSINE.value
    cfg.loss = cfg.train.loss
    return cfg


def odyssey2022_pfas_config(
    size: Union[str, ModelSize] = ModelSize.SMALL,
    loss_type: str = LossType.EXTENDED_SET_SOFTMAX.value,
    pure_paper_mode: bool = True,
    paper_faithful: Optional[bool] = None,
) -> ExperimentConfig:
    """Preset for Parameter-Free Attentive Scoring (Pelecanos et al., Odyssey 2022)."""
    if paper_faithful is not None:
        pure_paper_mode = bool(paper_faithful)
    cfg = ExperimentConfig()
    cfg.model.backbone = BackboneType.CONFORMER.value
    cfg.model.pooling.pooling_type = PoolingType.PFAS.value
    cfg.model.scoring_type = ScoringType.PFAS.value
    cfg.model.apply_size_variant(size)
    if pure_paper_mode:
        pf = cfg.model.pfas
        pf.attention_type = "atp_ffn"
        pf.atp_ffn_use_layer_norm = False
        pf.num_keys = 32
        pf.key_dim = 16
        pf.value_dim = 48
        pf.use_global_anchor = False
        pf.apply_l2_norm_to_keys = True
        pf.apply_l2_norm_to_values = False
        pf.apply_global_l2_norm_to_concat_form = True
        pf.use_trainable_scale_factor = True
        cfg.model.pooling.num_heads = 1
        cfg.model.pooling.attention_dim = 0
        cfg.model.pooling.subspace_heads = False
        cfg.model.pooling.bottleneck_dim = 512
        if loss_type == LossType.EXTENDED_SET_SOFTMAX.value:
            cfg.train.loss.pure_extended_set = True
            cfg.train.loss.ext_softmax_row_softmax_weight = 0.0
            cfg.train.loss.ext_softmax_normalize_by_num_speakers = False
    cfg.model.embedding_dim = cfg.model.pfas.representation_dim
    cfg.train.loss.loss_type = loss_type
    cfg.train.loss.scoring_type = ScoringType.PFAS.value
    cfg.loss = cfg.train.loss
    return cfg


icassp2018_lstm_ge2e_config = icassp2018_ge2e_config
interspeech2021_lstm_dr_vectors_config = interspeech2021_dr_vectors_config
odyssey2022_conformer_pfas_config = odyssey2022_pfas_config


def universal_multi_domain_config(
    backbone: str = BackboneType.CONFORMER.value,
    size: Union[str, ModelSize] = ModelSize.SMALL,
    loss_type: str = LossType.GE2E_SOFTMAX.value,
    scoring_type: str = ScoringType.COSINE.value,
) -> ExperimentConfig:
    """Preset for multi-domain universal speaker recognition across LibriSpeech, CN-Celeb2, and VoxCeleb1."""
    cfg = ExperimentConfig()
    cfg.model.backbone = backbone
    if scoring_type == ScoringType.PFAS.value:
        cfg.model.pooling.pooling_type = PoolingType.PFAS.value
    elif backbone in ("conformer", "transformer"):
        cfg.model.pooling.pooling_type = PoolingType.ATP.value
    else:
        cfg.model.pooling.pooling_type = PoolingType.ASP.value
    cfg.model.scoring_type = scoring_type
    cfg.model.apply_size_variant(size)
    if scoring_type == ScoringType.PFAS.value:
        cfg.model.embedding_dim = cfg.model.pfas.representation_dim
    cfg.train.loss.loss_type = loss_type
    cfg.train.loss.scoring_type = scoring_type
    cfg.loss = cfg.train.loss
    return cfg
