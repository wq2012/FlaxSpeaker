"""FlaxSpeaker: Modernized JAX/Flax Speaker Recognition & Verification Library.

Features:
- Neural Backbones: LSTM, Transformer, Conformer, Mamba (Selective SSM),
  ECAPA-TDNN, ResNet-34 / Thin-ResNet (`flaxspeaker.backbones`)
- Temporal Pooling: Mean, Last, Stats, SAP, ASP, ATP, Cumulative Stats, PFAS
  (`flaxspeaker.pooling`)
- Losses & Scoring:
  - Generalized End-to-End Loss (GE2E Softmax & Contrast)
  - Parameter-Free Attentive Scoring (PFAS)
  - Decision Residual Networks & Extended-Set Softmax Loss (Dr-Vectors)
  - ArcFace (AAM-Softmax), CosFace (AM-Softmax), SphereFace (A-Softmax),
    Softmax, Pairwise, and Triplet Loss (`flaxspeaker.losses`, `flaxspeaker.scoring`)
- Audio Frontend & VAD: Log-Mel, Whisper-Mel, HuggingFace `transformers`
  `AutoFeatureExtractor` compatibility, MFCC, and energy/spectral/hybrid VAD
  (`flaxspeaker.feature_extraction`, `flaxspeaker.vad`)
- Model Export & Ecosystem: TFLite (with int8 quantization), TensorFlow
  SavedModel, SafeTensors, and HuggingFace `PretrainedConfig` / `save_pretrained` /
  `from_pretrained` (`flaxspeaker.export`, `flaxspeaker.hf_compat`)
- 44 Pretrained Models on Hugging Face Hub (`wq2012/<model_id>`):
  https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595
"""

from flaxspeaker import backbones
from flaxspeaker import configs
from flaxspeaker import dataset
from flaxspeaker import evaluation
from flaxspeaker import export
from flaxspeaker import feature_extraction
from flaxspeaker import generate_csv
from flaxspeaker import hf_compat
from flaxspeaker import losses
from flaxspeaker import neural_net
from flaxspeaker import pooling
from flaxspeaker import scoring
from flaxspeaker import specaug
from flaxspeaker import vad

__version__ = "0.2.0"

HF_DEFAULT_NAMESPACE = hf_compat.HF_DEFAULT_NAMESPACE
HF_COLLECTION_URL = hf_compat.HF_COLLECTION_URL
PRETRAINED_MODEL_IDS = hf_compat.PRETRAINED_MODEL_IDS
HF_MODEL_REGISTRY = hf_compat.HF_MODEL_REGISTRY
FlaxSpeakerConfig = hf_compat.FlaxSpeakerConfig
FlaxSpeakerModel = hf_compat.FlaxSpeakerModel


def load_pretrained(
    model_dir_or_repo: str, **kwargs
) -> hf_compat.FlaxSpeakerModel:
    """Loads a pretrained FlaxSpeaker model from a local directory or Hugging Face repo (`wq2012/<model_id>`)."""
    return hf_compat.FlaxSpeakerModel.from_pretrained(
        model_dir_or_repo, **kwargs
    )


__all__ = [
    "FlaxSpeakerConfig",
    "FlaxSpeakerModel",
    "HF_COLLECTION_URL",
    "HF_DEFAULT_NAMESPACE",
    "HF_MODEL_REGISTRY",
    "PRETRAINED_MODEL_IDS",
    "backbones",
    "configs",
    "dataset",
    "evaluation",
    "export",
    "feature_extraction",
    "generate_csv",
    "hf_compat",
    "load_pretrained",
    "losses",
    "neural_net",
    "pooling",
    "scoring",
    "specaug",
    "vad",
]
