"""FlaxSpeaker: Modernized JAX/Flax Speaker Recognition & Verification Library.

Features:
- Neural Backbones: LSTM, Transformer, Conformer, Mamba (Selective SSM),
  ECAPA-TDNN, ResNet-34 / Thin-ResNet (`flaxspeaker.backbones`)
- Temporal Pooling: Mean, Last, Stats, SAP, ASP, Cumulative Stats, PFAS
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

__all__ = [
    "backbones",
    "configs",
    "dataset",
    "evaluation",
    "export",
    "feature_extraction",
    "generate_csv",
    "hf_compat",
    "losses",
    "neural_net",
    "pooling",
    "scoring",
    "specaug",
    "vad",
]
