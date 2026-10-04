---
library_name: flaxspeaker
tags:
- audio
- speaker-verification
- speaker-recognition
- jax
- flax
---
# FlaxSpeaker Model (`ecapa_tdnn` - `base`)

## Model Summary
- **Backbone**: `ecapa_tdnn` (`base`)
- **Pooling**: `asp`
- **Scoring**: `cosine`
- **Loss Function**: `ge2e_softmax`
- **Output Embedding Dimension**: `256`
- **Parameter Count**: `3,124,322`

## Usage
```python
from flaxspeaker.hf_compat import FlaxSpeakerModel

model = FlaxSpeakerModel.from_pretrained("/usr/local/google/home/quanw/Code/github/FlaxSpeaker/pretrained_models/ls_ecapa_tdnn_base_ge2e")
emb = model.extract_embedding("path/to/audio.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
```

## Evaluation Results
```json
{
  "experiment_name": "ls_ecapa_tdnn_base_ge2e",
  "train_dataset": "librispeech",
  "backbone": "ecapa_tdnn",
  "size_variant": "base",
  "pooling_type": "asp",
  "loss_type": "ge2e_softmax",
  "scoring_type": "cosine",
  "embedding_dim": 256,
  "num_parameters": 3124322,
  "num_steps": 200,
  "train_time_sec": 134.74,
  "initial_loss": 2.0595,
  "final_loss": 0.7359,
  "in_domain_eval": {
    "eer": 0.17266666666666666,
    "eer_percent": 17.2667,
    "eer_threshold": 0.6754601001739502,
    "min_dcf_001": 0.9647,
    "min_dcf_005": 0.94,
    "roc_auc": 0.9064,
    "num_trials": 3000,
    "num_unique_utterances": 999,
    "latency_ms_per_utterance": 17.334,
    "eval_time_sec": 17.357,
    "scoring_type": "cosine"
  },
  "multi_enroll_3utt_eval": {
    "eer": 0.14800000000000002,
    "eer_percent": 14.8,
    "eer_threshold": 0.7341094613075256,
    "min_dcf_001": 0.966,
    "roc_auc": 0.9352,
    "num_trials": 1000
  },
  "cross_dataset_eval": {
    "voxceleb1_o_official": {
      "eer": 0.27749999999999997,
      "eer_percent": 27.75,
      "eer_threshold": 0.5674952864646912,
      "min_dcf_001": 0.995,
      "min_dcf_005": 0.9935,
      "roc_auc": 0.8035,
      "num_trials": 4000,
      "num_unique_utterances": 3892,
      "latency_ms_per_utterance": 10.826,
      "eval_time_sec": 42.185,
      "scoring_type": "cosine"
    },
    "cnceleb2_zero_shot": {
      "eer": 0.35300000000000004,
      "eer_percent": 35.3,
      "eer_threshold": 0.6331055164337158,
      "min_dcf_001": 0.9947,
      "min_dcf_005": 0.9947,
      "roc_auc": 0.7139,
      "num_trials": 3000,
      "num_unique_utterances": 739,
      "latency_ms_per_utterance": 14.183,
      "eval_time_sec": 10.514,
      "scoring_type": "cosine"
    }
  },
  "config_yaml": "/usr/local/google/home/quanw/Code/github/FlaxSpeaker/configs/ls_ecapa_tdnn_base_ge2e.yml"
}
```
