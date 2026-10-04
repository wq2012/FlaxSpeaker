---
library_name: flaxspeaker
tags:
- audio
- speaker-verification
- speaker-recognition
- jax
- flax
---
# FlaxSpeaker Model (`ecapa_tdnn` - `tiny`)

## Model Summary
- **Backbone**: `ecapa_tdnn` (`tiny`)
- **Pooling**: `asp`
- **Scoring**: `cosine`
- **Loss Function**: `ge2e_softmax`
- **Output Embedding Dimension**: `128`
- **Parameter Count**: `527,650`

## Usage
```python
from flaxspeaker.hf_compat import FlaxSpeakerModel

model = FlaxSpeakerModel.from_pretrained("/usr/local/google/home/quanw/Code/github/FlaxSpeaker/pretrained_models/ls_ecapa_tdnn_tiny_ge2e")
emb = model.extract_embedding("path/to/audio.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
```

## Evaluation Results
```json
{
  "experiment_name": "ls_ecapa_tdnn_tiny_ge2e",
  "train_dataset": "librispeech",
  "backbone": "ecapa_tdnn",
  "size_variant": "tiny",
  "pooling_type": "asp",
  "loss_type": "ge2e_softmax",
  "scoring_type": "cosine",
  "embedding_dim": 128,
  "num_parameters": 527650,
  "num_steps": 200,
  "train_time_sec": 60.56,
  "initial_loss": 2.0422,
  "final_loss": 0.6309,
  "in_domain_eval": {
    "eer": 0.15800000000000003,
    "eer_percent": 15.8,
    "eer_threshold": 0.5431506633758545,
    "min_dcf_001": 0.97,
    "min_dcf_005": 0.868,
    "roc_auc": 0.9165,
    "num_trials": 3000,
    "num_unique_utterances": 999,
    "latency_ms_per_utterance": 7.53,
    "eval_time_sec": 7.561,
    "scoring_type": "cosine"
  },
  "multi_enroll_3utt_eval": {
    "eer": 0.124,
    "eer_percent": 12.4,
    "eer_threshold": 0.6178483963012695,
    "min_dcf_001": 0.838,
    "roc_auc": 0.9519,
    "num_trials": 1000
  },
  "cross_dataset_eval": {
    "voxceleb1_o_official": {
      "eer": 0.29175,
      "eer_percent": 29.175,
      "eer_threshold": 0.3976811170578003,
      "min_dcf_001": 0.999,
      "min_dcf_005": 0.9975,
      "roc_auc": 0.7883,
      "num_trials": 4000,
      "num_unique_utterances": 3892,
      "latency_ms_per_utterance": 4.606,
      "eval_time_sec": 17.967,
      "scoring_type": "cosine"
    },
    "cnceleb2_zero_shot": {
      "eer": 0.3433333333333334,
      "eer_percent": 34.3333,
      "eer_threshold": 0.44528841972351074,
      "min_dcf_001": 0.9813,
      "min_dcf_005": 0.972,
      "roc_auc": 0.7246,
      "num_trials": 3000,
      "num_unique_utterances": 739,
      "latency_ms_per_utterance": 6.283,
      "eval_time_sec": 4.676,
      "scoring_type": "cosine"
    }
  },
  "config_yaml": "/usr/local/google/home/quanw/Code/github/FlaxSpeaker/configs/ls_ecapa_tdnn_tiny_ge2e.yml"
}
```
