---
library_name: flaxspeaker
tags:
- audio
- speaker-verification
- speaker-recognition
- jax
- flax
---
# FlaxSpeaker Model (`transformer` - `small`)

## Model Summary
- **Backbone**: `transformer` (`small`)
- **Pooling**: `asp`
- **Scoring**: `cosine`
- **Loss Function**: `ge2e_softmax`
- **Output Embedding Dimension**: `192`
- **Parameter Count**: `1,080,322`

## Usage
```python
from flaxspeaker.hf_compat import FlaxSpeakerModel

model = FlaxSpeakerModel.from_pretrained("/usr/local/google/home/quanw/Code/github/FlaxSpeaker/pretrained_models/ls_transformer_small_ge2e")
emb = model.extract_embedding("path/to/audio.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
```

## Evaluation Results
```json
{
  "experiment_name": "ls_transformer_small_ge2e",
  "train_dataset": "librispeech",
  "backbone": "transformer",
  "size_variant": "small",
  "pooling_type": "asp",
  "loss_type": "ge2e_softmax",
  "scoring_type": "cosine",
  "embedding_dim": 192,
  "num_parameters": 1080322,
  "num_steps": 200,
  "train_time_sec": 44.13,
  "initial_loss": 1.9942,
  "final_loss": 0.5565,
  "in_domain_eval": {
    "eer": 0.14933333333333332,
    "eer_percent": 14.9333,
    "eer_threshold": 0.5312825441360474,
    "min_dcf_001": 0.868,
    "min_dcf_005": 0.7813,
    "roc_auc": 0.9218,
    "num_trials": 3000,
    "num_unique_utterances": 999,
    "latency_ms_per_utterance": 6.609,
    "eval_time_sec": 6.619,
    "scoring_type": "cosine"
  },
  "multi_enroll_3utt_eval": {
    "eer": 0.128,
    "eer_percent": 12.8,
    "eer_threshold": 0.5714007019996643,
    "min_dcf_001": 0.86,
    "roc_auc": 0.9492,
    "num_trials": 1000
  },
  "cross_dataset_eval": {
    "voxceleb1_o_official": {
      "eer": 0.2985,
      "eer_percent": 29.85,
      "eer_threshold": 0.5258704423904419,
      "min_dcf_001": 0.997,
      "min_dcf_005": 0.997,
      "roc_auc": 0.7673,
      "num_trials": 4000,
      "num_unique_utterances": 3892,
      "latency_ms_per_utterance": 3.296,
      "eval_time_sec": 12.85,
      "scoring_type": "cosine"
    },
    "cnceleb2_zero_shot": {
      "eer": 0.32833333333333337,
      "eer_percent": 32.8333,
      "eer_threshold": 0.5518258213996887,
      "min_dcf_001": 1.0,
      "min_dcf_005": 1.0,
      "roc_auc": 0.7317,
      "num_trials": 3000,
      "num_unique_utterances": 739,
      "latency_ms_per_utterance": 5.457,
      "eval_time_sec": 4.047,
      "scoring_type": "cosine"
    }
  },
  "config_yaml": "/usr/local/google/home/quanw/Code/github/FlaxSpeaker/configs/ls_transformer_small_ge2e.yml"
}
```
