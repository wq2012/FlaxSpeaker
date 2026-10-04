---
library_name: flaxspeaker
tags:
- audio
- speaker-verification
- speaker-recognition
- jax
- flax
---
# FlaxSpeaker Model (`conformer` - `small`)

## Model Summary
- **Backbone**: `conformer` (`small`)
- **Pooling**: `asp`
- **Scoring**: `cosine`
- **Loss Function**: `ge2e_softmax`
- **Output Embedding Dimension**: `192`
- **Parameter Count**: `1,871,362`

## Usage
```python
from flaxspeaker.hf_compat import FlaxSpeakerModel

model = FlaxSpeakerModel.from_pretrained("/usr/local/google/home/quanw/Code/github/FlaxSpeaker/pretrained_models/ls_conformer_small_ge2e")
emb = model.extract_embedding("path/to/audio.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
```

## Evaluation Results
```json
{
  "experiment_name": "ls_conformer_small_ge2e",
  "train_dataset": "librispeech",
  "backbone": "conformer",
  "size_variant": "small",
  "pooling_type": "asp",
  "loss_type": "ge2e_softmax",
  "scoring_type": "cosine",
  "embedding_dim": 192,
  "num_parameters": 1871362,
  "num_steps": 200,
  "train_time_sec": 86.98,
  "initial_loss": 1.9375,
  "final_loss": 0.7831,
  "in_domain_eval": {
    "eer": 0.17133333333333334,
    "eer_percent": 17.1333,
    "eer_threshold": 0.6939595937728882,
    "min_dcf_001": 0.9647,
    "min_dcf_005": 0.948,
    "roc_auc": 0.9009,
    "num_trials": 3000,
    "num_unique_utterances": 999,
    "latency_ms_per_utterance": 4.309,
    "eval_time_sec": 4.319,
    "scoring_type": "cosine"
  },
  "multi_enroll_3utt_eval": {
    "eer": 0.15000000000000002,
    "eer_percent": 15.0,
    "eer_threshold": 0.7469314932823181,
    "min_dcf_001": 0.922,
    "roc_auc": 0.9325,
    "num_trials": 1000
  },
  "cross_dataset_eval": {
    "voxceleb1_o_official": {
      "eer": 0.29074999999999995,
      "eer_percent": 29.075,
      "eer_threshold": 0.62399822473526,
      "min_dcf_001": 0.998,
      "min_dcf_005": 0.998,
      "roc_auc": 0.7794,
      "num_trials": 4000,
      "num_unique_utterances": 3892,
      "latency_ms_per_utterance": 4.215,
      "eval_time_sec": 16.423,
      "scoring_type": "cosine"
    },
    "cnceleb2_zero_shot": {
      "eer": 0.3433333333333333,
      "eer_percent": 34.3333,
      "eer_threshold": 0.5941939353942871,
      "min_dcf_001": 1.0,
      "min_dcf_005": 1.0,
      "roc_auc": 0.7206,
      "num_trials": 3000,
      "num_unique_utterances": 739,
      "latency_ms_per_utterance": 4.436,
      "eval_time_sec": 3.292,
      "scoring_type": "cosine"
    }
  },
  "config_yaml": "/usr/local/google/home/quanw/Code/github/FlaxSpeaker/configs/ls_conformer_small_ge2e.yml"
}
```
