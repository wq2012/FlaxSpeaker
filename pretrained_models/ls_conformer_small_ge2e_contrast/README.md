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
- **Loss Function**: `ge2e_contrast`
- **Output Embedding Dimension**: `192`
- **Parameter Count**: `1,871,362`

## Usage
```python
from flaxspeaker.hf_compat import FlaxSpeakerModel

model = FlaxSpeakerModel.from_pretrained("/usr/local/google/home/quanw/Code/github/FlaxSpeaker/pretrained_models/ls_conformer_small_ge2e_contrast")
emb = model.extract_embedding("path/to/audio.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
```

## Evaluation Results
```json
{
  "experiment_name": "ls_conformer_small_ge2e_contrast",
  "train_dataset": "librispeech",
  "backbone": "conformer",
  "size_variant": "small",
  "pooling_type": "asp",
  "loss_type": "ge2e_contrast",
  "scoring_type": "cosine",
  "embedding_dim": 192,
  "num_parameters": 1871362,
  "num_steps": 200,
  "train_time_sec": 65.5,
  "initial_loss": 1.0044,
  "final_loss": 1.0,
  "in_domain_eval": {
    "eer": 0.367,
    "eer_percent": 36.7,
    "eer_threshold": 0.9998452067375183,
    "min_dcf_001": 0.9987,
    "min_dcf_005": 0.9987,
    "roc_auc": 0.6803,
    "num_trials": 3000,
    "num_unique_utterances": 999,
    "latency_ms_per_utterance": 3.813,
    "eval_time_sec": 3.825,
    "scoring_type": "cosine"
  },
  "multi_enroll_3utt_eval": {
    "eer": 0.32999999999999996,
    "eer_percent": 33.0,
    "eer_threshold": 0.9998899102210999,
    "min_dcf_001": 0.99,
    "roc_auc": 0.7285,
    "num_trials": 1000
  },
  "cross_dataset_eval": {
    "voxceleb1_o_official": {
      "eer": 0.42,
      "eer_percent": 42.0,
      "eer_threshold": 0.9998083114624023,
      "min_dcf_001": 0.9995,
      "min_dcf_005": 0.9995,
      "roc_auc": 0.613,
      "num_trials": 4000,
      "num_unique_utterances": 3892,
      "latency_ms_per_utterance": 7.14,
      "eval_time_sec": 27.808,
      "scoring_type": "cosine"
    },
    "cnceleb2_zero_shot": {
      "eer": 0.38166666666666665,
      "eer_percent": 38.1667,
      "eer_threshold": 0.9997714757919312,
      "min_dcf_001": 0.99,
      "min_dcf_005": 0.99,
      "roc_auc": 0.6645,
      "num_trials": 3000,
      "num_unique_utterances": 739,
      "latency_ms_per_utterance": 7.605,
      "eval_time_sec": 5.639,
      "scoring_type": "cosine"
    }
  },
  "config_yaml": "/usr/local/google/home/quanw/Code/github/FlaxSpeaker/configs/ls_conformer_small_ge2e_contrast.yml"
}
```
