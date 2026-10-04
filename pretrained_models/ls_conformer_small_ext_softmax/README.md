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
- **Loss Function**: `extended_set_softmax`
- **Output Embedding Dimension**: `192`
- **Parameter Count**: `1,871,362`

## Usage
```python
from flaxspeaker.hf_compat import FlaxSpeakerModel

model = FlaxSpeakerModel.from_pretrained("/usr/local/google/home/quanw/Code/github/FlaxSpeaker/pretrained_models/ls_conformer_small_ext_softmax")
emb = model.extract_embedding("path/to/audio.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
```

## Evaluation Results
```json
{
  "experiment_name": "ls_conformer_small_ext_softmax",
  "train_dataset": "librispeech",
  "backbone": "conformer",
  "size_variant": "small",
  "pooling_type": "asp",
  "loss_type": "extended_set_softmax",
  "scoring_type": "cosine",
  "embedding_dim": 192,
  "num_parameters": 1871362,
  "num_steps": 200,
  "train_time_sec": 107.11,
  "initial_loss": 3.9906,
  "final_loss": 2.6855,
  "in_domain_eval": {
    "eer": 0.20733333333333337,
    "eer_percent": 20.7333,
    "eer_threshold": 0.7893890142440796,
    "min_dcf_001": 0.996,
    "min_dcf_005": 0.988,
    "roc_auc": 0.8724,
    "num_trials": 3000,
    "num_unique_utterances": 999,
    "latency_ms_per_utterance": 8.101,
    "eval_time_sec": 8.125,
    "scoring_type": "cosine"
  },
  "multi_enroll_3utt_eval": {
    "eer": 0.16100000000000003,
    "eer_percent": 16.1,
    "eer_threshold": 0.815463662147522,
    "min_dcf_001": 0.954,
    "roc_auc": 0.9169,
    "num_trials": 1000
  },
  "cross_dataset_eval": {
    "voxceleb1_o_official": {
      "eer": 0.3105,
      "eer_percent": 31.05,
      "eer_threshold": 0.7416742444038391,
      "min_dcf_001": 0.999,
      "min_dcf_005": 0.999,
      "roc_auc": 0.7601,
      "num_trials": 4000,
      "num_unique_utterances": 3892,
      "latency_ms_per_utterance": 6.885,
      "eval_time_sec": 26.853,
      "scoring_type": "cosine"
    },
    "cnceleb2_zero_shot": {
      "eer": 0.33866666666666667,
      "eer_percent": 33.8667,
      "eer_threshold": 0.7261396646499634,
      "min_dcf_001": 0.9993,
      "min_dcf_005": 0.9993,
      "roc_auc": 0.7065,
      "num_trials": 3000,
      "num_unique_utterances": 739,
      "latency_ms_per_utterance": 8.902,
      "eval_time_sec": 6.613,
      "scoring_type": "cosine"
    }
  },
  "config_yaml": "/usr/local/google/home/quanw/Code/github/FlaxSpeaker/configs/ls_conformer_small_ext_softmax.yml"
}
```
