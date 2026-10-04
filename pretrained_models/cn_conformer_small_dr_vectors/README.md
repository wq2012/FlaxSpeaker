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
- **Scoring**: `dr_vector`
- **Loss Function**: `extended_set_softmax`
- **Output Embedding Dimension**: `192`
- **Parameter Count**: `1,929,475`

## Usage
```python
from flaxspeaker.hf_compat import FlaxSpeakerModel

model = FlaxSpeakerModel.from_pretrained("/usr/local/google/home/quanw/Code/github/FlaxSpeaker/pretrained_models/cn_conformer_small_dr_vectors")
emb = model.extract_embedding("path/to/audio.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
```

## Evaluation Results
```json
{
  "experiment_name": "cn_conformer_small_dr_vectors",
  "train_dataset": "cnceleb2",
  "backbone": "conformer",
  "size_variant": "small",
  "pooling_type": "asp",
  "loss_type": "extended_set_softmax",
  "scoring_type": "dr_vector",
  "embedding_dim": 192,
  "num_parameters": 1929475,
  "num_steps": 200,
  "train_time_sec": 112.33,
  "initial_loss": 4.0303,
  "final_loss": 3.1686,
  "in_domain_eval": {
    "eer": 0.2843333333333333,
    "eer_percent": 28.4333,
    "eer_threshold": 0.817934513092041,
    "min_dcf_001": 0.992,
    "min_dcf_005": 0.992,
    "roc_auc": 0.7908,
    "num_trials": 3000,
    "num_unique_utterances": 739,
    "latency_ms_per_utterance": 15.838,
    "eval_time_sec": 13.065,
    "scoring_type": "dr_vector"
  },
  "multi_enroll_3utt_eval": {
    "eer": 0.22299999999999998,
    "eer_percent": 22.3,
    "eer_threshold": 0.8832591772079468,
    "min_dcf_001": 0.962,
    "roc_auc": 0.8439,
    "num_trials": 1000
  },
  "cross_dataset_eval": {},
  "config_yaml": "/usr/local/google/home/quanw/Code/github/FlaxSpeaker/configs/cn_conformer_small_dr_vectors.yml"
}
```
