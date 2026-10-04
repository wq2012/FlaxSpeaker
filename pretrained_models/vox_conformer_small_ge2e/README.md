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

model = FlaxSpeakerModel.from_pretrained("/usr/local/google/home/quanw/Code/github/FlaxSpeaker/pretrained_models/vox_conformer_small_ge2e")
emb = model.extract_embedding("path/to/audio.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
```

## Evaluation Results
```json
{
  "experiment_name": "vox_conformer_small_ge2e",
  "train_dataset": "voxceleb1",
  "backbone": "conformer",
  "size_variant": "small",
  "pooling_type": "asp",
  "loss_type": "ge2e_softmax",
  "scoring_type": "cosine",
  "embedding_dim": 192,
  "num_parameters": 1871362,
  "num_steps": 200,
  "train_time_sec": 181.42,
  "initial_loss": 2.0301,
  "final_loss": 1.0444,
  "in_domain_eval": {
    "eer": 0.28,
    "eer_percent": 28.0,
    "eer_threshold": 0.7694054841995239,
    "min_dcf_001": 0.994,
    "min_dcf_005": 0.994,
    "roc_auc": 0.794,
    "num_trials": 2000,
    "num_unique_utterances": 250,
    "latency_ms_per_utterance": 30.625,
    "eval_time_sec": 7.925,
    "scoring_type": "cosine"
  },
  "multi_enroll_3utt_eval": null,
  "cross_dataset_eval": {},
  "config_yaml": "/usr/local/google/home/quanw/Code/github/FlaxSpeaker/configs/vox_conformer_small_ge2e.yml"
}
```
