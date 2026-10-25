---
license: apache-2.0
library_name: flaxspeaker
pipeline_tag: feature-extraction
tags:
- audio
- speaker-verification
- speaker-recognition
- speaker-embeddings
- jax
- flax
- safetensors
- ecapa_tdnn
- ge2e_softmax
- cosine
datasets:
- librispeech_asr
- cnceleb
- voxceleb
metrics:
- eer
- min_dcf
- roc_auc
---

# FlaxSpeaker: `ls_ecapa_tdnn_tiny_ge2e` (`ecapa_tdnn` - `tiny`)

## Disclaimer & Reproduction Notice

> **This is not an officially supported Google product.**
>
> This checkpoint and the accompanying `FlaxSpeaker` library are a from-scratch re-implementation in JAX/Flax, written using only the published papers (no proprietary code or data), of speaker recognition algorithms from:
> - **GE2E**: *Generalized End-to-End Loss for Speaker Verification* ([arXiv:1710.10467](https://arxiv.org/abs/1710.10467))
> - **Dr-Vectors**: *Decision Residual Networks and an Improved Loss for Speaker Recognition* ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989))
> - **ATP** (pooling head repurposed from a *language identification* paper): *Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech* ([arXiv:2202.12163](https://arxiv.org/abs/2202.12163))
> - **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* ([arXiv:2203.05642](https://arxiv.org/abs/2203.05642))
>
> The experiments in those papers were run on Google-internal infrastructure and data, not with this library. **This is a converged small-corpus benchmark checkpoint (2,400 steps / ~13.27 epochs on 5,788 utterances from 232 speakers), not a full-corpus reproduction of the papers' large-scale results** -- see *Training budget* and *Deviations from the source papers* below.

## Training Budget (read this first)

This model was trained to convergence for **2,400 gradient steps** (`S_safe = X + 2 epochs` safety margin, warmup `240` steps) with mini-batches of 32 utterances (8 speakers x 4 utterances), i.e. **~13.27 epochs** over the 5,788-utterance LibriSpeech training pool (232 speakers), on a CPU workstation with seed 42. Across the training trajectory, in-domain LibriSpeech EER progressed from step `400`: **16.93%** (1-utt) / **13.10%** (3-utt) -> step `2000` (knee): **10.17%** (1-utt) / **5.50%** (3-utt) -> step `2400` (final): **9.67%** (1-utt) / **4.90%** (3-utt) (Universal Pooled final: **19.99%** 1-utt / **16.03%** 3-utt). While training loss and validation EER have reached their small-corpus convergence plateau on this 232-speaker open split (<= 25 utterances/speaker), absolute EERs remain bounded by training speaker diversity compared to industrial benchmarks trained on thousands of speakers (see below).

**Training-crop provenance.** This checkpoint was trained with `U(90, 180)`-frame variable-length crops and an exact validity mask (the current pipeline default); the shipped `config.json` / YAML (`variable_length_training: true`) describes the released weights. Step-0 loss fingerprint at the released seed (`scripts/fingerprint_released_checkpoints.py`, `pretrained_models/step0_fingerprints.json`): archived `2.0713`; current code with fixed 180-frame crops `2.0736`, with `U(90, 180)` crops `2.0713` (match: `variable`). Evaluating the released weights with the current code reproduces the archived metrics in `eval_results.json`. At the converged `S_safe` budget, the seed-to-seed standard deviation of pooled 1-utt EER across seeds 0-2 is 0.08-0.41 % absolute (`pretrained_models/r2_seed_and_control_results.json`), while the 8,000-trial 95 % bootstrap CI half-width is ~0.8 %.

For orientation only: published ECAPA-TDNN systems trained on VoxCeleb2-dev reach about 0.9% EER on the full VoxCeleb1-O list (Desplanques et al., 2020), and sub-1M-parameter systems about 2% (Wang et al., EfficientTDNN, 2022). The checkpoints released here are trained to convergence (including a +2-epoch safety margin) on the self-contained 381-speaker (<=25 utterances/speaker) benchmark corpus rather than VoxCeleb2-dev (5,994 speakers, ~1.1M utterances), and are intended for controlled relative comparisons within the FlaxSpeaker benchmark, not as state-of-the-art models trained on thousand-speaker corpora.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model ID** | `ls_ecapa_tdnn_tiny_ge2e` |
| **Hugging Face Repository** | [`wq2012/ls_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_tiny_ge2e) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `ecapa_tdnn` (`tiny`) |
| **Temporal Pooling** | `asp` |
| **Training Objective** | GE2E softmax (Wan et al. 2018, Eq. 6; exact leave-one-out centroids) (`ge2e_softmax`) |
| **Verification Scorer** | Cosine similarity (`cosine`) |
| **Output Representation Dim** | `128` |
| **Inference Parameters (encoder + scoring head; excludes training-only classifier weights)** | `626,722` |
| **Training Data** | LibriSpeech only (232 speakers, 5,788 utterances) |
| **Training Steps / Batch** | `2,400` steps x 32 utterances (~13.27 epochs), seed 42 |
| **Audio Frontend** | `80`-dim natural-log Mel @ `16000` Hz + frame-level energy VAD + CMVN (`flaxspeaker.AudioFrontend`; **not** a Hugging Face `AutoFeatureExtractor`) |

## Model Files Included

- `model.safetensors`: model weights (encoder + GE2E `w,b` / Dr-Vector / PFAS heads) in SafeTensors format
- `flax_model.msgpack`: native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` (full `ExperimentConfig`, incl. the frontend/VAD settings)
- `eval_results.json`: multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs, VoxCeleb1-O 10-speaker subset)

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

Requires `flaxspeaker >= 0.4.0` (`pip install -U flaxspeaker`) for full `margin_warmup_steps` config compatibility with the v0.4.0 converged checkpoints (`>= 0.3.0` introduced `load_pretrained`; the legacy `0.2.0` wheel cannot load Hub checkpoints).

```python
from flaxspeaker import load_pretrained

model = load_pretrained("wq2012/ls_ecapa_tdnn_tiny_ge2e")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/ls_ecapa_tdnn_tiny_ge2e \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

Trial lists are 50/50 target/impostor over held-out speakers (LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is close to its trivial ceiling of 1.0 for every model under this trial mix.

| Evaluation Split | Trials | EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **9.67%** | [8.50, 10.60] | 0.6913 | 0.4113 | 0.9671 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **29.43%** | [27.68, 30.87] | 0.9440 | 0.9240 | 0.7811 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **20.65%** | [18.70, 22.43] | 0.9770 | 0.8740 | 0.8787 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **19.99%** | [19.18, 20.98] | 0.9325 | 0.7933 | 0.8837 |
| **Universal Macro-Average (3 Domains)** | — | **19.92%** | — | 0.8708 | 0.7364 | 0.8756 |
| **VoxCeleb1-O 10-speaker subset** (see caveat) | 2,500 | **25.46%** | [23.43, 27.21] | 0.9335 | 0.9199 | 0.8382 |

Caveat: VoxCeleb1-O **10-speaker subset**: the official `veri_test2.txt` list filtered to trials whose enrollment and test speakers are both among the 10 held-out evaluation speakers (2,500 trials, ~72% target). **Not comparable to published VoxCeleb1-O numbers**; the other 30 of the 40 official test speakers are used as training data by `vox_*` and `univ_*` models.

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **4.90%** | [3.60, 6.60] | 0.4380 | 0.9889 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **25.10%** | [22.80, 28.20] | 0.9080 | 0.8273 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **17.20%** | [14.45, 19.20] | 0.7680 | 0.9237 |
| **Universal Pooled (3-Utt)** | 3,000 | **16.03%** | [14.77, 17.47] | 0.8773 | 0.9228 |
| **Universal Macro-Average (3-Utt)** | — | **15.73%** | — | 0.7047 | 0.9133 |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 2,400 gradient steps x 32 utterances (N = 8 speakers x M = 4, ~13.27 epochs) on <= 25 utterances/speaker from 232 speakers (5,788 utterances); the source papers train for ~500k steps with N = 16-64 / M = 8-10 on thousands of speakers.
- **Augmentation**: synthetic log-Mel-domain noise + one-tap smoothing (`specaug.apply_feature_mtr_augmentation`), not waveform MTR with recorded noise/RIR corpora.
- **ECAPA-TDNN**: LayerNorm instead of BatchNorm.
- **Frontend**: 80-D natural-log Mel + frame-level energy VAD (26 dB dynamic range) + CMVN; evaluation centre-crops to at most 500 frames (5 s).

## Citation

```bibtex
@inproceedings{wan2018generalized,
  title={Generalized End-to-End Loss for Speaker Verification},
  author={Wan, Li and Wang, Quan and Papir, Alan and Moreno, Ignacio Lopez},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4879--4883},
  year={2018}
}

@inproceedings{pelecanos2021dr,
  title={Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition},
  author={Pelecanos, Jason and Wang, Quan and Moreno, Ignacio Lopez},
  booktitle={Interspeech},
  pages={4603--4607},
  year={2021}
}

@inproceedings{wang2022attentive,
  title={Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech},
  author={Wang, Quan and Yu, Yang and Pelecanos, Jason and Huang, Yiling and Moreno, Ignacio Lopez},
  booktitle={The Speaker and Language Recognition Workshop (Odyssey)},
  year={2022}
}

@inproceedings{pelecanos2022parameter,
  title={Parameter-Free Attentive Scoring for Speaker Verification},
  author={Pelecanos, Jason and Wang, Quan and Huang, Yiling and Moreno, Ignacio Lopez},
  booktitle={The Speaker and Language Recognition Workshop (Odyssey)},
  year={2022}
}
```
