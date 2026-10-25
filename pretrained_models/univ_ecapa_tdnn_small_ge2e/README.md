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

# FlaxSpeaker: `univ_ecapa_tdnn_small_ge2e` (`ecapa_tdnn` - `small`)

## Disclaimer & Reproduction Notice

> **This is not an officially supported Google product.**
>
> This checkpoint and the accompanying `FlaxSpeaker` library are a from-scratch re-implementation in JAX/Flax, written using only the published papers (no proprietary code or data), of speaker recognition algorithms from:
> - **GE2E**: *Generalized End-to-End Loss for Speaker Verification* ([arXiv:1710.10467](https://arxiv.org/abs/1710.10467))
> - **Dr-Vectors**: *Decision Residual Networks and an Improved Loss for Speaker Recognition* ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989))
> - **ATP** (pooling head repurposed from a *language identification* paper): *Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech* ([arXiv:2202.12163](https://arxiv.org/abs/2202.12163))
> - **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* ([arXiv:2203.05642](https://arxiv.org/abs/2203.05642))
>
> The experiments in those papers were run on Google-internal infrastructure and data, not with this library. **This is a converged small-corpus benchmark checkpoint (4,600 steps / ~15.58 epochs on 9,447 utterances from 381 speakers), not a full-corpus reproduction of the papers' large-scale results** -- see *Training budget* and *Deviations from the source papers* below.

## Training Budget (read this first)

This model was trained to convergence for **4,600 gradient steps** (`S_safe = X + 2 epochs` safety margin, warmup `460` steps) with mini-batches of 32 utterances (8 speakers x 4 utterances), i.e. **~15.58 epochs** over the 9,447-utterance 3-domain training pool (381 speakers), on a CPU workstation with seed 42. Across the training trajectory, Universal Pooled EER progressed from step `400`: **24.60%** (1-utt) / **19.97%** (3-utt) -> step `4000` (knee): **16.20%** (1-utt) / **13.43%** (3-utt) -> step `4600` (final): **16.34%** (1-utt) / **12.87%** (3-utt). While training loss and validation EER have reached their small-corpus convergence plateau on this 381-speaker open split (<= 25 utterances/speaker), absolute EERs remain bounded by training speaker diversity compared to industrial benchmarks trained on thousands of speakers (see below).

**Training-crop provenance.** This checkpoint was trained with `U(90, 180)`-frame variable-length crops and an exact validity mask (the current pipeline default); the shipped `config.json` / YAML (`variable_length_training: true`) describes the released weights. Step-0 loss fingerprint at the released seed (`scripts/fingerprint_released_checkpoints.py`, `pretrained_models/step0_fingerprints.json`): archived `2.065`; current code with fixed 180-frame crops `2.066`, with `U(90, 180)` crops `2.065` (match: `variable`). Evaluating the released weights with the current code reproduces the archived metrics in `eval_results.json`. At the converged `S_safe` budget, the seed-to-seed standard deviation of pooled 1-utt EER across seeds 0-2 is 0.08-0.41 % absolute (`pretrained_models/r2_seed_and_control_results.json`), while the 8,000-trial 95 % bootstrap CI half-width is ~0.8 %.

For orientation only: published ECAPA-TDNN systems trained on VoxCeleb2-dev reach about 0.9% EER on the full VoxCeleb1-O list (Desplanques et al., 2020), and sub-1M-parameter systems about 2% (Wang et al., EfficientTDNN, 2022). The checkpoints released here are trained to convergence (including a +2-epoch safety margin) on the self-contained 381-speaker (<=25 utterances/speaker) benchmark corpus rather than VoxCeleb2-dev (5,994 speakers, ~1.1M utterances), and are intended for controlled relative comparisons within the FlaxSpeaker benchmark, not as state-of-the-art models trained on thousand-speaker corpora.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model ID** | `univ_ecapa_tdnn_small_ge2e` |
| **Hugging Face Repository** | [`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `ecapa_tdnn` (`small`) |
| **Temporal Pooling** | `asp` |
| **Training Objective** | GE2E softmax (Wan et al. 2018, Eq. 6; exact leave-one-out centroids) (`ge2e_softmax`) |
| **Verification Scorer** | Cosine similarity (`cosine`) |
| **Output Representation Dim** | `192` |
| **Inference Parameters (encoder + scoring head; excludes training-only classifier weights)** | `1,684,418` |
| **Training Data** | LibriSpeech + CN-Celeb2 + VoxCeleb1 (universal multi-domain, 381 speakers, 9,447 utterances) |
| **Training Steps / Batch** | `4,600` steps x 32 utterances (~15.58 epochs), seed 42 |
| **Audio Frontend** | `80`-dim natural-log Mel @ `16000` Hz + frame-level energy VAD + CMVN (`flaxspeaker.AudioFrontend`; **not** a Hugging Face `AutoFeatureExtractor`) |

## Model Files Included

- `model.safetensors`: model weights (encoder + GE2E `w,b` / Dr-Vector / PFAS heads) in SafeTensors format
- `flax_model.msgpack`: native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` (full `ExperimentConfig`, incl. the frontend/VAD settings)
- `eval_results.json`: multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs, VoxCeleb1-O 10-speaker subset)
- `model_fp32.tflite` / `model_int8.tflite`: `jax2tf` exports of the **encoder only** with a **fixed, unmasked 180-frame (1.8 s) input window** `[1, 180, 80]` -> `[1, 192]`. All EERs in the main tables below come from full-sequence masked JAX inference (up to 500 frames); the TFLite models see a centre crop and the reported cosine fidelity (>= 0.9999 vs. JAX) was measured on 180-frame inputs.
  - Measured 180-frame TFLite 1-utt EER (universal_pooled 8,000 1-utt trials (fixed 180-frame unmasked window)): FP32 **18.15%**, INT8 **18.24%**, mean `cos(FP32, INT8)` = `0.999972` (vs. full-sequence <=500-frame masked JAX: `16.34%`)

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

Requires `flaxspeaker >= 0.4.0` (`pip install -U flaxspeaker`) for full `margin_warmup_steps` config compatibility with the v0.4.0 converged checkpoints (`>= 0.3.0` introduced `load_pretrained`; the legacy `0.2.0` wheel cannot load Hub checkpoints).

```python
from flaxspeaker import load_pretrained

model = load_pretrained("wq2012/univ_ecapa_tdnn_small_ge2e")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_ecapa_tdnn_small_ge2e \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

Trial lists are 50/50 target/impostor over held-out speakers (LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is close to its trivial ceiling of 1.0 for every model under this trial mix.

| Evaluation Split | Trials | EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **10.37%** | [9.40, 11.57] | 0.6880 | 0.4607 | 0.9600 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **18.73%** | [17.20, 20.00] | 0.9493 | 0.8660 | 0.8922 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **19.75%** | [17.75, 21.45] | 0.8920 | 0.7940 | 0.8948 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **16.34%** | [15.56, 17.15] | 0.9467 | 0.7123 | 0.9199 |
| **Universal Macro-Average (3 Domains)** | — | **16.28%** | — | 0.8431 | 0.7069 | 0.9157 |
| **VoxCeleb1-O 10-speaker subset** (see caveat) | 2,500 | **20.68%** | [18.85, 22.77] | 0.9547 | 0.8717 | 0.8761 |

Caveat: VoxCeleb1-O **10-speaker subset**: the official `veri_test2.txt` list filtered to trials whose enrollment and test speakers are both among the 10 held-out evaluation speakers (2,500 trials, ~72% target). **Not comparable to published VoxCeleb1-O numbers**; the other 30 of the 40 official test speakers are used as training data by `vox_*` and `univ_*` models.

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **7.60%** | [5.60, 9.15] | 0.4580 | 0.9817 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **15.10%** | [12.80, 17.20] | 0.7320 | 0.9293 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **13.70%** | [11.90, 16.80] | 0.9380 | 0.9302 |
| **Universal Pooled (3-Utt)** | 3,000 | **12.87%** | [11.63, 14.15] | 0.7953 | 0.9484 |
| **Universal Macro-Average (3-Utt)** | — | **12.13%** | — | 0.7093 | 0.9471 |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 4,600 gradient steps x 32 utterances (N = 8 speakers x M = 4, ~15.58 epochs) on <= 25 utterances/speaker from 381 speakers (9,447 utterances); the source papers train for ~500k steps with N = 16-64 / M = 8-10 on thousands of speakers.
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
