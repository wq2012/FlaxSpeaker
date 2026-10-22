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
> The experiments in those papers were run on Google-internal infrastructure and data, not with this library. **This is a 400-step demonstration checkpoint, not a reproduction of the papers' results** -- see *Training budget* and *Deviations from the source papers* below.

## Training Budget (read this first)

This model was trained for **400 gradient steps** with mini-batches of 32 utterances (8 speakers x 4 utterances), i.e. about 1.35 epochs over the 9,447-utterance universal corpus, on a CPU workstation, with a single seed (42). Losses are not converged at this budget. All EERs below are therefore far from state of the art and are only meaningful for *relative* comparisons between FlaxSpeaker checkpoints trained under the same budget.

**Training-crop provenance.** This checkpoint was trained with **fixed 180-frame (1.8 s) crops and all-ones validity masks**, not with the `U(90, 180)` variable-length crops that are the current pipeline default: the training code in use at the time defaulted the batchers to `variable_length=False` and the benchmark driver did not pass the flag. The shipped `config.json` / YAML therefore set `variable_length_training: false`, which re-creates this regime. Step-0 loss fingerprint at the released seed (`scripts/fingerprint_released_checkpoints.py`, `pretrained_models/step0_fingerprints.json`): archived `2.066`; current code with fixed 180-frame crops `2.066`, with `U(90, 180)` crops `2.065` (match: `fixed`). Evaluating the released weights with the current code reproduces the archived metrics in `eval_results.json`. The seed-to-seed standard deviation of pooled 1-utt EER at this budget is about 0.7-0.8 % absolute, so sub-1.5 % differences between checkpoints are not rankings.

For orientation only: published ECAPA-TDNN systems trained on VoxCeleb2-dev reach about 0.9% EER on the full VoxCeleb1-O list (Desplanques et al., 2020), and sub-1M-parameter systems about 2% (Wang et al., EfficientTDNN, 2022). The 400-step checkpoints released here are 10-30x worse than that and are intended for relative comparisons within the FlaxSpeaker benchmark, not as state-of-the-art models.

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
| **Training Data** | LibriSpeech + CN-Celeb2 + VoxCeleb1 (universal multi-domain, 381 speakers) |
| **Training Steps / Batch** | `400` steps x 32 utterances, seed 42 |
| **Audio Frontend** | `80`-dim natural-log Mel @ `16000` Hz + frame-level energy VAD + CMVN (`flaxspeaker.AudioFrontend`; **not** a Hugging Face `AutoFeatureExtractor`) |

## Model Files Included

- `model.safetensors`: model weights (encoder + GE2E `w,b` / Dr-Vector / PFAS heads) in SafeTensors format
- `flax_model.msgpack`: native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` (full `ExperimentConfig`, incl. the frontend/VAD settings)
- `eval_results.json`: multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs, VoxCeleb1-O 10-speaker subset)
- `model_fp32.tflite` / `model_int8.tflite`: `jax2tf` exports of the **encoder only** with a **fixed, unmasked 180-frame (1.8 s) input window** `[1, 180, 80]` -> `[1, 192]`. All EERs in the main tables below come from full-sequence masked JAX inference (up to 500 frames); the TFLite models see a centre crop and the reported cosine fidelity (>= 0.9999 vs. JAX) was measured on 180-frame inputs.
  - Measured 180-frame TFLite 1-utt EER (universal_pooled 8,000 1-utt trials (fixed 180-frame unmasked window)): FP32 **25.39%**, INT8 **25.35%**, mean `cos(FP32, INT8)` = `0.999971` (vs. full-sequence <=500-frame masked JAX: `23.16%`)

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

Requires `flaxspeaker >= 0.3.0` (`pip install -U flaxspeaker`); the `0.2.0` wheel on PyPI predates the released checkpoints and cannot load them.

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
| **LibriSpeech (English Read)** | 3,000 | **19.00%** | [17.52, 20.45] | 0.9807 | 0.8947 | 0.8941 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **25.13%** | [23.67, 26.68] | 0.9980 | 0.9500 | 0.8258 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **23.30%** | [21.07, 24.90] | 0.9880 | 0.9850 | 0.8518 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **23.16%** | [22.25, 24.03] | 0.9973 | 0.9655 | 0.8536 |
| **Universal Macro-Average (3 Domains)** | — | **22.48%** | — | 0.9889 | 0.9432 | 0.8572 |
| **VoxCeleb1-O 10-speaker subset** (see caveat) | 2,500 | **24.73%** | [22.78, 27.14] | 0.9709 | 0.9266 | 0.8323 |

Caveat: VoxCeleb1-O **10-speaker subset**: the official `veri_test2.txt` list filtered to trials whose enrollment and test speakers are both among the 10 held-out evaluation speakers (2,500 trials, ~72% target). **Not comparable to published VoxCeleb1-O numbers**; the other 30 of the 40 official test speakers are used as training data by `vox_*` and `univ_*` models.

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **11.80%** | [9.75, 14.15] | 0.7780 | 0.9531 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **22.40%** | [19.60, 25.00] | 1.0000 | 0.8677 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **18.80%** | [16.05, 21.25] | 0.9580 | 0.8878 |
| **Universal Pooled (3-Utt)** | 3,000 | **18.00%** | [16.77, 19.40] | 0.9993 | 0.8993 |
| **Universal Macro-Average (3-Utt)** | — | **17.67%** | — | 0.9120 | 0.9029 |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 400 gradient steps x 32 utterances (N = 8 speakers x M = 4) on <= 25 utterances/speaker from 381 (universal) or fewer speakers; the source papers train for ~500k steps with N = 16-64 / M = 8-10 on thousands of speakers.
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
