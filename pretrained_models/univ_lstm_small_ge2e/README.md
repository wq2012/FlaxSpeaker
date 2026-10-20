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
- lstm
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

# FlaxSpeaker: `univ_lstm_small_ge2e` (`lstm` - `small`)

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

For orientation only: published ECAPA-TDNN systems trained on VoxCeleb2-dev reach about 0.9% EER on the full VoxCeleb1-O list (Desplanques et al., 2020), and sub-1M-parameter systems about 2% (Wang et al., EfficientTDNN, 2022). The 400-step checkpoints released here are 10-30x worse than that and are intended for relative comparisons within the FlaxSpeaker benchmark, not as state-of-the-art models.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model ID** | `univ_lstm_small_ge2e` |
| **Hugging Face Repository** | [`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `lstm` (`small`) |
| **Temporal Pooling** | `asp` |
| **Training Objective** | GE2E softmax (Wan et al. 2018, Eq. 6; exact leave-one-out centroids) (`ge2e_softmax`) |
| **Verification Scorer** | Cosine similarity (`cosine`) |
| **Output Representation Dim** | `192` |
| **Total Parameters (encoder + loss/scoring heads)** | `1,710,914` |
| **Training Data** | LibriSpeech + CN-Celeb2 + VoxCeleb1 (universal multi-domain, 381 speakers) |
| **Training Steps / Batch** | `400` steps x 32 utterances, seed 42 |
| **Audio Frontend** | `80`-dim natural-log Mel @ `16000` Hz + frame-level energy VAD + CMVN (`flaxspeaker.AudioFrontend`; **not** a Hugging Face `AutoFeatureExtractor`) |

## Model Files Included

- `model.safetensors`: model weights (encoder + GE2E `w,b` / Dr-Vector / PFAS heads) in SafeTensors format
- `flax_model.msgpack`: native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` (full `ExperimentConfig`, incl. the frontend/VAD settings)
- `eval_results.json`: multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs, VoxCeleb1-O 10-speaker subset)
- `model_fp32.tflite` / `model_int8.tflite`: `jax2tf` exports of the **encoder only** with a **fixed, unmasked 180-frame (1.8 s) input window** `[1, 180, 80]` -> `[1, 192]`. All EERs in this card come from full-sequence masked JAX inference (up to 500 frames); the TFLite models see a centre crop and the reported cosine fidelity (>= 0.9999 vs. JAX) was measured on 180-frame inputs only.
  - No TFLite EER has been measured for this repository.

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

```python
from flaxspeaker import load_pretrained

model = load_pretrained("wq2012/univ_lstm_small_ge2e")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_lstm_small_ge2e \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

Trial lists are 50/50 target/impostor over held-out speakers (LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is close to its trivial ceiling of 1.0 for every model under this trial mix.

| Evaluation Split | Trials | EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **14.97%** | [13.80, 16.35] | 0.9493 | 0.7667 | 0.9236 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **25.53%** | [23.83, 26.98] | 0.9787 | 0.9660 | 0.8305 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **22.50%** | [20.70, 24.20] | 0.9580 | 0.9070 | 0.8643 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **21.31%** | [20.42, 22.15] | 0.9750 | 0.9028 | 0.8730 |
| **Universal Macro-Average (3 Domains)** | — | **21.00%** | — | 0.9620 | 0.8799 | 0.8728 |
| **VoxCeleb1-O 10-speaker subset** (see caveat) | 2,500 | **24.73%** | [22.90, 26.92] | 0.9586 | 0.8913 | 0.8459 |

Caveat: VoxCeleb1-O **10-speaker subset**: the official `veri_test2.txt` list filtered to trials whose enrollment and test speakers are both among the 10 held-out evaluation speakers (2,500 trials, ~72% target). **Not comparable to published VoxCeleb1-O numbers**; the other 30 of the 40 official test speakers are used as training data by `vox_*` and `univ_*` models.

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **11.20%** | [9.20, 13.10] | 0.7840 | 0.9630 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **21.60%** | [18.75, 23.60] | 0.9860 | 0.8732 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **19.50%** | [16.60, 21.80] | 0.9820 | 0.8904 |
| **Universal Pooled (3-Utt)** | 3,000 | **18.00%** | [16.67, 19.33] | 0.9853 | 0.9082 |
| **Universal Macro-Average (3-Utt)** | — | **17.43%** | — | 0.9173 | 0.9089 |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 400 gradient steps x 32 utterances (N = 8 speakers x M = 4) on <= 25 utterances/speaker from 381 (universal) or fewer speakers; the source papers train for ~500k steps with N = 16-64 / M = 8-10 on thousands of speakers.
- **Augmentation**: synthetic log-Mel-domain noise + one-tap smoothing (`specaug.apply_feature_mtr_augmentation`), not waveform MTR with recorded noise/RIR corpora.
- **LSTM**: per-layer LayerNorm + residual connections and a post-pooling nonlinear bottleneck; the GE2E paper uses 3 x 768 LSTM + 256-D projections and a plain linear output (`icassp2018_ge2e_config(pure_paper_mode=True)`).
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
