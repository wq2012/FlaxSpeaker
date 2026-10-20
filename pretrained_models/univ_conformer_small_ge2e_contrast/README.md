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
- conformer
- ge2e_contrast
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

# FlaxSpeaker: `univ_conformer_small_ge2e_contrast` (`conformer` - `small`)

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

**Training-code provenance.** The released weights were produced by an earlier revision of the FlaxSpeaker training code than the current `main`. Evaluating released weights with the current code reproduces the archived numbers (verified for `univ_conformer_small_ge2e` and `ls_conformer_small_ge2e`); re-training the shipped `config.json` / YAML with the current code does **not** regenerate them (for `univ_conformer_small_ge2e` and `ls_conformer_small_ge2e`, seed 42 re-trained at the current revision gives 20.94 % / 21.40 % instead of the archived 19.40 % / 20.12 % pooled 1-utt EER; see `pretrained_models/r2_seed_and_control_results.json` in the GitHub repository). The seed-to-seed standard deviation of pooled 1-utt EER at the current revision is about 0.7-0.8 % absolute, so sub-1.5 % differences between checkpoints are not rankings.

For orientation only: published ECAPA-TDNN systems trained on VoxCeleb2-dev reach about 0.9% EER on the full VoxCeleb1-O list (Desplanques et al., 2020), and sub-1M-parameter systems about 2% (Wang et al., EfficientTDNN, 2022). The 400-step checkpoints released here are 10-30x worse than that and are intended for relative comparisons within the FlaxSpeaker benchmark, not as state-of-the-art models.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model ID** | `univ_conformer_small_ge2e_contrast` |
| **Hugging Face Repository** | [`wq2012/univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `conformer` (`small`) |
| **Temporal Pooling** | `atp` |
| **Training Objective** | GE2E hybrid contrast surrogate (FlaxSpeaker; value = Wan et al. Eq. 7, gradient = 0.25 Eq. 7 + 0.25 BCE + 0.50 GE2E-softmax CE; `ge2e_init_b = -1`) (`ge2e_contrast`) |
| **Verification Scorer** | Cosine similarity (`cosine`) |
| **Output Representation Dim** | `192` |
| **Total Parameters (encoder + loss/scoring heads)** | `2,029,894` |
| **Training Data** | LibriSpeech + CN-Celeb2 + VoxCeleb1 (universal multi-domain, 381 speakers) |
| **Training Steps / Batch** | `400` steps x 32 utterances, seed 42 |
| **Audio Frontend** | `80`-dim natural-log Mel @ `16000` Hz + frame-level energy VAD + CMVN (`flaxspeaker.AudioFrontend`; **not** a Hugging Face `AutoFeatureExtractor`) |

## Model Files Included

- `model.safetensors`: model weights (encoder + GE2E `w,b` / Dr-Vector / PFAS heads) in SafeTensors format
- `flax_model.msgpack`: native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` (full `ExperimentConfig`, incl. the frontend/VAD settings)
- `eval_results.json`: multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs, VoxCeleb1-O 10-speaker subset)

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

```python
from flaxspeaker import load_pretrained

model = load_pretrained("wq2012/univ_conformer_small_ge2e_contrast")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_conformer_small_ge2e_contrast \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

Trial lists are 50/50 target/impostor over held-out speakers (LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is close to its trivial ceiling of 1.0 for every model under this trial mix.

| Evaluation Split | Trials | EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **14.73%** | [13.58, 16.12] | 0.9080 | 0.6727 | 0.9276 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **23.27%** | [21.60, 24.67] | 0.9647 | 0.8553 | 0.8450 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **24.45%** | [22.60, 26.35] | 0.9460 | 0.9130 | 0.8411 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **20.69%** | [19.83, 21.50] | 0.9648 | 0.8110 | 0.8758 |
| **Universal Macro-Average (3 Domains)** | — | **20.82%** | — | 0.9396 | 0.8137 | 0.8712 |
| **VoxCeleb1-O 10-speaker subset** (see caveat) | 2,500 | **25.33%** | [23.29, 27.46] | 0.9642 | 0.9230 | 0.8289 |

Caveat: VoxCeleb1-O **10-speaker subset**: the official `veri_test2.txt` list filtered to trials whose enrollment and test speakers are both among the 10 held-out evaluation speakers (2,500 trials, ~72% target). **Not comparable to published VoxCeleb1-O numbers**; the other 30 of the 40 official test speakers are used as training data by `vox_*` and `univ_*` models.

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **8.30%** | [6.75, 10.35] | 0.6960 | 0.9721 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **17.10%** | [14.65, 19.80] | 0.9240 | 0.8969 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **19.40%** | [17.00, 21.95] | 0.9640 | 0.8866 |
| **Universal Pooled (3-Utt)** | 3,000 | **15.33%** | [14.10, 16.67] | 0.9073 | 0.9232 |
| **Universal Macro-Average (3-Utt)** | — | **14.93%** | — | 0.8613 | 0.9185 |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 400 gradient steps x 32 utterances (N = 8 speakers x M = 4) on <= 25 utterances/speaker from 381 (universal) or fewer speakers; the source papers train for ~500k steps with N = 16-64 / M = 8-10 on thousands of speakers.
- **Augmentation**: synthetic log-Mel-domain noise + one-tap smoothing (`specaug.apply_feature_mtr_augmentation`), not waveform MTR with recorded noise/RIR corpora.
- **Objective**: hybrid contrast surrogate (gradient = 0.25 Eq. 7 + 0.25 BCE + 0.50 softmax-CE) with `ge2e_init_b = -1`, not Wan et al. Eq. 7 alone. Set `ge2e_contrast_surrogate: false` for the pure Eq. 7 gradient.
- **Conformer**: parallel linear + Conv1D stem, no positional encoding, 3 blocks x 192-D, 4 heads, kernel 15, mask-aware GroupNorm(2), no dropout, multi-layer feature aggregation; the Google papers use 12 x 512 blocks, 4-8 heads, kernel 32, relative positional encoding and dropout.
- **ATP pooling**: 4 heads with a tanh bottleneck (128-D) and `subspace_heads = True`; the papers use a single head with a linear-sigmoid weight (`attention_dim = 0`). Note that arXiv:2202.12163 is a language-identification paper whose pooling head is repurposed here.
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
