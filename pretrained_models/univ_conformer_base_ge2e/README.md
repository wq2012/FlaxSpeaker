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

# FlaxSpeaker: `univ_conformer_base_ge2e` (`conformer` - `base`)

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
| **Model ID** | `univ_conformer_base_ge2e` |
| **Hugging Face Repository** | [`wq2012/univ_conformer_base_ge2e`](https://huggingface.co/wq2012/univ_conformer_base_ge2e) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `conformer` (`base`) |
| **Temporal Pooling** | `atp` |
| **Training Objective** | GE2E softmax (Wan et al. 2018, Eq. 6; exact leave-one-out centroids) (`ge2e_softmax`) |
| **Verification Scorer** | Cosine similarity (`cosine`) |
| **Output Representation Dim** | `256` |
| **Total Parameters (encoder + loss/scoring heads)** | `4,551,558` |
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

model = load_pretrained("wq2012/univ_conformer_base_ge2e")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_conformer_base_ge2e \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

Trial lists are 50/50 target/impostor over held-out speakers (LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is close to its trivial ceiling of 1.0 for every model under this trial mix.

| Evaluation Split | Trials | EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **15.37%** | [14.13, 16.64] | 0.9467 | 0.7647 | 0.9273 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **22.40%** | [20.68, 23.80] | 0.9893 | 0.9380 | 0.8569 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **22.85%** | [20.80, 24.45] | 0.9930 | 0.9280 | 0.8605 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **19.85%** | [19.04, 20.76] | 0.9890 | 0.9085 | 0.8840 |
| **Universal Macro-Average (3 Domains)** | — | **20.21%** | — | 0.9763 | 0.8769 | 0.8816 |
| **VoxCeleb1-O 10-speaker subset** (see caveat) | 2,500 | **25.01%** | [23.26, 27.02] | 0.9609 | 0.9609 | 0.8369 |

Caveat: VoxCeleb1-O **10-speaker subset**: the official `veri_test2.txt` list filtered to trials whose enrollment and test speakers are both among the 10 held-out evaluation speakers (2,500 trials, ~72% target). **Not comparable to published VoxCeleb1-O numbers**; the other 30 of the 40 official test speakers are used as training data by `vox_*` and `univ_*` models.

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **11.10%** | [9.15, 13.35] | 0.6420 | 0.9614 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **18.50%** | [16.05, 21.00] | 0.9200 | 0.8951 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **19.90%** | [17.05, 21.80] | 0.9420 | 0.8925 |
| **Universal Pooled (3-Utt)** | 3,000 | **16.80%** | [15.27, 18.02] | 0.9140 | 0.9163 |
| **Universal Macro-Average (3-Utt)** | — | **16.50%** | — | 0.8347 | 0.9163 |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 400 gradient steps x 32 utterances (N = 8 speakers x M = 4) on <= 25 utterances/speaker from 381 (universal) or fewer speakers; the source papers train for ~500k steps with N = 16-64 / M = 8-10 on thousands of speakers.
- **Augmentation**: synthetic log-Mel-domain noise + one-tap smoothing (`specaug.apply_feature_mtr_augmentation`), not waveform MTR with recorded noise/RIR corpora.
- **Conformer**: parallel linear + Conv1D stem, no positional encoding, 4 blocks x 256-D, 8 heads, kernel 15, mask-aware GroupNorm(2), no dropout, multi-layer feature aggregation; the Google papers use 12 x 512 blocks, 4-8 heads, kernel 32, relative positional encoding and dropout.
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
