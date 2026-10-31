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
- extended_set_softmax
- dr_vector
datasets:
- librispeech_asr
- cnceleb
- voxceleb
metrics:
- eer
- min_dcf
- roc_auc
---

# FlaxSpeaker: `univ_full_conformer_large_dr_vectors` (`conformer` - `large`)

## Disclaimer & Reproduction Notice

> **This is not an officially supported Google product.**
>
> This checkpoint and the accompanying `FlaxSpeaker` library are a from-scratch re-implementation in JAX/Flax, written using only the published papers (no proprietary code or data), of speaker recognition algorithms from:
> - **GE2E**: *Generalized End-to-End Loss for Speaker Verification* ([arXiv:1710.10467](https://arxiv.org/abs/1710.10467))
> - **Dr-Vectors**: *Decision Residual Networks and an Improved Loss for Speaker Recognition* ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989))
> - **ATP** (pooling head repurposed from a *language identification* paper): *Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech* ([arXiv:2202.12163](https://arxiv.org/abs/2202.12163))
> - **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* ([arXiv:2203.05642](https://arxiv.org/abs/2203.05642))
>
> The original papers were run on Google-internal infrastructure and data; this checkpoint is a **full-data open-corpus universal flagship** trained on **12,193 speakers (464,557 clean utterances + MUSAN/RIR/speed augmentation)** across VoxCeleb2-dev, VoxCeleb1-dev, CN-Celeb 1+2, and LibriSpeech 960h with zero speaker overlap against official evaluation sets.

## Training Budget (read this first)

This flagship model was trained for **2,400 gradient steps** (warmup `50` steps, batch size `32`) over the full multi-corpus universal training pool (VoxCeleb1/2 + CN-Celeb1/2 + LibriSpeech 960h) (12,193 base speakers, 464,557 clean utterances, expanded via 3-way speed perturbation + MUSAN/RIR).

**Training-crop provenance.** This checkpoint was trained with `U(80, 200)`-frame variable-length crops and an exact validity mask (the current pipeline default); the shipped `config.json` / YAML (`variable_length_training: true`) describes the released weights. Step-0 loss fingerprint at the released seed (`scripts/fingerprint_released_checkpoints.py`, `pretrained_models/step0_fingerprints.json`): archived `1.3912`; current code with fixed 200-frame crops `1.1641`, with `U(80, 200)` crops `1.3912` (match: `variable`). Evaluating the released weights with the current code reproduces the archived metrics in `eval_results.json`. At the converged `S_safe` budget, the seed-to-seed standard deviation of pooled 1-utt EER across seeds 0-2 is 0.08-0.41 % absolute (`pretrained_models/r2_seed_and_control_results.json`), while the 8,000-trial 95 % bootstrap CI half-width is ~0.8 %.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model ID** | `univ_full_conformer_large_dr_vectors` |
| **Hugging Face Repository** | [`wq2012/univ_full_conformer_large_dr_vectors`](https://huggingface.co/wq2012/univ_full_conformer_large_dr_vectors) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `conformer` (`large`) |
| **Temporal Pooling** | `stats` |
| **Training Objective** | Blended Extended-Set Softmax (FlaxSpeaker variant: 0.50 row-softmax + 0.50 extended-set with a -log N logit shift; not the paper's Eq. 2) (`extended_set_softmax`) |
| **Verification Scorer** | Down-scaled Dr-Vector variant (no base/side-info split, hidden (256, 128) relu + LayerNorm, fixed residual weight 0.15) (`dr_vector`) |
| **Output Representation Dim** | `256` |
| **Inference Parameters (encoder + scoring head; excludes training-only classifier weights)** | `32,059,395` |
| **Training Data** | VoxCeleb2-dev + VoxCeleb1-dev + CN-Celeb1/2 + LibriSpeech 960h (full multi-domain SOTA flagship, 12,193 speakers, 464,557 utterances) |
| **Training Steps / Batch** | `2,400` steps x 32 utterances (~0.17 epochs), seed 42 |
| **Audio Frontend** | `80`-dim natural-log Mel @ `16000` Hz + frame-level energy VAD + CMVN (`flaxspeaker.AudioFrontend`; **not** a Hugging Face `AutoFeatureExtractor`) |

## Model Files Included

- `model.safetensors`: model weights (encoder + GE2E `w,b` / Dr-Vector / PFAS heads) in SafeTensors format
- `flax_model.msgpack`: native Flax Msgpack serialized parameters
- `config.json`: `FlaxSpeakerConfig` (full `ExperimentConfig`, incl. the frontend/VAD settings)
- `eval_results.json`: multi-domain evaluation metrics (1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs, VoxCeleb1-O 10-speaker subset)

## Quickstart Usage

### Python API (from Hugging Face Hub or Local Directory)

Requires `flaxspeaker >= 0.5.0` (`pip install -U flaxspeaker`).

```python
from flaxspeaker import load_pretrained

model = load_pretrained("wq2012/univ_full_conformer_large_dr_vectors")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_full_conformer_large_dr_vectors \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Official Zero-Shot Speaker Verification Benchmarks

| Benchmark | Trials | Raw EER (%) | AS-Norm EER (%) | AS-Norm + QMF EER (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VoxCeleb1-O (`veri_test2.txt`, 40 spks)** | 37,611 | **2.17%** | **2.05%** | **2.07%** | 0.2303 | 0.9982 |
| **CN-Celeb-E (`eval/lists/trials.lst`, 200 spks)** | 3,484,292 | **11.38%** | **11.48%** | **11.03%** | 0.5944 | 0.9561 |
| **CN-Celeb-E Multi-Enroll (`enroll.map`)** | 3,484,292 | **10.28%** | **10.23%** | — | 0.5210 | 0.9645 |
| **LibriSpeech `test-clean` (40 spks)** | 10,000 | **1.05%** | **0.97%** | **0.84%** | 0.0682 | 0.9993 |
| **LibriSpeech `test-clean` (3-Utt)** | 3,000 | **0.53%** | **0.33%** | — | 0.0140 | 0.9999 |

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

Trial lists are 50/50 target/impostor over held-out speakers (LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is close to its trivial ceiling of 1.0 for every model under this trial mix.

| Evaluation Split | Trials | Raw EER (%) | AS-Norm EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **1.13%** | **1.13%** | [0.73, 1.57] | 0.0947 | 0.0500 | 0.9991 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **8.07%** | **8.03%** | [6.88, 9.00] | 0.3013 | 0.2613 | 0.9772 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **1.65%** | **1.55%** | [1.10, 2.20] | 0.0490 | 0.0490 | 0.9991 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **4.01%** | **3.83%** | [3.62, 4.44] | 0.2520 | 0.1597 | 0.9938 |
| **Universal Macro-Average (3 Domains)** | — | **3.62%** | **3.57%** | — | 0.1483 | 0.1201 | 0.9918 |

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **0.80%** | [0.30, 1.60] | 0.0300 | 0.9994 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **4.50%** | [3.40, 6.05] | 0.2500 | 0.9916 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **0.70%** | [0.20, 1.30] | 0.0240 | 0.9999 |
| **Universal Pooled (3-Utt)** | 3,000 | **2.17%** | [1.65, 2.67] | 0.1727 | 0.9979 |
| **Universal Macro-Average (3-Utt)** | — | **2.00%** | — | 0.1013 | 0.9970 |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 500 gradient steps x 32 utterances (~0.03 epochs) across 12,193 base speakers (464,557 clean utterances + 3x speed perturbation + MUSAN/RIR).
- **Augmentation**: synthetic log-Mel-domain noise + one-tap smoothing (`specaug.apply_feature_mtr_augmentation`), not waveform MTR with recorded noise/RIR corpora.
- **Objective**: 50/50 blend of row-softmax and `-log N`-shifted extended-set softmax; the paper's pure Eq. 2 is available via `pure_extended_set: true`.
- **Dr-Vector scorer**: no 200/56 base/side-information split, hidden (256, 128) with relu + LayerNorm, linear read-out and a fixed residual weight 0.15 instead of the paper's 3 x 256 leaky-ReLU(0.2) layers, weighted-sum read-out and trainable affine.
- **Conformer**: 2D `ResNet-34` acoustic stem (`base_channels=64`, `wespeaker_mode=True`, 8x time subsampling) + 4 Conformer blocks x 320-D (8 heads, kernel 15).
- **Frontend**: 80-D Kaldi HTK Mel-filterbank (Povey window, 20-7600 Hz, continuous frames, `vad.enabled=False`) + CMVN; evaluation processes up to 768 frames (7.7 s).

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
