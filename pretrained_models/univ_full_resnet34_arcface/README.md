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
- resnet
- subcenter_arcface
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

# FlaxSpeaker: `univ_full_resnet34_arcface` (`resnet` - `large`)

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

This flagship model was trained for **4,500 gradient steps** (warmup `50` steps, batch size `32`) over the full multi-corpus universal training pool (VoxCeleb1/2 + CN-Celeb1/2 + LibriSpeech 960h) (12,193 base speakers, 464,557 clean utterances, expanded via 3-way speed perturbation + MUSAN/RIR).

**Training-crop provenance.** This checkpoint was trained with `U(80, 200)`-frame variable-length crops and an exact validity mask (the current pipeline default); the shipped `config.json` / YAML (`variable_length_training: true`) describes the released weights. Step-0 loss fingerprint at the released seed (`scripts/fingerprint_released_checkpoints.py`, `pretrained_models/step0_fingerprints.json`): archived `2.7714`; current code with fixed 200-frame crops `2.6614`, with `U(80, 200)` crops `2.7714` (match: `variable`). Evaluating the released weights with the current code reproduces the archived metrics in `eval_results.json`. At the converged `S_safe` budget, the seed-to-seed standard deviation of pooled 1-utt EER across seeds 0-2 is 0.08-0.41 % absolute (`pretrained_models/r2_seed_and_control_results.json`), while the 8,000-trial 95 % bootstrap CI half-width is ~0.8 %.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Model ID** | `univ_full_resnet34_arcface` |
| **Hugging Face Repository** | [`wq2012/univ_full_resnet34_arcface`](https://huggingface.co/wq2012/univ_full_resnet34_arcface) |
| **Hugging Face Collection** | [FlaxSpeaker Collection](https://huggingface.co/collections/wq2012/flaxspeaker) |
| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |
| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |
| **Backbone Architecture** | `resnet` (`large`) |
| **Temporal Pooling** | `stats` |
| **Training Objective** | Sub-Center ArcFace (Deng et al. 2020; K=3, scale=32, margin=0.3) (`subcenter_arcface`) |
| **Verification Scorer** | Cosine similarity (`cosine`) |
| **Output Representation Dim** | `256` |
| **Inference Parameters (encoder + scoring head; excludes training-only classifier weights)** | `23,897,536` |
| **Training Data** | VoxCeleb2-dev + VoxCeleb1-dev + CN-Celeb1/2 + LibriSpeech 960h (full multi-domain SOTA flagship, 12,193 speakers, 464,557 utterances) |
| **Training Steps / Batch** | `4,500` steps x 32 utterances (~0.31 epochs), seed 42 |
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

model = load_pretrained("wq2012/univ_full_resnet34_arcface")
emb = model.extract_embedding("path/to/utterance.flac")
score = model.verify("path/to/enroll.flac", "path/to/test.flac")
print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")
```

### Command-Line Inference

```bash
python scripts/inference.py --model_dir wq2012/univ_full_resnet34_arcface \
    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac
```

## Official Zero-Shot Speaker Verification Benchmarks

| Benchmark | Trials | Raw EER (%) | AS-Norm EER (%) | AS-Norm + QMF EER (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VoxCeleb1-O (`veri_test2.txt`, 40 spks)** | 37,611 | **1.69%** | **1.26%** | **1.34%** | 0.2018 | 0.9988 |
| **CN-Celeb-E (`eval/lists/trials.lst`, 200 spks)** | 3,484,292 | **9.65%** | **10.07%** | **10.18%** | 0.6731 | 0.9672 |
| **CN-Celeb-E Multi-Enroll (`enroll.map`)** | 3,484,292 | **9.02%** | **9.02%** | — | 0.6135 | 0.9727 |
| **LibriSpeech `test-clean` (40 spks)** | 10,000 | **1.12%** | **0.87%** | **0.69%** | 0.0328 | 0.9993 |
| **LibriSpeech `test-clean` (3-Utt)** | 3,000 | **0.33%** | **0.20%** | — | 0.0033 | 1.0000 |

## Multi-Domain Evaluation Results

### 1-Utterance Verification Across Evaluation Domains

Trial lists are 50/50 target/impostor over held-out speakers (LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is close to its trivial ceiling of 1.0 for every model under this trial mix.

| Evaluation Split | Trials | Raw EER (%) | AS-Norm EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | minDCF@0.10 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (English Read)** | 3,000 | **0.87%** | **0.80%** | [0.60, 1.30] | 0.0753 | 0.0367 | 0.9991 |
| **CN-Celeb2 (Chinese Multi-Genre)** | 3,000 | **7.30%** | **7.00%** | [6.27, 8.32] | 0.3787 | 0.2220 | 0.9785 |
| **VoxCeleb1 (In-the-Wild Celebrity)** | 2,000 | **1.40%** | **1.15%** | [0.90, 2.00] | 0.0580 | 0.0530 | 0.9992 |
| **Universal Pooled (All 3 Domains)** | 8,000 | **3.67%** | **3.76%** | [3.26, 4.12] | 0.2870 | 0.1295 | 0.9932 |
| **Universal Macro-Average (3 Domains)** | — | **3.19%** | **2.98%** | — | 0.1707 | 0.1039 | 0.9923 |

### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains

| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech (3-Utt)** | 1,000 | **0.50%** | [0.20, 1.45] | 0.0200 | 0.9994 |
| **CN-Celeb2 (3-Utt)** | 1,000 | **5.00%** | [3.40, 6.40] | 0.3820 | 0.9912 |
| **VoxCeleb1 (3-Utt)** | 1,000 | **0.50%** | [0.10, 1.10] | 0.0520 | 0.9999 |
| **Universal Pooled (3-Utt)** | 3,000 | **2.23%** | [1.67, 2.70] | 0.1780 | 0.9976 |
| **Universal Macro-Average (3-Utt)** | — | **2.00%** | — | 0.1513 | 0.9968 |

Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences below about 1% absolute EER between FlaxSpeaker checkpoints are within the seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` (see the GitHub README) and should not be interpreted as rankings.

## Deviations from the Source Papers

- **Training regime**: 500 gradient steps x 32 utterances (~0.03 epochs) across 12,193 base speakers (464,557 clean utterances + 3x speed perturbation + MUSAN/RIR).
- **Augmentation**: synthetic log-Mel-domain noise + one-tap smoothing (`specaug.apply_feature_mtr_augmentation`), not waveform MTR with recorded noise/RIR corpora.
- **ResNet**: 2D Time-Frequency ResNet (`base_channels=64`, `num_blocks=(3, 4, 6, 3)`, `wespeaker_mode=True`).
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
