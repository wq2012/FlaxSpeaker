# FlaxSpeaker

[![Python application](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml/badge.svg)](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml) [![PyPI Version](https://img.shields.io/pypi/v/flaxspeaker.svg)](https://pypi.python.org/pypi/flaxspeaker) [![Python Versions](https://img.shields.io/pypi/pyversions/flaxspeaker.svg)](https://pypi.org/project/flaxspeaker) [![Downloads](https://static.pepy.tech/badge/flaxspeaker)](https://pepy.tech/project/flaxspeaker)

**FlaxSpeaker** is a modern, end-to-end speaker recognition and verification library built on [JAX](https://jax.readthedocs.io) and [Flax](https://flax.readthedocs.io). Designed for education, academic research, and production deployment, it unifies dataset preparation, Voice Activity Detection (VAD), HuggingFace `transformers`-compatible acoustic frontends, six neural sequence backbones, seven pooling heads, eight metric-learning and angular-margin loss functions, three scoring paradigms, and one-command export to **TFLite (FP32 / INT8)**, **TensorFlow SavedModel**, **SafeTensors**, and **HuggingFace Hub**.

For the companion PyTorch educational project, see [SpeakerRecognitionFromScratch](https://github.com/wq2012/SpeakerRecognitionFromScratch).

---

## Key Features

### 1. Six Neural Sequence Backbones & Four Size Presets (`< 100M` params)
All backbones share a unified `(B, T, F) -> (B, T', D)` sequence interface via `ModernSpeakerEncoder` ([`flaxspeaker/backbones.py`](flaxspeaker/backbones.py)) and scale automatically via `size_variant: tiny | small | base | large`:
- **Conformer** (`conformer`): Macaron FFN + Multi-Head Self-Attention + Depthwise Separable Convolution + LayerNorm.
- **ECAPA-TDNN** (`ecapa_tdnn`): 1D SE-Res2Net dilated multi-scale blocks with Multi-Layer Feature Aggregation (MFA).
- **Mamba** (`mamba`): Selective State Space Model (SSM) implemented in pure JAX via parallel `jax.lax.associative_scan` ($O(T)$ complexity).
- **ResNet** (`resnet`): 2D Time-Frequency Residual Network with Squeeze-and-Excitation (SE-ResNet).
- **Transformer** (`transformer`): Pre-LayerNorm Multi-Head Self-Attention encoder.
- **LSTM** (`lstm`): Deep stacked unidirectional LSTM with configurable hidden size and dropout.

### 2. Seven Temporal Pooling Heads ([`flaxspeaker/pooling.py`](flaxspeaker/pooling.py))
- **Mean Pooling** (`mean`) & **Last-Frame Pooling** (`last`)
- **Temporal Statistics Pooling** (`stats`): Concatenates temporal mean and standard deviation $(\mu \oplus \sigma)$.
- **Self-Attentive Pooling** (`sap`) & **Attentive Statistics Pooling** (`asp`): Attention-weighted frame mean and standard deviation.
- **Cumulative Statistics Pooling** (`cumulative_stats`): Frame-wise causal running statistics for streaming inference.
- **Parameter-Free Attentive Scoring Head** (`pfas`): Dual-branch $(K, V)$ projection head with unit-L2-normalized frame keys and values ([arXiv:2203.05642v3](https://arxiv.org/pdf/2203.05642v3)).

### 3. Eight Speaker Recognition Losses & Three Scoring Backends ([`flaxspeaker/losses.py`](flaxspeaker/losses.py), [`flaxspeaker/scoring.py`](flaxspeaker/scoring.py))
- **Generalized End-to-End (GE2E) Loss**: `ge2e` (Softmax) and `ge2e_contrast` (Contrast) with learnable scale $w$ and bias $b$ ([Wan et al., ICASSP 2018](https://arxiv.org/pdf/1710.10467)).
- **Extended-Set (Multi-Row) Softmax Loss**: `extended_set_softmax` for training with negative prototype banks ([Pelecanos et al., Interspeech 2021](https://arxiv.org/pdf/2104.01989)).
- **Decision Residual Networks (Dr-Vectors)**: `scoring.type: dr_vectors` — residual feed-forward network scoring elementwise interactions $[e_1, e_2, e_1 \odot e_2, |e_1 - e_2|, (e_1 - e_2)^2]$ ([arXiv:2104.01989](https://arxiv.org/pdf/2104.01989)).
- **Parameter-Free Attentive Scoring (PFAS)**: `scoring.type: pfas` — asymmetric cross-attention between test and enrollment $(K, V)$ representations with seamless multi-utterance concatenation ([Pelecanos et al., Odyssey 2022](https://arxiv.org/pdf/2203.05642v3)).
- **Angular Margin Classification Losses**:
  - **ArcFace / AAM-Softmax** (`arcface`): $\cos(\theta + m)$
  - **CosFace / AM-Softmax** (`cosface`): $\cos(\theta) - m$
  - **SphereFace / A-Softmax** (`sphereface`): $(-1)^k \cos(m\theta) - 2k$
  - **Standard Softmax** (`softmax`), **Pairwise Loss** (`pairwise`), and **Triplet Loss** (`triplet`).
- **Adaptive Symmetric Score Normalization (AS-Norm)**: Top-$K$ imposter cohort calibration (`scoring.use_as_norm: true`).

### 4. End-to-End Audio Pipeline, Datasets & Production Export
- **Voice Activity Detection (VAD)** ([`flaxspeaker/vad.py`](flaxspeaker/vad.py)): `energy`, `spectral`, `hybrid`, or `none`.
- **Acoustic Frontends** ([`flaxspeaker/feature_extraction.py`](flaxspeaker/feature_extraction.py)): `log_mel`, `whisper_mel`, `hf_compat` (100% compatible with HuggingFace `WhisperFeatureExtractor` / `SeamlessM4TFeatureExtractor`), `mfcc`, Cepstral Mean and Variance Normalization (CMVN), SpecAugment ([`flaxspeaker/specaug.py`](flaxspeaker/specaug.py)), and two-tier RAM + `.npy` disk caching (`FeatureCache`).
- **Academic Dataset Ingestion** ([`flaxspeaker/dataset.py`](flaxspeaker/dataset.py)): Out-of-the-box scanners and speaker-disjoint split/trial generators for **LibriSpeech**, **VoxCeleb1/2** (including official `veri_test2.txt`), **CN-Celeb1/2**, and custom CSV manifests.
- **Production Model Export** ([`flaxspeaker/export.py`](flaxspeaker/export.py), [`flaxspeaker/hf_compat.py`](flaxspeaker/hf_compat.py)): Export via `jax2tf` to **TFLite (FP32, dynamic-range INT8, full INT8)**, **TF SavedModel**, **SafeTensors**, and **HuggingFace Hub** directories (`config.json`, `model.safetensors`, `preprocessor_config.json`, auto-generated model card).

---

## Installation

```bash
pip install flaxspeaker
```

For development, HuggingFace export, and TFLite conversion extras:
```bash
pip install -e ".[dev,export,hf]"
```

---

## Quickstart Guide

### 1. Prepare Datasets & Academic Splits
Given a local directory containing **LibriSpeech**, **CN-Celeb**, or **VoxCeleb**, generate speaker-disjoint `train.csv`, `eval_enroll.csv`, `eval_test.csv`, single-utterance `eval_trials.csv`, and multi-utterance `eval_multi_trials.csv`:

```bash
python -m flaxspeaker \
  --mode prepare_dataset \
  --dataset_type librispeech \
  --path_to_dataset /path/to/LibriSpeech/dev-clean \
  --output_dir ./splits/librispeech \
  --train_speaker_ratio 0.8 \
  --num_eval_trials 3000
```

Or generate a flat CSV manifest from any directory of audio files:
```bash
python -m flaxspeaker \
  --mode generate_csv \
  --path_to_dataset "${HOME}/Downloads/CN-Celeb_flac/data" \
  --audio_format ".flac" \
  --speaker_label_index -2 \
  --output_csv "CN-Celeb.csv"
```

### 2. Configure an Experiment (`config.yml`)
FlaxSpeaker supports both the structured v0.2 YAML schema and legacy v0.1 `myconfig.yml` files:

```yaml
data:
  train_csv: "./splits/librispeech/train.csv"
  test_csv: "./splits/librispeech/eval_test.csv"
  eval_trials_csv: "./splits/librispeech/eval_trials.csv"

vad:
  mode: "energy"
  energy_threshold_db: -35.0

feature:
  feature_type: "log_mel"
  sample_rate: 16000
  n_mels: 80
  apply_cmvn: true
  cache_dir: "./cache/log_mel80"

specaug:
  enabled: true
  freq_mask_max_width: 8
  time_mask_max_width: 20

model:
  backbone: "conformer"      # lstm | transformer | conformer | mamba | ecapa_tdnn | resnet
  size_variant: "small"      # tiny | small | base | large
  pooling: "asp"             # mean | last | stats | sap | asp | cumulative_stats | pfas
  embedding_dim: 192
  dropout_rate: 0.1

scoring:
  scoring_type: "cosine"     # cosine | pfas | dr_vectors
  use_as_norm: false

loss:
  loss_type: "ge2e"          # ge2e | ge2e_contrast | extended_set_softmax | arcface | cosface | sphereface | softmax | pairwise | triplet

training:
  batch_num_speakers: 8
  batch_num_utterances: 6
  num_steps: 2500
  learning_rate: 0.001
  lr_schedule: "cosine"
  warmup_steps: 200
  model_dir: "./checkpoints/conformer_small_ge2e"
```

### 3. Train, Evaluate & Export

```bash
# Train
python -m flaxspeaker --mode train --config config.yml

# Evaluate (computes EER, minDCF@0.01, minDCF@0.05, ROC-AUC, 1-utt & multi-utt)
python -m flaxspeaker --mode eval --config config.yml

# Export to HuggingFace Hub directory + TFLite (FP32 & INT8)
python -m flaxspeaker \
  --mode export \
  --config config.yml \
  --export_dir ./exported_model \
  --export_formats hf,safetensors,tflite_fp32,tflite_int8
```

---

## Python & HuggingFace API

Load any exported model directory with `FlaxSpeakerForSpeakerVerification`:

```python
import numpy as np
from flaxspeaker.hf_compat import FlaxSpeakerForSpeakerVerification

# Load pretrained checkpoint + audio frontend
verifier = FlaxSpeakerForSpeakerVerification.from_pretrained(
    "pretrained_models/ls_ecapa_tdnn_small_ge2e"
)

# Extract L2-normalized speaker embeddings from 16kHz waveforms
wav_enroll = np.random.randn(16000 * 3).astype(np.float32)
wav_test = np.random.randn(16000 * 3).astype(np.float32)

emb_enroll = verifier.extract_embedding(wav_enroll)
emb_test = verifier.extract_embedding(wav_test)

# Compute verification similarity score (supports Cosine, PFAS, and Dr-Vectors)
score = verifier.verify(wav_enroll, wav_test)
print(f"Verification score: {score:.4f}")
```

---

## Benchmark Summary & Pretrained Model Zoo

We conducted a 24-model benchmark study across **LibriSpeech**, **CN-Celeb2**, and **VoxCeleb1-O** evaluating all 6 backbones, 8 loss functions, 4 size variants (`95K` to `6.81M` parameters), and 3 scoring paradigms.

- Full technical report & mathematical formulations: **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)**
- Raw verified evaluation metrics JSON: **[`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json)**
- Pretrained HuggingFace + TFLite checkpoints: **[`pretrained_models/`](pretrained_models/)**

### Highlight Results (Verified from [`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json))

| Category | Best Experiment ID | Backbone / Loss / Scoring | Params | 1-Utt EER (%) | 3-Utt EER (%) | Zero-Shot VoxCeleb1-O EER (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **LibriSpeech Best Overall** | `ls_ecapa_tdnn_base_ge2e` | `ecapa_tdnn` (`base`) + `ge2e` | `2,994,818` | **8.40%** | **5.70%** | 30.88% |
| **LibriSpeech Best Small** | `ls_ecapa_tdnn_small_ge2e` | `ecapa_tdnn` (`small`) + `ge2e` | `820,482` | **8.80%** | **5.50%** | 29.88% |
| **LibriSpeech Best Sub-300K** | `ls_ecapa_tdnn_tiny_ge2e` | `ecapa_tdnn` (`tiny`) + `ge2e` | `257,986` | **11.13%** | **8.30%** | 30.69% |
| **Best Zero-Shot Transfer** | `ls_mamba_small_ge2e` | `mamba` (`small`) + `ge2e` | `954,178` | 15.07% | 12.20% | **26.88%** (32.03% CN-Celeb2) |
| **CN-Celeb2 Best Multi-Genre** | `cn_conformer_small_dr_vectors` | `conformer` (`small`) + `extended_set` + `dr_vectors` | `786,497` | **28.43%** | **22.30%** | — |
| **CN-Celeb2 Best Cosine** | `cn_ecapa_tdnn_small_ge2e` | `ecapa_tdnn` (`small`) + `ge2e` | `820,482` | **28.47%** | 24.30% | — |

---

## Testing & Code Quality

Run the unit test suite (`34` tests covering all backbones, losses, pooling heads, VAD, SpecAugment, scoring, dataset loaders, HF compatibility, TFLite export, and legacy config compatibility):

```bash
pytest -v tests.py
flake8 flaxspeaker tests.py
```
