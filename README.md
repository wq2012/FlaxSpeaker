# FlaxSpeaker

[![Python application](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml/badge.svg)](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml) [![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-44%20Pretrained%20Models-blue)](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) [![PyPI Version](https://img.shields.io/pypi/v/flaxspeaker.svg)](https://pypi.python.org/pypi/flaxspeaker) [![Python Versions](https://img.shields.io/pypi/pyversions/flaxspeaker.svg)](https://pypi.org/project/flaxspeaker) [![Downloads](https://static.pepy.tech/badge/flaxspeaker)](https://pepy.tech/project/flaxspeaker) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**FlaxSpeaker** is a modern, end-to-end open-source speaker recognition and verification library built on [JAX](https://jax.readthedocs.io) and [Flax](https://flax.readthedocs.io). Designed for education, academic research, and production deployment, it unifies dataset preparation, Post-STFT Voice Activity Detection (VAD) with Automatic Gain Control (AGC), HuggingFace `transformers`-compatible acoustic frontends, six neural sequence backbones, eight temporal pooling heads, nine metric-learning and angular-margin loss functions, three scoring paradigms, universal multi-domain stratified training, and one-command export to **TFLite (FP32 / INT8)**, **TensorFlow SavedModel**, **SafeTensors**, and **HuggingFace Hub**.

For the companion PyTorch educational project, see [SpeakerRecognitionFromScratch](https://github.com/wq2012/SpeakerRecognitionFromScratch).

---

## Disclaimer & Open-Source Reproduction Notice

**This is not an officially supported Google product.**

This repository provides an independent, clean-room open-source re-implementation in **JAX / Flax** of the mathematical formulations and neural architectures described in the following published research papers:
1. **Generalized End-to-End Loss for Speaker Verification** — *Li Wan, Quan Wang, Alan Papir, Ignacio Lopez Moreno* ([ICASSP 2018, arXiv:1710.10467](https://arxiv.org/pdf/1710.10467))
2. **Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition** — *Jason Pelecanos, Quan Wang, Ignacio Lopez Moreno* ([Interspeech 2021, arXiv:2104.01989](https://arxiv.org/pdf/2104.01989))
3. **Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech** — *Quan Wang, Yang Yu, Jason Pelecanos, Yiling Huang, Ignacio Lopez Moreno* ([Odyssey 2022, arXiv:2202.12163](https://arxiv.org/pdf/2202.12163))
4. **Parameter-Free Attentive Scoring for Speaker Verification** — *Jason Pelecanos, Quan Wang, Yiling Huang, Ignacio Lopez Moreno* ([Odyssey 2022, arXiv:2203.05642v3](https://arxiv.org/pdf/2203.05642v3))

Because the original papers were trained and evaluated on proprietary internal datasets and infrastructure, **exact numerical parity with the original paper tables is neither expected nor claimed**. All pretrained checkpoints and benchmark tables in this repository are trained and evaluated from scratch on publicly available open-source speech corpora (**LibriSpeech**, **CN-Celeb2**, and **VoxCeleb1**) with speaker-disjoint splits.

---

## System Architecture

![FlaxSpeaker System Architecture](resources/architecture_diagram.png)

### 1. Six Neural Sequence Backbones & Four Size Presets (`< 100M` params)
All backbones share a unified `(B, T, F) -> (B, T', D)` sequence interface with **exact variable-length padding mask invariance** via `ModernSpeakerEncoder` ([`flaxspeaker/backbones.py`](flaxspeaker/backbones.py)) and scale automatically via `size_variant: tiny | small | base | large`:
- **Conformer** (`conformer`): 2-Layer ConvSubsampling Stem (`ConvStem`) + Macaron FFN + Masked Multi-Head Self-Attention + Mask-Aware GroupNorm Depthwise Separable Convolution + Multi-Layer Feature Aggregation (MFA).
- **ECAPA-TDNN** (`ecapa_tdnn`): 1D SE-Res2Net dilated multi-scale blocks with Multi-Layer Feature Aggregation (MFA).
- **Mamba** (`mamba`): Selective State Space Model (SSM) implemented in pure JAX via parallel `jax.lax.associative_scan` (`O(T)` linear time complexity, supporting both causal unidirectional and mask-aware bidirectional `bidirectional: true` modes).
- **ResNet** (`resnet`): 2D Time-Frequency Residual Network with Squeeze-and-Excitation (SE-ResNet).
- **Transformer** (`transformer`): 2-Layer ConvSubsampling Stem (`ConvStem`) + Pre-LayerNorm Multi-Head Self-Attention encoder with Multi-Layer Feature Aggregation (MFA).
- **LSTM** (`lstm`): Deep stacked unidirectional LSTM with configurable hidden size and dropout.

### 2. Eight Temporal Pooling Heads ([`flaxspeaker/pooling.py`](flaxspeaker/pooling.py))
- **Attentive Temporal Pooling** (`atp`): Sigmoid frame-gated subspace multi-head statistics pooling (`concat(mu_h, sigma_h)`) for Conformer and Transformer speaker encoders ([Wang et al., Odyssey 2022, `arXiv:2202.12163`](https://arxiv.org/pdf/2202.12163)).
- **Self-Attentive Pooling** (`sap`) & **Attentive Statistics Pooling** (`asp`): Softmax attention-weighted frame mean and standard deviation.
- **Mean Pooling** (`mean`), **Last-Frame Pooling** (`last`), & **Temporal Statistics Pooling** (`stats`).
- **Cumulative Statistics Pooling** (`cumulative_stats`): Frame-wise causal running statistics for streaming inference.
- **Parameter-Free Attentive Scoring Head** (`pfas`): Global `ATP` bottleneck value anchor + multi-head `(K, V)` residual projection with concatenated dual L2-normalization ([Pelecanos et al., Odyssey 2022, `arXiv:2203.05642v3`](https://arxiv.org/pdf/2203.05642v3)).

### 3. Nine Speaker Recognition Losses & Three Scoring Backends ([`flaxspeaker/losses.py`](flaxspeaker/losses.py), [`flaxspeaker/scoring.py`](flaxspeaker/scoring.py))
- **Generalized End-to-End (GE2E) Loss**: `ge2e_softmax` and `ge2e_contrast` with learnable scale `w` and bias `b` ([Wan et al., ICASSP 2018](https://arxiv.org/pdf/1710.10467)).
- **Extended-Set (Multi-Row) Softmax Loss**: `extended_set_softmax` for training with negative prototype banks ([Pelecanos et al., Interspeech 2021](https://arxiv.org/pdf/2104.01989)).
- **Decision Residual Networks (Dr-Vectors)**: `scoring_type: dr_vector` — residual feed-forward network scoring `[e_t, e_e, cos(e_t, e_e)]` ([arXiv:2104.01989](https://arxiv.org/pdf/2104.01989)).
- **Parameter-Free Attentive Scoring (PFAS)**: `scoring_type: pfas` — asymmetric cross-attention between test and enrollment `(K, V)` representations with learnable temperature scaling and seamless multi-utterance enrollment averaging ([Pelecanos et al., Odyssey 2022](https://arxiv.org/pdf/2203.05642v3)).
- **Angular Margin Classification Losses**:
  - **ArcFace / AAM-Softmax** (`arcface`): `cos(theta + m)`
  - **CosFace / AM-Softmax** (`cosface`): `cos(theta) - m`
  - **SphereFace / A-Softmax** (`sphereface`): `(-1)^k * cos(m * theta) - 2k`
  - **Standard Softmax** (`softmax`), **Pairwise Loss** (`pairwise`), and **Triplet Loss** (`triplet`).
- **Adaptive Symmetric Score Normalization (AS-Norm)**: Top-`K` imposter cohort calibration (`use_as_norm: true`).

### 4. End-to-End Audio Pipeline, Mutually Exclusive Augmentation, Variable-Length Training & Universal Multi-Domain Batching
- **Post-STFT Voice Activity Detection (VAD v2) & AGC** ([`flaxspeaker/vad.py`](flaxspeaker/vad.py)): Frame-level energy/spectral/hybrid VAD with Automatic Gain Control (`apply_agc`), relative-to-95th-percentile peak (`26.0 dB`) + adaptive noise floor thresholding, and hangover smoothing enabled identically for both training and evaluation.
- **Acoustic Frontends & Variable-Length Training** ([`flaxspeaker/feature_extraction.py`](flaxspeaker/feature_extraction.py)): `log_mel`, `whisper_mel`, `hf_compat` (100% compatible with HuggingFace `WhisperFeatureExtractor` / `SeamlessM4TFeatureExtractor`), `mfcc`, Cepstral Mean and Variance Normalization (CMVN), per-utterance variable sequence length sampling (`[min_seq_len, max_seq_len]`) with exact binary padding masks `(B, T)`, and two-tier RAM + `.npy` disk caching (`FeatureCache`).
- **Mutually Exclusive SpecAugment & Multi-Style Training (MTR)** ([`flaxspeaker/specaug.py`](flaxspeaker/specaug.py)): Calibrated moderate-SNR (`12–25 dB`) additive noise + room reverberation (MTR) and time/frequency masking (`SpecAugment`), strictly mutually exclusive (`never_simultaneous_with_specaug = True`: `35%` clean, `40%` SpecAugment only, `25%` MTR waveform noise/reverb only).
- **Universal Multi-Domain Stratified Batching** ([`flaxspeaker/dataset.py`](flaxspeaker/dataset.py)): Combines multiple heterogeneous datasets (`LibriSpeech`, `CN-Celeb2`, `VoxCeleb1`) using `75%` within-domain hard-negative batches and `25%` cross-domain score-calibration batches.
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

## Quickstart & CLI Workflows

FlaxSpeaker provides five standalone command-line scripts in [`scripts/`](scripts/) in addition to `python -m flaxspeaker`:

### 1. Prepare Datasets & Multi-Domain Splits (`scripts/prepare_datasets.py` / `scripts/prepare_data.py`)
Generate speaker-disjoint `train.csv`, `eval.csv`, single-utterance `trials.tsv`, 3-utterance `multi_enroll_trials.tsv`, and precomputed feature caches for **LibriSpeech**, **CN-Celeb2**, **VoxCeleb1**, and the combined **Universal** multi-domain split:

```bash
# Prepare all three academic datasets + combined Universal Multi-Domain split
python scripts/prepare_datasets.py --data_root ./data/speaker_datasets

# Or generate a synthetic smoke-test dataset split for local testing
python scripts/prepare_data.py --data_root ./data/synthetic --generate_synthetic
```

### 2. Train Models (`scripts/train.py`)
Train from a YAML config (`configs/*.yml`) or directly from one of the built-in paper preset configurations (`universal_multi_domain`, `icassp2018_ge2e`, `interspeech2021_dr_vectors`, `odyssey2022_conformer_atp`, `odyssey2022_pfas`):

```bash
# Train a Universal Multi-Domain Conformer + ATP + GE2E model
python scripts/train.py \
  --config configs/univ_conformer_small_ge2e.yml \
  --train_csv ./data/speaker_datasets/splits/universal/train.csv \
  --eval_csv ./data/speaker_datasets/splits/universal/eval.csv \
  --trials_file ./data/speaker_datasets/splits/universal/trials.tsv \
  --output_dir ./checkpoints/univ_conformer_small_ge2e
```

### 3. Evaluate 1-Utt and 3-Utt Verification Trials (`scripts/evaluate.py`)
Compute 1-utterance and 3-utterance EER (%), `minDCF@0.01`, `minDCF@0.05`, `minDCF@0.10`, ROC-AUC, and 95% bootstrap confidence intervals using any [Hugging Face Hub model ID (`wq2012/<model_id>`)](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) or local checkpoint directory:

```bash
python scripts/evaluate.py \
  --model_dir wq2012/univ_conformer_small_ge2e \
  --trials_file ./data/speaker_datasets/splits/universal/trials.tsv \
  --multi_trials_file ./data/speaker_datasets/splits/universal/multi_enroll_trials.tsv \
  --output_json eval_summary.json
```

### 4. Run End-to-End Speaker Verification Inference (`scripts/inference.py`)
Verify whether two (or more) audio files belong to the same speaker and optionally save their L2-normalized speaker embeddings:

```bash
python scripts/inference.py \
  --model_dir wq2012/univ_conformer_small_ge2e \
  --enroll_audio hf_space/examples/librispeech_spk1_utt1.flac \
  --test_audio hf_space/examples/librispeech_spk1_utt2.flac
```

### 5. Export to HuggingFace Hub & TFLite (`scripts/export_model.py`)
```bash
python scripts/export_model.py \
  --model_dir wq2012/univ_conformer_small_ge2e \
  --output_dir ./exported_models/univ_conformer_small_ge2e \
  --export_tflite \
  --quantize_int8
```

---

## Python & HuggingFace API

Load any exported model directory or Hugging Face Hub repository ID ([`wq2012/<model_id>`](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595)) with `load_pretrained` or `FlaxSpeakerModel.from_pretrained`:

```python
import flaxspeaker
from flaxspeaker.configs import universal_multi_domain_config

# 1. Load a pretrained universal multi-domain checkpoint from Hugging Face Hub or local path
verifier = flaxspeaker.load_pretrained(
    "wq2012/univ_conformer_small_ge2e"
)

# 2. Verify two audio files directly (supports 1-utt or multi-utt enrollment)
score = verifier.verify(
    "hf_space/examples/librispeech_spk1_utt1.flac",
    "hf_space/examples/librispeech_spk1_utt2.flac",
)
print(f"Verification similarity score: {score:.4f}")

# 3. Or build a paper preset configuration programmatically
cfg = universal_multi_domain_config(
    backbone="conformer",
    size="small",
    loss_type="ge2e_softmax",
    scoring_type="cosine",
)
```

---

## Interactive Gradio Demo (`hf_space/`)

An interactive [Gradio 5.50.0](https://gradio.app) web application is included in [`hf_space/`](hf_space/) featuring:
- Model selector across all 44 Universal Multi-Domain (`univ_*`) and Single-Domain (`ls_*`, `cn_*`, `vox_*`) checkpoints from [`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) (`Cosine`, `PFAS`, and `Dr-Vectors` scoring).
- 1-utterance and 3-utterance speaker enrollment with real speech samples from **LibriSpeech**, **CN-Celeb2**, and **VoxCeleb1**.
- Visual log-Mel spectrogram, Post-STFT VAD frame mask, and 192-D speaker embedding bar plots.

```bash
python hf_space/app.py
```

---

## 44-Model Benchmark Summary & Pretrained Model Zoo

We trained and evaluated **44 models** (`19` Universal Multi-Domain models trained jointly on `LibriSpeech + CN-Celeb2 + VoxCeleb1` = `381` speakers / `9,447` utterances, plus `25` Single-Domain baseline models) across **all three speaker-disjoint evaluation domains** (`8,000` 1-utterance trials + `3,000` 3-utterance trials per model = `484,000` total trial evaluations).

- **Hugging Face Collection (44 Pretrained Models)**: **[`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595)**
- Full technical report, ablations & mathematical analysis: **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)**
- Complete verified evaluation metrics JSON (`44` models + `12` TFLite exports): **[`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json)**
- Local Pretrained HuggingFace + SafeTensors + TFLite checkpoints: **[`pretrained_models/`](pretrained_models/)**

### 1. Original Published Paper Reference Results (Internal Proprietary Datasets)

| Paper & Venue | Architecture & Loss / Scoring | Original Training Corpus | Original Evaluation Benchmark | Reported Metric in Original Paper |
| :--- | :--- | :--- | :--- | :--- |
| **GE2E** ([Wan et al., ICASSP 2018](https://arxiv.org/pdf/1710.10467)) | 3-layer LSTM (`768` hidden, `256` proj) + `GE2E-Softmax` | Internal Proprietary Vendor + Voice Search (~`18K` spk) | Internal TD-SV ("OK Google") & TI-SV | `2.18%` EER (TD-SV) / `7.61%` EER (TI-SV) |
| **Dr-Vectors** ([Pelecanos et al., Interspeech 2021](https://arxiv.org/pdf/2104.01989)) | LSTM + `Extended-Set Softmax` + `Dr-Vectors` Residual Scorer | Internal Proprietary Multi-Condition Corpus | Internal Multi-Condition Speaker Verification | `~11%` relative EER reduction over Cosine + GE2E |
| **Conformer + ATP** ([Wang et al., Odyssey 2022](https://arxiv.org/pdf/2202.12163)) | Conformer Encoder + Multi-Head `Attentive Temporal Pooling (ATP)` | Internal Multi-Language Speech Corpus (`65` languages) | Internal Streaming & Long-Form Identification | Significant EER & latency improvement over Mean/Last pooling |
| **PFAS** ([Pelecanos et al., Odyssey 2022](https://arxiv.org/pdf/2203.05642v3)) | Conformer + Multi-Head `(K, V)` `Parameter-Free Attentive Scoring` | Internal Proprietary Multi-Vendor Corpus | Internal 1-Utt & Multi-Utt Speaker Verification | `~8–14%` relative EER reduction over Cosine baseline |

### 2. Universal Multi-Domain Pretrained Models (`19` `univ_*` Models Across All 3 Domains)

All 19 `univ_*` models below are trained simultaneously on **LibriSpeech (`232` spk) + CN-Celeb2 (`119` spk) + VoxCeleb1 (`30` spk)** (`381` speakers, `9,447` utterances) with Domain-Balanced Stratified Batching (`75%` within-domain, `25%` cross-domain) and evaluated on all three speaker-disjoint evaluation sets (verified from [`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json)):

| Hugging Face Repository | Backbone / Size | Pooling / Loss / Scoring | Params | LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | Macro-Avg 1-Utt / 3-Utt EER (%) | Pooled 1-Utt / 3-Utt EER (%) | Macro 1-Utt / 3-Utt minDCF@0.01 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/univ_conformer_tiny_ge2e`](https://huggingface.co/wq2012/univ_conformer_tiny_ge2e) | `conformer` (`tiny`) | `atp` / `ge2e_softmax` / `cosine` | `693,126` | 13.23% / **8.80%** | 22.00% / 19.20% | 22.55% / 17.60% | **19.26%** / **15.20%** | **18.75%** / **15.03%** | `0.9211` / `0.8020` |
| [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | **12.50%** / **8.80%** | 22.33% / 20.00% | 24.80% / 19.70% | 19.88% / 16.17% | 19.40% / 16.03% | `0.9359` / `0.8433` |
| [`wq2012/univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors) | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 15.20% / 9.90% | 22.33% / 18.20% | 22.75% / **17.50%** | 20.09% / **15.20%** | 20.10% / 15.40% | `0.9348` / **`0.7500`** |
| [`wq2012/univ_conformer_small_ext_softmax`](https://huggingface.co/wq2012/univ_conformer_small_ext_softmax) | `conformer` (`small`) | `atp` / `extended_set_softmax` / `cosine` | `2,029,894` | 14.83% / 9.80% | 22.40% / 18.80% | 23.30% / 19.90% | 20.18% / 16.17% | 20.50% / 16.17% | `0.9635` / `0.8973` |
| [`wq2012/univ_conformer_base_ge2e`](https://huggingface.co/wq2012/univ_conformer_base_ge2e) | `conformer` (`base`) | `atp` / `ge2e_softmax` / `cosine` | `4,551,558` | 15.37% / 11.10% | 22.40% / 18.50% | 22.85% / 19.90% | 20.21% / 16.50% | 19.85% / 16.80% | `0.9763` / `0.8347` |
| [`wq2012/univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e) | `conformer` (`small`) | `pfas` / `ge2e_softmax` / `pfas` | `2,162,187` | 15.20% / 11.00% | **21.93%** / **17.40%** | 23.65% / 19.80% | 20.26% / 16.07% | 19.86% / 16.27% | `0.9741` / `0.8727` |
| [`wq2012/univ_mamba_small_ge2e`](https://huggingface.co/wq2012/univ_mamba_small_ge2e) | `mamba` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,028,674` | 14.20% / 9.50% | 23.20% / 18.50% | 23.75% / 17.70% | 20.38% / 15.23% | 20.41% / 16.07% | `0.9571` / `0.7920` |
| [`wq2012/univ_transformer_small_ge2e`](https://huggingface.co/wq2012/univ_transformer_small_ge2e) | `transformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `1,239,622` | 15.17% / 10.70% | 22.67% / 19.00% | 23.95% / 18.00% | 20.59% / 15.90% | 20.00% / 16.33% | `0.9694` / `0.8427` |
| [`wq2012/univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas) | `conformer` (`small`) | `pfas` / `extended_set_softmax` / `pfas` | `2,162,187` | 15.83% / 10.80% | 23.17% / 18.60% | 22.95% / 18.00% | 20.65% / 15.80% | 20.35% / 16.00% | **`0.8967`** / `0.8060` |
| [`wq2012/univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast) | `conformer` (`small`) | `atp` / `ge2e_contrast` / `cosine` | `2,029,894` | 14.73% / **8.30%** | 23.27% / **17.10%** | 24.45% / 19.40% | 20.82% / **14.93%** | 20.69% / 15.40% | `0.9461` / `0.8553` |
| [`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e) | `lstm` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,710,914` | 14.97% / 11.20% | 25.53% / 21.60% | 22.50% / 19.50% | 21.00% / 17.43% | 21.31% / 18.00% | `0.9620` / `0.9173` |
| [`wq2012/univ_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_base_ge2e) | `ecapa_tdnn` (`base`) | `asp` / `ge2e_softmax` / `cosine` | `3,190,626` | 18.63% / 12.20% | 25.97% / 21.20% | **22.20%** / 17.30% | 22.27% / 16.90% | 22.81% / 18.20% | `0.9760` / `0.9340` |
| [`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e) | `ecapa_tdnn` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,684,418` | 19.00% / 11.80% | 25.13% / 22.40% | 23.30% / 18.80% | 22.48% / 17.67% | 23.16% / 18.00% | `0.9889` / `0.9120` |
| [`wq2012/univ_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_tiny_ge2e) | `ecapa_tdnn` (`tiny`) | `asp` / `ge2e_softmax` / `cosine` | `626,722` | 19.07% / 12.50% | 25.27% / 21.40% | 23.35% / 17.20% | 22.56% / 17.03% | 23.02% / 18.00% | `0.9613` / `0.8933` |
| [`wq2012/univ_conformer_small_triplet`](https://huggingface.co/wq2012/univ_conformer_small_triplet) | `conformer` (`small`) | `atp` / `triplet` / `cosine` | `2,029,892` | 18.50% / 14.40% | 26.67% / 24.00% | 25.40% / 22.00% | 23.52% / 20.13% | 24.07% / 20.33% | `0.9477` / `0.9247` |
| [`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface) | `conformer` (`small`) | `atp` / `arcface` / `cosine` | `2,029,892` | 20.57% / 17.40% | 25.27% / 21.40% | 25.00% / 21.80% | 23.61% / 20.20% | 25.11% / 21.67% | `0.9869` / `0.9160` |
| [`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface) | `conformer` (`small`) | `atp` / `cosface` / `cosine` | `2,029,892` | 20.37% / 17.10% | 25.73% / 22.20% | 26.40% / 20.30% | 24.17% / 19.87% | 25.60% / 21.83% | `0.9914` / `0.9193` |
| [`wq2012/univ_conformer_small_sphereface`](https://huggingface.co/wq2012/univ_conformer_small_sphereface) | `conformer` (`small`) | `atp` / `sphereface` / `cosine` | `2,029,892` | 22.27% / 18.10% | 25.67% / 23.40% | 25.15% / 20.40% | 24.36% / 20.63% | 25.84% / 22.07% | `0.9858` / `0.9347` |
| [`wq2012/univ_resnet_small_ge2e`](https://huggingface.co/wq2012/univ_resnet_small_ge2e) | `resnet` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `2,239,922` | 24.53% / 21.40% | 29.33% / 26.40% | 28.25% / 25.80% | 27.37% / 24.53% | 27.89% / 25.47% | `0.9987` / `0.9847` |

### 3. Why Universal Multi-Domain Training Outperforms Single-Domain Models

| Model Paradigm | Representative Checkpoint | LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | Universal Macro-Avg 1-Utt / 3-Utt EER (%) | Universal Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech-Only Specialist** | [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) | **9.67%** / **5.50%** | 28.50% / 26.10% | 22.00% / 17.50% | 20.06% / 16.37% | 20.12% / 16.73% |
| **CN-Celeb2-Only Specialist** | [`wq2012/cn_conformer_small_ge2e`](https://huggingface.co/wq2012/cn_conformer_small_ge2e) | 20.80% / 16.00% | **21.33%** / **16.80%** | 27.20% / 23.90% | 23.11% / 18.90% | 23.93% / 20.13% |
| **VoxCeleb1-Only Specialist** | [`wq2012/vox_conformer_small_ge2e`](https://huggingface.co/wq2012/vox_conformer_small_ge2e) | 16.33% / 11.70% | 27.87% / 25.80% | **20.80%** / **16.40%** | 21.67% / 17.97% | 22.40% / 17.63% |
| **Universal Multi-Domain (`small`)** | [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) | 12.50% / 8.80% | 22.33% / 20.00% | 24.80% / 19.70% | **19.88%** / 16.17% | **19.40%** / 16.03% |
| **Universal Multi-Domain (`tiny`)** | [`wq2012/univ_conformer_tiny_ge2e`](https://huggingface.co/wq2012/univ_conformer_tiny_ge2e) | 13.23% / 8.80% | 22.00% / 19.20% | 22.55% / 17.60% | **19.26%** / **15.20%** | **18.75%** / **15.03%** |

### 4. Single-Domain Pretrained Baseline Models (`25` `ls_*`, `cn_*`, `vox_*` Models)

| Hugging Face Repository | Train Domain | Backbone / Size | Pooling / Loss / Scoring | Params | LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | Macro-Avg 1-Utt / 3-Utt EER (%) | Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) | `librispeech` | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | **9.67%** / **5.50%** | 28.50% / 26.10% | 22.00% / 17.50% | **20.06%** / 16.37% | **20.12%** / 16.73% |
| [`wq2012/ls_conformer_small_ext_softmax`](https://huggingface.co/wq2012/ls_conformer_small_ext_softmax) | `librispeech` | `conformer` (`small`) | `atp` / `extended_set_softmax` / `cosine` | `2,029,894` | 10.03% / 6.20% | 29.87% / 25.50% | 24.10% / 20.20% | 21.33% / 17.30% | 21.05% / 17.53% |
| [`wq2012/ls_conformer_small_pfas`](https://huggingface.co/wq2012/ls_conformer_small_pfas) | `librispeech` | `conformer` (`small`) | `pfas` / `extended_set_softmax` / `pfas` | `2,162,187` | 10.50% / 6.20% | 28.83% / 24.90% | 22.90% / 18.70% | 20.74% / 16.60% | 20.88% / 16.80% |
| [`wq2012/ls_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_pfas_ge2e) | `librispeech` | `conformer` (`small`) | `pfas` / `ge2e_softmax` / `pfas` | `2,162,187` | 10.50% / 6.20% | 28.87% / 25.00% | 23.40% / 20.20% | 20.92% / 17.13% | 20.69% / 17.17% |
| [`wq2012/ls_conformer_tiny_ge2e`](https://huggingface.co/wq2012/ls_conformer_tiny_ge2e) | `librispeech` | `conformer` (`tiny`) | `atp` / `ge2e_softmax` / `cosine` | `693,126` | 10.53% / 6.20% | 30.00% / 25.80% | 25.70% / 21.60% | 22.08% / 17.87% | 21.91% / 18.07% |
| [`wq2012/ls_transformer_small_ge2e`](https://huggingface.co/wq2012/ls_transformer_small_ge2e) | `librispeech` | `transformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `1,239,622` | 10.60% / 6.20% | 28.10% / 23.30% | 22.30% / 16.90% | 20.33% / **15.47%** | 20.80% / **16.37%** |
| [`wq2012/ls_conformer_base_ge2e`](https://huggingface.co/wq2012/ls_conformer_base_ge2e) | `librispeech` | `conformer` (`base`) | `atp` / `ge2e_softmax` / `cosine` | `4,551,558` | 10.77% / 6.60% | 28.87% / 23.90% | 24.65% / 19.90% | 21.43% / 16.80% | 21.26% / 17.00% |
| [`wq2012/ls_conformer_small_dr_vectors`](https://huggingface.co/wq2012/ls_conformer_small_dr_vectors) | `librispeech` | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 11.17% / 6.20% | 28.80% / 24.00% | 22.30% / 19.60% | 20.76% / 16.60% | 20.69% / 16.60% |
| [`wq2012/ls_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/ls_conformer_small_ge2e_contrast) | `librispeech` | `conformer` (`small`) | `atp` / `ge2e_contrast` / `cosine` | `2,029,894` | 11.43% / 7.10% | 30.87% / 25.40% | 24.45% / 20.10% | 22.25% / 17.53% | 22.27% / 17.63% |
| [`wq2012/ls_mamba_small_ge2e`](https://huggingface.co/wq2012/ls_mamba_small_ge2e) | `librispeech` | `mamba` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,028,674` | 12.17% / 7.90% | 29.13% / 23.30% | 21.05% / 16.10% | 20.78% / 15.77% | 21.30% / 17.10% |
| [`wq2012/ls_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_tiny_ge2e) | `librispeech` | `ecapa_tdnn` (`tiny`) | `asp` / `ge2e_softmax` / `cosine` | `626,722` | 12.60% / 8.40% | 30.87% / 26.40% | 21.70% / 17.30% | 21.72% / 17.37% | 22.10% / 17.67% |
| [`wq2012/ls_lstm_small_ge2e`](https://huggingface.co/wq2012/ls_lstm_small_ge2e) | `librispeech` | `lstm` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,710,914` | 12.77% / 7.90% | 28.03% / 24.90% | 23.35% / 19.70% | 21.38% / 17.50% | 21.25% / 17.73% |
| [`wq2012/ls_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_small_ge2e) | `librispeech` | `ecapa_tdnn` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,684,418` | 13.83% / 9.10% | 31.70% / 28.40% | **19.65%** / **15.90%** | 21.73% / 17.80% | 22.48% / 17.83% |
| [`wq2012/ls_conformer_small_cosface`](https://huggingface.co/wq2012/ls_conformer_small_cosface) | `librispeech` | `conformer` (`small`) | `atp` / `cosface` / `cosine` | `2,029,892` | 14.13% / 10.80% | 30.53% / 25.00% | 23.00% / 17.40% | 22.56% / 17.73% | 23.26% / 18.17% |
| [`wq2012/ls_conformer_small_arcface`](https://huggingface.co/wq2012/ls_conformer_small_arcface) | `librispeech` | `conformer` (`small`) | `atp` / `arcface` / `cosine` | `2,029,892` | 14.47% / 9.60% | 28.73% / 25.50% | 23.85% / 18.30% | 22.35% / 17.80% | 22.80% / 18.23% |
| [`wq2012/ls_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_base_ge2e) | `librispeech` | `ecapa_tdnn` (`base`) | `asp` / `ge2e_softmax` / `cosine` | `3,190,626` | 14.47% / 10.60% | 33.33% / 30.20% | 19.90% / 17.00% | 22.57% / 19.27% | 23.29% / 18.90% |
| [`wq2012/ls_conformer_small_sphereface`](https://huggingface.co/wq2012/ls_conformer_small_sphereface) | `librispeech` | `conformer` (`small`) | `atp` / `sphereface` / `cosine` | `2,029,892` | 14.73% / 11.00% | 29.93% / 23.10% | 22.20% / 16.30% | 22.29% / 16.80% | 23.29% / 17.97% |
| [`wq2012/ls_resnet_small_ge2e`](https://huggingface.co/wq2012/ls_resnet_small_ge2e) | `librispeech` | `resnet` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `2,239,922` | 15.20% / 11.30% | 32.17% / 29.70% | 21.40% / 20.20% | 22.92% / 20.40% | 23.94% / 20.47% |
| [`wq2012/ls_conformer_small_triplet`](https://huggingface.co/wq2012/ls_conformer_small_triplet) | `librispeech` | `conformer` (`small`) | `atp` / `triplet` / `cosine` | `2,029,892` | 16.07% / 13.50% | 28.93% / 25.20% | 22.00% / 20.20% | 22.33% / 19.63% | 23.41% / 19.80% |
| [`wq2012/cn_conformer_small_dr_vectors`](https://huggingface.co/wq2012/cn_conformer_small_dr_vectors) | `cnceleb2` | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 23.00% / 16.20% | **21.17%** / **16.70%** | 26.30% / 20.40% | 23.49% / 17.77% | 23.95% / 18.97% |
| [`wq2012/cn_conformer_small_ge2e`](https://huggingface.co/wq2012/cn_conformer_small_ge2e) | `cnceleb2` | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 20.80% / 16.00% | 21.33% / 16.80% | 27.20% / 23.90% | 23.11% / 18.90% | 23.93% / 20.13% |
| [`wq2012/cn_conformer_small_pfas`](https://huggingface.co/wq2012/cn_conformer_small_pfas) | `cnceleb2` | `conformer` (`small`) | `pfas` / `extended_set_softmax` / `pfas` | `2,162,187` | 24.47% / 18.90% | 22.13% / 19.80% | 28.15% / 24.80% | 24.92% / 21.17% | 26.20% / 23.20% |
| [`wq2012/cn_ecapa_tdnn_small_arcface`](https://huggingface.co/wq2012/cn_ecapa_tdnn_small_arcface) | `cnceleb2` | `ecapa_tdnn` (`small`) | `asp` / `arcface` / `cosine` | `1,684,416` | 30.53% / 23.40% | 28.07% / 23.80% | 31.40% / 31.20% | 30.00% / 26.13% | 30.23% / 26.47% |
| [`wq2012/vox_conformer_small_ge2e`](https://huggingface.co/wq2012/vox_conformer_small_ge2e) | `voxceleb1` | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 16.33% / 11.70% | 27.87% / 25.80% | **20.80%** / **16.40%** | 21.67% / 17.97% | 22.40% / 17.63% |
| [`wq2012/vox_ecapa_tdnn_small_ext_softmax`](https://huggingface.co/wq2012/vox_ecapa_tdnn_small_ext_softmax) | `voxceleb1` | `ecapa_tdnn` (`small`) | `asp` / `extended_set_softmax` / `cosine` | `1,684,418` | 20.07% / 14.60% | 32.40% / 30.60% | 21.10% / 17.80% | 24.52% / 21.00% | 24.57% / 20.97% |

---

## Testing, Linting & Documentation

Run the unit test suite (`42` tests covering all backbones, exact padding mask invariance including `PFAS` `maxpool`, losses including non-saturating `ge2e_contrast` and pure/blended `extended_set_softmax`, pooling heads including `ATP` and `PFAS`, VAD + AGC, mutually exclusive `SpecAugment`/`MTR`, `Dr-Vectors` 1-term and 3-term scoring, dataset loaders, HF compatibility & Hub resolution, TFLite export, and legacy config compatibility):

```bash
pytest -v tests.py
flake8 .
./run_pdoc.sh
```
