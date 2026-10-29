# FlaxSpeaker

[![Python application](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml/badge.svg)](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml) [![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-48%20Pretrained%20Models-blue)](https://huggingface.co/collections/wq2012/flaxspeaker) [![PyPI Version](https://img.shields.io/pypi/v/flaxspeaker.svg)](https://pypi.python.org/pypi/flaxspeaker) [![Python Versions](https://img.shields.io/pypi/pyversions/flaxspeaker.svg)](https://pypi.org/project/flaxspeaker) [![Downloads](https://static.pepy.tech/badge/flaxspeaker)](https://pepy.tech/project/flaxspeaker) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**FlaxSpeaker** (`v0.5.0`) is an end-to-end open-source speaker verification library built on [JAX](https://jax.readthedocs.io) and [Flax](https://flax.readthedocs.io). Designed for education and academic research, it unifies dataset preparation, real-noise waveform augmentation (`MUSAN` + `RIRS_NOISES` + `3x` speed perturbation) and zero-copy `float16` memory-mapped feature shards, frame-level energy Voice Activity Detection (VAD) with peak Automatic Gain Control (AGC), an 80-D log-Mel frontend with CMVN, six neural sequence backbones (`tiny`, `small`, `base`, `large`), eight temporal pooling heads, ten metric-learning and angular-margin training objectives (including `Sub-Center ArcFace K=3`), three scoring paradigms (`Cosine`, `PFAS`, `Dr-Vectors`) with Top-300 `AS-Norm` and Quality-Aware Score Calibration (`QMF`), universal multi-domain stratified training, and one-command export to **TFLite (FP32 / dynamic-range INT8)**, **TensorFlow SavedModel**, **SafeTensors**, and the **Hugging Face Hub**.

> **Two-Tier `v0.5.0` Model Zoo (`48` Pretrained Checkpoints = `44` Controlled Ablation Models + `4` Full-Data Universal Flagships).**
> 1. **Tier-2 Full-Data Universal Flagships (`4` `univ_full_*` models, `6.98M–14.92M` parameters):** Trained on **`12,193` physical speakers** (`36,579` speed-perturbed classes, `464,557` clean utterances / `1,857,149` augmented items across `VoxCeleb2-dev + VoxCeleb1-dev + CN-Celeb1/2 + LibriSpeech-960h` with `MUSAN` noise/music/babble and `RIRS_NOISES` room impulse responses) and evaluated on the full official trial lists (**VoxCeleb1-O** `37,611` trials, **CN-Celeb-E** `3,484,292` trials, and **LibriSpeech `test-clean`** `10,000` trials), reaching **`13.92%` VoxCeleb1-O QMF EER** (`univ_full_conformer_large_dr_vectors`), **`17.98%` CN-Celeb-E Multi-Enroll EER** (`3`-way fusion; `18.24%` single-model), and **`5.20%` LibriSpeech `test-clean` 3-Utt EER** (`5.43%` single-model).
> 2. **Tier-1 Controlled Benchmark Zoo (`44` `univ_*`, `ls_*`, `cn_*`, `vox_*` models, `0.63M–4.55M` parameters):** Trained to calibrated `S_safe = X + 2` epoch step budgets (`4,600` steps / `15.58` epochs on Universal `381` spk / `9,447` utts; `2,400–2,600` steps on LibriSpeech `232` spk; `1,600–1,800` steps on CN-Celeb2 `119` spk; `1,000` steps on VoxCeleb1 `30` spk) with uniform variable-length masked crops `T ~ U(90, 180)` (`48 / 48` released checkpoints verified to `5e-5` step-0 loss tolerance as `variable` in `pretrained_models/step0_fingerprints.json`), `Sub-Center ArcFace (K=3)`, and `BatchNorm` after activation in `ECAPA-TDNN`. The 44 Tier-1 models reach **`13.94–28.91%`** pooled 1-utt EER (`13.94–23.23%` across the 19 `univ_*` models; `9.67–19.57%` pooled 3-utt EER), with universal multi-domain training outperforming the matched LibriSpeech specialist in all **`19 / 19` pairs**. Multi-seed stability across seeds `{0, 1, 2}` has a pooled 1-utt EER standard deviation of **`0.05%`** (`univ_conformer_small_ge2e`), **`0.29%`** (`univ_ecapa_tdnn_small_ge2e`), and **`0.41%`** (`ls_conformer_small_ge2e`; see [RESEARCH_REPORT.md §10](RESEARCH_REPORT.md)).

For the companion PyTorch educational project, see [SpeakerRecognitionFromScratch](https://github.com/wq2012/SpeakerRecognitionFromScratch).

---

## Disclaimer & Open-Source Reproduction Notice

**This is not an officially supported Google product.**

This repository provides a from-scratch re-implementation in **JAX / Flax**, written using only the published papers (no proprietary code or data), of the mathematical formulations and neural architectures described in the following research papers:
1. **Generalized End-to-End Loss for Speaker Verification** — *Li Wan, Quan Wang, Alan Papir, Ignacio Lopez Moreno* ([ICASSP 2018, arXiv:1710.10467](https://arxiv.org/pdf/1710.10467))
2. **Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition** — *Jason Pelecanos, Quan Wang, Ignacio Lopez Moreno* ([Interspeech 2021, arXiv:2104.01989](https://arxiv.org/pdf/2104.01989))
3. **Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech** — *Quan Wang, Yang Yu, Jason Pelecanos, Yiling Huang, Ignacio Lopez Moreno* ([Odyssey 2022, arXiv:2202.12163](https://arxiv.org/pdf/2202.12163)) — a *language identification* paper whose attentive temporal pooling head is repurposed here for speaker embeddings
4. **Parameter-Free Attentive Scoring for Speaker Verification** — *Jason Pelecanos, Quan Wang, Yiling Huang, Ignacio Lopez Moreno* ([Odyssey 2022, arXiv:2203.05642](https://arxiv.org/pdf/2203.05642))

Because the original papers were trained and evaluated on proprietary internal datasets and infrastructure (for `~500k` steps on thousands of speakers), **exact numerical parity with the original paper tables is neither expected nor claimed**. All pretrained checkpoints and benchmark tables in this repository are trained and evaluated from scratch on publicly available speech corpora (**VoxCeleb2**, **VoxCeleb1**, **CN-Celeb1/2**, and **LibriSpeech**) with speaker-disjoint splits.

---

## System Architecture

![FlaxSpeaker System Architecture](resources/architecture_diagram.png)

### 1. Six Neural Sequence Backbones & Four Size Presets (`0.63 M–14.92 M` parameters across `tiny`, `small`, `base`, `large`)
All backbones share a unified `(B, T, F) -> (B, T, D)` sequence interface with **exact variable-length padding mask invariance** via `ModernSpeakerEncoder` ([`flaxspeaker/backbones.py`](flaxspeaker/backbones.py)) and scale automatically via `size_variant: tiny | small | base | large`:
- **Conformer** (`conformer`): Parallel Linear + 1D Conv Stem (`Dense(D) + SiLU(LayerNorm(Conv1D(k=3)))`) + Macaron FFN + Masked Multi-Head Self-Attention + Mask-Aware GroupNorm Depthwise Separable Convolution + Multi-Layer Feature Aggregation (MFA), with `large` preset (`dim=384, num_layers=6, num_heads=6, ffn_dim=1536, subsample_factor=4`, `6.98M–7.31M` params).
- **ECAPA-TDNN** (`ecapa_tdnn`): 1D SE-Res2Net dilated multi-scale blocks with Multi-Layer Feature Aggregation (MFA), mask-aware `BatchNorm` running statistics after activation (`use_batch_norm: true, norm_after_activation: true` default in `v0.5.0`), and `large` preset (`channels=1024, scale=8, mfa_dim=1536, se_bottleneck_dim=128`, `14.92M` params).
- **Mamba** (`mamba`): Selective State Space Model (SSM) implemented in pure JAX via parallel `jax.lax.associative_scan` (`O(T)` linear time complexity, supporting both causal unidirectional and mask-aware bidirectional `bidirectional: true` modes).
- **ResNet** (`resnet`): 2D Time-Frequency Residual Network with Squeeze-and-Excitation (SE-ResNet-18, 8 residual blocks across 4 stages `(2, 2, 2, 2)`).
- **Transformer** (`transformer`): Parallel Linear + 1D Conv Stem (`Dense(D) + SiLU(LayerNorm(Conv1D(k=3)))`) + Pre-LayerNorm Multi-Head Self-Attention encoder with Multi-Layer Feature Aggregation (MFA).
- **LSTM** (`lstm`): Deep stacked uni- or bidirectional LSTM (`bidirectional: true`, mask-aware with `seq_lengths` in both directions) with configurable hidden size, optional per-layer projection, and per-layer LayerNorm + residual connections.

### 2. Eight Temporal Pooling Heads ([`flaxspeaker/pooling.py`](flaxspeaker/pooling.py))
- **Attentive Temporal Pooling** (`atp`): Sigmoid frame-gated statistics pooling (`concat(mu_h, sigma_h)`) extended from the single-head linear-sigmoid ATP of [Wang et al., Odyssey 2022, `arXiv:2202.12163`](https://arxiv.org/pdf/2202.12163) (a language-identification paper). FlaxSpeaker defaults add a tanh bottleneck (`attention_dim: 128`) and 4 channel-subspace heads (`subspace_heads: true`); `attention_dim: 0, num_heads: 1` recovers the paper form. Followed by a nonlinear bottleneck FFN (`LayerNorm -> Dense(256) -> LayerNorm -> ReLU -> Dense(d_e) -> L2_norm`).
- **Self-Attentive Pooling** (`sap`) & **Attentive Statistics Pooling** (`asp`): Softmax attention-weighted frame mean and standard deviation.
- **Mean Pooling** (`mean`), **Last-Frame Pooling** (`last`), & **Temporal Statistics Pooling** (`stats`).
- **Cumulative Statistics Pooling** (`cumulative_stats`): Frame-wise causal running statistics for streaming inference.
- **Parameter-Free Attentive Scoring Head** (`pfas`): Supports `attention_type: atp_ffn` (the head of [Pelecanos et al., Odyssey 2022, `arXiv:2203.05642`](https://arxiv.org/pdf/2203.05642): ATP → affine + ReLU → linear → keys/values; `univ_full_conformer_large_pfas` uses the 16-config sweep winner `pfas_paper_atpffn_k16_d32_v64_ln` with `K=16` tied keys × `(d_k=32 + d_v=64)` = `1,536-D`, while `odyssey2022_pfas_config(pure_paper_mode=True)` uses 32 tied keys × `(16 + 48)` = `2,048-D`), as well as `attention_type: shared_nonlinear | unshared_nonlinear | dot | maxpool` + `use_global_anchor: true` (**PFAS-SN**, used by the Tier-1 `*_pfas*` checkpoints: 4 keys × `(32 + 128)` = `640-D`). All variants apply per-key L2 normalisation and global L2 normalisation of the concatenated value form (paper Eqs. 5–7).

### 3. Ten Speaker Recognition Losses & Three Scoring Backends ([`flaxspeaker/losses.py`](flaxspeaker/losses.py), [`flaxspeaker/scoring.py`](flaxspeaker/scoring.py))
- **Generalized End-to-End (GE2E) Loss**: `ge2e_softmax` (Eq. 6, exact leave-one-out centroids) and `ge2e_contrast` (Eq. 7) with learnable scale `w` and bias `b` ([Wan et al., ICASSP 2018](https://arxiv.org/pdf/1710.10467)). By default `ge2e_contrast` back-propagates Eq. 7 exactly; the released `*_ge2e_contrast` checkpoints use the **hybrid contrast surrogate** (`ge2e_contrast_surrogate: true`: forward value = Eq. 7, gradient = `0.25·Eq7 + 0.25·BCE + 0.50·GE2E-softmax-CE`, `ge2e_init_b: -1.0`, `ge2e_contrast_b_grad_scale: 0.5`).
- **Extended-Set Softmax Loss**: `extended_set_softmax`, supporting both the paper's pure form (`pure_extended_set: true`, Eq. 2 of [Pelecanos et al., Interspeech 2021](https://arxiv.org/pdf/2104.01989)) and the **Blended Extended-Set Softmax (BXS, FlaxSpeaker variant)** (`pure_extended_set: false`: 50/50 blend of row-softmax and extended-set softmax with a `-log(N)` logit shift).
- **Decision Residual Networks (Dr-Vectors)**: `scoring_type: dr_vector` — single-stage end-to-end residual feed-forward scorer on `[e_t, e_e, c]`. `interspeech2021_dr_vectors_config(pure_paper_mode=True)` builds the paper's network (`256-D` embedding split into `cosine_dims: 200` base + `56` side dims, `3 × Dense(256)` leaky-ReLU(0.2) without normalisation, `readout: weighted_sum`, `trainable_affine: true`, pure extended-set softmax, reaching `18.30%` pooled 1-utt EER at 4,600 steps in `ablation_results.json`). The released `*_dr_vectors` checkpoints use the compact **down-scaled Dr-Vector variant** (`192-D` or `256-D` cosine, hidden `(128, 64)` ReLU + LayerNorm, linear read-out, `residual_weight: 0.25`, blended extended-set softmax).
- **Parameter-Free Attentive Scoring (PFAS)**: `scoring_type: pfas` — cross-attention between test and enrollment `(K, V)` representations with per-trial softmax key–key attention weights `A_{ij}`, per-key and global L2 normalisation (Eqs. 5–7 of [Pelecanos et al., Odyssey 2022](https://arxiv.org/pdf/2203.05642)), trainable scale, and key stacking (`N = M × E`) for multi-utterance enrollment.
- **Angular Margin Classification Losses (with Sub-Center $K=3$ & Linear Margin Warm-Up)**:
  - **Sub-Center ArcFace** (`subcenter_arcface` / `arcface` with `num_subcenters: 3`): takes $\max_{k \in \{1..K\}} \cos(\theta_{i,j,k})$ per speaker class to absorb noisy labels before applying $\cos(\theta + m_t)$ with linear margin ramp `m_t = m * min(1, t / margin_warmup_steps)` (`margin_warmup_steps = 500` on Universal, `300` on LibriSpeech, `250` on CN-Celeb2).
  - **CosFace / AM-Softmax** (`cosface`): `cos(theta) - m_t` with linear margin ramp `margin_warmup_steps`.
  - **SphereFace / A-Softmax** (`sphereface`): `(-1)^k * cos(m * theta) - 2k` with $\lambda$-annealing (`1000 -> 5`).
  - **Standard Softmax** (`softmax`), **Pairwise Loss** (`pairwise`), and **Triplet Loss** (`triplet`).
- **Top-300 Adaptive Symmetric Score Normalization (`AS-Norm`) & Quality-Aware Score Calibration (`QMF`)**: Top-`K` (`K=300`) impostor-cohort z-normalisation (`use_as_norm: true`) with enforced trial-cohort disjointness (evaluated alongside raw scores), plus logistic Quality-Aware Score Calibration (`qmf.py`) combining raw/AS-Norm score, log duration frames, post-VAD speech ratio, and pre-L2 embedding norm.

### 4. End-to-End Audio Pipeline, Real Waveform Augmentation, `float16` Shards & Model Export
- **Frame-level energy VAD & AGC** ([`flaxspeaker/vad.py`](flaxspeaker/vad.py)): Frame energies are computed on the peak-normalised waveform (`apply_agc = True`, target peak `1.0`) and the resulting mask is applied to the log-Mel frames (`max(-42.0 dB, P_95(E) - 26.0 dB, P_10(E) + 4.5 dB)`, hangover smoothing `min_speech_frames = 3`, `hangover_frames = 8`, top-8-frame fallback).
- **Acoustic Frontends, Full-Utterance Inference & `float16` Memory-Mapped Shards** ([`flaxspeaker/feature_extraction.py`](flaxspeaker/feature_extraction.py), [`flaxspeaker/sharded_cache.py`](flaxspeaker/sharded_cache.py)): `log_mel` (natural-log Mel + energy VAD + CMVN), **per-utterance variable sequence length sampling `T ~ U(90, 180)` frames across all 48 `v0.5.0` checkpoints** (`variable_length_training: true`) with exact binary padding masks `(B, T)`, bucketed full-utterance inference (`max_eval_seq_len = 0`, capped at `3,000` frames / `30 s`), and zero-copy `np.memmap` `float16` feature shards (`ShardedFeatureDataset`).
- **Real Waveform Augmentation & Feature-Domain Regularization** ([`flaxspeaker/waveform_aug.py`](flaxspeaker/waveform_aug.py), [`flaxspeaker/specaug.py`](flaxspeaker/specaug.py)): Tier-2 full-data training applies `3x` polyphase speed perturbation (`0.9x`, `1.0x`, `1.1x` -> `3x` speaker classes) and online/offline `MUSAN` (`speech`, `music`, `noise`) + `RIRS_NOISES` room reverberation (`MusanRirAugmenter`), while Tier-1 controlled training applies mutually exclusive feature-domain noise (`35%`, `SNR ~ U(12, 20) dB`), `SpecAugment` (`32.5%`), and clean features (`32.5%`).
- **Model Export** ([`flaxspeaker/export.py`](flaxspeaker/export.py), [`flaxspeaker/hf_compat.py`](flaxspeaker/hf_compat.py)): Export via `jax2tf` to **TFLite (FP32, dynamic-range INT8)** and **TF SavedModel** with a fixed, unmasked `180`-frame (`1.8 s`) encoder window, and to **SafeTensors** / **Hugging Face Hub** (`config.json`, `model.safetensors`, `flax_model.msgpack`, `eval_results.json`, model card). Across all 16 exported `.tflite` files (`8` models × FP32/INT8), dynamic-range INT8 quantization reduces file size by **`3.44–3.88×`** (`6.82–59.84 MB` → `1.86–15.42 MB`) while changing 180-frame verification EER by at most **`0.00–0.14` percentage points** relative to FP32 TFLite, with mean utterance-level cosine similarity `cos(FP32, INT8)` between **`0.999972` and `0.999990`** and a **`1.85–1.93×` CPU speedup** on LSTM models (see [RESEARCH_REPORT.md §11](RESEARCH_REPORT.md)).

---

## Installation

```bash
pip install -U flaxspeaker   # >= 0.5.0 recommended for the 48-model v0.5.0 zoo
```

To install from source: `pip install git+https://github.com/wq2012/FlaxSpeaker`.

For development, HuggingFace export, and TFLite conversion extras:
```bash
pip install -e ".[dev,export,hf]"
```

---

## Quickstart & CLI Workflows

FlaxSpeaker provides standalone command-line scripts in [`scripts/`](scripts/) covering dataset preparation, shard generation, training, PFAS sweeps, evaluation, ablations, figure/table generation, and Hub deployment (in addition to `python -m flaxspeaker`):

### 1. Prepare Datasets & `float16` Feature Shards (`scripts/prepare_datasets.py` / `scripts/prepare_controlled_shards.py` / `scripts/prepare_full_corpora_shards.py`)
```bash
# Prepare controlled academic datasets + combined Universal Multi-Domain split
python scripts/prepare_datasets.py --data_root ./data/speaker_datasets
python scripts/prepare_controlled_shards.py --data_root ./data/speaker_datasets

# Or download and shard the full 12,193-speaker multi-corpus datasets (VoxCeleb1/2 + CN-Celeb1/2 + LibriSpeech-960h + MUSAN + RIRS)
python scripts/download_full_corpora.py
python scripts/prepare_full_corpora_shards.py
```

### 2. Train Models (`scripts/train.py` / `scripts/train_full_data_flagships.py`)
Train from any converged YAML config (`configs/*.yml`) or run the Tier-2 full-data universal flagship pipeline:

```bash
# Train the top Tier-1 Universal Conformer + ATP + CosFace model (4,600 steps, margin_warmup_steps=500)
python scripts/train.py \
  --config configs/univ_conformer_small_cosface.yml \
  --train_csv ./data/speaker_datasets/splits/universal/train.csv \
  --eval_csv ./data/speaker_datasets/splits/universal/eval.csv \
  --trials_file ./data/speaker_datasets/splits/universal/trials.tsv \
  --output_dir ./checkpoints/univ_conformer_small_cosface

# Run the 16-config PFAS hyperparameter sweep and Tier-2 full-data universal flagships
python scripts/run_pfas_hparam_sweep.py
python scripts/train_full_data_flagships.py
```

### 3. Evaluate 1-Utt, 3-Utt, and VoxCeleb1-O-subset Trials (`scripts/evaluate.py` / `scripts/eval_vox1_o_subset.py`)
```bash
python scripts/evaluate.py \
  --model_dir wq2012/univ_full_conformer_large_arcface \
  --trials_file ./data/speaker_datasets/splits/universal/trials.tsv \
  --multi_trials_file ./data/speaker_datasets/splits/universal/multi_enroll_trials.tsv \
  --output_json eval_summary.json

# Evaluate all 48 models on the VoxCeleb1-O 10-speaker subset (writes pretrained_models/vox1_o_10spk_subset_results.json)
python scripts/eval_vox1_o_subset.py --data_root ./data/speaker_datasets --models_dir ./pretrained_models
```

### 4. Run End-to-End Speaker Verification Inference (`scripts/inference.py`)
```bash
python scripts/inference.py \
  --model_dir wq2012/univ_full_conformer_large_arcface \
  --enroll_audio hf_space/examples/librispeech_spk1_utt1.flac \
  --test_audio hf_space/examples/librispeech_spk1_utt2.flac \
  --apply_vad --vad_top_db 26.0 --full_sequence
```

### 5. Export to TFLite & Publish to the Hugging Face Hub (`scripts/export_model.py` / `scripts/eval_tflite_models.py` / `scripts/update_hf_model_cards.py`)
```bash
python scripts/export_model.py \
  --model_dir wq2012/univ_full_conformer_large_arcface \
  --output_dir ./exported_models/univ_full_conformer_large_arcface \
  --export_tflite \
  --quantize_int8

# Evaluate exported FP32 and INT8 TFLite models
python scripts/eval_tflite_models.py --data_root ./data/speaker_datasets --models_dir ./pretrained_models

# Regenerate all 48 model cards and push weights + TFLite + metadata to Hugging Face Hub
python scripts/update_hf_model_cards.py --push --include_weights
```

### 6. Reproduce Pilot Calibration, Full 48-Model Benchmark Suite, Ablations, Provenance & Paper Figures
```bash
# Phase-1 5,000-step convergence pilot & S_safe = X + 2 epoch schedule calibration
python scripts/run_phase1_convergence_pilot.py --data_root ./data/speaker_datasets

# Run the complete 44-model Tier-1 controlled benchmark matrix across all domains
python scripts/run_research_benchmarks.py --data_root ./data/speaker_datasets

# Reproduce controlled ablation studies + pure_paper_mode=True 4,600-step runs (ablation_results.json)
python scripts/run_r1_ablations.py --data_root ./data/speaker_datasets

# Multi-seed stability (3 seeds x 3 models), own-schedule horizons (400..4,600), matched-data controls
python scripts/run_r2_controls.py --data_root ./data/speaker_datasets

# Step-0 loss fingerprint of all 48 released checkpoints & training_provenance blocks
python scripts/fingerprint_released_checkpoints.py --data_root ./data/speaker_datasets
python scripts/write_training_provenance.py

# Regenerate Markdown / LaTeX benchmark tables and publication figures
python scripts/generate_report_tables.py --section all > /tmp/report_tables.md
python scripts/generate_paper_tables.py --paper_dir ../flax_speaker_paper
python scripts/generate_figures.py \
  --benchmark-json pretrained_models/benchmark_results.json \
  --output-dir ./figures
```

---

## Python & HuggingFace API

Load any exported model directory or Hugging Face Hub repository ID ([`wq2012/<model_id>`](https://huggingface.co/collections/wq2012/flaxspeaker)) with `load_pretrained` or `FlaxSpeakerModel.from_pretrained`:

```python
import flaxspeaker
from flaxspeaker.configs import universal_multi_domain_config

# 1. Load a v0.5.0 full-data universal flagship or controlled checkpoint from Hugging Face Hub
verifier = flaxspeaker.load_pretrained(
    "wq2012/univ_full_conformer_large_arcface"
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
    size="large",
    loss_type="arcface",
    scoring_type="cosine",
)
```

---

## Interactive Gradio Demo (`hf_space/`)

An interactive [Gradio 5.50.0](https://gradio.app) web application is included in [`hf_space/`](hf_space/). **Hosting status**: creating a Gradio Space under `wq2012` on the Hugging Face Hub returns HTTP 402 (free CPU Spaces require a paid PRO subscription on that account), so run it locally (`python hf_space/app.py`) or deploy it to an account with Space creation enabled via `python scripts/deploy_hf_space.py --repo_id <your-namespace>/FlaxSpeaker`. Features:
- Real inference only (no precomputed outputs) across all **48 `v0.5.0` models** (`4` Tier-2 `univ_full_*` Full-Data Universal Flagships + `19` Tier-1 `univ_*` + `25` Tier-1 Single-Domain `ls_*`, `cn_*`, `vox_*` checkpoints) from [`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker) (`Cosine`, `PFAS`, and `Dr-Vectors` scoring).
- Automatic operating-point selection from each model's `eval_results.json` for 1-utterance vs. multi-utterance enrollment, configurable VAD thresholds (`--vad_snr_margin_db`, `--vad_top_db`), and steady-state JIT latency measurement.
- 1-utterance and 3-utterance enrollment with speech samples from **LibriSpeech**, **CN-Celeb2**, and **VoxCeleb1**, plus waveform + VAD gate, post-VAD log-Mel spectrogram, and embedding-profile visualizations.

```bash
python hf_space/app.py
```

---

## 48-Model Benchmark Summary & Pretrained Model Zoo (`v0.5.0`)

![FlaxSpeaker v0.5.0 Benchmark Summary](resources/benchmark_charts.png)

We trained and evaluated **48 `v0.5.0` models** across two tiers:
- **Tier-2 Full-Data Universal Flagships (`4` `univ_full_*` models):** Trained on `12,193` speakers (`36,579` speed-perturbed classes, `464,557` clean utterances / `1,857,149` augmented items across `VoxCeleb2-dev + VoxCeleb1-dev + CN-Celeb1/2 + LibriSpeech-960h` with `MUSAN` + `RIRS_NOISES`) and evaluated on the full official benchmarks (**VoxCeleb1-O** `37,611` trials, **CN-Celeb-E** `3,484,292` trials, **LibriSpeech `test-clean`** `10,000` trials) plus all controlled splits.
- **Tier-1 Controlled Benchmark Zoo (`44` models):** `19` Universal Multi-Domain models (`381` speakers / `9,447` utterances, `4,600` steps) + `25` Single-Domain specialist models (`2,400–2,600` steps on LibriSpeech, `1,600–1,800` steps on CN-Celeb2, `1,000` steps on VoxCeleb1; total training compute `47.51` CPU hours, `5.8–397.3` min/model, median `36.2` min).

Key artifacts:
- **Hugging Face Collection (48 Pretrained Models)**: **[`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker)**
- Full technical report, convergence calibration, ablations & mathematical analysis: **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)**
- Complete verified evaluation metrics JSON (`48` models + `16` TFLite exports): **[`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json)**
- Tier-2 Full-Data Flagships & Official Benchmarks (`VoxCeleb1-O`, `CN-Celeb-E`, `LibriSpeech test-clean` + 2-way/3-way fusion): **[`pretrained_models/flagship_official_results.json`](pretrained_models/flagship_official_results.json)**
- 16-Candidate Wide `PFAS` Hyperparameter Sweep JSON: **[`pretrained_models/pfas_sweep_results.json`](pretrained_models/pfas_sweep_results.json)**
- Phase-1 5,000-step convergence pilot & `S_safe` schedule calibration JSON: **[`pretrained_models/phase1_convergence_pilot.json`](pretrained_models/phase1_convergence_pilot.json)**
- Controlled ablation study & `pure_paper_mode=True` metrics JSON: **[`pretrained_models/ablation_results.json`](pretrained_models/ablation_results.json)**
- VoxCeleb1-O **10-speaker subset** evaluation JSON (`48` models, 26 dB VAD): **[`pretrained_models/vox1_o_10spk_subset_results.json`](pretrained_models/vox1_o_10spk_subset_results.json)**
- Multi-seed stability (`3` seeds × `3` models), own-schedule horizons (`400..4,600`), and matched-data controls (`2,400` & `4,600` steps): **[`pretrained_models/r2_seed_and_control_results.json`](pretrained_models/r2_seed_and_control_results.json)**
- Step-0 loss fingerprint of all 48 released checkpoints (`48 / 48` `variable`): **[`pretrained_models/step0_fingerprints.json`](pretrained_models/step0_fingerprints.json)**
- Local Pretrained HuggingFace + SafeTensors + TFLite checkpoints: **[`pretrained_models/`](pretrained_models/)**

### 1. Tier-2 Full-Data Universal Flagships (`12,193` Speakers) & Official Benchmarks (`VoxCeleb1-O`, `CN-Celeb-E`, `LibriSpeech test-clean`)

<!-- BEGIN README_FLAGSHIP_TABLE -->
| Hugging Face Repository | Backbone / Head / Objective | Params | Steps | VoxCeleb1-O Official (`37,611` trials) Raw / AS-Norm / QMF EER (%) | CN-Celeb-E Official (`3.48M` trials) Single Raw / QMF / Multi-Enroll EER (%) | LibriSpeech `test-clean` (`10,000` trials) 1-Utt Raw / QMF / 3-Utt EER (%) | Controlled 3-Domain Pooled 1-Utt / 3-Utt EER (%) | Controlled Macro-Avg 1-Utt / 3-Utt EER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/univ_full_conformer_large_dr_vectors`](https://huggingface.co/wq2012/univ_full_conformer_large_dr_vectors) | `conformer` (`large`) / `atp` / `extended_set_softmax` (`dr_vector`) | `7,148,743` | `2,400` | 14.32% / — / **13.92%** | 19.79% / 18.96% / 18.24% | 9.13% / 8.52% / 5.60% | 14.10% / 10.33% (raw 10.33%) | 13.52% / 9.77% |
| [`wq2012/univ_full_conformer_large_pfas`](https://huggingface.co/wq2012/univ_full_conformer_large_pfas) | `conformer` (`large`) / `pfas` / `extended_set_softmax` (`pfas`) | `7,312,327` | `2,400` | 14.18% / — / 14.05% | 19.78% / **18.83%** / 18.25% | 9.49% / 8.56% / 5.47% | 14.19% / 10.20% (raw 10.20%) | 13.36% / 9.43% |
| [`wq2012/univ_full_ecapa_c1024_arcface`](https://huggingface.co/wq2012/univ_full_ecapa_c1024_arcface) | `ecapa_tdnn` (`large`) / `asp` / `arcface` (`cosine`) | `14,923,136` | `4,500` | 15.14% / 14.23% / 14.06% | 21.18% / 19.67% / 18.89% | 9.41% / **8.06%** / **5.43%** | 14.69% / 11.00% (raw 11.03%) | 14.30% / 10.67% |
| [`wq2012/univ_full_conformer_large_arcface`](https://huggingface.co/wq2012/univ_full_conformer_large_arcface) | `conformer` (`large`) / `atp` / `arcface` (`cosine`) | `6,983,364` | `4,500` | 15.20% / 14.56% / 14.58% | 21.11% / 19.94% / 18.89% | 9.90% / 8.73% / 6.13% | 14.55% / 10.50% (raw 11.50%) | 14.62% / 10.97% |
| **2-Way Score Fusion (`0.1·Conformer-L + 0.9·ECAPA-c1024`)** | Heterogeneous Cosine + AS-Norm + QMF | `21,906,500` | — | — / 14.18% / **13.98%** | — / 19.59% / — | 8.44% / 8.12% / — | — | — |
| **3-Way Multi-Enroll Fusion (`0.5·Conf-L + 0.3·ECAPA + 0.2·PFAS`)** | Cosine AS-Norm + Frame-Level PFAS | `29,218,827` | — | — | — / — / **17.98%** (`AUC 0.9098`) | — / — / **5.20%** (`AUC 0.9871`) | — | — |
<!-- END README_FLAGSHIP_TABLE -->

### 2. Original Published Paper Reference Results (Internal Proprietary Datasets)

| Paper & Venue | Architecture & Loss / Scoring | Original Training Corpus | Original Evaluation Benchmark | Reported Metric in Original Paper |
| :--- | :--- | :--- | :--- | :--- |
| **GE2E** ([Wan et al., ICASSP 2018](https://arxiv.org/pdf/1710.10467)) | 3-layer LSTM (`768` hidden, `256` proj) + `GE2E-Softmax` | ~36 M utterances from ~`18K` speakers (anonymised logs) | Internal TI-SV: 1,000 speakers, ≈6.3 enrollment / 7.2 test utterances per speaker | `3.55%` EER (GE2E) vs. `4.06%` (softmax) and `4.13%` (TE2E), Table 3 — multi-utterance enrollment, not comparable to 1-utt numbers |
| **Dr-Vectors** ([Pelecanos et al., Interspeech 2021](https://arxiv.org/pdf/2104.01989)) | LSTM + `Extended-Set Softmax` + `Dr-Vectors` Residual Scorer | Internal Proprietary Multi-Condition Corpus | Internal Multi-Condition Speaker Verification | `~4–11%` relative EER reduction over Cosine + GE2E |
| **Conformer + ATP** ([Wang et al., Odyssey 2022](https://arxiv.org/pdf/2202.12163)) | Conformer Encoder + `Attentive Temporal Pooling (ATP)` | Internal Multi-Language Speech Corpus (`65` languages) | Internal Streaming & Long-Form Identification | Significant EER & latency improvement over Mean/Last pooling |
| **PFAS** ([Pelecanos et al., Odyssey 2022](https://arxiv.org/pdf/2203.05642v3)) | Conformer + Multi-Head `(K, V)` `Parameter-Free Attentive Scoring` | Internal Proprietary Multi-Vendor Corpus | Internal 1-Utt & Multi-Utt Speaker Verification | `~8–14%` relative EER reduction over Cosine baseline |

### 3. Tier-1 Universal Multi-Domain Pretrained Models (`19` `univ_*` Models, `4,600` Steps / `15.58` Epochs)

All 19 `univ_*` models below are trained simultaneously on **LibriSpeech (`232` spk) + CN-Celeb2 (`119` spk) + VoxCeleb1 (`30` spk)** (`381` speakers, `9,447` utterances) for `4,600` steps with Domain-Balanced Stratified Batching (`75%` within-domain, `25%` cross-domain) and variable-length crops `T ~ U(90, 180)` (verified from [`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json)):

<!-- BEGIN README_UNIV_TABLE -->
| Hugging Face Repository | Steps | Backbone / Size | Pooling / Loss / Scoring | Params | LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | Macro-Avg 1-Utt / 3-Utt EER (%) | Pooled 1-Utt / 3-Utt EER (%) | Macro 1-Utt / 3-Utt minDCF@0.01 |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface) | `4,600` | `conformer` (`small`) | `atp` / `cosface` / `cosine` | `2,029,892` | **7.67%** / **3.80%** | **16.73%** / **11.90%** | **17.05%** / **11.80%** | **13.82%** / **9.17%** | **13.94%** / **9.67%** | `0.7159` / `0.5180` |
| [`wq2012/univ_conformer_small_sphereface`](https://huggingface.co/wq2012/univ_conformer_small_sphereface) | `4,600` | `conformer` (`small`) | `atp` / `sphereface` / `cosine` | `2,029,892` | 8.00% / 4.20% | 17.00% / 12.00% | 18.40% / 14.20% | 14.47% / 10.13% | 14.20% / 11.07% | `0.7021` / **`0.5100`** |
| [`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface) | `4,600` | `conformer` (`small`) | `atp` / `arcface` / `cosine` | `2,029,892` | 8.27% / 4.60% | 16.93% / 12.60% | 18.05% / 12.30% | 14.42% / 9.83% | 14.80% / 10.10% | **`0.6943`** / `0.6440` |
| [`wq2012/univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast) | `4,600` | `conformer` (`small`) | `atp` / `ge2e_contrast` / `cosine` | `2,029,894` | 10.00% / 6.90% | 18.63% / 14.10% | 19.25% / 16.00% | 15.96% / 12.33% | 16.56% / 12.67% | `0.7788` / `0.6440` |
| [`wq2012/univ_resnet_small_ge2e`](https://huggingface.co/wq2012/univ_resnet_small_ge2e) | `4,600` | `resnet` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `2,239,922` | 11.03% / 5.70% | 18.47% / 12.70% | 21.45% / 17.90% | 16.98% / 12.10% | 16.79% / 12.57% | `0.7926` / `0.8053` |
| [`wq2012/univ_conformer_tiny_ge2e`](https://huggingface.co/wq2012/univ_conformer_tiny_ge2e) | `4,600` | `conformer` (`tiny`) | `atp` / `ge2e_softmax` / `cosine` | `693,126` | 12.10% / 7.80% | 18.73% / 15.00% | 20.60% / 16.40% | 17.14% / 13.07% | 16.93% / 13.03% | `0.9275` / `0.7800` |
| [`wq2012/univ_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_base_ge2e) | `4,600` | `ecapa_tdnn` (`base`) | `asp` / `ge2e_softmax` / `cosine` | `3,190,626` | 11.73% / 7.90% | 20.10% / 15.70% | 20.20% / 17.20% | 17.34% / 13.60% | 17.25% / 14.00% | `0.9028` / `0.7533` |
| [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) | `4,600` | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 13.47% / 9.20% | 19.40% / 16.30% | 20.85% / 16.40% | 17.91% / 13.97% | 17.54% / 13.87% | `0.9139` / `0.8227` |
| [`wq2012/univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas) | `4,600` | `conformer` (`small`) | `pfas` / `extended_set_softmax` / `pfas` | `2,162,187` | 11.73% / 8.10% | 20.93% / 18.80% | 19.45% / 16.20% | 17.37% / 14.37% | 17.60% / 14.70% | `0.8726` / `0.7607` |
| [`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e) | `4,600` | `ecapa_tdnn` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,684,418` | 12.60% / 8.60% | 19.93% / 15.40% | 21.10% / 18.00% | 17.88% / 14.00% | 17.62% / 14.07% | `0.9548` / `0.8473` |
| [`wq2012/univ_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_tiny_ge2e) | `4,600` | `ecapa_tdnn` (`tiny`) | `asp` / `ge2e_softmax` / `cosine` | `626,722` | 13.00% / 8.50% | 20.17% / 16.00% | 20.10% / 16.30% | 17.76% / 13.60% | 17.80% / 14.20% | `0.9285` / `0.8100` |
| [`wq2012/univ_conformer_base_ge2e`](https://huggingface.co/wq2012/univ_conformer_base_ge2e) | `4,600` | `conformer` (`base`) | `atp` / `ge2e_softmax` / `cosine` | `4,551,558` | 12.83% / 8.70% | 19.53% / 15.60% | 21.90% / 17.00% | 18.09% / 13.77% | 17.90% / 14.40% | `0.9341` / `0.7893` |
| [`wq2012/univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e) | `4,600` | `conformer` (`small`) | `pfas` / `ge2e_softmax` / `pfas` | `2,162,187` | 12.57% / 9.80% | 20.60% / 17.90% | 20.40% / 16.40% | 17.86% / 14.70% | 18.31% / 15.53% | `0.8347` / `0.7593` |
| [`wq2012/univ_conformer_small_ext_softmax`](https://huggingface.co/wq2012/univ_conformer_small_ext_softmax) | `4,600` | `conformer` (`small`) | `atp` / `extended_set_softmax` / `cosine` | `2,029,894` | 13.03% / 9.00% | 20.80% / 16.30% | 21.50% / 16.40% | 18.44% / 13.90% | 18.61% / 14.93% | `0.9206` / `0.7707` |
| [`wq2012/univ_transformer_small_ge2e`](https://huggingface.co/wq2012/univ_transformer_small_ge2e) | `4,600` | `transformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `1,239,622` | 13.63% / 10.10% | 21.43% / 16.50% | 21.55% / 16.60% | 18.87% / 14.40% | 18.73% / 14.87% | `0.8972` / `0.8613` |
| [`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e) | `4,600` | `lstm` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,710,914` | 14.17% / 9.60% | 21.73% / 16.20% | 21.10% / 17.30% | 19.00% / 14.37% | 19.14% / 14.87% | `0.9560` / `0.9040` |
| [`wq2012/univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors) | `4,600` | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 13.43% / 10.20% | 21.93% / 16.40% | 20.60% / 16.90% | 18.66% / 14.50% | 19.43% / 15.13% | `0.9531` / `0.7827` |
| [`wq2012/univ_mamba_small_ge2e`](https://huggingface.co/wq2012/univ_mamba_small_ge2e) | `4,600` | `mamba` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,028,674` | 16.23% / 12.20% | 21.37% / 18.00% | 20.95% / 17.40% | 19.52% / 15.87% | 19.57% / 16.57% | `0.9644` / `0.9180` |
| [`wq2012/univ_conformer_small_triplet`](https://huggingface.co/wq2012/univ_conformer_small_triplet) | `4,600` | `conformer` (`small`) | `atp` / `triplet` / `cosine` | `2,029,892` | 20.70% / 17.20% | 24.87% / 20.00% | 22.30% / 17.80% | 22.62% / 18.33% | 23.23% / 19.57% | `0.9720` / `0.9213` |
<!-- END README_UNIV_TABLE -->

### 4. Universal Multi-Domain vs. Single-Domain Training (`19 / 19` Matched Pairs Improved)

At full convergence (`S_safe`), **Universal Multi-Domain training outperforms the matched LibriSpeech specialist in all `19 / 19` `univ_*` vs. `ls_*` pairs** across **Pooled 1-Utt EER** (`-0.92` to `-3.71 pp`), **Macro-Avg 1-Utt EER** (`-0.89` to `-3.63 pp`), and **Pooled 3-Utt EER** (`-0.93` to `-4.00 pp`). Crucially, the matched-data control in `r2_seed_and_control_results.json` isolates speaker/domain diversity from utterance count: subsampling the 381-speaker Universal mix to **`5,788` utterances** (matching LibriSpeech) and training for **`2,400` steps** (matching compute) achieves **`16.32%` Pooled 1-Utt EER** (`16.64%` Macro 1-Utt, `11.93%` Pooled 3-Utt) vs. **`20.20%`** for `ls_conformer_small_ge2e` (**`-3.88 pp`** improvement at identical utterance count and step budget), and **`15.29%` Pooled 1-Utt EER** (`11.20%` Pooled 3-Utt) at `4,600` steps:

| Model Paradigm | Representative Checkpoint (`Steps`) | LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | Universal Macro-Avg 1-Utt / 3-Utt EER (%) | Universal Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech-Only Specialist (`GE2E`)** | [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) (`2,400`) | 12.07% / 7.50% | 26.20% / 23.20% | 21.25% / 18.50% | 19.84% / 16.40% | 20.20% / 16.93% |
| **CN-Celeb2-Only Specialist (`GE2E`)** | [`wq2012/cn_conformer_small_ge2e`](https://huggingface.co/wq2012/cn_conformer_small_ge2e) (`1,600`) | 25.63% / 20.70% | 21.93% / 17.50% | 27.75% / 24.70% | 25.11% / 20.97% | 25.68% / 21.23% |
| **VoxCeleb1-Only Specialist (`GE2E`)** | [`wq2012/vox_conformer_small_ge2e`](https://huggingface.co/wq2012/vox_conformer_small_ge2e) (`1,000`) | 20.00% / 15.20% | 29.83% / 29.60% | 20.50% / 16.00% | 23.44% / 20.27% | 23.64% / 20.70% |
| **Universal Multi-Domain (`GE2E`)** | [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) (`4,600`) | 13.47% / 9.20% | 19.40% / 16.30% | 20.85% / 16.40% | 17.91% / 13.97% | 17.54% / 13.87% |
| **LibriSpeech-Only Specialist (`ArcFace`)** | [`wq2012/ls_conformer_small_arcface`](https://huggingface.co/wq2012/ls_conformer_small_arcface) (`2,400`) | 8.13% / 4.20% | 21.80% / 18.20% | 17.65% / 13.60% | 15.86% / 12.00% | 16.23% / 12.40% |
| **Universal Multi-Domain (`ArcFace`)** | [`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface) (`4,600`) | 8.27% / 4.60% | 16.93% / 12.60% | 18.05% / 12.30% | 14.42% / 9.83% | 14.80% / 10.10% |
| **Universal Multi-Domain (`CosFace`)** | [`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface) (`4,600`) | **7.67%** / **3.80%** | **16.73%** / **11.90%** | **17.05%** / **11.80%** | **13.82%** / **9.17%** | **13.94%** / **9.67%** |
| **Full-Data Universal Flagship (`PFAS`)** | [`wq2012/univ_full_conformer_large_pfas`](https://huggingface.co/wq2012/univ_full_conformer_large_pfas) (`2,400`) | 8.07% / 4.20% | 18.17% / 12.30% | 15.85% / 11.80% | **13.36%** / **9.43%** | 14.19% / 10.20% |

### 5. Tier-1 Single-Domain Pretrained Specialist Models (`25` `ls_*`, `cn_*`, `vox_*` Models)

<!-- BEGIN README_SINGLE_TABLE -->
| Hugging Face Repository | Train Domain (`Steps`) | Backbone / Size | Pooling / Loss / Scoring | Params | LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | Macro-Avg 1-Utt / 3-Utt EER (%) | Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/ls_conformer_small_cosface`](https://huggingface.co/wq2012/ls_conformer_small_cosface) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `cosface` / `cosine` | `2,029,892` | **8.03%** / **4.10%** | 24.30% / 19.40% | 19.00% / 13.60% | 17.11% / 12.37% | 17.62% / 13.67% |
| [`wq2012/ls_conformer_small_arcface`](https://huggingface.co/wq2012/ls_conformer_small_arcface) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `arcface` / `cosine` | `2,029,892` | 8.13% / 4.20% | 21.80% / 18.20% | 17.65% / 13.60% | **15.86%** / 12.00% | **16.23%** / 12.40% |
| [`wq2012/ls_conformer_small_sphereface`](https://huggingface.co/wq2012/ls_conformer_small_sphereface) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `sphereface` / `cosine` | `2,029,892` | 8.40% / **4.10%** | 23.17% / 18.60% | **16.40%** / **12.80%** | 15.99% / **11.83%** | 16.60% / **12.13%** |
| [`wq2012/ls_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/ls_conformer_small_ge2e_contrast) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `ge2e_contrast` / `cosine` | `2,029,894` | 9.47% / 5.60% | 26.27% / 22.00% | 20.50% / 15.60% | 18.74% / 14.40% | 18.65% / 14.27% |
| [`wq2012/ls_resnet_small_ge2e`](https://huggingface.co/wq2012/ls_resnet_small_ge2e) | `librispeech` (`2,600`) | `resnet` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `2,239,922` | 9.53% / 6.00% | 26.90% / 20.20% | 22.60% / 19.10% | 19.68% / 15.10% | 20.12% / 15.87% |
| [`wq2012/ls_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_base_ge2e) | `librispeech` (`2,600`) | `ecapa_tdnn` (`base`) | `asp` / `ge2e_softmax` / `cosine` | `3,190,626` | 10.60% / 6.20% | 26.87% / 22.10% | 22.10% / 17.80% | 19.86% / 15.37% | 20.12% / 16.47% |
| [`wq2012/ls_conformer_tiny_ge2e`](https://huggingface.co/wq2012/ls_conformer_tiny_ge2e) | `librispeech` (`2,400`) | `conformer` (`tiny`) | `atp` / `ge2e_softmax` / `cosine` | `693,126` | 11.07% / 6.30% | 26.00% / 22.10% | 20.90% / 17.70% | 19.32% / 15.37% | 19.76% / 15.73% |
| [`wq2012/ls_conformer_small_pfas`](https://huggingface.co/wq2012/ls_conformer_small_pfas) | `librispeech` (`2,400`) | `conformer` (`small`) | `pfas` / `extended_set_softmax` / `pfas` | `2,162,187` | 11.20% / 7.80% | 26.87% / 23.80% | 22.05% / 18.00% | 20.04% / 16.53% | 20.43% / 16.67% |
| [`wq2012/ls_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_small_ge2e) | `librispeech` (`2,400`) | `ecapa_tdnn` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,684,418` | 11.37% / 6.10% | 26.47% / 21.10% | 21.10% / 18.10% | 19.64% / 15.10% | 20.05% / 15.77% |
| [`wq2012/ls_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_pfas_ge2e) | `librispeech` (`2,400`) | `conformer` (`small`) | `pfas` / `ge2e_softmax` / `pfas` | `2,162,187` | 11.60% / 9.20% | 27.53% / 26.20% | 20.90% / 17.00% | 20.01% / 17.47% | 20.57% / 17.70% |
| [`wq2012/ls_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_tiny_ge2e) | `librispeech` (`2,400`) | `ecapa_tdnn` (`tiny`) | `asp` / `ge2e_softmax` / `cosine` | `626,722` | 11.63% / 6.60% | 27.80% / 22.50% | 20.95% / 18.90% | 20.13% / 16.00% | 20.55% / 16.13% |
| [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 12.07% / 7.50% | 26.20% / 23.20% | 21.25% / 18.50% | 19.84% / 16.40% | 20.20% / 16.93% |
| [`wq2012/ls_conformer_base_ge2e`](https://huggingface.co/wq2012/ls_conformer_base_ge2e) | `librispeech` (`2,600`) | `conformer` (`base`) | `atp` / `ge2e_softmax` / `cosine` | `4,551,558` | 12.13% / 8.90% | 27.30% / 22.50% | 21.10% / 18.50% | 20.18% / 16.63% | 20.32% / 16.73% |
| [`wq2012/ls_conformer_small_dr_vectors`](https://huggingface.co/wq2012/ls_conformer_small_dr_vectors) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 13.20% / 8.90% | 27.93% / 22.60% | 21.80% / 17.90% | 20.98% / 16.47% | 21.40% / 16.93% |
| [`wq2012/ls_transformer_small_ge2e`](https://huggingface.co/wq2012/ls_transformer_small_ge2e) | `librispeech` (`2,400`) | `transformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `1,239,622` | 13.53% / 8.10% | 28.20% / 24.20% | 21.50% / 17.70% | 21.08% / 16.67% | 21.35% / 17.00% |
| [`wq2012/ls_lstm_small_ge2e`](https://huggingface.co/wq2012/ls_lstm_small_ge2e) | `librispeech` (`2,400`) | `lstm` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,710,914` | 14.00% / 10.20% | 29.07% / 25.20% | 22.20% / 19.00% | 21.76% / 18.13% | 21.73% / 18.27% |
| [`wq2012/ls_conformer_small_ext_softmax`](https://huggingface.co/wq2012/ls_conformer_small_ext_softmax) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `extended_set_softmax` / `cosine` | `2,029,894` | 14.00% / 9.20% | 29.40% / 25.80% | 22.80% / 20.40% | 22.07% / 18.47% | 22.32% / 18.23% |
| [`wq2012/ls_mamba_small_ge2e`](https://huggingface.co/wq2012/ls_mamba_small_ge2e) | `librispeech` (`2,400`) | `mamba` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,028,674` | 14.43% / 10.40% | 27.60% / 23.80% | 21.15% / 18.20% | 21.06% / 17.47% | 21.38% / 17.63% |
| [`wq2012/ls_conformer_small_triplet`](https://huggingface.co/wq2012/ls_conformer_small_triplet) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `triplet` / `cosine` | `2,029,892` | 17.87% / 14.20% | 32.10% / 28.70% | 20.55% / 17.60% | 23.51% / 20.17% | 24.15% / 20.50% |
| [`wq2012/cn_ecapa_tdnn_small_arcface`](https://huggingface.co/wq2012/cn_ecapa_tdnn_small_arcface) | `cnceleb2` (`1,800`) | `ecapa_tdnn` (`small`) | `asp` / `arcface` / `cosine` | `1,684,416` | 14.13% / 8.80% | **17.63%** / **13.00%** | 18.10% / 15.00% | 16.62% / 12.27% | 17.32% / 12.83% |
| [`wq2012/cn_conformer_small_ge2e`](https://huggingface.co/wq2012/cn_conformer_small_ge2e) | `cnceleb2` (`1,600`) | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 25.63% / 20.70% | 21.93% / 17.50% | 27.75% / 24.70% | 25.11% / 20.97% | 25.68% / 21.23% |
| [`wq2012/cn_conformer_small_pfas`](https://huggingface.co/wq2012/cn_conformer_small_pfas) | `cnceleb2` (`1,600`) | `conformer` (`small`) | `pfas` / `extended_set_softmax` / `pfas` | `2,162,187` | 25.57% / 22.40% | 22.20% / 20.70% | 28.45% / 26.80% | 25.41% / 23.30% | 27.65% / 25.47% |
| [`wq2012/cn_conformer_small_dr_vectors`](https://huggingface.co/wq2012/cn_conformer_small_dr_vectors) | `cnceleb2` (`1,600`) | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 29.47% / 23.40% | 24.53% / 20.80% | 28.00% / 26.10% | 27.33% / 23.43% | 28.91% / 25.57% |
| [`wq2012/vox_conformer_small_ge2e`](https://huggingface.co/wq2012/vox_conformer_small_ge2e) | `voxceleb1` (`1,000`) | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 20.00% / 15.20% | 29.83% / 29.60% | 20.50% / 16.00% | 23.44% / 20.27% | 23.64% / 20.70% |
| [`wq2012/vox_ecapa_tdnn_small_ext_softmax`](https://huggingface.co/wq2012/vox_ecapa_tdnn_small_ext_softmax) | `voxceleb1` (`1,000`) | `ecapa_tdnn` (`small`) | `asp` / `extended_set_softmax` / `cosine` | `1,684,418` | 19.83% / 16.80% | 30.97% / 30.00% | 22.40% / 16.80% | 24.40% / 21.20% | 25.10% / 21.73% |
<!-- END README_SINGLE_TABLE -->

---

## Deviations from the Source Papers

Several released zoo configurations implement **FlaxSpeaker variants** of the cited algorithms alongside the exact paper-faithful configurations (`icassp2018_ge2e_config`, `interspeech2021_dr_vectors_config`, `odyssey2022_conformer_atp_config`, `odyssey2022_pfas_config` with `pure_paper_mode=True`). In `v0.5.0`, **all four `pure_paper_mode=True` presets have been trained for `4,600` steps on the Universal corpus** and benchmarked in `pretrained_models/ablation_results.json` (`studies.paper_faithful_presets_smoke_test`).

| Component | Source paper | Released FlaxSpeaker `v0.5.0` zoo checkpoints | Paper-faithful preset (`4,600`-step Universal Pooled 1 / 3-utt EER) |
| :--- | :--- | :--- | :--- |
| Training budget / data | GE2E: ~36 M utts, 18 K spk, N=64×M=10; Dr-Vectors/PFAS: 16 spk × 8 utts, 500 k steps | Tier-1 (`44` models): `1,000–4,600` steps (`S_safe`) × (`8` spk × `4` utts), 381 / 232 / 119 / 30 spk; Tier-2 (`4` `univ_full_*` models): `2,400–4,500` steps on `12,193` speakers (`464,557` clean utts / `1.86M` augmented items); `T ~ U(90, 180)` | Evaluated on the 381-speaker Universal split (`4,600` steps) |
| GE2E encoder | 3 × LSTM-768 + 256-D projection, linear output | `ls_lstm_*` / `univ_lstm_*`: 3 × LSTM-256, no projection, per-layer LayerNorm + residual, ASP pooling, nonlinear bottleneck FFN (`19.14% / 14.87%`) | `icassp2018_ge2e_config(pure_paper_mode=True)` (`30.52% / 29.13%` — unnormalized 3×768 LSTM overfits/stalls on 9.4k utts) |
| GE2E contrast loss | Eq. 7 | `*_ge2e_contrast`: **hybrid contrast surrogate** (value = Eq. 7; gradient = 0.25 Eq. 7 + 0.25 BCE + 0.50 softmax-CE), `ge2e_init_b = −1`, `b` grad scale 0.5 (`16.56% / 12.67%`) | `ge2e_contrast_surrogate: false` (default) |
| Extended-set softmax | Eq. 2 (pure) | `*_ext_softmax`, `*_dr_vectors`, `*_pfas`: **blended** 0.5 row-softmax + 0.5 extended-set with `−log N` shift (`18.61% / 14.93%`) | `pure_extended_set: true` (`18.80% / 14.30%` on Conformer-Small) |
| Dr-Vector scorer | 200 base + 56 side dims; 3 × 256 leaky-ReLU(0.2); weighted sum; trainable affine | `*_dr_vectors`: cosine over all 192/256 dims, hidden `(128, 64)` ReLU + LayerNorm, linear read-out, fixed 0.25 residual weight (`14.10% / 10.33%` on `univ_full_conformer_large_dr_vectors`; `19.43% / 15.13%` on Tier-1) | `interspeech2021_dr_vectors_config(pure_paper_mode=True)` (`18.30% / 14.07%`) |
| ATP pooling | single head, linear-sigmoid weights, full-D mean/std | `attention_dim: 128` tanh bottleneck, 4 channel-subspace heads (`subspace_heads: true`); source paper is a language-ID paper | `odyssey2022_conformer_atp_config(pure_paper_mode=True)` (`18.98% / 16.27%`) |
| PFAS head | ATP → affine + ReLU(512) → linear; 32 tied keys (2,048-D) | `univ_full_conformer_large_pfas`: `atp_ffn` with `K=16` tied keys × `(32 + 64)` = `1,536-D` (`14.19% / 10.20%`); Tier-1 `*_pfas*`: **PFAS-SN** (`shared_nonlinear` + global ATP anchor, 4 keys × `(32 + 128)`, `17.60% / 14.70%`) | `odyssey2022_pfas_config(pure_paper_mode=True)` (`23.02% / 21.00%`) |
| ECAPA-TDNN / Angular Margin | BatchNorm / fixed margin or integer `m = 4` | `BatchNorm` after activation (`17.62% / 14.07%` vs `18.32% / 15.30%` legacy `LayerNorm` before activation); `Sub-Center ArcFace K=3` + linear `margin_warmup_steps` (`13.94%` CosFace, `14.80%` ArcFace); real-valued SphereFace `m = 1.35` with λ-annealing (`14.20%`) | — |
| Augmentation | waveform MTR with recorded noise + room simulator | Tier-2 `univ_full_*`: real `MUSAN` + `RIRS_NOISES` + `3x` speed perturbation; Tier-1: synthetic log-Mel noise (SNR 12–20 dB) + `SpecAugment` | — |
| Evaluation | VoxCeleb1-O 37,611 trials (cleaned `veri_test2.txt`) over 40 speakers | Tier-2 `univ_full_*`: Official 37,611-trial `VoxCeleb1-O` (`13.92%` QMF), 3.48M-trial `CN-Celeb-E` (`17.98%` Multi-Enroll), 10,000-trial `LibriSpeech test-clean` (`5.20%` 3-Utt); Tier-1 + Tier-2: 10-speaker subset of `veri_test2.txt` (2,500 trials) | — |

---

## Testing, Linting & Documentation

Run the unit test suite (covering all backbones, exact padding-mask invariance across all backbone variants including `BatchNorm` ECAPA-TDNN, bidirectional LSTM, and Mamba, all losses including `Sub-Center ArcFace`, pure-vs-surrogate `ge2e_contrast`, pure/blended `extended_set_softmax`, and `margin_warmup_steps`, all pooling heads including `"dot"` and `"unshared_nonlinear"` PFAS attention, VAD + AGC, real waveform augmentation and `float16` shards, Top-300 `AS-Norm` and `QMF` calibration, Dr-Vectors paper options, every `pure_paper_mode` preset field reaching the built module, strict config-key validation, export-window resolution, local-vs-Hub checkpoint equivalence, TFLite export, version consistency across `pyproject.toml` / `setup.py` / `__init__.py` / `CHANGELOG.md`, documentation script existence, and `training_provenance` consistency across all 48 checkpoints):

```bash
pytest -v tests.py
flake8 .
./run_pdoc.sh
```
