# FlaxSpeaker

[![Python application](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml/badge.svg)](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml) [![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-44%20Pretrained%20Models-blue)](https://huggingface.co/collections/wq2012/flaxspeaker) [![PyPI Version](https://img.shields.io/pypi/v/flaxspeaker.svg)](https://pypi.python.org/pypi/flaxspeaker) [![Python Versions](https://img.shields.io/pypi/pyversions/flaxspeaker.svg)](https://pypi.org/project/flaxspeaker) [![Downloads](https://static.pepy.tech/badge/flaxspeaker)](https://pepy.tech/project/flaxspeaker) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**FlaxSpeaker** is an end-to-end open-source speaker verification library built on [JAX](https://jax.readthedocs.io) and [Flax](https://flax.readthedocs.io). Designed for education and academic research, it unifies dataset preparation, frame-level energy Voice Activity Detection (VAD) with peak Automatic Gain Control (AGC), an 80-D log-Mel frontend with CMVN, six neural sequence backbones, eight temporal pooling heads, nine metric-learning and angular-margin training objectives (seven of which are benchmarked), three scoring paradigms, universal multi-domain stratified training, and one-command export to **TFLite (FP32 / dynamic-range INT8)**, **TensorFlow SavedModel**, **SafeTensors**, and the **Hugging Face Hub**.

> **Read this before using the released checkpoints.** All 44 published models are **400-step, single-seed CPU demonstration checkpoints** (32 utterances per step, ≈1.35 epochs over 9,447 utterances from 381 speakers). They reach 18.75–30.23 % pooled 1-utt EER on the project's trial lists and 22.5–32.2 % EER on a *10-speaker subset* of VoxCeleb1-O (not comparable to published VoxCeleb1-O numbers; published systems reach ≈1 %). They are intended for *relative* comparisons of architectures and objectives under an identical budget, not as state-of-the-art models. Several released configurations are **FlaxSpeaker variants** of the cited algorithms (see [Deviations from the source papers](#deviations-from-the-source-papers)); the paper-faithful configurations are available as `pure_paper_mode=True` presets but have **not** been trained or released. The released weights were trained by earlier revisions of this code base: *evaluating* them with the current code reproduces the archived numbers exactly (checked for `univ_conformer_small_ge2e` and `ls_conformer_small_ge2e`), but *re-training* seed 42 with the current code does **not** reproduce them (20.94 % / 21.40 % vs. the archived 19.40 % / 20.12 % pooled 1-utt EER; see [RESEARCH_REPORT.md §10](RESEARCH_REPORT.md) and `pretrained_models/r2_seed_and_control_results.json`).

For the companion PyTorch educational project, see [SpeakerRecognitionFromScratch](https://github.com/wq2012/SpeakerRecognitionFromScratch).

---

## Disclaimer & Open-Source Reproduction Notice

**This is not an officially supported Google product.**

This repository provides a from-scratch re-implementation in **JAX / Flax**, written using only the published papers (no proprietary code or data), of the mathematical formulations and neural architectures described in the following research papers:
1. **Generalized End-to-End Loss for Speaker Verification** — *Li Wan, Quan Wang, Alan Papir, Ignacio Lopez Moreno* ([ICASSP 2018, arXiv:1710.10467](https://arxiv.org/pdf/1710.10467))
2. **Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition** — *Jason Pelecanos, Quan Wang, Ignacio Lopez Moreno* ([Interspeech 2021, arXiv:2104.01989](https://arxiv.org/pdf/2104.01989))
3. **Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech** — *Quan Wang, Yang Yu, Jason Pelecanos, Yiling Huang, Ignacio Lopez Moreno* ([Odyssey 2022, arXiv:2202.12163](https://arxiv.org/pdf/2202.12163)) — a *language identification* paper whose attentive temporal pooling head is repurposed here for speaker embeddings
4. **Parameter-Free Attentive Scoring for Speaker Verification** — *Jason Pelecanos, Quan Wang, Yiling Huang, Ignacio Lopez Moreno* ([Odyssey 2022, arXiv:2203.05642](https://arxiv.org/pdf/2203.05642))

Because the original papers were trained and evaluated on proprietary internal datasets and infrastructure (for ~500k steps on thousands of speakers), **exact numerical parity with the original paper tables is neither expected nor claimed**, and the released 400-step checkpoints should not be read as reproductions of the papers' results. All pretrained checkpoints and benchmark tables in this repository are trained and evaluated from scratch on publicly available speech corpora (**LibriSpeech**, **CN-Celeb2**, and the 40-speaker **VoxCeleb1 test partition**) with speaker-disjoint splits.

---

## System Architecture

![FlaxSpeaker System Architecture](resources/architecture_diagram.png)

### 1. Six Neural Sequence Backbones & Four Size Presets (0.63 M–4.55 M parameters in the released zoo; `large` is defined but never trained)
All backbones share a unified `(B, T, F) -> (B, T, D)` frame-rate-preserving sequence interface with **exact variable-length padding mask invariance** via `ModernSpeakerEncoder` ([`flaxspeaker/backbones.py`](flaxspeaker/backbones.py)) and scale automatically via `size_variant: tiny | small | base | large`:
- **Conformer** (`conformer`): Parallel Linear + 1D Conv Stem (`Dense(D) + SiLU(LayerNorm(Conv1D(k=3)))`) + Macaron FFN + Masked Multi-Head Self-Attention + Mask-Aware GroupNorm Depthwise Separable Convolution + Multi-Layer Feature Aggregation (MFA).
- **ECAPA-TDNN** (`ecapa_tdnn`): 1D SE-Res2Net dilated multi-scale blocks with Multi-Layer Feature Aggregation (MFA).
- **Mamba** (`mamba`): Selective State Space Model (SSM) implemented in pure JAX via parallel `jax.lax.associative_scan` (`O(T)` linear time complexity, supporting both causal unidirectional and mask-aware bidirectional `bidirectional: true` modes).
- **ResNet** (`resnet`): 2D Time-Frequency Residual Network with Squeeze-and-Excitation (SE-ResNet, 8 residual blocks across 4 stages `(2, 2, 2, 2)`).
- **Transformer** (`transformer`): Parallel Linear + 1D Conv Stem (`Dense(D) + SiLU(LayerNorm(Conv1D(k=3)))`) + Pre-LayerNorm Multi-Head Self-Attention encoder with Multi-Layer Feature Aggregation (MFA).
- **LSTM** (`lstm`): Deep stacked uni- or bidirectional LSTM (`bidirectional: true`, mask-aware with `seq_lengths` in both directions) with configurable hidden size, optional per-layer projection, and per-layer LayerNorm + residual connections. No dropout is implemented in any backbone.

### 2. Eight Temporal Pooling Heads ([`flaxspeaker/pooling.py`](flaxspeaker/pooling.py))
- **Attentive Temporal Pooling** (`atp`): Sigmoid frame-gated statistics pooling (`concat(mu_h, sigma_h)`) extended from the single-head linear-sigmoid ATP of [Wang et al., Odyssey 2022, `arXiv:2202.12163`](https://arxiv.org/pdf/2202.12163) (a language-identification paper). FlaxSpeaker defaults add a tanh bottleneck (`attention_dim: 128`) and 4 channel-subspace heads (`subspace_heads: true`); `attention_dim: 0, num_heads: 1` recovers the paper form. Followed by a nonlinear bottleneck FFN (`LayerNorm -> Dense(256) -> LayerNorm -> ReLU -> Dense(d_e) -> L2_norm`).
- **Self-Attentive Pooling** (`sap`) & **Attentive Statistics Pooling** (`asp`): Softmax attention-weighted frame mean and standard deviation.
- **Mean Pooling** (`mean`), **Last-Frame Pooling** (`last`), & **Temporal Statistics Pooling** (`stats`).
- **Cumulative Statistics Pooling** (`cumulative_stats`): Frame-wise causal running statistics for streaming inference.
- **Parameter-Free Attentive Scoring Head** (`pfas`): Two representation heads. `attention_type: atp_ffn` is the head of [Pelecanos et al., Odyssey 2022, `arXiv:2203.05642`](https://arxiv.org/pdf/2203.05642) (ATP → affine + ReLU → linear → keys/values; `odyssey2022_pfas_config(pure_paper_mode=True)` uses 32 tied keys × (16 + 48) = 2,048-D). `attention_type: shared_nonlinear` + `use_global_anchor: true` (**PFAS-SN, a FlaxSpeaker extension used by all released `*_pfas*` checkpoints**) uses per-frame key/value projections with multi-head tanh frame attention anchored on a global ATP value summary, 4 keys × (32 + 128) = 640-D. Both apply per-key L2 normalisation and global L2 normalisation of the concatenated value form (paper Eqs. 5–7).

### 3. Nine Speaker Recognition Losses & Three Scoring Backends ([`flaxspeaker/losses.py`](flaxspeaker/losses.py), [`flaxspeaker/scoring.py`](flaxspeaker/scoring.py))
- **Generalized End-to-End (GE2E) Loss**: `ge2e_softmax` (Eq. 6, exact leave-one-out centroids) and `ge2e_contrast` (Eq. 7) with learnable scale `w` and bias `b` ([Wan et al., ICASSP 2018](https://arxiv.org/pdf/1710.10467)). By default `ge2e_contrast` back-propagates Eq. 7 exactly. The released `*_ge2e_contrast` checkpoints were trained with the **hybrid contrast surrogate** (`ge2e_contrast_surrogate: true`): the value is Eq. 7 but the gradient is `0.25·Eq7 + 0.25·BCE + 0.50·GE2E-softmax-CE`, with `ge2e_init_b: -1.0` and `ge2e_contrast_b_grad_scale: 0.5` — a FlaxSpeaker objective, not Wan et al.'s contrast loss. When `scoring_type` is `pfas` or `dr_vector`, GE2E uses the even/odd split-batch test-vs-enrollment matrix instead of leave-one-out centroids.
- **Extended-Set Softmax Loss**: `extended_set_softmax`, supporting the paper's pure form (`pure_extended_set: true`, Eq. 2 of [Pelecanos et al., Interspeech 2021](https://arxiv.org/pdf/2104.01989)) and the **Blended Extended-Set Softmax (FlaxSpeaker variant)** used by all released `*_ext_softmax` / `*_dr_vectors` / `*_pfas` checkpoints (`pure_extended_set: false`: 50/50 blend of row-softmax and extended-set softmax with a `-log(N)` logit shift).
- **Decision Residual Networks (Dr-Vectors)**: `scoring_type: dr_vector` — single-stage end-to-end residual feed-forward scorer on `[e_t, e_e, c]`. `interspeech2021_dr_vectors_config(pure_paper_mode=True)` builds the paper's network ([arXiv:2104.01989](https://arxiv.org/pdf/2104.01989) Sec. 2/4: 256-D embedding split into `cosine_dims: 200` base dims + 56 side-information dims, 3 × Dense(256) leaky-ReLU(0.2) without normalisation, `readout: weighted_sum`, score = cosine + network output, `trainable_affine: true`, trained with pure extended-set softmax). The released `*_dr_vectors` checkpoints are a **down-scaled Dr-Vector variant** (no base/side split, hidden `(128, 64)` ReLU + LayerNorm, linear read-out, fixed `residual_weight: 0.25`, blended extended-set softmax). The 3-term cosine expansion `[c, c^2, sqrt(max(c, 0))]` (`num_cosine_terms: 3`) is a FlaxSpeaker extension that does **not** appear in the paper (which has only two numbered equations).
- **Parameter-Free Attentive Scoring (PFAS)**: `scoring_type: pfas` — cross-attention between test and enrollment `(K, V)` representations with per-trial softmax key–key attention weights `A_{ij}`, per-key and global L2 normalisation (Eqs. 5–7 of [Pelecanos et al., Odyssey 2022](https://arxiv.org/pdf/2203.05642)), a trainable scale, and key stacking (`N = M × E`) for multi-utterance enrollment.
- **Angular Margin Classification Losses**:
  - **ArcFace / AAM-Softmax** (`arcface`): `cos(theta + m)`
  - **CosFace / AM-Softmax** (`cosface`): `cos(theta) - m`
  - **SphereFace / A-Softmax** (`sphereface`): `(-1)^k * cos(m * theta) - 2k`
  - **Standard Softmax** (`softmax`), **Pairwise Loss** (`pairwise`), and **Triplet Loss** (`triplet`). (`softmax` and `pairwise` are implemented and unit-tested but **no released checkpoint uses them**.)
- **Adaptive Symmetric Score Normalization (AS-Norm)**: Top-`K` impostor-cohort z-normalisation (`use_as_norm: true`). The cohort must be passed explicitly (`cohort_embeddings=`, e.g. training-speaker embeddings) and must be disjoint from the trial utterances; `evaluate_verification_trials` raises otherwise. No reported number uses AS-Norm.

### 4. End-to-End Audio Pipeline, Mutually Exclusive Augmentation, Variable-Length Training & Universal Multi-Domain Batching
- **Frame-level energy VAD & AGC** ([`flaxspeaker/vad.py`](flaxspeaker/vad.py)): Frame energies are computed on the peak-normalised waveform (`apply_agc = True`, target peak `1.0`) and the resulting mask is applied to the log-Mel frames; 3-term dynamic thresholding `max(-42.0 dB, P_95(E) - 26.0 dB, P_10(E) + 4.5 dB)`, optional spectral flatness `<= 0.65` and speech-band (`300-3400 Hz`) energy ratio `>= 0.25` gates, hangover smoothing (`min_speech_frames = 3`, `hangover_frames = 8`), and a top-8-frame fallback. **One VAD constant (26 dB, the `VadConfig()` default) is used for data preparation, training, in-domain evaluation and the VoxCeleb1-O-subset evaluation.**
- **Acoustic Frontends & Variable-Length Training** ([`flaxspeaker/feature_extraction.py`](flaxspeaker/feature_extraction.py)): `log_mel` (used by every released checkpoint: natural-log Mel + energy VAD + CMVN; **not** reproducible with a Hugging Face `AutoFeatureExtractor`, which is why no `preprocessor_config.json` is shipped), optional `whisper_mel` / `hf_compat` / `mfcc` frontends, per-utterance variable sequence length sampling (`U(90, 180)` frames in all benchmark YAMLs) with exact binary padding masks `(B, T)`, evaluation centre-cropping to `max_eval_seq_len = 500` frames (5 s), and two-tier RAM + `.npy` disk caching (`FeatureCache`).
- **Mutually Exclusive SpecAugment & feature-domain noise augmentation** ([`flaxspeaker/specaug.py`](flaxspeaker/specaug.py)): The training-path "MTR" is **synthetic log-Mel-domain** augmentation — frequency-shaped Gaussian noise at `SNR ~ U(12, 20) dB` gated towards low-energy bins, plus a one-tap temporal smoothing with probability 0.4 — **not** waveform-domain multi-style training with recorded noise/RIR corpora. It is strictly mutually exclusive with `SpecAugment` (up to 2 frequency masks `<= 8` mel bins and 2 time masks `<= 12` frames): `35%` noise only, `32.5%` SpecAugment only, `32.5%` clean. A standalone waveform helper (`apply_waveform_augmentation`) exists but is not on the training path.
- **Universal Multi-Domain Stratified Batching** ([`flaxspeaker/feature_extraction.py`](flaxspeaker/feature_extraction.py)): Combines multiple heterogeneous datasets (`LibriSpeech`, `CN-Celeb2`, `VoxCeleb1`) using `75%` within-domain hard-negative batches and `25%` cross-domain score-calibration batches with temperature-balanced domain sampling `p_d proportional to |S_d|^0.35`.
- **Model Export** ([`flaxspeaker/export.py`](flaxspeaker/export.py), [`flaxspeaker/hf_compat.py`](flaxspeaker/hf_compat.py)): Export via `jax2tf` to **TFLite (FP32, dynamic-range INT8)** and **TF SavedModel** with a **fixed, unmasked input window** (`export.sequence_length`; `0` = `model.seq_len`; the 12 released `.tflite` files use **180 frames = 1.8 s**, encoder only, no scoring heads), and to **SafeTensors** / **Hugging Face Hub** directories (`config.json`, `model.safetensors`, `eval_results.json`, auto-generated model card). No TFLite EER is reported; the `≥ 0.9999` cosine fidelity was measured on 180-frame inputs.

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

FlaxSpeaker provides standalone command-line scripts in [`scripts/`](scripts/) covering the full dataset preparation, training, evaluation, ablation, figure generation, and deployment lifecycle (in addition to `python -m flaxspeaker`):

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

### 3. Evaluate 1-Utt, 3-Utt, and VoxCeleb1-O-subset Trials (`scripts/evaluate.py` / `scripts/eval_vox1_o_subset.py`)
Compute 1-utterance and 3-utterance EER (%), `minDCF@0.01`, `minDCF@0.05`, `minDCF@0.10`, ROC-AUC, and 95% bootstrap confidence intervals using any [Hugging Face Hub model ID (`wq2012/<model_id>`)](https://huggingface.co/collections/wq2012/flaxspeaker) or local checkpoint directory, as well as the **VoxCeleb1-O 10-speaker subset** (official `veri_test2.txt` filtered to trials whose two speakers are both among the 10 held-out evaluation speakers, 2,500 trials, ≈72 % target; **not comparable to published VoxCeleb1-O numbers** — the other 30 of the 40 official test speakers are training data for `vox_*` and `univ_*` models):

```bash
python scripts/evaluate.py \
  --model_dir wq2012/univ_conformer_small_ge2e \
  --trials_file ./data/speaker_datasets/splits/universal/trials.tsv \
  --multi_trials_file ./data/speaker_datasets/splits/universal/multi_enroll_trials.tsv \
  --output_json eval_summary.json

# Evaluate all 44 models on the VoxCeleb1-O 10-speaker subset (writes pretrained_models/vox1_o_10spk_subset_results.json)
python scripts/eval_vox1_o_subset.py --data_root ./data/speaker_datasets --models_dir ./pretrained_models
```

### 4. Run End-to-End Speaker Verification Inference (`scripts/inference.py`)
Verify whether two (or more) audio files belong to the same speaker and optionally save their L2-normalized speaker embeddings:

```bash
python scripts/inference.py \
  --model_dir wq2012/univ_conformer_small_ge2e \
  --enroll_audio hf_space/examples/librispeech_spk1_utt1.flac \
  --test_audio hf_space/examples/librispeech_spk1_utt2.flac
```

### 5. Export to HuggingFace Hub & TFLite (`scripts/export_model.py` / `scripts/publish_hf.py`)
```bash
python scripts/export_model.py \
  --model_dir wq2012/univ_conformer_small_ge2e \
  --output_dir ./exported_models/univ_conformer_small_ge2e \
  --export_tflite \
  --quantize_int8
```

### 6. Reproduce Full 44-Model Benchmark Suite, Controlled Ablations & Paper Figures (`scripts/run_benchmark_suite.py`, `scripts/run_r1_ablations.py`, `scripts/generate_figures.py`, `scripts/generate_demo_examples.py`)
```bash
# Run the complete 44-model benchmark matrix across all domains
python scripts/run_benchmark_suite.py --data-root ./data/speaker_datasets

# Reproduce the 5 controlled ablation studies (pretrained_models/ablation_results.json)
python scripts/run_r1_ablations.py --data_root ./data/speaker_datasets

# Seed-noise floor (3 seeds x 2 models), honest 800-step horizon point, matched-data control
# (pretrained_models/r2_seed_and_control_results.json)
python scripts/run_r2_controls.py --data_root ./data/speaker_datasets

# Regenerate all 6 paper figures from the JSON benchmark & ablation artifacts
python scripts/generate_figures.py \
  --benchmark-json pretrained_models/benchmark_results.json \
  --ablation-json pretrained_models/ablation_results.json \
  --vox1o-json pretrained_models/vox1_o_10spk_subset_results.json \
  --output-dir ./figures
```

---

## Python & HuggingFace API

Load any exported model directory or Hugging Face Hub repository ID ([`wq2012/<model_id>`](https://huggingface.co/collections/wq2012/flaxspeaker)) with `load_pretrained` or `FlaxSpeakerModel.from_pretrained`:

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

An interactive [Gradio 5.50.0](https://gradio.app) web application is included in [`hf_space/`](hf_space/). **It is not hosted on the Hugging Face Hub**: creating a Gradio Space under `wq2012` currently returns HTTP 402 (free CPU Spaces require a PRO subscription), so run it locally (`python hf_space/app.py`) or deploy it yourself with `python scripts/deploy_hf_space.py --repo_id <your-namespace>/FlaxSpeaker`. Features:
- Real inference only (no precomputed outputs) with a model selector across all 44 Universal Multi-Domain (`univ_*`) and Single-Domain (`ls_*`, `cn_*`, `vox_*`) checkpoints from [`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker) (`Cosine`, `PFAS`, and `Dr-Vectors` scoring).
- The decision threshold defaults to the selected model's **own pooled 1-utt EER operating point** from `eval_results.json` (slider range adapts to the scoring back-end); latency is reported as steady-state model inference after a JIT warm-up.
- 1-utterance and 3-utterance speaker enrollment with real speech samples from **LibriSpeech**, **CN-Celeb2**, and **VoxCeleb1**; waveform + VAD gate, post-VAD log-Mel spectrogram, and embedding-profile plots.
- AS-Norm is deliberately not offered (a web demo has no disjoint impostor cohort). `hf_space/` is self-contained (`requirements.txt` installs `flaxspeaker`).

```bash
python hf_space/app.py
```

---

## 44-Model Benchmark Summary & Pretrained Model Zoo

We trained and evaluated **44 models** (`19` Universal Multi-Domain models trained jointly on `LibriSpeech + CN-Celeb2 + VoxCeleb1` = `381` speakers / `9,447` utterances, plus `25` Single-Domain baseline models) across **all three speaker-disjoint evaluation domains** (`8,000` 1-utterance trials + `3,000` 3-utterance trials + `2,500` VoxCeleb1-O-subset trials per model = `594,000` total trial evaluations). **Every model: 400 steps × 32 utterances, seed 42, single run** — see the seed-noise floor below before reading any sub-1 % difference as a ranking.

- **Hugging Face Collection (44 Pretrained Models)**: **[`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker)**
- Full technical report, ablations & mathematical analysis: **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)**
- Complete verified evaluation metrics JSON (`44` models + `12` TFLite exports): **[`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json)**
- Controlled ablation study metrics JSON (`5` studies): **[`pretrained_models/ablation_results.json`](pretrained_models/ablation_results.json)**
- VoxCeleb1-O **10-speaker subset** evaluation JSON (`44` models, 26 dB VAD matched to training): **[`pretrained_models/vox1_o_10spk_subset_results.json`](pretrained_models/vox1_o_10spk_subset_results.json)**
- Round-2 controls (seed-noise floor over seeds {0, 1, 2}, honest 800-step horizon point, matched-data control): **[`pretrained_models/r2_seed_and_control_results.json`](pretrained_models/r2_seed_and_control_results.json)**
- Local Pretrained HuggingFace + SafeTensors + TFLite checkpoints: **[`pretrained_models/`](pretrained_models/)**

### 1. Original Published Paper Reference Results (Internal Proprietary Datasets)

| Paper & Venue | Architecture & Loss / Scoring | Original Training Corpus | Original Evaluation Benchmark | Reported Metric in Original Paper |
| :--- | :--- | :--- | :--- | :--- |
| **GE2E** ([Wan et al., ICASSP 2018](https://arxiv.org/pdf/1710.10467)) | 3-layer LSTM (`768` hidden, `256` proj) + `GE2E-Softmax` | ~36 M utterances from ~`18K` speakers (anonymised logs) | Internal TI-SV: 1,000 speakers, ≈6.3 enrollment / 7.2 test utterances per speaker | `3.55%` EER (GE2E) vs. `4.06%` (softmax) and `4.13%` (TE2E), Table 3 — multi-utterance enrollment, not comparable to our 1-utt numbers |
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
| [`wq2012/univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast) | `conformer` (`small`) | `atp` / `ge2e_contrast` / `cosine` | `2,029,894` | 14.73% / **8.30%** | 23.27% / **17.10%** | 24.45% / 19.40% | 20.82% / **14.93%** | 20.69% / 15.33% | `0.9396` / `0.8613` |
| [`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e) | `lstm` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,710,914` | 14.97% / 11.20% | 25.53% / 21.60% | 22.50% / 19.50% | 21.00% / 17.43% | 21.31% / 18.00% | `0.9620` / `0.9173` |
| [`wq2012/univ_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_base_ge2e) | `ecapa_tdnn` (`base`) | `asp` / `ge2e_softmax` / `cosine` | `3,190,626` | 18.63% / 12.20% | 25.97% / 21.20% | **22.20%** / 17.30% | 22.27% / 16.90% | 22.81% / 18.20% | `0.9760` / `0.9340` |
| [`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e) | `ecapa_tdnn` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,684,418` | 19.00% / 11.80% | 25.13% / 22.40% | 23.30% / 18.80% | 22.48% / 17.67% | 23.16% / 18.00% | `0.9889` / `0.9120` |
| [`wq2012/univ_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_tiny_ge2e) | `ecapa_tdnn` (`tiny`) | `asp` / `ge2e_softmax` / `cosine` | `626,722` | 19.07% / 12.50% | 25.27% / 21.40% | 23.35% / 17.20% | 22.56% / 17.03% | 23.02% / 18.00% | `0.9613` / `0.8933` |
| [`wq2012/univ_conformer_small_triplet`](https://huggingface.co/wq2012/univ_conformer_small_triplet) | `conformer` (`small`) | `atp` / `triplet` / `cosine` | `2,029,892` | 18.50% / 14.40% | 26.67% / 24.00% | 25.40% / 22.00% | 23.52% / 20.13% | 24.07% / 20.33% | `0.9477` / `0.9247` |
| [`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface) | `conformer` (`small`) | `atp` / `arcface` / `cosine` | `2,029,892` | 20.57% / 17.40% | 25.27% / 21.40% | 25.00% / 21.80% | 23.61% / 20.20% | 25.11% / 21.67% | `0.9869` / `0.9160` |
| [`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface) | `conformer` (`small`) | `atp` / `cosface` / `cosine` | `2,029,892` | 20.37% / 17.10% | 25.73% / 22.20% | 26.40% / 20.30% | 24.17% / 19.87% | 25.60% / 21.83% | `0.9914` / `0.9193` |
| [`wq2012/univ_conformer_small_sphereface`](https://huggingface.co/wq2012/univ_conformer_small_sphereface) | `conformer` (`small`) | `atp` / `sphereface` / `cosine` | `2,029,892` | 22.27% / 18.10% | 25.67% / 23.40% | 25.15% / 20.40% | 24.36% / 20.63% | 25.84% / 22.07% | `0.9858` / `0.9347` |
| [`wq2012/univ_resnet_small_ge2e`](https://huggingface.co/wq2012/univ_resnet_small_ge2e) | `resnet` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `2,239,922` | 24.53% / 21.40% | 29.33% / 26.40% | 28.25% / 25.80% | 27.37% / 24.53% | 27.89% / 25.47% | `0.9987` / `0.9847` |

### 3. Universal Multi-Domain vs. Single-Domain Training (matched Conformer-Small / GE2E)

Universal training lowers *pooled* 1-utt EER relative to the matched LibriSpeech specialist in 11 of the 19 `univ_*` vs. `ls_*` pairs (every Conformer metric-learning objective incl. the contrast surrogate, Dr-Vector, PFAS-SN and BXS variants, plus Transformer, Mamba and ECAPA-TDNN-base with GE2E) and raises it in the other 8 (LSTM, ECAPA-TDNN-small/tiny, ResNet-18, and the Conformer ArcFace/CosFace/SphereFace/triplet models); it is worse on *macro* 1-utt EER in 8 of 19 pairs (Transformer, ECAPA-TDNN-small/tiny, ResNet-18, ArcFace, CosFace, SphereFace, triplet). It never wins in-domain (the LibriSpeech specialist is 9.67 % vs. 12.50 % on LibriSpeech; the VoxCeleb1 specialist 20.80 % vs. 24.80 % on VoxCeleb1). The comparison also confounds domain diversity with 1.6× more training data (9,447 vs. 5,788 utterances); see the matched-data control in `r2_seed_and_control_results.json`. All of these are single-seed differences; see the seed-noise floor in [RESEARCH_REPORT.md §10](RESEARCH_REPORT.md).

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
| [`wq2012/ls_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/ls_conformer_small_ge2e_contrast) | `librispeech` | `conformer` (`small`) | `atp` / `ge2e_contrast` / `cosine` | `2,029,894` | 11.43% / 7.10% | 29.57% / 25.10% | 25.75% / 20.40% | 22.25% / 17.53% | 22.80% / 19.07% |
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

## Deviations from the Source Papers

The released checkpoints implement **FlaxSpeaker variants** of the cited algorithms. The paper-faithful configurations are available as presets (`icassp2018_ge2e_config`, `interspeech2021_dr_vectors_config`, `odyssey2022_conformer_atp_config`, `odyssey2022_pfas_config` with `pure_paper_mode=True`; every field they set is consumed by the built module and covered by `tests.py`), but **no paper-faithful preset has been trained or released**, so the table below describes what was actually run.

| Component | Source paper | Released FlaxSpeaker checkpoints | Paper-faithful preset |
| :--- | :--- | :--- | :--- |
| Training budget / data | GE2E: ~36 M utts, 18 K spk, N=64×M=10; Dr-Vectors/PFAS: 16 spk × 8 utts, 500 k steps | 400 steps × (8 spk × 4 utts); ≤ 25 utts/spk; 381 (universal) / 232 / 119 / 30 speakers; single seed | not available (single CPU workstation) |
| GE2E encoder | 3 × LSTM-768 + 256-D projection, linear output | `ls_lstm_*` / `univ_lstm_*`: 3 × LSTM-256, no projection, per-layer LayerNorm + residual, ASP pooling, nonlinear bottleneck FFN | `icassp2018_ge2e_config(pure_paper_mode=True)` |
| GE2E contrast loss | Eq. 7 | `*_ge2e_contrast`: **hybrid contrast surrogate** (value = Eq. 7; gradient = 0.25 Eq. 7 + 0.25 BCE + 0.50 softmax-CE), `ge2e_init_b = −1`, `b` gradient scale 0.5 | `ge2e_contrast_surrogate: false` (default) |
| Extended-set softmax | Eq. 2 (pure) | `*_ext_softmax`, `*_dr_vectors`, `*_pfas`: **blended** 0.5 row-softmax + 0.5 extended-set with `−log N` shift | `pure_extended_set: true` |
| Dr-Vector scorer | 200 base + 56 side dims; 3 × 256 leaky-ReLU(0.2); weighted sum; trainable affine | `*_dr_vectors`: cosine over all 192 dims, hidden (128, 64) ReLU + LayerNorm, linear read-out, fixed 0.25 residual weight. The 3-term `[c, c², √max(c,0)]` expansion is a FlaxSpeaker extension (ablation only) | `interspeech2021_dr_vectors_config(pure_paper_mode=True)` |
| ATP pooling | single head, linear-sigmoid weights, full-D mean/std | `attention_dim: 128` tanh bottleneck, 4 channel-subspace heads (`subspace_heads: true`); source paper is a language-ID paper | `attention_dim: 0, num_heads: 1` |
| PFAS head | ATP → affine + ReLU(512) → linear; 32 tied keys (2,048-D) | `*_pfas*`: **PFAS-SN** — per-frame key/value projections, multi-head tanh frame attention + global ATP anchor, 4 keys × (32 + 128) | `odyssey2022_pfas_config(pure_paper_mode=True)` (32 × (16 + 48); dims inferred from the paper's totals) |
| Conformer | 12 × 512, 4–8 heads, kernel 32, relative PE, dropout, stack-by-2 | 2–4 blocks × 128–256, 4–8 heads, kernel 15, no PE, GroupNorm(2) instead of BatchNorm, conv stem + MFA, no dropout | `odyssey2022_conformer_atp_config(pure_paper_mode=True)` (conv-before-MHSA, sinusoidal PE, stack-by-2 after block 3) |
| ECAPA-TDNN / SphereFace | BatchNorm / integer `m = 4` | LayerNorm / real-valued `m = 1.35` with λ-annealing | — |
| Augmentation | waveform MTR with recorded noise + room simulator | synthetic log-Mel noise (SNR 12–20 dB) + one-tap smoothing | — |
| Evaluation | VoxCeleb1-O 37,720 trials over 40 speakers after VoxCeleb1-dev / VoxCeleb2 training | 10-speaker subset of `veri_test2.txt` (2,500 trials, 72 % target); 30/40 official test speakers used for training | — |

---

## Testing, Linting & Documentation

Run the unit test suite (covering all backbones, exact padding-mask invariance on all seven backbone variants including bidirectional LSTM, losses including pure-vs-surrogate `ge2e_contrast` gradients and pure/blended `extended_set_softmax`, pooling heads, VAD + AGC, mutually exclusive augmentation, Dr-Vectors paper options, every `pure_paper_mode` preset field reaching the built module, strict config-key validation, AS-Norm cohort disjointness, export-window resolution, HF compatibility, and TFLite export):

```bash
pytest -v tests.py
flake8 .
./run_pdoc.sh
```
