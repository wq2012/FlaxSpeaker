# FlaxSpeaker

[![Python application](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml/badge.svg)](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml) [![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-44%20Pretrained%20Models-blue)](https://huggingface.co/collections/wq2012/flaxspeaker) [![PyPI Version](https://img.shields.io/pypi/v/flaxspeaker.svg)](https://pypi.python.org/pypi/flaxspeaker) [![Python Versions](https://img.shields.io/pypi/pyversions/flaxspeaker.svg)](https://pypi.org/project/flaxspeaker) [![Downloads](https://static.pepy.tech/badge/flaxspeaker)](https://pepy.tech/project/flaxspeaker) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**FlaxSpeaker** (`v0.4.0`) is an end-to-end open-source speaker verification library built on [JAX](https://jax.readthedocs.io) and [Flax](https://flax.readthedocs.io). Designed for education and academic research, it unifies dataset preparation, frame-level energy Voice Activity Detection (VAD) with peak Automatic Gain Control (AGC), an 80-D log-Mel frontend with CMVN, six neural sequence backbones, eight temporal pooling heads, nine metric-learning and angular-margin training objectives (seven of which are released in the 44-model zoo), three scoring paradigms, universal multi-domain stratified training, and one-command export to **TFLite (FP32 / dynamic-range INT8)**, **TensorFlow SavedModel**, **SafeTensors**, and the **Hugging Face Hub**.

> **Scope & Corpus-Scale Notice (`v0.4.0`).** All 44 published checkpoints in `v0.4.0` are **converged CPU research checkpoints** trained to calibrated `S_safe = X + 2` epoch step budgets (`4,600` steps / `15.58` epochs on Universal; `2,400–2,600` steps / `13.27–14.37` epochs on LibriSpeech; `1,600–1,800` steps / `17.60–19.80` epochs on CN-Celeb2; `1,000` steps / `42.67` epochs on VoxCeleb1; `32` utterances per step, `N = 8 × M = 4`) with uniform variable-length masked crops `T ~ U(90, 180)` (`variable_length_training: true` across all 44 models, verified to `5e-5` step-0 loss tolerance in `pretrained_models/step0_fingerprints.json`: `42` `variable` + `2` crop-invariant `both`, `0` `fixed`, `0` `neither`) and linear angular-margin warm-up (`margin_warmup_steps`). Because the training corpora intentionally fit on a single workstation (`≤ 381` speakers, `≤ 9,447` utterances, `≤ 25` utterances/speaker), the 44 models reach **`14.60–22.35%`** pooled 1-utt EER (`14.60–19.25%` across the 19 `univ_*` models; `10.40–19.67%` pooled 3-utt EER) on the speaker-disjoint multi-domain trial lists and **`19.15–29.96%`** EER on a *10-speaker subset* of VoxCeleb1-O (**not comparable to published VoxCeleb1-O systems** trained on 7,000+ VoxCeleb2 speakers, which reach `~1%` EER). Several released configurations are **FlaxSpeaker variants** of the cited algorithms (see [Deviations from the Source Papers](#deviations-from-the-source-papers)); all four paper-faithful `pure_paper_mode=True` presets are also trained for `4,600` steps and benchmarked in `pretrained_models/ablation_results.json` (`studies.paper_faithful_presets_smoke_test`). Multi-seed stability across seeds `{0, 1, 2}` has a pooled 1-utt EER standard deviation of **`0.08%`** (`univ_conformer_small_ge2e`), **`0.14%`** (`univ_ecapa_tdnn_small_ge2e`), and **`0.41%`** (`ls_conformer_small_ge2e`; see [RESEARCH_REPORT.md §10](RESEARCH_REPORT.md)).

For the companion PyTorch educational project, see [SpeakerRecognitionFromScratch](https://github.com/wq2012/SpeakerRecognitionFromScratch).

---

## Disclaimer & Open-Source Reproduction Notice

**This is not an officially supported Google product.**

This repository provides a from-scratch re-implementation in **JAX / Flax**, written using only the published papers (no proprietary code or data), of the mathematical formulations and neural architectures described in the following research papers:
1. **Generalized End-to-End Loss for Speaker Verification** — *Li Wan, Quan Wang, Alan Papir, Ignacio Lopez Moreno* ([ICASSP 2018, arXiv:1710.10467](https://arxiv.org/pdf/1710.10467))
2. **Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition** — *Jason Pelecanos, Quan Wang, Ignacio Lopez Moreno* ([Interspeech 2021, arXiv:2104.01989](https://arxiv.org/pdf/2104.01989))
3. **Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech** — *Quan Wang, Yang Yu, Jason Pelecanos, Yiling Huang, Ignacio Lopez Moreno* ([Odyssey 2022, arXiv:2202.12163](https://arxiv.org/pdf/2202.12163)) — a *language identification* paper whose attentive temporal pooling head is repurposed here for speaker embeddings
4. **Parameter-Free Attentive Scoring for Speaker Verification** — *Jason Pelecanos, Quan Wang, Yiling Huang, Ignacio Lopez Moreno* ([Odyssey 2022, arXiv:2203.05642](https://arxiv.org/pdf/2203.05642))

Because the original papers were trained and evaluated on proprietary internal datasets and infrastructure (for `~500k` steps on thousands of speakers), **exact numerical parity with the original paper tables is neither expected nor claimed**, and the `≤ 381`-speaker checkpoints should not be read as reproductions of the papers' large-corpus numbers. All pretrained checkpoints and benchmark tables in this repository are trained and evaluated from scratch on publicly available speech corpora (**LibriSpeech**, **CN-Celeb2**, and the 40-speaker **VoxCeleb1 test partition**) with speaker-disjoint splits.

---

## System Architecture

![FlaxSpeaker System Architecture](resources/architecture_diagram.png)

### 1. Six Neural Sequence Backbones & Four Size Presets (`0.63 M–4.55 M` parameters in the released zoo; `large` is defined for custom runs)
All backbones share a unified `(B, T, F) -> (B, T, D)` frame-rate-preserving sequence interface with **exact variable-length padding mask invariance** via `ModernSpeakerEncoder` ([`flaxspeaker/backbones.py`](flaxspeaker/backbones.py)) and scale automatically via `size_variant: tiny | small | base | large`:
- **Conformer** (`conformer`): Parallel Linear + 1D Conv Stem (`Dense(D) + SiLU(LayerNorm(Conv1D(k=3)))`) + Macaron FFN + Masked Multi-Head Self-Attention + Mask-Aware GroupNorm Depthwise Separable Convolution + Multi-Layer Feature Aggregation (MFA).
- **ECAPA-TDNN** (`ecapa_tdnn`): 1D SE-Res2Net dilated multi-scale blocks with Multi-Layer Feature Aggregation (MFA) and configurable normalization order (`norm_after_activation`).
- **Mamba** (`mamba`): Selective State Space Model (SSM) implemented in pure JAX via parallel `jax.lax.associative_scan` (`O(T)` linear time complexity, supporting both causal unidirectional and mask-aware bidirectional `bidirectional: true` modes).
- **ResNet** (`resnet`): 2D Time-Frequency Residual Network with Squeeze-and-Excitation (SE-ResNet-18, 8 residual blocks across 4 stages `(2, 2, 2, 2)`).
- **Transformer** (`transformer`): Parallel Linear + 1D Conv Stem (`Dense(D) + SiLU(LayerNorm(Conv1D(k=3)))`) + Pre-LayerNorm Multi-Head Self-Attention encoder with Multi-Layer Feature Aggregation (MFA).
- **LSTM** (`lstm`): Deep stacked uni- or bidirectional LSTM (`bidirectional: true`, mask-aware with `seq_lengths` in both directions) with configurable hidden size, optional per-layer projection, and per-layer LayerNorm + residual connections.

### 2. Eight Temporal Pooling Heads ([`flaxspeaker/pooling.py`](flaxspeaker/pooling.py))
- **Attentive Temporal Pooling** (`atp`): Sigmoid frame-gated statistics pooling (`concat(mu_h, sigma_h)`) extended from the single-head linear-sigmoid ATP of [Wang et al., Odyssey 2022, `arXiv:2202.12163`](https://arxiv.org/pdf/2202.12163) (a language-identification paper). FlaxSpeaker defaults add a tanh bottleneck (`attention_dim: 128`) and 4 channel-subspace heads (`subspace_heads: true`); `attention_dim: 0, num_heads: 1` recovers the paper form. Followed by a nonlinear bottleneck FFN (`LayerNorm -> Dense(256) -> LayerNorm -> ReLU -> Dense(d_e) -> L2_norm`).
- **Self-Attentive Pooling** (`sap`) & **Attentive Statistics Pooling** (`asp`): Softmax attention-weighted frame mean and standard deviation.
- **Mean Pooling** (`mean`), **Last-Frame Pooling** (`last`), & **Temporal Statistics Pooling** (`stats`).
- **Cumulative Statistics Pooling** (`cumulative_stats`): Frame-wise causal running statistics for streaming inference.
- **Parameter-Free Attentive Scoring Head** (`pfas`): Supports `attention_type: atp_ffn` (the head of [Pelecanos et al., Odyssey 2022, `arXiv:2203.05642`](https://arxiv.org/pdf/2203.05642): ATP → affine + ReLU → linear → keys/values; `odyssey2022_pfas_config(pure_paper_mode=True)` uses 32 tied keys × `(16 + 48)` = `2,048-D`), as well as `attention_type: shared_nonlinear | dot | maxpool` + `use_global_anchor: true` (**PFAS-SN, a FlaxSpeaker extension used by all released `*_pfas*` checkpoints**: per-frame key/value projections with multi-head tanh or scaled dot-product frame attention anchored on a global ATP value summary, 4 keys × `(32 + 128)` = `640-D`). All variants apply per-key L2 normalisation and global L2 normalisation of the concatenated value form (paper Eqs. 5–7).

### 3. Nine Speaker Recognition Losses & Three Scoring Backends ([`flaxspeaker/losses.py`](flaxspeaker/losses.py), [`flaxspeaker/scoring.py`](flaxspeaker/scoring.py))
- **Generalized End-to-End (GE2E) Loss**: `ge2e_softmax` (Eq. 6, exact leave-one-out centroids) and `ge2e_contrast` (Eq. 7) with learnable scale `w` and bias `b` ([Wan et al., ICASSP 2018](https://arxiv.org/pdf/1710.10467)). By default `ge2e_contrast` back-propagates Eq. 7 exactly; the released `*_ge2e_contrast` checkpoints use the **hybrid contrast surrogate** (`ge2e_contrast_surrogate: true`: forward value = Eq. 7, gradient = `0.25·Eq7 + 0.25·BCE + 0.50·GE2E-softmax-CE`, `ge2e_init_b: -1.0`, `ge2e_contrast_b_grad_scale: 0.5`). When `scoring_type` is `pfas` or `dr_vector`, GE2E uses the even/odd split-batch test-vs-enrollment matrix.
- **Extended-Set Softmax Loss**: `extended_set_softmax`, supporting both the paper's pure form (`pure_extended_set: true`, Eq. 2 of [Pelecanos et al., Interspeech 2021](https://arxiv.org/pdf/2104.01989)) and the **Blended Extended-Set Softmax (BXS, FlaxSpeaker variant)** (`pure_extended_set: false`: 50/50 blend of row-softmax and extended-set softmax with a `-log(N)` logit shift).
- **Decision Residual Networks (Dr-Vectors)**: `scoring_type: dr_vector` — single-stage end-to-end residual feed-forward scorer on `[e_t, e_e, c]`. `interspeech2021_dr_vectors_config(pure_paper_mode=True)` builds the paper's network (`256-D` embedding split into `cosine_dims: 200` base + `56` side dims, `3 × Dense(256)` leaky-ReLU(0.2) without normalisation, `readout: weighted_sum`, `trainable_affine: true`, pure extended-set softmax, reaching `15.47%` pooled 1-utt EER at 4,600 steps in `ablation_results.json`). The released `*_dr_vectors` checkpoints use the compact **down-scaled Dr-Vector variant** (192-D cosine, hidden `(128, 64)` ReLU + LayerNorm, linear read-out, `residual_weight: 0.25`, blended extended-set softmax, `15.88%` pooled 1-utt EER).
- **Parameter-Free Attentive Scoring (PFAS)**: `scoring_type: pfas` — cross-attention between test and enrollment `(K, V)` representations with per-trial softmax key–key attention weights `A_{ij}`, per-key and global L2 normalisation (Eqs. 5–7 of [Pelecanos et al., Odyssey 2022](https://arxiv.org/pdf/2203.05642)), trainable scale, and key stacking (`N = M × E`) for multi-utterance enrollment.
- **Angular Margin Classification Losses (with Linear Margin Warm-Up)**:
  - **ArcFace / AAM-Softmax** (`arcface`): `cos(theta + m_t)` with linear margin ramp `m_t = m * min(1, t / margin_warmup_steps)` (`margin_warmup_steps = 500` on Universal, `300` on LibriSpeech, `250` on CN-Celeb2)
  - **CosFace / AM-Softmax** (`cosface`): `cos(theta) - m_t` with linear margin ramp `margin_warmup_steps`
  - **SphereFace / A-Softmax** (`sphereface`): `(-1)^k * cos(m * theta) - 2k` with $\lambda$-annealing (`1000 -> 5`)
  - **Standard Softmax** (`softmax`), **Pairwise Loss** (`pairwise`), and **Triplet Loss** (`triplet`). (`softmax` and `pairwise` are implemented and unit-tested; 7 objectives appear in the 44-model zoo.)
- **Adaptive Symmetric Score Normalization (AS-Norm)**: Top-`K` impostor-cohort z-normalisation (`use_as_norm: true`) with enforced trial-cohort disjointness. No reported benchmark number uses AS-Norm.

### 4. End-to-End Audio Pipeline, Mutually Exclusive Augmentation, Variable-Length Training & Model Export
- **Frame-level energy VAD & AGC** ([`flaxspeaker/vad.py`](flaxspeaker/vad.py)): Frame energies are computed on the peak-normalised waveform (`apply_agc = True`, target peak `1.0`) and the resulting mask is applied to the log-Mel frames (`max(-42.0 dB, P_95(E) - 26.0 dB, P_10(E) + 4.5 dB)`, hangover smoothing `min_speech_frames = 3`, `hangover_frames = 8`, top-8-frame fallback). **One VAD constant (`26 dB`) is used across data preparation, training, in-domain evaluation, and the VoxCeleb1-O-subset evaluation.**
- **Acoustic Frontends & Variable-Length Training** ([`flaxspeaker/feature_extraction.py`](flaxspeaker/feature_extraction.py)): `log_mel` (natural-log Mel + energy VAD + CMVN; **not** reproducible with a Hugging Face `AutoFeatureExtractor`, so no `preprocessor_config.json` is shipped), optional `whisper_mel` / `hf_compat` / `mfcc` frontends, **per-utterance variable sequence length sampling `T ~ U(90, 180)` frames across all 44 `v0.4.0` checkpoints** (`variable_length_training: true`) with exact binary padding masks `(B, T)`, evaluation centre-cropping up to `max_eval_seq_len = 500` frames (`5.0 s`), and two-tier RAM + `.npy` disk caching (`FeatureCache`).
- **Mutually Exclusive SpecAugment & Feature-Domain Noise Augmentation** ([`flaxspeaker/specaug.py`](flaxspeaker/specaug.py)): Strictly mutually exclusive per training utterance: `35%` synthetic log-Mel noise (`SNR ~ U(12, 20) dB` + one-tap temporal smoothing with probability `0.4`), `32.5%` `SpecAugment` (`<= 2` frequency masks `<= 8` bins, `<= 2` time masks `<= 12` frames), and `32.5%` clean features.
- **Universal Multi-Domain Stratified Batching** ([`flaxspeaker/feature_extraction.py`](flaxspeaker/feature_extraction.py)): Combines `LibriSpeech`, `CN-Celeb2`, and `VoxCeleb1` using `75%` within-domain hard-negative batches and `25%` cross-domain score-calibration batches with temperature-balanced domain sampling `p_d ∝ |S_d|^0.35`.
- **Model Export** ([`flaxspeaker/export.py`](flaxspeaker/export.py), [`flaxspeaker/hf_compat.py`](flaxspeaker/hf_compat.py)): Export via `jax2tf` to **TFLite (FP32, dynamic-range INT8)** and **TF SavedModel** with a fixed, unmasked `180`-frame (`1.8 s`) encoder window, and to **SafeTensors** / **Hugging Face Hub** (`config.json`, `model.safetensors`, `flax_model.msgpack`, `eval_results.json`, model card). Across all 12 exported `.tflite` files (`scripts/eval_tflite_models.py`), dynamic-range INT8 quantization reduces file size by **`3.44–3.61×`** (`6.85–8.25 MB` → `1.90–2.33 MB`) while changing 180-frame verification EER by at most **`0.00–0.10` percentage points** relative to FP32 TFLite (`17.40% → 17.30%`, `18.15% → 18.24%`, `18.25% → 18.25%` on `universal_pooled`; `8.53% → 8.60%`, `11.47% → 11.47%`, `10.53% → 10.47%` on `librispeech`), with mean utterance-level cosine similarity `cos(FP32, INT8)` between **`0.999935` and `0.999972`** across all 1,988 unique evaluation utterances and a **`1.8–1.9×` CPU speedup** on memory-bandwidth-bound LSTM models (see [RESEARCH_REPORT.md §11](RESEARCH_REPORT.md)).

---

## Installation

```bash
pip install -U flaxspeaker   # >= 0.4.0 recommended for the converged v0.4.0 model zoo
```

`flaxspeaker >= 0.3.0` is required to load Hub checkpoints (`0.2.0` predated the `load_pretrained` / `encode_features` API), and `flaxspeaker >= 0.4.0` includes the `margin_warmup_steps` schedule, `"dot"` PFAS attention, and converged `v0.4.0` model cards and defaults. To install from source: `pip install git+https://github.com/wq2012/FlaxSpeaker`.

For development, HuggingFace export, and TFLite conversion extras:
```bash
pip install -e ".[dev,export,hf]"
```

---

## Quickstart & CLI Workflows

FlaxSpeaker provides standalone command-line scripts in [`scripts/`](scripts/) covering dataset preparation, pilot calibration, training, evaluation, ablations, figure/table generation, and Hub deployment (in addition to `python -m flaxspeaker`):

### 1. Prepare Datasets & Multi-Domain Splits (`scripts/prepare_datasets.py` / `scripts/prepare_data.py`)
```bash
# Prepare all three academic datasets + combined Universal Multi-Domain split
python scripts/prepare_datasets.py --data_root ./data/speaker_datasets

# Or generate a synthetic smoke-test dataset split for local testing
python scripts/prepare_data.py --data_root ./data/synthetic --generate_synthetic
```

### 2. Train Models (`scripts/train.py`)
Train from any converged YAML config (`configs/*.yml`, preconfigured with its calibrated `S_safe` step budget and `warmup_steps`) or from a built-in paper preset (`universal_multi_domain`, `icassp2018_ge2e`, `interspeech2021_dr_vectors`, `odyssey2022_conformer_atp`, `odyssey2022_pfas`):

```bash
# Train the top Universal Multi-Domain Conformer + ATP + ArcFace model (4,600 steps, margin_warmup_steps=500)
python scripts/train.py \
  --config configs/univ_conformer_small_arcface.yml \
  --train_csv ./data/speaker_datasets/splits/universal/train.csv \
  --eval_csv ./data/speaker_datasets/splits/universal/eval.csv \
  --trials_file ./data/speaker_datasets/splits/universal/trials.tsv \
  --output_dir ./checkpoints/univ_conformer_small_arcface
```

### 3. Evaluate 1-Utt, 3-Utt, and VoxCeleb1-O-subset Trials (`scripts/evaluate.py` / `scripts/eval_vox1_o_subset.py`)
```bash
python scripts/evaluate.py \
  --model_dir wq2012/univ_conformer_small_arcface \
  --trials_file ./data/speaker_datasets/splits/universal/trials.tsv \
  --multi_trials_file ./data/speaker_datasets/splits/universal/multi_enroll_trials.tsv \
  --output_json eval_summary.json

# Evaluate all 44 models on the VoxCeleb1-O 10-speaker subset (writes pretrained_models/vox1_o_10spk_subset_results.json)
python scripts/eval_vox1_o_subset.py --data_root ./data/speaker_datasets --models_dir ./pretrained_models
```

### 4. Run End-to-End Speaker Verification Inference (`scripts/inference.py`)
```bash
python scripts/inference.py \
  --model_dir wq2012/univ_conformer_small_arcface \
  --enroll_audio hf_space/examples/librispeech_spk1_utt1.flac \
  --test_audio hf_space/examples/librispeech_spk1_utt2.flac \
  --apply_vad --vad_top_db 26.0 --full_sequence
```

### 5. Export to TFLite & Publish to the Hugging Face Hub (`scripts/export_model.py` / `scripts/eval_tflite_models.py` / `scripts/update_hf_model_cards.py`)
`export_model.py` writes a Hub-format directory (`config.json`, `model.safetensors`, `flax_model.msgpack`, model card, optional `.tflite` files) locally. `scripts/update_hf_model_cards.py --push --include_weights` regenerates all 44 model cards and uploads weights, TFLite binaries, `config.json`, `eval_results.json`, and `README.md` to `wq2012/<model_id>`. `scripts/deploy_hf_space.py` deploys the Gradio demo to a Hugging Face Space.
```bash
python scripts/export_model.py \
  --model_dir wq2012/univ_conformer_small_ge2e \
  --output_dir ./exported_models/univ_conformer_small_ge2e \
  --export_tflite \
  --quantize_int8

# Evaluate exported FP32 and INT8 TFLite models on the 1-utterance trial lists
python scripts/eval_tflite_models.py --data_root ./data/speaker_datasets --models_dir ./pretrained_models

# Regenerate all 44 model cards and push weights + TFLite + metadata to Hugging Face Hub
python scripts/update_hf_model_cards.py --push --include_weights
```

### 6. Reproduce Pilot Calibration, Full 44-Model Benchmark Suite, Ablations, Provenance & Paper Figures
```bash
# Phase-1 5,000-step convergence pilot & S_safe = X + 2 epoch schedule calibration
python scripts/run_phase1_convergence_pilot.py --data_root ./data/speaker_datasets

# Run the complete 44-model converged benchmark matrix across all domains
python scripts/run_research_benchmarks.py --data_root ./data/speaker_datasets

# Reproduce controlled ablation studies + pure_paper_mode=True 4,600-step runs (ablation_results.json)
python scripts/run_r1_ablations.py --data_root ./data/speaker_datasets

# Multi-seed stability (3 seeds x 3 models), own-schedule horizons (400..4,600), matched-data controls
python scripts/run_r2_controls.py --data_root ./data/speaker_datasets

# Step-0 loss fingerprint of all 44 released checkpoints & training_provenance blocks
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

# 1. Load a converged v0.4.0 universal multi-domain checkpoint from Hugging Face Hub or local path
verifier = flaxspeaker.load_pretrained(
    "wq2012/univ_conformer_small_arcface"
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
    loss_type="arcface",
    scoring_type="cosine",
)
```

---

## Interactive Gradio Demo (`hf_space/`)

An interactive [Gradio 5.50.0](https://gradio.app) web application is included in [`hf_space/`](hf_space/). **Hosting status**: creating a Gradio Space under `wq2012` on the Hugging Face Hub returns HTTP 402 (free CPU Spaces require a paid PRO subscription on that account), so run it locally (`python hf_space/app.py`) or deploy it to an account with Space creation enabled via `python scripts/deploy_hf_space.py --repo_id <your-namespace>/FlaxSpeaker`. Features:
- Real inference only (no precomputed outputs) across all 44 converged `v0.4.0` Universal (`univ_*`) and Single-Domain (`ls_*`, `cn_*`, `vox_*`) checkpoints from [`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker) (`Cosine`, `PFAS`, and `Dr-Vectors` scoring).
- Automatic operating-point selection from each model's `eval_results.json` for 1-utterance vs. multi-utterance enrollment (e.g., `0.384` / `0.474` on `univ_conformer_small_arcface`, `0.501` / `0.583` on `univ_conformer_tiny_ge2e`), configurable VAD thresholds (`--vad_snr_margin_db`, `--vad_top_db`), and steady-state JIT latency measurement.
- 1-utterance and 3-utterance enrollment with speech samples from **LibriSpeech**, **CN-Celeb2**, and **VoxCeleb1**, plus waveform + VAD gate, post-VAD log-Mel spectrogram, and embedding-profile visualizations.

```bash
python hf_space/app.py
```

---

## 44-Model Benchmark Summary & Pretrained Model Zoo (`v0.4.0`)

![FlaxSpeaker v0.4.0 Benchmark Summary](resources/benchmark_charts.png)

We trained and evaluated **44 converged models** (`19` Universal Multi-Domain models trained jointly on `LibriSpeech + CN-Celeb2 + VoxCeleb1` = `381` speakers / `9,447` utterances for `4,600` steps, plus `25` Single-Domain specialist models trained for `2,400–2,600` steps on LibriSpeech, `1,600–1,800` steps on CN-Celeb2, and `1,000` steps on VoxCeleb1; total training compute `40.69` CPU hours, `5.1–311.5` min/model, median `30.1` min) across **all three speaker-disjoint evaluation domains** (`8,000` 1-utterance trials + `3,000` 3-utterance trials + `2,500` VoxCeleb1-O-subset trials per model = `594,000` trial evaluations):

- **Hugging Face Collection (44 Pretrained Models)**: **[`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker)**
- Full technical report, convergence calibration, ablations & mathematical analysis: **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)**
- Complete verified evaluation metrics JSON (`44` models + `12` TFLite exports): **[`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json)**
- Phase-1 5,000-step convergence pilot & `S_safe` schedule calibration JSON: **[`pretrained_models/phase1_convergence_pilot.json`](pretrained_models/phase1_convergence_pilot.json)**
- Controlled ablation study & `pure_paper_mode=True` metrics JSON: **[`pretrained_models/ablation_results.json`](pretrained_models/ablation_results.json)**
- VoxCeleb1-O **10-speaker subset** evaluation JSON (`44` models, 26 dB VAD): **[`pretrained_models/vox1_o_10spk_subset_results.json`](pretrained_models/vox1_o_10spk_subset_results.json)**
- Multi-seed stability (`3` seeds × `3` models), own-schedule horizons (`400..4,600`), and matched-data controls (`2,400` & `4,600` steps): **[`pretrained_models/r2_seed_and_control_results.json`](pretrained_models/r2_seed_and_control_results.json)**
- Step-0 loss fingerprint of all 44 released checkpoints (`42` `variable` + `2` `both`): **[`pretrained_models/step0_fingerprints.json`](pretrained_models/step0_fingerprints.json)**
- Local Pretrained HuggingFace + SafeTensors + TFLite checkpoints: **[`pretrained_models/`](pretrained_models/)**

### 1. Original Published Paper Reference Results (Internal Proprietary Datasets)

| Paper & Venue | Architecture & Loss / Scoring | Original Training Corpus | Original Evaluation Benchmark | Reported Metric in Original Paper |
| :--- | :--- | :--- | :--- | :--- |
| **GE2E** ([Wan et al., ICASSP 2018](https://arxiv.org/pdf/1710.10467)) | 3-layer LSTM (`768` hidden, `256` proj) + `GE2E-Softmax` | ~36 M utterances from ~`18K` speakers (anonymised logs) | Internal TI-SV: 1,000 speakers, ≈6.3 enrollment / 7.2 test utterances per speaker | `3.55%` EER (GE2E) vs. `4.06%` (softmax) and `4.13%` (TE2E), Table 3 — multi-utterance enrollment, not comparable to 1-utt numbers |
| **Dr-Vectors** ([Pelecanos et al., Interspeech 2021](https://arxiv.org/pdf/2104.01989)) | LSTM + `Extended-Set Softmax` + `Dr-Vectors` Residual Scorer | Internal Proprietary Multi-Condition Corpus | Internal Multi-Condition Speaker Verification | `~4–11%` relative EER reduction over Cosine + GE2E |
| **Conformer + ATP** ([Wang et al., Odyssey 2022](https://arxiv.org/pdf/2202.12163)) | Conformer Encoder + `Attentive Temporal Pooling (ATP)` | Internal Multi-Language Speech Corpus (`65` languages) | Internal Streaming & Long-Form Identification | Significant EER & latency improvement over Mean/Last pooling |
| **PFAS** ([Pelecanos et al., Odyssey 2022](https://arxiv.org/pdf/2203.05642v3)) | Conformer + Multi-Head `(K, V)` `Parameter-Free Attentive Scoring` | Internal Proprietary Multi-Vendor Corpus | Internal 1-Utt & Multi-Utt Speaker Verification | `~8–14%` relative EER reduction over Cosine baseline |

### 2. Universal Multi-Domain Pretrained Models (`19` `univ_*` Models, `4,600` Steps / `15.58` Epochs)

All 19 `univ_*` models below are trained simultaneously on **LibriSpeech (`232` spk) + CN-Celeb2 (`119` spk) + VoxCeleb1 (`30` spk)** (`381` speakers, `9,447` utterances) for `4,600` steps with Domain-Balanced Stratified Batching (`75%` within-domain, `25%` cross-domain) and variable-length crops `T ~ U(90, 180)` (verified from [`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json)):

| Hugging Face Repository | Steps | Backbone / Size | Pooling / Loss / Scoring | Params | LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | Macro-Avg 1-Utt / 3-Utt EER (%) | Pooled 1-Utt / 3-Utt EER (%) | Macro 1-Utt / 3-Utt minDCF@0.01 |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface) | `4,600` | `conformer` (`small`) | `atp` / `arcface` / `cosine` | `2,029,892` | 5.77% / 2.80% | 18.53% / 13.70% | 19.60% / 15.80% | **14.63%** / 10.77% | **14.60%** / 10.87% | **`0.6800`** / `0.6100` |
| [`wq2012/univ_conformer_small_sphereface`](https://huggingface.co/wq2012/univ_conformer_small_sphereface) | `4,600` | `conformer` (`small`) | `atp` / `sphereface` / `cosine` | `2,029,892` | 7.07% / 3.00% | 18.40% / 13.70% | 20.10% / 14.70% | 15.19% / 10.47% | **14.60%** / 10.53% | `0.7100` / `0.6173` |
| [`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface) | `4,600` | `conformer` (`small`) | `atp` / `cosface` / `cosine` | `2,029,892` | **5.43%** / **2.50%** | 19.60% / **12.40%** | 19.30% / 14.00% | 14.78% / **9.63%** | 14.71% / **10.40%** | `0.7572` / **`0.4847`** |
| [`wq2012/univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast) | `4,600` | `conformer` (`small`) | `atp` / `ge2e_contrast` / `cosine` | `2,029,894` | 8.27% / 5.20% | 18.27% / 12.80% | 19.60% / **12.60%** | 15.38% / 10.20% | 14.95% / 10.73% | `0.7510` / `0.5720` |
| [`wq2012/univ_conformer_tiny_ge2e`](https://huggingface.co/wq2012/univ_conformer_tiny_ge2e) | `4,600` | `conformer` (`tiny`) | `atp` / `ge2e_softmax` / `cosine` | `693,126` | 8.47% / 6.00% | 18.30% / 13.20% | 19.25% / 14.80% | 15.34% / 11.33% | 15.05% / 11.33% | `0.8238` / `0.5893` |
| [`wq2012/univ_conformer_base_ge2e`](https://huggingface.co/wq2012/univ_conformer_base_ge2e) | `4,600` | `conformer` (`base`) | `atp` / `ge2e_softmax` / `cosine` | `4,551,558` | 8.17% / 6.40% | **17.97%** / 12.50% | 19.60% / 14.40% | 15.24% / 11.10% | 15.14% / 11.63% | `0.7933` / `0.6673` |
| [`wq2012/univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e) | `4,600` | `conformer` (`small`) | `pfas` / `ge2e_softmax` / `pfas` | `2,162,187` | 8.47% / 4.70% | 19.33% / 14.80% | 19.55% / 15.00% | 15.78% / 11.50% | 15.39% / 11.80% | `0.7817` / `0.6667` |
| [`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e) | `4,600` | `lstm` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,710,914` | 9.33% / 6.70% | 18.33% / 15.90% | 19.05% / 15.10% | 15.57% / 12.57% | 15.50% / 12.50% | `0.8007` / `0.7107` |
| [`wq2012/univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas) | `4,600` | `conformer` (`small`) | `pfas` / `extended_set_softmax` / `pfas` | `2,162,187` | 8.67% / 5.60% | 19.23% / 14.60% | 20.30% / 15.20% | 16.07% / 11.80% | 15.55% / 11.53% | `0.7778` / `0.6620` |
| [`wq2012/univ_resnet_small_ge2e`](https://huggingface.co/wq2012/univ_resnet_small_ge2e) | `4,600` | `resnet` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `2,239,922` | 9.23% / 6.60% | 19.10% / 14.30% | 19.85% / 15.70% | 16.06% / 12.20% | 15.60% / 12.20% | `0.8415` / `0.5407` |
| [`wq2012/univ_conformer_small_ext_softmax`](https://huggingface.co/wq2012/univ_conformer_small_ext_softmax) | `4,600` | `conformer` (`small`) | `atp` / `extended_set_softmax` / `cosine` | `2,029,894` | 8.57% / 5.20% | 19.00% / 12.90% | 19.90% / 15.80% | 15.82% / 11.30% | 15.62% / 11.63% | `0.8579` / `0.6760` |
| [`wq2012/univ_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_tiny_ge2e) | `4,600` | `ecapa_tdnn` (`tiny`) | `asp` / `ge2e_softmax` / `cosine` | `626,722` | 9.57% / 6.60% | 18.87% / 15.70% | **18.25%** / 13.00% | 15.56% / 11.77% | 15.70% / 12.63% | `0.8404` / `0.7093` |
| [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) | `4,600` | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 9.00% / 6.00% | 19.07% / 13.60% | 21.40% / 15.00% | 16.49% / 11.53% | 15.81% / 11.93% | `0.8254` / `0.6340` |
| [`wq2012/univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors) | `4,600` | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 8.67% / 5.20% | 19.03% / 13.00% | 20.30% / 15.20% | 16.00% / 11.13% | 15.88% / 11.67% | `0.8410` / `0.6980` |
| [`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e) | `4,600` | `ecapa_tdnn` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,684,418` | 10.37% / 7.60% | 18.73% / 15.10% | 19.75% / 13.70% | 16.28% / 12.13% | 16.34% / 12.87% | `0.8431` / `0.7093` |
| [`wq2012/univ_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_base_ge2e) | `4,600` | `ecapa_tdnn` (`base`) | `asp` / `ge2e_softmax` / `cosine` | `3,190,626` | 10.30% / 6.60% | 18.40% / 15.00% | 19.90% / 14.60% | 16.20% / 12.07% | 16.40% / 12.60% | `0.8390` / `0.6647` |
| [`wq2012/univ_transformer_small_ge2e`](https://huggingface.co/wq2012/univ_transformer_small_ge2e) | `4,600` | `transformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `1,239,622` | 9.47% / 6.80% | 19.43% / 14.00% | 20.35% / 15.00% | 16.42% / 11.93% | 16.40% / 12.40% | `0.8138` / `0.6687` |
| [`wq2012/univ_mamba_small_ge2e`](https://huggingface.co/wq2012/univ_mamba_small_ge2e) | `4,600` | `mamba` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,028,674` | 9.67% / 7.40% | 19.87% / 15.70% | 20.50% / 14.30% | 16.68% / 12.47% | 16.56% / 12.67% | `0.8228` / `0.6740` |
| [`wq2012/univ_conformer_small_triplet`](https://huggingface.co/wq2012/univ_conformer_small_triplet) | `4,600` | `conformer` (`small`) | `atp` / `triplet` / `cosine` | `2,029,892` | 10.73% / 6.30% | 23.27% / 18.70% | 23.95% / 20.80% | 19.32% / 15.27% | 19.25% / 15.67% | `0.9391` / `0.7653` |

### 3. Universal Multi-Domain vs. Single-Domain Training (`19 / 19` Matched Pairs Improved)

At full convergence (`S_safe`), **Universal Multi-Domain training outperforms the matched LibriSpeech specialist in all `19 / 19` `univ_*` vs. `ls_*` pairs** across **Pooled 1-Utt EER** (`-2.39` to `-5.66 pp`), **Macro-Avg 1-Utt EER** (`-2.24` to `-5.86 pp`), and **Pooled 3-Utt EER** (`-2.16` to `-6.87 pp`). Crucially, the matched-data control in `r2_seed_and_control_results.json` isolates speaker/domain diversity from utterance count: subsampling the 381-speaker Universal mix to **`5,788` utterances** (matching LibriSpeech) and training for **`2,400` steps** (matching compute) achieves **`17.04%` Pooled 1-Utt EER** (`17.28%` Macro 1-Utt, `12.87%` Pooled 3-Utt) vs. **`20.79%`** for `ls_conformer_small_ge2e` (**`-3.75 pp`** improvement at identical utterance count and step budget), and **`15.28%` Pooled 1-Utt EER** at `4,600` steps. Meanwhile, single-domain specialists retain a narrow edge strictly on their own training domain (`ls_conformer_small_arcface` `5.23%` vs. `univ_conformer_small_arcface` `5.77%` on LibriSpeech; `cn_conformer_small_ge2e` `18.97%` vs. `univ_conformer_small_ge2e` `19.07%` on CN-Celeb2; `vox_conformer_small_ge2e` `19.10%` vs. `21.40%` on VoxCeleb1) while degrading sharply out-of-domain:

| Model Paradigm | Representative Checkpoint (`Steps`) | LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | Universal Macro-Avg 1-Utt / 3-Utt EER (%) | Universal Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech-Only Specialist (`GE2E`)** | [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) (`2,400`) | 7.77% / 4.60% | 29.57% / 25.90% | 25.00% / 22.60% | 20.78% / 17.70% | 20.79% / 18.23% |
| **CN-Celeb2-Only Specialist (`GE2E`)** | [`wq2012/cn_conformer_small_ge2e`](https://huggingface.co/wq2012/cn_conformer_small_ge2e) (`1,600`) | 18.30% / 12.60% | 18.97% / 15.40% | 24.30% / 21.00% | 20.52% / 16.33% | 21.01% / 16.53% |
| **VoxCeleb1-Only Specialist (`GE2E`)** | [`wq2012/vox_conformer_small_ge2e`](https://huggingface.co/wq2012/vox_conformer_small_ge2e) (`1,000`) | 16.50% / 11.60% | 26.57% / 25.80% | **19.10%** / 14.20% | 20.72% / 17.20% | 20.84% / 16.47% |
| **Universal Multi-Domain (`GE2E`)** | [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) (`4,600`) | 9.00% / 6.00% | 19.07% / 13.60% | 21.40% / 15.00% | 16.49% / 11.53% | 15.81% / 11.93% |
| **LibriSpeech-Only Specialist (`ArcFace`)** | [`wq2012/ls_conformer_small_arcface`](https://huggingface.co/wq2012/ls_conformer_small_arcface) (`2,400`) | **5.23%** / **2.20%** | 28.80% / 23.30% | 24.00% / 19.20% | 19.34% / 14.90% | 19.48% / 15.93% |
| **Universal Multi-Domain (`ArcFace`)** | [`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface) (`4,600`) | 5.77% / 2.80% | **18.53%** / **13.70%** | 19.60% / 15.80% | **14.63%** / **10.77%** | **14.60%** / **10.87%** |

### 4. Single-Domain Pretrained Specialist Models (`25` `ls_*`, `cn_*`, `vox_*` Models)

| Hugging Face Repository | Train Domain (`Steps`) | Backbone / Size | Pooling / Loss / Scoring | Params | LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | Macro-Avg 1-Utt / 3-Utt EER (%) | Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/ls_conformer_small_cosface`](https://huggingface.co/wq2012/ls_conformer_small_cosface) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `cosface` / `cosine` | `2,029,892` | **5.17%** / 2.50% | 29.57% / 23.20% | 25.90% / 21.90% | 20.21% / 15.87% | 19.76% / 16.53% |
| [`wq2012/ls_conformer_small_arcface`](https://huggingface.co/wq2012/ls_conformer_small_arcface) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `arcface` / `cosine` | `2,029,892` | 5.23% / **2.20%** | 28.80% / 23.30% | 24.00% / 19.20% | 19.34% / 14.90% | 19.48% / 15.93% |
| [`wq2012/ls_conformer_small_sphereface`](https://huggingface.co/wq2012/ls_conformer_small_sphereface) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `sphereface` / `cosine` | `2,029,892` | 5.83% / 2.80% | 28.67% / 23.40% | 21.25% / 17.80% | 18.58% / 14.67% | **18.54%** / 15.50% |
| [`wq2012/ls_conformer_base_ge2e`](https://huggingface.co/wq2012/ls_conformer_base_ge2e) | `librispeech` (`2,600`) | `conformer` (`base`) | `atp` / `ge2e_softmax` / `cosine` | `4,551,558` | 7.40% / 4.20% | 28.87% / 24.10% | 25.75% / 21.70% | 20.67% / 16.67% | 20.10% / 17.07% |
| [`wq2012/ls_conformer_small_pfas`](https://huggingface.co/wq2012/ls_conformer_small_pfas) | `librispeech` (`2,400`) | `conformer` (`small`) | `pfas` / `extended_set_softmax` / `pfas` | `2,162,187` | 7.47% / 3.80% | 28.90% / 26.40% | 24.70% / 20.60% | 20.36% / 16.93% | 20.30% / 17.27% |
| [`wq2012/ls_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/ls_conformer_small_ge2e_contrast) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `ge2e_contrast` / `cosine` | `2,029,894` | 7.60% / 4.40% | 28.30% / 22.70% | 22.95% / 18.10% | 19.62% / 15.07% | 19.56% / 16.20% |
| [`wq2012/ls_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_pfas_ge2e) | `librispeech` (`2,400`) | `conformer` (`small`) | `pfas` / `ge2e_softmax` / `pfas` | `2,162,187` | 7.60% / 4.00% | 29.93% / 26.00% | 25.00% / 20.80% | 20.84% / 16.93% | 20.39% / 17.63% |
| [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 7.77% / 4.60% | 29.57% / 25.90% | 25.00% / 22.60% | 20.78% / 17.70% | 20.79% / 18.23% |
| [`wq2012/ls_conformer_tiny_ge2e`](https://huggingface.co/wq2012/ls_conformer_tiny_ge2e) | `librispeech` (`2,400`) | `conformer` (`tiny`) | `atp` / `ge2e_softmax` / `cosine` | `693,126` | 7.87% / 4.60% | 29.43% / 25.20% | 26.30% / 22.30% | 21.20% / 17.37% | 20.70% / 18.20% |
| [`wq2012/ls_transformer_small_ge2e`](https://huggingface.co/wq2012/ls_transformer_small_ge2e) | `librispeech` (`2,400`) | `transformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `1,239,622` | 8.23% / 5.40% | 29.87% / 25.40% | 23.05% / 19.10% | 20.38% / 16.63% | 20.55% / 17.73% |
| [`wq2012/ls_conformer_small_ext_softmax`](https://huggingface.co/wq2012/ls_conformer_small_ext_softmax) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `extended_set_softmax` / `cosine` | `2,029,894` | 8.27% / 3.80% | 28.70% / 24.90% | 24.60% / 20.20% | 20.52% / 16.30% | 20.62% / 17.30% |
| [`wq2012/ls_conformer_small_dr_vectors`](https://huggingface.co/wq2012/ls_conformer_small_dr_vectors) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 8.60% / 3.60% | 29.83% / 25.70% | 24.45% / 20.20% | 20.96% / 16.50% | 21.07% / 17.57% |
| [`wq2012/ls_mamba_small_ge2e`](https://huggingface.co/wq2012/ls_mamba_small_ge2e) | `librispeech` (`2,400`) | `mamba` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,028,674` | 8.87% / 5.10% | 28.10% / 22.70% | 22.85% / 18.30% | 19.94% / 15.37% | 20.00% / 16.70% |
| [`wq2012/ls_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_base_ge2e) | `librispeech` (`2,600`) | `ecapa_tdnn` (`base`) | `asp` / `ge2e_softmax` / `cosine` | `3,190,626` | 9.20% / 4.90% | 29.30% / 25.00% | 20.00% / 16.30% | 19.50% / 15.40% | 20.07% / 16.13% |
| [`wq2012/ls_lstm_small_ge2e`](https://huggingface.co/wq2012/ls_lstm_small_ge2e) | `librispeech` (`2,400`) | `lstm` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,710,914` | 9.33% / 6.40% | 28.87% / 24.00% | 20.90% / 17.00% | 19.70% / 15.80% | 19.71% / 15.90% |
| [`wq2012/ls_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_tiny_ge2e) | `librispeech` (`2,400`) | `ecapa_tdnn` (`tiny`) | `asp` / `ge2e_softmax` / `cosine` | `626,722` | 9.67% / 4.90% | 29.43% / 25.10% | 20.65% / 17.20% | 19.92% / 15.73% | 19.99% / 16.03% |
| [`wq2012/ls_resnet_small_ge2e`](https://huggingface.co/wq2012/ls_resnet_small_ge2e) | `librispeech` (`2,600`) | `resnet` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `2,239,922` | 10.27% / 6.30% | 26.73% / 22.20% | 19.85% / 14.70% | 18.95% / 14.40% | 19.52% / 15.73% |
| [`wq2012/ls_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_small_ge2e) | `librispeech` (`2,400`) | `ecapa_tdnn` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,684,418` | 10.37% / 4.90% | 29.80% / 24.80% | 19.75% / 15.60% | 19.97% / 15.10% | 19.81% / 15.93% |
| [`wq2012/ls_conformer_small_triplet`](https://huggingface.co/wq2012/ls_conformer_small_triplet) | `librispeech` (`2,400`) | `conformer` (`small`) | `atp` / `triplet` / `cosine` | `2,029,892` | 10.70% / 6.60% | 29.27% / 25.40% | 24.70% / 20.80% | 21.56% / 17.60% | 21.64% / 17.83% |
| [`wq2012/cn_ecapa_tdnn_small_arcface`](https://huggingface.co/wq2012/cn_ecapa_tdnn_small_arcface) | `cnceleb2` (`1,800`) | `ecapa_tdnn` (`small`) | `asp` / `arcface` / `cosine` | `1,684,416` | 15.23% / 9.40% | **18.67%** / **13.80%** | 20.75% / 17.40% | **18.22%** / **13.53%** | 18.98% / **14.80%** |
| [`wq2012/cn_conformer_small_ge2e`](https://huggingface.co/wq2012/cn_conformer_small_ge2e) | `cnceleb2` (`1,600`) | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 18.30% / 12.60% | 18.97% / 15.40% | 24.30% / 21.00% | 20.52% / 16.33% | 21.01% / 16.53% |
| [`wq2012/cn_conformer_small_pfas`](https://huggingface.co/wq2012/cn_conformer_small_pfas) | `cnceleb2` (`1,600`) | `conformer` (`small`) | `pfas` / `extended_set_softmax` / `pfas` | `2,162,187` | 17.73% / 12.30% | 19.33% / 15.70% | 26.35% / 22.10% | 21.14% / 16.70% | 22.35% / 19.03% |
| [`wq2012/cn_conformer_small_dr_vectors`](https://huggingface.co/wq2012/cn_conformer_small_dr_vectors) | `cnceleb2` (`1,600`) | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 17.53% / 12.20% | 19.77% / 14.80% | 26.05% / 22.10% | 21.12% / 16.37% | 21.05% / 16.93% |
| [`wq2012/vox_ecapa_tdnn_small_ext_softmax`](https://huggingface.co/wq2012/vox_ecapa_tdnn_small_ext_softmax) | `voxceleb1` (`1,000`) | `ecapa_tdnn` (`small`) | `asp` / `extended_set_softmax` / `cosine` | `1,684,418` | 16.60% / 12.80% | 29.30% / 27.50% | **18.40%** / **14.10%** | 21.43% / 18.13% | 22.25% / 18.63% |
| [`wq2012/vox_conformer_small_ge2e`](https://huggingface.co/wq2012/vox_conformer_small_ge2e) | `voxceleb1` (`1,000`) | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 16.50% / 11.60% | 26.57% / 25.80% | 19.10% / 14.20% | 20.72% / 17.20% | 20.84% / 16.47% |

---

## Deviations from the Source Papers

Several released zoo configurations implement **FlaxSpeaker variants** of the cited algorithms tailored for `< 5M`-parameter models on `≤ 381` speakers, alongside the exact paper-faithful configurations (`icassp2018_ge2e_config`, `interspeech2021_dr_vectors_config`, `odyssey2022_conformer_atp_config`, `odyssey2022_pfas_config` with `pure_paper_mode=True`). In `v0.4.0`, **all four `pure_paper_mode=True` presets have also been trained for `4,600` steps on the Universal corpus** and benchmarked in `pretrained_models/ablation_results.json` (`studies.paper_faithful_presets_smoke_test`).

| Component | Source paper | Released FlaxSpeaker `v0.4.0` zoo checkpoints | Paper-faithful preset (`4,600`-step Universal Pooled 1 / 3-utt EER) |
| :--- | :--- | :--- | :--- |
| Training budget / data | GE2E: ~36 M utts, 18 K spk, N=64×M=10; Dr-Vectors/PFAS: 16 spk × 8 utts, 500 k steps | `1,000–4,600` steps (`S_safe = X + 2` ep) × (`8` spk × `4` utts); ≤ 25 utts/spk; 381 / 232 / 119 / 30 speakers; `T ~ U(90, 180)` | Evaluated on the 381-speaker Universal split (`4,600` steps) |
| GE2E encoder | 3 × LSTM-768 + 256-D projection, linear output | `ls_lstm_*` / `univ_lstm_*`: 3 × LSTM-256, no projection, per-layer LayerNorm + residual, ASP pooling, nonlinear bottleneck FFN (`15.50% / 12.50%`) | `icassp2018_ge2e_config(pure_paper_mode=True)` (`26.59% / 24.13%` — unnormalized 3×768 LSTM overfits/stalls on 9.4k utts) |
| GE2E contrast loss | Eq. 7 | `*_ge2e_contrast`: **hybrid contrast surrogate** (value = Eq. 7; gradient = 0.25 Eq. 7 + 0.25 BCE + 0.50 softmax-CE), `ge2e_init_b = −1`, `b` grad scale 0.5 (`14.95% / 10.73%`) | `ge2e_contrast_surrogate: false` (default) |
| Extended-set softmax | Eq. 2 (pure) | `*_ext_softmax`, `*_dr_vectors`, `*_pfas`: **blended** 0.5 row-softmax + 0.5 extended-set with `−log N` shift (`15.62% / 11.63%`) | `pure_extended_set: true` (`16.44% / 11.70%` on Conformer-Small) |
| Dr-Vector scorer | 200 base + 56 side dims; 3 × 256 leaky-ReLU(0.2); weighted sum; trainable affine | `*_dr_vectors`: cosine over all 192 dims, hidden `(128, 64)` ReLU + LayerNorm, linear read-out, fixed 0.25 residual weight (`15.88% / 11.67%`) | `interspeech2021_dr_vectors_config(pure_paper_mode=True)` (`15.47% / 11.53%`) |
| ATP pooling | single head, linear-sigmoid weights, full-D mean/std | `attention_dim: 128` tanh bottleneck, 4 channel-subspace heads (`subspace_heads: true`); source paper is a language-ID paper | `odyssey2022_conformer_atp_config(pure_paper_mode=True)` (`16.82% / 13.87%`) |
| PFAS head | ATP → affine + ReLU(512) → linear; 32 tied keys (2,048-D) | `*_pfas*`: **PFAS-SN** — per-frame key/value projections, multi-head tanh frame attention + global ATP anchor, 4 keys × `(32 + 128)` (`15.39% / 11.80%` GE2E, `15.55% / 11.53%` BXS) | `odyssey2022_pfas_config(pure_paper_mode=True)` (`18.34% / 14.43%`) |
| ECAPA-TDNN / Angular Margin | BatchNorm / fixed margin or integer `m = 4` | LayerNorm (`norm_after_activation` ablated: `15.84%` vs `16.34%`) / linear `margin_warmup_steps` on ArcFace & CosFace (`14.60%` & `14.71%`); real-valued SphereFace `m = 1.35` with λ-annealing (`1000 → 5`, `14.60%`) | — |
| Augmentation | waveform MTR with recorded noise + room simulator | synthetic log-Mel noise (SNR 12–20 dB) + one-tap smoothing | — |
| Evaluation | VoxCeleb1-O 37,611 trials (cleaned `veri_test2.txt`) over 40 speakers after VoxCeleb2 training | 10-speaker subset of `veri_test2.txt` (2,500 trials, 71.6 % target); 30/40 official test speakers used for training | — |

---

## Testing, Linting & Documentation

Run the unit test suite (covering all backbones, exact padding-mask invariance across all backbone variants including bidirectional LSTM and Mamba, all losses including pure-vs-surrogate `ge2e_contrast`, pure/blended `extended_set_softmax`, and `margin_warmup_steps`, all pooling heads including `"dot"` PFAS attention, VAD + AGC, mutually exclusive augmentation, Dr-Vectors paper options, every `pure_paper_mode` preset field reaching the built module, strict config-key validation, AS-Norm cohort disjointness, export-window resolution, local-vs-Hub checkpoint equivalence, TFLite export, version consistency across `pyproject.toml` / `setup.py` / `__init__.py` / `CHANGELOG.md`, documentation script existence, and `training_provenance` consistency across all 44 checkpoints):

```bash
pytest -v tests.py
flake8 .
./run_pdoc.sh
```
