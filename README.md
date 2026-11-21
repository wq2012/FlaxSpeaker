# FlaxSpeaker

[![Python application](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml/badge.svg)](https://github.com/wq2012/FlaxSpeaker/actions/workflows/python-app.yml) [![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-50%20Pretrained%20Models-blue)](https://huggingface.co/collections/wq2012/flaxspeaker) [![PyPI Version](https://img.shields.io/pypi/v/flaxspeaker.svg)](https://pypi.python.org/pypi/flaxspeaker) [![Python Versions](https://img.shields.io/pypi/pyversions/flaxspeaker.svg)](https://pypi.org/project/flaxspeaker) [![Downloads](https://static.pepy.tech/badge/flaxspeaker)](https://pepy.tech/project/flaxspeaker) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

**FlaxSpeaker** (`v0.6.0`) is an open-source Python library built on [JAX](https://jax.readthedocs.io) and [Flax](https://flax.readthedocs.io) for **speaker verification** (*"Are these two audio recordings spoken by the same person?"*) and **speaker embedding extraction**.

**Do you need FlaxSpeaker?**
- **Drop-in Pretrained Speaker Verification**: Download one universal model from Hugging Face that works out of the box across English and Chinese, clean speech and noisy in-the-wild recordings—with ready-to-deploy **SafeTensors** and **FP32 / INT8 TFLite** edge models.
- **JAX / Flax Research & Education**: Train, compare, or customize **7 neural backbones** (`SimAM-ERes2Conformer`, `Conformer`, `ECAPA-TDNN`, `SE-ResNet`, `Mamba`, `Transformer`, `LSTM`), **9 temporal pooling heads**, and **10 metric / angular-margin losses** (`Sub-Center ArcFace`, `CosFace`, `GE2E`, `Dr-Vectors`, `PFAS`) in pure functional JAX (`jax.jit` / `jax.pmap`). For the companion PyTorch educational project, see [SpeakerRecognitionFromScratch](https://github.com/wq2012/SpeakerRecognitionFromScratch).

> **Note**: **This is not an officially supported Google product.** FlaxSpeaker is a clean-room open-source JAX/Flax implementation trained from scratch on public speech corpora; see [Reproduction Scope & Paper References](#reproduction-scope--paper-references).

### Which Pretrained Model Should You Use?

If you just want the strongest speaker verification or embedding model, use **[`wq2012/univ_full_eres2conformer_large`](https://huggingface.co/wq2012/univ_full_eres2conformer_large)**. It is a single multi-domain model trained from scratch on `12,193` speakers (`VoxCeleb1/2 + CN-Celeb1/2 + LibriSpeech`) that beats all other standalone models and our previous 5-model ensemble:

| Use Case | Recommended Hugging Face Model | Params (INT8 TFLite) | VoxCeleb1-O EER | CN-Celeb-E Multi-Enroll EER | LibriSpeech `test-clean` EER |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Best Overall Accuracy (Recommended)** | **[`wq2012/univ_full_eres2conformer_large`](https://huggingface.co/wq2012/univ_full_eres2conformer_large)** | `31.8M` (`33.1 MB`) | **`0.71%`** (`0.84%` raw) | **`6.72%`** (`7.45%` 1-utt) | **`0.65%`** (`0.27%` 3-utt raw) |
| **Lighter 2D CNN Backbone** | [`wq2012/univ_full_resnet34_arcface`](https://huggingface.co/wq2012/univ_full_resnet34_arcface) | `24.6M` (`25.0 MB`) | `1.60%` (`1.81%` raw) | `9.79%` (`10.26%` 1-utt) | `1.04%` (**`0.23%`** 3-utt) |
| **Fast Compact 1D Encoder** | [`wq2012/univ_full_ecapa_c1024_arcface`](https://huggingface.co/wq2012/univ_full_ecapa_c1024_arcface) | `14.9M` (`15.4 MB`) | `2.29%` (`2.54%` raw) | `10.82%` (`11.51%` 1-utt) | `1.55%` (`0.47%` 3-utt) |
| **Tiny CPU / Ablation Baseline** | [`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface) | `2.0M` | — (`13.94%` 381-spk pooled) | — | — |

All **50 pretrained checkpoints** (`6` full-data universal flagships + `44` controlled ablation models) are hosted in the **[`wq2012/FlaxSpeaker` Hugging Face Collection](https://huggingface.co/collections/wq2012/flaxspeaker)**.

### Quickstart (Install & Run in 30 Seconds)

```bash
pip install -U flaxspeaker
```

Verify two audio files or extract a 256-D L2-normalized speaker embedding in Python:

```python
import flaxspeaker

# Automatically downloads from Hugging Face Hub on first use
model = flaxspeaker.load_pretrained("wq2012/univ_full_eres2conformer_large")

# 1. Compare two audio files (also accepts a list of enrollment files)
same_score = model.verify(
    "hf_space/examples/librispeech_spk1_utt1.flac",
    "hf_space/examples/librispeech_spk1_utt2.flac",
)
diff_score = model.verify(
    "hf_space/examples/librispeech_spk1_utt1.flac",
    "hf_space/examples/librispeech_spk2_utt1.flac",
)
print(f"Same speaker: {same_score:.4f} | Different speaker: {diff_score:.4f}")
# -> Same speaker: 0.7899 | Different speaker: 0.2846 (English EER threshold ~0.36)

# 2. Extract a 256-D L2-normalized speaker embedding (NumPy array)
embedding = model.extract_embedding("hf_space/examples/librispeech_spk1_utt1.flac")
```

Or run from the command line or launch the interactive [Gradio](https://gradio.app) web app:

```bash
python scripts/inference.py \
  --model_dir wq2012/univ_full_eres2conformer_large \
  --enroll_audio hf_space/examples/librispeech_spk1_utt1.flac \
  --test_audio hf_space/examples/librispeech_spk1_utt2.flac

python hf_space/app.py   # interactive Gradio demo (50 models, VAD & spectrogram plots)
```

### Where to Find Detailed Information

- **[1. Flagship Benchmarks & Architecture Ablation](#1-flagship-benchmarks--architecture-ablation)** — Full metrics across all 6 full-data flagships, score fusion, and why `SimAM-ERes2Conformer` outperforms pure Conformer and ResNet.
- **[2. System Architecture](#2-system-architecture)** — Overview of the 7 backbones, 9 pooling heads, 10 losses, audio frontend, and FP32/INT8 TFLite export.
- **[3. CLI Workflows: Train, Evaluate & Export](#3-cli-workflows-train-evaluate--export)** — Commands to prepare datasets, train custom models, export to TFLite, and run tests.
- **[4. Controlled 44-Model Ablation Zoo](#4-controlled-44-model-ablation-zoo)** — Multi-domain vs. single-domain findings and collapsible tables for all 44 ablation checkpoints.
- **[5. Reproduction Scope & Paper References](#5-reproduction-scope--paper-references)** — Relationship to the GE2E, Dr-Vectors, ATP, and PFAS papers, source-paper deviations, and **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)**.

---

## 1. Flagship Benchmarks & Architecture Ablation

Our 6 Tier-2 full-data universal flagships (`univ_full_*`) are trained 100% from scratch via `jax.pmap` on TPU accelerators across **`12,193` physical speakers** (`36,579` speed-perturbed classes across `VoxCeleb2-dev + VoxCeleb1-dev + CN-Celeb1/2 + LibriSpeech-960h`, augmented with `MUSAN` noise/music/babble and `RIRS_NOISES` room reverberation) with zero speaker overlap against any evaluation split.

![FlaxSpeaker v0.6.0 Tier-2 Flagships & SimAM-ERes2Conformer Ablation](resources/flagship_comparison.png)

<!-- BEGIN README_FLAGSHIP_TABLE -->
| Hugging Face Repository | Backbone / Head / Objective | Params | Steps | VoxCeleb1-O Official (`37,611` trials) Raw / AS-Norm / QMF EER (%) | CN-Celeb-E Official (`3.48M` trials) Single Raw / QMF / Multi-Enroll EER (%) | LibriSpeech `test-clean` (`10,000` trials) 1-Utt Raw / QMF / 3-Utt EER (%) | Controlled 3-Domain Pooled 1-Utt / 3-Utt EER (%) | Controlled Macro-Avg 1-Utt / 3-Utt EER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/univ_full_conformer_large_dr_vectors`](https://huggingface.co/wq2012/univ_full_conformer_large_dr_vectors) | `conformer` (`large`) / `atp` / `extended_set_softmax` (`dr_vector`) | `22,171,529` | `16,000` | 2.58% / 2.51% / 2.36% | 12.89% / 11.83% / 11.12% | 2.04% / 1.84% / 0.70% | 5.97% / 2.97% (raw 2.97%) | 5.58% / 3.10% |
| [`wq2012/univ_full_conformer_large_pfas`](https://huggingface.co/wq2012/univ_full_conformer_large_pfas) | `conformer` (`large`) / `pfas` / `extended_set_softmax` (`pfas`) | `22,582,159` | `24,000` | 2.79% / 2.78% / 2.57% | 12.36% / 11.18% / 10.43% | 2.05% / 1.99% / 0.77% | 6.36% / 3.87% (raw 3.87%) | 5.72% / 3.47% |
| [`wq2012/univ_full_ecapa_c1024_arcface`](https://huggingface.co/wq2012/univ_full_ecapa_c1024_arcface) | `ecapa_tdnn` (`large`) / `asp` / `arcface` (`cosine`) | `14,856,832` | `81,000` | 2.54% / 2.45% / 2.29% | 12.63% / 11.51% / 10.82% | 2.07% / 1.55% / 0.47% | 7.10% / 4.07% (raw 4.07%) | 5.48% / 3.13% |
| [`wq2012/univ_full_conformer_large_arcface`](https://huggingface.co/wq2012/univ_full_conformer_large_arcface) | `conformer` (`large`) / `atp` / `arcface` (`cosine`) | `22,005,636` | `81,000` | 2.66% / 2.66% / 2.46% | 11.78% / 11.04% / 10.43% | 2.19% / 1.80% / 0.43% | 7.53% / 3.47% (raw 3.97%) | 5.53% / 2.60% |
| [`wq2012/univ_full_resnet34_arcface`](https://huggingface.co/wq2012/univ_full_resnet34_arcface) | `resnet` (`large`) / `stats` / `arcface` (`cosine`) | `24,551,472` | `81,000` | 1.81% / 1.73% / 1.60% | 10.48% / 10.26% / 9.79% | 1.22% / 1.04% / **0.23%** | 5.81% / 2.97% (raw 2.97%) | 4.56% / 2.33% |
| [`wq2012/univ_full_eres2conformer_large`](https://huggingface.co/wq2012/univ_full_eres2conformer_large) | `eres2conformer` (`large`) / `catp` / `subcenter_arcface` (`cosine`) | `31,834,352` | `93,000` | 0.84% / 0.85% / **0.71%** | 7.83% / **7.45%** / **6.72%** | 0.72% / **0.65%** / 0.40% | **4.30%** / **2.30%** (raw 2.30%) | **3.21%** / **1.83%** |
| **2-Way Score Fusion (`0.3·Conformer-L + 0.7·ECAPA-c1024`)** | Heterogeneous Cosine + AS-Norm + QMF | `36,862,468` | — | — / 2.20% / 2.01% | — / 10.44% / — | — / 1.62% / — | — | — |
| **3-Way Backbone Fusion (`0.25·Conf-L + 0.2·ECAPA + 0.55·ResNet-34`)** | 3-Architecture Cosine + AS-Norm + QMF | `61,413,940` | — | — / 1.62% / 1.43% | — / 9.37% / — | — / 1.14% / — | — | — |
| **4-Way Backbone Fusion (`0.02·Conf-L + 0.1·ECAPA + 0.2·ResNet-34 + 0.68·ERes2Conf`)** | 4-Architecture Cosine + AS-Norm + QMF | `93,248,292` | — | — / **0.85%** / **0.69%** | — / 7.62% / — | — / 0.74% / — | — | — |
| **6-Way Flagship Fusion (`Conf-L + ECAPA + ResNet-34 + ERes2Conf + PFAS + DrV`)** | Heterogeneous 6-Way AS-Norm + QMF | `138,001,980` | — | — / 0.88% / 0.74% | — / **7.45%** / — | — / 0.80% / — | — | — |
| **6-Way Multi-Enroll Fusion** | Cosine AS-Norm + Frame-Level PFAS | `138,001,980` | — | — | — / — / 6.95% (`AUC 0.9798`) | — / — / 0.43% (`AUC 0.9999`) | — | — |
<!-- END README_FLAGSHIP_TABLE -->

### Why `SimAM-ERes2Conformer` Beats Pure `Conformer-Large` and `SE-ResNet-34`

Pure 1D `Conformer-Large` (`2.46%` VoxCeleb1-O QMF) and its warm-started `Dr-Vectors` (`2.36%`) / `PFAS` (`2.57%`) heads trailed 2D `SE-ResNet-34` (`1.60%`) for three reasons:
1. **2D Time-Frequency Inductive Bias**: Flattening all 80 Mel bins at layer 0 discards local 2D translation equivariance along formant trajectories and harmonic spacing.
2. **Optimizer & Prototype Regularization**: Across `36,579` speed-perturbed speaker classes, `Nesterov SGD` with weight decay on the `(36,579 × 3)` `Sub-Center ArcFace` prototypes (`decay_class_weights=True`) generalizes better than `AdamW`.
3. **Global Margin vs. Batch-Local Metric Losses**: `PFAS` and `Dr-Vectors` optimize pairwise scores within a `512`-utterance batch (`Extended-Set Softmax`), whereas `Sub-Center ArcFace` + `5.2 s` Large-Margin Fine-Tuning (`LMFT`) enforces global separation against all `36,579` classes simultaneously.

Combining a 4-stage 2D `SimAM-ERes2NetV2 + AFF` stem, `2` global `RoPE + QK-Norm` Conformer blocks, `Context-ATP` (`catp`) pooling, and `Nesterov SGD` into **`SimAM-ERes2Conformer`** confirms each improvement in a controlled `27,500`-step ablation ([`pretrained_models/eres2conformer_ablation_results.json`](pretrained_models/eres2conformer_ablation_results.json)):

<!-- BEGIN ERES2CONF_ABLATION_TABLE -->
| Ablation Arm (`27,500` steps = `25k` Stage-1 + `2.5k` `5.0 s` LMFT, batch `512`) | Conformer Blocks | Optimizer | Params | VoxCeleb1-O Raw / AS-Norm / QMF EER (%) | CN-Celeb-E Single Raw / QMF / Multi-Enroll EER (%) | LibriSpeech `test-clean` 1-Utt Raw / QMF / 3-Utt EER (%) | Controlled Pooled 1/3-Utt EER (%) | Controlled Macro 1/3-Utt EER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eres2conformer_catp_sgd`** | `2` | `sgd` | `31,834,352` | 1.83% / 1.69% / **1.61%** | 10.21% / **8.94%** / **8.83%** | 1.37% / **1.01%** / **0.23%** | **6.14%** / **3.50%** | **4.31%** / **2.77%** |
| `eres2net_noconf_catp_sgd` | `0` | `sgd` | `20,846,816` | 2.04% / 1.95% / 1.85% | 10.62% / 9.20% / 8.97% | 1.56% / 1.16% / 0.33% | 6.56% / 3.87% | 4.57% / 2.77% |
| `eres2conformer_catp_adamw` | `2` | `adamw` | `31,834,352` | 2.19% / 2.09% / 1.90% | 11.10% / 9.87% / 9.55% | 1.57% / 1.19% / 0.43% | 6.53% / 4.17% | 4.73% / 2.73% |
<!-- END ERES2CONF_ABLATION_TABLE -->

---

## 2. System Architecture

![FlaxSpeaker System Architecture](resources/architecture_diagram.png)

All components share a unified, padding-mask-invariant `(B, T, F) -> (B, T, D)` interface configured via typed `ExperimentConfig` YAML/JSON files (full mathematical details in [RESEARCH_REPORT.md](RESEARCH_REPORT.md)):

- **7 Sequence Backbones ([`flaxspeaker/backbones.py`](flaxspeaker/backbones.py))**:
  - **SimAM-ERes2Conformer** (`eres2conformer`): 4-stage 2D multi-scale `ERes2NetV2` + `AFF` + 3D `SimAM` stem, Stage-3/4 `MFA`, and `RoPE + QK-Norm + BatchNorm` Macaron Conformer blocks (`31.83M` params in `large`).
  - **SE-ResNet** (`resnet`): 2D time-frequency residual CNN with Squeeze-and-Excitation (`SE-ResNet-18` `2.24M` and `SE-ResNet-34` `24.55M`).
  - **Conformer** (`conformer`): Macaron FFN + Masked MHSA + GroupNorm Depthwise Conv + MFA (`tiny` to `large`).
  - **ECAPA-TDNN** (`ecapa_tdnn`): 1D dilated SE-Res2Net blocks with running-statistics `BatchNorm` after activation and MFA (`c=1024` in `large`).
  - **Mamba** (`mamba`): Linear-time Selective State Space Model via `jax.lax.associative_scan` (unidirectional or bidirectional).
  - **Transformer** (`transformer`) & **LSTM** (`lstm`): Pre-LN MHSA + MFA, and mask-aware uni-/bidirectional stacked LSTMs.
- **9 Temporal Pooling Heads ([`flaxspeaker/pooling.py`](flaxspeaker/pooling.py))**:
  - **Context-ATP** (`catp`, 16 channel-subspace sigmoid heads conditioned on utterance mean/std + `BN-Neck`), **ATP** (`atp`), **ASP** (`asp`), **SAP** (`sap`), **Stats** (`stats`), **Mean** (`mean`), **Last** (`last`), causal **Cumulative Stats** (`cumulative_stats`), and multi-key **PFAS** (`pfas`: `atp_ffn` and `shared_nonlinear` PFAS-SN).
- **10 Training Objectives & 3 Scorers ([`flaxspeaker/losses.py`](flaxspeaker/losses.py), [`flaxspeaker/scoring.py`](flaxspeaker/scoring.py))**:
  - **Angular Margin**: `Sub-Center ArcFace (K=3)` (`subcenter_arcface` / `arcface` with linear `margin_warmup_steps` and optional `decay_class_weights`), `CosFace` (`cosface`), `SphereFace` (`sphereface`), and `Softmax` (`softmax`).
  - **Metric & End-to-End Scoring**: `GE2E-Softmax` (`ge2e_softmax`), `GE2E-Contrast` (`ge2e_contrast`), `Extended-Set Softmax` (`extended_set_softmax`, pure and blended BXS), `Triplet` (`triplet`), and `Pairwise` (`pairwise`), paired with **Cosine**, **Dr-Vectors** (`dr_vector`), or **PFAS** (`pfas`) scoring plus Top-300 **AS-Norm** and 4-feature **QMF** calibration.
- **Frontend, Sharded Data & Edge Export ([`flaxspeaker/feature_extraction.py`](flaxspeaker/feature_extraction.py), [`flaxspeaker/export.py`](flaxspeaker/export.py))**:
  - 16 kHz 80-D log-Mel + adaptive frame-level energy VAD + CMVN, `MUSAN` + `RIRS_NOISES` + `3x` speed perturbation into memory-mapped `float16` shards, and non-XLA `jax2tf` export to **FP32 / INT8 TFLite** (`3.44–3.94×` size reduction across 20 exported binaries with at most `0.14 pp` EER change; `2.38%` FP32 → `2.41%` INT8 on `univ_full_eres2conformer_large`).

---

## 3. CLI Workflows: Train, Evaluate & Export

For development, Hugging Face Hub, and TFLite export extras:

```bash
pip install -e ".[dev,export,hf]"
```

### Prepare Datasets & `float16` Shards
```bash
# Controlled 381-speaker benchmark splits (LibriSpeech + CN-Celeb2 + VoxCeleb1)
python scripts/prepare_datasets.py --data_root ./data/speaker_datasets
python scripts/prepare_controlled_shards.py --data_root ./data/speaker_datasets

# Full 12,193-speaker multi-corpus datasets (VoxCeleb1/2 + CN-Celeb1/2 + LibriSpeech-960h + MUSAN + RIRS)
python scripts/download_full_corpora.py
python scripts/prepare_full_corpora_shards.py
```

### Train Custom Models or Reproduce Flagships
```bash
# Train a single model from any YAML config in configs/*.yml
python scripts/train.py \
  --config configs/univ_conformer_small_cosface.yml \
  --train_csv ./data/speaker_datasets/splits/universal/train.csv \
  --eval_csv ./data/speaker_datasets/splits/universal/eval.csv \
  --trials_file ./data/speaker_datasets/splits/universal/trials.tsv \
  --output_dir ./checkpoints/univ_conformer_small_cosface

# Run the full-data multi-device flagship training pipeline or 16-config PFAS sweep
python scripts/train_full_data_flagships.py
python scripts/run_pfas_hparam_sweep.py
```

### Evaluate & Export to TFLite / Hugging Face Hub
```bash
# Evaluate 1-utterance and multi-utterance verification trials
python scripts/evaluate.py \
  --model_dir wq2012/univ_full_eres2conformer_large \
  --trials_file ./data/speaker_datasets/splits/universal/trials.tsv \
  --multi_trials_file ./data/speaker_datasets/splits/universal/multi_enroll_trials.tsv \
  --output_json eval_summary.json

# Export any cosine-scored checkpoint to FP32 and dynamic-range INT8 TFLite
python scripts/export_model.py \
  --model_dir wq2012/univ_full_eres2conformer_large \
  --output_dir ./exported_models/univ_full_eres2conformer_large \
  --export_tflite --quantize_int8

# Evaluate all exported TFLite models and regenerate Hub model cards
python scripts/eval_tflite_models.py --data_root ./data/speaker_datasets --models_dir ./pretrained_models
python scripts/update_hf_model_cards.py --push --include_weights
```

### Run Unit Tests, Linting & Benchmark Reproducers
```bash
pytest -v tests.py
flake8 .
pytype .
./run_pdoc.sh

# Reproduce Tier-1 ablations, multi-seed controls, provenance checks & paper figures
python scripts/run_phase1_convergence_pilot.py --data_root ./data/speaker_datasets
python scripts/run_research_benchmarks.py --data_root ./data/speaker_datasets
python scripts/run_r1_ablations.py --data_root ./data/speaker_datasets
python scripts/run_r2_controls.py --data_root ./data/speaker_datasets
python scripts/eval_vox1_o_subset.py --data_root ./data/speaker_datasets --models_dir ./pretrained_models
python scripts/fingerprint_released_checkpoints.py --data_root ./data/speaker_datasets
python scripts/write_training_provenance.py
python scripts/generate_report_tables.py --section all > /tmp/report_tables.md
python scripts/generate_paper_tables.py --paper_dir ../flax_speaker_paper
python scripts/generate_figures.py --benchmark-json pretrained_models/benchmark_results.json --output-dir ./figures
```

---

## 4. Controlled 44-Model Ablation Zoo

Alongside the 6 full-data flagships, we release **44 controlled CPU-trained models** (`0.63M–4.55M` params) trained on speaker-disjoint subsets (`381` speakers / `9,447` utterances across LibriSpeech, CN-Celeb2, and VoxCeleb1) to isolate backbone, pooling, loss, and multi-domain effects under identical conditions:

![FlaxSpeaker v0.6.0 Tier-1 Controlled Benchmark Summary](resources/benchmark_charts.png)

- **Universal Multi-Domain Beats Single-Domain Specialists (`19 / 19` Pairs)**: Every universal `univ_*` model outperforms its matched LibriSpeech-only `ls_*` specialist on pooled 1-utt, macro 1-utt, and pooled 3-utt EER—including by **`3.88` percentage points** (`16.32%` vs. `20.20%`) when subsampled to the exact same utterance count (`5,788`) and step budget (`2,400` steps).
- **Angular-Margin Crossover**: With linear margin warm-up (`margin_warmup_steps=500`), `CosFace` (`13.94%` pooled 1-utt / `9.67%` 3-utt), `SphereFace` (`14.20%`), and `Sub-Center ArcFace` (`14.80%`) lead all 19 controlled universal models.

| Model Paradigm | Representative Checkpoint (`Steps`) | LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | Universal Macro-Avg 1-Utt / 3-Utt EER (%) | Universal Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LibriSpeech-Only Specialist (`GE2E`)** | [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) (`2,400`) | 12.07% / 7.50% | 26.20% / 23.20% | 21.25% / 18.50% | 19.84% / 16.40% | 20.20% / 16.93% |
| **Universal Multi-Domain (`GE2E`)** | [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) (`4,600`) | 13.47% / 9.20% | 19.40% / 16.30% | 20.85% / 16.40% | 17.91% / 13.97% | 17.54% / 13.87% |
| **LibriSpeech-Only Specialist (`ArcFace`)** | [`wq2012/ls_conformer_small_arcface`](https://huggingface.co/wq2012/ls_conformer_small_arcface) (`2,400`) | 8.13% / 4.20% | 21.80% / 18.20% | 17.65% / 13.60% | 15.86% / 12.00% | 16.23% / 12.40% |
| **Universal Multi-Domain (`CosFace`)** | [`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface) (`4,600`) | 7.67% / 3.80% | 16.73% / 11.90% | 17.05% / 11.80% | 13.82% / 9.17% | 13.94% / 9.67% |
| **Full-Data Flagship (`SE-ResNet-34`)** | [`wq2012/univ_full_resnet34_arcface`](https://huggingface.co/wq2012/univ_full_resnet34_arcface) (`81,000`) | 1.47% / **0.40%** | 11.00% / 6.30% | 1.20% / 0.30% | 4.56% / 2.33% | 5.81% / 2.97% |
| **Full-Data Flagship (`SimAM-ERes2Conformer`)** | [`wq2012/univ_full_eres2conformer_large`](https://huggingface.co/wq2012/univ_full_eres2conformer_large) (`93,000`) | **0.90%** / **0.40%** | **8.17%** / **5.10%** | **0.55%** / **0.00%** | **3.21%** / **1.83%** | **4.30%** / **2.30%** |

<details>
<summary><strong>Click to expand all 19 Tier-1 Universal Multi-Domain Models (<code>univ_*</code>)</strong></summary>

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

</details>

<details>
<summary><strong>Click to expand all 25 Tier-1 Single-Domain Specialist Models (<code>ls_*</code>, <code>cn_*</code>, <code>vox_*</code>)</strong></summary>

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

</details>

Machine-readable result archives in [`pretrained_models/`](pretrained_models/): [`benchmark_results.json`](pretrained_models/benchmark_results.json), [`full_data_flagship_results.json`](pretrained_models/full_data_flagship_results.json), [`eres2conformer_ablation_results.json`](pretrained_models/eres2conformer_ablation_results.json), [`pfas_hparam_sweep.json`](pretrained_models/pfas_hparam_sweep.json), [`ablation_results.json`](pretrained_models/ablation_results.json), [`r2_seed_and_control_results.json`](pretrained_models/r2_seed_and_control_results.json), [`vox1_o_10spk_subset_results.json`](pretrained_models/vox1_o_10spk_subset_results.json), and [`step0_fingerprints.json`](pretrained_models/step0_fingerprints.json).

---

## 5. Reproduction Scope & Paper References

**This is not an officially supported Google product.**

FlaxSpeaker includes from-scratch JAX/Flax re-implementations—written using only the published papers, with no proprietary code or data—of the mathematical formulations in:
1. **GE2E**: *Generalized End-to-End Loss for Speaker Verification* — Li Wan, Quan Wang, Alan Papir, Ignacio Lopez Moreno ([ICASSP 2018, arXiv:1710.10467](https://arxiv.org/pdf/1710.10467))
2. **Dr-Vectors**: *Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition* — Jason Pelecanos, Quan Wang, Ignacio Lopez Moreno ([Interspeech 2021, arXiv:2104.01989](https://arxiv.org/pdf/2104.01989))
3. **ATP**: *Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech* — Quan Wang, Yang Yu, Jason Pelecanos, Yiling Huang, Ignacio Lopez Moreno ([Odyssey 2022, arXiv:2202.12163](https://arxiv.org/pdf/2202.12163)) *(language-ID pooling head repurposed here for speaker embeddings)*
4. **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* — Jason Pelecanos, Quan Wang, Yiling Huang, Ignacio Lopez Moreno ([Odyssey 2022, arXiv:2203.05642](https://arxiv.org/pdf/2203.05642))

Because those original papers were trained and evaluated on proprietary internal corpora (`~8K–18K` speakers, `~500k` steps), exact numerical parity with their internal tables is neither expected nor claimed. All checkpoints in this repository are trained from scratch on public speech datasets (**VoxCeleb1/2**, **CN-Celeb1/2**, **LibriSpeech**). Both exact paper-faithful presets (`pure_paper_mode=True` in `flaxspeaker.configs`) and our modern FlaxSpeaker extensions (`SimAM-ERes2Conformer`, `Context-ATP`, `Sub-Center ArcFace`, `PFAS-SN`, `BXS`) are provided and benchmarked side by side in **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)**:

| Paper & Venue | Original Internal Setup & Metric | FlaxSpeaker `v0.6.0` Open-Source Summary |
| :--- | :--- | :--- |
| **GE2E** ([ICASSP 2018](https://arxiv.org/pdf/1710.10467)) | `3×768` LSTM on ~36M utts / ~18K spk; `3.55%` multi-enroll TI-SV EER | Exact Eq. 6 (`ge2e_softmax`), Eq. 7 (`ge2e_contrast`), and hybrid contrast surrogate; `icassp2018_ge2e_config(pure_paper_mode=True)` |
| **Dr-Vectors** ([Interspeech 2021](https://arxiv.org/pdf/2104.01989)) | LSTM + `Extended-Set Softmax` + residual scorer; `~4–11%` rel. EER reduction | `univ_full_conformer_large_dr_vectors`: `2.36%` VoxCeleb1-O QMF EER (`5.97% / 2.97%` Controlled Pooled 1/3-Utt); `interspeech2021_dr_vectors_config(pure_paper_mode=True)` |
| **ATP** ([Odyssey 2022](https://arxiv.org/pdf/2202.12163)) | Conformer + sigmoid Attentive Temporal Pooling (`mean + std`) | Extended to multi-head subspace `ATP` and utterance-conditioned `Context-ATP` (`catp`, `0.71%` VoxCeleb1-O QMF on `univ_full_eres2conformer_large`); `odyssey2022_conformer_atp_config(pure_paper_mode=True)` |
| **PFAS** ([Odyssey 2022](https://arxiv.org/pdf/2203.05642)) | Conformer + parameter-free `(K, V)` cross-attention; `~8–14%` rel. EER reduction | `univ_full_conformer_large_pfas`: `2.57%` VoxCeleb1-O QMF / `10.43%` CN-Celeb-E Multi-Enroll EER; `odyssey2022_pfas_config(pure_paper_mode=True)` |
