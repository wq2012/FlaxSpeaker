---
title: FlaxSpeaker — Speaker Verification Demo (JAX/Flax)
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Speaker verification in JAX/Flax (GE2E, Dr-Vectors, ATP, PFAS) — 49 converged v0.5.0 models
---

# FlaxSpeaker (`v0.5.0`) — Interactive Speaker Verification Demo

## Disclaimer & Reproduction Notice

> **This is not an officially supported Google product.**
>
> This demo and the `FlaxSpeaker` library are a from-scratch JAX/Flax re-implementation, written using only the published papers (no proprietary code or data), of speaker recognition algorithms from:
> - **GE2E**: *Generalized End-to-End Loss for Speaker Verification* ([arXiv:1710.10467](https://arxiv.org/abs/1710.10467))
> - **Dr-Vectors**: *Decision Residual Networks and an Improved Loss for Speaker Recognition* ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989))
> - **ATP** (pooling head repurposed from a *language identification* paper): *Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech* ([arXiv:2202.12163](https://arxiv.org/abs/2202.12163))
> - **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* ([arXiv:2203.05642](https://arxiv.org/abs/2203.05642))
>
> The original papers were evaluated on Google-internal infrastructure and data, not with this library.

## Two-Tier `v0.5.0` Model Zoo (read first)

All **49 selectable checkpoints** (`v0.5.0`) are organized into two tiers:
1. **Tier-2 Full-Data Universal Flagships (`5` `univ_full_*` models, `14.86M–24.55M` parameters):** Trained across **12,193 speakers** (`36,579` speed-perturbed classes, `464,557` clean utterances / `1,857,149` MUSAN + RIR + speed-augmented items across `VoxCeleb2-dev + VoxCeleb1-dev + CN-Celeb1/2 + LibriSpeech-960h`) with zero speaker overlap against any evaluation split (`VoxCeleb1-O`, `CN-Celeb-E`, `LibriSpeech test-clean`, and the 80-speaker controlled suite). Covers three complementary encoder backbones (`Conformer-Large`, `ECAPA-TDNN c=1024`, `SE-ResNet-34`) plus warm-started `PFAS-SN` and `Dr-Vectors` scoring heads.
2. **Tier-1 Controlled Benchmark Zoo (`44` `univ_*`, `ls_*`, `cn_*`, `vox_*` models, `0.63M–4.55M` parameters):** Trained to calibrated `S_safe = X + 2` epoch step budgets (`4,600` steps / `15.58` epochs on Universal `381` spk / `9,447` utts; `2,400–2,600` steps on LibriSpeech; `1,600–1,800` steps on CN-Celeb2; `1,000` steps on VoxCeleb1) with uniform variable-length masked crops `T ~ U(90, 180)` frames for controlled architectural, pooling, objective, and scorer comparisons.

## What the demo does

- **Real inference only.** Every verdict runs the selected checkpoint on the uploaded/recorded audio (energy VAD → 80-D log-Mel → JAX encoder → cosine / Dr-Vector / PFAS scoring). Nothing is precomputed; curated examples are *not* cached.
- **Calibrated default threshold.** When you pick a model, the decision threshold resets to that model's **own EER operating point** read from its `eval_results.json` (both 1-utt and 3-utt operating points); the slider range adapts to the scoring back-end (cosine vs. Dr-Vector vs. PFAS scales).
- **Multi-utterance enrollment** (centroid for cosine / Dr-Vector, key stacking for PFAS), a VAD on/off toggle with SNR-margin and dynamic-range sliders, and a diagnostic plot (waveform + VAD gate, post-VAD log-Mel spectrograms, embedding profile).
- **Latency** is reported as steady-state model inference time separately from end-to-end time: embeddings go through the masked power-of-two bucketed full-utterance evaluation path (`flaxspeaker.evaluation.encode_features`), pre-warmed at model load.
- AS-Norm is intentionally **not** offered in the demo: a meaningful AS-Norm needs an impostor cohort disjoint from the trial audio, which a stateless web demo does not have. The library's `evaluate_verification_trials(..., cohort_embeddings=...)` enforces this requirement.

## Links & Pretrained Models

- **Hugging Face Collection (49 Pretrained Models)**: [`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker)
- **GitHub Repository**: [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker)
- **PyPI Package**: [`flaxspeaker`](https://pypi.org/project/flaxspeaker/)

### Model selection

The dropdown lists all **49 checkpoints**: **5 full-data universal flagships** (`univ_full_conformer_large_arcface`, `univ_full_ecapa_c1024_arcface`, `univ_full_resnet34_arcface`, `univ_full_conformer_large_pfas`, `univ_full_conformer_large_dr_vectors`), **19 controlled universal multi-domain models** (`univ_*`), and **25 controlled single-domain specialists** (`ls_*`, `cn_*`, `vox_*`). Per-model metrics, 95% bootstrap confidence intervals, and paper-deviation notes are included in each model card.

## Hosting status

This folder is a complete Gradio Space; `scripts/deploy_hf_space.py` deploys it to `wq2012/FlaxSpeaker` (or any target Space namespace with Gradio Space hosting enabled).

## Running locally

```bash
pip install -r requirements.txt   # installs flaxspeaker>=0.5.0 + gradio
python app.py
```

Checkpoints are downloaded from the Hub on first use (or read from `../pretrained_models/` when running inside the repository).
