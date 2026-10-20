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
short_description: Speaker verification in JAX/Flax (GE2E, Dr-Vectors, ATP, PFAS) — 400-step demo checkpoints
---

# FlaxSpeaker — Interactive Speaker Verification Demo

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

## Accuracy caveat (read first)

All 44 selectable checkpoints were trained for **400 gradient steps × 32 utterances on a CPU workstation with a single seed** (≈1.35 epochs over 9,447 utterances from 381 speakers). They reach **18.75–30.23% pooled 1-utt EER** on the project's in-house trial lists and **22.5–32.2% EER on a 10-speaker VoxCeleb1-O subset** (not comparable to published VoxCeleb1-O numbers); published systems trained on VoxCeleb2 reach about 1%. They are *demonstration* checkpoints for comparing architectures and losses under an identical budget, not production or state-of-the-art models.

## What the demo does

- **Real inference only.** Every verdict runs the selected checkpoint on the uploaded/recorded audio (energy VAD → 80-D log-Mel → JAX encoder → cosine / Dr-Vector / PFAS scoring). Nothing is precomputed; curated examples are *not* cached.
- **Calibrated default threshold.** When you pick a model, the decision threshold resets to that model's **own pooled 1-utt EER operating point** read from its `eval_results.json` (e.g. 0.625 for `univ_conformer_tiny_ge2e`, 0.671 for `univ_conformer_small_ge2e`); the slider range adapts to the scoring back-end (cosine vs. Dr-Vector vs. PFAS scales).
- **Multi-utterance enrollment** (centroid for cosine / Dr-Vector, key stacking for PFAS), a VAD on/off toggle with an SNR-margin slider, and a diagnostic plot (waveform + VAD gate, post-VAD log-Mel spectrograms, embedding profile).
- **Latency** is reported as steady-state model inference time separately from end-to-end time: embeddings go through the benchmark's masked fixed-window path (`flaxspeaker.evaluation.encode_features`, centre-crop / zero-pad to `model.max_eval_seq_len` frames with a validity mask), so a single kernel is compiled when the model is loaded and no re-compilation happens for new clip lengths.
- AS-Norm is intentionally **not** offered in the demo: a meaningful AS-Norm needs an impostor cohort disjoint from the trial audio, which a stateless web demo does not have. The library's `evaluate_verification_trials(..., cohort_embeddings=...)` enforces this requirement.

## Links & Pretrained Models

- **Hugging Face Collection (44 Pretrained Models)**: [`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595)
- **GitHub Repository**: [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker)
- **PyPI Package**: [`flaxspeaker`](https://pypi.org/project/flaxspeaker/)

### Model selection

The dropdown lists all 44 checkpoints: **19 universal multi-domain models** (`univ_*`, trained jointly on LibriSpeech + CN-Celeb2 + VoxCeleb1 with domain-balanced batching) and **25 single-domain specialists** (`ls_*` LibriSpeech-only, `cn_*` CN-Celeb2-only, `vox_*` VoxCeleb1-only). The default `univ_conformer_tiny_ge2e` (0.69 M parameters) has the lowest pooled 1-utt EER (18.75%) and the best VoxCeleb1-O-subset EER among universal models (23.94%) in `pretrained_models/benchmark_results.json`. Per-model numbers, confidence intervals, and a per-checkpoint "Deviations from the source papers" section are in each model card.

## Hosting status

This folder is a complete Gradio Space, but **no public Space is currently hosted** under `wq2012`: the Hub returns HTTP 402 for Gradio Spaces on the free tier (PRO subscription required). `scripts/deploy_hf_space.py` deploys it to any namespace that can host Gradio Spaces.

## Running locally

```bash
pip install -r requirements.txt   # installs flaxspeaker from GitHub + gradio
python app.py
```

Checkpoints are downloaded from the Hub on first use (or read from `../pretrained_models/` when running inside the repository).
