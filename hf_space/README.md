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
short_description: Speaker verification in JAX/Flax (GE2E, Dr-Vectors, ATP, PFAS) — 44 converged v0.4.0 models
---

# FlaxSpeaker (`v0.4.0`) — Interactive Speaker Verification Demo

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

## Corpus-scale caveat (read first)

All 44 selectable checkpoints (`v0.4.0`) were trained to full convergence plus a 2-epoch safety margin (`S_safe = X + 2 epochs`: 4,600 steps / 15.58 epochs on Universal; 2,400–2,600 steps on LibriSpeech; 1,600–1,800 steps on CN-Celeb2; 1,000 steps on VoxCeleb1) with variable-length masked crops ($T \sim \mathcal{U}(90, 180)$ frames). Because the training corpora are bounded public subsets ($\le 381$ speakers, $\le 9,447$ utterances), the zoo spans **14.60–22.35% pooled 1-utt EER** (**10.40–19.67% pooled 3-utt EER**, and **5.17%** LibriSpeech 1-utt EER / **2.20%** 3-utt EER in-domain) and **19.15–29.96% EER on a 10-speaker VoxCeleb1-O subset** (not comparable to published VoxCeleb1-O numbers); published systems trained on 6,000-speaker VoxCeleb2 with MUSAN/RIR reach about 1%. They are controlled research checkpoints for comparing backbones, pooling heads, objectives, and scorers, not industrial-scale models.

## What the demo does

- **Real inference only.** Every verdict runs the selected checkpoint on the uploaded/recorded audio (energy VAD → 80-D log-Mel → JAX encoder → cosine / Dr-Vector / PFAS scoring). Nothing is precomputed; curated examples are *not* cached.
- **Calibrated default threshold.** When you pick a model, the decision threshold resets to that model's **own pooled EER operating point** read from its `eval_results.json` (e.g. `0.384` 1-utt / `0.474` 3-utt for `univ_conformer_small_arcface`, `0.501` / `0.583` for `univ_conformer_tiny_ge2e`); the slider range adapts to the scoring back-end (cosine vs. Dr-Vector vs. PFAS scales).
- **Multi-utterance enrollment** (centroid for cosine / Dr-Vector, key stacking for PFAS), a VAD on/off toggle with SNR-margin and dynamic-range sliders, and a diagnostic plot (waveform + VAD gate, post-VAD log-Mel spectrograms, embedding profile).
- **Latency** is reported as steady-state model inference time separately from end-to-end time: embeddings go through the benchmark's masked fixed-window path (`flaxspeaker.evaluation.encode_features`, centre-crop / zero-pad to `model.max_eval_seq_len` frames with a validity mask), so a single kernel is compiled when the model is loaded and no re-compilation happens for new clip lengths.
- AS-Norm is intentionally **not** offered in the demo: a meaningful AS-Norm needs an impostor cohort disjoint from the trial audio, which a stateless web demo does not have. The library's `evaluate_verification_trials(..., cohort_embeddings=...)` enforces this requirement.

## Links & Pretrained Models

- **Hugging Face Collection (44 Pretrained Models)**: [`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker)
- **GitHub Repository**: [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker)
- **PyPI Package**: [`flaxspeaker`](https://pypi.org/project/flaxspeaker/)

### Model selection

The dropdown lists all 44 checkpoints: **19 universal multi-domain models** (`univ_*`, trained jointly on LibriSpeech + CN-Celeb2 + VoxCeleb1 with domain-balanced batching for 4,600 steps) and **25 single-domain specialists** (`ls_*` LibriSpeech-only, `cn_*` CN-Celeb2-only, `vox_*` VoxCeleb1-only). Top universal checkpoints include `univ_conformer_small_arcface` (14.60% pooled 1-utt / 10.87% 3-utt EER), `univ_conformer_small_sphereface` (14.60% / 10.53%), `univ_conformer_small_cosface` (14.71% / 10.40%; 19.15% VoxCeleb1-O subset), and the default lightweight `univ_conformer_tiny_ge2e` (0.69 M parameters, 15.05% / 11.33%). Per-model numbers, confidence intervals, and a per-checkpoint "Deviations from the source papers" section are in each model card.

## Hosting status

This folder is a complete Gradio Space; `scripts/deploy_hf_space.py` deploys it to `wq2012/FlaxSpeaker` (or any target Space namespace with Gradio Space hosting enabled).

## Running locally

```bash
pip install -r requirements.txt   # installs flaxspeaker>=0.4.0 + gradio
python app.py
```

Checkpoints are downloaded from the Hub on first use (or read from `../pretrained_models/` when running inside the repository).
