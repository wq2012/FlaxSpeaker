---
title: FlaxSpeaker — Universal Speaker Recognition & Verification
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Universal Speaker Recognition in JAX/Flax (GE2E, Dr-Vectors, Conformer-ATP, PFAS)
---

# FlaxSpeaker — Interactive Universal Speaker Verification Demo

## Disclaimer & Open-Source Reproduction Notice

> **This is not an officially supported Google product.**
>
> This interactive demo and the `FlaxSpeaker` library are an independent, clean-room open-source reproduction in JAX/Flax of speaker recognition algorithms published in:
> - **GE2E**: *Generalized End-to-End Loss for Speaker Verification* ([arXiv:1710.10467](https://arxiv.org/abs/1710.10467))
> - **Dr-Vectors**: *Decision Residual Networks and an Improved Loss for Speaker Recognition* ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989))
> - **Conformer-ATP**: *Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech* ([arXiv:2202.12163](https://arxiv.org/abs/2202.12163))
> - **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* ([arXiv:2203.05642v3](https://arxiv.org/abs/2203.05642v3))
>
> The original papers were conducted using Google's internal proprietary infrastructure and internal datasets, not this open-source library.

## Links & Pretrained Models

- **Hugging Face Collection (44 Pretrained Models)**: [`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595)
- **GitHub Repository**: [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker)
- **PyPI Package**: [`flaxspeaker`](https://pypi.org/project/flaxspeaker/)

### Highlighted Pretrained Checkpoints on Hugging Face Hub

| Hugging Face Repository | Backbone / Size | Pooling / Loss / Scoring | Params | Macro-Avg 1-Utt / 3-Utt EER (%) | Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| [`wq2012/univ_conformer_tiny_ge2e`](https://huggingface.co/wq2012/univ_conformer_tiny_ge2e) | `conformer` (`tiny`) | `atp` / `ge2e_softmax` / `cosine` | `693,126` | **19.26%** / **15.20%** | **18.75%** / **15.03%** |
| [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 19.88% / 16.17% | 19.40% / 16.03% |
| [`wq2012/univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors) | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 20.09% / **15.20%** | 20.10% / 15.40% |
| [`wq2012/univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e) | `conformer` (`small`) | `pfas` / `ge2e_softmax` / `pfas` | `2,162,187` | 20.26% / 16.07% | 19.86% / 16.27% |
| [`wq2012/univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas) | `conformer` (`small`) | `pfas` / `extended_set_softmax` / `pfas` | `2,162,187` | 20.65% / 15.80% | 20.35% / 16.00% |
| [`wq2012/univ_mamba_small_ge2e`](https://huggingface.co/wq2012/univ_mamba_small_ge2e) | `mamba` (`small`) | `asp` / `ge2e_softmax` / `cosine` | `1,028,674` | 20.38% / 15.23% | 20.41% / 16.07% |
| [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 20.06% / 16.37% | 20.12% / 16.73% |
| [`wq2012/cn_conformer_small_dr_vectors`](https://huggingface.co/wq2012/cn_conformer_small_dr_vectors) | `conformer` (`small`) | `atp` / `extended_set_softmax` / `dr_vector` | `2,088,007` | 23.49% / 17.77% | 23.95% / 18.97% |
| [`wq2012/vox_conformer_small_ge2e`](https://huggingface.co/wq2012/vox_conformer_small_ge2e) | `conformer` (`small`) | `atp` / `ge2e_softmax` / `cosine` | `2,029,894` | 21.67% / 17.97% | 22.40% / 17.63% |

## Features

1. **Universal Multi-Domain Pretrained Models**: Select between all 44 Universal Conformer-ATP (`GE2E`, `Dr-Vectors`, `PFAS`, `Extended-Set Softmax`, `ArcFace`), Universal ECAPA-TDNN, Universal Mamba, Universal Transformer, Universal LSTM, Universal ResNet, and Single-Domain models from the [FlaxSpeaker Hugging Face Collection](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) trained simultaneously on **LibriSpeech + CN-Celeb2 + VoxCeleb1**.
2. **Real-Time Verification & Scoring**: Supports Cosine Similarity, Parameter-Free Attentive Scoring (PFAS), and Decision Residual Network (Dr-Vector) scoring.
3. **Acoustic & VAD Diagnostics**: Visualizes the 80-bin Log-Mel Spectrogram after Automatic Gain Control (AGC) and Frame-Energy Voice Activity Detection (26 dB below 95th-percentile frame energy), alongside the 192-D speaker embedding profile.
4. **Curated Multi-Domain Examples**: Pre-loaded genuine and impostor trials from English read speech (LibriSpeech), Chinese multi-genre speech (CN-Celeb2), and in-the-wild English celebrity interviews (VoxCeleb1).
