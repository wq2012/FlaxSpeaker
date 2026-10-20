# FlaxSpeaker v0.2.0: Universal Multi-Domain Speaker Recognition Research Report

## Disclaimer & Open-Source Reproduction Notice

> **This is not an officially supported Google product.**
>
> `FlaxSpeaker` is a from-scratch re-implementation in JAX/Flax, written using only the published papers (no proprietary code or data), of speaker recognition algorithms published in:
> 1. **GE2E**: Li Wan, Quan Wang, Alan Papir, and Ignacio Lopez Moreno, *"Generalized End-to-End Loss for Speaker Verification"*, IEEE ICASSP 2018 ([arXiv:1710.10467](https://arxiv.org/abs/1710.10467)).
> 2. **Dr-Vectors**: Jason Pelecanos, Quan Wang, and Ignacio Lopez Moreno, *"Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition"*, Interspeech 2021 ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989)).
> 3. **ATP** (a *language identification* paper whose pooling head is repurposed here): Quan Wang, Yang Yu, Jason Pelecanos, Yiling Huang, and Ignacio Lopez Moreno, *"Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech"*, Odyssey 2022 ([arXiv:2202.12163](https://arxiv.org/abs/2202.12163)).
> 4. **PFAS**: Jason Pelecanos, Quan Wang, Yiling Huang, and Ignacio Lopez Moreno, *"Parameter-Free Attentive Scoring for Speaker Verification"*, Odyssey 2022 ([arXiv:2203.05642v3](https://arxiv.org/abs/2203.05642v3)).
>
> The experiments reported in those original papers were conducted using Google's internal proprietary infrastructure and internal datasets, not this open-source library. All benchmark numbers in this report are measured directly from our open-source JAX/Flax implementation on public datasets (`LibriSpeech`, `CN-Celeb2`, `VoxCeleb1`).
>
> **Read this first.** Every released checkpoint is a **400-step, single-seed CPU demonstration model** (32 utterances per step, ≈1.35 epochs over 9,447 utterances from 381 speakers). pooled 1-utt EERs of 18.75–30.23% (22.5–32.2% on the VoxCeleb1-O 10-speaker subset) are 10–30× worse than published systems; comparisons are only meaningful *between* FlaxSpeaker checkpoints under the same budget and only beyond the ~0.7–0.8% seed standard deviation measured in Section 10. The released `ge2e_contrast`, `ext_softmax`, `dr_vectors` and `pfas*` checkpoints are **FlaxSpeaker variants** of the cited algorithms (hybrid contrast surrogate, blended extended-set softmax, down-scaled Dr-Vector scorer, PFAS-SN head); the paper-faithful presets exist (`pure_paper_mode=True`) but have not been trained. See the README section *Deviations from the Source Papers*.

---

## 1. Executive Summary & Verification Provenance

This report documents a 44-model empirical investigation of **Universal Multi-Domain Speaker Recognition** versus **Single-Domain Speaker Recognition** in `FlaxSpeaker v0.2.0` across 6 neural backbones (`Conformer`, `Transformer`, `Mamba`, `LSTM`, `ECAPA-TDNN`, `ResNet`), 9 training objectives (`GE2E-Softmax`, `GE2E-Contrast`, `Extended-Set Softmax`, `Dr-Vectors`, `PFAS + Extended-Set Softmax`, `PFAS + GE2E-Softmax`, `ArcFace`, `CosFace`, `SphereFace`, `Triplet`), and 3 parameter scales (`tiny`, `small`, `base`, all under 5M parameters), supplemented by 5 controlled ablation studies on multi-domain batching, objective formulations, variable-length training, training horizon scaling (`400` to `1,200` steps), and bidirectional Mamba.

- **Hugging Face Pretrained Model Collection (44 Models)**: [`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) (with individual repositories at `https://huggingface.co/wq2012/<model_id>`)
- **Machine-Readable Metrics Artifact**: [`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json) (and per-model [`pretrained_models/<model_id>/eval_results.json`](pretrained_models/))
- **Controlled Ablation Artifact**: [`pretrained_models/ablation_results.json`](pretrained_models/ablation_results.json)
- **Dataset Splits Manifest**: `./data/speaker_datasets/splits/dataset_summary.json`
- **Execution Command**:
  ```bash
  PYTHONPATH=. python scripts/run_research_benchmarks.py --num_steps 400 --start_idx 0 --end_idx 44
  ```
- **Execution Logs**: Verified across parallel training and evaluation shards `task-938` through `task-943`, `task-1205` through `task-1211`, `task-1246` through `task-1248`, `task-1288` through `task-1291`, `task-1318`, `task-203`, `task-204`, `task-205`, and `task-221`.

![FlaxSpeaker Architecture Diagram](resources/architecture_diagram.png)

### Key Research Highlights

1. **Universal Multi-Domain Training Lowers Pooled/Macro EER for Metric-Learning Objectives (but not in-domain, not for classification losses, and by less than the seed floor at matched data size)**:
   - Models trained on a single dataset suffer severe domain shift on out-of-domain speech: [`ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) (trained only on English read speech) achieves `9.67%` 1-utt EER (`5.50%` 3-utt EER) on LibriSpeech, but degrades to `28.50%` on Chinese multi-genre speech (`CN-Celeb2`). Conversely, [`cn_conformer_small_ge2e`](https://huggingface.co/wq2012/cn_conformer_small_ge2e) (trained only on CN-Celeb2) achieves `21.33%` on CN-Celeb2, but degrades to `20.80%` on LibriSpeech (`+11.13%` worse than in-domain) and `27.20%` on VoxCeleb1 (`23.11%` Macro-Average EER).
   - Training a single universal model on all three domains (`LibriSpeech + CN-Celeb2 + VoxCeleb1`, `381` speakers, `9,447` utterances) with **Domain-Balanced Stratified Batching** (`75%` within-domain hard-negative batches + `25%` cross-domain calibration batches) lowers pooled 1-utt EER relative to the matched LibriSpeech specialist in 11 of 19 matched `univ_*`/`ls_*` pairs (GE2E, extended-set, Dr-Vector and PFAS objectives on Conformer/Mamba), but **not** for ArcFace/CosFace/SphereFace/Triplet, ResNet-18 or ECAPA-TDNN-small/tiny (universal is worse on macro 1-utt EER in 8/19 pairs), and never in-domain (VoxCeleb1 specialist 20.80% vs universal 24.80% on VoxCeleb1). The universal corpus is also 1.6× larger than the LibriSpeech one (Section 10, matched-data control):
     - **[`univ_conformer_tiny_ge2e`](https://huggingface.co/wq2012/univ_conformer_tiny_ge2e) (`693K` params)** achieves **19.26% Macro-Average 1-Utt EER**, **18.75% Pooled 8,000-Trial 1-Utt EER** (`13.23%` LibriSpeech, `22.00%` CN-Celeb2, `22.55%` VoxCeleb1), **15.20% Macro-Average 3-Utt EER**, and **15.03% Pooled 3,000-Trial 3-Utt EER** at `400` training steps.
     - **[`univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) (`2.03M` params)** achieves **19.88% Macro-Average 1-Utt EER** and **19.40% Pooled 8,000-Trial 1-Utt EER** (`12.50%` LibriSpeech, `22.33%` CN-Celeb2, `24.80%` VoxCeleb1) at `400` steps — and when scaled to `1,200` training steps (Section 10), further improves to **18.50% Macro-Average 1-Utt EER**, **18.48% Pooled 1-Utt EER**, and **14.17% Macro-Average 3-Utt EER**.
2. **Conformer with (multi-head, FlaxSpeaker-variant) Attentive Temporal Pooling Has the Lowest Pooled EER Among the 6 Backbones, Within Overlapping CIs of Transformer and Mamba**:
   - On both Universal Multi-Domain training and Single-Domain training, **Conformer-ATP** has the lowest single-seed pooled 1-utt EER; the gaps to Transformer-ATP and Mamba are inside the 95% CIs and below the seed floor:
     - **Universal Multi-Domain (1-Utt Pooled / 3-Utt Macro-Avg EER)**: `Conformer-ATP` (`19.40%` / `16.17%`) and `Transformer-ATP` (`20.00%` / `15.90%`) and `Mamba` (`20.41%` / `15.23%`) outperform `LSTM` (`21.31%` / `17.43%`), `ECAPA-TDNN` (`23.16%` / `17.67%`), and `ResNet` (`27.89%` / `24.53%`).
     - **LibriSpeech In-Domain (1-Utt / 3-Utt EER)**: `Conformer-ATP` (`9.67%` / `5.50%`) > `Transformer-ATP` (`10.60%` / `6.20%`) > `Mamba` (`12.17%` / `7.90%`) > `LSTM` (`12.77%` / `7.90%`) > `ECAPA-TDNN` (`13.83%` / `9.10%`) > `ResNet` (`15.20%` / `11.30%`).
3. **Down-Scaled Dr-Vector Scorer, PFAS-SN, and the Hybrid GE2E-Contrast Surrogate (all FlaxSpeaker variants) Have the Lowest 3-Utt EERs Among Universal Conformer-Small Models (overlapping CIs)**:
   - **Universal Dr-Vectors ([`univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors))**: Achieves **15.20% Macro-Average 3-Utt EER** (`9.90%` LibriSpeech, `18.20%` CN-Celeb2, `17.50%` VoxCeleb1) and **15.40% Pooled 3-Utt EER** (`minDCF@0.01 = 0.7500`, `ROC-AUC = 0.9267`), improving over its base [`univ_conformer_small_ext_softmax`](https://huggingface.co/wq2012/univ_conformer_small_ext_softmax) cosine model (`16.17%` Macro 3-Utt EER, `minDCF@0.01 = 0.8973`) across every metric.
   - **Universal GE2E-Contrast surrogate ([`univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast))**: trained with the **hybrid contrast surrogate** (loss value = Wan et al. Eq. 7, but gradient = 0.25·Eq. 7 + 0.25·BCE + 0.50·GE2E-softmax-CE, `ge2e_init_b = -1`, `b` gradient scale 0.5 — a FlaxSpeaker objective, **not** the paper's contrast loss; `ge2e_contrast_surrogate: false` gives the pure Eq. 7 gradient), achieves **14.93% Macro-Average 3-Utt EER** (`8.30%` LibriSpeech, `17.10%` CN-Celeb2, `19.40%` VoxCeleb1) and **15.33% Pooled 3-Utt EER** (`ROC-AUC = 0.9232`), along with **11.43% LibriSpeech 1-Utt EER** (`7.10%` 3-Utt EER) on [`ls_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/ls_conformer_small_ge2e_contrast).
   - **Universal PFAS ([`univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas) & [`univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e))**:
     - [`univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas) achieves **15.80% Macro-Average 3-Utt EER** (`16.00%` Pooled 3-Utt EER) and **0.8967 Macro-Average 1-Utt minDCF@0.01** (the best 1-utt minDCF of any universal model, beating cosine scoring's `0.9635` by **-0.0668**).
     - [`univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e) achieves **21.93% 1-Utt EER** and **17.40% 3-Utt EER on CN-Celeb2**.

---

## 2. Original Published Paper Reference Results vs. Open-Source Reproduction Context

The table below summarizes the reference results reported in the four original publications (public text only) next to the FlaxSpeaker setting. The two columns are **not comparable**: the papers train for ~500k steps on thousands of speakers with multi-utterance enrollment; FlaxSpeaker trains 400 steps on ≤381 speakers. No reproduction of any published number is claimed.

| Paper & Algorithm | Original Paper Training Corpus (Internal / Public) | Original Paper Reported Benchmark & Metric | FlaxSpeaker v0.2.0 Open-Source Reproduction Result (Public Corpora, < 5M Params) |
| :--- | :--- | :--- | :--- |
| **GE2E-Softmax & GE2E-Contrast** ([Wan et al., ICASSP 2018](https://arxiv.org/abs/1710.10467)) | Google Internal Voice Search (`~18K` speakers, `~36M` utterances) + TD-SV / TI-SV | TI-SV (Table 3 of the paper; ~6.3 enrollment utterances per speaker, 1,000 test speakers): **3.55% EER** for GE2E vs 4.06% softmax and 4.13% TE2E (`3-layer LSTM`, `768` hidden, `256` proj). Multi-utterance enrollment — not comparable to the 1-utt numbers in the next column. | [`ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e): **9.67% 1-Utt EER** / **5.50% 3-Utt EER** (LibriSpeech); [`ls_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/ls_conformer_small_ge2e_contrast): **11.43% 1-Utt EER** / **7.10% 3-Utt EER**; [`univ_conformer_tiny_ge2e`](https://huggingface.co/wq2012/univ_conformer_tiny_ge2e): **18.75% Pooled 1-Utt EER** / **15.03% Pooled 3-Utt EER** |
| **Dr-Vectors + Extended-Set Softmax** ([Pelecanos et al., Interspeech 2021](https://arxiv.org/abs/2104.01989)) | Google Internal Vendor English + Multi-Condition (`8K`+ speakers) | Relative EER reduction of `~4%–9%` from Extended-Set Softmax + Decision Residual Network over GE2E baseline | [`univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors): **15.20% Macro 3-Utt EER** / **15.40% Pooled 3-Utt EER** (`minDCF@0.01 = 0.7500`), improving `-0.97%` absolute EER (`-6.0%` relative) over [`univ_conformer_small_ext_softmax`](https://huggingface.co/wq2012/univ_conformer_small_ext_softmax) cosine (`16.17%`) |
| **Conformer with Attentive Temporal Pooling (ATP)** ([Wang et al., Odyssey 2022](https://arxiv.org/abs/2202.12163)) | Google Internal Multilingual / Long-Form Speech | Multi-head Attentive Temporal Pooling (`mean ⊕ std` with shared bottleneck `b = 128`) stabilizes Conformer sequence pooling | [`ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) (4-head ATP): **9.67% 1-Utt EER** / **5.50% 3-Utt EER** on LibriSpeech, improving `-6.86%` absolute 1-Utt EER over single-query ASP (`16.53%`) and beating ECAPA-TDNN (`13.83%`) |
| **Parameter-Free Attentive Scoring (PFAS)** ([Pelecanos et al., Odyssey 2022](https://arxiv.org/abs/2203.05642v3)) | Google Internal Multi-Enrollment Speaker Verification | Cross-attention over `(K, V)` frame subsets improves multi-enrollment EER and minDCF without adding scoring parameters | [`univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas): **15.80% Macro 3-Utt EER** and **0.8967 Macro 1-Utt minDCF@0.01** (best universal minDCF); [`ls_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_pfas_ge2e): **0.6627 1-Utt minDCF@0.01** (`-18.5%` relative vs cosine `0.8133`) |

---

## 3. Multi-Domain Dataset & Evaluation Protocol

All datasets are stored under `./data/speaker_datasets` (or `$FLAXSPEAKER_DATA_ROOT`) and processed with strictly speaker-disjoint training and evaluation splits (`scripts/prepare_datasets.py`), plus evaluation on the **VoxCeleb1-O 10-speaker subset**: the official `veri_test2.txt` list filtered to trials whose two speakers are both among the 10 held-out evaluation speakers (`2,500` trials: `1,789` target, `711` impostor, 71.6% target, across `937` unique utterances). **This subset is not comparable to published VoxCeleb1-O numbers** (37,720 balanced trials over 40 speakers after VoxCeleb1-dev/VoxCeleb2 training); the other 30 of the 40 official test speakers are *training* data for the `vox_*` and `univ_*` models. One VAD constant (26 dB) is used for training and all evaluations:

| Dataset Split | Domain & Language | Train Speakers | Train Utterances | Eval Speakers (Unseen) | Eval Utterances | 1-Utt Trials | 3-Utt Multi-Enroll Trials |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`librispeech`** | English Read Audiobooks (`train-clean-100` + `dev-clean` / `test-clean`, 16 kHz) | 232 | 5,788 | 40 | 1,000 | 3,000 (`50%` Pos / `50%` Neg) | 1,000 |
| **`cnceleb2`** | Chinese Multi-Genre Speech (`CN-Celeb2`娛樂/Interview/Singing/Vlog, 16 kHz) | 119 | 2,909 | 30 | 739 | 3,000 (`50%` Pos / `50%` Neg) | 1,000 |
| **`voxceleb1`** | English In-the-Wild Celebrity Interviews (`VoxCeleb1`, 16 kHz) | 30 (`seed=42`) | 750 | 10 (`seed=42`, unseen) | 250 | 2,000 (`50%` Pos / `50%` Neg) | 1,000 |
| **`vox1_o_10spk_subset`** | VoxCeleb1-O **10-speaker subset** of `veri_test2.txt` (not the published protocol) | — (Eval-Only) | — | 10 (`id10271, id10275, id10276, id10277, id10278, id10284, id10285, id10287, id10297, id10308`) | 937 | 2,500 (`1,789` Pos / `711` Neg, 71.6% target) | — |
| **`universal`** | **Combined Multi-Domain Corpus (`librispeech + cnceleb2 + voxceleb1`)** | **381** | **9,447** | **80** | **1,989** | **8,000** | **3,000** |

### Critical Multi-Domain Engineering Details

1. **Domain-Prefixed Speaker Namespaces**: Raw speaker IDs in `CN-Celeb2` (`id10029..id11975`) and `VoxCeleb1` (`id10270..id10309`) collide on 5 speaker IDs (`id10272`, `id10293`, `id10295`, `id10297`, `id10309`). In `universal/train.csv` and `universal/eval.csv`, every speaker ID is prefixed with `<domain>:<spk_id>` (`librispeech:103`, `cnceleb2:id10272`, `voxceleb1:id10272`) to prevent cross-dataset speaker merging.
2. **Domain-Balanced Stratified Batching (`flaxspeaker/feature_extraction.py`)**:
   - Naively sampling speakers uniformly from the concatenated 381-speaker pool under-represents `VoxCeleb1` (`30` speakers) relative to `LibriSpeech` (`232` speakers) and `CN-Celeb2` (`119` speakers) and mixes acoustic domains inside every GE2E / Extended-Set Softmax batch (making negative rejection trivially solvable via channel/language cues rather than voice identity).
   - Our multi-domain sampler (`_sample_speakers_domain_balanced`) uses temperature-balanced domain probabilities `p_d proportional to |S_d|^0.35` and alternates between **75% within-domain hard-negative batches** (all `N = 8` speakers drawn from the same domain `d`, forcing fine-grained speaker discrimination) and **25% cross-domain calibration batches** (speakers stratified across all 3 domains, aligning cross-domain score calibration).

---

## 4. Deep-Dive Answers to Core Architectural & Algorithmic Questions

### 4.1 Why Conformer Originally Underperformed — and How Conformer + ATP (`arXiv:2202.12163`) Became #1
1. **Conformer Block Architecture (`flaxspeaker/backbones.py`)**:
   - Implements the Macaron-style Conformer block (`HalfStepFFN -> MultiHeadSelfAttention -> ConformerConvModule -> HalfStepFFN -> LayerNorm`) with depthwise separable convolution (`kernel_size = 15`), GLU gating, Mask-Aware GroupNorm (`G = 2` groups), and Multi-Layer Feature Aggregation (MFA), preceded by a frame-rate-preserving ($T'=T$) parallel linear + convolutional stem (`Dense(D) + SiLU(LayerNorm(Conv1D(k=3)))`) with relative local context captured by the stem and depthwise convolutions (`use_positional_encoding = False` by default; `odyssey2022_conformer_atp_config(pure_paper_mode=True)` sets `layer_order = "conv_before_mhsa"`, `use_conv_stem = False`, `use_positional_encoding = True`, `subsample_factor = 2`, single-head linear-sigmoid ATP `attention_dim = 0, num_heads = 1`; every one of these fields is consumed by the built module and unit-tested).
2. **Attentive Temporal Pooling (`ATP`, `arXiv:2202.12163`, `flaxspeaker/pooling.py`)**:
   - **FlaxSpeaker variant** of the single-head linear-sigmoid ATP of Wang et al. (Odyssey 2022, a language-identification paper): adds a shared tanh bottleneck `d_a = 128` (`z_t = tanh(W_1 h_t + b_1)`) and `N_h = 4` query heads computing masked sigmoid temporal attention weights `alpha_{t,k}`, weighted mean `mu_k` and standard deviation `sigma_k` over channel subspaces (`subspace_heads = True`), concatenated across all heads (`2 * D`) and passed through a nonlinear bottleneck FFN (`LayerNorm -> Dense(256) -> LayerNorm -> ReLU -> Dense(192) -> L2_norm`).
   - **Impact**: Upgrading Conformer-Small from single-query ASP to 4-head ATP + parallel ConvStem + Cosine Warmup (`1.5e-3`) reduced LibriSpeech 1-Utt EER from **`16.53%` → `9.67%`** (`3-Utt EER = 5.50%`), outperforming ECAPA-TDNN (`13.83%`).

### 4.2 Parameter-Free Attentive Scoring (`PFAS`, `arXiv:2203.05642v3`)
1. **Key-Value Representation & Two-Stage Concatenated Value L2 Normalization (`flaxspeaker/pooling.py`, `flaxspeaker/scoring.py`)**:
   - The released `*_pfas*` checkpoints use the **PFAS-SN head, a FlaxSpeaker extension** (`attention_type = shared_nonlinear` + global ATP anchor); the paper's head is `attention_type = atp_ffn` (`odyssey2022_pfas_config(pure_paper_mode=True)`: single-head ATP → affine + ReLU → linear, 32 tied keys × (16 + 48)). With `K = 4` learnable frame-selection queries, `PfasRepresentationHead` extracts `K` L2-normalized key vectors `k_i` (`dim = 32`, `apply_l2_norm_to_keys = True`) and `K` value vectors `v_i = v_global + 0.25 * v_local,i` (`dim = 128`, `apply_l2_norm_to_values = False`; when `use_global_anchor = False` in `odyssey2022_pfas_config(pure_paper_mode=True)`, `v_i = v_local,i`). When `apply_global_l2_norm_to_concat_form = True` (default), `PfasRepresentationHead` first globally L2-normalizes the unweighted concatenated value vector `[v_1; ...; v_K]` in `R^{K * d_v}` at pooling time (`flaxspeaker/pooling.py:410-413`), packing the normalized keys and values into a `640-D` representation (`4 * (32 + 128)`).
2. **Pairwise Cross-Attention & Trial-Time Attention-Weighted Normalization**:
   - For each trial pair `(test, enroll)`, PFAS computes the cross-similarity matrix `S_{i,j} = <k_i^test, k_j^enroll>`, computes normalized attention weights `A_{i,j}` over the `K_t x K_e` key grid (`sum_{i,j} A_{i,j} = 1`, with `scale_factor = 16.0` or learnable `_aux_pfas_log_scale`), and normalizes the aligned test and enrollment value matrices dynamically using attention-weighted global concatenated L2 norms (`nu_t = sqrt(sum_i (sum_j A_{i,j}) ||v_i^test||^2)` and `nu_e = sqrt(sum_j (sum_i A_{i,j}) ||v_j^enroll||^2)`, returning `sum_{i,j} A_{i,j} <v_i^test / nu_t, v_j^enroll / nu_e>`).

### 4.3 Voice Activity Detection (VAD) & Automatic Gain Control (AGC) (`flaxspeaker/vad.py`)
- Enabled identically in both training and evaluation (`vad.enabled = True`, `vad.mode = "energy"`):
  1. Peak-normalizes waveform amplitude to `[-1, 1]` (`apply_agc = True`, target peak `1.0`).
  2. `VadProcessor` computes rectangular-frame log-RMS energy in dB (`25 ms` window, `10 ms` hop) on the normalized *waveform*; the resulting mask is applied to the log-Mel frames before utterance CMVN, retaining frames exceeding the 3-term dynamic threshold `max(-42.0 dB, P_95(E) - 26.0 dB, P_10(E) + 4.5 dB)` (with optional spectral flatness `<= 0.65` and speech-band `300-3400 Hz` energy ratio `>= 0.25` in `spectral`/`hybrid` modes), followed by `min_speech_frames = 3` and `hangover_frames = 8` (`80 ms`) smoothing and a top-`min(max(min_speech_frames, 8), T)` frame fallback for very quiet recordings.

### 4.4 Mutually Exclusive Data Augmentation (`flaxspeaker/specaug.py`)
- Enforces **strict mutual exclusion** (`augment_features_mutually_exclusive`) on every training utterance. The "MTR" branch is **synthetic log-Mel-domain** augmentation (frequency-shaped Gaussian noise gated towards low-energy bins at `SNR ~ U(12, 20) dB`, plus a one-tap temporal smoothing with probability 0.4) — **not** waveform multi-style training with recorded noise/RIR corpora. The standalone waveform helper `apply_waveform_augmentation` is not on the training path:
  - `35%` feature-domain noise *only* (`aug_prob = 0.35`, `snr_min_db = 12`, `snr_max_db = 20`)
  - `32.5%` SpecAugment *only* (`(1 - 0.35) * 0.50 = 0.325`, up to `2` frequency masks `<= 8` mel bins and `2` time masks `<= 12` frames, each sampled independently with probability `0.5`)
  - `32.5%` clean unaugmented features
  - Never applies SpecAugment and multi-style noise simultaneously.

### 4.5 Variable-Length Masked Training & Full-Sequence Inference
- Training utterances are sampled with variable duration `T ~ Uniform(90, 180)` frames (`0.9 s` to `1.8 s`; identical in all 44 benchmark YAML configs) with an explicit binary frame mask `(B, T)` propagated through every Conformer/Transformer/ECAPA-TDNN/ResNet/Mamba/LSTM layer and temporal pooling head so padded frames receive zero weight.
- At evaluation time (`full_sequence_inference = True`), utterances up to `max_eval_seq_len = 500` frames (`5.0 s`) are encoded in a single masked forward pass.

---

## 5. Study 5: Universal Multi-Domain Neural Backbones (`LibriSpeech + CN-Celeb2 + VoxCeleb1`)

All models in Study 5 are `small` variants (`1.0M–2.2M` parameters) trained for `400` steps (`batch_size = 32`, `N = 8` speakers × `M = 4` utterances) on the **Universal Multi-Domain training set (`381` speakers, `9,447` utterances)** using `GE2E-Softmax` and evaluated on all 3 unseen speaker-disjoint evaluation domains (`LibriSpeech` 3,000 trials, `CN-Celeb2` 3,000 trials, `VoxCeleb1` 2,000 trials), their **Universal Macro-Average**, and the **Universal Pooled 8,000-Trial / 3,000-Trial** sets:

### 5.1 Universal Backbones — 1-Utterance Verification Results

| Hugging Face Repository | Backbone (`Pooling`) | Params | LibriSpeech EER (%) [95% CI] | CN-Celeb2 EER (%) [95% CI] | VoxCeleb1 EER (%) [95% CI] | Universal Macro-Avg EER (%) | Universal Pooled EER (%) [95% CI] | Pooled ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | `conformer` (`atp`) | 2,029,894 | **12.50%** [11.16, 13.63] | **22.33%** [20.73, 23.78] | 24.80% [22.55, 26.60] | **19.88%** | **19.40%** [18.61, 20.26] | **0.8906** |
| **[`wq2012/univ_mamba_small_ge2e`](https://huggingface.co/wq2012/univ_mamba_small_ge2e)** | `mamba` (`asp`) | **1,028,674** | 14.20% [12.77, 15.40] | 23.20% [21.63, 24.70] | 23.75% [21.72, 25.40] | 20.38% | 20.41% [19.64, 21.33] | 0.8846 |
| **[`wq2012/univ_transformer_small_ge2e`](https://huggingface.co/wq2012/univ_transformer_small_ge2e)** | `transformer` (`atp`) | 1,239,622 | 15.17% [13.95, 16.60] | 22.67% [20.96, 24.27] | 23.95% [21.87, 25.80] | 20.59% | 20.00% [19.25, 20.90] | 0.8831 |
| **[`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e)** | `lstm` (`asp`) | 1,710,914 | 14.97% [13.80, 16.35] | 25.53% [23.83, 26.98] | **22.50%** [20.70, 24.20] | 21.00% | 21.31% [20.42, 22.15] | 0.8730 |
| **[`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e)** | `ecapa_tdnn` (`asp`) | 1,684,418 | 19.00% [17.52, 20.45] | 25.13% [23.67, 26.68] | 23.30% [21.07, 24.90] | 22.48% | 23.16% [22.25, 24.03] | 0.8536 |
| **[`wq2012/univ_resnet_small_ge2e`](https://huggingface.co/wq2012/univ_resnet_small_ge2e)** | `resnet` (`asp`) | 2,239,922 | 24.53% [23.10, 26.17] | 29.33% [27.67, 30.93] | 28.25% [26.60, 30.55] | 27.37% | 27.89% [26.88, 28.82] | 0.7989 |

### 5.2 Universal Backbones — 3-Utterance Multi-Enrollment Verification Results

| Hugging Face Repository | Backbone (`Pooling`) | LibriSpeech 3-Utt EER (%) | CN-Celeb2 3-Utt EER (%) | VoxCeleb1 3-Utt EER (%) | Universal Macro-Avg 3-Utt EER (%) | Universal Pooled 3-Utt EER (%) [95% CI] | Macro 3-Utt minDCF@0.01 | Macro 3-Utt ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/univ_mamba_small_ge2e`](https://huggingface.co/wq2012/univ_mamba_small_ge2e)** | `mamba` (`asp`) | 9.50% | **18.50%** | **17.70%** | **15.23%** | 16.07% [14.82, 17.38] | **0.7920** | **0.9252** |
| **[`wq2012/univ_transformer_small_ge2e`](https://huggingface.co/wq2012/univ_transformer_small_ge2e)** | `transformer` (`atp`) | 10.70% | 19.00% | 18.00% | 15.90% | 16.33% [14.90, 17.64] | 0.8427 | 0.9159 |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | `conformer` (`atp`) | **8.80%** | 20.00% | 19.70% | 16.17% | **16.03%** [14.73, 17.25] | 0.8433 | 0.9198 |
| **[`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e)** | `lstm` (`asp`) | 11.20% | 21.60% | 19.50% | 17.43% | 18.00% [16.67, 19.33] | 0.9173 | 0.9089 |
| **[`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e)** | `ecapa_tdnn` (`asp`) | 11.80% | 22.40% | 18.80% | 17.67% | 18.00% [16.77, 19.40] | 0.9120 | 0.9029 |
| **[`wq2012/univ_resnet_small_ge2e`](https://huggingface.co/wq2012/univ_resnet_small_ge2e)** | `resnet` (`asp`) | 21.40% | 26.40% | 25.80% | 24.53% | 25.47% [23.85, 27.18] | 0.9847 | 0.8343 |

---

## 6. Study 6: Universal Multi-Domain Losses & Scoring (`Conformer-Small` on 3-Dataset Mix)

Holding the backbone fixed (`Conformer-Small`, `3` layers, `d = 192`, `4` heads, `2,029,894` parameters), we compare all 9 loss and scoring combinations on the **Universal Multi-Domain** corpus:

### 6.1 Universal Losses & Scoring — 1-Utterance Verification

| Hugging Face Repository | Loss Function | Scoring | Pooling | LibriSpeech EER (%) | CN-Celeb2 EER (%) | VoxCeleb1 EER (%) | Macro-Avg 1-Utt EER (%) | Pooled 1-Utt EER (%) [95% CI] | Macro 1-Utt minDCF@0.01 | Pooled 1-Utt ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | `ge2e_softmax` | `cosine` | `atp` | **12.50%** | 22.33% | 24.80% | **19.88%** | **19.40%** [18.61, 20.26] | 0.9359 | **0.8906** |
| **[`wq2012/univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors)** | `extended_set_softmax` | **`dr_vector`** | `atp` | 15.20% | 22.33% | **22.75%** | 20.09% | 20.10% [19.31, 20.96] | 0.9348 | 0.8866 |
| **[`wq2012/univ_conformer_small_ext_softmax`](https://huggingface.co/wq2012/univ_conformer_small_ext_softmax)** | `extended_set_softmax` | `cosine` | `atp` | 14.83% | 22.40% | 23.30% | 20.18% | 20.50% [19.64, 21.26] | 0.9635 | 0.8808 |
| **[`wq2012/univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e)** | `ge2e_softmax` | **`pfas`** | `pfas` | 15.20% | **21.93%** | 23.65% | 20.26% | 19.86% [19.09, 20.77] | 0.9741 | 0.8869 |
| **[`wq2012/univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas)** | `extended_set_softmax` | **`pfas`** | `pfas` | 15.83% | 23.17% | 22.95% | 20.65% | 20.35% [19.48, 21.21] | **0.8967** | 0.8888 |
| **[`wq2012/univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast)** | `ge2e_contrast` | `cosine` | `atp` | 14.73% | 23.27% | 24.45% | 20.82% | 20.69% [19.83, 21.50] | 0.9396 | 0.8758 |
| **[`wq2012/univ_conformer_small_triplet`](https://huggingface.co/wq2012/univ_conformer_small_triplet)** | `triplet` | `cosine` | `atp` | 18.50% | 26.67% | 25.40% | 23.52% | 24.07% [23.12, 25.02] | 0.9477 | 0.8491 |
| **[`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface)** | `arcface` | `cosine` | `atp` | 20.57% | 25.27% | 25.00% | 23.61% | 25.11% [24.30, 26.06] | 0.9869 | 0.8306 |
| **[`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface)** | `cosface` | `cosine` | `atp` | 20.37% | 25.73% | 26.40% | 24.17% | 25.60% [24.64, 26.58] | 0.9914 | 0.8256 |
| **[`wq2012/univ_conformer_small_sphereface`](https://huggingface.co/wq2012/univ_conformer_small_sphereface)** | `sphereface` | `cosine` | `atp` | 22.27% | 25.67% | 25.15% | 24.36% | 25.84% [24.83, 26.82] | 0.9858 | 0.8248 |

### 6.2 Universal Losses & Scoring — 3-Utterance Multi-Enrollment Verification

| Hugging Face Repository | Loss Function | Scoring | LibriSpeech 3-Utt EER (%) | CN-Celeb2 3-Utt EER (%) | VoxCeleb1 3-Utt EER (%) | Macro-Avg 3-Utt EER (%) | Pooled 3-Utt EER (%) [95% CI] | Macro 3-Utt minDCF@0.01 | Pooled 3-Utt ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast)** | `ge2e_contrast` | `cosine` | **8.30%** | **17.10%** | 19.40% | **14.93%** | **15.33%** [14.10, 16.67] | 0.8613 | 0.9232 |
| **[`wq2012/univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors)** | `extended_set_softmax` | **`dr_vector`** | 9.90% | 18.20% | **17.50%** | 15.20% | 15.40% [14.08, 16.67] | **0.7500** | **0.9267** |
| **[`wq2012/univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas)** | `extended_set_softmax` | **`pfas`** | 10.80% | 18.60% | 18.00% | 15.80% | 16.00% [14.73, 17.33] | 0.8060 | 0.9251 |
| **[`wq2012/univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e)** | `ge2e_softmax` | **`pfas`** | 11.00% | 17.40% | 19.80% | 16.07% | 16.27% [15.00, 17.73] | 0.8727 | 0.9179 |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | `ge2e_softmax` | `cosine` | 8.80% | 20.00% | 19.70% | 16.17% | 16.03% [14.73, 17.25] | 0.8433 | 0.9212 |
| **[`wq2012/univ_conformer_small_ext_softmax`](https://huggingface.co/wq2012/univ_conformer_small_ext_softmax)** | `extended_set_softmax` | `cosine` | 9.80% | 18.80% | 19.90% | 16.17% | 16.17% [14.98, 17.53] | 0.8973 | 0.9217 |
| **[`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface)** | `cosface` | `cosine` | 17.10% | 22.20% | 20.30% | 19.87% | 21.83% [20.17, 23.20] | 0.9193 | 0.8720 |
| **[`wq2012/univ_conformer_small_triplet`](https://huggingface.co/wq2012/univ_conformer_small_triplet)** | `triplet` | `cosine` | 14.40% | 24.00% | 22.00% | 20.13% | 20.33% [18.95, 21.55] | 0.9247 | 0.8854 |
| **[`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface)** | `arcface` | `cosine` | 17.40% | 21.40% | 21.80% | 20.20% | 21.67% [20.03, 23.17] | 0.9160 | 0.8730 |
| **[`wq2012/univ_conformer_small_sphereface`](https://huggingface.co/wq2012/univ_conformer_small_sphereface)** | `sphereface` | `cosine` | 18.10% | 23.40% | 20.40% | 20.63% | 22.07% [20.40, 23.50] | 0.9347 | 0.8663 |

---

## 7. Study 7: Universal Multi-Domain Model Scaling (`Tiny` vs. `Small` vs. `Base`)

| Hugging Face Repository | Backbone & Size | Params | LibriSpeech 1-Utt EER (%) | CN-Celeb2 1-Utt EER (%) | VoxCeleb1 1-Utt EER (%) | Macro-Avg 1-Utt EER (%) | Pooled 1-Utt EER (%) [95% CI] | Macro-Avg 3-Utt EER (%) | Pooled 3-Utt EER (%) [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/univ_conformer_tiny_ge2e`](https://huggingface.co/wq2012/univ_conformer_tiny_ge2e)** | `conformer-tiny` | **693,126** | 13.23% | **22.00%** | **22.55%** | **19.26%** | **18.75%** [17.89, 19.68] | **15.20%** | **15.03%** [13.67, 16.20] |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | `conformer-small` | 2,029,894 | **12.50%** | 22.33% | 24.80% | 19.88% | 19.40% [18.61, 20.26] | 16.17% | 16.03% [14.73, 17.25] |
| **[`wq2012/univ_conformer_base_ge2e`](https://huggingface.co/wq2012/univ_conformer_base_ge2e)** | `conformer-base` | 4,551,558 | 15.37% | 22.40% | 22.85% | 20.21% | 19.85% [19.04, 20.76] | 16.50% | 16.80% [15.27, 18.02] |
| **[`wq2012/univ_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_tiny_ge2e)** | `ecapa_tdnn-tiny` | **626,722** | 19.07% | 25.27% | 23.35% | 22.56% | 23.02% [22.14, 23.96] | 17.03% | 18.00% [16.75, 19.37] |
| **[`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e)** | `ecapa_tdnn-small` | 1,684,418 | 19.00% | **25.13%** | 23.30% | 22.48% | 23.16% [22.25, 24.03] | 17.67% | 18.00% [16.77, 19.40] |
| **[`wq2012/univ_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_base_ge2e)** | `ecapa_tdnn-base` | 3,190,626 | **18.63%** | 25.97% | **22.20%** | **22.27%** | **22.81%** [21.91, 23.86] | **16.90%** | 18.20% [16.96, 19.70] |

---

## 8. Universal Multi-Domain vs. Single-Domain Direct Comparison

To quantify why universal multi-domain training is essential for real-world deployment, the table below compares identical `Conformer-Small` models trained on single domains (`librispeech`, `cnceleb2`, `voxceleb1`) versus the Universal 3-Dataset Mix (`universal`):

| Hugging Face Repository | Training Dataset (`Speakers / Utts`) | LibriSpeech 1-Utt / 3-Utt EER (%) | CN-Celeb2 1-Utt / 3-Utt EER (%) | VoxCeleb1 1-Utt / 3-Utt EER (%) | Universal Macro-Avg 1-Utt / 3-Utt EER (%) | Universal Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) | LibriSpeech Only (`232 / 5,788`) | **9.67% / 5.50%** | 28.50% / 26.10% | 22.00% / 17.50% | 20.06% / 16.37% | 20.12% / 16.73% |
| [`wq2012/cn_conformer_small_ge2e`](https://huggingface.co/wq2012/cn_conformer_small_ge2e) | CN-Celeb2 Only (`119 / 2,909`) | 20.80% / 16.00% | **21.33% / 16.80%** | 27.20% / 23.90% | 23.11% / 18.90% | 23.93% / 20.13% |
| [`wq2012/vox_conformer_small_ge2e`](https://huggingface.co/wq2012/vox_conformer_small_ge2e) | VoxCeleb1 Only (`30 / 750`) | 16.33% / 11.70% | 27.87% / 25.80% | **20.80% / 16.40%** | 21.67% / 17.97% | 22.40% / 17.63% |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | **Universal 3-Dataset Mix (`381 / 9,447`)** | 12.50% / 8.80% | 22.33% / 20.00% | 24.80% / 19.70% | **19.88% / 16.17%** | **19.40% / 16.03%** |
| **[`wq2012/univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors)** | **Universal 3-Dataset Mix (`381 / 9,447`)** | 15.20% / 9.90% | 22.33% / 18.20% | 22.75% / 17.50% | 20.09% / **15.20%** | 20.10% / **15.40%** |
| **[`wq2012/univ_conformer_tiny_ge2e`](https://huggingface.co/wq2012/univ_conformer_tiny_ge2e)** | **Universal 3-Dataset Mix (`381 / 9,447`)** | 13.23% / 8.80% | 22.00% / 19.20% | 22.55% / 17.60% | **19.26% / 15.20%** | **18.75% / 15.03%** |

---

## 9. Studies 1–4: Complete Single-Domain Baselines Evaluated Across All 3 Domains

Every single-domain baseline model (`25` models) was also evaluated across all 3 evaluation domains (`LibriSpeech`, `CN-Celeb2`, `VoxCeleb1`) and the Universal Pooled set:

| Hugging Face Repository | Train Domain | Backbone (`Size`) | Loss / Scoring | Params | LibriSpeech 1-Utt / 3-Utt EER (%) | CN-Celeb2 1-Utt / 3-Utt EER (%) | VoxCeleb1 1-Utt / 3-Utt EER (%) | Macro-Avg 1-Utt / 3-Utt EER (%) | Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e)** | `librispeech` | `conformer` (`small`) | `ge2e_softmax` / `cosine` | 2,029,894 | **9.67% / 5.50%** | 28.50% / 26.10% | 22.00% / 17.50% | **20.06%** / 16.37% | **20.12%** / 16.73% |
| **[`wq2012/ls_conformer_small_ext_softmax`](https://huggingface.co/wq2012/ls_conformer_small_ext_softmax)** | `librispeech` | `conformer` (`small`) | `ext_softmax` / `cosine` | 2,029,894 | 10.03% / 6.20% | 29.87% / 25.50% | 24.10% / 20.20% | 21.33% / 17.30% | 21.05% / 17.53% |
| **[`wq2012/ls_conformer_small_pfas`](https://huggingface.co/wq2012/ls_conformer_small_pfas)** | `librispeech` | `conformer` (`small`) | `ext_softmax` / `pfas` | 2,162,187 | 10.50% / 6.20% | 28.83% / 24.90% | 22.90% / 18.70% | 20.74% / 16.60% | 20.88% / 16.80% |
| **[`wq2012/ls_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_pfas_ge2e)** | `librispeech` | `conformer` (`small`) | `ge2e_softmax` / `pfas` | 2,162,187 | 10.50% / 6.20% | 28.87% / 25.00% | 23.40% / 20.20% | 20.92% / 17.13% | 20.69% / 17.17% |
| **[`wq2012/ls_conformer_tiny_ge2e`](https://huggingface.co/wq2012/ls_conformer_tiny_ge2e)** | `librispeech` | `conformer` (`tiny`) | `ge2e_softmax` / `cosine` | 693,126 | 10.53% / 6.20% | 30.00% / 25.80% | 25.70% / 21.60% | 22.08% / 17.87% | 21.91% / 18.07% |
| **[`wq2012/ls_transformer_small_ge2e`](https://huggingface.co/wq2012/ls_transformer_small_ge2e)** | `librispeech` | `transformer` (`small`) | `ge2e_softmax` / `cosine` | 1,239,622 | 10.60% / 6.20% | 28.10% / 23.30% | 22.30% / 16.90% | 20.33% / **15.47%** | 20.80% / **16.37%** |
| **[`wq2012/ls_conformer_base_ge2e`](https://huggingface.co/wq2012/ls_conformer_base_ge2e)** | `librispeech` | `conformer` (`base`) | `ge2e_softmax` / `cosine` | 4,551,558 | 10.77% / 6.60% | 28.87% / 23.90% | 24.65% / 19.90% | 21.43% / 16.80% | 21.26% / 17.00% |
| **[`wq2012/ls_conformer_small_dr_vectors`](https://huggingface.co/wq2012/ls_conformer_small_dr_vectors)** | `librispeech` | `conformer` (`small`) | `ext_softmax` / `dr_vector` | 2,088,007 | 11.17% / 6.20% | 28.80% / 24.00% | 22.30% / 19.60% | 20.76% / 16.60% | 20.69% / 16.60% |
| **[`wq2012/ls_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/ls_conformer_small_ge2e_contrast)** | `librispeech` | `conformer` (`small`) | `ge2e_contrast` / `cosine` | 2,029,894 | 11.43% / 7.10% | 29.57% / 25.10% | 25.75% / 20.40% | 22.25% / 17.53% | 22.80% / 19.07% |
| **[`wq2012/ls_mamba_small_ge2e`](https://huggingface.co/wq2012/ls_mamba_small_ge2e)** | `librispeech` | `mamba` (`small`) | `ge2e_softmax` / `cosine` | 1,028,674 | 12.17% / 7.90% | 29.13% / 23.30% | 21.05% / 16.10% | 20.78% / 15.77% | 21.30% / 17.10% |
| **[`wq2012/ls_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_tiny_ge2e)** | `librispeech` | `ecapa_tdnn` (`tiny`) | `ge2e_softmax` / `cosine` | 626,722 | 12.60% / 8.40% | 30.87% / 26.40% | 21.70% / 17.30% | 21.72% / 17.37% | 22.10% / 17.67% |
| **[`wq2012/ls_lstm_small_ge2e`](https://huggingface.co/wq2012/ls_lstm_small_ge2e)** | `librispeech` | `lstm` (`small`) | `ge2e_softmax` / `cosine` | 1,710,914 | 12.77% / 7.90% | 28.03% / 24.90% | 23.35% / 19.70% | 21.38% / 17.50% | 21.25% / 17.73% |
| **[`wq2012/ls_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_small_ge2e)** | `librispeech` | `ecapa_tdnn` (`small`) | `ge2e_softmax` / `cosine` | 1,684,418 | 13.83% / 9.10% | 31.70% / 28.40% | **19.65% / 15.90%** | 21.73% / 17.80% | 22.48% / 17.83% |
| **[`wq2012/ls_conformer_small_cosface`](https://huggingface.co/wq2012/ls_conformer_small_cosface)** | `librispeech` | `conformer` (`small`) | `cosface` / `cosine` | 2,029,892 | 14.13% / 10.80% | 30.53% / 25.00% | 23.00% / 17.40% | 22.56% / 17.73% | 23.26% / 18.17% |
| **[`wq2012/ls_conformer_small_arcface`](https://huggingface.co/wq2012/ls_conformer_small_arcface)** | `librispeech` | `conformer` (`small`) | `arcface` / `cosine` | 2,029,892 | 14.47% / 9.60% | 28.73% / 25.50% | 23.85% / 18.30% | 22.35% / 17.80% | 22.80% / 18.23% |
| **[`wq2012/ls_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_base_ge2e)** | `librispeech` | `ecapa_tdnn` (`base`) | `ge2e_softmax` / `cosine` | 3,190,626 | 14.47% / 10.60% | 33.33% / 30.20% | 19.90% / 17.00% | 22.57% / 19.27% | 23.29% / 18.90% |
| **[`wq2012/ls_conformer_small_sphereface`](https://huggingface.co/wq2012/ls_conformer_small_sphereface)** | `librispeech` | `conformer` (`small`) | `sphereface` / `cosine` | 2,029,892 | 14.73% / 11.00% | 29.93% / 23.10% | 22.20% / 16.30% | 22.29% / 16.80% | 23.29% / 17.97% |
| **[`wq2012/ls_resnet_small_ge2e`](https://huggingface.co/wq2012/ls_resnet_small_ge2e)** | `librispeech` | `resnet` (`small`) | `ge2e_softmax` / `cosine` | 2,239,922 | 15.20% / 11.30% | 32.17% / 29.70% | 21.40% / 20.20% | 22.92% / 20.40% | 23.94% / 20.47% |
| **[`wq2012/ls_conformer_small_triplet`](https://huggingface.co/wq2012/ls_conformer_small_triplet)** | `librispeech` | `conformer` (`small`) | `triplet` / `cosine` | 2,029,892 | 16.07% / 13.50% | 28.93% / 25.20% | 22.00% / 20.20% | 22.33% / 19.63% | 23.41% / 19.80% |
| **[`wq2012/cn_conformer_small_dr_vectors`](https://huggingface.co/wq2012/cn_conformer_small_dr_vectors)** | `cnceleb2` | `conformer` (`small`) | `ext_softmax` / `dr_vector` | 2,088,007 | 23.00% / 16.20% | **21.17% / 16.70%** | 26.30% / 20.40% | 23.49% / 17.77% | 23.95% / 18.97% |
| **[`wq2012/cn_conformer_small_ge2e`](https://huggingface.co/wq2012/cn_conformer_small_ge2e)** | `cnceleb2` | `conformer` (`small`) | `ge2e_softmax` / `cosine` | 2,029,894 | 20.80% / 16.00% | 21.33% / 16.80% | 27.20% / 23.90% | 23.11% / 18.90% | 23.93% / 20.13% |
| **[`wq2012/cn_conformer_small_pfas`](https://huggingface.co/wq2012/cn_conformer_small_pfas)** | `cnceleb2` | `conformer` (`small`) | `ext_softmax` / `pfas` | 2,162,187 | 24.47% / 18.90% | 22.13% / 19.80% | 28.15% / 24.80% | 24.92% / 21.17% | 26.20% / 23.20% |
| **[`wq2012/cn_ecapa_tdnn_small_arcface`](https://huggingface.co/wq2012/cn_ecapa_tdnn_small_arcface)** | `cnceleb2` | `ecapa_tdnn` (`small`) | `arcface` / `cosine` | 1,684,416 | 30.53% / 23.40% | 28.07% / 23.80% | 31.40% / 31.20% | 30.00% / 26.13% | 30.23% / 26.47% |
| **[`wq2012/vox_conformer_small_ge2e`](https://huggingface.co/wq2012/vox_conformer_small_ge2e)** | `voxceleb1` | `conformer` (`small`) | `ge2e_softmax` / `cosine` | 2,029,894 | 16.33% / 11.70% | 27.87% / 25.80% | **20.80% / 16.40%** | 21.67% / 17.97% | 22.40% / 17.63% |
| **[`wq2012/vox_ecapa_tdnn_small_ext_softmax`](https://huggingface.co/wq2012/vox_ecapa_tdnn_small_ext_softmax)** | `voxceleb1` | `ecapa_tdnn` (`small`) | `ext_softmax` / `cosine` | 1,684,418 | 20.07% / 14.60% | 32.40% / 30.60% | 21.10% / 17.80% | 24.52% / 21.00% | 24.57% / 20.97% |

---

## 10. Controlled Ablations, VoxCeleb1-O 10-Speaker Subset, and Round-2 Controls

All tables in this section are generated from the JSON archives by `scripts/generate_report_tables.py` (nothing below is typed by hand). **Every run is a single seed**; see the seed-noise floor before interpreting sub-1% differences.

### VoxCeleb1-O 10-speaker subset (all 44 models)

Protocol: `veri_test2_10spk_subset_2500trials` — 2,500 trials (1,789 target / 711 impostor, 71.6% target), 937 utterances from 10 held-out speakers (`id10271`, `id10275`, `id10276`, `id10277`, `id10278`, `id10284`, `id10285`, `id10287`, `id10297`, `id10308`), VAD 26 dB (matched to training). **Not comparable to published VoxCeleb1-O numbers.**

| Model | Subset EER (%) [95% CI] | minDCF@0.01 | ROC-AUC | Pooled 1-utt EER (%) |
| :--- | :---: | :---: | :---: | :---: |
| `vox_ecapa_tdnn_small_ext_softmax` | 22.52 [20.50, 24.38] | 0.9899 | 0.8627 | 24.57 |
| `vox_conformer_small_ge2e` | 23.01 [20.53, 24.58] | 0.9665 | 0.8562 | 22.40 |
| `univ_conformer_tiny_ge2e` | 23.94 [21.75, 25.80] | 0.9748 | 0.8564 | 18.75 |
| `univ_conformer_small_cosface` | 24.07 [22.39, 25.97] | 0.9905 | 0.8454 | 25.60 |
| `ls_ecapa_tdnn_base_ge2e` | 24.10 [22.11, 25.92] | 0.9994 | 0.8492 | 23.29 |
| `ls_lstm_small_ge2e` | 24.17 [22.18, 26.06] | 0.9782 | 0.8400 | 21.25 |
| `univ_ecapa_tdnn_base_ge2e` | 24.20 [22.03, 26.12] | 0.9860 | 0.8415 | 22.81 |
| `ls_ecapa_tdnn_small_ge2e` | 24.60 [22.72, 26.49] | 0.9530 | 0.8436 | 22.48 |
| `univ_lstm_small_ge2e` | 24.73 [22.90, 26.92] | 0.9586 | 0.8459 | 21.31 |
| `univ_ecapa_tdnn_small_ge2e` | 24.73 [22.78, 27.14] | 0.9709 | 0.8323 | 23.16 |
| `univ_conformer_small_arcface` | 25.01 [22.93, 26.81] | 0.9463 | 0.8411 | 25.11 |
| `univ_conformer_base_ge2e` | 25.01 [23.26, 27.02] | 0.9609 | 0.8369 | 19.85 |
| `univ_conformer_small_sphereface` | 25.04 [23.06, 26.81] | 0.9838 | 0.8475 | 25.84 |
| `ls_mamba_small_ge2e` | 25.07 [23.06, 27.14] | 0.9704 | 0.8374 | 21.30 |
| `univ_conformer_small_pfas_ge2e` | 25.16 [23.01, 27.00] | 0.9810 | 0.8399 | 19.86 |
| `ls_conformer_small_ge2e` | 25.19 [23.16, 26.95] | 0.9536 | 0.8252 | 20.12 |
| `univ_conformer_small_ge2e_contrast` | 25.33 [23.29, 27.46] | 0.9642 | 0.8289 | 20.69 |
| `ls_transformer_small_ge2e` | 25.45 [22.88, 27.08] | 0.9452 | 0.8313 | 20.80 |
| `univ_mamba_small_ge2e` | 25.58 [23.45, 27.35] | 0.9877 | 0.8418 | 20.41 |
| `univ_ecapa_tdnn_tiny_ge2e` | 25.63 [23.41, 27.56] | 0.9419 | 0.8358 | 23.02 |
| `univ_conformer_small_ge2e` | 25.64 [23.75, 27.74] | 0.9855 | 0.8295 | 19.40 |
| `ls_conformer_small_sphereface` | 25.75 [23.97, 27.90] | 0.9704 | 0.8288 | 23.29 |
| `univ_transformer_small_ge2e` | 25.98 [24.08, 28.19] | 0.9732 | 0.8235 | 20.00 |
| `univ_conformer_small_triplet` | 26.05 [24.03, 28.09] | 0.9827 | 0.8225 | 24.07 |
| `univ_conformer_small_ext_softmax` | 26.06 [23.92, 28.00] | 0.9955 | 0.8361 | 20.50 |
| `ls_conformer_small_cosface` | 26.16 [24.04, 28.19] | 0.9799 | 0.8261 | 23.26 |
| `ls_conformer_small_pfas` | 26.29 [24.17, 28.25] | 0.9871 | 0.8267 | 20.88 |
| `univ_conformer_small_dr_vectors` | 26.29 [24.28, 28.38] | 0.9726 | 0.8390 | 20.10 |
| `univ_conformer_small_pfas` | 26.34 [23.47, 28.38] | 0.9866 | 0.8369 | 20.35 |
| `ls_conformer_small_triplet` | 26.37 [24.08, 28.29] | 0.9843 | 0.8246 | 23.41 |
| `ls_conformer_small_ext_softmax` | 26.43 [24.56, 28.43] | 0.9631 | 0.8133 | 21.05 |
| `ls_resnet_small_ge2e` | 26.44 [24.18, 28.60] | 0.9570 | 0.8195 | 23.94 |
| `ls_conformer_small_dr_vectors` | 26.47 [24.48, 28.34] | 0.9709 | 0.8153 | 20.69 |
| `ls_conformer_small_arcface` | 26.66 [24.58, 28.59] | 0.9503 | 0.8197 | 22.80 |
| `ls_ecapa_tdnn_tiny_ge2e` | 26.75 [24.52, 28.73] | 0.9793 | 0.8228 | 22.10 |
| `ls_conformer_base_ge2e` | 27.03 [24.94, 28.97] | 0.9441 | 0.8056 | 21.26 |
| `ls_conformer_small_pfas_ge2e` | 27.16 [25.06, 29.08] | 0.9681 | 0.8167 | 20.69 |
| `cn_conformer_small_dr_vectors` | 27.66 [25.61, 29.59] | 0.9894 | 0.8192 | 23.95 |
| `cn_conformer_small_ge2e` | 28.15 [25.94, 30.14] | 0.9916 | 0.8027 | 23.93 |
| `ls_conformer_small_ge2e_contrast` | 28.22 [26.02, 30.02] | 0.9100 | 0.7949 | 22.80 |
| `ls_conformer_tiny_ge2e` | 28.66 [26.59, 30.61] | 0.9838 | 0.7962 | 21.91 |
| `cn_conformer_small_pfas` | 29.81 [28.03, 31.88] | 0.9821 | 0.7876 | 26.20 |
| `univ_resnet_small_ge2e` | 31.60 [29.46, 33.75] | 0.9950 | 0.7607 | 27.89 |
| `cn_ecapa_tdnn_small_arcface` | 32.20 [30.20, 33.95] | 0.9704 | 0.7442 | 30.23 |

### Controlled ablations (`ablation_results.json`; § = copied from the main benchmark, not re-trained)

| Variant | LS 1/3-utt | CN 1/3-utt | VOX 1/3-utt | Macro 1/3-utt | Pooled 1-utt [95% CI] / 3-utt |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. `within_domain_prob = 0.0` (pure cross-domain) | 14.47 / 11.20 | 24.33 / 18.60 | 22.80 / 18.20 | 20.53 / 16.00 | 21.31 [20.49, 22.26] / 17.03 |
| 1. `within_domain_prob = 0.75` (default) § | 12.50 / 8.80 | 22.33 / 20.00 | 24.80 / 19.70 | 19.88 / 16.17 | 19.40 [18.61, 20.26] / 16.03 |
| 1. `within_domain_prob = 1.0` (pure within-domain) | 15.23 / 11.10 | 24.30 / 19.20 | 24.90 / 21.20 | 21.48 / 17.17 | 21.34 [20.52, 22.24] / 17.03 |
| 2. Pure extended-set softmax (paper Eq. 2), cosine | 22.20 / 17.80 | 25.00 / 20.50 | 26.35 / 23.20 | 24.52 / 20.50 | 24.52 [23.67, 25.55] / 20.77 |
| 2. Blended extended-set softmax (FlaxSpeaker variant), cosine § | 14.83 / 9.80 | 22.40 / 18.80 | 23.30 / 19.90 | 20.18 / 16.17 | 20.50 [19.64, 21.26] / 16.17 |
| 2. 3-term Dr-Vector variant (FlaxSpeaker extension) + pure extended-set | 17.03 / 12.20 | 24.80 / 21.20 | 22.90 / 19.40 | 21.58 / 17.60 | 21.91 [20.99, 22.82] / 18.17 |
| 2. 1-term down-scaled Dr-Vector variant + blended extended-set § | 15.20 / 9.90 | 22.33 / 18.20 | 22.75 / 17.50 | 20.09 / 15.20 | 20.10 [19.31, 20.96] / 15.40 |
| 3. Variable length U(90,180), full-sequence eval § | 12.50 / 8.80 | 22.33 / 20.00 | 24.80 / 19.70 | 19.88 / 16.17 | 19.40 [18.61, 20.26] / 16.03 |
| 3. Fixed length 180, full-sequence eval | 13.20 / 9.00 | 22.20 / 19.00 | 22.75 / 16.40 | 19.38 / 14.80 | 18.99 [18.14, 19.82] / 15.50 |
| 3. Variable length, 1.0 s (100-frame) eval § | 18.93 / 12.70 | 30.33 / 28.10 | 26.95 / 25.20 | 25.41 / 22.00 | 26.04 [25.05, 26.98] / 22.23 |
| 3. Fixed length, 1.0 s (100-frame) eval | 19.57 / 13.60 | 28.57 / 26.30 | 25.90 / 23.60 | 24.68 / 21.17 | 24.73 [23.86, 25.77] / 21.57 |
| 4. 400 steps (separate run) § | 12.50 / 8.80 | 22.33 / 20.00 | 24.80 / 19.70 | 19.88 / 16.17 | 19.40 [18.61, 20.26] / 16.03 |
| 4. 1,200-step run, snapshot at step 800 (LR mid-schedule; **not** an 800-step budget run) | 14.80 / 10.00 | 22.80 / 17.80 | 23.40 / 19.60 | 20.33 / 15.80 | 20.02 [19.19, 20.93] / 16.90 |
| 4. 800 steps with its own 800-step schedule (Round 2) | 12.87 / 9.20 | 21.23 / 16.80 | 23.20 / 18.80 | 19.10 / 14.93 | 19.06 [18.32, 19.99] / 14.87 |
| 4. 1,200 steps | 12.73 / 8.00 | 20.47 / 16.60 | 22.30 / 17.90 | 18.50 / 14.17 | 18.48 [17.61, 19.30] / 14.47 |
| 5. Mamba unidirectional § | 12.17 / 7.90 | 29.13 / 23.30 | 21.05 / 16.10 | 20.78 / 15.77 | 21.30 [20.40, 22.10] / 17.10 |
| 5. Mamba bidirectional | 13.53 / 10.80 | 29.83 / 25.30 | 21.75 / 17.30 | 21.71 / 17.80 | 22.07 [21.20, 22.96] / 17.73 |

Mamba 1-utt minDCF@0.01 (bidirectional vs unidirectional): librispeech 0.8547 vs 0.9080, cnceleb2 0.9320 vs 0.9760, voxceleb1 0.9570 vs 0.8780, universal_pooled 0.9160 vs 0.9523, macro 0.9146 vs 0.9207.

### Seed-noise floor (`r2_seed_and_control_results.json`)

| Model | Seeds | Pooled 1-utt EER per seed (%) | Pooled 1-utt mean ± sd | Macro 1-utt mean ± sd | Pooled 3-utt mean ± sd | Released seed-42 pooled / macro 1-utt (earlier code) | Seed 42 re-trained, current code: pooled / macro 1-utt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `univ_conformer_small_ge2e` | 0, 1, 2 | 20.14 / 20.30 / 21.48 | 20.64 ± 0.73 | 20.60 ± 0.74 | 16.50 ± 0.61 | 19.40 / 19.88 | 20.94 / 21.29 |
| `ls_conformer_small_ge2e` | 0, 1, 2 | 21.07 / 22.45 / 22.39 | 21.97 ± 0.78 | 21.85 ± 0.77 | 17.90 ± 0.44 | 20.12 / 20.06 | 21.40 / 21.60 |

Initial / final training loss, released (earlier code) vs. seed-42 re-run (current code): `univ_conformer_small_ge2e` 1.9180 / 0.7821 vs 2.0525 / 0.9060; `ls_conformer_small_ge2e` 2.0965 / 0.3651 vs 2.1204 / 0.5535.

### Matched-data control (`r2_seed_and_control_results.json`)

| Training set | Speakers / utterances | LS | CN | VOX | Macro 1-utt | Pooled 1-utt | Macro 3-utt | Pooled 3-utt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ls_conformer_small_ge2e` (released; earlier code) | 232 / 5,788 | 9.67 | 28.50 | 22.00 | 20.06 | 20.12 | 16.37 | 16.73 |
| `ls_conformer_small_ge2e` (seed 42 re-trained; current code) | 232 / 5,788 | 11.80 | 28.30 | 24.70 | 21.60 | 21.40 | 17.53 | 17.67 |
| universal corpus subsampled to the LibriSpeech count (seed 42; current code) | 381 / 5,788 | 15.00 | 25.17 | 22.00 | 20.72 | 21.09 | 16.03 | 16.83 |
| `univ_conformer_small_ge2e` (seed 42 re-trained; current code) | 381 / 9,447 | 14.93 | 25.10 | 23.85 | 21.29 | 20.94 | 17.10 | 17.57 |
| `univ_conformer_small_ge2e` (released; earlier code) | 381 / 9,447 | 12.50 | 22.33 | 24.80 | 19.88 | 19.40 | 16.17 | 16.03 |


### Interpretation (what the controls change)

- **Study 4 was mis-described in earlier revisions.** The former "800 steps" row was a snapshot of the single 1,200-step run taken at step 800 with the learning rate still mid-schedule; it is kept above only for transparency. The honest 800-step run (own schedule) makes the horizon trend monotone. Wall-clock minutes previously printed for this study did not correspond to any archived value and have been removed; the archived `train_time_sec` values are not comparable across rows because the 400-step row was trained in a different session.
- **Baseline rows marked § were copied from the main benchmark**, so each re-trained variant starts from a different RNG stream (`initial_loss` 2.0525 vs 1.918).
- **Mamba directionality**: the minDCF values quoted in earlier revisions (0.9947 / 0.9481 / 0.9760) did not exist in the archive; the archived values are listed above, and the archive also shows VoxCeleb1 moving the other way (bidirectional 0.9570 vs unidirectional 0.8780).
- **Seed floor and training reproducibility**: the standard deviation of pooled 1-utt EER across seeds {0, 1, 2} at the current code revision is about 0.7–0.8% absolute for both models. The released seed-42 checkpoints (19.40% / 20.12%) lie *below* both seed ranges, and **re-training seed 42 with the current code does not reproduce them**: it gives 20.94% / 21.40% pooled 1-utt EER, and the first-step loss already differs (2.0525 vs 1.9180 archived; 2.1204 vs 2.0965), i.e. the released weights were produced by earlier revisions of the training code (`model.safetensors` mtimes 2026-10-05 and 2026-10-15; the frontend, augmentation, pooling and backbone modules changed afterwards). Re-*evaluating* these two released checkpoints with the current code reproduces their archived per-domain EERs exactly (`released_checkpoint_reeval_current_code` in the JSON), so the archived numbers describe the released weights, but `scripts/train.py` at this revision will not regenerate them from the shipped YAMLs. The earlier claim that the released checkpoints were merely a lucky draw from the same distribution is withdrawn: they are draws from a different (earlier) pipeline. Differences between two single-seed models smaller than ~1.5% absolute should not be read as rankings; this covers all backbone comparisons except ResNet-18 and all objective/scorer comparisons in Sections 5–7.
- **Matched data**: trained on a 5,788-utterance subsample of the universal corpus (the LibriSpeech training-set size) at the current code revision, the universal model reaches 21.09% pooled 1-utt EER — close to the current-code seed-42 full-corpus universal run (20.94%) and better than the current-code seed-42 specialist (21.40%); against the seed means it sits between the universal (20.64%) and specialist (21.97%) distributions. At matched data quantity the universal-vs-specialist difference is within the seed floor; a substantial part of the pooled-EER advantage in Section 8 comes from the 1.6× larger corpus.
- **Protocol disagreement**: across the 44 models the Spearman rank correlation between pooled 1-utt EER and the VoxCeleb1-O-subset EER is 0.16 (p = 0.29); the in-house VoxCeleb1 EER correlates better (0.55). Model selection by pooled EER says little about this semi-external protocol.

---

## 11. Production Export: Hugging Face Hub & TensorFlow Lite (`FP32` and `INT8`)

Every trained model (`44/44`) is published on the Hugging Face Hub in the [**`wq2012/FlaxSpeaker` Collection**](https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) (`https://huggingface.co/wq2012/<experiment_name>`) and exported locally under [`pretrained_models/<experiment_name>/`](pretrained_models/) containing:
- `config.json` (`FlaxSpeakerConfig`; the frontend is **not** expressible as a Hugging Face `AutoFeatureExtractor`, so no `preprocessor_config.json` is shipped — use `flaxspeaker.AudioFrontend`)
- `model.safetensors` (`safetensors` serialized weights)
- `flax_model.msgpack` (native Flax msgpack checkpoint)
- `eval_results.json` & `README.md` (complete Hugging Face Model Card with multi-domain tables, 95% bootstrap CIs, disclaimer, and BibTeX)

In addition, 6 representative models (3 Universal + 3 Single-Domain across Conformer-ATP, ECAPA-TDNN, and LSTM) were exported to **TensorFlow Lite (`.tflite`)** in both **FP32** and **dynamic-range INT8 quantized** formats via `jax2tf`. **Caveats**: the exports are encoder-only with a **fixed, unmasked 180-frame (1.8 s) input window** (`export.sequence_length = 180`), all EERs in this report come from full-sequence masked JAX inference (≤ 500 frames), no TFLite EER has been measured, cosine fidelity was measured on 180-frame inputs only, and the latencies below are **single-shot** measurements on one x86 CPU (not averaged over runs; differences between rows are within timing noise):

| Exported TFLite Model | Precision | File Size (KB) | Compression Ratio | Single-shot latency (ms, 180-frame input, x86 CPU, not averaged) | Cosine Fidelity vs. Flax (180-frame input) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) | `FP32` | 8,052.27 KB | `1.00×` | 28.57 ms | `0.999999` |
| [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) | `INT8` | **2,267.91 KB** | **`3.55×`** | 33.55 ms | **`0.999948`** |
| [`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e) | `FP32` | 6,683.24 KB | `1.00×` | 30.63 ms | `0.999999` |
| [`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e) | `INT8` | **1,845.59 KB** | **`3.62×`** | **15.33 ms** | **`0.999966`** |
| [`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e) | `FP32` | 6,926.32 KB | `1.00×` | 52.06 ms | `0.999999` |
| [`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e) | `INT8` | **2,016.31 KB** | **`3.44×`** | 85.99 ms | **`0.999950`** |
| [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) | `FP32` | 8,052.27 KB | `1.00×` | 20.35 ms | `0.999999` |
| [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) | `INT8` | **2,267.91 KB** | **`3.55×`** | 29.48 ms | **`0.999961`** |
| [`wq2012/ls_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_small_ge2e) | `FP32` | 6,683.24 KB | `1.00×` | 27.94 ms | `0.999999` |
| [`wq2012/ls_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_small_ge2e) | `INT8` | **1,845.59 KB** | **`3.62×`** | 21.65 ms | **`0.999970`** |
| [`wq2012/ls_lstm_small_ge2e`](https://huggingface.co/wq2012/ls_lstm_small_ge2e) | `FP32` | 6,926.32 KB | `1.00×` | 61.68 ms | `0.999999` |
| [`wq2012/ls_lstm_small_ge2e`](https://huggingface.co/wq2012/ls_lstm_small_ge2e) | `INT8` | **2,016.31 KB** | **`3.44×`** | 37.21 ms | **`0.999946`** |

---

## 12. Citations

```bibtex
@inproceedings{wan2018generalized,
  title={Generalized End-to-End Loss for Speaker Verification},
  author={Wan, Li and Wang, Quan and Papir, Alan and Moreno, Ignacio Lopez},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4879--4883},
  year={2018}
}

@inproceedings{pelecanos2021dr,
  title={Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition},
  author={Pelecanos, Jason and Wang, Quan and Moreno, Ignacio Lopez},
  booktitle={Interspeech},
  pages={4603--4607},
  year={2021}
}

@inproceedings{wang2022attentive,
  title={Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech},
  author={Wang, Quan and Yu, Yang and Pelecanos, Jason and Huang, Yiling and Moreno, Ignacio Lopez},
  booktitle={The Speaker and Language Recognition Workshop (Odyssey)},
  year={2022}
}

@inproceedings{pelecanos2022parameter,
  title={Parameter-Free Attentive Scoring for Speaker Verification},
  author={Pelecanos, Jason and Wang, Quan and Huang, Yiling and Moreno, Ignacio Lopez},
  booktitle={The Speaker and Language Recognition Workshop (Odyssey)},
  year={2022}
}
```
