# FlaxSpeaker v0.4.0: Universal Multi-Domain Speaker Recognition Research Report

## Disclaimer & Open-Source Reproduction Notice

> **This is not an officially supported Google product.**
>
> `FlaxSpeaker` is a from-scratch re-implementation in JAX/Flax, written using only the published papers (no proprietary code or data), of speaker recognition algorithms published in:
> 1. **GE2E**: Li Wan, Quan Wang, Alan Papir, and Ignacio Lopez Moreno, *"Generalized End-to-End Loss for Speaker Verification"*, IEEE ICASSP 2018 ([arXiv:1710.10467](https://arxiv.org/abs/1710.10467)).
> 2. **Dr-Vectors**: Jason Pelecanos, Quan Wang, and Ignacio Lopez Moreno, *"Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition"*, Interspeech 2021 ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989)).
> 3. **ATP** (a *language identification* paper whose pooling head is repurposed here): Quan Wang, Yang Yu, Jason Pelecanos, Yiling Huang, and Ignacio Lopez Moreno, *"Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech"*, Odyssey 2022 ([arXiv:2202.12163](https://arxiv.org/abs/2202.12163)).
> 4. **PFAS**: Jason Pelecanos, Quan Wang, Yiling Huang, and Ignacio Lopez Moreno, *"Parameter-Free Attentive Scoring for Speaker Verification"*, Odyssey 2022 ([arXiv:2203.05642v3](https://arxiv.org/abs/2203.05642v3)).
>
> The experiments reported in those original papers were conducted using internal proprietary infrastructure and large-scale datasets (`~8K–18K` speakers, `~500k` steps), not this open-source library. All benchmark numbers in this report are measured directly from our open-source JAX/Flax implementation on public datasets (`LibriSpeech`, `CN-Celeb2`, `VoxCeleb1`).
>
> **Scope & Corpus-Scale Notice (`v0.4.0`).** Every released checkpoint in `v0.4.0` is a **converged CPU research checkpoint** trained to a calibrated `S_safe = X + 2` epoch step budget (`4,600` steps / `15.58` epochs on Universal; `2,400–2,600` steps / `13.27–14.37` epochs on LibriSpeech; `1,600–1,800` steps / `17.60–19.80` epochs on CN-Celeb2; `1,000` steps / `42.67` epochs on VoxCeleb1; `32` utterances per step) with uniform variable-length masked crops `T ~ U(90, 180)` (`variable_length_training: true` across all 44 models) and linear angular-margin warm-up (`margin_warmup_steps`). Because the training splits intentionally fit on a single workstation (`≤ 381` speakers, `≤ 9,447` utterances), pooled 1-utt EERs of **`14.60–22.35%`** (`19.15–29.96%` on the 10-speaker VoxCeleb1-O subset) are bounded by speaker diversity rather than optimization under-training and are **not comparable to published VoxCeleb2 systems** (`7,000+` speakers, `~1%` EER). Several released zoo models use **FlaxSpeaker variants** of the cited algorithms (hybrid contrast surrogate, blended extended-set softmax, down-scaled Dr-Vector scorer, PFAS-SN head); all four paper-faithful `pure_paper_mode=True` presets are also trained for `4,600` steps and benchmarked in Section 10 (`ablation_results.json → studies.paper_faithful_presets_smoke_test`).

---

## 1. Executive Summary & Verification Provenance

This report documents a 44-model converged empirical investigation of **Universal Multi-Domain Speaker Recognition** versus **Single-Domain Speaker Recognition** in `FlaxSpeaker v0.4.0` across 6 neural backbones (`Conformer`, `Transformer`, `Mamba`, `LSTM`, `ECAPA-TDNN`, `ResNet-18`), 7 released training objectives (`ArcFace`, `CosFace`, `SphereFace`, `GE2E-Softmax`, `GE2E-Contrast`, `Extended-Set Softmax`, `Triplet`) with 3 scoring heads (`Cosine`, `Dr-Vectors`, `PFAS`), and 3 parameter scales (`tiny`, `small`, `base`, `0.63M–4.55M` parameters), backed by a 5,000-step Phase-1 convergence pilot, 7 controlled ablation groups, 3-seed stability runs, matched-data controls at `2,400` and `4,600` steps, and a step-0 loss fingerprint of every released checkpoint (§10).

- **Hugging Face Pretrained Model Collection (44 Models)**: [`wq2012/FlaxSpeaker`](https://huggingface.co/collections/wq2012/flaxspeaker) (individual repositories at `https://huggingface.co/wq2012/<model_id>`)
- **Machine-Readable Metrics Artifact**: [`pretrained_models/benchmark_results.json`](pretrained_models/benchmark_results.json) (and per-model [`pretrained_models/<model_id>/eval_results.json`](pretrained_models/))
- **Phase-1 Convergence Pilot & Schedule Calibration**: [`pretrained_models/phase1_convergence_pilot.json`](pretrained_models/phase1_convergence_pilot.json) (`scripts/run_phase1_convergence_pilot.py`)
- **Controlled Ablations & Paper-Faithful Presets**: [`pretrained_models/ablation_results.json`](pretrained_models/ablation_results.json) (`scripts/run_r1_ablations.py`)
- **Multi-Seed Stability, Own-Schedule Horizons & Matched-Data Controls**: [`pretrained_models/r2_seed_and_control_results.json`](pretrained_models/r2_seed_and_control_results.json) (`scripts/run_r2_controls.py`)
- **VoxCeleb1-O 10-Speaker Subset Evaluation**: [`pretrained_models/vox1_o_10spk_subset_results.json`](pretrained_models/vox1_o_10spk_subset_results.json) (`scripts/eval_vox1_o_subset.py`)
- **Step-0 Loss Fingerprint**: [`pretrained_models/step0_fingerprints.json`](pretrained_models/step0_fingerprints.json) (`scripts/fingerprint_released_checkpoints.py`)
- **Reproducible Table & Figure Generators**: `scripts/generate_report_tables.py`, `scripts/generate_paper_tables.py`, `scripts/generate_figures.py`

![FlaxSpeaker Architecture Diagram](resources/architecture_diagram.png)

![FlaxSpeaker Benchmark Summary](resources/benchmark_charts.png)

### Key Research Highlights (`v0.4.0`)

1. **Schedule Calibration (`S_safe = X + 2` Epochs) & Linear Margin Warm-Up Invert the 400-Step Objective Ranking**:
   - In the 5,000-step Phase-1 convergence pilot (`phase1_convergence_pilot.json`), 381-class angular-margin classifiers (`ArcFace`, `CosFace`, `SphereFace`) were severely under-converged at 400 steps (`~25%` Pooled 1-utt EER) and suffered early gradient shock when an additive margin was applied to random embeddings. Introducing linear margin warm-up (`margin_warmup_steps = 500` on Universal, `300` on LibriSpeech, `250` on CN-Celeb2) and training to the empirically calibrated **`S_safe = X + 2` epoch** plateau (`4,600` steps on Universal) enables angular-margin classifiers to cross over GE2E around step `1,200–1,600` and lead the entire 44-model zoo:
     - **[`univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface)** achieves **`14.60%` Pooled 1-Utt EER** (`[13.82, 15.48]`), **`14.63%` Macro 1-Utt EER** (`5.77%` LibriSpeech, `18.53%` CN-Celeb2, `19.60%` VoxCeleb1), **`10.87%` Pooled 3-Utt EER**, **`0.6800` Macro 1-Utt minDCF@0.01**, and **`19.68%`** on the VoxCeleb1-O 10-speaker subset.
     - **[`univ_conformer_small_sphereface`](https://huggingface.co/wq2012/univ_conformer_small_sphereface)** ties at **`14.60%` Pooled 1-Utt EER** (`10.53%` Pooled 3-Utt EER), while **[`univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface)** achieves **`14.71%` Pooled 1-Utt EER**, the best **`10.40%` Pooled 3-Utt EER** (`9.63%` Macro 3-Utt EER, `2.50%` LibriSpeech 3-Utt EER, `0.4847` Macro 3-Utt minDCF@0.01), and the best **`19.15%`** EER on the VoxCeleb1-O 10-speaker subset.
2. **Universal Multi-Domain Training Beats Matched LibriSpeech Specialists Across All `19 / 19` Pairs — Including at Matched Utterances & Compute**:
   - Single-domain specialists suffer severe out-of-domain degradation (`ls_conformer_small_ge2e`: `7.77%` LibriSpeech vs. `29.57%` CN-Celeb2 and `25.00%` VoxCeleb1, `20.79%` Pooled 1-utt EER; `ls_conformer_small_arcface`: `5.23%` LibriSpeech vs. `28.80%` CN-Celeb2, `19.48%` Pooled).
   - At converged `S_safe` budgets, **`univ_*` improves over the matched `ls_*` specialist in `19 / 19` pairs** on Pooled 1-utt EER (`-2.39` to `-5.65 pp`), Macro 1-utt EER (`-2.24` to `-5.86 pp`), and Pooled 3-utt EER (`-2.17` to `-6.87 pp`).
   - Holding utterance count (`5,788` utterances) and compute (`2,400` steps) strictly equal to `ls_conformer_small_ge2e`, subsampling the 381-speaker Universal mix lowers Pooled 1-utt EER from **`20.79%` to `17.04%`** (**`-3.75 pp`**, non-overlapping 95% CIs `[16.31, 17.91]` vs. `[19.95, 21.68]`), proving that multi-domain speaker diversity directly drives cross-domain generalization.
3. **2D ResNet-18 Recovery, Tight 6-Backbone Convergence & Resolved Capacity Scaling**:
   - Under 4,600-step convergence, 2D `SE-ResNet-18` (`univ_resnet_small_ge2e`) drops from its 400-step cold-start (`27.89%`) to **`15.60%` Pooled 1-Utt EER** (`12.20%` Pooled 3-Utt EER, `0.5407` Macro 3-Utt minDCF@0.01), placing all six Small GE2E backbones within a `1.06 pp` band (`LSTM` `15.50%`, `ResNet-18` `15.60%`, `Conformer` `15.81%`, `ECAPA-TDNN` `16.34%`, `Transformer` `16.40%`, causal `Mamba` `16.56%`; bidirectional `Mamba` reaches **`15.59% / 11.93%`** in Study 5b).
   - On LibriSpeech, capacity scaling resolves monotonically: `Conformer-Base` (`7.40%`) < `Small` (`7.77%`) < `Tiny` (`7.87%`).
4. **Tight Multi-Seed Convergence & Strong Cross-Protocol Rank Agreement**:
   - Across seeds `{0, 1, 2}` at `S_safe`, pooled 1-utt EER standard deviation shrinks from `~0.75 pp` (at 400 steps) to **`0.08 pp`** (`15.13 ± 0.08%` on `univ_conformer_small_ge2e`), **`0.14 pp`** (`16.58 ± 0.14%` on `univ_ecapa_tdnn_small_ge2e`), and **`0.41 pp`** (`20.16 ± 0.41%` on `ls_conformer_small_ge2e`).
   - Across all 44 converged models, pooled 1-utt EER correlates strongly with the held-out VoxCeleb1-O 10-speaker subset (**Spearman $\rho = 0.73$, $p = 2.4 \times 10^{-8}$**; in-house VoxCeleb1 1-utt EER $\rho = 0.83$, $p = 2.4 \times 10^{-12}$).

---

## 2. Original Published Paper Reference Results vs. Open-Source Reproduction Context

The table below places the four reference publications alongside our `FlaxSpeaker v0.4.0` converged results (both the released `<5M`-param variants and the 4,600-step `pure_paper_mode=True` runs). Because the original papers train on `8K–18K` speakers for `~500k` steps whereas our splits have `≤ 381` speakers, absolute EERs are not comparable.

| Paper & Algorithm | Original Paper Training Corpus | Original Paper Reported Benchmark & Metric | FlaxSpeaker v0.4.0 Converged Open-Source Result (`≤ 381` Speakers, Public Corpora) |
| :--- | :--- | :--- | :--- |
| **GE2E-Softmax & GE2E-Contrast** ([Wan et al., ICASSP 2018](https://arxiv.org/abs/1710.10467)) | Internal Voice Search (`~18K` spk, `~36M` utts) | TI-SV (~6.3 enroll utts/spk, 1,000 test spk): **3.55% EER** (`3×768` LSTM + `256` proj) | [`univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e) (LayerNorm+residual): **`15.50% / 12.50%`** Pooled 1/3-Utt EER; [`univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast): **`14.95% / 10.73%`**; `icassp2018_ge2e_config(pure_paper_mode=True)`: `26.59% / 24.13%` |
| **Dr-Vectors + Extended-Set Softmax** ([Pelecanos et al., Interspeech 2021](https://arxiv.org/abs/2104.01989)) | Internal Multi-Condition (`8K+` spk) | `~4%–11%` relative EER reduction over Cosine + GE2E | `interspeech2021_dr_vectors_config(pure_paper_mode=True)`: **`15.47% / 11.53%`** Pooled 1/3-Utt EER (`15.79% / 11.33%` Macro); [`univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors): **`15.88% / 11.67%`** (`16.00% / 11.13%` Macro) |
| **Conformer with Attentive Temporal Pooling (ATP)** ([Wang et al., Odyssey 2022](https://arxiv.org/abs/2202.12163)) | Internal Multilingual Speech (`65` languages) | Sigmoid Attentive Temporal Pooling (`mean ⊕ std`) stabilizes long-form Conformer pooling | [`univ_conformer_base_ge2e`](https://huggingface.co/wq2012/univ_conformer_base_ge2e): **`15.14% / 11.63%`** Pooled 1/3-Utt EER; [`ls_conformer_base_ge2e`](https://huggingface.co/wq2012/ls_conformer_base_ge2e): **`7.40% / 4.20%`** LibriSpeech; `odyssey2022_conformer_atp_config(pure_paper_mode=True)`: `16.82% / 13.87%` |
| **Parameter-Free Attentive Scoring (PFAS)** ([Pelecanos et al., Odyssey 2022](https://arxiv.org/abs/2203.05642v3)) | Internal Multi-Enrollment SV | `(K, V)` cross-attention improves multi-enrollment EER & minDCF without scoring parameters | [`univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e): **`15.39% / 11.80%`** Pooled 1/3-Utt EER (`4.70%` LS 3-Utt); [`univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas): **`15.55% / 11.53%`**; `odyssey2022_pfas_config(pure_paper_mode=True)`: `18.34% / 14.43%` |

---

## 3. Multi-Domain Dataset, Step-Budget Calibration & Evaluation Protocol

All datasets are stored under `./data/speaker_datasets` (or `$FLAXSPEAKER_DATA_ROOT`) with strictly speaker-disjoint training and evaluation splits (`scripts/prepare_datasets.py`), plus evaluation on the **VoxCeleb1-O 10-speaker subset** (`veri_test2.txt` filtered to pairs where both speakers belong to the 10 held-out VoxCeleb1 evaluation speakers: `2,500` trials, `1,789` target / `711` impostor, `71.6%` target, `937` utterances). One VAD threshold (`26 dB`) is used everywhere:

| Dataset Split | Domain & Language | Train Speakers | Train Utterances | Calibrated `S_safe` Budget (`Epochs`) | Eval Speakers (Unseen) | Eval Utterances | 1-Utt / 3-Utt Trials |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`librispeech`** | English Read Audiobooks (`16 kHz`) | 232 | 5,788 | `2,400` (`13.27` ep) / `2,600` (`14.37` ep, Base & ResNet) | 40 | 1,000 | 3,000 / 1,000 |
| **`cnceleb2`** | Chinese Multi-Genre Speech (`16 kHz`) | 119 | 2,909 | `1,600` (`17.60` ep) / `1,800` (`19.80` ep, ArcFace) | 30 | 739 | 3,000 / 1,000 |
| **`voxceleb1`** | English Celebrity Interviews (`16 kHz`) | 30 (`seed=42`) | 750 | `1,000` (`42.67` ep) | 10 (`seed=42`, unseen) | 250 | 2,000 / 1,000 |
| **`vox1_o_10spk_subset`** | VoxCeleb1-O **10-speaker subset** of `veri_test2.txt` | — (Eval-Only) | — | — | 10 (`id10271..id10308`) | 937 | 2,500 / — |
| **`universal`** | **Combined 3-Domain Mix (`LS + CN + VOX`)** | **381** | **9,447** | **`4,600` (`15.58` ep)** | **80** | **1,989** | **8,000 / 3,000** |

### Multi-Domain Engineering & Convergence Calibration Details

1. **Domain-Prefixed Speaker Namespaces**: Raw speaker IDs in `CN-Celeb2` (`id10029..id11975`) and `VoxCeleb1` (`id10270..id10309`) collide on 5 speaker IDs (`id10272`, `id10293`, `id10295`, `id10297`, `id10309`). Every universal speaker ID is prefixed with `<domain>:<spk_id>` to prevent accidental cross-corpus speaker merging.
2. **Domain-Balanced Stratified Batching (`flaxspeaker/feature_extraction.py`)**: Uses temperature-balanced domain probabilities `p_d ∝ |S_d|^0.35` and alternates between `75%` within-domain hard-negative batches (`N = 8` speakers from the same domain) and `25%` cross-domain calibration batches.
3. **Phase-1 Convergence Pilot & `S_safe = X + 2` Epoch Rule (`scripts/run_phase1_convergence_pilot.py`)**: A 5,000-step sweep over 7 representative models located the validation EER knee `X` for each model family (`X ≈ 4,000` on Universal at `295.22` steps/epoch; `X ≈ 2,000–2,200` on LibriSpeech at `180.88` steps/epoch; `X ≈ 1,400–1,600` on CN-Celeb2 at `90.91` steps/epoch; `X ≈ 900` on VoxCeleb1 at `23.44` steps/epoch) and added a `+2`-epoch cosine-tail safety margin (`+600`, `+400`, `+200`, `+100` steps respectively) with `10%` linear LR warm-up (`warmup_steps = 0.1 * S_safe`).

---

## 4. Core Architectural & Algorithmic Components

### 4.1 Conformer + Multi-Head Attentive Temporal Pooling (`flaxspeaker/backbones.py`, `flaxspeaker/pooling.py`)
- Macaron Conformer blocks (`HalfStepFFN -> MHSA -> ConformerConvModule(k=15, GroupNorm(2)) -> HalfStepFFN -> LayerNorm`) with a frame-rate-preserving parallel linear + conv stem and Multi-Layer Feature Aggregation (MFA), paired with 4-head subspace Attentive Temporal Pooling (`attention_dim = 128`) and a nonlinear bottleneck FFN. `odyssey2022_conformer_atp_config(pure_paper_mode=True)` recovers the exact Odyssey 2022 ordering (`conv_before_mhsa`, sinusoidal PE, `subsample_factor = 2`, single-head linear-sigmoid ATP).

### 4.2 Parameter-Free Attentive Scoring (`PFAS`, `flaxspeaker/pooling.py`, `flaxspeaker/scoring.py`)
- Supports both the paper's `atp_ffn` head (`32` tied keys × `(16 + 48)` = `2,048-D`) and the compact `PFAS-SN` head (`shared_nonlinear`, `dot`, or `maxpool` frame attention with an optional global ATP value anchor `v_i = v_global + 0.25 * v_local,i`, `4` keys × `(32 + 128)` = `640-D`), with two-stage concatenated value L2 normalization and multi-enrollment key stacking (`N = M × E`).

### 4.3 Linear Angular-Margin Warm-Up (`margin_warmup_steps`, `flaxspeaker/losses.py`, `flaxspeaker/training.py`)
- Applying full additive angular margin `m` (`0.20–0.25`) at step 0 on 381 random speaker prototypes destabilizes early feature formation. Linearly ramping `m_t = m * min(1, t / margin_warmup_steps)` (`500` steps on Universal, `300` on LibriSpeech, `250` on CN-Celeb2) lets speaker clusters separate under scaled cosine softmax before enforcing the geodesic margin, reducing converged Universal Pooled 1-utt EER to **`14.60%`** (`ArcFace`) and **`14.71%`** (`CosFace`).

### 4.4 Frame-Level Energy VAD, Mutually Exclusive Augmentation & Uniform Variable-Length Crops
- **VAD + AGC**: Peak-normalized waveform energy VAD (`26 dB` dynamic SNR margin, hangover `8` frames) applied identically in training and evaluation.
- **Mutually Exclusive Augmentation**: `35%` synthetic log-Mel noise (`SNR ~ U(12, 20) dB`), `32.5%` SpecAugment, `32.5%` clean.
- **Uniform Variable-Length Training (`T ~ U(90, 180)`)**: Every `v0.4.0` checkpoint is trained with per-utterance variable sequence lengths `T ~ U(90, 180)` (`0.9–1.8 s`) and exact binary masks `(B, T)`, and evaluated up to `max_eval_seq_len = 500` frames (`5.0 s`). Step-0 loss fingerprinting (`step0_fingerprints.json`) confirms `variable` (`42/44`) or `both` (`2/44`, where fixed and variable step-0 losses round to the identical 4-decimal value) across the entire zoo (`0` `fixed`, `0` `neither`).

---

## 5. Study 5: Universal Multi-Domain Neural Backbones (`4,600` Steps on 3-Dataset Mix)

All models in Study 5 are `small` variants (`1.03M–2.24M` parameters) trained for `4,600` steps (`15.58` epochs, `batch_size = 32`, `N = 8 × M = 4`, `T ~ U(90, 180)`) on the **Universal Multi-Domain training set (`381` speakers, `9,447` utterances)** using `GE2E-Softmax`:

### 5.1 Universal Backbones — 1-Utterance Verification Results

| Hugging Face Repository | Backbone (`Pooling`) | Params | LibriSpeech EER (%) [95% CI] | CN-Celeb2 EER (%) [95% CI] | VoxCeleb1 EER (%) [95% CI] | Universal Macro-Avg EER (%) | Universal Pooled EER (%) [95% CI] | Pooled ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e)** | `lstm` (`asp`) | 1,710,914 | 9.33% [8.30, 10.53] | **18.33%** [16.97, 19.88] | **19.05%** [17.05, 20.80] | **15.57%** | **15.50%** [14.84, 16.44] | 0.9234 |
| **[`wq2012/univ_resnet_small_ge2e`](https://huggingface.co/wq2012/univ_resnet_small_ge2e)** | `resnet` (`asp`) | 2,239,922 | 9.23% [8.33, 10.37] | 19.10% [17.65, 20.52] | 19.85% [18.20, 21.73] | 16.06% | 15.60% [14.85, 16.40] | **0.9272** |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | `conformer` (`atp`) | 2,029,894 | **9.00%** [7.87, 10.00] | 19.07% [17.73, 20.42] | 21.40% [19.90, 23.30] | 16.49% | 15.81% [15.10, 16.70] | 0.9255 |
| **[`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e)** | `ecapa_tdnn` (`asp`) | 1,684,418 | 10.37% [9.40, 11.57] | 18.73% [17.20, 20.00] | 19.75% [17.75, 21.45] | 16.28% | 16.34% [15.56, 17.15] | 0.9199 |
| **[`wq2012/univ_transformer_small_ge2e`](https://huggingface.co/wq2012/univ_transformer_small_ge2e)** | `transformer` (`atp`) | 1,239,622 | 9.47% [8.53, 10.72] | 19.43% [17.98, 21.00] | 20.35% [18.60, 22.20] | 16.42% | 16.40% [15.57, 17.21] | 0.9219 |
| **[`wq2012/univ_mamba_small_ge2e`](https://huggingface.co/wq2012/univ_mamba_small_ge2e)** | `mamba` (`asp`) | **1,028,674** | 9.67% [8.62, 10.77] | 19.87% [18.45, 21.35] | 20.50% [18.87, 22.53] | 16.68% | 16.56% [15.79, 17.43] | 0.9210 |

*(Note: Enabling bidirectional scan `bidirectional: true` on `univ_mamba_small_ge2e` in Study 5b lowers Pooled 1-Utt EER from `16.56%` to **`15.59%`** `[14.82, 16.32]` and Pooled 3-Utt EER from `12.67%` to **`11.93%`**.)*

### 5.2 Universal Backbones — 3-Utterance Multi-Enrollment Verification Results

| Hugging Face Repository | Backbone (`Pooling`) | LibriSpeech 3-Utt EER (%) | CN-Celeb2 3-Utt EER (%) | VoxCeleb1 3-Utt EER (%) | Universal Macro-Avg 3-Utt EER (%) | Universal Pooled 3-Utt EER (%) [95% CI] | Macro 3-Utt minDCF@0.01 | Macro 3-Utt ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | `conformer` (`atp`) | **6.00%** | **13.60%** | 15.00% | **11.53%** | **11.93%** [10.67, 13.08] | 0.6340 | **0.9533** |
| **[`wq2012/univ_resnet_small_ge2e`](https://huggingface.co/wq2012/univ_resnet_small_ge2e)** | `resnet` (`asp`) | 6.60% | 14.30% | 15.70% | 12.20% | 12.20% [11.07, 13.53] | **0.5407** | **0.9533** |
| **[`wq2012/univ_transformer_small_ge2e`](https://huggingface.co/wq2012/univ_transformer_small_ge2e)** | `transformer` (`atp`) | 6.80% | 14.00% | 15.00% | 11.93% | 12.40% [10.98, 13.47] | 0.6687 | 0.9532 |
| **[`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e)** | `lstm` (`asp`) | 6.70% | 15.90% | 15.10% | 12.57% | 12.50% [11.27, 13.77] | 0.7107 | 0.9502 |
| **[`wq2012/univ_mamba_small_ge2e`](https://huggingface.co/wq2012/univ_mamba_small_ge2e)** | `mamba` (`asp`) | 7.40% | 15.70% | 14.30% | 12.47% | 12.67% [11.40, 13.77] | 0.6740 | 0.9460 |
| **[`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e)** | `ecapa_tdnn` (`asp`) | 7.60% | 15.10% | **13.70%** | 12.13% | 12.87% [11.63, 14.15] | 0.7093 | 0.9471 |

---

## 6. Study 6: Universal Multi-Domain Losses & Scoring (`Conformer-Small`, `4,600` Steps)

Holding the backbone fixed (`Conformer-Small`, `3` layers, `d = 192`, `4` heads, `~2.03M–2.16M` parameters), we compare all 10 objective and scoring configurations on the **Universal Multi-Domain** corpus at `4,600` steps:

### 6.1 Universal Losses & Scoring — 1-Utterance Verification

| Hugging Face Repository | Loss Function | Scoring | Pooling | LibriSpeech EER (%) | CN-Celeb2 EER (%) | VoxCeleb1 EER (%) | Macro-Avg 1-Utt EER (%) | Pooled 1-Utt EER (%) [95% CI] | Macro 1-Utt minDCF@0.01 | Pooled 1-Utt ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface)** | `arcface` | `cosine` | `atp` | 5.77% | 18.53% | 19.60% | **14.63%** | **14.60%** [13.82, 15.48] | **0.6800** | **0.9363** |
| **[`wq2012/univ_conformer_small_sphereface`](https://huggingface.co/wq2012/univ_conformer_small_sphereface)** | `sphereface` | `cosine` | `atp` | 7.07% | 18.40% | 20.10% | 15.19% | **14.60%** [13.89, 15.49] | 0.7100 | 0.9354 |
| **[`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface)** | `cosface` | `cosine` | `atp` | **5.43%** | 19.60% | **19.30%** | 14.78% | 14.71% [13.97, 15.46] | 0.7572 | 0.9320 |
| **[`wq2012/univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast)** | `ge2e_contrast` | `cosine` | `atp` | 8.27% | **18.27%** | 19.60% | 15.38% | 14.95% [14.24, 15.79] | 0.7510 | 0.9281 |
| **[`wq2012/univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e)** | `ge2e_softmax` | **`pfas`** | `pfas` | 8.47% | 19.33% | 19.55% | 15.78% | 15.39% [14.61, 16.23] | 0.7817 | 0.9317 |
| **[`wq2012/univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas)** | `extended_set_softmax` | **`pfas`** | `pfas` | 8.67% | 19.23% | 20.30% | 16.07% | 15.55% [14.82, 16.38] | 0.7778 | 0.9311 |
| **[`wq2012/univ_conformer_small_ext_softmax`](https://huggingface.co/wq2012/univ_conformer_small_ext_softmax)** | `extended_set_softmax` | `cosine` | `atp` | 8.57% | 19.00% | 19.90% | 15.82% | 15.62% [14.91, 16.39] | 0.8579 | 0.9280 |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | `ge2e_softmax` | `cosine` | `atp` | 9.00% | 19.07% | 21.40% | 16.49% | 15.81% [15.10, 16.70] | 0.8254 | 0.9255 |
| **[`wq2012/univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors)** | `extended_set_softmax` | **`dr_vector`** | `atp` | 8.67% | 19.03% | 20.30% | 16.00% | 15.88% [15.12, 16.68] | 0.8410 | 0.9285 |
| **[`wq2012/univ_conformer_small_triplet`](https://huggingface.co/wq2012/univ_conformer_small_triplet)** | `triplet` | `cosine` | `atp` | 10.73% | 23.27% | 23.95% | 19.32% | 19.25% [18.37, 20.06] | 0.9391 | 0.8922 |

### 6.2 Universal Losses & Scoring — 3-Utterance Multi-Enrollment Verification

| Hugging Face Repository | Loss Function | Scoring | LibriSpeech 3-Utt EER (%) | CN-Celeb2 3-Utt EER (%) | VoxCeleb1 3-Utt EER (%) | Macro-Avg 3-Utt EER (%) | Pooled 3-Utt EER (%) [95% CI] | Macro 3-Utt minDCF@0.01 | Pooled 3-Utt ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface)** | `cosface` | `cosine` | **2.50%** | **12.40%** | 14.00% | **9.63%** | **10.40%** [9.27, 11.67] | **0.4847** | **0.9668** |
| **[`wq2012/univ_conformer_small_sphereface`](https://huggingface.co/wq2012/univ_conformer_small_sphereface)** | `sphereface` | `cosine` | 3.00% | 13.70% | 14.70% | 10.47% | 10.53% [9.27, 11.60] | 0.6173 | 0.9638 |
| **[`wq2012/univ_conformer_small_ge2e_contrast`](https://huggingface.co/wq2012/univ_conformer_small_ge2e_contrast)** | `ge2e_contrast` | `cosine` | 5.20% | 12.80% | **12.60%** | 10.20% | 10.73% [9.47, 11.85] | 0.5720 | 0.9631 |
| **[`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface)** | `arcface` | `cosine` | 2.80% | 13.70% | 15.80% | 10.77% | 10.87% [9.93, 12.12] | 0.6100 | 0.9625 |
| **[`wq2012/univ_conformer_small_pfas`](https://huggingface.co/wq2012/univ_conformer_small_pfas)** | `extended_set_softmax` | **`pfas`** | 5.60% | 14.60% | 15.20% | 11.80% | 11.53% [10.08, 12.72] | 0.6620 | 0.9570 |
| **[`wq2012/univ_conformer_small_ext_softmax`](https://huggingface.co/wq2012/univ_conformer_small_ext_softmax)** | `extended_set_softmax` | `cosine` | 5.20% | 12.90% | 15.80% | 11.30% | 11.63% [10.40, 12.68] | 0.6760 | 0.9583 |
| **[`wq2012/univ_conformer_small_dr_vectors`](https://huggingface.co/wq2012/univ_conformer_small_dr_vectors)** | `extended_set_softmax` | **`dr_vector`** | 5.20% | 13.00% | 15.20% | 11.13% | 11.67% [10.50, 12.80] | 0.6980 | 0.9571 |
| **[`wq2012/univ_conformer_small_pfas_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_pfas_ge2e)** | `ge2e_softmax` | **`pfas`** | 4.70% | 14.80% | 15.00% | 11.50% | 11.80% [10.55, 13.00] | 0.6667 | 0.9570 |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | `ge2e_softmax` | `cosine` | 6.00% | 13.60% | 15.00% | 11.53% | 11.93% [10.67, 13.08] | 0.6340 | 0.9560 |
| **[`wq2012/univ_conformer_small_triplet`](https://huggingface.co/wq2012/univ_conformer_small_triplet)** | `triplet` | `cosine` | 6.30% | 18.70% | 20.80% | 15.27% | 15.67% [14.35, 17.04] | 0.7653 | 0.9230 |

---

## 7. Study 7: Model Capacity Scaling (`Tiny` vs. `Small` vs. `Base`)

On `LibriSpeech` (`232` speakers) and the high-resource sub-domains (`LS` & `CN` inside Universal), `Conformer-Base` (`4.55M` params) achieves the best EER (`7.40%` on LibriSpeech specialist; `8.17%` LS and `17.97%` CN on Universal), resolving the 400-step capacity inversion. On the 30-speaker `VoxCeleb1` slice inside Universal (`750` utterances total), smaller parameterizations (`univ_conformer_tiny_ge2e` `19.25%`, `univ_ecapa_tdnn_tiny_ge2e` `18.25%`) avoid overfitting the 30 training speakers:

| Hugging Face Repository | Train (`Steps`) | Backbone & Size | Params | LibriSpeech 1-Utt EER (%) | CN-Celeb2 1-Utt EER (%) | VoxCeleb1 1-Utt EER (%) | Macro-Avg 1-Utt EER (%) | Pooled 1-Utt EER (%) [95% CI] | Macro-Avg 3-Utt EER (%) | Pooled 3-Utt EER (%) [95% CI] |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`wq2012/univ_conformer_tiny_ge2e`](https://huggingface.co/wq2012/univ_conformer_tiny_ge2e)** | `universal` (`4,600`) | `conformer-tiny` | **693,126** | 8.47% | 18.30% | 19.25% | 15.34% | **15.05%** [14.38, 15.87] | 11.33% | **11.33%** [10.18, 12.47] |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | `universal` (`4,600`) | `conformer-small` | 2,029,894 | 9.00% | 19.07% | 21.40% | 16.49% | 15.81% [15.10, 16.70] | 11.53% | 11.93% [10.67, 13.08] |
| **[`wq2012/univ_conformer_base_ge2e`](https://huggingface.co/wq2012/univ_conformer_base_ge2e)** | `universal` (`4,600`) | `conformer-base` | 4,551,558 | **8.17%** | **17.97%** | 19.60% | **15.24%** | 15.14% [14.43, 15.93] | **11.10%** | 11.63% [10.35, 12.73] |
| **[`wq2012/univ_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_tiny_ge2e)** | `universal` (`4,600`) | `ecapa_tdnn-tiny` | **626,722** | 9.57% | 18.87% | **18.25%** | 15.56% | 15.70% [14.97, 16.54] | 11.77% | 12.63% [11.27, 13.67] |
| **[`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e)** | `universal` (`4,600`) | `ecapa_tdnn-small` | 1,684,418 | 10.37% | 18.73% | 19.75% | 16.28% | 16.34% [15.56, 17.15] | 12.13% | 12.87% [11.63, 14.15] |
| **[`wq2012/univ_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_base_ge2e)** | `universal` (`4,600`) | `ecapa_tdnn-base` | 3,190,626 | 10.30% | 18.40% | 19.90% | 16.20% | 16.40% [15.66, 17.24] | 12.07% | 12.60% [11.23, 13.78] |
| **[`wq2012/ls_conformer_tiny_ge2e`](https://huggingface.co/wq2012/ls_conformer_tiny_ge2e)** | `librispeech` (`2,400`) | `conformer-tiny` | 693,126 | 7.87% | 29.43% | 26.30% | 21.20% | 20.70% [19.88, 21.64] | 17.37% | 18.20% [16.70, 19.48] |
| **[`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e)** | `librispeech` (`2,400`) | `conformer-small` | 2,029,894 | 7.77% | 29.57% | 25.00% | 20.78% | 20.79% [19.95, 21.68] | 17.70% | 18.23% [16.90, 19.58] |
| **[`wq2012/ls_conformer_base_ge2e`](https://huggingface.co/wq2012/ls_conformer_base_ge2e)** | `librispeech` (`2,600`) | `conformer-base` | 4,551,558 | **7.40%** | 28.87% | 25.75% | 20.67% | 20.10% [19.24, 20.97] | 16.67% | 17.07% [15.33, 18.27] |
| **[`wq2012/ls_ecapa_tdnn_tiny_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_tiny_ge2e)** | `librispeech` (`2,400`) | `ecapa_tdnn-tiny` | 626,722 | 9.67% | 29.43% | 20.65% | 19.92% | 19.99% [19.18, 20.98] | 15.73% | 16.03% [14.77, 17.47] |
| **[`wq2012/ls_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_small_ge2e)** | `librispeech` (`2,400`) | `ecapa_tdnn-small` | 1,684,418 | 10.37% | 29.80% | 19.75% | 19.97% | 19.81% [19.02, 20.82] | 15.10% | 15.93% [14.43, 17.05] |
| **[`wq2012/ls_ecapa_tdnn_base_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_base_ge2e)** | `librispeech` (`2,600`) | `ecapa_tdnn-base` | 3,190,626 | **9.20%** | 29.30% | 20.00% | 19.50% | 20.07% [19.23, 21.05] | 15.40% | 16.13% [14.83, 17.33] |

---

## 8. Universal Multi-Domain vs. Single-Domain Direct Comparison

Comparing matched `Conformer-Small` models across single-domain specialists (`librispeech`, `cnceleb2`, `voxceleb1`) and the Universal 3-Dataset Mix (`universal`) shows universal training winning across all multi-domain aggregates while staying within `0.10–2.30 pp` of each domain specialist on its home domain:

| Hugging Face Repository | Training Dataset (`Spk / Utts`, `Steps`) | LibriSpeech 1-Utt / 3-Utt EER (%) | CN-Celeb2 1-Utt / 3-Utt EER (%) | VoxCeleb1 1-Utt / 3-Utt EER (%) | Universal Macro-Avg 1-Utt / 3-Utt EER (%) | Universal Pooled 1-Utt / 3-Utt EER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) | LibriSpeech Only (`232 / 5,788`, `2,400`) | 7.77% / 4.60% | 29.57% / 25.90% | 25.00% / 22.60% | 20.78% / 17.70% | 20.79% / 18.23% |
| [`wq2012/cn_conformer_small_ge2e`](https://huggingface.co/wq2012/cn_conformer_small_ge2e) | CN-Celeb2 Only (`119 / 2,909`, `1,600`) | 18.30% / 12.60% | 18.97% / 15.40% | 24.30% / 21.00% | 20.52% / 16.33% | 21.01% / 16.53% |
| [`wq2012/vox_conformer_small_ge2e`](https://huggingface.co/wq2012/vox_conformer_small_ge2e) | VoxCeleb1 Only (`30 / 750`, `1,000`) | 16.50% / 11.60% | 26.57% / 25.80% | **19.10% / 14.20%** | 20.72% / 17.20% | 20.84% / 16.47% |
| **[`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e)** | **Universal Mix (`381 / 9,447`, `4,600`)** | 9.00% / 6.00% | 19.07% / 13.60% | 21.40% / 15.00% | 16.49% / 11.53% | 15.81% / 11.93% |
| [`wq2012/ls_conformer_small_arcface`](https://huggingface.co/wq2012/ls_conformer_small_arcface) | LibriSpeech Only (`232 / 5,788`, `2,400`) | **5.23% / 2.20%** | 28.80% / 23.30% | 24.00% / 19.20% | 19.34% / 14.90% | 19.48% / 15.93% |
| **[`wq2012/univ_conformer_small_arcface`](https://huggingface.co/wq2012/univ_conformer_small_arcface)** | **Universal Mix (`381 / 9,447`, `4,600`)** | 5.77% / 2.80% | **18.53% / 13.70%** | 19.60% / 15.80% | **14.63% / 10.77%** | **14.60% / 10.87%** |
| **[`wq2012/univ_conformer_small_cosface`](https://huggingface.co/wq2012/univ_conformer_small_cosface)** | **Universal Mix (`381 / 9,447`, `4,600`)** | 5.43% / 2.50% | 19.60% / **12.40%** | 19.30% / **14.00%** | 14.78% / **9.63%** | 14.71% / **10.40%** |

---

## 9. Studies 1–4: Complete Single-Domain Baselines Evaluated Across All 3 Domains

Every single-domain baseline model (`25` models) was trained to its calibrated `S_safe` step budget and evaluated across all 3 evaluation domains and the Universal Pooled set:

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

## 10. Controlled Ablations, VoxCeleb1-O 10-Speaker Subset, Multi-Seed Stability, Matched-Data Controls & Step-0 Provenance

All tables in this section are generated directly from the JSON archives by `scripts/generate_report_tables.py`.

### VoxCeleb1-O 10-speaker subset (all 44 models)

Protocol: `veri_test2_10spk_subset_2500trials` — 2,500 trials (1,789 target / 711 impostor, 71.6% target), 937 utterances from 10 held-out speakers (`id10271`, `id10275`, `id10276`, `id10277`, `id10278`, `id10284`, `id10285`, `id10287`, `id10297`, `id10308`), VAD 26 dB (matched to training). **Not comparable to published VoxCeleb1-O numbers.**

| Model | Subset EER (%) [95% CI] | minDCF@0.01 | ROC-AUC | Pooled 1-utt EER (%) |
| :--- | :---: | :---: | :---: | :---: |
| `univ_conformer_small_cosface` | 19.15 [17.76, 20.80] | 0.8368 | 0.8887 | 14.71 |
| `univ_conformer_small_arcface` | 19.68 [18.32, 21.91] | 0.7937 | 0.8857 | 14.60 |
| `vox_conformer_small_ge2e` | 19.84 [18.01, 21.28] | 0.9380 | 0.8814 | 20.84 |
| `univ_ecapa_tdnn_tiny_ge2e` | 20.24 [18.32, 22.21] | 0.9715 | 0.8817 | 15.70 |
| `univ_ecapa_tdnn_base_ge2e` | 20.36 [18.32, 21.95] | 0.9173 | 0.8834 | 16.40 |
| `univ_conformer_small_dr_vectors` | 20.37 [18.23, 22.12] | 0.8921 | 0.8821 | 15.88 |
| `univ_conformer_small_sphereface` | 20.43 [18.72, 22.18] | 0.8256 | 0.8871 | 14.60 |
| `univ_conformer_small_pfas` | 20.52 [18.81, 22.85] | 0.9648 | 0.8785 | 15.55 |
| `univ_ecapa_tdnn_small_ge2e` | 20.68 [18.85, 22.77] | 0.9547 | 0.8761 | 16.34 |
| `univ_conformer_small_ge2e_contrast` | 20.68 [18.85, 22.50] | 0.9279 | 0.8754 | 14.95 |
| `univ_conformer_small_ext_softmax` | 20.83 [19.04, 22.96] | 0.9195 | 0.8787 | 15.62 |
| `univ_mamba_small_ge2e` | 20.96 [18.81, 22.85] | 0.9704 | 0.8787 | 16.56 |
| `vox_ecapa_tdnn_small_ext_softmax` | 21.09 [18.90, 23.16] | 0.9564 | 0.8699 | 22.25 |
| `univ_conformer_small_pfas_ge2e` | 21.30 [18.80, 23.45] | 0.9301 | 0.8811 | 15.39 |
| `univ_conformer_tiny_ge2e` | 21.42 [19.50, 23.36] | 0.9312 | 0.8760 | 15.05 |
| `univ_transformer_small_ge2e` | 21.49 [19.74, 23.39] | 0.9335 | 0.8743 | 16.40 |
| `univ_conformer_small_ge2e` | 21.58 [19.66, 23.29] | 0.9396 | 0.8739 | 15.81 |
| `univ_conformer_base_ge2e` | 21.70 [19.54, 23.75] | 0.9536 | 0.8789 | 15.14 |
| `univ_lstm_small_ge2e` | 21.93 [19.87, 23.74] | 0.9715 | 0.8812 | 15.50 |
| `cn_ecapa_tdnn_small_arcface` | 22.21 [20.51, 23.99] | 0.9866 | 0.8636 | 18.98 |
| `univ_resnet_small_ge2e` | 22.43 [20.69, 24.44] | 0.9039 | 0.8710 | 15.60 |
| `ls_lstm_small_ge2e` | 22.89 [20.94, 24.86] | 0.8988 | 0.8553 | 19.71 |
| `ls_conformer_small_ge2e_contrast` | 23.43 [21.60, 25.14] | 0.9357 | 0.8339 | 19.56 |
| `ls_ecapa_tdnn_small_ge2e` | 23.92 [22.00, 25.81] | 0.8781 | 0.8503 | 19.81 |
| `ls_ecapa_tdnn_base_ge2e` | 23.92 [21.95, 25.66] | 0.9469 | 0.8455 | 20.07 |
| `ls_conformer_small_sphereface` | 24.42 [22.57, 26.75] | 0.8910 | 0.8303 | 18.54 |
| `ls_resnet_small_ge2e` | 24.86 [22.73, 26.60] | 0.9849 | 0.8367 | 19.52 |
| `ls_ecapa_tdnn_tiny_ge2e` | 25.46 [23.43, 27.21] | 0.9335 | 0.8382 | 19.99 |
| `ls_conformer_small_arcface` | 25.63 [23.55, 27.59] | 0.9156 | 0.8247 | 19.48 |
| `ls_mamba_small_ge2e` | 26.20 [23.99, 28.17] | 0.9570 | 0.8126 | 20.00 |
| `univ_conformer_small_triplet` | 26.29 [24.26, 28.43] | 0.9648 | 0.8202 | 19.25 |
| `cn_conformer_small_dr_vectors` | 26.31 [24.07, 28.07] | 0.9966 | 0.8311 | 21.05 |
| `ls_transformer_small_ge2e` | 26.69 [24.51, 28.82] | 0.8949 | 0.8183 | 20.55 |
| `ls_conformer_small_pfas_ge2e` | 26.82 [24.87, 29.02] | 0.8994 | 0.8146 | 20.39 |
| `cn_conformer_small_ge2e` | 27.00 [24.36, 28.78] | 0.9625 | 0.8199 | 21.01 |
| `cn_conformer_small_pfas` | 27.00 [24.84, 28.66] | 0.9676 | 0.8213 | 22.35 |
| `ls_conformer_small_pfas` | 27.24 [24.77, 28.95] | 0.9402 | 0.8105 | 20.30 |
| `ls_conformer_small_cosface` | 27.56 [25.81, 29.31] | 0.8496 | 0.8072 | 19.76 |
| `ls_conformer_small_dr_vectors` | 28.05 [26.03, 30.13] | 0.9542 | 0.7974 | 21.07 |
| `ls_conformer_small_ge2e` | 28.30 [26.09, 30.27] | 0.9463 | 0.7914 | 20.79 |
| `ls_conformer_small_ext_softmax` | 28.30 [26.38, 30.38] | 0.9307 | 0.7950 | 20.62 |
| `ls_conformer_base_ge2e` | 29.24 [27.48, 31.23] | 0.9497 | 0.7832 | 20.10 |
| `ls_conformer_small_triplet` | 29.71 [27.69, 31.49] | 0.9776 | 0.7902 | 21.64 |
| `ls_conformer_tiny_ge2e` | 29.96 [27.65, 32.15] | 0.9816 | 0.7831 | 20.70 |

### Controlled ablations (`ablation_results.json`, `r2_seed_and_control_results.json`; § = matched converged v0.4.0 checkpoint in `benchmark_results.json`, 4,600 steps on Universal / 2,400 steps on LibriSpeech, seed 42, variable-length crops `T ~ U(90, 180)`)

| Variant | LS 1/3-utt | CN 1/3-utt | VOX 1/3-utt | Macro 1/3-utt | Pooled 1-utt [95% CI] / 3-utt |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Stratified batching: `within_domain_prob = 0.00` (pure cross-domain, 4,600 steps) | 9.03 / 6.00 | 18.63 / 14.00 | 22.85 / 17.10 | 16.84 / 12.37 | 16.51 [15.67, 17.38] / 12.90 |
| 1. Stratified batching: `within_domain_prob = 0.75` (default hybrid, 4,600 steps) § | 9.00 / 6.00 | 19.07 / 13.60 | 21.40 / 15.00 | 16.49 / 11.53 | 15.81 [15.10, 16.70] / 11.93 |
| 1. Stratified batching: `within_domain_prob = 1.00` (pure within-domain, 4,600 steps) | 8.63 / 5.40 | 17.87 / 13.20 | 19.80 / 14.30 | 15.43 / 10.97 | 15.00 [14.32, 15.82] / 11.13 |
| 2. Pure extended-set softmax (paper Eq. 2), cosine (4,600 steps) | 8.87 / 5.00 | 20.13 / 14.20 | 21.80 / 15.20 | 16.93 / 11.47 | 16.44 [15.59, 17.28] / 11.70 |
| 2. Blended extended-set softmax (FlaxSpeaker BXS variant), cosine (4,600 steps) § | 8.57 / 5.20 | 19.00 / 12.90 | 19.90 / 15.80 | 15.82 / 11.30 | 15.62 [14.91, 16.39] / 11.63 |
| 2. 3-term Dr-Vector variant (`[c, c^2, sqrt(max(c,0))]`) + pure extended-set (4,600 steps) | 8.63 / 4.60 | 20.53 / 14.70 | 21.10 / 16.30 | 16.76 / 11.87 | 16.55 [15.71, 17.34] / 12.17 |
| 2. 1-term down-scaled Dr-Vector variant + blended extended-set (4,600 steps) § | 8.67 / 5.20 | 19.03 / 13.00 | 20.30 / 15.20 | 16.00 / 11.13 | 15.88 [15.12, 16.68] / 11.67 |
| 2. 3-term Dr-Vector symmetric skip + blended extended-set (4,600 steps) | 8.17 / 4.40 | 19.87 / 14.70 | 22.35 / 16.80 | 16.79 / 11.97 | 16.65 [15.84, 17.42] / 12.27 |
| 2. Hybrid `ge2e_contrast` surrogate (4,600 steps) § | 8.27 / 5.20 | 18.27 / 12.80 | 19.60 / 12.60 | 15.38 / 10.20 | 14.95 [14.24, 15.79] / 10.73 |
| 2b. PFAS-SN `shared_nonlinear` + global ATP anchor (4,600 steps) § | 8.67 / 5.60 | 19.23 / 14.60 | 20.30 / 15.20 | 16.07 / 11.80 | 15.55 [14.82, 16.38] / 11.53 |
| 2b. PFAS-SN `shared_nonlinear` without global ATP anchor (4,600 steps) | 9.00 / 5.40 | 19.97 / 16.60 | 20.70 / 18.00 | 16.56 / 13.33 | 16.71 [15.99, 17.54] / 14.13 |
| 2b. PFAS-SN `dot` attention + global ATP anchor (4,600 steps) | 8.13 / 5.00 | 19.07 / 14.60 | 21.00 / 15.40 | 16.07 / 11.67 | 15.72 [15.00, 16.56] / 12.23 |
| 2b. PFAS-SN `maxpool` + global ATP anchor (4,600 steps) | 10.03 / 7.80 | 18.57 / 14.30 | 20.30 / 15.90 | 16.30 / 12.67 | 16.46 [15.67, 17.22] / 12.83 |
| 3. Variable crops `T ~ U(90, 180)` (4,600 steps), full-sequence (`<=500f`) eval § | 9.00 / 6.00 | 19.07 / 13.60 | 21.40 / 15.00 | 16.49 / 11.53 | 15.81 [15.10, 16.70] / 11.93 |
| 3. Fixed crops `T = 180` (4,600 steps), full-sequence (`<=500f`) eval | 8.53 / 4.90 | 18.17 / 12.30 | 18.50 / 12.90 | 15.07 / 10.03 | 15.19 [14.49, 16.11] / 11.20 |
| 3. Variable crops `T ~ U(90, 180)` (4,600 steps), short 1.0 s (`100f`) eval § | 14.10 / 10.10 | 23.97 / 18.40 | 25.00 / 21.50 | 21.02 / 16.67 | 20.69 [19.67, 21.50] / 18.13 |
| 3. Fixed crops `T = 180` (4,600 steps), short 1.0 s (`100f`) eval | 15.20 / 10.80 | 24.67 / 19.60 | 26.60 / 20.50 | 22.16 / 16.97 | 21.75 [20.93, 22.76] / 17.43 |
| 4. Horizon: 400 steps (own 400-step cosine schedule) | 14.07 / 11.60 | 24.53 / 21.80 | 23.55 / 18.30 | 20.72 / 17.23 | 20.79 [20.04, 21.68] / 17.33 |
| 4. Horizon: 800 steps (own 800-step cosine schedule) | 12.67 / 8.60 | 21.80 / 16.80 | 24.30 / 17.80 | 19.59 / 14.40 | 19.21 [18.27, 20.03] / 14.90 |
| 4. Horizon: 1,200 steps (own 1,200-step cosine schedule) | 11.80 / 7.80 | 20.87 / 17.20 | 22.25 / 18.40 | 18.31 / 14.47 | 18.39 [17.60, 19.28] / 14.67 |
| 4. Horizon: 2,400 steps (own 2,400-step cosine schedule) | 9.47 / 6.60 | 20.20 / 14.60 | 21.65 / 18.20 | 17.11 / 13.13 | 16.88 [16.09, 17.76] / 13.20 |
| 4. Horizon: 4,600 steps (`S_safe`, own 4,600-step cosine schedule) § | 9.00 / 6.00 | 19.07 / 13.60 | 21.40 / 15.00 | 16.49 / 11.53 | 15.81 [15.10, 16.70] / 11.93 |
| 5a. Mamba unidirectional on LibriSpeech (2,400 steps) § | 8.87 / 5.10 | 28.10 / 22.70 | 22.85 / 18.30 | 19.94 / 15.37 | 20.00 [19.18, 20.91] / 16.70 |
| 5a. Mamba bidirectional on LibriSpeech (2,400 steps) | 8.60 / 5.60 | 29.10 / 24.80 | 22.30 / 19.10 | 20.00 / 16.50 | 20.07 [19.27, 21.04] / 16.83 |
| 5b. Mamba unidirectional on Universal (4,600 steps) § | 9.67 / 7.40 | 19.87 / 15.70 | 20.50 / 14.30 | 16.68 / 12.47 | 16.56 [15.79, 17.43] / 12.67 |
| 5b. Mamba bidirectional on Universal (4,600 steps) | 8.87 / 6.50 | 18.67 / 13.70 | 20.75 / 14.70 | 16.09 / 11.63 | 15.59 [14.82, 16.32] / 11.93 |
| 6a. ECAPA-TDNN `LayerNorm` before activation (default, 4,600 steps) § | 10.37 / 7.60 | 18.73 / 15.10 | 19.75 / 13.70 | 16.28 / 12.13 | 16.34 [15.56, 17.15] / 12.87 |
| 6a. ECAPA-TDNN `LayerNorm` after activation (BN-style order, 4,600 steps) | 9.93 / 6.40 | 18.60 / 14.40 | 19.40 / 13.70 | 15.98 / 11.50 | 15.84 [15.10, 16.66] / 12.63 |
| 6b. Frontend VAD 26 dB + peak AGC (default, 4,600 steps) § | 9.00 / 6.00 | 19.07 / 13.60 | 21.40 / 15.00 | 16.49 / 11.53 | 15.81 [15.10, 16.70] / 11.93 |
| 6b. Frontend VAD 26 dB without AGC (4,600 steps) | 8.00 / 5.80 | 18.27 / 12.10 | 21.00 / 13.40 | 15.76 / 10.43 | 15.30 [14.53, 16.10] / 11.27 |
| 6b. Frontend VAD 20 dB without AGC (4,600 steps) | 8.20 / 4.90 | 18.43 / 12.50 | 21.90 / 15.50 | 16.18 / 10.97 | 15.95 [15.04, 16.74] / 11.33 |
| 7. `icassp2018_ge2e_config(pure_paper_mode=True)` (3×768 LSTM + 256 proj, 4,600 steps) | 23.93 / 21.70 | 29.77 / 27.90 | 24.90 / 22.80 | 26.20 / 24.13 | 26.59 [25.69, 27.59] / 24.13 |
| 7. `interspeech2021_dr_vectors_config(pure_paper_mode=True)` (200+56 split, 3×256 LReLU, 4,600 steps) | 8.13 / 4.90 | 19.10 / 14.20 | 20.15 / 14.90 | 15.79 / 11.33 | 15.47 [14.69, 16.23] / 11.53 |
| 7. `odyssey2022_conformer_atp_config(pure_paper_mode=True)` (conv-before-MHSA, sin PE, ×2 subsample, 4,600 steps) | 10.13 / 7.40 | 20.57 / 15.30 | 20.70 / 16.80 | 17.13 / 13.17 | 16.82 [16.14, 17.77] / 13.87 |
| 7. `odyssey2022_pfas_config(pure_paper_mode=True)` (`atp_ffn`, 32 tied keys × (16+48), 4,600 steps) | 10.63 / 6.30 | 21.20 / 17.20 | 22.95 / 18.20 | 18.26 / 13.90 | 18.34 [17.55, 19.23] / 14.43 |

### Multi-seed stability at converged `S_safe` (`r2_seed_and_control_results.json`)

| Model (`Steps`) | Seeds | Pooled 1-utt EER per seed (%) | Pooled 1-utt mean ± sd | Macro 1-utt mean ± sd | LS 1-utt mean ± sd | Pooled 3-utt mean ± sd | Released seed-42 pooled / macro 1-utt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `univ_conformer_small_ge2e` (`4,600`) | 0, 1, 2 | 15.11 / 15.22 / 15.06 | 15.13 ± 0.08 | 15.62 ± 0.11 | 8.23 ± 0.44 | 11.30 ± 0.55 | 15.81 / 16.49 |
| `univ_ecapa_tdnn_small_ge2e` (`4,600`) | 0, 1, 2 | 16.48 / 16.52 / 16.74 | 16.58 ± 0.14 | 16.39 ± 0.04 | 9.83 ± 0.06 | 12.63 ± 0.26 | 16.34 / 16.28 |
| `ls_conformer_small_ge2e` (`2,400`) | 0, 1, 2 | 19.71 / 20.52 / 20.25 | 20.16 ± 0.41 | 20.31 ± 0.28 | 7.48 ± 0.60 | 16.96 ± 0.47 | 20.79 / 20.78 |

### Matched-data control (`r2_seed_and_control_results.json`)

| Training configuration | Steps | Speakers / utterances | LS 1-utt | CN 1-utt | VOX 1-utt | Macro 1-utt | Pooled 1-utt [95% CI] | Macro 3-utt | Pooled 3-utt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ls_conformer_small_ge2e` (LibriSpeech-only specialist) § | `2,400` | 232 / 5,788 | 7.77 | 29.57 | 25.00 | 20.78 | 20.79 [19.95, 21.68] | 17.70 | 18.23 |
| Universal subsampled to LibriSpeech utterance count (matched utterances & compute) | `2,400` | 381 / 5,788 | 10.53 | 20.20 | 21.10 | 17.28 | 17.04 [16.31, 17.91] | 12.33 | 12.87 |
| Universal subsampled to LibriSpeech utterance count (`S_safe` 4,600-step schedule) | `4,600` | 381 / 5,788 | 8.17 | 18.47 | 19.10 | 15.24 | 15.28 [14.57, 16.11] | 10.83 | 11.07 |
| `univ_conformer_small_ge2e` (full Universal 3-domain mix) § | `4,600` | 381 / 9,447 | 9.00 | 19.07 | 21.40 | 16.49 | 15.81 [15.10, 16.70] | 11.53 | 11.93 |

### Step-0 loss fingerprint of all 44 released checkpoints (`step0_fingerprints.json`, code `9b091a5+dirty`; tolerance 5e-05)

Histogram: `both` = 2, `variable` = 42.

| Checkpoint | Loss | Released `initial_loss` | `HEAD`, fixed T = 180 | `HEAD`, T ~ U(90,180) | Match |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `univ_conformer_small_ge2e_contrast` | `ge2e_contrast` | 1.0001 | 1.0001 | 1.0001 | `both` |
| `univ_ecapa_tdnn_tiny_ge2e` | `ge2e_softmax` | 2.0697 | 2.0697 | 2.0697 | `both` |
| `cn_conformer_small_dr_vectors` | `extended_set_softmax` | 2.1125 | 1.9661 | 2.1125 | `variable` |
| `cn_conformer_small_ge2e` | `ge2e_softmax` | 2.0926 | 1.9294 | 2.0926 | `variable` |
| `cn_conformer_small_pfas` | `extended_set_softmax` | 2.0599 | 1.9827 | 2.0599 | `variable` |
| `cn_ecapa_tdnn_small_arcface` | `arcface` | 6.7785 | 7.5219 | 6.7785 | `variable` |
| `ls_conformer_base_ge2e` | `ge2e_softmax` | 2.1163 | 2.0807 | 2.1163 | `variable` |
| `ls_conformer_small_arcface` | `arcface` | 7.8938 | 7.2812 | 7.8938 | `variable` |
| `ls_conformer_small_cosface` | `cosface` | 7.8938 | 7.2812 | 7.8938 | `variable` |
| `ls_conformer_small_dr_vectors` | `extended_set_softmax` | 2.0849 | 2.0240 | 2.0849 | `variable` |
| `ls_conformer_small_ext_softmax` | `extended_set_softmax` | 2.0882 | 2.0232 | 2.0882 | `variable` |
| `ls_conformer_small_ge2e` | `ge2e_softmax` | 2.1204 | 2.0965 | 2.1204 | `variable` |
| `ls_conformer_small_ge2e_contrast` | `ge2e_contrast` | 1.0005 | 1.0000 | 1.0005 | `variable` |
| `ls_conformer_small_pfas` | `extended_set_softmax` | 2.0415 | 2.0493 | 2.0415 | `variable` |
| `ls_conformer_small_pfas_ge2e` | `ge2e_softmax` | 2.0093 | 2.0452 | 2.0093 | `variable` |
| `ls_conformer_small_sphereface` | `sphereface` | 7.9095 | 7.2967 | 7.9095 | `variable` |
| `ls_conformer_small_triplet` | `triplet` | 0.1044 | 0.0935 | 0.1044 | `variable` |
| `ls_conformer_tiny_ge2e` | `ge2e_softmax` | 2.1332 | 2.0900 | 2.1332 | `variable` |
| `ls_ecapa_tdnn_base_ge2e` | `ge2e_softmax` | 2.0906 | 2.0638 | 2.0906 | `variable` |
| `ls_ecapa_tdnn_small_ge2e` | `ge2e_softmax` | 2.0840 | 2.0730 | 2.0840 | `variable` |
| `ls_ecapa_tdnn_tiny_ge2e` | `ge2e_softmax` | 2.0713 | 2.0736 | 2.0713 | `variable` |
| `ls_lstm_small_ge2e` | `ge2e_softmax` | 2.0686 | 2.0370 | 2.0686 | `variable` |
| `ls_mamba_small_ge2e` | `ge2e_softmax` | 2.0537 | 2.0599 | 2.0537 | `variable` |
| `ls_resnet_small_ge2e` | `ge2e_softmax` | 2.0695 | 2.0740 | 2.0695 | `variable` |
| `ls_transformer_small_ge2e` | `ge2e_softmax` | 2.1206 | 2.0829 | 2.1206 | `variable` |
| `univ_conformer_base_ge2e` | `ge2e_softmax` | 2.0638 | 1.9922 | 2.0638 | `variable` |
| `univ_conformer_small_arcface` | `arcface` | 8.9101 | 8.4811 | 8.9101 | `variable` |
| `univ_conformer_small_cosface` | `cosface` | 8.9101 | 8.4811 | 8.9101 | `variable` |
| `univ_conformer_small_dr_vectors` | `extended_set_softmax` | 2.1347 | 2.0420 | 2.1347 | `variable` |
| `univ_conformer_small_ext_softmax` | `extended_set_softmax` | 2.1339 | 2.0379 | 2.1339 | `variable` |
| `univ_conformer_small_ge2e` | `ge2e_softmax` | 2.0525 | 1.9180 | 2.0525 | `variable` |
| `univ_conformer_small_pfas` | `extended_set_softmax` | 2.1081 | 1.9822 | 2.1081 | `variable` |
| `univ_conformer_small_pfas_ge2e` | `ge2e_softmax` | 2.0883 | 1.9704 | 2.0883 | `variable` |
| `univ_conformer_small_sphereface` | `sphereface` | 8.9258 | 8.4967 | 8.9258 | `variable` |
| `univ_conformer_small_triplet` | `triplet` | 0.0875 | 0.1020 | 0.0875 | `variable` |
| `univ_conformer_tiny_ge2e` | `ge2e_softmax` | 2.0288 | 1.9935 | 2.0288 | `variable` |
| `univ_ecapa_tdnn_base_ge2e` | `ge2e_softmax` | 2.0573 | 2.0546 | 2.0573 | `variable` |
| `univ_ecapa_tdnn_small_ge2e` | `ge2e_softmax` | 2.0650 | 2.0660 | 2.0650 | `variable` |
| `univ_lstm_small_ge2e` | `ge2e_softmax` | 2.0644 | 2.0203 | 2.0644 | `variable` |
| `univ_mamba_small_ge2e` | `ge2e_softmax` | 2.0539 | 1.9903 | 2.0539 | `variable` |
| `univ_resnet_small_ge2e` | `ge2e_softmax` | 2.0738 | 2.0825 | 2.0738 | `variable` |
| `univ_transformer_small_ge2e` | `ge2e_softmax` | 2.0179 | 2.0160 | 2.0179 | `variable` |
| `vox_conformer_small_ge2e` | `ge2e_softmax` | 2.2154 | 2.0441 | 2.2154 | `variable` |
| `vox_ecapa_tdnn_small_ext_softmax` | `extended_set_softmax` | 2.0999 | 2.0428 | 2.0999 | `variable` |

### Key Takeaways from the Converged Ablations & Controls

- **Study 1 (Stratified Batching at 4,600 Steps)**: Increasing within-domain hard-negative batch probability monotonically improves verification accuracy at full convergence (`p_within = 0.00`: `16.51%` → `0.75`: `15.81%` → `1.00`: **`15.00%`** Pooled 1-utt EER, `12.90% → 11.93% → 11.13%` 3-utt), because GE2E batch negatives drawn from the *same* acoustic corpus prevent trivial channel/language shortcut learning while temperature-balanced domain sampling (`|S_d|^0.35`) still trains a shared embedding space across all three domains.
- **Study 2 (Extended-Set, Dr-Vectors, Contrast & PFAS-SN)**: Blended extended-set softmax (`15.62%`) improves over pure Eq. 2 (`16.44%`) at `N = 8 × M = 4`; within PFAS-SN (`Study 2b`), the global ATP value anchor is decisive (`15.55%` with anchor vs. `16.71%` without on 1-utt, and `11.53%` vs. `14.13%` on 3-utt), while `shared_nonlinear` (`15.55%`) and `dot` (`15.72%`) outperform `maxpool` (`16.46%`).
- **Study 3 (Variable `T ~ U(90, 180)` vs. Fixed `T = 180` Crops)**: Fixed `T = 180` crops score `0.62 pp` lower on full-length (`≤ 5 s`) trials (`15.19%` vs. `15.81%`), whereas **variable-length `T ~ U(90, 180)` training lowers short `1.0 s` (`100`-frame) verification EER by `-1.06 pp` Pooled (`20.69%` vs. `21.75%`) and `-1.10 pp` on LibriSpeech (`14.10%` vs. `15.20%`)**.
- **Study 4 (Own-Schedule Horizon Scaling `400 → 4,600` Steps)**: Training separate cosine schedules per budget yields a strictly monotone reduction in Pooled 1-utt / 3-utt EER: `400` (`20.79% / 17.33%`) → `800` (`19.21% / 14.90%`) → `1,200` (`18.39% / 14.67%`) → `2,400` (`16.88% / 13.20%`) → `4,600` (`15.81% / 11.93%`).
- **Study 5 (Mamba Directionality)**: Bidirectional Mamba improves Universal Pooled 1-utt EER by **`-0.97 pp`** (`15.59%` vs. `16.56%`) and LibriSpeech in-domain 1-utt EER from `8.87%` to `8.60%`.
- **Study 7 (`pure_paper_mode=True` Presets at 4,600 Steps)**: Training the four exact paper presets on our 381-speaker Universal split shows `interspeech2021_dr_vectors_config(pure_paper_mode=True)` performing strongly (**`15.47% / 11.53%`** Pooled 1/3-utt EER), `odyssey2022_conformer_atp_config(pure_paper_mode=True)` reaching `16.82% / 13.87%`, `odyssey2022_pfas_config(pure_paper_mode=True)` (`32` tied keys, `2,048-D`) reaching `18.34% / 14.43%` (behind compact 4-key anchored `PFAS-SN` `15.55% / 11.53%` on `9.4k` utterances), and unnormalized `icassp2018_ge2e_config(pure_paper_mode=True)` (`3×768` LSTM without LayerNorm/residual) stalling at `26.59%` vs. `15.50%` for our LayerNorm+residual `univ_lstm_small_ge2e`.

---

## 11. Production Export: Hugging Face Hub & TensorFlow Lite (`FP32` and `INT8`)

Every trained model (`44/44`) is published on the Hugging Face Hub in the [**`wq2012/FlaxSpeaker` Collection**](https://huggingface.co/collections/wq2012/flaxspeaker) (`https://huggingface.co/wq2012/<experiment_name>`) and exported locally under [`pretrained_models/<experiment_name>/`](pretrained_models/) containing `config.json`, `model.safetensors`, `flax_model.msgpack`, `eval_results.json`, and `README.md`.

In addition, 6 converged models (3 Universal + 3 Single-Domain across Conformer-ATP, ECAPA-TDNN, and LSTM) were exported to **TensorFlow Lite (`.tflite`)** in both **FP32** and **dynamic-range INT8 quantized** formats (`180`-frame = `1.8 s` fixed encoder window) and evaluated via `scripts/eval_tflite_models.py`:

| Model | FP32 / INT8 size (MB) | Ratio | Primary eval set | TFLite FP32 EER (%) [95% CI] | TFLite INT8 EER (%) [95% CI] | JAX full-seq (`<=500f`) EER (%) | `cos(FP32, INT8)` mean (min) | CPU latency FP32 / INT8 (ms, 50-run mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/univ_conformer_small_ge2e`](https://huggingface.co/wq2012/univ_conformer_small_ge2e) | 8.25 / **2.33** | **`3.54×`** | `universal_pooled` | 17.40% [16.57, 18.21] | **17.30%** [16.52, 18.15] | 15.81% | **`0.999953`** (`0.999802`) | 9.93 / 9.35 ms |
| [`wq2012/univ_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/univ_ecapa_tdnn_small_ge2e) | 6.85 / **1.90** | **`3.61×`** | `universal_pooled` | 18.15% [17.41, 19.01] | **18.24%** [17.47, 19.03] | 16.34% | **`0.999972`** (`0.999904`) | 7.37 / 7.03 ms |
| [`wq2012/univ_lstm_small_ge2e`](https://huggingface.co/wq2012/univ_lstm_small_ge2e) | 7.09 / **2.06** | **`3.44×`** | `universal_pooled` | 18.25% [17.40, 19.12] | **18.25%** [17.44, 19.18] | 15.50% | **`0.999940`** (`0.999731`) | 28.57 / 15.71 ms |
| [`wq2012/ls_conformer_small_ge2e`](https://huggingface.co/wq2012/ls_conformer_small_ge2e) | 8.25 / **2.33** | **`3.54×`** | `librispeech` | 8.53% [7.53, 9.47] | **8.60%** [7.48, 9.47] | 7.77% | **`0.999958`** (`0.999754`) | 9.28 / 9.43 ms |
| [`wq2012/ls_ecapa_tdnn_small_ge2e`](https://huggingface.co/wq2012/ls_ecapa_tdnn_small_ge2e) | 6.85 / **1.90** | **`3.61×`** | `librispeech` | 11.47% [10.43, 12.73] | **11.47%** [10.41, 12.73] | 10.37% | **`0.999965`** (`0.999839`) | 7.33 / 7.09 ms |
| [`wq2012/ls_lstm_small_ge2e`](https://huggingface.co/wq2012/ls_lstm_small_ge2e) | 7.09 / **2.06** | **`3.44×`** | `librispeech` | 10.53% [9.33, 11.53] | **10.47%** [9.33, 11.55] | 9.33% | **`0.999935`** (`0.999757`) | 29.00 / 15.40 ms |

- **Quantization impact (`FP32` vs. `INT8`)**: Dynamic-range INT8 quantization shrinks `.tflite` binaries by **`3.44–3.61×`** while changing 180-frame verification EER by at most **`0.10` percentage points** (`cos(FP32, INT8)` mean `0.999935–0.999972`, min `0.999731` across all 1,988 evaluation utterances), and accelerates sequential LSTM inference on x86 CPU by **`1.8–1.9×`** (`28.57–29.00 ms → 15.40–15.71 ms`).
- **Fixed-window truncation cost (`180` frames vs. full sequence `≤ 500` frames)**: Because all `v0.4.0` encoders were trained with variable-length `T ~ U(90, 180)` crops, the 1.8 s fixed-window truncation gap vs. `≤ 5.0 s` JAX inference narrows to **`+0.76` to `+2.75` percentage points EER** (down from `+2.23` to `+3.74 pp` in `v0.3.0`).

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
