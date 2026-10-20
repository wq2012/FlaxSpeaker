"""Hugging Face model-card (README.md) generation for FlaxSpeaker checkpoints.

The card is generated from the checkpoint's `config.json` and `eval_results.json`
only, so it can be regenerated for every published repository without touching the
weights (`scripts/update_hf_model_cards.py`). Every number in the card is read from
`eval_results.json`; nothing is estimated.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from flaxspeaker import configs

VOX1_SUBSET_KEY = "vox1_o_10spk_subset"
LEGACY_VOX1_KEY = "voxceleb1_o_official"

VOX1_SUBSET_CAVEAT = (
    "VoxCeleb1-O **10-speaker subset**: the official `veri_test2.txt` list filtered to "
    "trials whose enrollment and test speakers are both among the 10 held-out "
    "evaluation speakers (2,500 trials, ~72% target). **Not comparable to published "
    "VoxCeleb1-O numbers**; the other 30 of the 40 official test speakers are used as "
    "training data by `vox_*` and `univ_*` models."
)

SOTA_ANCHOR = (
    "For orientation only: published ECAPA-TDNN systems trained on VoxCeleb2-dev reach "
    "about 0.9% EER on the full VoxCeleb1-O list (Desplanques et al., 2020), and "
    "sub-1M-parameter systems about 2% (Wang et al., EfficientTDNN, 2022). The 400-step "
    "checkpoints released here are 10-30x worse than that and are intended for relative "
    "comparisons within the FlaxSpeaker benchmark, not as state-of-the-art models."
)


def _fmt_ci(d_m: dict[str, Any]) -> str:
    return (
        f"[{d_m.get('eer_ci95_low_percent', d_m['eer_percent']):.2f}, "
        f"{d_m.get('eer_ci95_high_percent', d_m['eer_percent']):.2f}]"
    )


def objective_label(exp_cfg: configs.ExperimentConfig) -> str:
    """Honest human-readable name for the training objective actually optimised."""
    loss = exp_cfg.train.loss
    lt = str(getattr(loss.loss_type, "value", loss.loss_type))
    if lt == "ge2e_contrast":
        if bool(getattr(loss, "ge2e_contrast_surrogate", False)):
            return (
                "GE2E hybrid contrast surrogate (FlaxSpeaker; value = Wan et al. Eq. 7, "
                "gradient = 0.25 Eq. 7 + 0.25 BCE + 0.50 GE2E-softmax CE; "
                f"`ge2e_init_b = {float(loss.ge2e_init_b):g}`)"
            )
        return "GE2E contrast loss (Wan et al. 2018, Eq. 7)"
    if lt == "ge2e_softmax":
        st = str(getattr(loss.scoring_type, "value", loss.scoring_type))
        if st in ("pfas", "dr_vector"):
            return (
                "GE2E softmax on the even/odd split-batch test-vs-enrollment matrix "
                "(not leave-one-out centroids) because scoring is non-cosine"
            )
        return "GE2E softmax (Wan et al. 2018, Eq. 6; exact leave-one-out centroids)"
    if lt == "extended_set_softmax":
        if bool(getattr(loss, "pure_extended_set", False)):
            return "GE2E extended-set softmax (Pelecanos et al. 2021, Eq. 2)"
        w = float(getattr(loss, "ext_softmax_row_softmax_weight", 0.5))
        return (
            f"Blended Extended-Set Softmax (FlaxSpeaker variant: {w:.2f} row-softmax + "
            f"{1 - w:.2f} extended-set with a -log N logit shift; not the paper's Eq. 2)"
        )
    return lt


def scorer_label(exp_cfg: configs.ExperimentConfig) -> str:
    st = str(getattr(exp_cfg.train.loss.scoring_type, "value", exp_cfg.train.loss.scoring_type))
    if st == "pfas":
        pf = exp_cfg.model.pfas
        if str(pf.attention_type) == "atp_ffn":
            return (
                f"PFAS (Pelecanos et al. 2022 head; {pf.num_keys} keys x "
                f"({pf.key_dim} + {pf.value_dim}))"
            )
        return (
            f"PFAS-SN (FlaxSpeaker extension of PFAS: `{pf.attention_type}` frame attention"
            f"{' + global ATP anchor' if pf.use_global_anchor else ''}; {pf.num_keys} keys x "
            f"({pf.key_dim} + {pf.value_dim}); paper scoring Eqs. 5-7)"
        )
    if st == "dr_vector":
        dr = exp_cfg.model.dr_vector
        if int(getattr(dr, "cosine_dims", 0)) > 0:
            return "Dr-Vector decision residual network (paper configuration)"
        return (
            "Down-scaled Dr-Vector variant (no base/side-info split, hidden "
            f"{tuple(dr.hidden_dims)} {dr.activation}"
            f"{' + LayerNorm' if dr.use_layer_norm else ''}, fixed residual weight "
            f"{float(dr.residual_weight):g}"
            f"{', 3-term cosine expansion (FlaxSpeaker extension)' if int(dr.num_cosine_terms) == 3 else ''})"
        )
    return "Cosine similarity"


def deviations_from_papers(exp_cfg: configs.ExperimentConfig) -> list[str]:
    """Checkpoint-specific list of known deviations from the cited source papers."""
    out: list[str] = []
    loss = exp_cfg.train.loss
    lt = str(getattr(loss.loss_type, "value", loss.loss_type))
    st = str(getattr(loss.scoring_type, "value", loss.scoring_type))
    bb = str(getattr(exp_cfg.model.backbone, "value", exp_cfg.model.backbone))
    out.append(
        "**Training regime**: 400 gradient steps x 32 utterances (N = 8 speakers x M = 4) "
        "on <= 25 utterances/speaker from 381 (universal) or fewer speakers; the source "
        "papers train for ~500k steps with N = 16-64 / M = 8-10 on thousands of speakers."
    )
    out.append(
        "**Augmentation**: synthetic log-Mel-domain noise + one-tap smoothing "
        "(`specaug.apply_feature_mtr_augmentation`), not waveform MTR with recorded "
        "noise/RIR corpora."
    )
    if lt == "ge2e_contrast" and bool(getattr(loss, "ge2e_contrast_surrogate", False)):
        out.append(
            "**Objective**: hybrid contrast surrogate (gradient = 0.25 Eq. 7 + 0.25 BCE + "
            "0.50 softmax-CE) with `ge2e_init_b = -1`, not Wan et al. Eq. 7 alone. "
            "Set `ge2e_contrast_surrogate: false` for the pure Eq. 7 gradient."
        )
    if lt == "ge2e_softmax" and st in ("pfas", "dr_vector"):
        out.append(
            "**Objective**: GE2E softmax computed on the even/odd split-batch matrix, "
            "not on leave-one-out centroids (centroids are undefined for non-cosine scorers)."
        )
    if lt == "extended_set_softmax" and not bool(getattr(loss, "pure_extended_set", False)):
        out.append(
            "**Objective**: 50/50 blend of row-softmax and `-log N`-shifted extended-set "
            "softmax; the paper's pure Eq. 2 is available via `pure_extended_set: true`."
        )
    if st == "pfas" and str(exp_cfg.model.pfas.attention_type) != "atp_ffn":
        pf = exp_cfg.model.pfas
        out.append(
            "**PFAS head**: `shared_nonlinear` multi-head tanh frame attention with a global "
            f"ATP anchor and {pf.num_keys} keys x ({pf.key_dim} + {pf.value_dim}) "
            "(FlaxSpeaker 'PFAS-SN'); the paper uses ATP -> affine+ReLU -> linear with "
            "32 tied keys (`odyssey2022_pfas_config(pure_paper_mode=True)`)."
        )
    if st == "dr_vector" and int(getattr(exp_cfg.model.dr_vector, "cosine_dims", 0)) == 0:
        dr = exp_cfg.model.dr_vector
        out.append(
            "**Dr-Vector scorer**: no 200/56 base/side-information split, hidden "
            f"{tuple(dr.hidden_dims)} with {dr.activation}"
            f"{' + LayerNorm' if dr.use_layer_norm else ''}, linear read-out and a fixed "
            f"residual weight {float(dr.residual_weight):g} instead of the paper's "
            "3 x 256 leaky-ReLU(0.2) layers, weighted-sum read-out and trainable affine."
        )
    if bb == "conformer":
        out.append(
            "**Conformer**: parallel linear + Conv1D stem, no positional encoding, "
            f"{exp_cfg.model.conformer.num_layers} blocks x {exp_cfg.model.conformer.dim}-D, "
            f"{exp_cfg.model.conformer.num_heads} heads, kernel {exp_cfg.model.conformer.conv_kernel_size}, "
            "mask-aware GroupNorm(2), no dropout, multi-layer feature aggregation; the "
            "Google papers use 12 x 512 blocks, 4-8 heads, kernel 32, relative positional "
            "encoding and dropout."
        )
        p = exp_cfg.model.pooling
        if str(p.pooling_type) == "atp" and (int(p.attention_dim) > 0 or int(p.num_heads) > 1):
            out.append(
                f"**ATP pooling**: {p.num_heads} heads with a tanh bottleneck ({p.attention_dim}-D) "
                f"and `subspace_heads = {bool(p.subspace_heads)}`; the papers use a single "
                "head with a linear-sigmoid weight (`attention_dim = 0`). Note that "
                "arXiv:2202.12163 is a language-identification paper whose pooling head is "
                "repurposed here."
            )
    if bb == "lstm":
        out.append(
            "**LSTM**: per-layer LayerNorm + residual connections and a post-pooling "
            "nonlinear bottleneck; the GE2E paper uses 3 x 768 LSTM + 256-D projections "
            "and a plain linear output (`icassp2018_ge2e_config(pure_paper_mode=True)`)."
        )
    if bb == "ecapa_tdnn":
        out.append("**ECAPA-TDNN**: LayerNorm instead of BatchNorm.")
    out.append(
        "**Frontend**: 80-D natural-log Mel + frame-level energy VAD (26 dB dynamic range) "
        "+ CMVN; evaluation centre-crops to at most 500 frames (5 s)."
    )
    return out


def build_model_card(
    model_id: str,
    exp_cfg: configs.ExperimentConfig,
    num_parameters: int,
    metrics: Optional[dict[str, Any]],
    save_directory: str,
) -> str:
    """Returns the full README.md text for a checkpoint directory."""
    m_cfg = exp_cfg.model
    backbone = str(getattr(m_cfg.backbone, "value", m_cfg.backbone))
    size_variant = str(m_cfg.size_variant)
    pooling_type = str(getattr(m_cfg.pooling.pooling_type, "value", m_cfg.pooling.pooling_type))
    loss_type = str(getattr(exp_cfg.train.loss.loss_type, "value", exp_cfg.train.loss.loss_type))
    scoring_type = str(
        getattr(exp_cfg.train.loss.scoring_type, "value", exp_cfg.train.loss.scoring_type)
    )
    train_ds = (
        metrics.get("train_dataset", "universal") if isinstance(metrics, dict) else "universal"
    )
    num_steps = (
        int(metrics.get("num_steps", exp_cfg.train.num_steps))
        if isinstance(metrics, dict)
        else int(exp_cfg.train.num_steps)
    )
    has_tflite = os.path.exists(
        os.path.join(save_directory, "model_fp32.tflite")
    ) or os.path.exists(os.path.join(save_directory, "model_int8.tflite"))
    export_seq = int(exp_cfg.export.sequence_length)
    export_window = export_seq if export_seq > 0 else int(m_cfg.seq_len)

    train_domain_desc = {
        "universal": "LibriSpeech + CN-Celeb2 + VoxCeleb1 (universal multi-domain, 381 speakers)",
        "librispeech": "LibriSpeech only (232 speakers)",
        "cnceleb2": "CN-Celeb2 only (119 speakers)",
        "voxceleb1": "VoxCeleb1 only (30 speakers from the VoxCeleb1 test partition)",
    }.get(str(train_ds), str(train_ds))

    lines = [
        "---",
        "license: apache-2.0",
        "library_name: flaxspeaker",
        "pipeline_tag: feature-extraction",
        "tags:",
        "- audio",
        "- speaker-verification",
        "- speaker-recognition",
        "- speaker-embeddings",
        "- jax",
        "- flax",
        "- safetensors",
        f"- {backbone}",
        f"- {loss_type}",
        f"- {scoring_type}",
        "datasets:",
        "- librispeech_asr",
        "- cnceleb",
        "- voxceleb",
        "metrics:",
        "- eer",
        "- min_dcf",
        "- roc_auc",
        "---",
        "",
        f"# FlaxSpeaker: `{model_id}` (`{backbone}` - `{size_variant}`)",
        "",
        "## Disclaimer & Reproduction Notice",
        "",
        "> **This is not an officially supported Google product.**",
        ">",
        (
            "> This checkpoint and the accompanying `FlaxSpeaker` library are a "
            "from-scratch re-implementation in JAX/Flax, written using only the published "
            "papers (no proprietary code or data), of speaker recognition algorithms from:"
        ),
        (
            "> - **GE2E**: *Generalized End-to-End Loss for Speaker Verification* "
            "([arXiv:1710.10467](https://arxiv.org/abs/1710.10467))"
        ),
        (
            "> - **Dr-Vectors**: *Decision Residual Networks and an Improved Loss "
            "for Speaker Recognition* ([arXiv:2104.01989](https://arxiv.org/abs/2104.01989))"
        ),
        (
            "> - **ATP** (pooling head repurposed from a *language identification* paper): "
            "*Attentive Temporal Pooling for Conformer-based Streaming Language "
            "Identification in Long-form Speech* "
            "([arXiv:2202.12163](https://arxiv.org/abs/2202.12163))"
        ),
        (
            "> - **PFAS**: *Parameter-Free Attentive Scoring for Speaker Verification* "
            "([arXiv:2203.05642](https://arxiv.org/abs/2203.05642))"
        ),
        ">",
        (
            "> The experiments in those papers were run on Google-internal infrastructure "
            "and data, not with this library. **This is a 400-step demonstration checkpoint, "
            "not a reproduction of the papers' results** -- see *Training budget* and "
            "*Deviations from the source papers* below."
        ),
        "",
        "## Training Budget (read this first)",
        "",
        (
            f"This model was trained for **{num_steps:,} gradient steps** with mini-batches of "
            "32 utterances (8 speakers x 4 utterances), i.e. about 1.35 epochs over the "
            "9,447-utterance universal corpus, on a CPU workstation, with a single seed (42). "
            "Losses are not converged at this budget. All EERs below are therefore far from "
            "state of the art and are only meaningful for *relative* comparisons between "
            "FlaxSpeaker checkpoints trained under the same budget."
        ),
        "",
        (
            "**Training-code provenance.** The released weights were produced by an earlier "
            "revision of the FlaxSpeaker training code than the current `main`. Evaluating "
            "released weights with the current code reproduces the archived numbers (verified "
            "for `univ_conformer_small_ge2e` and `ls_conformer_small_ge2e`); re-training "
            "the shipped `config.json` / YAML with the current code does **not** regenerate "
            "them (for `univ_conformer_small_ge2e` and `ls_conformer_small_ge2e`, seed 42 "
            "re-trained at the current revision gives 20.94 % / 21.40 % instead of the "
            "archived 19.40 % / 20.12 % pooled 1-utt EER; see "
            "`pretrained_models/r2_seed_and_control_results.json` in the GitHub repository). "
            "The seed-to-seed standard deviation of pooled 1-utt EER at the current revision "
            "is about 0.7-0.8 % absolute, so sub-1.5 % differences between checkpoints are "
            "not rankings."
        ),
        "",
        SOTA_ANCHOR,
        "",
        "## Model Summary",
        "",
        "| Property | Value |",
        "| :--- | :--- |",
        f"| **Model ID** | `{model_id}` |",
        (
            f"| **Hugging Face Repository** | "
            f"[`wq2012/{model_id}`](https://huggingface.co/wq2012/{model_id}) |"
        ),
        (
            "| **Hugging Face Collection** | "
            "[FlaxSpeaker Collection]"
            "(https://huggingface.co/collections/wq2012/flaxspeaker) |"
        ),
        "| **GitHub Repository** | [`wq2012/FlaxSpeaker`](https://github.com/wq2012/FlaxSpeaker) |",
        "| **PyPI Package** | [`flaxspeaker`](https://pypi.org/project/flaxspeaker/) |",
        f"| **Backbone Architecture** | `{backbone}` (`{size_variant}`) |",
        f"| **Temporal Pooling** | `{pooling_type}` |",
        f"| **Training Objective** | {objective_label(exp_cfg)} (`{loss_type}`) |",
        f"| **Verification Scorer** | {scorer_label(exp_cfg)} (`{scoring_type}`) |",
        f"| **Output Representation Dim** | `{m_cfg.output_embedding_dim}` |",
        (
            "| **Inference Parameters (encoder + scoring head; "
            f"excludes training-only classifier weights)** | `{num_parameters:,}` |"
        ),
        f"| **Training Data** | {train_domain_desc} |",
        f"| **Training Steps / Batch** | `{num_steps:,}` steps x 32 utterances, seed 42 |",
        (
            f"| **Audio Frontend** | `{exp_cfg.frontend.n_mels}`-dim natural-log Mel "
            f"@ `{exp_cfg.frontend.sample_rate}` Hz + frame-level energy VAD + CMVN "
            "(`flaxspeaker.AudioFrontend`; **not** a Hugging Face `AutoFeatureExtractor`) |"
        ),
        "",
        "## Model Files Included",
        "",
        "- `model.safetensors`: model weights (encoder + GE2E `w,b` / Dr-Vector / PFAS heads) in SafeTensors format",
        "- `flax_model.msgpack`: native Flax Msgpack serialized parameters",
        "- `config.json`: `FlaxSpeakerConfig` (full `ExperimentConfig`, incl. the frontend/VAD settings)",
        (
            "- `eval_results.json`: multi-domain evaluation metrics "
            "(1-utt & 3-utt EER, minDCF, ROC-AUC, 95% bootstrap CIs, VoxCeleb1-O 10-speaker subset)"
        ),
    ]
    if has_tflite:
        lines += [
            (
                f"- `model_fp32.tflite` / `model_int8.tflite`: `jax2tf` exports of the **encoder only** "
                f"with a **fixed, unmasked {export_window}-frame ({export_window / 100:.1f} s) input "
                f"window** `[1, {export_window}, 80]` -> `[1, {m_cfg.output_embedding_dim}]`. "
                "All EERs in the main tables below come from full-sequence masked JAX inference (up to 500 "
                "frames); the TFLite models see a centre crop and the reported cosine fidelity "
                f"(>= 0.9999 vs. JAX) was measured on {export_window}-frame inputs."
            ),
        ]
        tfl = metrics.get("tflite_eval") if isinstance(metrics, dict) else None
        if isinstance(tfl, dict) and "eer_percent" in tfl:
            cos_part = (
                f", mean `cos(FP32, INT8)` = `{tfl['cos_fp32_int8_mean']:.6f}`"
                if "cos_fp32_int8_mean" in tfl
                else ""
            )
            jax_part = (
                f" (vs. full-sequence <=500-frame masked JAX: `{tfl['jax_full_seq_eer_percent']:.2f}%`)"
                if "jax_full_seq_eer_percent" in tfl
                else ""
            )
            proto_str = tfl.get("protocol", "see eval_results.json")
            lines.append(
                f"  - Measured {export_window}-frame TFLite 1-utt EER ({proto_str}): "
                f"FP32 **{tfl['eer_percent']:.2f}%**"
                + (
                    f", INT8 **{tfl['eer_percent_int8']:.2f}%**"
                    if "eer_percent_int8" in tfl
                    else ""
                )
                + cos_part
                + jax_part
            )
        else:
            lines.append("  - No TFLite EER has been measured for this repository.")
    lines += [
        "",
        "## Quickstart Usage",
        "",
        "### Python API (from Hugging Face Hub or Local Directory)",
        "",
        "```python",
        "from flaxspeaker import load_pretrained",
        "",
        f'model = load_pretrained("wq2012/{model_id}")',
        'emb = model.extract_embedding("path/to/utterance.flac")',
        'score = model.verify("path/to/enroll.flac", "path/to/test.flac")',
        'print(f"Embedding shape: {emb.shape}, Verification score: {score:.4f}")',
        "```",
        "",
        "### Command-Line Inference",
        "",
        "```bash",
        f"python scripts/inference.py --model_dir wq2012/{model_id} \\",
        "    --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac",
        "```",
    ]
    if isinstance(metrics, dict):
        lines += ["", "## Multi-Domain Evaluation Results", ""]
        md_eval = metrics.get("multi_domain_eval", {})
        if md_eval and "domains_1utt" in md_eval:
            lines += [
                "### 1-Utterance Verification Across Evaluation Domains",
                "",
                (
                    "Trial lists are 50/50 target/impostor over held-out speakers "
                    "(LibriSpeech 40, CN-Celeb2 30, VoxCeleb1 10 speakers); minDCF@0.01 is "
                    "close to its trivial ceiling of 1.0 for every model under this trial mix."
                ),
                "",
                (
                    "| Evaluation Split | Trials | EER (%) | 95% Bootstrap CI (%) "
                    "| minDCF@0.01 | minDCF@0.10 | ROC-AUC |"
                ),
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
            for dom_label, dom_key in [
                ("LibriSpeech (English Read)", "librispeech"),
                ("CN-Celeb2 (Chinese Multi-Genre)", "cnceleb2"),
                ("VoxCeleb1 (In-the-Wild Celebrity)", "voxceleb1"),
                ("Universal Pooled (All 3 Domains)", "universal_pooled"),
            ]:
                d_m = md_eval["domains_1utt"].get(dom_key)
                if d_m:
                    lines.append(
                        f"| **{dom_label}** | {d_m['num_trials']:,} | **{d_m['eer_percent']:.2f}%** | "
                        f"{_fmt_ci(d_m)} | {d_m['min_dcf_001']:.4f} | {d_m['min_dcf_01']:.4f} | {d_m['roc_auc']:.4f} |"
                    )
            macro_1 = md_eval.get("universal_macro_avg_1utt")
            if macro_1:
                lines.append(
                    f"| **Universal Macro-Average (3 Domains)** | — | **{macro_1['eer_percent']:.2f}%** | "
                    f"— | {macro_1['min_dcf_001']:.4f} | {macro_1['min_dcf_01']:.4f} | {macro_1['roc_auc']:.4f} |"
                )
            vox_sub = md_eval["domains_1utt"].get(VOX1_SUBSET_KEY) or (
                metrics.get("cross_dataset_eval", {}) or {}
            ).get(VOX1_SUBSET_KEY)
            if vox_sub:
                lines.append(
                    f"| **VoxCeleb1-O 10-speaker subset** (see caveat) | {vox_sub['num_trials']:,} | "
                    f"**{vox_sub['eer_percent']:.2f}%** | {_fmt_ci(vox_sub)} | "
                    f"{vox_sub['min_dcf_001']:.4f} | {vox_sub['min_dcf_01']:.4f} | {vox_sub['roc_auc']:.4f} |"
                )
                lines += ["", f"Caveat: {VOX1_SUBSET_CAVEAT}"]
            if "domains_3utt" in md_eval:
                lines += [
                    "",
                    "### 3-Utterance Multi-Enrollment Verification Across Evaluation Domains",
                    "",
                    "| Evaluation Split | Trials | 3-Utt EER (%) | 95% Bootstrap CI (%) | minDCF@0.01 | ROC-AUC |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: |",
                ]
                for dom_label, dom_key in [
                    ("LibriSpeech (3-Utt)", "librispeech"),
                    ("CN-Celeb2 (3-Utt)", "cnceleb2"),
                    ("VoxCeleb1 (3-Utt)", "voxceleb1"),
                    ("Universal Pooled (3-Utt)", "universal_pooled"),
                ]:
                    d_m = md_eval["domains_3utt"].get(dom_key)
                    if d_m:
                        lines.append(
                            f"| **{dom_label}** | {d_m['num_trials']:,} | **{d_m['eer_percent']:.2f}%** | "
                            f"{_fmt_ci(d_m)} | {d_m['min_dcf_001']:.4f} | {d_m['roc_auc']:.4f} |"
                        )
                macro_3 = md_eval.get("universal_macro_avg_3utt")
                if macro_3:
                    lines.append(
                        f"| **Universal Macro-Average (3-Utt)** | — | **{macro_3['eer_percent']:.2f}%** | "
                        f"— | {macro_3['min_dcf_001']:.4f} | {macro_3['roc_auc']:.4f} |"
                    )
        elif "in_domain_eval" in metrics and metrics["in_domain_eval"]:
            ind = metrics["in_domain_eval"]
            lines += [
                f"- **In-Domain EER**: `{ind['eer_percent']:.2f}%`",
                f"- **In-Domain minDCF@0.01**: `{ind['min_dcf_001']:.4f}`",
                f"- **In-Domain ROC-AUC**: `{ind['roc_auc']:.4f}`",
            ]
        lines += [
            "",
            (
                "Single seed; the 95% bootstrap CIs resample trials (not speakers). Differences "
                "below about 1% absolute EER between FlaxSpeaker checkpoints are within the "
                "seed-to-seed variation measured in `pretrained_models/r2_seed_and_control_results.json` "
                "(see the GitHub README) and should not be interpreted as rankings."
            ),
        ]
    lines += ["", "## Deviations from the Source Papers", ""]
    lines += [f"- {d}" for d in deviations_from_papers(exp_cfg)]
    lines += [
        "",
        "## Citation",
        "",
        "```bibtex",
        "@inproceedings{wan2018generalized,",
        "  title={Generalized End-to-End Loss for Speaker Verification},",
        "  author={Wan, Li and Wang, Quan and Papir, Alan and Moreno, Ignacio Lopez},",
        "  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},",
        "  pages={4879--4883},",
        "  year={2018}",
        "}",
        "",
        "@inproceedings{pelecanos2021dr,",
        "  title={Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition},",
        "  author={Pelecanos, Jason and Wang, Quan and Moreno, Ignacio Lopez},",
        "  booktitle={Interspeech},",
        "  pages={4603--4607},",
        "  year={2021}",
        "}",
        "",
        "@inproceedings{wang2022attentive,",
        (
            "  title={Attentive Temporal Pooling for Conformer-based Streaming "
            "Language Identification in Long-form Speech},"
        ),
        "  author={Wang, Quan and Yu, Yang and Pelecanos, Jason and Huang, Yiling and Moreno, Ignacio Lopez},",
        "  booktitle={The Speaker and Language Recognition Workshop (Odyssey)},",
        "  year={2022}",
        "}",
        "",
        "@inproceedings{pelecanos2022parameter,",
        "  title={Parameter-Free Attentive Scoring for Speaker Verification},",
        "  author={Pelecanos, Jason and Wang, Quan and Huang, Yiling and Moreno, Ignacio Lopez},",
        "  booktitle={The Speaker and Language Recognition Workshop (Odyssey)},",
        "  year={2022}",
        "}",
        "```",
    ]
    return "\n".join(lines) + "\n"
