#!/usr/bin/env python3
"""Generate publication-quality vector PDF and PNG figures for the FlaxSpeaker paper."""

import argparse
import json
import os

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

# Embed TrueType (Type 42) fonts in PDF/PS output (no Type 3 bitmap fonts)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if os.path.basename(_PARENT_DIR) == "FlaxSpeaker":
    _DEFAULT_REPO_ROOT = _PARENT_DIR
    _DEFAULT_FIG_DIR = os.path.join(_PARENT_DIR, "resources")
else:
    _DEFAULT_REPO_ROOT = os.path.join(
        os.path.dirname(_PARENT_DIR), "FlaxSpeaker"
    )
    _DEFAULT_FIG_DIR = _THIS_DIR

PAPER_FIG_DIR = os.environ.get("FLAXSPEAKER_PAPER_FIG_DIR", _DEFAULT_FIG_DIR)
REPO_ROOT = os.environ.get("FLAXSPEAKER_REPO_ROOT", _DEFAULT_REPO_ROOT)


def _ci(entry, lo_default=0.8):
    """Returns (err_low, err_high) for an eval entry with bootstrap CI fields."""
    v = entry["eer_percent"]
    return (
        v - entry.get("eer_ci95_low_percent", v - lo_default),
        entry.get("eer_ci95_high_percent", v + lo_default) - v,
    )


def draw_architecture_diagram(
    repo_root: str = REPO_ROOT,
    paper_fig_dir: str = PAPER_FIG_DIR,
) -> None:
    """Draws Figure 1: End-to-End FlaxSpeaker Universal Speaker Recognition Pipeline."""
    os.makedirs(paper_fig_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_root, "resources"), exist_ok=True)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(11.6, 4.05), dpi=300)
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4.05)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    ax.text(
        5.8,
        3.86,
        "FlaxSpeaker: Multi-Domain Speaker Verification Pipeline (JAX / Flax)",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#0F172A",
    )

    def add_stage(x, y, w, h, title, shape_tag, items, bg, border, hdr):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.10",
            linewidth=1.5,
            edgecolor=border,
            facecolor=bg,
        )
        ax.add_patch(rect)
        hdr_h = 0.36
        hdr_rect = FancyBboxPatch(
            (x, y + h - hdr_h),
            w,
            hdr_h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.1,
            edgecolor=border,
            facecolor=hdr,
        )
        ax.add_patch(hdr_rect)
        ax.text(
            x + w / 2,
            y + h - hdr_h / 2,
            title,
            ha="center",
            va="center",
            fontsize=9.3,
            fontweight="bold",
            color="#FFFFFF",
        )
        # Tensor shape pill
        ax.text(
            x + w / 2,
            y + h - hdr_h - 0.19,
            shape_tag,
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color=hdr,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="#FFFFFF",
                edgecolor=border,
                lw=0.8,
            ),
        )
        top_y = y + h - hdr_h - 0.41
        step = (h - hdr_h - 0.46) / max(len(items), 1)
        for idx, item in enumerate(items):
            prefix = "  " if item.startswith("  ") else "• "
            ax.text(
                x + 0.10,
                top_y - idx * step,
                f"{prefix}{item}",
                ha="left",
                va="center",
                fontsize=8.1,
                color="#1E293B",
            )

    def add_arrow(x1, y1, x2, y2, label=""):
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>,head_length=5,head_width=3.2",
            mutation_scale=1.0,
            lw=1.6,
            color="#334155",
        )
        ax.add_patch(arr)
        if label:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.11,
                label,
                ha="center",
                va="center",
                fontsize=7.6,
                fontweight="bold",
                color="#0F172A",
            )

    # Top row: 4 main forward-pass stages
    y_top, h_top = 1.74, 1.92
    add_stage(
        0.15,
        y_top,
        2.50,
        h_top,
        "1. Multi-Domain & VAD",
        "Waveform (B, L) @ 16 kHz",
        [
            "LibriSpeech + CN-Celeb2 + Vox1",
            "Namespace <dom>:<spk> (381 spk)",
            "Peak AGC (1.0) + 80-d log-Mel",
            "Energy VAD: max($-42$ dB,",
            "  $P_{95}-26$ dB, $P_{10}+4.5$ dB) + CMVN",
        ],
        "#EFF6FF",
        "#3B82F6",
        "#1D4ED8",
    )

    add_stage(
        2.95,
        y_top,
        2.50,
        h_top,
        "2. Stratified Batch & Aug",
        "Features (B, T, 80) + Mask (B, T)",
        [
            "75% Within / 25% Cross-Domain",
            "Domain temp: $p_d \\propto |\\mathcal{S}_d|^{0.35}$",
            "Mutually exclusive aug: 35% noise,",
            "  32.5% SpecAug, 32.5% clean",
            "Variable crop: $T \\sim \\mathcal{U}(90, 180)$ + mask",
            "Converged $S_{\\mathrm{safe}}{=}X{+}2$ ep (1k–4.6k st)",
        ],
        "#F0FDF4",
        "#22C55E",
        "#15803D",
    )

    add_stage(
        5.75,
        y_top,
        2.65,
        h_top,
        "3. Masked Backbones (x6)",
        "Frame states $\\mathbf{H} \\in \\mathbb{R}^{B \\times T \\times D}$",
        [
            "Conformer (linear + Conv1D stem,",
            "  GroupNorm $G{=}2$, MFA)",
            "Mamba SSM (associative scan)",
            "Transformer (pre-LN + MFA)",
            "ECAPA-TDNN | ResNet-18 | (Bi)LSTM",
        ],
        "#FEFCE8",
        "#EAB308",
        "#A16207",
    )

    add_stage(
        8.70,
        y_top,
        2.75,
        h_top,
        "4. Pooling Heads & Bottleneck",
        "Embedding $\\mathbf{e} \\in \\mathbb{R}^{B \\times d_e}$ (PFAS-SN: 640-D)",
        [
            "7 heads: ATP ($N_h{=}4$ subspace), ASP,",
            "  SAP, Stats, Mean, Last, CumStats",
            "  → LN → FC(256) → LN → ReLU → $L_2$",
            "PFAS-SN (bypasses FC(256)): $K{=}4$ keys",
            "  (32-D) + values (128-D) = 640-D",
        ],
        "#FAF5FF",
        "#A855F7",
        "#7E22CE",
    )

    # Arrows across top row
    add_arrow(2.65, y_top + 0.96, 2.95, y_top + 0.96)
    add_arrow(5.45, y_top + 0.96, 5.75, y_top + 0.96)
    add_arrow(8.40, y_top + 0.96, 8.70, y_top + 0.96)

    # Bottom row: Losses & Scoring + Deployment
    y_bot, h_bot = 0.08, 1.24
    add_stage(
        0.15,
        y_bot,
        6.60,
        h_bot,
        "5. Nine Losses (7 Benchmarked + 2 Reference) & Three Scorers",
        "Batch ($N{=}8$, $M{=}4$) → loss $\\mathcal{L}$;  trial ($\\mathbf{e}_t$, $\\mathbf{e}_e$) → score $s$",
        [
            "Losses: GE2E softmax / contrast (+ hybrid surrogate) | extended-set softmax (pure Eq. 2 or blended)",
            "  + ArcFace, CosFace, SphereFace (linear margin warm-up), Triplet + Softmax, Pairwise (tested only)",
            (
                "Scoring: cosine | Dr-Vector decision net "
                "($\\lambda_{res}{=}0.25$ variant / paper preset) | PFAS (global $L_2$)"
            ),
        ],
        "#FFF1F2",
        "#F43F5E",
        "#BE123C",
    )

    add_stage(
        7.10,
        y_bot,
        4.35,
        h_bot,
        "6. Edge Export & Open-Source Ecosystem",
        "44 HF Hub Models + TFLite FP32/INT8",
        [
            "Hugging Face Hub (wq2012/*): SafeTensors + config.json",
            "jax2tf → SavedModel → TFLite FP32 / INT8 (1.8 s window)",
            "Gradio demo (real inference) + tests.py CI (flake8 / pytype)",
        ],
        "#ECFEFF",
        "#06B6D4",
        "#0E7490",
    )

    # Connecting arrows from Stage 4 to Stages 5 & 6
    # Stage 4 -> Stage 5 (loss / scoring) and Stage 4 -> Stage 6 (export): orthogonal routing.
    y_mid = (y_top + y_bot + h_bot) / 2
    for (xa, ya, xb, yb) in ((9.40, y_top, 9.40, y_mid), (9.40, y_mid, 6.40, y_mid)):
        ax.add_patch(
            FancyArrowPatch(
                (xa, ya), (xb, yb), arrowstyle="-", lw=1.6, color="#334155", zorder=6
            )
        )
    add_arrow(6.40, y_mid, 6.40, y_bot + h_bot)
    add_arrow(10.60, y_top, 10.60, y_bot + h_bot)
    ax.text(7.90, y_mid + 0.03, "train / score", ha="center", va="bottom", fontsize=7.4, color="#0F172A")
    ax.text(10.66, y_mid, "export", ha="left", va="center", fontsize=7.4, color="#0F172A")

    plt.tight_layout(pad=0.15)
    fig.savefig(
        os.path.join(paper_fig_dir, "fig1_architecture.pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(repo_root, "resources", "architecture_diagram.png"),
        bbox_inches="tight",
        dpi=250,
    )
    plt.close(fig)


def draw_benchmark_charts(
    repo_root: str = REPO_ROOT,
    paper_fig_dir: str = PAPER_FIG_DIR,
    benchmark_json: str | None = None,
) -> None:
    """Draws Figure 2: 4-Panel Empirical Benchmark Summary from benchmark_results.json."""
    os.makedirs(paper_fig_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_root, "resources"), exist_ok=True)
    bench_path = benchmark_json or os.path.join(
        repo_root, "pretrained_models", "benchmark_results.json"
    )
    with open(
        bench_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
    by_name = data["experiments"]

    plt.close("all")
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel (a): Universal Backbones 1-Utt EER across 3 domains + Pooled
    ax = axes[0, 0]
    backbones = [
        ("Conformer", "univ_conformer_small_ge2e"),
        ("Mamba", "univ_mamba_small_ge2e"),
        ("Transformer", "univ_transformer_small_ge2e"),
        ("LSTM", "univ_lstm_small_ge2e"),
        ("ECAPA-TDNN", "univ_ecapa_tdnn_small_ge2e"),
        ("ResNet-18", "univ_resnet_small_ge2e"),
    ]
    x = np.arange(len(backbones))
    w = 0.20
    ls_vals = [
        by_name[k]["multi_domain_eval"]["domains_1utt"]["librispeech"][
            "eer_percent"
        ]
        for _, k in backbones
    ]
    cn_vals = [
        by_name[k]["multi_domain_eval"]["domains_1utt"]["cnceleb2"][
            "eer_percent"
        ]
        for _, k in backbones
    ]
    vx_vals = [
        by_name[k]["multi_domain_eval"]["domains_1utt"]["voxceleb1"][
            "eer_percent"
        ]
        for _, k in backbones
    ]
    pl_vals = [
        by_name[k]["multi_domain_eval"]["domains_1utt"]["universal_pooled"][
            "eer_percent"
        ]
        for _, k in backbones
    ]
    pl_ci_low = [
        pl_vals[i]
        - by_name[k]["multi_domain_eval"]["domains_1utt"][
            "universal_pooled"
        ].get("eer_ci95_low_percent", pl_vals[i] - 0.8)
        for i, (_, k) in enumerate(backbones)
    ]
    pl_ci_high = [
        by_name[k]["multi_domain_eval"]["domains_1utt"][
            "universal_pooled"
        ].get("eer_ci95_high_percent", pl_vals[i] + 0.8)
        - pl_vals[i]
        for i, (_, k) in enumerate(backbones)
    ]

    def _dom_ci(dom):
        errs = [_ci(by_name[k]["multi_domain_eval"]["domains_1utt"][dom]) for _, k in backbones]
        return [[e[0] for e in errs], [e[1] for e in errs]]

    err_kw = {"lw": 0.8, "ecolor": "#475569"}
    ax.bar(
        x - 1.5 * w,
        ls_vals,
        w,
        yerr=_dom_ci("librispeech"),
        capsize=1.8,
        label="LibriSpeech (EN)",
        color="#3B82F6",
        error_kw=err_kw,
    )
    ax.bar(
        x - 0.5 * w,
        cn_vals,
        w,
        yerr=_dom_ci("cnceleb2"),
        capsize=1.8,
        label="CN-Celeb2 (ZH)",
        color="#F59E0B",
        error_kw=err_kw,
    )
    vx_ci = _dom_ci("voxceleb1")
    ax.bar(
        x + 0.5 * w,
        vx_vals,
        w,
        yerr=vx_ci,
        capsize=1.8,
        label="VoxCeleb1 (EN)",
        color="#10B981",
        error_kw=err_kw,
    )
    bars_p = ax.bar(
        x + 1.5 * w,
        pl_vals,
        w,
        yerr=[pl_ci_low, pl_ci_high],
        capsize=2.2,
        label="Universal Pooled",
        color="#1E293B",
        error_kw={"lw": 0.9, "ecolor": "#475569"},
    )
    for b, c_hi in zip(bars_p, pl_ci_high):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + c_hi + 0.45,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7.0,
            fontweight="bold",
            color="#0F172A",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in backbones], fontsize=8.8)
    ax.set_ylabel("1-Utterance EER (%) [Lower is Better]", fontsize=9.2)
    ax.set_title(
        "(a) Universal backbones (Small, GE2E, 4,600 steps); 95% CIs",
        fontsize=10.0,
        fontweight="bold",
    )
    ax.set_ylim(0, 32.0)
    ax.legend(fontsize=7.8, ncol=2, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    # Panel (b): Universal Losses & Scoring (Conformer-Small)
    ax = axes[0, 1]
    losses = [
        ("GE2E\nATP+cos", "univ_conformer_small_ge2e"),
        ("GE2E-C$^*$\nATP+cos", "univ_conformer_small_ge2e_contrast"),
        ("BXS$^*$\nATP+DrV$^*$", "univ_conformer_small_dr_vectors"),
        ("GE2E\nPFAS-SN$^*$", "univ_conformer_small_pfas_ge2e"),
        ("BXS$^*$\nPFAS-SN$^*$", "univ_conformer_small_pfas"),
        ("BXS$^*$\nATP+cos", "univ_conformer_small_ext_softmax"),
        ("Triplet\nATP+cos", "univ_conformer_small_triplet"),
        ("ArcFace\nATP+cos", "univ_conformer_small_arcface"),
        ("CosFace\nATP+cos", "univ_conformer_small_cosface"),
        ("SphereF.\nATP+cos", "univ_conformer_small_sphereface"),
    ]
    x2 = np.arange(len(losses))
    w2 = 0.35
    e1 = [
        by_name[k]["multi_domain_eval"]["domains_1utt"]["universal_pooled"][
            "eer_percent"
        ]
        for _, k in losses
    ]
    e1_lo = [
        e1[i]
        - by_name[k]["multi_domain_eval"]["domains_1utt"][
            "universal_pooled"
        ].get("eer_ci95_low_percent", e1[i] - 0.8)
        for i, (_, k) in enumerate(losses)
    ]
    e1_hi = [
        by_name[k]["multi_domain_eval"]["domains_1utt"][
            "universal_pooled"
        ].get("eer_ci95_high_percent", e1[i] + 0.8)
        - e1[i]
        for i, (_, k) in enumerate(losses)
    ]
    e3 = [
        by_name[k]["multi_domain_eval"]["domains_3utt"]["universal_pooled"][
            "eer_percent"
        ]
        for _, k in losses
    ]
    e3_lo = [
        e3[i]
        - by_name[k]["multi_domain_eval"]["domains_3utt"][
            "universal_pooled"
        ].get("eer_ci95_low_percent", e3[i] - 1.2)
        for i, (_, k) in enumerate(losses)
    ]
    e3_hi = [
        by_name[k]["multi_domain_eval"]["domains_3utt"][
            "universal_pooled"
        ].get("eer_ci95_high_percent", e3[i] + 1.2)
        - e3[i]
        for i, (_, k) in enumerate(losses)
    ]
    b1 = ax.bar(
        x2 - w2 / 2,
        e1,
        w2,
        yerr=[e1_lo, e1_hi],
        capsize=1.8,
        label="1-Utt Pooled EER (%) [95% CI]",
        color="#2563EB",
        error_kw={"lw": 0.85, "ecolor": "#1E3A8A"},
    )
    b3 = ax.bar(
        x2 + w2 / 2,
        e3,
        w2,
        yerr=[e3_lo, e3_hi],
        capsize=1.8,
        label="3-Utt Pooled EER (%) [95% CI]",
        color="#16A34A",
        error_kw={"lw": 0.85, "ecolor": "#14532D"},
    )
    for b, hi in zip(b1, e1_hi):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + hi + 0.45,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6.8,
            color="#1E3A8A",
            fontweight="bold",
        )
    for b, hi in zip(b3, e3_hi):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + hi + 0.45,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6.8,
            color="#15803D",
            fontweight="bold",
        )
    ax.set_xticks(x2)
    ax.set_xticklabels([item[0] for item in losses], fontsize=6.9)
    ax.set_ylabel("Universal Pooled EER (%)", fontsize=9.2)
    ax.set_title(
        "(b) Objective × pool+scorer (Conformer-Small, 4,600 st); $^*$ = variant",
        fontsize=10.0,
        fontweight="bold",
    )
    ax.set_ylim(0, 29.5)
    ax.legend(fontsize=8.0, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    # Panel (c): Single-Domain vs Universal Cross-Domain Generalization
    ax = axes[1, 0]
    train_sets = [
        ("LibriSpeech\nOnly", "ls_conformer_small_ge2e"),
        ("CN-Celeb2\nOnly", "cn_conformer_small_ge2e"),
        ("VoxCeleb1\nOnly", "vox_conformer_small_ge2e"),
        ("Univ Mix\n(GE2E)", "univ_conformer_small_ge2e"),
        ("Univ Mix\n(Dr-Vec)", "univ_conformer_small_dr_vectors"),
        ("Univ Mix\n(ArcFace)", "univ_conformer_small_arcface"),
    ]
    x3 = np.arange(len(train_sets))
    w3 = 0.19
    ls3 = [
        by_name[k]["multi_domain_eval"]["domains_1utt"]["librispeech"][
            "eer_percent"
        ]
        for _, k in train_sets
    ]
    cn3 = [
        by_name[k]["multi_domain_eval"]["domains_1utt"]["cnceleb2"][
            "eer_percent"
        ]
        for _, k in train_sets
    ]
    vx3 = [
        by_name[k]["multi_domain_eval"]["domains_1utt"]["voxceleb1"][
            "eer_percent"
        ]
        for _, k in train_sets
    ]
    mc3 = [
        by_name[k]["multi_domain_eval"]["universal_macro_avg_1utt"][
            "eer_percent"
        ]
        for _, k in train_sets
    ]

    def _dom_ci3(dom):
        errs = [_ci(by_name[k]["multi_domain_eval"]["domains_1utt"][dom]) for _, k in train_sets]
        return [[e[0] for e in errs], [e[1] for e in errs]]

    err_kw3 = {"lw": 0.8, "ecolor": "#475569"}
    ax.bar(
        x3 - 1.5 * w3,
        ls3,
        w3,
        yerr=_dom_ci3("librispeech"),
        capsize=1.8,
        label="Eval: LibriSpeech",
        color="#3B82F6",
        error_kw=err_kw3,
    )
    ax.bar(
        x3 - 0.5 * w3,
        cn3,
        w3,
        yerr=_dom_ci3("cnceleb2"),
        capsize=1.8,
        label="Eval: CN-Celeb2",
        color="#F59E0B",
        error_kw=err_kw3,
    )
    vx3_ci = _dom_ci3("voxceleb1")
    ax.bar(
        x3 + 0.5 * w3,
        vx3,
        w3,
        yerr=vx3_ci,
        capsize=1.8,
        label="Eval: VoxCeleb1",
        color="#10B981",
        error_kw=err_kw3,
    )
    bm = ax.bar(
        x3 + 1.5 * w3, mc3, w3, label="3-Domain Macro-Avg (point estimate)", color="#7C3AED"
    )
    for b in bm:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.45,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7.0,
            fontweight="bold",
            color="#5B21B6",
        )
    ax.set_xticks(x3)
    ax.set_xticklabels([t[0] for t in train_sets], fontsize=8.0)
    ax.set_ylabel("1-Utterance EER (%)", fontsize=9.2)
    ax.set_title(
        "(c) Single-domain specialists vs. universal training\n"
        "(95% CIs on per-domain bars; macro-average = point estimate)",
        fontsize=10.0,
        fontweight="bold",
    )
    ax.set_ylim(0, 34.5)
    ax.legend(fontsize=7.8, ncol=2, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    # Panel (d): TFLite FP32 vs INT8 Size (decimal MB = 10^6 bytes), Latency & Cosine Fidelity
    ax = axes[1, 1]
    tflite_models = [
        ("Univ Conformer-S", "univ_conformer_small_ge2e"),
        ("Univ ECAPA-S", "univ_ecapa_tdnn_small_ge2e"),
        ("Univ LSTM-S", "univ_lstm_small_ge2e"),
    ]
    t_by_name = data["tflite_exports"]
    all_ratios = [
        t_by_name[k]["size_bytes"] / t_by_name[k.replace("_fp32", "_int8")]["size_bytes"]
        for k in t_by_name
        if k.endswith("_fp32")
    ]
    r_min, r_max = min(all_ratios), max(all_ratios)
    x4 = np.arange(len(tflite_models))
    w4 = 0.30
    fp32_mb = [
        t_by_name[f"{k}_fp32"]["size_bytes"] / 1e6
        for _, k in tflite_models
    ]
    int8_mb = [
        t_by_name[f"{k}_int8"]["size_bytes"] / 1e6
        for _, k in tflite_models
    ]
    int8_cos = [
        by_name[k].get("tflite_eval", {}).get(
            "cos_fp32_int8_mean",
            t_by_name[f"{k}_int8"]["cosine_fidelity_vs_flax"],
        )
        for _, k in tflite_models
    ]
    fp32_lat = [
        by_name[k].get("tflite_eval", {}).get("latency_ms_fp32_mean_50runs", 0.0)
        for _, k in tflite_models
    ]
    int8_lat = [
        by_name[k].get("tflite_eval", {}).get("latency_ms_int8_mean_50runs", 0.0)
        for _, k in tflite_models
    ]

    bf = ax.bar(
        x4 - w4 / 2, fp32_mb, w4, label="TFLite FP32 (MB)", color="#64748B"
    )
    bi = ax.bar(
        x4 + w4 / 2,
        int8_mb,
        w4,
        label=f"TFLite INT8 (MB, {r_min:.2f}–{r_max:.2f}× smaller)",
        color="#0891B2",
    )
    for idx, b in enumerate(bf):
        lat_str = f"\n({fp32_lat[idx]:.1f} ms)" if fp32_lat[idx] > 0 else ""
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.16,
            f"{b.get_height():.2f} MB{lat_str}",
            ha="center",
            va="bottom",
            fontsize=6.9,
            color="#334155",
        )
    for idx, b in enumerate(bi):
        lat_str = f" ({int8_lat[idx]:.1f} ms)" if int8_lat[idx] > 0 else ""
        ax.text(
            b.get_x() + 0.01,
            b.get_height() + 0.16,
            f"{b.get_height():.2f} MB{lat_str}\ncos={int8_cos[idx]:.5f}",
            ha="left",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            color="#0E7490",
        )
    ax.set_xticks(x4)
    ax.set_xticklabels([m[0] for m in tflite_models], fontsize=8.8)
    ax.set_xlim(-0.55, 2.82)
    ax.set_ylabel("Model File Size (MB, $10^6$ bytes)", fontsize=9.2)
    ax.set_title(
        "(d) TFLite export (x86 CPU, fixed 1.8 s window): FP32 vs. INT8",
        fontsize=10.0,
        fontweight="bold",
    )
    ax.set_ylim(0, 10.9)
    ax.legend(fontsize=8.0, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(
        os.path.join(paper_fig_dir, "fig2_benchmarks.pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(repo_root, "resources", "benchmark_charts.png"),
        bbox_inches="tight",
        dpi=250,
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate the 2 FlaxSpeaker paper figures (fig1_architecture.pdf and "
            "fig2_benchmarks.pdf) and repository PNGs."
        )
    )
    parser.add_argument(
        "--repo_root", "--repo-root", type=str, default=REPO_ROOT
    )
    parser.add_argument(
        "--paper_fig_dir",
        "--paper-fig-dir",
        "--output-dir",
        "--output_dir",
        dest="paper_fig_dir",
        type=str,
        default=PAPER_FIG_DIR,
    )
    parser.add_argument("--benchmark-json", type=str, default=None)
    parser.add_argument("--ablation-json", type=str, default=None)
    parser.add_argument("--vox1o-json", type=str, default=None)
    args = parser.parse_args()
    draw_architecture_diagram(args.repo_root, args.paper_fig_dir)
    draw_benchmark_charts(
        args.repo_root, args.paper_fig_dir, benchmark_json=args.benchmark_json
    )
    print("Generated fig1_architecture.pdf and fig2_benchmarks.pdf")
