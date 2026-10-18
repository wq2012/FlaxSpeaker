#!/usr/bin/env python3
"""Generate publication-quality vector PDF and PNG figures for the FlaxSpeaker paper and repository."""

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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PAPER_FIG_DIR = os.environ.get(
    "FLAXSPEAKER_PAPER_FIG_DIR",
    os.path.join(REPO_ROOT, "resources"),
)


def draw_architecture_diagram(
    repo_root: str = REPO_ROOT,
    paper_fig_dir: str = DEFAULT_PAPER_FIG_DIR,
) -> None:
    """Draws Figure 1: End-to-End FlaxSpeaker Universal Speaker Recognition Pipeline."""
    os.makedirs(paper_fig_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_root, "resources"), exist_ok=True)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(11.6, 3.85), dpi=300)
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 3.85)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    ax.text(
        5.8,
        3.62,
        "FlaxSpeaker: Universal Multi-Domain Speaker Recognition Architecture (JAX / Flax)",
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
        ax.text(
            x + w / 2,
            y + h - hdr_h - 0.17,
            shape_tag,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=hdr,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="#FFFFFF",
                edgecolor=border,
                lw=0.8,
            ),
        )
        top_y = y + h - hdr_h - 0.39
        step = (h - hdr_h - 0.45) / max(len(items), 1)
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

    y_top, h_top = 1.48, 1.92
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
            "Peak Norm (1.0) + 80-d Log-Mel",
            "Post-STFT VAD: max(-42 dB,",
            "  $P_{95}-26$ dB, $P_{10}+4.5$ dB)",
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
            "Mutually Exclusive Feature Aug:",
            "  35% MTR | 32.5% SpecAug",
            "  32.5% Clean; $T \\in [80, 200]$",
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
        "Frame States H in (B, T, D)",
        [
            "Conformer (Linear+Conv1D Stem,",
            "  GroupNorm G=2, MFA)",
            "Mamba SSM (associative_scan)",
            "Transformer (Pre-LN + MFA)",
            "ECAPA-TDNN | ResNet-18 | LSTM",
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
        "4. Pooling (x8) & Bottleneck",
        "Embedding e in (B, d_e) or (B, 640)",
        [
            "ATP ($N_h=4$ Sigmoid Subspace)",
            "ASP / SAP / Stats / Mean / Last",
            "Cumulative Stats (Streaming)",
            "PFAS Head: $K=4$ Keys + Values",
            "LN -> FC(256) -> LN -> ReLU -> L2",
        ],
        "#FAF5FF",
        "#A855F7",
        "#7E22CE",
    )

    add_arrow(2.65, y_top + 0.96, 2.95, y_top + 0.96)
    add_arrow(5.45, y_top + 0.96, 5.75, y_top + 0.96)
    add_arrow(8.40, y_top + 0.96, 8.70, y_top + 0.96)

    y_bot, h_bot = 0.08, 1.24
    add_stage(
        0.15,
        y_bot,
        6.80,
        h_bot,
        "5. Nine Training Losses & Three Verification Scoring Backends",
        "Batch (N=8, M=4) -> Scalar Loss L / Trial Pair (e_t, e_e) -> Score s",
        [
            "Losses: GE2E-Softmax & GE2E-Contrast | Extended-Set Softmax (0.5 Row + 0.5 Ext - log N, Eq. 4)",
            "  + Angular Margin & Metric Losses: ArcFace, CosFace, SphereFace, Softmax, Pairwise, Triplet",
            "Scoring: Cosine + AS-Norm | Dr-Vectors MLP (lambda_res=0.25, Eq. 5) | PFAS Concat L2 (Eq. 6-7)",
        ],
        "#FFF1F2",
        "#F43F5E",
        "#BE123C",
    )

    add_stage(
        7.35,
        y_bot,
        4.10,
        h_bot,
        "6. Edge Export & Open-Source Ecosystem",
        "44 HF Hub Models + TFLite FP32/INT8",
        [
            "HuggingFace Hub (wq2012/*): SafeTensors + AutoModel",
            "jax2tf -> TFLite INT8 (3.55x smaller, >=0.99994 cosine)",
            "Interactive Gradio Web App + Pytest CI Suite",
        ],
        "#ECFEFF",
        "#06B6D4",
        "#0E7490",
    )

    add_arrow(9.40, y_top, 5.80, y_bot + h_bot)
    add_arrow(10.20, y_top, 9.60, y_bot + h_bot)
    add_arrow(6.95, y_bot + 0.58, 7.35, y_bot + 0.58)

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
    paper_fig_dir: str = DEFAULT_PAPER_FIG_DIR,
) -> None:
    """Draws Figure 2: 4-Panel Empirical Benchmark Summary from benchmark_results.json."""
    os.makedirs(paper_fig_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_root, "resources"), exist_ok=True)
    with open(
        os.path.join(repo_root, "pretrained_models", "benchmark_results.json"),
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
    by_name = data["experiments"]

    plt.close("all")
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.2), dpi=300)
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

    ax.bar(x - 1.5 * w, ls_vals, w, label="LibriSpeech (EN)", color="#3B82F6")
    ax.bar(x - 0.5 * w, cn_vals, w, label="CN-Celeb2 (ZH)", color="#F59E0B")
    ax.bar(x + 0.5 * w, vx_vals, w, label="VoxCeleb1 (EN)", color="#10B981")
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
            b.get_height() + c_hi + 0.35,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in backbones], fontsize=8.8)
    ax.set_ylabel("1-Utterance EER (%) [Lower is Better]", fontsize=9.2)
    ax.set_title(
        "(a) Universal Multi-Domain Backbones (Small, GE2E-Softmax)",
        fontsize=10.0,
        fontweight="bold",
    )
    ax.set_ylim(0, 35.0)
    ax.legend(fontsize=7.8, ncol=2, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    # Panel (b): Universal Losses & Scoring (Conformer-Small)
    ax = axes[0, 1]
    losses = [
        ("GE2E\nCosine", "univ_conformer_small_ge2e"),
        ("GE2E-C\nCosine", "univ_conformer_small_ge2e_contrast"),
        ("ExtSoft\nDr-Vec", "univ_conformer_small_dr_vectors"),
        ("GE2E\nPFAS", "univ_conformer_small_pfas_ge2e"),
        ("ExtSoft\nPFAS", "univ_conformer_small_pfas"),
        ("ExtSoft\nCosine", "univ_conformer_small_ext_softmax"),
        ("Triplet\nCosine", "univ_conformer_small_triplet"),
        ("ArcFace\nCosine", "univ_conformer_small_arcface"),
    ]
    x2 = np.arange(len(losses))
    w2 = 0.34
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
        capsize=2.2,
        label="1-Utt Pooled EER (%) [95% CI]",
        color="#2563EB",
        error_kw={"lw": 0.9, "ecolor": "#1E3A8A"},
    )
    b3 = ax.bar(
        x2 + w2 / 2,
        e3,
        w2,
        yerr=[e3_lo, e3_hi],
        capsize=2.2,
        label="3-Utt Pooled EER (%) [95% CI]",
        color="#16A34A",
        error_kw={"lw": 0.9, "ecolor": "#14532D"},
    )
    for b, hi in zip(b1, e1_hi):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + hi + 0.3,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    for b, hi in zip(b3, e3_hi):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + hi + 0.3,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#15803D",
            fontweight="bold",
        )
    ax.set_xticks(x2)
    ax.set_xticklabels([item[0] for item in losses], fontsize=8.0)
    ax.set_ylabel("Universal Pooled EER (%)", fontsize=9.2)
    ax.set_title(
        "(b) Loss Functions & Scoring Paradigms (Conformer-Small)",
        fontsize=10.0,
        fontweight="bold",
    )
    ax.set_ylim(0, 36.0)
    ax.legend(fontsize=8.0, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    # Panel (c): Single-Domain vs Universal Cross-Domain Generalization
    ax = axes[1, 0]
    train_sets = [
        ("LibriSpeech\nOnly", "ls_conformer_small_ge2e"),
        ("CN-Celeb2\nOnly", "cn_conformer_small_ge2e"),
        ("VoxCeleb1\nOnly", "vox_conformer_small_ge2e"),
        ("Universal Mix\n(GE2E)", "univ_conformer_small_ge2e"),
        ("Universal Mix\n(Dr-Vectors)", "univ_conformer_small_dr_vectors"),
    ]
    x3 = np.arange(len(train_sets))
    w3 = 0.20
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

    ax.bar(
        x3 - 1.5 * w3, ls3, w3, label="Eval: LibriSpeech", color="#3B82F6"
    )
    ax.bar(x3 - 0.5 * w3, cn3, w3, label="Eval: CN-Celeb2", color="#F59E0B")
    ax.bar(x3 + 0.5 * w3, vx3, w3, label="Eval: VoxCeleb1", color="#10B981")
    bm = ax.bar(
        x3 + 1.5 * w3, mc3, w3, label="3-Domain Macro-Avg", color="#7C3AED"
    )
    for b in bm:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.35,
            f"{b.get_height():.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color="#5B21B6",
        )
    ax.set_xticks(x3)
    ax.set_xticklabels([t[0] for t in train_sets], fontsize=8.3)
    ax.set_ylabel("1-Utterance EER (%)", fontsize=9.2)
    ax.set_title(
        "(c) Cross-Domain Generalization: Single-Domain vs. Universal",
        fontsize=10.0,
        fontweight="bold",
    )
    ax.set_ylim(0, 34.5)
    ax.legend(fontsize=7.8, ncol=2, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25)

    # Panel (d): TFLite FP32 vs INT8 Size (decimal MB = 10^6 bytes) & Cosine Fidelity
    ax = axes[1, 1]
    tflite_models = [
        ("Univ Conformer-S", "univ_conformer_small_ge2e"),
        ("Univ ECAPA-S", "univ_ecapa_tdnn_small_ge2e"),
        ("Univ LSTM-S", "univ_lstm_small_ge2e"),
    ]
    t_by_name = data["tflite_exports"]
    x4 = np.arange(len(tflite_models))
    w4 = 0.32
    fp32_mb = [
        t_by_name[f"{k}_fp32"]["size_bytes"] / 1e6
        for _, k in tflite_models
    ]
    int8_mb = [
        t_by_name[f"{k}_int8"]["size_bytes"] / 1e6
        for _, k in tflite_models
    ]
    int8_cos = [
        t_by_name[f"{k}_int8"]["cosine_fidelity_vs_flax"]
        for _, k in tflite_models
    ]

    bf = ax.bar(
        x4 - w4 / 2, fp32_mb, w4, label="TFLite FP32 (MB)", color="#64748B"
    )
    bi = ax.bar(
        x4 + w4 / 2,
        int8_mb,
        w4,
        label="TFLite INT8 (MB, 3.5x smaller)",
        color="#0891B2",
    )
    for b in bf:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.18,
            f"{b.get_height():.2f} MB",
            ha="center",
            va="bottom",
            fontsize=7.6,
        )
    for idx, b in enumerate(bi):
        ax.text(
            b.get_x() + b.get_width() / 2 + 0.08,
            b.get_height() + 0.18,
            f"{b.get_height():.2f} MB\n(cos={int8_cos[idx]:.5f})",
            ha="center",
            va="bottom",
            fontsize=7.1,
            fontweight="bold",
            color="#0E7490",
        )
    ax.set_xticks(x4)
    ax.set_xticklabels([m[0] for m in tflite_models], fontsize=8.8)
    ax.set_ylabel("Model File Size (MB, $10^6$ bytes)", fontsize=9.2)
    ax.set_title(
        "(d) On-Device TFLite Export: FP32 vs. Dynamic-Range INT8",
        fontsize=10.0,
        fontweight="bold",
    )
    ax.set_ylim(0, 10.2)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=REPO_ROOT)
    parser.add_argument("--paper_fig_dir", type=str, default=DEFAULT_PAPER_FIG_DIR)
    args = parser.parse_args()
    draw_architecture_diagram(args.repo_root, args.paper_fig_dir)
    draw_benchmark_charts(args.repo_root, args.paper_fig_dir)
    print("Generated fig1_architecture.pdf and fig2_benchmarks.pdf")
