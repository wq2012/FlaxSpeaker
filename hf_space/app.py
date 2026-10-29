#!/usr/bin/env python3
"""Interactive Gradio demo for FlaxSpeaker speaker verification.

Every verdict is produced by *real* inference of the selected checkpoint on the
uploaded audio (feature extraction -> JAX encoder -> cosine / Dr-Vector / PFAS
scoring). Nothing is precomputed or cached across examples.

The folder is self-contained: `flaxspeaker` is installed from `requirements.txt`
(falling back to the parent checkout for local development) and checkpoints are
downloaded from the Hugging Face Hub when `pretrained_models/` is not present.
"""

import json
import os
import sys
import time
from typing import Any, Optional

try:
    import gradio as gr  # pytype: disable=import-error
except ImportError:  # pragma: no cover
    gr = None  # type: ignore
from matplotlib.figure import Figure
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
try:  # pip-installed `flaxspeaker` (Hugging Face Space)
    from flaxspeaker import evaluation
    from flaxspeaker import feature_extraction
    from flaxspeaker import hf_compat
    from flaxspeaker import vad
except ImportError:  # local development from the repository checkout
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from flaxspeaker import evaluation  # noqa: E402
    from flaxspeaker import feature_extraction  # noqa: E402
    from flaxspeaker import hf_compat  # noqa: E402
    from flaxspeaker import vad  # noqa: E402


PRETRAINED_ROOT = os.environ.get(
    "FLAXSPEAKER_PRETRAINED_ROOT",
    os.path.join(REPO_ROOT, "pretrained_models"),
)
EXAMPLES_DIR = os.path.join(THIS_DIR, "examples")
DEFAULT_MODEL = "univ_full_conformer_large_arcface"
FALLBACK_MODEL = "univ_conformer_small_arcface"
_MODEL_CACHE: dict[str, hf_compat.FlaxSpeakerModel] = {}
_THRESHOLD_CACHE: dict[str, dict[str, float]] = {}
_EVAL_SUMMARY_CACHE: dict[str, dict[str, float]] = {}
# Threshold slider ranges per scoring back-end (score scales differ: cosine is in
# [-1, 1]; Dr-Vector adds a residual to the cosine; PFAS is a normalised dot product
# of attention-weighted value vectors). Defaults come from each model's own pooled
# 1-utt EER operating point in `eval_results.json` (see `get_default_threshold`).
_SLIDER_RANGE = {
    "cosine": (-0.2, 1.0),
    "dr_vector": (-0.5, 1.5),
    "pfas": (-0.5, 1.5),
}


def list_available_models() -> list[str]:
    default_order = (
        list(hf_compat.FULL_DATA_FLAGSHIP_IDS)
        + [m for m in hf_compat.CONTROLLED_ABLATION_MODEL_IDS if m.startswith("univ_")]
        + [m for m in hf_compat.CONTROLLED_ABLATION_MODEL_IDS if not m.startswith("univ_")]
    )
    if not os.path.isdir(PRETRAINED_ROOT):
        return default_order
    models = sorted(
        d
        for d in os.listdir(PRETRAINED_ROOT)
        if hf_compat.FlaxSpeakerModel._has_local_weights(os.path.join(PRETRAINED_ROOT, d))
    )
    flagships = [m for m in models if m.startswith("univ_full_")]
    univ = [m for m in models if m.startswith("univ_") and not m.startswith("univ_full_")]
    others = [m for m in models if not m.startswith("univ_")]
    return (flagships + univ + others) or default_order


def _resolve_model_target(model_name: str) -> str:
    local_path = os.path.join(PRETRAINED_ROOT, model_name)
    if hf_compat.FlaxSpeakerModel._has_local_weights(local_path):
        return local_path
    return f"wq2012/{model_name}"


def get_model(model_name: str) -> hf_compat.FlaxSpeakerModel:
    """Loads (and caches) a checkpoint and warms up the bucketed JIT kernels.

    Inference goes through `evaluation.encode_features`, the masked power-of-two
    bucketed full-utterance evaluation path (`256, 512, 1024, 2048, 3000` frames).
    Warming up the primary duration buckets (`256, 512, 1024` frames) compiles the
    steady-state XLA kernels ahead of interactive requests.
    """
    if model_name not in _MODEL_CACHE:
        model = hf_compat.FlaxSpeakerModel.from_pretrained(_resolve_model_target(model_name))
        feat_dim = int(model.exp_config.frontend.feature_dim)
        warm_emb = evaluation.encode_features(
            np.zeros((64, feat_dim), dtype=np.float32), model.state, model.exp_config
        )
        for bucket_frames in (300, 600):
            evaluation.encode_features(
                np.zeros((bucket_frames, feat_dim), dtype=np.float32),
                model.state,
                model.exp_config,
            )
        # The Dr-Vector / PFAS scorers are JIT-compiled lazily and the PFAS kernel is
        # shape-specialised on the enrollment count; warm 1-, 2- and 3-utt enrollment.
        model.score_embeddings(warm_emb, warm_emb)
        model.score_embeddings([warm_emb, warm_emb], warm_emb)
        model.score_embeddings([warm_emb, warm_emb, warm_emb], warm_emb)
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def _load_eval_results_dict(model_name: str) -> dict[str, Any]:
    target = _resolve_model_target(model_name)
    path = os.path.join(target, "eval_results.json")
    if not os.path.exists(path):
        try:
            from huggingface_hub import hf_hub_download  # pytype: disable=import-error

            path = hf_hub_download(repo_id=target, filename="eval_results.json")
        except Exception:  # pragma: no cover - offline fallback
            path = ""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_default_threshold(model_name: str) -> dict[str, float]:
    """Per-model raw-score EER operating points read from `eval_results.json`."""
    if model_name in _THRESHOLD_CACHE:
        return _THRESHOLD_CACHE[model_name]
    out: dict[str, float] = {}
    metrics = _load_eval_results_dict(model_name)
    if metrics:
        d1 = metrics.get("multi_domain_eval", {}).get("domains_1utt", {})
        for key, name in (
            ("universal_pooled", "pooled"),
            ("librispeech", "librispeech"),
            ("cnceleb2", "cnceleb2"),
            ("voxceleb1", "voxceleb1"),
        ):
            if key in d1 and "eer_threshold" in d1[key]:
                out[name] = float(d1[key]["eer_threshold"])
        d3 = metrics.get("multi_domain_eval", {}).get("domains_3utt", {})
        if "universal_pooled" in d3 and "eer_threshold" in d3["universal_pooled"]:
            out["pooled_3utt"] = float(d3["universal_pooled"]["eer_threshold"])
        off = metrics.get("official_benchmarks", {})
        for key in ("voxceleb1_o", "cnceleb_e", "librispeech_test_clean"):
            if key in off and "eer_threshold" in off[key]:
                out[key] = float(off[key]["eer_threshold"])
    if "pooled" not in out:
        out["pooled"] = out.get("voxceleb1_o", 0.5)
    if "pooled_3utt" not in out:
        out["pooled_3utt"] = out["pooled"]
    _THRESHOLD_CACHE[model_name] = out
    return out


def get_model_eval_summary(model_name: str) -> dict[str, float]:
    """Reads measured EER percentages recorded in the checkpoint's `eval_results.json`."""
    if model_name in _EVAL_SUMMARY_CACHE:
        return _EVAL_SUMMARY_CACHE[model_name]
    summary: dict[str, float] = {}
    metrics = _load_eval_results_dict(model_name)
    if metrics:
        d1 = metrics.get("multi_domain_eval", {}).get("domains_1utt", {})
        if "universal_pooled" in d1 and "eer_percent" in d1["universal_pooled"]:
            summary["controlled_pooled_1utt_eer_pct"] = round(
                float(d1["universal_pooled"]["eer_percent"]), 2
            )
        d3 = metrics.get("multi_domain_eval", {}).get("domains_3utt", {})
        if "universal_pooled" in d3 and "eer_percent" in d3["universal_pooled"]:
            summary["controlled_pooled_3utt_eer_pct"] = round(
                float(d3["universal_pooled"]["eer_percent"]), 2
            )
        off = metrics.get("official_benchmarks", {})
        for key in ("voxceleb1_o", "cnceleb_e", "librispeech_test_clean"):
            if key in off and "eer_percent" in off[key]:
                best_block = off[key].get("qmf") or off[key].get("as_norm") or off[key]
                summary[f"official_{key}_eer_pct"] = round(
                    float(best_block["eer_percent"]), 2
                )
    _EVAL_SUMMARY_CACHE[model_name] = summary
    return summary


def threshold_slider_update(model_name: str) -> Any:
    """Gradio callback: reset the threshold slider to the selected model's operating point."""
    model = get_model(model_name)
    scoring_type_str = str(model.config.scoring_type).lower()
    lo, hi = _SLIDER_RANGE.get(scoring_type_str, (-0.5, 1.5))
    t = get_default_threshold(model_name)["pooled"]
    if gr is None:
        return t
    return gr.update(
        minimum=lo,
        maximum=hi,
        value=round(t, 3),
        label=(
            f"Manual Decision Threshold ({scoring_type_str} score; this model's pooled "
            f"1-utt EER operating point is {t:.3f})"
        ),
    )


def build_diagnostic_plot(
    enroll_wav: np.ndarray,
    enroll_vad_mask: np.ndarray,
    enroll_feats: np.ndarray,
    test_feats: np.ndarray,
    enroll_emb: np.ndarray,
    test_emb: np.ndarray,
    score: float,
    model_name: str,
    num_enroll: int = 1,
    sample_rate: int = 16000,
    scoring_type: str = "cosine",
) -> Figure:
    fig = Figure(figsize=(16.0, 3.6), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    axes = fig.subplots(1, 4)

    t_axis = np.arange(enroll_wav.shape[0]) / float(max(1, sample_rate))
    axes[0].plot(t_axis, enroll_wav, color="#334155", lw=0.7, alpha=0.85)
    if enroll_vad_mask.size > 0 and enroll_wav.size > 0:
        sample_mask = np.repeat(
            enroll_vad_mask.astype(np.float32),
            max(1, int(np.ceil(enroll_wav.shape[0] / enroll_vad_mask.shape[0]))),
        )[: enroll_wav.shape[0]]
        axes[0].fill_between(
            t_axis,
            -1.0,
            1.0,
            where=sample_mask > 0.5,
            color="#22C55E",
            alpha=0.22,
            label="Active VAD",
        )
    axes[0].set_title("Enrollment Waveform & VAD Gate", fontsize=9.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Normalized Amplitude")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].legend(loc="upper right", fontsize=7.5)

    axes[1].imshow(
        enroll_feats[:250].T,
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    axes[1].set_title(
        f"Enroll Log-Mel ({enroll_feats.shape[0]} active frames, {num_enroll}-utt)",
        fontsize=9.5,
    )
    axes[1].set_xlabel("Frame (10 ms stride)")
    axes[1].set_ylabel("Mel Bin (80)")

    axes[2].imshow(
        test_feats[:250].T,
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    axes[2].set_title(
        f"Test Log-Mel ({test_feats.shape[0]} active frames)",
        fontsize=9.5,
    )
    axes[2].set_xlabel("Frame (10 ms stride)")
    axes[2].set_ylabel("Mel Bin (80)")

    dims = min(64, enroll_emb.shape[0], test_emb.shape[0])
    x = np.arange(dims)
    is_pfas = scoring_type.lower() == "pfas"
    enroll_lbl = f"Enroll Keys (1..{dims})" if is_pfas else "Enrollment Emb"
    test_lbl = f"Test Keys (1..{dims})" if is_pfas else "Test Emb"
    axes[3].plot(
        x, enroll_emb[:dims], label=enroll_lbl, color="#2563EB", lw=1.3
    )
    axes[3].plot(
        x,
        test_emb[:dims],
        label=test_lbl,
        color="#DC2626",
        lw=1.3,
        alpha=0.85,
    )
    panel4_title = (
        f"PFAS Key-Value Rep (first {dims}/{enroll_emb.shape[0]} dims, score={score:.4f})"
        if is_pfas
        else f"{model_name} (first {dims}/{enroll_emb.shape[0]} dims, score={score:.4f})"
    )
    axes[3].set_title(panel4_title, fontsize=9.2)
    axes[3].set_xlabel(
        f"PFAS Key-Value Slice ({enroll_emb.shape[0]} total dims)"
        if is_pfas
        else "Embedding Dimension"
    )
    axes[3].set_ylabel("Activation")
    axes[3].legend(loc="upper right", fontsize=7.5)
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def verify_pair(
    enroll_audio: Optional[str],
    test_audio: Optional[str],
    model_name: str,
    threshold: Optional[float] = None,
    enroll_audio_2: Optional[str] = None,
    enroll_audio_3: Optional[str] = None,
    use_vad: bool = True,
    vad_snr_margin_db: float = 4.5,
    vad_top_db: float = 26.0,
    auto_threshold: bool = True,
) -> tuple[str, dict[str, Any], Any]:
    """Scores one trial.

    Threshold selection: with `auto_threshold=True` (default) the decision threshold
    is the checkpoint's own EER operating point for the *enrollment count actually
    used* -- `domains_1utt.universal_pooled.eer_threshold` for one enrollment
    utterance and `domains_3utt.universal_pooled.eer_threshold` for two or three
    (the multi-enrollment protocol of the benchmark). With `auto_threshold=False`
    the slider value `threshold` is used verbatim. The VAD knobs are the same two
    exposed by `scripts/inference.py` (`--vad_snr_margin_db`, `--vad_top_db`).
    """
    if not enroll_audio or not test_audio:
        return "Please provide both Enrollment and Test audio files.", {}, None

    t0 = time.time()
    model = get_model(model_name)
    enroll_paths = [
        p
        for p in (enroll_audio, enroll_audio_2, enroll_audio_3)
        if p and os.path.exists(p)
    ]
    if not enroll_paths:
        enroll_paths = [enroll_audio]
    operating_points = get_default_threshold(model_name)
    eval_summary = get_model_eval_summary(model_name)
    op_key = "pooled_3utt" if len(enroll_paths) >= 2 else "pooled"
    if threshold is None or auto_threshold:
        threshold = operating_points[op_key]
        threshold_source = (
            f"eval_results.json {op_key} EER operating point "
            f"({len(enroll_paths)} enrollment utterance(s))"
        )
    else:
        threshold_source = "manual slider"
    frontend = model.resolve_frontend(
        apply_vad=bool(use_vad),
        vad_top_db=float(vad_top_db),
        vad_snr_margin_db=float(vad_snr_margin_db),
    )
    vad_cfg = frontend.vad_processor.config if frontend.vad_processor else model.exp_config.vad
    vad_proc = vad.VadProcessor(vad_cfg)

    enroll_wav_1 = feature_extraction.load_waveform(
        enroll_paths[0], target_sr=frontend.config.sample_rate
    )
    norm_wav_1 = vad_proc.normalize_waveform(enroll_wav_1)
    enroll_vad_mask = vad_proc.detect_frames_from_waveform(
        norm_wav_1, pre_normalized=True
    )

    enroll_feats_list = [frontend.extract_from_file(p) for p in enroll_paths]
    test_feats = frontend.extract_from_file(test_audio)
    t_infer0 = time.time()
    enroll_embs = [
        evaluation.encode_features(feats, model.state, model.exp_config)
        for feats in enroll_feats_list
    ]
    test_emb = evaluation.encode_features(test_feats, model.state, model.exp_config)
    raw_score = float(
        model.score_embeddings(
            enroll_embs if len(enroll_embs) > 1 else enroll_embs[0],
            test_emb,
        )
    )
    score = raw_score
    infer_ms = (time.time() - t_infer0) * 1000.0
    scoring_type_str = str(model.config.scoring_type).lower()
    elapsed_ms = (time.time() - t0) * 1000.0

    is_same = score >= threshold
    verdict = (
        f"SAME SPEAKER (Score = {score:.4f} >= Threshold {threshold:.3f})"
        if is_same
        else f"DIFFERENT SPEAKER (Score = {score:.4f} < Threshold {threshold:.3f})"
    )

    display_enroll_emb = np.mean(np.stack(enroll_embs, axis=0), axis=0)
    tier_label = (
        "Tier-2 Full-Data Universal Flagship (`universal_full`)"
        if model_name.startswith("univ_full_")
        else "Tier-1 Controlled Benchmark Model (381-speaker ablation suite)"
    )
    details = {
        "model_id": model_name,
        "model_tier": tier_label,
        "hf_repo_id": f"wq2012/{model_name}",
        "hf_model_url": f"https://huggingface.co/wq2012/{model_name}",
        "hf_collection_url": hf_compat.HF_COLLECTION_URL,
        "backbone": model.config.backbone,
        "size_variant": model.config.size_variant,
        "pooling_type": model.config.pooling_type,
        "loss_type": model.config.loss_type,
        "scoring_type": model.config.scoring_type,
        "num_enrollment_utterances": len(enroll_embs),
        "vad_enabled": bool(use_vad),
        "vad_snr_margin_db": round(float(vad_snr_margin_db), 2),
        "vad_top_db": round(float(vad_top_db), 2),
        "similarity_score": round(float(score), 4),
        "threshold": round(float(threshold), 4),
        "threshold_source": threshold_source,
        "model_eer_operating_points": {k: round(v, 4) for k, v in operating_points.items()},
        "measured_benchmark_eers_percent": eval_summary,
        "decision": "same_speaker" if is_same else "different_speaker",
        "enroll_vad_frames": int(sum(f.shape[0] for f in enroll_feats_list)),
        "test_vad_frames": int(test_feats.shape[0]),
        "embedding_dim": int(display_enroll_emb.shape[0]),
        "model_inference_latency_ms": round(infer_ms, 2),
        "end_to_end_latency_ms": round(elapsed_ms, 2),
        "note": (
            "Steady-state latency: inference uses masked power-of-two bucketed full-utterance "
            "evaluation (up to 30 s per utterance). Check `measured_benchmark_eers_percent` "
            "and the linked Hugging Face model card for exact evaluation metrics."
        ),
    }
    fig = build_diagnostic_plot(
        norm_wav_1,
        enroll_vad_mask,
        enroll_feats_list[0],
        test_feats,
        display_enroll_emb,
        test_emb,
        score,
        model_name,
        num_enroll=len(enroll_embs),
        sample_rate=frontend.config.sample_rate,
        scoring_type=scoring_type_str,
    )
    return verdict, details, fig


def build_demo() -> Any:
    if gr is None:
        raise ImportError("gradio is required to build the web UI (`pip install flaxspeaker[demo]`).")
    models = list_available_models()
    if DEFAULT_MODEL in models:
        default_model = DEFAULT_MODEL
    elif FALLBACK_MODEL in models:
        default_model = FALLBACK_MODEL
    else:
        default_model = models[0]
    default_threshold = round(get_default_threshold(default_model)["pooled"], 3)

    example_rows = []
    for dom, ext in [
        ("librispeech", ".flac"),
        ("cnceleb2", ".flac"),
        ("voxceleb1", ".wav"),
    ]:
        p1 = os.path.join(EXAMPLES_DIR, f"{dom}_spk1_utt1{ext}")
        p2 = os.path.join(EXAMPLES_DIR, f"{dom}_spk1_utt2{ext}")
        p3 = os.path.join(EXAMPLES_DIR, f"{dom}_spk2_utt1{ext}")
        if os.path.exists(p1) and os.path.exists(p2):
            example_rows.append(
                [p1, p2, default_model, default_threshold, None, None, True, 4.5, 26.0, True]
            )
        if os.path.exists(p1) and os.path.exists(p3):
            example_rows.append(
                [p1, p3, default_model, default_threshold, None, None, True, 4.5, 26.0, True]
            )

    with gr.Blocks(title="FlaxSpeaker — Universal Speaker Recognition") as demo:
        gr.Markdown(
            "# FlaxSpeaker (`v0.5.0`) — Universal Multi-Domain Speaker Verification\n"
            "> **Disclaimer**: **This is not an officially supported Google product.** "
            "FlaxSpeaker is a from-scratch JAX/Flax re-implementation (using only the published "
            "papers, no proprietary code) of "
            "[GE2E (`arXiv:1710.10467`)](https://arxiv.org/abs/1710.10467), "
            "[Dr-Vectors (`arXiv:2104.01989`)](https://arxiv.org/abs/2104.01989), "
            "[ATP pooling (`arXiv:2202.12163`)](https://arxiv.org/abs/2202.12163), and "
            "[PFAS (`arXiv:2203.05642`)](https://arxiv.org/abs/2203.05642), together with "
            "Sub-Center ArcFace (`K=3`), ECAPA-TDNN, Conformer, and Mamba encoders.\n\n"
            "> **Two-Tier Model Zoo (`48` Pretrained Models)**:\n"
            "> 1. **Tier 2 — Full-Data Universal Flagships (`univ_full_*`, 4 models)**: "
            "`Conformer-Large` (`6.98M`) and `ECAPA-TDNN-c1024` (`14.92M`) trained on the full "
            "speaker-disjoint multi-corpus mixture (`VoxCeleb1/2-dev` + `CN-Celeb 1/2` + "
            "`LibriSpeech 960h`: `12,193` base speakers, `464,557` clean utterances / "
            "`1,857,149` augmented items with `MUSAN` + `RIRS_NOISES` + `3x` speed perturbation "
            "and `3.6s` Large-Margin Fine-Tuning), plus warm-started `PFAS` (`7.31M`) and "
            "`Dr-Vectors` (`7.15M`) heads.\n"
            "> 2. **Tier 1 — Controlled Multi-Domain Benchmark (`univ_*`, `ls_*`, `cn_*`, `vox_*`, "
            "44 models)**: trained on fixed bounded subsets (`<= 381` speakers, `<= 9,447` clean "
            "utterances) for controlled architectural, pooling, and loss ablations.\n\n"
            "Every verdict below runs real full-utterance JAX inference on your uploaded audio; "
            "the default decision threshold is read directly from the selected checkpoint's "
            "`eval_results.json` operating point (`1-utt` for single enrollment, `3-utt` for "
            "multi-utterance enrollment).\n\n"
            "**Links**: "
            "[Hugging Face Collection (48 Pretrained Models)]"
            "(https://huggingface.co/collections/wq2012/flaxspeaker) | "
            "[GitHub Repository (`wq2012/FlaxSpeaker`)](https://github.com/wq2012/FlaxSpeaker) | "
            "[PyPI (`flaxspeaker`)](https://pypi.org/project/flaxspeaker/)"
        )
        with gr.Row():
            with gr.Column(scale=1):
                enroll_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Enrollment Utterance 1 (Required)",
                )
                with gr.Accordion(
                    "Multi-Utterance Enrollment (Optional Utterances 2 & 3 for Centroid / PFAS)",
                    open=False,
                ):
                    enroll_input_2 = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Enrollment Utterance 2 (Optional)",
                    )
                    enroll_input_3 = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Enrollment Utterance 3 (Optional)",
                    )
                test_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Test Utterance",
                )
                model_dropdown = gr.Dropdown(
                    choices=models,
                    value=default_model,
                    label=f"Pretrained FlaxSpeaker Model ({len(models)} Checkpoints)",
                )
                auto_threshold_checkbox = gr.Checkbox(
                    value=True,
                    label=(
                        "Use the model's own EER operating point for the enrollment count "
                        "(1-utt / 3-utt; overrides the slider)"
                    ),
                )
                threshold_slider = gr.Slider(
                    minimum=-0.2,
                    maximum=1.0,
                    value=default_threshold,
                    step=0.005,
                    label=(
                        f"Manual Decision Threshold (cosine score; this model's pooled "
                        f"1-utt EER operating point is {default_threshold:.3f})"
                    ),
                )
                with gr.Row():
                    vad_checkbox = gr.Checkbox(
                        value=True, label="Enable frame-level energy VAD"
                    )
                vad_margin_slider = gr.Slider(
                    minimum=1.0,
                    maximum=12.0,
                    value=4.5,
                    step=0.5,
                    label="VAD SNR margin above adaptive noise floor (dB) [--vad_snr_margin_db]",
                )
                vad_top_db_slider = gr.Slider(
                    minimum=10.0,
                    maximum=60.0,
                    value=26.0,
                    step=1.0,
                    label="VAD dynamic-range threshold below peak energy (dB) [--vad_top_db]",
                )
                verify_btn = gr.Button("Run Speaker Verification", variant="primary")
            with gr.Column(scale=1):
                verdict_box = gr.Textbox(
                    label="Verification Decision", interactive=False
                )
                details_json = gr.JSON(label="Detailed Verification Telemetry")
        diag_plot = gr.Plot(
            label="Waveform + VAD Gate, Log-Mel Spectrograms (Post-VAD) & Speaker Embedding Profile"
        )

        all_inputs = [
            enroll_input,
            test_input,
            model_dropdown,
            threshold_slider,
            enroll_input_2,
            enroll_input_3,
            vad_checkbox,
            vad_margin_slider,
            vad_top_db_slider,
            auto_threshold_checkbox,
        ]
        model_dropdown.change(
            fn=threshold_slider_update, inputs=[model_dropdown], outputs=[threshold_slider]
        )
        verify_btn.click(
            fn=verify_pair,
            inputs=all_inputs,
            outputs=[verdict_box, details_json, diag_plot],
        )

        if example_rows:
            gr.Examples(
                examples=example_rows,
                inputs=all_inputs,
                outputs=[verdict_box, details_json, diag_plot],
                fn=verify_pair,
                cache_examples=False,
                label="Curated Multi-Domain Examples (LibriSpeech EN, CN-Celeb2 ZH, VoxCeleb1 EN)",
            )

    return demo


if __name__ == "__main__":
    import inspect

    app = build_demo()
    launch_kwargs: dict[str, Any] = {"ssr_mode": False}
    if gr is not None and "theme" in inspect.signature(app.launch).parameters:
        launch_kwargs["theme"] = gr.themes.Soft()
    app.launch(**launch_kwargs)
