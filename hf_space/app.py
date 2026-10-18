#!/usr/bin/env python3
"""Interactive Gradio 5.50.0 Demo for FlaxSpeaker Universal Speaker Verification."""

import dataclasses
import glob
import os
import sys
import time
from typing import Any, Optional

try:
    import gradio as gr  # pytype: disable=import-error
except ImportError:  # pragma: no cover
    gr = None  # type: ignore
import jax.numpy as jnp
from matplotlib.figure import Figure
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import evaluation  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402
from flaxspeaker import scoring  # noqa: E402
from flaxspeaker import vad  # noqa: E402


PRETRAINED_ROOT = os.path.join(REPO_ROOT, "pretrained_models")
EXAMPLES_DIR = os.path.join(THIS_DIR, "examples")
_MODEL_CACHE: dict[str, hf_compat.FlaxSpeakerModel] = {}
_COHORT_CACHE: dict[str, np.ndarray] = {}


def list_available_models() -> list[str]:
    if not os.path.isdir(PRETRAINED_ROOT):
        return list(hf_compat.PRETRAINED_MODEL_IDS)
    models = sorted(
        d
        for d in os.listdir(PRETRAINED_ROOT)
        if os.path.exists(os.path.join(PRETRAINED_ROOT, d, "config.json"))
    )
    univ = [m for m in models if m.startswith("univ_")]
    others = [m for m in models if not m.startswith("univ_")]
    return (univ + others) or list(hf_compat.PRETRAINED_MODEL_IDS)


def get_model(model_name: str) -> hf_compat.FlaxSpeakerModel:
    if model_name not in _MODEL_CACHE:
        local_path = os.path.join(PRETRAINED_ROOT, model_name)
        target = (
            local_path
            if hf_compat.FlaxSpeakerModel._has_local_weights(local_path)
            else f"wq2012/{model_name}"
        )
        _MODEL_CACHE[model_name] = hf_compat.FlaxSpeakerModel.from_pretrained(
            target
        )
    return _MODEL_CACHE[model_name]


def _get_example_cohort(
    model_name: str, model: hf_compat.FlaxSpeakerModel
) -> Optional[np.ndarray]:
    if model_name in _COHORT_CACHE:
        return _COHORT_CACHE[model_name]
    files = sorted(
        glob.glob(os.path.join(EXAMPLES_DIR, "*.flac"))
        + glob.glob(os.path.join(EXAMPLES_DIR, "*.wav"))
    )
    if len(files) < 2:
        return None
    embs = []
    for fpath in files:
        feats = model.frontend.extract_from_file(fpath)
        emb = np.asarray(
            evaluation.run_inference(
                jnp.asarray(feats, dtype=jnp.float32),
                model.state,
                model.exp_config,
            ),
            dtype=np.float32,
        )
        embs.append(emb)
    cohort = np.stack(embs, axis=0)
    _COHORT_CACHE[model_name] = cohort
    return cohort


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
    is_pfas = scoring_type.lower() == "pfas" or enroll_emb.shape[0] >= 640
    enroll_lbl = "Enroll Keys (1..64)" if is_pfas else "Enrollment Emb"
    test_lbl = "Test Keys (1..64)" if is_pfas else "Test Emb"
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
        f"PFAS Key-Value Rep (first {dims}/640 dims, score={score:.4f})"
        if is_pfas
        else f"{model_name} (first {dims} dims, score={score:.4f})"
    )
    axes[3].set_title(panel4_title, fontsize=9.2)
    axes[3].set_xlabel(
        "PFAS Key Slice (4x32 Keys + 4x128 Values)" if is_pfas else "Embedding Dimension"
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
    threshold: float = 0.55,
    enroll_audio_2: Optional[str] = None,
    enroll_audio_3: Optional[str] = None,
    use_vad: bool = True,
    vad_snr_margin_db: float = 4.5,
    use_as_norm: bool = False,
) -> tuple[str, dict[str, Any], Any]:
    if not enroll_audio or not test_audio:
        return "Please provide both Enrollment and Test audio files.", {}, None

    t0 = time.time()
    model = get_model(model_name)
    vad_cfg = dataclasses.replace(
        model.exp_config.vad,
        enabled=bool(use_vad),
        snr_margin_db=float(vad_snr_margin_db),
    )
    frontend = feature_extraction.AudioFrontend(
        model.exp_config.frontend, vad_cfg
    )
    vad_proc = vad.VadProcessor(vad_cfg)

    enroll_paths = [
        p
        for p in (enroll_audio, enroll_audio_2, enroll_audio_3)
        if p and os.path.exists(p)
    ]
    if not enroll_paths:
        enroll_paths = [enroll_audio]

    enroll_wav_1 = feature_extraction.load_waveform(
        enroll_paths[0], target_sr=frontend.config.sample_rate
    )
    norm_wav_1 = vad_proc.normalize_waveform(enroll_wav_1)
    enroll_vad_mask = vad_proc.detect_frames_from_waveform(
        norm_wav_1, pre_normalized=True
    )

    enroll_feats_list = [frontend.extract_from_file(p) for p in enroll_paths]
    test_feats = frontend.extract_from_file(test_audio)
    enroll_embs = [
        np.asarray(
            evaluation.run_inference(
                jnp.asarray(feats, dtype=jnp.float32),
                model.state,
                model.exp_config,
            ),
            dtype=np.float32,
        )
        for feats in enroll_feats_list
    ]
    test_emb = np.asarray(
        evaluation.run_inference(
            jnp.asarray(test_feats, dtype=jnp.float32),
            model.state,
            model.exp_config,
        ),
        dtype=np.float32,
    )
    raw_score = float(
        model.score_embeddings(
            enroll_embs if len(enroll_embs) > 1 else enroll_embs[0],
            test_emb,
        )
    )
    score = raw_score
    as_norm_applied = False
    as_norm_note = "disabled"
    scoring_type_str = str(model.config.scoring_type).lower()
    if use_as_norm:
        if scoring_type_str == "cosine":
            cohort = _get_example_cohort(model_name, model)
            if cohort is not None and cohort.shape[0] >= 2:
                e_centroid = np.mean(np.stack(enroll_embs, axis=0), axis=0)
                e_centroid = e_centroid / (np.linalg.norm(e_centroid) + 1e-6)
                norm_arr = scoring.apply_as_norm(
                    np.array([raw_score], dtype=np.float32),
                    e_centroid[None, :],
                    test_emb[None, :],
                    cohort,
                    top_k=min(6, cohort.shape[0]),
                )
                score = float(norm_arr[0])
                as_norm_applied = True
                as_norm_note = (
                    "AS-Norm z-score applied (score is in cohort standard-deviation units)"
                )
        else:
            as_norm_note = (
                f"skipped (AS-Norm applies to cosine similarity; model uses '{scoring_type_str}')"
            )

    elapsed_ms = (time.time() - t0) * 1000.0

    is_same = score >= threshold
    verdict = (
        f"SAME SPEAKER (Score = {score:.4f} >= Threshold {threshold:.2f})"
        if is_same
        else f"DIFFERENT SPEAKER (Score = {score:.4f} < Threshold {threshold:.2f})"
    )
    if as_norm_applied:
        verdict += f" [AS-Norm z-score; raw cosine = {raw_score:.4f}]"

    display_enroll_emb = np.mean(np.stack(enroll_embs, axis=0), axis=0)
    details = {
        "model_id": model_name,
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
        "as_norm_applied": as_norm_applied,
        "as_norm_status": as_norm_note,
        "raw_similarity_score": round(float(raw_score), 4),
        "similarity_score": round(float(score), 4),
        "threshold": round(float(threshold), 4),
        "decision": "same_speaker" if is_same else "different_speaker",
        "enroll_vad_frames": int(sum(f.shape[0] for f in enroll_feats_list)),
        "test_vad_frames": int(test_feats.shape[0]),
        "embedding_dim": int(display_enroll_emb.shape[0]),
        "inference_latency_ms": round(elapsed_ms, 2),
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
    default_model = (
        "univ_conformer_small_ge2e"
        if "univ_conformer_small_ge2e" in models
        else models[0]
    )

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
                [p1, p2, default_model, 0.55, None, None, True, 4.5, False]
            )
        if os.path.exists(p1) and os.path.exists(p3):
            example_rows.append(
                [p1, p3, default_model, 0.55, None, None, True, 4.5, False]
            )

    with gr.Blocks(title="FlaxSpeaker — Universal Speaker Recognition") as demo:
        gr.Markdown(
            "# FlaxSpeaker — Universal Multi-Domain Speaker Verification\n"
            "> **Disclaimer**: **This is not an officially supported Google product.** "
            "This is an independent open-source JAX/Flax reproduction of "
            "[GE2E (`arXiv:1710.10467`)](https://arxiv.org/abs/1710.10467), "
            "[Dr-Vectors (`arXiv:2104.01989`)](https://arxiv.org/abs/2104.01989), "
            "[Conformer-ATP (`arXiv:2202.12163`)](https://arxiv.org/abs/2202.12163), and "
            "[PFAS (`arXiv:2203.05642v3`)](https://arxiv.org/abs/2203.05642v3).\n\n"
            "**Links**: "
            "[Hugging Face Collection (44 Pretrained Models)]"
            "(https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595) | "
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
                    label="Pretrained FlaxSpeaker Model (44 Checkpoints)",
                )
                threshold_slider = gr.Slider(
                    minimum=-0.2,
                    maximum=0.95,
                    value=0.55,
                    step=0.01,
                    label="Verification Decision Threshold",
                )
                with gr.Row():
                    vad_checkbox = gr.Checkbox(
                        value=True, label="Enable Post-STFT VAD"
                    )
                    as_norm_checkbox = gr.Checkbox(
                        value=False, label="Apply AS-Norm Cohort Calibration"
                    )
                vad_margin_slider = gr.Slider(
                    minimum=1.0,
                    maximum=12.0,
                    value=4.5,
                    step=0.5,
                    label="VAD Dynamic SNR Margin (dB)",
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
            as_norm_checkbox,
        ]
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
