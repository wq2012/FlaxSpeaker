#!/usr/bin/env python3
"""Interactive Gradio 5.50.0 Demo for FlaxSpeaker Universal Speaker Verification."""

import os
import sys
import time
from typing import Any, Optional

import gradio as gr  # pytype: disable=import-error
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import hf_compat  # noqa: E402


PRETRAINED_ROOT = os.path.join(REPO_ROOT, "pretrained_models")
EXAMPLES_DIR = os.path.join(THIS_DIR, "examples")
_MODEL_CACHE: dict[str, hf_compat.FlaxSpeakerModel] = {}


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


def build_diagnostic_plot(
    enroll_feats: np.ndarray,
    test_feats: np.ndarray,
    enroll_emb: np.ndarray,
    test_emb: np.ndarray,
    score: float,
    model_name: str,
) -> Any:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")

    axes[0].imshow(
        enroll_feats[:250].T,
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    axes[0].set_title(
        f"Enrollment Log-Mel + VAD ({enroll_feats.shape[0]} frames)",
        fontsize=10,
    )
    axes[0].set_xlabel("Frame (10 ms stride)")
    axes[0].set_ylabel("Mel Bin (80)")

    axes[1].imshow(
        test_feats[:250].T,
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    axes[1].set_title(
        f"Test Log-Mel + VAD ({test_feats.shape[0]} frames)",
        fontsize=10,
    )
    axes[1].set_xlabel("Frame (10 ms stride)")
    axes[1].set_ylabel("Mel Bin (80)")

    dims = min(64, enroll_emb.shape[0])
    x = np.arange(dims)
    axes[2].plot(
        x, enroll_emb[:dims], label="Enrollment Emb", color="#2563EB", lw=1.4
    )
    axes[2].plot(
        x,
        test_emb[:dims],
        label="Test Emb",
        color="#DC2626",
        lw=1.4,
        alpha=0.85,
    )
    axes[2].set_title(
        f"{model_name} (first {dims} dims, score={score:.4f})", fontsize=10
    )
    axes[2].set_xlabel("Embedding Dimension")
    axes[2].set_ylabel("Activation")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


def verify_pair(
    enroll_audio: Optional[str],
    test_audio: Optional[str],
    model_name: str,
    threshold: float,
) -> tuple[str, dict[str, Any], Any]:
    if not enroll_audio or not test_audio:
        return "Please provide both Enrollment and Test audio files.", {}, None

    t0 = time.time()
    model = get_model(model_name)
    enroll_feats = model.frontend.extract_from_file(enroll_audio)
    test_feats = model.frontend.extract_from_file(test_audio)
    enroll_emb = model.extract_embedding(enroll_audio)
    test_emb = model.extract_embedding(test_audio)
    score = model.verify(enroll_audio, test_audio)
    elapsed_ms = (time.time() - t0) * 1000.0

    is_same = score >= threshold
    verdict = (
        f"SAME SPEAKER (Score = {score:.4f} >= Threshold {threshold:.2f})"
        if is_same
        else f"DIFFERENT SPEAKER (Score = {score:.4f} < Threshold {threshold:.2f})"
    )

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
        "similarity_score": round(float(score), 4),
        "threshold": round(float(threshold), 4),
        "decision": "same_speaker" if is_same else "different_speaker",
        "enroll_vad_frames": int(enroll_feats.shape[0]),
        "test_vad_frames": int(test_feats.shape[0]),
        "embedding_dim": int(enroll_emb.shape[0]),
        "inference_latency_ms": round(elapsed_ms, 2),
    }
    fig = build_diagnostic_plot(
        enroll_feats, test_feats, enroll_emb, test_emb, score, model_name
    )
    return verdict, details, fig


def build_demo() -> gr.Blocks:
    models = list_available_models()
    default_model = (
        "univ_conformer_small_ge2e"
        if "univ_conformer_small_ge2e" in models
        else models[0]
    )

    example_rows = []
    for dom, ext, label_same, label_diff in [
        (
            "librispeech",
            ".flac",
            "LibriSpeech Genuine (Same Speaker)",
            "LibriSpeech Impostor (Diff Speaker)",
        ),
        (
            "cnceleb2",
            ".flac",
            "CN-Celeb2 Genuine (Same Speaker)",
            "CN-Celeb2 Impostor (Diff Speaker)",
        ),
        (
            "voxceleb1",
            ".wav",
            "VoxCeleb1 Genuine (Same Speaker)",
            "VoxCeleb1 Impostor (Diff Speaker)",
        ),
    ]:
        p1 = os.path.join(EXAMPLES_DIR, f"{dom}_spk1_utt1{ext}")
        p2 = os.path.join(EXAMPLES_DIR, f"{dom}_spk1_utt2{ext}")
        p3 = os.path.join(EXAMPLES_DIR, f"{dom}_spk2_utt1{ext}")
        if os.path.exists(p1) and os.path.exists(p2):
            example_rows.append([p1, p2, default_model, 0.55])
        if os.path.exists(p1) and os.path.exists(p3):
            example_rows.append([p1, p3, default_model, 0.55])

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
                    label="Enrollment Utterance",
                )
                test_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Test Utterance",
                )
                model_dropdown = gr.Dropdown(
                    choices=models,
                    value=default_model,
                    label="Pretrained FlaxSpeaker Model",
                )
                threshold_slider = gr.Slider(
                    minimum=-0.2,
                    maximum=0.95,
                    value=0.55,
                    step=0.01,
                    label="Verification Decision Threshold",
                )
                verify_btn = gr.Button("Run Speaker Verification", variant="primary")
            with gr.Column(scale=1):
                verdict_box = gr.Textbox(label="Verification Decision", interactive=False)
                details_json = gr.JSON(label="Detailed Verification Telemetry")
        diag_plot = gr.Plot(label="Log-Mel Spectrogram (Post-VAD) & Speaker Embedding Profile")

        verify_btn.click(
            fn=verify_pair,
            inputs=[enroll_input, test_input, model_dropdown, threshold_slider],
            outputs=[verdict_box, details_json, diag_plot],
        )

        if example_rows:
            gr.Examples(
                examples=example_rows,
                inputs=[enroll_input, test_input, model_dropdown, threshold_slider],
                outputs=[verdict_box, details_json, diag_plot],
                fn=verify_pair,
                cache_examples=False,
                label="Curated Multi-Domain Examples (LibriSpeech EN, CN-Celeb2 ZH, VoxCeleb1 EN)",
            )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
