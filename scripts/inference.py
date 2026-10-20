#!/usr/bin/env python3
"""Run end-to-end speaker embedding extraction or speaker verification inference.

All 44 pretrained models are available in the Hugging Face FlaxSpeaker Collection:
https://huggingface.co/collections/wq2012/flaxspeaker

Usage:
    # Verify two audio files using a pretrained Hugging Face model ID or local directory:
    python scripts/inference.py --model_dir wq2012/univ_conformer_small_ge2e \\
        --enroll_audio path/to/enroll.flac --test_audio path/to/test.flac

    # Multi-utterance enrollment (comma-separated audio paths):
    python scripts/inference.py --model_dir wq2012/univ_conformer_small_ge2e \\
        --enroll_audio path/to/e1.flac,path/to/e2.flac,path/to/e3.flac \\
        --test_audio path/to/test.flac

    # Extract speaker embedding from a single audio file (Flax or TFLite):
    python scripts/inference.py --model_dir pretrained_models/univ_conformer_small_ge2e \\
        --audio path/to/utterance.flac
"""

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import export  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run speaker embedding extraction or speaker verification on audio files."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help=(
            "Hugging Face Hub model ID (e.g. wq2012/univ_conformer_small_ge2e) or local path "
            "to a pretrained FlaxSpeaker directory."
        ),
    )
    parser.add_argument(
        "--tflite_path",
        type=str,
        default="",
        help="Optional path to a .tflite model file for edge inference.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="",
        help="Path to a single audio file for speaker embedding extraction.",
    )
    parser.add_argument(
        "--enroll_audio",
        type=str,
        default="",
        help="Path (or comma-separated paths) to enrollment audio file(s).",
    )
    parser.add_argument(
        "--test_audio",
        type=str,
        default="",
        help="Path to test audio file for speaker verification.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Verification decision threshold. Default: the checkpoint's own pooled "
            "1-utt EER operating point from eval_results.json (falls back to 0.5 if "
            "the checkpoint ships no metrics)."
        ),
    )
    parser.add_argument(
        "--apply_vad",
        dest="apply_vad",
        action="store_true",
        default=None,
        help="Explicitly enable frame-level energy VAD (default: use checkpoint config).",
    )
    parser.add_argument(
        "--no_vad",
        dest="apply_vad",
        action="store_false",
        help="Disable frame-level energy VAD during feature extraction.",
    )
    parser.add_argument(
        "--vad_top_db",
        type=float,
        default=None,
        help="Override VAD dynamic-range threshold in dB (default: checkpoint's 26.0 dB).",
    )
    parser.add_argument(
        "--full_sequence",
        dest="full_sequence",
        action="store_true",
        default=None,
        help=(
            "Use single-pass masked full-sequence inference (<= max_eval_seq_len frames; "
            "default for released checkpoints)."
        ),
    )
    parser.add_argument(
        "--sliding_window",
        dest="full_sequence",
        action="store_false",
        help="Use sliding-window / multi-window inference instead of full-sequence inference.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model = hf_compat.FlaxSpeakerModel.from_pretrained(args.model_dir)
    tfl_runner = export.TFLiteSpeakerRunner(args.tflite_path) if args.tflite_path else None

    def _extract_one(audio_path: str) -> np.ndarray:
        if tfl_runner is not None:
            fe = model._resolve_frontend(
                apply_vad=args.apply_vad, vad_top_db=args.vad_top_db
            )
            feats = fe.extract_from_file(audio_path)
            return tfl_runner(feats)
        return model.extract_embedding(
            audio_path,
            full_sequence=args.full_sequence,
            apply_vad=args.apply_vad,
            vad_top_db=args.vad_top_db,
        )

    short_id = os.path.basename(os.path.normpath(args.model_dir))
    output: dict[str, Any] = {
        "model_dir": args.model_dir,
        "backbone": model.config.backbone,
        "size_variant": model.config.size_variant,
        "scoring_type": model.config.scoring_type,
        "backend": "tflite" if tfl_runner is not None else "jax_flax",
    }
    if short_id in hf_compat.PRETRAINED_MODEL_IDS:
        output["hf_repo_id"] = f"{hf_compat.HF_DEFAULT_NAMESPACE}/{short_id}"
        output["hf_model_url"] = (
            f"https://huggingface.co/{hf_compat.HF_DEFAULT_NAMESPACE}/{short_id}"
        )
        output["hf_collection_url"] = hf_compat.HF_COLLECTION_URL

    if args.audio:
        t0 = time.time()
        emb = _extract_one(args.audio)
        elapsed_ms = (time.time() - t0) * 1000.0
        output["audio"] = args.audio
        output["embedding_dim"] = int(emb.shape[0])
        output["embedding_l2_norm"] = round(float(np.linalg.norm(emb)), 6)
        output["embedding_preview_first8"] = [round(float(x), 5) for x in emb[:8]]
        output["inference_time_ms"] = round(elapsed_ms, 2)

    if args.enroll_audio and args.test_audio:
        if args.threshold is not None:
            threshold = float(args.threshold)
            threshold_source = "--threshold"
        else:
            eer_thr = model.eer_threshold()
            threshold = eer_thr if eer_thr is not None else 0.5
            threshold_source = (
                "eval_results.json pooled 1-utt EER operating point"
                if eer_thr is not None
                else "fallback 0.5 (no eval_results.json)"
            )
        t0 = time.time()
        enroll_paths = [p.strip() for p in args.enroll_audio.split(",") if p.strip()]
        if tfl_runner is None:
            score = model.verify(
                enroll_paths[0] if len(enroll_paths) == 1 else enroll_paths,
                args.test_audio,
                full_sequence=args.full_sequence,
                apply_vad=args.apply_vad,
                vad_top_db=args.vad_top_db,
            )
        else:
            e_embs = [_extract_one(p) for p in enroll_paths]
            t_emb = _extract_one(args.test_audio)
            score = model.score_embeddings(e_embs, t_emb)
        elapsed_ms = (time.time() - t0) * 1000.0
        output["enroll_audio"] = enroll_paths
        output["test_audio"] = args.test_audio
        output["num_enrollment_utterances"] = len(enroll_paths)
        output["similarity_score"] = round(float(score), 4)
        output["threshold"] = round(float(threshold), 4)
        output["threshold_source"] = threshold_source
        output["same_speaker_decision"] = bool(score >= threshold)
        output["inference_time_ms"] = round(elapsed_ms, 2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
