#!/usr/bin/env python3
"""Export a FlaxSpeaker model to HuggingFace SafeTensors and TensorFlow Lite (FP32 / INT8).

All 44 pretrained models are available in the Hugging Face FlaxSpeaker Collection:
https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595

Usage:
    python scripts/export_model.py --model_dir wq2012/univ_conformer_small_ge2e \\
        --output_dir /tmp/exported_model --export_tflite --quantize_int8
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import export  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a FlaxSpeaker model to HuggingFace format and/or TFLite."
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
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory (defaults to --model_dir).",
    )
    parser.add_argument(
        "--export_tflite",
        action="store_true",
        help="Export FP32 .tflite flatbuffer.",
    )
    parser.add_argument(
        "--quantize_int8",
        action="store_true",
        help="Also export dynamic-range INT8 quantized .tflite flatbuffer.",
    )
    args = parser.parse_args()

    model = hf_compat.FlaxSpeakerModel.from_pretrained(args.model_dir)
    if args.output_dir:
        out_dir = args.output_dir
    elif os.path.isdir(args.model_dir):
        out_dir = args.model_dir
    else:
        short_id = os.path.basename(os.path.normpath(args.model_dir))
        local_pm = os.path.join("pretrained_models", short_id)
        out_dir = local_pm if os.path.isdir("pretrained_models") else short_id

    os.makedirs(out_dir, exist_ok=True)
    if (
        args.output_dir
        and os.path.abspath(out_dir) != os.path.abspath(args.model_dir)
    ) or not hf_compat.FlaxSpeakerModel._has_local_weights(out_dir):
        model.save_pretrained(out_dir)

    summary = {"model_dir": out_dir, "tflite_exports": {}}
    if args.export_tflite or args.quantize_int8:
        quant_modes = [False, True] if (args.export_tflite and args.quantize_int8) else [args.quantize_int8]
        rng = np.random.default_rng(42)
        sample_feat = rng.standard_normal(
            (model.exp_config.model.seq_len, model.exp_config.frontend.feature_dim)
        ).astype(np.float32)
        flax_emb = np.asarray(
            model.state.apply_fn({"params": model.state.params}, sample_feat[None, :, :])[0]
        )

        for q in quant_modes:
            suffix = "int8" if q else "fp32"
            tfl_path = os.path.join(out_dir, f"model_{suffix}.tflite")
            info = export.export_to_tflite(
                model.state, model.exp_config, tfl_path, quantize_int8=q
            )
            runner = export.TFLiteSpeakerRunner(tfl_path)
            t0 = time.time()
            tfl_emb = runner(sample_feat)
            latency_ms = (time.time() - t0) * 1000.0
            cos_sim = float(
                np.dot(tfl_emb, flax_emb)
                / (np.linalg.norm(tfl_emb) * np.linalg.norm(flax_emb) + 1e-6)
            )
            info["latency_ms"] = round(latency_ms, 3)
            info["cosine_fidelity_vs_flax"] = round(cos_sim, 6)
            summary["tflite_exports"][suffix] = info

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
