#!/usr/bin/env python3
"""Evaluate a pretrained FlaxSpeaker model on 1-utt or multi-enrollment verification trials.

All 44 pretrained models are available in the Hugging Face FlaxSpeaker Collection:
https://huggingface.co/collections/wq2012/flaxspeaker-6ad02d999a4180c794870595

Usage:
    # Evaluate on a single trials TSV (using Hugging Face Hub ID or local directory):
    python scripts/evaluate.py --model_dir wq2012/univ_conformer_small_ge2e \\
        --trials_file /path/to/trials.tsv

    # Evaluate across all 3 evaluation domains (LibriSpeech, CN-Celeb2, VoxCeleb1):
    python scripts/evaluate.py --model_dir pretrained_models/univ_conformer_small_ge2e \\
        --data_root ~/Data/speaker_datasets
"""

import argparse
import json
import os
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import evaluation  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402


def _load_multi_enroll_trials(path: str) -> list[dataset.MultiEnrollTrial]:
    path = dataset.resolve_data_path(path)
    trials: list[dataset.MultiEnrollTrial] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            trials.append(
                (
                    int(parts[0]),
                    [dataset.resolve_data_path(x) for x in parts[1].split(",") if x],
                    dataset.resolve_data_path(parts[2]),
                )
            )
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained FlaxSpeaker model on speaker verification trials."
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
        "--trials_file",
        type=str,
        default="",
        help="Path to a 1-utterance verification trials TSV file.",
    )
    parser.add_argument(
        "--multi_trials_file",
        type=str,
        default="",
        help="Optional path to a multi-enrollment verification trials TSV file.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Optional data_root containing splits/dataset_summary.json for full multi-domain evaluation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to write JSON evaluation metrics.",
    )
    args = parser.parse_args()

    model = hf_compat.FlaxSpeakerModel.from_pretrained(args.model_dir)
    exp_cfg = model.exp_config
    frontend = model.frontend

    short_id = os.path.basename(os.path.normpath(args.model_dir))
    results: dict[str, Any] = {"model_dir": args.model_dir}
    if short_id in hf_compat.PRETRAINED_MODEL_IDS:
        results["hf_repo_id"] = f"{hf_compat.HF_DEFAULT_NAMESPACE}/{short_id}"
        results["hf_model_url"] = (
            f"https://huggingface.co/{hf_compat.HF_DEFAULT_NAMESPACE}/{short_id}"
        )
        results["hf_collection_url"] = hf_compat.HF_COLLECTION_URL

    if args.trials_file:
        trials = dataset.load_trials_file(args.trials_file)
        cache = feature_extraction.FeatureCache(frontend)
        results["trials_1utt"] = evaluation.evaluate_verification_trials(
            model.state, trials, exp_cfg, feature_cache=cache
        )

    if args.multi_trials_file:
        m_trials = _load_multi_enroll_trials(args.multi_trials_file)
        cache = feature_extraction.FeatureCache(frontend)
        results["trials_multi_enroll"] = evaluation.evaluate_multi_enroll_trials(
            model.state, m_trials, exp_cfg, feature_cache=cache
        )

    if args.data_root:
        summary_path = os.path.join(args.data_root, "splits", "dataset_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                ds_summary = json.load(f)
            domains_1utt: dict[str, Any] = {}
            domains_3utt: dict[str, Any] = {}
            for dname in ("librispeech", "cnceleb2", "voxceleb1", "universal"):
                if dname not in ds_summary:
                    continue
                info = ds_summary[dname]
                c_dir = os.path.join(args.data_root, "feature_cache", dname)
                cache = feature_extraction.FeatureCache(frontend, cache_dir=c_dir)
                t_1 = dataset.load_trials_file(info["trials_file"])
                domains_1utt[dname] = evaluation.evaluate_verification_trials(
                    model.state, t_1, exp_cfg, feature_cache=cache
                )
                if "multi_trials_file" in info and os.path.exists(info["multi_trials_file"]):
                    t_3 = _load_multi_enroll_trials(info["multi_trials_file"])
                    domains_3utt[dname] = evaluation.evaluate_multi_enroll_trials(
                        model.state, t_3, exp_cfg, feature_cache=cache
                    )
            results["multi_domain_1utt"] = domains_1utt
            results["multi_domain_3utt"] = domains_3utt

    print(json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
