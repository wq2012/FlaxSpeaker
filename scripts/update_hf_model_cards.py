#!/usr/bin/env python3
"""Regenerate and (optionally) push model cards, metadata, and weights for all 44 HF repos.

For every `pretrained_models/<model_id>/`:
  1. regenerates `README.md` from `config.json` + `eval_results.json` via
     `flaxspeaker.model_card.build_model_card` (local weights are untouched);
  2. with `--push`, uploads `README.md`, `config.json`, `eval_results.json` (plus
     `model.safetensors`, `flax_model.msgpack`, `model_fp32.tflite`, `model_int8.tflite`
     when `--include_weights` is passed) to `wq2012/<model_id>` and deletes any stale
     `preprocessor_config.json` from the Hub.

Usage:
    python scripts/update_hf_model_cards.py                           # regenerate locally only
    python scripts/update_hf_model_cards.py --push                    # push cards + metadata
    python scripts/update_hf_model_cards.py --push --include_weights  # push full release artifacts
"""

import argparse
import json
import os
import sys
import warnings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import configs  # noqa: E402
from flaxspeaker import model_card  # noqa: E402

PM = os.path.join(REPO_ROOT, "pretrained_models")

METADATA_FILES = ("README.md", "config.json", "eval_results.json")
WEIGHT_FILES = (
    "model.safetensors",
    "flax_model.msgpack",
    "model_fp32.tflite",
    "model_int8.tflite",
)


def regenerate(model_dir: str) -> str:
    model_id = os.path.basename(os.path.normpath(model_dir))
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg_json = json.load(f)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # config.json must be clean
        exp_cfg = configs.ExperimentConfig.from_dict(cfg_json["experiment_dict"])
    metrics = None
    eval_path = os.path.join(model_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    num_params = int(metrics["num_parameters"]) if metrics and "num_parameters" in metrics else 0
    text = model_card.build_model_card(model_id, exp_cfg, num_params, metrics, model_dir)
    with open(os.path.join(model_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return model_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--include_weights",
        action="store_true",
        help="Also upload model.safetensors, flax_model.msgpack, and *.tflite when present.",
    )
    parser.add_argument("--only", nargs="*", default=None, help="subset of model ids")
    parser.add_argument(
        "--commit_message",
        default=(
            "v0.4.0: full-convergence checkpoints (S_safe = X + 2 epochs), "
            "TFLite exports, and updated evaluation archives"
        ),
    )
    args = parser.parse_args()

    model_dirs = sorted(
        d for d in (os.path.join(PM, x) for x in os.listdir(PM))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json"))
    )
    if args.only:
        model_dirs = [d for d in model_dirs if os.path.basename(d) in set(args.only)]
    ids = [regenerate(d) for d in model_dirs]
    print(f"Regenerated {len(ids)} model cards")

    if not args.push:
        return
    from huggingface_hub import (  # pytype: disable=import-error
        CommitOperationAdd,
        CommitOperationDelete,
        HfApi,
    )

    api = HfApi()
    for idx, d in enumerate(model_dirs, 1):
        model_id = os.path.basename(d)
        repo_id = f"wq2012/{model_id}"
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        ops = []
        upload_candidates = list(METADATA_FILES)
        if args.include_weights:
            upload_candidates.extend(WEIGHT_FILES)
        uploaded_names = []
        for fname in upload_candidates:
            fpath = os.path.join(d, fname)
            if os.path.exists(fpath):
                ops.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=fpath))
                uploaded_names.append(fname)
        existing = set(api.list_repo_files(repo_id))
        if "preprocessor_config.json" in existing:
            ops.append(CommitOperationDelete(path_in_repo="preprocessor_config.json"))
        commit_info = api.create_commit(
            repo_id=repo_id,
            operations=ops,
            commit_message=args.commit_message,
        )
        print(
            f"[{idx}/{len(model_dirs)}] pushed {repo_id} "
            f"({len(uploaded_names)} files: {', '.join(uploaded_names)}) -> {commit_info.oid[:8]}"
        )


if __name__ == "__main__":
    main()
