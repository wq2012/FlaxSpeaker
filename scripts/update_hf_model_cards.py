#!/usr/bin/env python3
"""Regenerate and (optionally) push model cards + metadata for all 44 Hugging Face repos.

For every `pretrained_models/<model_id>/`:
  1. regenerates `README.md` from `config.json` + `eval_results.json` via
     `flaxspeaker.model_card.build_model_card` (no weights are touched);
  2. with `--push`, uploads `README.md`, `config.json`, `eval_results.json` to
     `wq2012/<model_id>` and deletes the stale `preprocessor_config.json` from the Hub.

Usage:
    python scripts/update_hf_model_cards.py            # regenerate locally only
    python scripts/update_hf_model_cards.py --push     # also push to the Hub
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
    parser.add_argument("--only", nargs="*", default=None, help="subset of model ids")
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
    from huggingface_hub import HfApi  # pytype: disable=import-error

    api = HfApi()
    for d in model_dirs:
        model_id = os.path.basename(d)
        repo_id = f"wq2012/{model_id}"
        ops = []
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete  # pytype: disable=import-error

        for fname in ("README.md", "config.json", "eval_results.json"):
            ops.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=os.path.join(d, fname)))
        existing = set(api.list_repo_files(repo_id))
        if "preprocessor_config.json" in existing:
            ops.append(CommitOperationDelete(path_in_repo="preprocessor_config.json"))
        api.create_commit(
            repo_id=repo_id,
            operations=ops,
            commit_message=(
                "Round 2: honest model card (training budget, variant labels, Vox1-O 10-spk subset, "
                "TFLite window, deviations), cleaned config.json, 26 dB Vox1-O subset metrics; "
                "remove misleading preprocessor_config.json"
            ),
        )
        print(f"pushed {repo_id}")


if __name__ == "__main__":
    main()
