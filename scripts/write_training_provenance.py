#!/usr/bin/env python3
"""Write the *actual* training-crop provenance of the released checkpoints.

Reads ``pretrained_models/step0_fingerprints.json`` (produced by
``scripts/fingerprint_released_checkpoints.py``) and, for every released checkpoint,

1. adds a top-level ``training_provenance`` block to
   ``pretrained_models/<id>/config.json`` (round-tripped by
   ``FlaxSpeakerConfig.from_pretrained`` / ``save_pretrained``);
2. sets ``experiment_dict.model.variable_length_training`` in that ``config.json`` and
   ``model.variable_length_training`` in ``configs/<id>.yml`` to the value the released
   weights were actually trained with, so that re-training the shipped recipe
   reproduces the archived step-0 loss (the framework *default* for new experiments
   remains ``True``, i.e. ``T ~ U(min_seq_len, seq_len)`` crops).

Decision rule (``match`` from the fingerprint JSON):

* ``fixed`` / ``fixed_approx``  -> ``variable_length_training_actual = False``, ``crop_frames = 180``;
* ``variable`` / ``variable_approx`` -> ``True``, ``crop_frames = "90-180"``;
* ``both`` (not discriminable at 4 decimals) -> decided from git history: the two
  ``*_ge2e_contrast`` checkpoints were retrained at ``9accf95`` with the
  variable-length default (``True``); every other checkpoint predates ``9accf95``
  and was produced by the ``variable_length=False`` batcher default (``False``).
  The block records ``fingerprint_discriminable: false`` in that case;
* ``neither`` -> ``"unknown"``; the YAML / ``config.json`` recipe is left unchanged.

Usage:
    python scripts/write_training_provenance.py            # write
    python scripts/write_training_provenance.py --check    # verify only (used by tests)
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import configs  # noqa: E402

PM = os.path.join(REPO_ROOT, "pretrained_models")
CONFIGS = os.path.join(REPO_ROOT, "configs")
FINGERPRINTS = os.path.join(PM, "step0_fingerprints.json")

TRAINING_CODE_FIXED = (
    "uncommitted working tree before 1cb90c7 (2026-10-15): get_batched_nm_input / "
    "get_batched_classification_input defaulted to variable_length=False and "
    "run_research_benchmarks.py did not pass the flag, so crops were fixed "
    "seq_len=180 frames with all-ones masks although the serialized config said "
    "variable_length_training=True"
)
TRAINING_CODE_VARIABLE = (
    "9accf95 (2026-10-18) or later: variable_length_training honoured by the "
    "benchmark driver (T ~ U(min_seq_len, seq_len) crops with an exact validity mask)"
)
TRAINING_CODE_UNKNOWN = (
    "uncommitted working tree before 1cb90c7; the triplet branch passed "
    "variable_length=True explicitly but drew triplets from the unseeded global "
    "`random` module, so the step-0 loss cannot be fingerprinted"
)


def provenance_for(mid: str, fp: dict) -> tuple[dict, object]:
    """Returns (training_provenance block, actual variable_length_training or 'unknown')."""
    match = fp["match"]
    if match in ("fixed", "fixed_approx"):
        actual: object = False
        crop: object = int(fp["seq_len"])
        code = TRAINING_CODE_FIXED
    elif match in ("variable", "variable_approx"):
        actual = True
        crop = f"{fp['min_seq_len']}-{fp['seq_len']}"
        code = TRAINING_CODE_VARIABLE
    elif match == "both":
        if mid.endswith("_ge2e_contrast") or bool(
            fp.get("yaml_variable_length_training", True)
        ):
            actual = True
            crop = f"{fp['min_seq_len']}-{fp['seq_len']}"
            code = TRAINING_CODE_VARIABLE
        else:
            actual = False
            crop = int(fp["seq_len"])
            code = TRAINING_CODE_FIXED
    else:
        actual = "unknown"
        crop = "unknown"
        code = TRAINING_CODE_UNKNOWN
    block = {
        "variable_length_training_actual": actual,
        "crop_frames": crop,
        "training_code": code,
        "step0_fingerprint_match": match,
        "released_initial_loss": fp["released_initial_loss"],
        "head_fixed_step0_loss": fp["head_fixed"],
        "head_variable_step0_loss": fp["head_variable"],
        "fingerprint_code_commit": fp.get("code_commit"),
        "fingerprint_source": "pretrained_models/step0_fingerprints.json",
        "note": (
            "How the released weights were actually trained. Evaluating the released "
            "weights with the current code reproduces the archived metrics; re-training "
            "the shipped recipe reproduces the archived step-0 loss only under "
            "'crop_frames'. See README.md 'Training-crop provenance'."
        ),
    }
    if match == "both":
        block["fingerprint_discriminable"] = False
        block["fingerprint_note"] = (
            "Both crop regimes reproduce the archived 4-decimal step-0 loss "
            f"(fixed {fp['head_fixed']}, variable {fp['head_variable']}); the crop regime "
            "is assigned from the git history, not from the fingerprint."
        )
    if fp.get("legacy_lambda_used"):
        block["sphereface_note"] = (
            "Released weights were trained with a constant SphereFace annealing "
            "lambda = sphereface_lambda_min (pre-9accf95 behaviour); the current "
            "schedule starts at sphereface_lambda_base. The fingerprint used the "
            "constant-lambda arms (head_*_legacy_lambda in step0_fingerprints.json)."
        )
        block["head_fixed_step0_loss_legacy_lambda"] = fp["head_fixed_legacy_lambda"]
        block["head_variable_step0_loss_legacy_lambda"] = fp["head_variable_legacy_lambda"]
    return block, actual


def expected_blocks() -> dict[str, tuple[dict, object]]:
    with open(FINGERPRINTS, "r", encoding="utf-8") as f:
        fps = json.load(f)
    out = {}
    for mid, fp in fps["checkpoints"].items():
        fp = dict(fp)
        fp["code_commit"] = fps.get("code_commit")
        out[mid] = provenance_for(mid, fp)
    return out


def main(check_only: bool) -> int:
    blocks = expected_blocks()
    problems = []
    for mid, (block, actual) in sorted(blocks.items()):
        cfg_path = os.path.join(PM, mid, "config.json")
        yml_path = os.path.join(CONFIGS, f"{mid}.yml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
        want_flag = cfg_json["experiment_dict"]["model"]["variable_length_training"]
        if actual != "unknown":
            want_flag = bool(actual)
        ok = (
            cfg_json.get("training_provenance") == block
            and cfg_json["experiment_dict"]["model"]["variable_length_training"] == want_flag
        )
        ycfg = configs.ExperimentConfig.from_yaml(yml_path)
        ok = ok and bool(ycfg.model.variable_length_training) == bool(want_flag)
        if ok:
            continue
        if check_only:
            problems.append(mid)
            continue
        cfg_json["training_provenance"] = block
        cfg_json["experiment_dict"]["model"]["variable_length_training"] = want_flag
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f, indent=2)
            f.write("\n")
        if bool(ycfg.model.variable_length_training) != bool(want_flag):
            ycfg.model.variable_length_training = bool(want_flag)
            ycfg.to_yaml(yml_path)
        print(f"updated {mid}: variable_length_training={want_flag} ({block['step0_fingerprint_match']})")
    if check_only:
        if problems:
            print("provenance out of date for: " + ", ".join(problems))
            return 1
        print(f"provenance up to date for {len(blocks)} checkpoints")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify only; exit 1 if stale")
    args = parser.parse_args()
    sys.exit(main(args.check))
