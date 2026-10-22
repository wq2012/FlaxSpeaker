#!/usr/bin/env python3
"""Fixed-crop control experiments (CPU-only, ~5 min per run on a 64-core workstation).

Background: the released checkpoint zoo (except the two ``*_ge2e_contrast`` models)
was trained with *fixed* 180-frame crops and all-ones validity masks, because the
batchers in the pre-``9accf95`` working tree defaulted to ``variable_length=False``
and the benchmark driver did not pass the flag. The current pipeline defaults to
``T ~ U(90, 180)`` crops. ``scripts/fingerprint_released_checkpoints.py`` establishes
this per checkpoint; this script provides the same-code, same-crop controls needed to
re-state the stratified-batching and reproducibility claims:

1. ``domain_mixing_fixed_crop``: ``univ_conformer_small_ge2e`` with
   ``within_domain_prob`` in {0.0, 1.0}, seed 42, fixed 180-frame crops.
   (The 0.75 arm is ``ablation_results.json -> variable_length_ablation ->
   variable_length_false``, seed 42, which this script does not duplicate.)
2. ``seed_noise_floor_fixed_crop``: ``univ_conformer_small_ge2e`` with seeds {0, 1, 2},
   fixed crops, giving the fixed-crop seed floor to compare with the variable-length
   floor in ``r2_seed_and_control_results.json``.
3. ``ls_fixed_crop_seed42``: ``ls_conformer_small_ge2e`` seed 42, fixed crops -- the
   end-to-end confirmation of the second step-0 fingerprint (released ``initial_loss``
   2.0965).

Outputs ``pretrained_models/r3_fixed_crop_controls.json`` (same per-run schema as
``r2_seed_and_control_results.json``). Nothing is estimated; every number is produced
by the training/evaluation code in this repository.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from scripts import run_r1_ablations as r1  # noqa: E402
from scripts import run_r2_controls as r2  # noqa: E402
from scripts import run_research_benchmarks as rrb  # noqa: E402

DEFAULT_DATA_ROOT = os.environ.get(
    "FLAXSPEAKER_DATA_ROOT", os.path.expanduser("~/Data/speaker_datasets")
)
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "pretrained_models")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:  # pragma: no cover - git not available
        return "unknown"


def main(data_root: str, output_root: str, output_json: str, seeds: list[int], skip: list[str]) -> None:
    if not output_json:
        output_json = os.path.join(output_root, "r3_fixed_crop_controls.json")
    summary_path = os.path.join(data_root, "splits", "dataset_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        ds_summary = json.load(f)

    def _cfg(dname: str, seed: int, within_prob: float = 0.75):
        info = ds_summary[dname]
        cfg = rrb.make_exp_config(
            dataset_name=dname,
            train_csv=info["train_csv"],
            eval_csv=info["eval_csv"],
            trials_file=info["trials_file"],
            backbone="conformer",
            size_variant="small",
            loss_type="ge2e_softmax",
            scoring_type="cosine",
            pooling_type="atp",
            num_steps=400,
        )
        cfg.model.variable_length_training = False  # fixed T = seq_len = 180 crops
        cfg.train.seed = int(seed)
        cfg.data.within_domain_prob = float(within_prob)
        return cfg

    sample_cfg = _cfg("universal", 42)
    frontend = feature_extraction.AudioFrontend(sample_cfg.frontend, sample_cfg.vad)
    caches: dict[str, feature_extraction.FeatureCache] = {}
    spk_train_sets: dict[str, dataset.SpkToUtts] = {}
    trials_sets: dict[str, list[dataset.TrialPair]] = {}
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]] = {}
    for dname in ("librispeech", "cnceleb2", "voxceleb1"):
        info = ds_summary[dname]
        spk_train_sets[dname] = dataset.get_csv_spk_to_utts(info["train_csv"])
        spk_eval = dataset.get_csv_spk_to_utts(info["eval_csv"])
        trials_sets[dname] = dataset.load_trials_file(info["trials_file"])
        multi_trials_sets[dname] = rrb.load_multi_enroll_trials(info["multi_trials_file"])
        cache = feature_extraction.FeatureCache(
            frontend, cache_dir=os.path.join(data_root, "feature_cache", dname)
        )
        cache.precompute_dataset(spk_eval, num_workers=16)
        cache.precompute_dataset(spk_train_sets[dname], num_workers=16)
        caches[dname] = cache
    univ_info = ds_summary["universal"]
    spk_train_sets["universal"] = dataset.get_csv_spk_to_utts(univ_info["train_csv"])
    trials_sets["universal"] = dataset.load_trials_file(univ_info["trials_file"])
    multi_trials_sets["universal"] = rrb.load_multi_enroll_trials(univ_info["multi_trials_file"])
    caches["universal"] = feature_extraction.FeatureCache.merged(
        frontend,
        [caches[d] for d in ("librispeech", "cnceleb2", "voxceleb1")],
        cache_dir=os.path.join(data_root, "feature_cache", "universal"),
    )

    results: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "description": (
            "Same-code controls with variable_length_training=False (fixed 180-frame "
            "crops, all-ones masks), the regime that produced the released zoo except "
            "the *_ge2e_contrast checkpoints. 400 steps, Conformer-Small, GE2E, ATP."
        ),
        "domain_mixing_fixed_crop": {},
        "seed_noise_floor_fixed_crop": {},
        "ls_fixed_crop_seed42": {},
    }
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            prev = json.load(f)
        for k in ("domain_mixing_fixed_crop", "seed_noise_floor_fixed_crop", "ls_fixed_crop_seed42"):
            results[k] = prev.get(k, {})

    def _save() -> None:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def _run(cfg, dname: str, tag: str) -> dict[str, Any]:
        print(f"\n=== {tag} (dataset={dname}, seed={cfg.train.seed}, "
              f"p_within={cfg.data.within_domain_prob}, fixed T={cfg.model.seq_len}) ===")
        t0 = time.time()
        res, _, _ = r1.run_single_config(
            cfg, spk_train_sets[dname], frontend, caches[dname], caches,
            trials_sets, multi_trials_sets,
        )
        res["seed"] = int(cfg.train.seed)
        res["variable_length_training"] = False
        res["crop_frames"] = int(cfg.model.seq_len)
        res["within_domain_prob"] = float(cfg.data.within_domain_prob)
        res["num_steps"] = int(cfg.train.num_steps)
        res["wall_time_sec"] = round(time.time() - t0, 2)
        res["summary"] = r2._summarise(res["multi_domain_eval"])
        return res

    # 1. Domain mixing at fixed crops (seed 42), p_within in {0.0, 1.0}
    if "mixing" not in skip:
        for p_val in (0.0, 1.0):
            key = f"p_within_{p_val}"
            if key in results["domain_mixing_fixed_crop"]:
                continue
            cfg = _cfg("universal", 42, within_prob=p_val)
            results["domain_mixing_fixed_crop"][key] = _run(
                cfg, "universal", f"Study 1 fixed-crop: univ_conformer_small_ge2e p_within={p_val}"
            )
            _save()

    # 2. Fixed-crop seed floor
    if "seeds" not in skip:
        entry = results["seed_noise_floor_fixed_crop"].setdefault(
            "univ_conformer_small_ge2e", {"per_seed": {}}
        )
        for seed in seeds:
            if str(seed) in entry["per_seed"]:
                continue
            cfg = _cfg("universal", seed)
            entry["per_seed"][str(seed)] = _run(
                cfg, "universal", f"Fixed-crop seed floor: univ_conformer_small_ge2e seed={seed}"
            )
            _save()
        rows = [v["summary"] for v in entry["per_seed"].values()]
        entry["mean_std_over_seeds"] = r2._mean_std(rows)
        entry["seeds"] = sorted(int(k) for k in entry["per_seed"].keys())
        _save()

    # 3. ls_conformer_small_ge2e seed 42 at fixed crops
    if "ls" not in skip and "ls_conformer_small_ge2e" not in results["ls_fixed_crop_seed42"]:
        cfg = _cfg("librispeech", 42)
        results["ls_fixed_crop_seed42"]["ls_conformer_small_ge2e"] = _run(
            cfg, "librispeech", "Fixed-crop seed-42 re-run: ls_conformer_small_ge2e"
        )
        _save()

    print(f"Saved fixed-crop control results to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed-crop (T=180) control experiments.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip", type=str, nargs="*", default=[], choices=["mixing", "seeds", "ls"])
    args = parser.parse_args()
    main(args.data_root, args.output_root, args.output_json, args.seeds, args.skip)
