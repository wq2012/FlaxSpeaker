#!/usr/bin/env python3
"""Round-2 control experiments requested by the reviewers (cheap, CPU-only).

1. **Seed-noise floor**: `univ_conformer_small_ge2e` and `ls_conformer_small_ge2e`
   re-trained with seeds {0, 1, 2} (400 steps, identical configs otherwise). The seed
   controls both parameter initialisation and the data order. Reports per-seed metrics
   plus mean / std, so that sub-1% EER gaps in the paper can be qualified.
2. **Study 4 re-run**: an honest 800-step run of `univ_conformer_small_ge2e` with its
   *own* 800-step cosine schedule (the original "800 steps" row was a mid-schedule
   snapshot of a 1,200-step run).
3. **Matched-data control**: `univ_conformer_small_ge2e` trained on a random
   5,788-utterance subsample of the universal corpus (the size of the LibriSpeech-only
   training set), 400 steps, seed 42, to separate domain diversity from data quantity.

Outputs `pretrained_models/r2_seed_and_control_results.json`. Nothing is estimated; every
number is produced by the training/evaluation code in this repository.
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from scripts import run_r1_ablations as r1  # noqa: E402
from scripts import run_research_benchmarks as rrb  # noqa: E402

DEFAULT_DATA_ROOT = os.environ.get(
    "FLAXSPEAKER_DATA_ROOT", os.path.expanduser("~/Data/speaker_datasets")
)
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "pretrained_models")


def _summarise(md_eval: dict[str, Any]) -> dict[str, float]:
    d1 = md_eval["domains_1utt"]
    d3 = md_eval.get("domains_3utt", {})
    out = {
        "ls_1utt_eer": d1["librispeech"]["eer_percent"],
        "cn_1utt_eer": d1["cnceleb2"]["eer_percent"],
        "vox_1utt_eer": d1["voxceleb1"]["eer_percent"],
        "macro_1utt_eer": md_eval["universal_macro_avg_1utt"]["eer_percent"],
        "pooled_1utt_eer": d1["universal_pooled"]["eer_percent"],
    }
    if d3:
        out["macro_3utt_eer"] = md_eval["universal_macro_avg_3utt"]["eer_percent"]
        out["pooled_3utt_eer"] = d3["universal_pooled"]["eer_percent"]
    return out


def _mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = rows[0].keys()
    return {
        k: {
            "mean": round(float(np.mean([r[k] for r in rows])), 3),
            "std": round(float(np.std([r[k] for r in rows], ddof=1)), 3) if len(rows) > 1 else 0.0,
            "min": round(float(np.min([r[k] for r in rows])), 2),
            "max": round(float(np.max([r[k] for r in rows])), 2),
        }
        for k in keys
    }


def _subsample_spk_to_utts(
    spk_to_utts: dataset.SpkToUtts, num_utts: int, seed: int
) -> dataset.SpkToUtts:
    """Randomly keeps `num_utts` utterances (>= 2 per retained speaker) from a corpus."""
    rng = random.Random(seed)
    all_pairs = [(spk, u) for spk, utts in spk_to_utts.items() for u in utts]
    rng.shuffle(all_pairs)
    kept: dict[str, list[str]] = {}
    for spk, u in all_pairs[:num_utts]:
        kept.setdefault(spk, []).append(u)
    # GE2E needs >= 2 utterances per speaker; drop singletons.
    return {spk: utts for spk, utts in kept.items() if len(utts) >= 2}


def main(data_root: str, output_root: str, output_json: str, seeds: list[int], skip: list[str]) -> None:
    if not output_json:
        output_json = os.path.join(output_root, "r2_seed_and_control_results.json")
    summary_path = os.path.join(data_root, "splits", "dataset_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        ds_summary = json.load(f)

    sample_cfg = rrb.make_exp_config(
        dataset_name="universal",
        train_csv=ds_summary["universal"]["train_csv"],
        eval_csv=ds_summary["universal"]["eval_csv"],
        trials_file=ds_summary["universal"]["trials_file"],
        backbone="conformer",
        size_variant="small",
        loss_type="ge2e_softmax",
    )
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
    univ_cache = feature_extraction.FeatureCache(
        frontend, cache_dir=os.path.join(data_root, "feature_cache", "universal")
    )
    for dname in ("librispeech", "cnceleb2", "voxceleb1"):
        univ_cache._cache.update(caches[dname]._cache)
    caches["universal"] = univ_cache

    results: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed_noise_floor": {},
        "training_horizon_800_own_schedule": {},
        "matched_data_control": {},
    }
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            results.update(json.load(f))

    def _cfg(dname: str, num_steps: int = 400):
        info = ds_summary[dname]
        return rrb.make_exp_config(
            dataset_name=dname,
            train_csv=info["train_csv"],
            eval_csv=info["eval_csv"],
            trials_file=info["trials_file"],
            backbone="conformer",
            size_variant="small",
            loss_type="ge2e_softmax",
            scoring_type="cosine",
            pooling_type="atp",
            num_steps=num_steps,
        )

    def _save() -> None:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # 1. Seed-noise floor
    if "seeds" not in skip:
        for model_name, dname in (
            ("univ_conformer_small_ge2e", "universal"),
            ("ls_conformer_small_ge2e", "librispeech"),
        ):
            entry = results["seed_noise_floor"].setdefault(model_name, {"per_seed": {}})
            for seed in seeds:
                if str(seed) in entry["per_seed"]:
                    continue
                print(f"\n=== Seed floor: {model_name} seed={seed} ===")
                cfg = _cfg(dname)
                cfg.train.seed = int(seed)
                t0 = time.time()
                res, _, _ = r1.run_single_config(
                    cfg, spk_train_sets[dname], frontend, caches[dname], caches,
                    trials_sets, multi_trials_sets,
                )
                res["seed"] = int(seed)
                res["wall_time_sec"] = round(time.time() - t0, 2)
                res["summary"] = _summarise(res["multi_domain_eval"])
                entry["per_seed"][str(seed)] = res
                _save()
            rows = [v["summary"] for v in entry["per_seed"].values()]
            entry["mean_std_over_seeds"] = _mean_std(rows)
            entry["seeds"] = sorted(int(k) for k in entry["per_seed"].keys())
            _save()

    # 2. Study 4: 800 steps with its own schedule
    if "horizon" not in skip and "steps_800_own_schedule" not in results["training_horizon_800_own_schedule"]:
        print("\n=== Study 4 re-run: univ_conformer_small_ge2e, 800 steps, own cosine schedule ===")
        cfg = _cfg("universal", num_steps=800)
        res, _, _ = r1.run_single_config(
            cfg, spk_train_sets["universal"], frontend, caches["universal"], caches,
            trials_sets, multi_trials_sets,
        )
        res["num_steps"] = 800
        res["warmup_steps"] = int(cfg.train.warmup_steps)
        res["summary"] = _summarise(res["multi_domain_eval"])
        results["training_horizon_800_own_schedule"]["steps_800_own_schedule"] = res
        _save()

    # 3. Matched-data control
    if "matched" not in skip and "univ_subsampled_5788" not in results["matched_data_control"]:
        ls_count = sum(len(v) for v in spk_train_sets["librispeech"].values())
        print(f"\n=== Matched-data control: universal subsampled to {ls_count} utterances ===")
        sub = _subsample_spk_to_utts(spk_train_sets["universal"], ls_count, seed=42)
        n_utts = sum(len(v) for v in sub.values())
        cfg = _cfg("universal")
        res, _, _ = r1.run_single_config(
            cfg, sub, frontend, caches["universal"], caches, trials_sets, multi_trials_sets,
        )
        res["num_train_utterances"] = n_utts
        res["num_train_speakers"] = len(sub)
        res["target_num_utterances"] = ls_count
        res["summary"] = _summarise(res["multi_domain_eval"])
        results["matched_data_control"]["univ_subsampled_5788"] = res
        _save()

    print(f"Saved Round-2 control results to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round-2 seed-floor and control experiments.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip", type=str, nargs="*", default=[], choices=["seeds", "horizon", "matched"])
    args = parser.parse_args()
    main(args.data_root, args.output_root, args.output_json, args.seeds, args.skip)
