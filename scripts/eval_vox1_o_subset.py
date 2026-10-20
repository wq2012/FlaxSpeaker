#!/usr/bin/env python3
"""Evaluate all pretrained FlaxSpeaker models on the *VoxCeleb1-O 10-speaker subset*.

IMPORTANT -- this is NOT the published VoxCeleb1-O protocol and the resulting
numbers are NOT comparable to published VoxCeleb1-O EERs:

- The official `veri_test2.txt` list (37,720 balanced trials over the 40
  VoxCeleb1 test speakers) is filtered to trials where *both* sides belong to the
  10 held-out evaluation speakers of `splits/voxceleb1/eval.csv`, then capped at
  `max_trials` (default 2,500) -> ~72 % target trials, 45 impostor speaker pairs.
- The other 30 of the 40 official test speakers are used as *training* data by
  the `vox_*` and `univ_*` models (`splits/voxceleb1/train.csv`), because the
  self-contained benchmark deliberately avoids the 1,211-speaker VoxCeleb1-dev
  download. Speaker-disjointness between train and eval is enforced below.
- Features are extracted with exactly the same `VadConfig()` defaults as
  training and the in-domain evaluation (`energy_db_threshold = 26 dB`), so the
  train / test frontend is matched.

Produces:
- `splits/voxceleb1/vox1_o_10spk_subset_<N>.tsv` (the filtered trial list)
- `pretrained_models/vox1_o_10spk_subset_results.json`
- Updated `pretrained_models/benchmark_results.json` and
  `pretrained_models/<model_id>/eval_results.json` (`vox1_o_10spk_subset` fields)
"""

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import configs  # noqa: E402
from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import evaluation  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402

DEFAULT_DATA_ROOT = os.environ.get(
    "FLAXSPEAKER_DATA_ROOT", os.path.expanduser("~/Data/speaker_datasets")
)
DEFAULT_MODELS_DIR = os.path.join(REPO_ROOT, "pretrained_models")

PROTOCOL_KEY = "vox1_o_10spk_subset"
PROTOCOL_NAME = "veri_test2_10spk_subset_2500trials"
PROTOCOL_NOTE = (
    "VoxCeleb1-O 10-speaker subset: official veri_test2.txt filtered to trials whose "
    "enrollment AND test speakers are both among the 10 held-out eval speakers, capped "
    "at 2,500 trials (~72% target). NOT comparable to published VoxCeleb1-O numbers; "
    "30/40 official test speakers are used for training by vox_* and univ_* models."
)
LEGACY_PROTOCOL_KEY = "voxceleb1_o_official"


def _strip_legacy_keys(record: dict[str, Any]) -> None:
    """Removes the pre-Round-2 `voxceleb1_o_official*` fields from a result record."""
    for k in list(record.keys()):
        if k.startswith(LEGACY_PROTOCOL_KEY):
            record.pop(k, None)
    for parent_key in ("cross_dataset_eval",):
        if isinstance(record.get(parent_key), dict):
            record[parent_key].pop(LEGACY_PROTOCOL_KEY, None)
    md = record.get("multi_domain_eval")
    if isinstance(md, dict) and isinstance(md.get("domains_1utt"), dict):
        md["domains_1utt"].pop(LEGACY_PROTOCOL_KEY, None)


def _write_protocol_fields(record: dict[str, Any], metrics: dict[str, Any]) -> None:
    _strip_legacy_keys(record)
    record.setdefault("cross_dataset_eval", {})[PROTOCOL_KEY] = metrics
    record.setdefault("multi_domain_eval", {}).setdefault("domains_1utt", {})[
        PROTOCOL_KEY
    ] = metrics
    record[f"{PROTOCOL_KEY}_eer_percent"] = metrics["eer_percent"]
    record[f"{PROTOCOL_KEY}_min_dcf_001"] = metrics["min_dcf_001"]
    record[f"{PROTOCOL_KEY}_roc_auc"] = metrics["roc_auc"]


def evaluate_all_models_on_vox1_subset(
    data_root: str = DEFAULT_DATA_ROOT,
    models_dir: str = DEFAULT_MODELS_DIR,
    max_trials: int = 2500,
    seed: int = 42,
) -> dict[str, Any]:
    vox_root = os.path.join(data_root, "voxceleb1")
    veri_txt = os.path.join(vox_root, "veri_test2.txt")
    vox_splits_dir = os.path.join(data_root, "splits", "voxceleb1")
    train_csv = os.path.join(vox_splits_dir, "train.csv")
    eval_csv = os.path.join(vox_splits_dir, "eval.csv")

    train_spks = set(dataset.get_csv_spk_to_utts(train_csv).keys())
    eval_spks = set(dataset.get_csv_spk_to_utts(eval_csv).keys())
    overlap = train_spks & eval_spks
    if overlap:
        raise RuntimeError(f"Speaker leakage detected between train and eval splits: {overlap}")

    subset_trials = dataset.load_voxceleb_veri_test(
        veri_txt,
        vox_root,
        max_trials=max_trials,
        seed=seed,
        allowed_speakers=eval_spks,
    )
    subset_trials_path = os.path.join(
        vox_splits_dir, f"{PROTOCOL_KEY}_{len(subset_trials)}.tsv"
    )
    dataset.save_trials_file(subset_trials, subset_trials_path)

    n_pos = sum(1 for lbl, _, _ in subset_trials if lbl == 1)
    n_neg = sum(1 for lbl, _, _ in subset_trials if lbl == 0)
    unique_utts = sorted({u for _, u1, u2 in subset_trials for u in (u1, u2)})
    print(
        f"Loaded {len(subset_trials)} VoxCeleb1-O 10-speaker-subset trials "
        f"(pos={n_pos}, neg={n_neg}, unique_utts={len(unique_utts)}, "
        f"eval_speakers={len(eval_spks)}, train_speakers_from_vox1_test={len(train_spks)})"
    )

    summary_path = os.path.join(data_root, "splits", "dataset_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            ds_summary = json.load(f)
        if "voxceleb1" in ds_summary:
            ds_summary["voxceleb1"][f"num_{PROTOCOL_KEY}_trials"] = len(subset_trials)
            ds_summary["voxceleb1"][f"{PROTOCOL_KEY}_trials_file"] = subset_trials_path
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(ds_summary, f, indent=2)

    # One VAD constant for the whole project: the `VadConfig()` default (26 dB),
    # identical to training (`run_research_benchmarks.py`) and `prepare_datasets.py`.
    frontend_cfg = configs.FrontendConfig(
        feature_type=configs.FeatureType.LOG_MEL,
        sample_rate=16000,
        n_mels=80,
        apply_cmvn=True,
    )
    vad_cfg = configs.VadConfig(mode=configs.VadMode.ENERGY.value)
    frontend = feature_extraction.AudioFrontend(frontend_cfg, vad_cfg)
    cache_dir = os.path.join(data_root, "feature_cache", "voxceleb1")
    cache = feature_extraction.FeatureCache(frontend, cache_dir=cache_dir)
    cache.precompute_dataset({"subset": unique_utts}, num_workers=16)

    bench_path = os.path.join(models_dir, "benchmark_results.json")
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    if "datasets" in bench_data and "voxceleb1" in bench_data["datasets"]:
        ds_entry = bench_data["datasets"]["voxceleb1"]
        ds_entry.pop("num_official_veri_test2_trials", None)
        ds_entry.pop("official_trials_file", None)
        ds_entry[f"num_{PROTOCOL_KEY}_trials"] = len(subset_trials)
        ds_entry[f"{PROTOCOL_KEY}_trials_file"] = subset_trials_path
        ds_entry[f"{PROTOCOL_KEY}_note"] = PROTOCOL_NOTE

    t0 = time.time()
    per_model_results: dict[str, Any] = {}
    experiments_dict = bench_data["experiments"]
    total_models = len(experiments_dict)
    for idx, (exp_id, entry) in enumerate(experiments_dict.items()):
        model_dir = os.path.join(models_dir, exp_id)
        t_m = time.time()
        model = hf_compat.FlaxSpeakerModel.from_pretrained(model_dir)
        metrics = evaluation.evaluate_verification_trials(
            model.state,
            subset_trials,
            model.exp_config,
            feature_cache=cache,
        )
        metrics["num_target_trials"] = n_pos
        metrics["num_impostor_trials"] = n_neg
        metrics["num_eval_speakers"] = len(eval_spks)
        metrics["protocol"] = PROTOCOL_NAME
        metrics["vad_energy_db_threshold"] = float(vad_cfg.energy_db_threshold)
        per_model_results[exp_id] = metrics

        _write_protocol_fields(entry, metrics)

        eval_json_path = os.path.join(model_dir, "eval_results.json")
        if os.path.exists(eval_json_path):
            with open(eval_json_path, "r", encoding="utf-8") as f:
                model_eval = json.load(f)
            _write_protocol_fields(model_eval, metrics)
            with open(eval_json_path, "w", encoding="utf-8") as f:
                json.dump(model_eval, f, indent=2)

        print(
            f"[{idx + 1:02d}/{total_models:02d}] {exp_id}: "
            f"Vox1-O-10spk-subset EER={metrics['eer_percent']:.2f}% "
            f"[{metrics['eer_ci95_low_percent']:.2f}, {metrics['eer_ci95_high_percent']:.2f}], "
            f"minDCF0.01={metrics['min_dcf_001']:.4f}, AUC={metrics['roc_auc']:.4f} "
            f"({time.time() - t_m:.2f}s)"
        )

    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2)

    out_payload = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": PROTOCOL_NAME,
        "protocol_note": PROTOCOL_NOTE,
        "comparable_to_published_voxceleb1_o": False,
        "veri_test2_source": (
            "VoxCeleb1 official veri_test2.txt filtered to the 10 held-out eval speakers"
        ),
        "eval_speakers": sorted(eval_spks),
        "num_vox1_test_speakers_used_for_training": len(train_spks),
        "vad_energy_db_threshold": float(vad_cfg.energy_db_threshold),
        "num_trials": len(subset_trials),
        "num_target_trials": n_pos,
        "num_impostor_trials": n_neg,
        "num_unique_utterances": len(unique_utts),
        "total_eval_time_sec": round(time.time() - t0, 2),
        "models": per_model_results,
    }
    out_json_path = os.path.join(models_dir, f"{PROTOCOL_KEY}_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    legacy_json = os.path.join(models_dir, "voxceleb1_o_official_results.json")
    if os.path.exists(legacy_json):
        os.remove(legacy_json)
    print(f"Saved VoxCeleb1-O 10-speaker-subset results to {out_json_path}")
    return out_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all pretrained models on the VoxCeleb1-O 10-speaker subset "
            "(filtered veri_test2.txt; NOT comparable to published VoxCeleb1-O)."
        )
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--max_trials", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate_all_models_on_vox1_subset(
        data_root=args.data_root,
        models_dir=args.models_dir,
        max_trials=args.max_trials,
        seed=args.seed,
    )
