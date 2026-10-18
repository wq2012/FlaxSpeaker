#!/usr/bin/env python3
"""Evaluate all 44 pretrained FlaxSpeaker models on the speaker-disjoint official VoxCeleb1-O (veri_test2.txt) protocol.

Enforces strict open-set speaker-disjointness by filtering official `veri_test2.txt`
trials to pairs where both enrollment and test speakers belong to the 10 held-out
unseen evaluation speakers in `splits/voxceleb1/eval.csv` (strictly disjoint from
the 30 training speakers in `splits/voxceleb1/train.csv` and `splits/universal/train.csv`).

Produces:
- `splits/voxceleb1/vox1_o_official_2500.tsv` (2,500 speaker-disjoint official trials)
- `pretrained_models/voxceleb1_o_official_results.json`
- Updated `pretrained_models/benchmark_results.json` and `pretrained_models/<model_id>/eval_results.json`
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


def evaluate_all_models_on_vox1_official(
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

    official_trials = dataset.load_voxceleb_veri_test(
        veri_txt,
        vox_root,
        max_trials=max_trials,
        seed=seed,
        allowed_speakers=eval_spks,
    )
    official_trials_path = os.path.join(
        vox_splits_dir, f"vox1_o_official_{len(official_trials)}.tsv"
    )
    dataset.save_trials_file(official_trials, official_trials_path)

    n_pos = sum(1 for lbl, _, _ in official_trials if lbl == 1)
    n_neg = sum(1 for lbl, _, _ in official_trials if lbl == 0)
    unique_utts = sorted({u for _, u1, u2 in official_trials for u in (u1, u2)})
    print(
        f"Loaded {len(official_trials)} speaker-disjoint official veri_test2.txt trials "
        f"(pos={n_pos}, neg={n_neg}, unique_utts={len(unique_utts)}, unseen_speakers={len(eval_spks)})"
    )

    # Update splits/dataset_summary.json if present
    summary_path = os.path.join(data_root, "splits", "dataset_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            ds_summary = json.load(f)
        if "voxceleb1" in ds_summary:
            ds_summary["voxceleb1"]["num_official_veri_test2_trials"] = len(official_trials)
            ds_summary["voxceleb1"]["official_trials_file"] = official_trials_path
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(ds_summary, f, indent=2)

    frontend_cfg = configs.FrontendConfig(
        feature_type=configs.FeatureType.LOG_MEL,
        sample_rate=16000,
        n_mels=80,
        apply_cmvn=True,
    )
    vad_cfg = configs.VadConfig(mode=configs.VadMode.ENERGY, energy_db_threshold=30.0)
    frontend = feature_extraction.AudioFrontend(frontend_cfg, vad_cfg)
    cache_dir = os.path.join(data_root, "feature_cache", "voxceleb1")
    cache = feature_extraction.FeatureCache(frontend, cache_dir=cache_dir)
    cache.precompute_dataset({"official": unique_utts}, num_workers=16)

    bench_path = os.path.join(models_dir, "benchmark_results.json")
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    if "datasets" in bench_data and "voxceleb1" in bench_data["datasets"]:
        bench_data["datasets"]["voxceleb1"]["num_official_veri_test2_trials"] = len(
            official_trials
        )
        bench_data["datasets"]["voxceleb1"]["official_trials_file"] = official_trials_path

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
            official_trials,
            model.exp_config,
            feature_cache=cache,
        )
        metrics["num_target_trials"] = n_pos
        metrics["num_impostor_trials"] = n_neg
        metrics["num_unseen_speakers"] = len(eval_spks)
        metrics["protocol"] = "veri_test2_speaker_disjoint_10spk_2500trials"
        per_model_results[exp_id] = metrics

        entry.setdefault("cross_dataset_eval", {})["voxceleb1_o_official"] = metrics
        entry.setdefault("multi_domain_eval", {}).setdefault(
            "domains_1utt", {}
        )["voxceleb1_o_official"] = metrics
        entry["voxceleb1_o_official_eer_percent"] = metrics["eer_percent"]
        entry["voxceleb1_o_official_min_dcf_001"] = metrics["min_dcf_001"]
        entry["voxceleb1_o_official_roc_auc"] = metrics["roc_auc"]

        eval_json_path = os.path.join(model_dir, "eval_results.json")
        if os.path.exists(eval_json_path):
            with open(eval_json_path, "r", encoding="utf-8") as f:
                model_eval = json.load(f)
            model_eval.setdefault("cross_dataset_eval", {})[
                "voxceleb1_o_official"
            ] = metrics
            model_eval.setdefault("multi_domain_eval", {}).setdefault(
                "domains_1utt", {}
            )["voxceleb1_o_official"] = metrics
            model_eval["voxceleb1_o_official_eer_percent"] = metrics["eer_percent"]
            model_eval["voxceleb1_o_official_min_dcf_001"] = metrics["min_dcf_001"]
            model_eval["voxceleb1_o_official_roc_auc"] = metrics["roc_auc"]
            with open(eval_json_path, "w", encoding="utf-8") as f:
                json.dump(model_eval, f, indent=2)

        print(
            f"[{idx + 1:02d}/{total_models:02d}] {exp_id}: "
            f"Vox1-O Official EER={metrics['eer_percent']:.2f}% "
            f"[{metrics['eer_ci95_low_percent']:.2f}, {metrics['eer_ci95_high_percent']:.2f}], "
            f"minDCF0.01={metrics['min_dcf_001']:.4f}, AUC={metrics['roc_auc']:.4f} "
            f"({time.time() - t_m:.2f}s)"
        )

    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2)

    out_payload = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": "veri_test2_speaker_disjoint_10spk_2500trials",
        "veri_test2_source": "VoxCeleb1 official veri_test2.txt filtered to 10 unseen eval speakers",
        "unseen_eval_speakers": sorted(eval_spks),
        "num_trials": len(official_trials),
        "num_target_trials": n_pos,
        "num_impostor_trials": n_neg,
        "num_unique_utterances": len(unique_utts),
        "total_eval_time_sec": round(time.time() - t0, 2),
        "models": per_model_results,
    }
    official_json_path = os.path.join(models_dir, "voxceleb1_o_official_results.json")
    with open(official_json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"Saved official VoxCeleb1-O results to {official_json_path}")
    return out_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all 44 pretrained models on speaker-disjoint "
            "official VoxCeleb1-O (veri_test2.txt) trials."
        )
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--max_trials", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate_all_models_on_vox1_official(
        data_root=args.data_root,
        models_dir=args.models_dir,
        max_trials=args.max_trials,
        seed=args.seed,
    )
