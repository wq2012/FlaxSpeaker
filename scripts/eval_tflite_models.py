#!/usr/bin/env python3
"""Evaluate the 6 exported TFLite models (FP32 and dynamic-range INT8) on the verification trial lists.

Measures:
  1. 180-frame fixed-window 1-utterance EER (%) for both `model_fp32.tflite` and
     `model_int8.tflite` across `librispeech` (3,000 trials), `cnceleb2` (3,000 trials),
     `voxceleb1` (2,000 trials), `universal_pooled` (8,000 trials), and the 3-domain
     macro-average, alongside the full-sequence (<=3000-frame) masked JAX EER.
     Every EER carries a stratified trial-bootstrap 95 % CI
     (`evaluation.compute_bootstrap_eer_ci`, 500 resamples, seed 42) under
     `eer_ci95_low` / `eer_ci95_high` of the per-domain records.
  2. Utterance-level cosine similarity `cos(FP32, INT8)` (mean and min over all 1,988
     unique evaluation utterances).
  3. Averaged per-utterance CPU latency (ms) over 50 timed invocations (after 5
     warm-up calls), measured sequentially on an otherwise idle interpreter.

Updates `pretrained_models/<model_id>/eval_results.json` (`"tflite_eval"`) and
`pretrained_models/benchmark_results.json` (`experiments.<id>.tflite_eval`,
`tflite_exports.<id>_fp32|_int8.tflite_eval` and their `latency_ms`, which is
overwritten with the 50-run mean so that the single-call export-time number is not
left behind).

Features are extracted with each model's own frontend / VAD configuration
(`pretrained_models/<model_id>/config.json`), not the library defaults.
"""

import argparse
from concurrent import futures
import json
import os
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import evaluation  # noqa: E402
from flaxspeaker import export  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402

TFLITE_MODEL_IDS = (
    "univ_conformer_small_ge2e",
    "univ_ecapa_tdnn_small_ge2e",
    "univ_lstm_small_ge2e",
    "ls_conformer_small_ge2e",
    "ls_ecapa_tdnn_small_ge2e",
    "ls_lstm_small_ge2e",
)


def _score_trials_cosine(
    trials: list[tuple[int, str, str]],
    emb_map: dict[str, np.ndarray],
) -> dict[str, Any]:
    labels = np.array([t[0] for t in trials], dtype=np.int32)
    e1 = np.stack([emb_map[t[1]] for t in trials], axis=0)
    e2 = np.stack([emb_map[t[2]] for t in trials], axis=0)
    n1 = np.linalg.norm(e1, axis=-1, keepdims=True) + 1e-6
    n2 = np.linalg.norm(e2, axis=-1, keepdims=True) + 1e-6
    scores = np.sum((e1 / n1) * (e2 / n2), axis=-1)
    eer, thr = evaluation.compute_eer_exact(labels, scores)
    ci_low, ci_high = evaluation.compute_bootstrap_eer_ci(labels, scores)
    dcf001 = evaluation.compute_min_dcf(labels, scores, p_target=0.01)
    auc = evaluation.compute_roc_auc(labels, scores)
    return {
        "eer_percent": round(eer * 100.0, 2),
        "eer_ci95_low": round(ci_low * 100.0, 2),
        "eer_ci95_high": round(ci_high * 100.0, 2),
        "eer_threshold": round(float(thr), 6),
        "min_dcf_001": round(float(dcf001), 4),
        "roc_auc": round(float(auc), 4),
        "num_trials": len(trials),
    }


def _load_feat_map(
    model_id: str, pm_dir: str, cache_root: str, all_utts: list[str]
) -> dict[str, np.ndarray]:
    """Features for `all_utts` extracted with the model's own frontend / VAD config."""
    model_dir = os.path.join(pm_dir, model_id)
    exp_cfg = hf_compat.FlaxSpeakerModel.load_exp_config(model_dir)
    frontend = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)
    fcache = feature_extraction.FeatureCache(
        frontend=frontend,
        n_mfcc=exp_cfg.frontend.feature_dim,
        cache_dir=os.path.join(cache_root, "universal"),
    )
    fcache.preload(all_utts, num_workers=8)
    return {u: fcache.get(u) for u in all_utts}


def _eval_one_model(
    model_id: str,
    pm_dir: str,
    domain_trials: dict[str, list[tuple[int, str, str]]],
    feat_map: dict[str, np.ndarray],
) -> tuple[str, dict[str, Any]]:
    model_dir = os.path.join(pm_dir, model_id)
    fp32_path = os.path.join(model_dir, "model_fp32.tflite")
    int8_path = os.path.join(model_dir, "model_int8.tflite")
    runner_fp32 = export.TFLiteSpeakerRunner(fp32_path)
    runner_int8 = export.TFLiteSpeakerRunner(int8_path)

    all_utts = sorted(feat_map.keys())
    embs_fp32: dict[str, np.ndarray] = {}
    embs_int8: dict[str, np.ndarray] = {}
    cos_vals: list[float] = []

    for u in all_utts:
        feat = feat_map[u]
        v32 = runner_fp32(feat)
        v8 = runner_int8(feat)
        embs_fp32[u] = v32
        embs_int8[u] = v8
        c = float(
            np.dot(v32, v8)
            / (np.linalg.norm(v32) * np.linalg.norm(v8) + 1e-12)
        )
        cos_vals.append(c)

    domains_fp32: dict[str, Any] = {}
    domains_int8: dict[str, Any] = {}
    for dom_name, trials in domain_trials.items():
        domains_fp32[dom_name] = _score_trials_cosine(trials, embs_fp32)
        domains_int8[dom_name] = _score_trials_cosine(trials, embs_int8)

    macro_fp32 = round(
        float(
            np.mean(
                [
                    domains_fp32[d]["eer_percent"]
                    for d in ("librispeech", "cnceleb2", "voxceleb1")
                ]
            )
        ),
        2,
    )
    macro_int8 = round(
        float(
            np.mean(
                [
                    domains_int8[d]["eer_percent"]
                    for d in ("librispeech", "cnceleb2", "voxceleb1")
                ]
            )
        ),
        2,
    )

    with open(
        os.path.join(model_dir, "eval_results.json"), "r", encoding="utf-8"
    ) as f:
        eval_res = json.load(f)

    primary_dom = (
        "universal_pooled" if model_id.startswith("univ_") else "librispeech"
    )
    primary_protocol = (
        "universal_pooled 8,000 1-utt trials (fixed 180-frame unmasked window)"
        if primary_dom == "universal_pooled"
        else "librispeech 3,000 1-utt trials (fixed 180-frame unmasked window)"
    )
    jax_full_eer = float(
        eval_res["multi_domain_eval"]["domains_1utt"][primary_dom][
            "eer_percent"
        ]
    )
    jax_pooled_eer = float(
        eval_res["multi_domain_eval"]["domains_1utt"]["universal_pooled"][
            "eer_percent"
        ]
    )
    jax_ls_eer = float(
        eval_res["multi_domain_eval"]["domains_1utt"]["librispeech"][
            "eer_percent"
        ]
    )

    tfl_eval = {
        "window_frames": runner_fp32.seq_len,
        "primary_domain": primary_dom,
        "protocol": primary_protocol,
        "eer_percent": domains_fp32[primary_dom]["eer_percent"],
        "eer_ci95_fp32": [
            domains_fp32[primary_dom]["eer_ci95_low"],
            domains_fp32[primary_dom]["eer_ci95_high"],
        ],
        "eer_percent_int8": domains_int8[primary_dom]["eer_percent"],
        "eer_ci95_int8": [
            domains_int8[primary_dom]["eer_ci95_low"],
            domains_int8[primary_dom]["eer_ci95_high"],
        ],
        "jax_full_seq_eer_percent": round(jax_full_eer, 2),
        "pooled_1utt_eer_fp32_percent": domains_fp32["universal_pooled"][
            "eer_percent"
        ],
        "pooled_1utt_eer_int8_percent": domains_int8["universal_pooled"][
            "eer_percent"
        ],
        "jax_full_seq_pooled_1utt_eer_percent": round(jax_pooled_eer, 2),
        "librispeech_1utt_eer_fp32_percent": domains_fp32["librispeech"][
            "eer_percent"
        ],
        "librispeech_1utt_eer_int8_percent": domains_int8["librispeech"][
            "eer_percent"
        ],
        "jax_full_seq_librispeech_1utt_eer_percent": round(jax_ls_eer, 2),
        "macro_1utt_eer_fp32_percent": macro_fp32,
        "macro_1utt_eer_int8_percent": macro_int8,
        "cos_fp32_int8_mean": round(float(np.mean(cos_vals)), 6),
        "cos_fp32_int8_min": round(float(np.min(cos_vals)), 6),
        "num_unique_utterances": len(all_utts),
        "domains_1utt_fp32": domains_fp32,
        "domains_1utt_int8": domains_int8,
    }
    return model_id, tfl_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get(
            "FLAXSPEAKER_DATA_ROOT",
            os.path.expanduser("~/Data/speaker_datasets"),
        ),
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "pretrained_models"),
    )
    args = parser.parse_args()

    os.environ["FLAXSPEAKER_DATA_ROOT"] = os.path.abspath(args.data_root)
    splits_dir = os.path.join(args.data_root, "splits")
    cache_root = os.path.join(args.data_root, "feature_cache")

    domain_trials: dict[str, list[tuple[int, str, str]]] = {}
    all_utts_set: set[str] = set()
    for dom in ("librispeech", "cnceleb2", "voxceleb1"):
        t_path = os.path.join(splits_dir, dom, "trials.tsv")
        trials = dataset.load_trials_file(t_path)
        domain_trials[dom] = trials
        for _, u1, u2 in trials:
            all_utts_set.add(u1)
            all_utts_set.add(u2)
    domain_trials["universal_pooled"] = (
        domain_trials["librispeech"]
        + domain_trials["cnceleb2"]
        + domain_trials["voxceleb1"]
    )

    all_utts = sorted(all_utts_set)
    feat_maps = {
        mid: _load_feat_map(mid, args.models_dir, cache_root, all_utts)
        for mid in TFLITE_MODEL_IDS
    }
    print(f"Loaded features for {len(all_utts)} unique evaluation utterances.")

    results: dict[str, dict[str, Any]] = {}
    with futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [
            ex.submit(
                _eval_one_model, mid, args.models_dir, domain_trials, feat_maps[mid]
            )
            for mid in TFLITE_MODEL_IDS
        ]
        for fut in futures.as_completed(futs):
            mid, tfl_eval = fut.result()
            results[mid] = tfl_eval

    # Sequentially measure single-thread / uncontended per-utterance CPU latency
    sample_feat = feat_maps[TFLITE_MODEL_IDS[0]][all_utts[0]]
    for mid in TFLITE_MODEL_IDS:
        model_dir = os.path.join(args.models_dir, mid)
        runner_fp32 = export.TFLiteSpeakerRunner(
            os.path.join(model_dir, "model_fp32.tflite")
        )
        runner_int8 = export.TFLiteSpeakerRunner(
            os.path.join(model_dir, "model_int8.tflite")
        )
        for _ in range(5):
            runner_fp32(sample_feat)
            runner_int8(sample_feat)
        t0 = time.perf_counter()
        for _ in range(50):
            runner_fp32(sample_feat)
        lat_fp32_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
        t0 = time.perf_counter()
        for _ in range(50):
            runner_int8(sample_feat)
        lat_int8_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
        tfl_eval = results[mid]
        tfl_eval["latency_ms_fp32_mean_50runs"] = round(lat_fp32_ms, 2)
        tfl_eval["latency_ms_int8_mean_50runs"] = round(lat_int8_ms, 2)
        print(
            f"  {mid}: primary ({tfl_eval['primary_domain']}) "
            f"FP32={tfl_eval['eer_percent']:.2f}% INT8={tfl_eval['eer_percent_int8']:.2f}% "
            f"(JAX full-seq={tfl_eval['jax_full_seq_eer_percent']:.2f}%), "
            f"cos(FP32,INT8)={tfl_eval['cos_fp32_int8_mean']:.6f} "
            f"(min={tfl_eval['cos_fp32_int8_min']:.6f}), "
            f"lat FP32={lat_fp32_ms:.2f}ms INT8={lat_int8_ms:.2f}ms"
        )

    # Save into each model's eval_results.json and into benchmark_results.json
    bench_path = os.path.join(args.models_dir, "benchmark_results.json")
    with open(bench_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    import datetime

    for mid in TFLITE_MODEL_IDS:
        tfl_eval = results[mid]
        ev_path = os.path.join(args.models_dir, mid, "eval_results.json")
        with open(ev_path, "r", encoding="utf-8") as f:
            ev = json.load(f)
        ev["tflite_eval"] = tfl_eval
        ev_exports = ev.get("tflite_exports", {})
        if isinstance(ev_exports, dict):
            for suffix, lat_key in (
                ("fp32", "latency_ms_fp32_mean_50runs"),
                ("int8", "latency_ms_int8_mean_50runs"),
            ):
                exp_key = f"{mid}_{suffix}"
                if exp_key in ev_exports:
                    ev_exports[exp_key]["latency_ms"] = tfl_eval[lat_key]
                    ev_exports[exp_key]["latency_ms_source"] = (
                        "scripts/eval_tflite_models.py: mean of 50 sequential calls "
                        "after 5 warm-up calls"
                    )
                    ev_exports[exp_key]["tflite_eval"] = {
                        "eer_percent": tfl_eval[
                            "eer_percent" if suffix == "fp32" else "eer_percent_int8"
                        ],
                        "eer_ci95": tfl_eval[f"eer_ci95_{suffix}"],
                        "primary_domain": tfl_eval["primary_domain"],
                    }
        with open(ev_path, "w", encoding="utf-8") as f:
            json.dump(ev, f, indent=2)

        if mid in bench.get("experiments", {}):
            bench["experiments"][mid]["tflite_eval"] = tfl_eval
            if isinstance(ev_exports, dict):
                bench["experiments"][mid]["tflite_exports"] = ev_exports
        # `tflite_exports` is keyed `<model_id>_fp32` / `<model_id>_int8`; overwrite the
        # export-time single-call `latency_ms` with the 50-run mean measured above.
        exports = bench.setdefault("tflite_exports", {})
        for suffix, lat_key in (
            ("fp32", "latency_ms_fp32_mean_50runs"),
            ("int8", "latency_ms_int8_mean_50runs"),
        ):
            exp_key = f"{mid}_{suffix}"
            if exp_key in exports:
                exports[exp_key]["latency_ms"] = tfl_eval[lat_key]
                exports[exp_key]["latency_ms_source"] = (
                    "scripts/eval_tflite_models.py: mean of 50 sequential calls "
                    "after 5 warm-up calls"
                )
                exports[exp_key]["tflite_eval"] = {
                    "eer_percent": tfl_eval[
                        "eer_percent" if suffix == "fp32" else "eer_percent_int8"
                    ],
                    "eer_ci95": tfl_eval[f"eer_ci95_{suffix}"],
                    "primary_domain": tfl_eval["primary_domain"],
                }

    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2)

    tfl_json_path = os.path.join(args.models_dir, "tflite_eval_results.json")
    with open(tfl_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "window_frames": 180,
                "num_models": len(results),
                "models": results,
            },
            f,
            indent=2,
        )
    print(
        f"Updated eval_results.json, benchmark_results.json, and {tfl_json_path} with tflite_eval."
    )


if __name__ == "__main__":
    main()
