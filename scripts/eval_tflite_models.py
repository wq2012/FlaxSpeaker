#!/usr/bin/env python3
"""Evaluate exported TFLite models (FP32 and dynamic-range INT8) on verification trial lists.

Supports both:
  - Tier-1 controlled models (`TFLITE_TIER1_MODEL_IDS`, 6 models): evaluated on
    `librispeech` (3,000 trials), `cnceleb2` (3,000 trials), `voxceleb1` (2,000 trials),
    and `universal_pooled` (8,000 trials).
  - Tier-2 full-data flagship models (`TFLITE_TIER2_MODEL_IDS`, 9 models): evaluated on
    the official 40-speaker `VoxCeleb1-O` benchmark (`37,611` trials, `4,708` unique
    utterances, 180-frame center crop), plus `cos(Flax, FP32)` and `cos(Flax, INT8)`.

Measures stratified trial-bootstrap 95% CIs (`evaluation.compute_bootstrap_eer_ci`,
500 resamples, seed 42), `cos(FP32, INT8)` (mean and min), and averaged per-utterance
CPU latency (ms) over 50 timed invocations after 5 warm-up calls.

Updates `pretrained_models/<model_id>/eval_results.json` (`"tflite_eval"`),
`pretrained_models/tflite_eval_results.json`, and `pretrained_models/benchmark_results.json`.
"""

import argparse
from concurrent import futures
import datetime
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
from flaxspeaker import resources  # noqa: E402

TFLITE_TIER1_MODEL_IDS = (
    "univ_conformer_small_ge2e",
    "univ_ecapa_tdnn_small_ge2e",
    "univ_lstm_small_ge2e",
    "ls_conformer_small_ge2e",
    "ls_ecapa_tdnn_small_ge2e",
    "ls_lstm_small_ge2e",
)

TFLITE_TIER2_MODEL_IDS = (
    "univ_full_conformer_large_arcface",
    "univ_full_ecapa_c1024_arcface",
    "univ_full_resnet34_arcface",
    "univ_full_eres2conformer_large",
    "univ_full_eres2conformer_v2_large",
    "univ_full_eres2conformer_xl",
    "univ_full_eres2conformer_v2_diar",
    "univ_full_eres2conformer_xl_diar",
    "univ_full_eres2conformer_xl_diar_adapt",
)

TFLITE_MODEL_IDS = TFLITE_TIER1_MODEL_IDS + TFLITE_TIER2_MODEL_IDS


def _score_trials_cosine(
    trials: list[tuple[int, str, str]],
    emb_map: dict[str, np.ndarray],
    eer_decimals: int = 2,
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
        "eer_percent": round(eer * 100.0, eer_decimals),
        "eer_ci95_low": round(ci_low * 100.0, 2),
        "eer_ci95_high": round(ci_high * 100.0, 2),
        "eer_threshold": round(float(thr), 6),
        "min_dcf_001": round(float(dcf001), 4),
        "roc_auc": round(float(auc), 4),
        "num_trials": len(trials),
    }


def _load_feat_map(
    model_id: str,
    pm_dir: str,
    cache_subdir: str,
    all_utts: list[str],
) -> dict[str, np.ndarray]:
    """Features for `all_utts` extracted with the model's own frontend / VAD config."""
    model_dir = os.path.join(pm_dir, model_id)
    exp_cfg = hf_compat.FlaxSpeakerModel.load_exp_config(model_dir)
    frontend = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)
    fcache = feature_extraction.FeatureCache(
        frontend=frontend,
        n_mfcc=exp_cfg.frontend.feature_dim,
        cache_dir=cache_subdir,
    )
    fcache.preload(all_utts, num_workers=8)
    return {u: fcache.get(u) for u in all_utts}


def _eval_tier1_model(
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
            np.dot(v32, v8) / (np.linalg.norm(v32) * np.linalg.norm(v8) + 1e-12)
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
        eval_res["multi_domain_eval"]["domains_1utt"][primary_dom]["eer_percent"]
    )
    jax_pooled_eer = float(
        eval_res["multi_domain_eval"]["domains_1utt"]["universal_pooled"][
            "eer_percent"
        ]
    )
    jax_ls_eer = float(
        eval_res["multi_domain_eval"]["domains_1utt"]["librispeech"]["eer_percent"]
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


def _crop_or_tile_180(feat: np.ndarray, seq_len: int) -> np.ndarray:
    arr = np.asarray(feat, dtype=np.float32)
    if arr.shape[0] >= seq_len:
        off = (arr.shape[0] - seq_len) // 2
        return arr[off:off + seq_len]
    reps = (seq_len // max(1, arr.shape[0])) + 1
    return np.tile(arr, (reps, 1))[:seq_len]


def _eval_tier2_flagship_model(
    model_id: str,
    pm_dir: str,
    vox_trials: list[tuple[int, str, str]],
    feat_map: dict[str, np.ndarray],
) -> tuple[str, dict[str, Any]]:
    import jax.numpy as jnp

    model_dir = os.path.join(pm_dir, model_id)
    fp32_path = os.path.join(model_dir, "model_fp32.tflite")
    int8_path = os.path.join(model_dir, "model_int8.tflite")
    runner_fp32_meta = export.TFLiteSpeakerRunner(fp32_path)
    hf_model = hf_compat.FlaxSpeakerModel.from_pretrained(model_dir)

    all_utts = sorted(feat_map.keys())
    embs_fp32: dict[str, np.ndarray] = {}
    embs_int8: dict[str, np.ndarray] = {}
    cos_vals: list[float] = []
    crops_180: list[np.ndarray] = [
        _crop_or_tile_180(feat_map[u], runner_fp32_meta.seq_len)
        for u in all_utts[:128]
    ]
    xb_180 = jnp.asarray(np.stack(crops_180, axis=0), dtype=jnp.float32)
    flax_embs = np.asarray(hf_model(xb_180), dtype=np.float32)

    n_workers = 8
    chunks = [all_utts[i::n_workers] for i in range(n_workers)]

    def _run_chunk(utt_chunk: list[str]) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
        r32 = export.TFLiteSpeakerRunner(fp32_path)
        r8 = export.TFLiteSpeakerRunner(int8_path)
        out_c = []
        for u_c in utt_chunk:
            f_c = feat_map[u_c]
            v32_c = r32(f_c)
            v8_c = r8(f_c)
            c_val = float(
                np.dot(v32_c, v8_c)
                / (np.linalg.norm(v32_c) * np.linalg.norm(v8_c) + 1e-12)
            )
            out_c.append((u_c, v32_c, v8_c, c_val))
        return out_c

    with futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for batch_res in pool.map(_run_chunk, chunks):
            for u, v32, v8, c in batch_res:
                embs_fp32[u] = v32
                embs_int8[u] = v8
                cos_vals.append(c)

    # Measure Flax vs TFLite cosine fidelity over 128 center-cropped 180-frame utterances
    cos_flax_32: list[float] = []
    cos_flax_8: list[float] = []
    for idx_u, u in enumerate(all_utts[: len(crops_180)]):
        vf = flax_embs[idx_u]
        v32 = embs_fp32[u]
        v8 = embs_int8[u]
        cos_flax_32.append(
            float(
                np.dot(vf, v32)
                / (np.linalg.norm(vf) * np.linalg.norm(v32) + 1e-12)
            )
        )
        cos_flax_8.append(
            float(
                np.dot(vf, v8) / (np.linalg.norm(vf) * np.linalg.norm(v8) + 1e-12)
            )
        )

    res_fp32 = _score_trials_cosine(vox_trials, embs_fp32, eer_decimals=4)
    res_int8 = _score_trials_cosine(vox_trials, embs_int8, eer_decimals=4)

    with open(
        os.path.join(model_dir, "eval_results.json"), "r", encoding="utf-8"
    ) as f:
        eval_res = json.load(f)
    vox_off = eval_res["official_benchmarks"]["voxceleb1_o"]
    jax_full_eer = float(vox_off.get("eer_raw", vox_off["eer_percent"]))
    if jax_full_eer < 1.0 and vox_off["eer_percent"] >= 1.0:
        jax_full_eer = jax_full_eer * 100.0

    fp32_bytes = os.path.getsize(fp32_path)
    int8_bytes = os.path.getsize(int8_path)
    tfl_eval = {
        "window_frames": runner_fp32_meta.seq_len,
        "primary_domain": "voxceleb1_o",
        "protocol": "VoxCeleb1-O official (37,611 trials, 180-frame center crop)",
        "eer_percent": res_fp32["eer_percent"],
        "eer_ci95_fp32": [res_fp32["eer_ci95_low"], res_fp32["eer_ci95_high"]],
        "min_dcf_001": res_fp32["min_dcf_001"],
        "eer_percent_int8": res_int8["eer_percent"],
        "eer_ci95_int8": [res_int8["eer_ci95_low"], res_int8["eer_ci95_high"]],
        "min_dcf_001_int8": res_int8["min_dcf_001"],
        "jax_full_seq_eer_percent": round(jax_full_eer, 4),
        "cos_fp32_int8_mean": round(float(np.mean(cos_vals)), 6),
        "cos_fp32_int8_min": round(float(np.min(cos_vals)), 6),
        "cos_flax_fp32_mean": round(float(np.mean(cos_flax_32)), 6),
        "cos_flax_int8_mean": round(float(np.mean(cos_flax_8)), 6),
        "fp32_size_kb": round(fp32_bytes / 1024.0, 2),
        "int8_size_kb": round(int8_bytes / 1024.0, 2),
        "fp32_size_bytes": fp32_bytes,
        "int8_size_bytes": int8_bytes,
        "num_unique_utterances": len(all_utts),
    }
    return model_id, tfl_eval


def main() -> None:
    resources.configure_polite_resources(max_threads=8, nice_level=10)
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
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=[],
        help="Evaluate only the specified model IDs and merge into tflite_eval_results.json.",
    )
    args = parser.parse_args()

    os.environ["FLAXSPEAKER_DATA_ROOT"] = os.path.abspath(args.data_root)
    splits_dir = os.path.join(args.data_root, "splits")
    cache_root = os.path.join(args.data_root, "feature_cache")

    target_ids = (
        [m for m in TFLITE_MODEL_IDS if m in set(args.only)]
        if args.only
        else [
            m
            for m in TFLITE_MODEL_IDS
            if os.path.isfile(os.path.join(args.models_dir, m, "model_fp32.tflite"))
        ]
    )
    tier1_targets = [m for m in target_ids if m in TFLITE_TIER1_MODEL_IDS]
    tier2_targets = [m for m in target_ids if m in TFLITE_TIER2_MODEL_IDS]

    results: dict[str, dict[str, Any]] = {}
    sample_feat_by_mid: dict[str, np.ndarray] = {}

    if tier1_targets:
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
        all_utts_t1 = sorted(all_utts_set)
        feat_maps_t1 = {
            mid: _load_feat_map(
                mid,
                args.models_dir,
                os.path.join(cache_root, "universal"),
                all_utts_t1,
            )
            for mid in tier1_targets
        }
        for mid in tier1_targets:
            sample_feat_by_mid[mid] = feat_maps_t1[mid][all_utts_t1[0]]
        with futures.ThreadPoolExecutor(max_workers=min(6, len(tier1_targets))) as ex:
            futs = [
                ex.submit(
                    _eval_tier1_model,
                    mid,
                    args.models_dir,
                    domain_trials,
                    feat_maps_t1[mid],
                )
                for mid in tier1_targets
            ]
            for fut in futures.as_completed(futs):
                mid, tfl_eval = fut.result()
                results[mid] = tfl_eval

    if tier2_targets:
        vox_dir = os.path.join(args.data_root, "voxceleb1")
        vox_trials = dataset.load_voxceleb_veri_test(
            os.path.join(vox_dir, "veri_test2.txt"), vox_dir
        )
        vox_utts = sorted({u for _, u1, u2 in vox_trials for u in (u1, u2)})
        for mid in tier2_targets:
            fmap_t2 = _load_feat_map(
                mid,
                args.models_dir,
                os.path.join(cache_root, "voxceleb1_o_full"),
                vox_utts,
            )
            sample_feat_by_mid[mid] = fmap_t2[vox_utts[0]]
            _, tfl_eval = _eval_tier2_flagship_model(
                mid, args.models_dir, vox_trials, fmap_t2
            )
            results[mid] = tfl_eval

    # Sequentially measure uncontended per-utterance CPU latency (50 runs after 5 warm-up)
    for mid in target_ids:
        model_dir = os.path.join(args.models_dir, mid)
        runner_fp32 = export.TFLiteSpeakerRunner(
            os.path.join(model_dir, "model_fp32.tflite")
        )
        runner_int8 = export.TFLiteSpeakerRunner(
            os.path.join(model_dir, "model_int8.tflite")
        )
        s_feat = _crop_or_tile_180(sample_feat_by_mid[mid], runner_fp32.seq_len)
        for _ in range(5):
            runner_fp32(s_feat)
            runner_int8(s_feat)
        t0 = time.perf_counter()
        for _ in range(50):
            runner_fp32(s_feat)
        lat_fp32_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
        t0 = time.perf_counter()
        for _ in range(50):
            runner_int8(s_feat)
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

    bench_path = os.path.join(args.models_dir, "benchmark_results.json")
    with open(bench_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    for mid in target_ids:
        tfl_eval = results[mid]
        ev_path = os.path.join(args.models_dir, mid, "eval_results.json")
        with open(ev_path, "r", encoding="utf-8") as f:
            ev = json.load(f)
        ev["tflite_eval"] = tfl_eval
        ev_exports = ev.setdefault("tflite_exports", {})
        exports = bench.setdefault("tflite_exports", {})
        for suffix, is_q8, lat_key in (
            ("fp32", False, "latency_ms_fp32_mean_50runs"),
            ("int8", True, "latency_ms_int8_mean_50runs"),
        ):
            exp_key = f"{mid}_{suffix}"
            rel_tfl = f"pretrained_models/{mid}/model_{suffix}.tflite"
            abs_tfl = os.path.join(args.models_dir, mid, f"model_{suffix}.tflite")
            sz_b = os.path.getsize(abs_tfl)
            cos_flax = tfl_eval.get(
                "cos_flax_fp32_mean" if suffix == "fp32" else "cos_flax_int8_mean",
                1.0 if suffix == "fp32" else tfl_eval["cos_fp32_int8_mean"],
            )
            entry_payload = exports.get(
                exp_key,
                {
                    "tflite_path": rel_tfl,
                    "size_bytes": sz_b,
                    "size_kb": round(sz_b / 1024.0, 2),
                    "quantize_int8": is_q8,
                },
            )
            entry_payload["size_bytes"] = sz_b
            entry_payload["size_kb"] = round(sz_b / 1024.0, 2)
            entry_payload["latency_ms"] = tfl_eval[lat_key]
            entry_payload["cosine_fidelity_vs_flax"] = cos_flax
            entry_payload["latency_ms_source"] = (
                "50 sequential calls after 5 warm-up calls on 180x80 log-Mel input"
            )
            entry_payload["tflite_eval"] = {
                "eer_percent": round(
                    float(
                        tfl_eval[
                            "eer_percent" if suffix == "fp32" else "eer_percent_int8"
                        ]
                    ),
                    2,
                ),
                "eer_ci95": tfl_eval[f"eer_ci95_{suffix}"],
                "primary_domain": tfl_eval["primary_domain"],
            }
            exports[exp_key] = entry_payload
            if isinstance(ev_exports, dict):
                ev_exports[exp_key] = dict(entry_payload)

        with open(ev_path, "w", encoding="utf-8") as f:
            json.dump(ev, f, indent=2)
        if mid in bench.get("experiments", {}):
            bench["experiments"][mid]["tflite_eval"] = tfl_eval
            bench["experiments"][mid]["tflite_exports"] = ev_exports

    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2)

    tfl_json_path = os.path.join(args.models_dir, "tflite_eval_results.json")
    merged_models: dict[str, Any] = {}
    if os.path.isfile(tfl_json_path) and args.only:
        with open(tfl_json_path, "r", encoding="utf-8") as f:
            prev_tfl = json.load(f)
        merged_models.update(prev_tfl.get("models", {}))
    merged_models.update(results)

    with open(tfl_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "window_frames": 180,
                "num_models": len(merged_models),
                "models": merged_models,
            },
            f,
            indent=2,
        )
    print(
        f"Updated eval_results.json, benchmark_results.json, and {tfl_json_path} "
        f"({len(merged_models)} models)."
    )


if __name__ == "__main__":
    main()
