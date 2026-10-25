#!/usr/bin/env python3
"""Converged multi-seed stability, matched-data control, and own-schedule horizon (Phase 2).

Runs three experiments requested by the reviewers at converged Phase-2 step budgets:

1. **Multi-seed stability** (3 seeds: 0, 1, 2) for:
   - `univ_conformer_small_ge2e` (`4,600` steps, `15.58` epochs)
   - `ls_conformer_small_ge2e` (`2,400` steps, `13.27` epochs)
   (plus `univ_ecapa_tdnn_small_ge2e` at `4,600` steps in `multi_seed_stability`).
   Reports per-seed metrics plus mean / std / min / max across seeds.

2. **Utterance-budget matched universal vs. single-domain control**: trains
   `univ_conformer_small_ge2e` on a deterministic exact 5,788-utterance subsample
   of the universal split (matching the LibriSpeech training utterance count of
   5,788 across all 381 speakers, ~15.2 utterances/speaker, seed 42) at both:
   - `2,400` steps (`13.27` epochs, matching `ls_conformer_small_ge2e` in both
     total utterances and gradient steps)
   - `4,600` steps (`25.43` epochs, matching `univ_conformer_small_ge2e` in
     gradient steps)

3. **Own-schedule horizon ablation**: trains `univ_conformer_small_ge2e` (seed 42)
   for `400`, `800`, `1,200`, `2,400`, and `4,600` steps where each run uses its
   *own* cosine decay horizon (`decay_steps = num_steps`, `warmup_steps = num_steps // 10`).

Writes `pretrained_models/r2_seed_and_control_results.json`.
"""

import argparse
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

from flaxspeaker import configs  # noqa: E402
from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import neural_net  # noqa: E402
from scripts import run_research_benchmarks as rrb  # noqa: E402


DEFAULT_DATA_ROOT = rrb.DEFAULT_DATA_ROOT
DEFAULT_OUTPUT_ROOT = rrb.DEFAULT_OUTPUT_ROOT
OUT_PATH = os.path.join(DEFAULT_OUTPUT_ROOT, "r2_seed_and_control_results.json")
CACHE_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "_ablation_cache")


def _summarise(md_eval: dict[str, Any]) -> dict[str, float]:
    """Extract canonical flat EER summary dict from a `multi_domain_eval` block."""
    d1 = md_eval["domains_1utt"]
    d3 = md_eval.get("domains_3utt", {})
    out = {
        "ls_1utt_eer": d1["librispeech"]["eer_percent"],
        "cn_1utt_eer": d1["cnceleb2"]["eer_percent"],
        "vox_1utt_eer": d1["voxceleb1"]["eer_percent"],
        "macro_1utt_eer": md_eval["universal_macro_avg_1utt"]["eer_percent"],
        "pooled_1utt_eer": d1["universal_pooled"]["eer_percent"],
    }
    if d3 and "universal_pooled" in d3:
        out["macro_3utt_eer"] = md_eval["universal_macro_avg_3utt"]["eer_percent"]
        out["pooled_3utt_eer"] = d3["universal_pooled"]["eer_percent"]
    return out


def _summarise_from_cached(rec: dict[str, Any]) -> dict[str, float]:
    """Extract canonical `summary` from either `multi_domain_eval` or `eer_*_percent`."""
    if "summary" in rec and isinstance(rec["summary"], dict) and "pooled_1utt_eer" in rec["summary"]:
        return rec["summary"]
    if "multi_domain_eval" in rec:
        return _summarise(rec["multi_domain_eval"])
    e1 = rec["eer_1utt_percent"]
    e3 = rec.get("eer_3utt_percent", {})
    out = {
        "ls_1utt_eer": e1["librispeech"],
        "cn_1utt_eer": e1["cnceleb2"],
        "vox_1utt_eer": e1["voxceleb1"],
        "macro_1utt_eer": e1["macro_avg"],
        "pooled_1utt_eer": e1["universal_pooled"],
    }
    if "universal_pooled" in e3:
        out["macro_3utt_eer"] = e3["macro_avg"]
        out["pooled_3utt_eer"] = e3["universal_pooled"]
    return out


def _mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute mean, sample std (`ddof=1`), min, and max across seed summary dicts."""
    keys = rows[0].keys()
    return {
        k: {
            "mean": round(float(np.mean([r[k] for r in rows])), 3),
            "std": (
                round(float(np.std([r[k] for r in rows], ddof=1)), 3)
                if len(rows) > 1
                else 0.0
            ),
            "min": round(float(np.min([r[k] for r in rows])), 2),
            "max": round(float(np.max([r[k] for r in rows])), 2),
        }
        for k in keys
    }


def _load_env(data_root: str = DEFAULT_DATA_ROOT):
    summary_path = os.path.join(data_root, "splits", "dataset_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    fe_cfg = configs.FrontendConfig(
        feature_type="log_mel", sample_rate=16000, n_mels=80, apply_cmvn=True
    )
    fe = feature_extraction.AudioFrontend(fe_cfg, configs.VadConfig(mode="energy"))
    caches: dict[str, feature_extraction.FeatureCache] = {}
    spk_train: dict[str, dataset.SpkToUtts] = {}
    trials: dict[str, list[dataset.TrialPair]] = {}
    multi_trials: dict[str, list[dataset.MultiEnrollTrial]] = {}
    for d in ("librispeech", "cnceleb2", "voxceleb1"):
        spk_train[d] = dataset.get_csv_spk_to_utts(ds[d]["train_csv"])
        spk_eval = dataset.get_csv_spk_to_utts(ds[d]["eval_csv"])
        trials[d] = dataset.load_trials_file(ds[d]["trials_file"])
        multi_trials[d] = rrb.load_multi_enroll_trials(ds[d]["multi_trials_file"])
        c = feature_extraction.FeatureCache(
            fe, cache_dir=os.path.join(data_root, "feature_cache", d)
        )
        c.precompute_dataset(spk_eval, num_workers=8)
        c.precompute_dataset(spk_train[d], num_workers=8)
        caches[d] = c
    spk_train["universal"] = dataset.get_csv_spk_to_utts(ds["universal"]["train_csv"])
    trials["universal"] = dataset.load_trials_file(ds["universal"]["trials_file"])
    multi_trials["universal"] = rrb.load_multi_enroll_trials(
        ds["universal"]["multi_trials_file"]
    )
    caches["universal"] = feature_extraction.FeatureCache.merged(
        fe,
        [caches[d] for d in ("librispeech", "cnceleb2", "voxceleb1")],
        cache_dir=os.path.join(data_root, "feature_cache", "universal"),
    )
    return ds, fe, caches, spk_train, trials, multi_trials


def _train_and_eval(
    cfg: configs.ExperimentConfig,
    spk_map: dataset.SpkToUtts,
    train_cache: feature_extraction.FeatureCache,
    caches: dict[str, feature_extraction.FeatureCache],
    trials: dict[str, list[dataset.TrialPair]],
    multi_trials: dict[str, list[dataset.MultiEnrollTrial]],
    fe: feature_extraction.AudioFrontend,
    init_seed: int,
    label: str = "",
) -> dict[str, Any]:
    t0 = time.time()
    cfg.train.seed = int(init_seed)
    _, state = neural_net.get_speaker_encoder(cfg, num_classes=0)
    step_fn = neural_net.make_modern_train_step(cfg)
    rng = np.random.default_rng(int(init_seed))
    losses_list: list[float] = []

    def _sample():
        return feature_extraction.get_batched_nm_input(
            spk_map,
            num_speakers=cfg.train.loss.num_speakers_per_batch,
            num_utts_per_speaker=cfg.train.loss.num_utts_per_speaker,
            seq_len=cfg.model.seq_len,
            frontend=fe,
            specaug_config=cfg.train.specaug,
            waveform_aug_config=cfg.train.waveform_aug,
            cache=train_cache,
            rng=rng,
            min_seq_len=cfg.model.min_seq_len,
            variable_length=bool(cfg.model.variable_length_training),
            return_mask=True,
            within_domain_prob=cfg.data.within_domain_prob,
            domain_temp_alpha=cfg.data.domain_temp_alpha,
        )

    cur_batch = _sample()
    for step in range(cfg.train.num_steps):
        x, y, m = cur_batch
        state, loss_val = step_fn(state, x, y, m)
        if step + 1 < cfg.train.num_steps:
            cur_batch = _sample()
        loss_s = float(loss_val.item())
        losses_list.append(loss_s)
        if (step + 1) % 400 == 0 or step == 0 or step + 1 == cfg.train.num_steps:
            mean100 = float(np.mean(losses_list[-min(len(losses_list), 100):]))
            print(
                f"  [{label}] step {step + 1}/{cfg.train.num_steps} "
                f"loss={loss_s:.4f} (mean100={mean100:.4f})",
                flush=True,
            )

    train_time_sec = round(time.time() - t0, 2)
    md_eval = rrb.evaluate_all_domains(
        state, cfg, caches, trials, multi_trials, shared_embs={}
    )
    d1 = md_eval["domains_1utt"]
    d3 = md_eval["domains_3utt"]
    out_1utt = {
        k: d1[k]["eer_percent"]
        for k in ("librispeech", "cnceleb2", "voxceleb1", "universal_pooled")
    }
    out_1utt["macro_avg"] = md_eval["universal_macro_avg_1utt"]["eer_percent"]
    out_3utt = {
        k: d3[k]["eer_percent"]
        for k in ("librispeech", "cnceleb2", "voxceleb1", "universal_pooled")
    }
    out_3utt["macro_avg"] = md_eval["universal_macro_avg_3utt"]["eer_percent"]
    summ = _summarise(md_eval)
    return {
        "seed": int(init_seed),
        "num_steps": int(cfg.train.num_steps),
        "warmup_steps": int(cfg.train.warmup_steps),
        "initial_loss": round(losses_list[0], 4),
        "final_loss": round(
            float(np.mean(losses_list[-min(len(losses_list), 100):])), 4
        ),
        "train_time_sec": train_time_sec,
        "multi_domain_eval": md_eval,
        "summary": summ,
        "eer_1utt_percent": out_1utt,
        "eer_3utt_percent": out_3utt,
        "eer_percent": out_1utt,
    }


def subsample_spk_map(
    spk_map: dataset.SpkToUtts, target_total_utts: int, seed: int = 42
) -> dataset.SpkToUtts:
    """Subsample utterances across speakers via water-filling to total EXACTLY `target_total_utts`."""
    rng = np.random.default_rng(seed)
    spks = sorted(spk_map.keys())
    caps = {s: len(spk_map[s]) for s in spks}
    total_avail = sum(caps.values())
    if target_total_utts > total_avail:
        raise ValueError(
            f"target_total_utts={target_total_utts} exceeds available={total_avail}"
        )
    alloc = {s: 0 for s in spks}
    remaining = int(target_total_utts)
    active = [s for s in spks if caps[s] > 0]
    while remaining > 0 and active:
        q, r = divmod(remaining, len(active))
        if q > 0:
            for s in active:
                add = min(q, caps[s] - alloc[s])
                alloc[s] += add
                remaining -= add
            active = [s for s in active if alloc[s] < caps[s]]
        else:
            chosen = set(rng.choice(active, size=r, replace=False).tolist())
            for s in active:
                if s in chosen:
                    alloc[s] += 1
                    remaining -= 1
            break

    out: dataset.SpkToUtts = {}
    for s in spks:
        utts = list(spk_map[s])
        k = alloc[s]
        if k >= 2:
            idx = rng.choice(len(utts), size=k, replace=False)
            out[s] = [utts[int(i)] for i in sorted(idx)]
    actual_total = sum(len(v) for v in out.values())
    if actual_total != target_total_utts:
        raise RuntimeError(
            f"Subsample produced {actual_total} utterances, expected {target_total_utts}"
        )
    return out


ALL_R2_TASKS = [
    # 1. Multi-seed stability (seeds 0, 1, 2)
    "seed_univ_conformer_small_ge2e_s0",
    "seed_univ_conformer_small_ge2e_s1",
    "seed_univ_conformer_small_ge2e_s2",
    "seed_univ_ecapa_tdnn_small_ge2e_s0",
    "seed_univ_ecapa_tdnn_small_ge2e_s1",
    "seed_univ_ecapa_tdnn_small_ge2e_s2",
    "seed_ls_conformer_small_ge2e_s0",
    "seed_ls_conformer_small_ge2e_s1",
    "seed_ls_conformer_small_ge2e_s2",
    # 2. Exact 5,788-utt universal subsample controls (seed 42; 2,400 steps & 4,600 steps)
    "matched_exact5788_2400steps",
    "matched_exact5788_4600steps",
    # 3. Own-schedule horizon runs (seed 42; 400, 800, 1200, 2400; 4600 reuses univ_conformer_small_ge2e)
    "own_horizon_s42_400",
    "own_horizon_s42_800",
    "own_horizon_s42_1200",
    "own_horizon_s42_2400",
]


def run_single_r2_task(
    task_name: str,
    data_root: str = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Execute one atomic R2 control task and save its JSON to `_ablation_cache`."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"r2_{task_name}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"[R2 Control] Skipping {task_name} (already cached)")
        return cached

    ds, fe, caches, spk_train, trials, multi_trials = _load_env(data_root)

    if task_name.startswith("seed_"):
        prefix, seed_part = task_name.rsplit("_s", 1)
        mid = prefix[len("seed_"):]
        s = int(seed_part)
        spec_map = {
            "univ_conformer_small_ge2e": ("universal", "conformer", "atp", 4600),
            "univ_ecapa_tdnn_small_ge2e": ("universal", "ecapa_tdnn", "asp", 4600),
            "ls_conformer_small_ge2e": ("librispeech", "conformer", "atp", 2400),
        }
        dname, bb, pool, n_steps = spec_map[mid]
        cfg = rrb.make_exp_config(
            dname,
            ds[dname]["train_csv"],
            ds[dname]["eval_csv"],
            ds[dname]["trials_file"],
            bb,
            "small",
            "ge2e_softmax",
            "cosine",
            pool,
            num_steps=n_steps,
        )
        print(f"\n[R2 Seed] {mid} seed={s} steps={n_steps} ...", flush=True)
        res = _train_and_eval(
            cfg,
            spk_train[dname],
            caches[dname],
            caches,
            trials,
            multi_trials,
            fe,
            init_seed=s,
            label=task_name,
        )
    elif task_name in ("matched_exact5788_2400steps", "matched_exact5788_4600steps"):
        n_steps = 2400 if task_name.endswith("2400steps") else 4600
        target_utts = sum(len(u) for u in spk_train["librispeech"].values())
        sub_univ = subsample_spk_map(spk_train["universal"], target_utts, seed=42)
        actual_utts = sum(len(u) for u in sub_univ.values())
        cfg = rrb.make_exp_config(
            "universal",
            ds["universal"]["train_csv"],
            ds["universal"]["eval_csv"],
            ds["universal"]["trials_file"],
            "conformer",
            "small",
            "ge2e_softmax",
            "cosine",
            "atp",
            num_steps=n_steps,
        )
        print(
            f"\n[R2 Matched-Data] universal subsample ({actual_utts} utts / {len(sub_univ)} spks) "
            f"steps={n_steps} seed=42 ...",
            flush=True,
        )
        res = _train_and_eval(
            cfg,
            sub_univ,
            caches["universal"],
            caches,
            trials,
            multi_trials,
            fe,
            init_seed=42,
            label=task_name,
        )
        res["num_train_utterances"] = actual_utts
        res["total_train_utterances"] = actual_utts
        res["num_train_speakers"] = len(sub_univ)
        res["target_num_utterances"] = target_utts
    elif task_name.startswith("own_horizon_s42_"):
        n_steps = int(task_name.split("_")[-1])
        cfg = rrb.make_exp_config(
            "universal",
            ds["universal"]["train_csv"],
            ds["universal"]["eval_csv"],
            ds["universal"]["trials_file"],
            "conformer",
            "small",
            "ge2e_softmax",
            "cosine",
            "atp",
            num_steps=n_steps,
        )
        print(
            f"\n[R2 Own-Schedule Horizon] num_steps={n_steps} seed=42 ...", flush=True
        )
        res = _train_and_eval(
            cfg,
            spk_train["universal"],
            caches["universal"],
            caches,
            trials,
            multi_trials,
            fe,
            init_seed=42,
            label=task_name,
        )
    else:
        raise ValueError(f"Unknown R2 task: {task_name}")

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def _from_released(mid: str) -> dict[str, Any]:
    p = os.path.join(DEFAULT_OUTPUT_ROOT, mid, "eval_results.json")
    with open(p, "r", encoding="utf-8") as f:
        rec = json.load(f)
    md = rec["multi_domain_eval"]
    d1 = md["domains_1utt"]
    d3 = md.get("domains_3utt", {})
    e1 = {
        k: d1[k]["eer_percent"]
        for k in ("librispeech", "cnceleb2", "voxceleb1", "universal_pooled")
    }
    e1["macro_avg"] = md["universal_macro_avg_1utt"]["eer_percent"]
    e3 = {
        k: d3[k]["eer_percent"]
        for k in ("librispeech", "cnceleb2", "voxceleb1", "universal_pooled")
        if k in d3
    }
    if "universal_macro_avg_3utt" in md:
        e3["macro_avg"] = md["universal_macro_avg_3utt"]["eer_percent"]
    return {
        "seed": 42,
        "num_steps": rec["num_steps"],
        "warmup_steps": rec.get("warmup_steps", max(10, rec["num_steps"] // 10)),
        "initial_loss": rec["initial_loss"],
        "final_loss": rec["final_loss"],
        "train_time_sec": rec["train_time_sec"],
        "multi_domain_eval": md,
        "summary": _summarise(md),
        "eer_1utt_percent": e1,
        "eer_3utt_percent": e3,
        "eer_percent": e1,
    }


def assemble_r2_results(output_path: str = OUT_PATH) -> dict[str, Any]:
    """Assemble `pretrained_models/r2_seed_and_control_results.json` from cached R2 tasks."""
    cached: dict[str, Any] = {}
    for tname in ALL_R2_TASKS:
        cpath = os.path.join(CACHE_DIR, f"r2_{tname}.json")
        if os.path.exists(cpath):
            with open(cpath, "r", encoding="utf-8") as f:
                cached[tname] = json.load(f)

    seed_noise_floor: dict[str, Any] = {}
    multi_seed: dict[str, Any] = {}
    for mid, n_steps in (
        ("univ_conformer_small_ge2e", 4600),
        ("univ_ecapa_tdnn_small_ge2e", 4600),
        ("ls_conformer_small_ge2e", 2400),
    ):
        runs: list[dict[str, Any]] = []
        per_seed_dict: dict[str, Any] = {}
        for s in (0, 1, 2):
            key = f"seed_{mid}_s{s}"
            if key in cached:
                rec_s = dict(cached[key])
                rec_s["summary"] = _summarise_from_cached(rec_s)
                runs.append(rec_s)
                per_seed_dict[str(s)] = rec_s

        summary_1utt: dict[str, Any] = {}
        summary_3utt: dict[str, Any] = {}
        if runs:
            for k in runs[0]["eer_1utt_percent"]:
                vals1 = [r["eer_1utt_percent"][k] for r in runs]
                summary_1utt[k] = {
                    "mean": round(float(np.mean(vals1)), 2),
                    "std": (
                        round(float(np.std(vals1, ddof=1)), 2)
                        if len(vals1) > 1
                        else 0.0
                    ),
                    "values": vals1,
                }
            for k in runs[0]["eer_3utt_percent"]:
                vals3 = [r["eer_3utt_percent"][k] for r in runs]
                summary_3utt[k] = {
                    "mean": round(float(np.mean(vals3)), 2),
                    "std": (
                        round(float(np.std(vals3, ddof=1)), 2)
                        if len(vals3) > 1
                        else 0.0
                    ),
                    "values": vals3,
                }
        multi_seed[mid] = {
            "num_steps": n_steps,
            "per_seed": runs,
            "summary": summary_1utt,
            "summary_1utt": summary_1utt,
            "summary_3utt": summary_3utt,
        }
        if mid in ("univ_conformer_small_ge2e", "ls_conformer_small_ge2e") and per_seed_dict:
            rows = [v["summary"] for v in per_seed_dict.values()]
            seed_noise_floor[mid] = {
                "num_steps": n_steps,
                "seeds": sorted(int(k) for k in per_seed_dict.keys()),
                "per_seed": per_seed_dict,
                "mean_std_over_seeds": _mean_std(rows),
            }

    matched_2400 = cached.get("matched_exact5788_2400steps", {})
    matched_4600 = cached.get("matched_exact5788_4600steps", {})
    ls_rel = _from_released("ls_conformer_small_ge2e")
    univ_rel = _from_released("univ_conformer_small_ge2e")

    own_horizon: dict[str, Any] = {}
    for n_steps in (400, 800, 1200, 2400):
        k = f"own_horizon_s42_{n_steps}"
        if k in cached:
            own_horizon[f"steps_{n_steps}"] = cached[k]
    own_horizon["steps_4600"] = univ_rel

    h800_block: dict[str, Any] = {}
    if "own_horizon_s42_800" in cached:
        h800_block["steps_800_own_schedule"] = cached["own_horizon_s42_800"]

    results = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed_noise_floor": seed_noise_floor,
        "training_horizon_800_own_schedule": h800_block,
        "multi_seed_stability": multi_seed,
        "matched_data_control": {
            "description": (
                "Universal Conformer-Small GE2E trained on an exact 5,788-utterance "
                "subsample (381 speakers, ~15.2 utts/speaker, seed 42) matching the "
                "LibriSpeech total utterance count, evaluated at both 2,400 steps "
                "(matched steps & 13.27 epochs to ls_conformer_small_ge2e) and "
                "4,600 steps (matched steps to univ_conformer_small_ge2e)."
            ),
            "univ_subsampled_5788": matched_2400,
            "univ_subsampled_5788_2400steps": matched_2400,
            "univ_subsampled_5788_4600steps": matched_4600,
            "univ_conformer_small_ge2e_matched_5788utts": matched_2400,
            "univ_conformer_small_ge2e_matched_5788utts_2400steps": matched_2400,
            "univ_conformer_small_ge2e_matched_5788utts_4600steps": matched_4600,
            "ls_conformer_small_ge2e_5788utts": {
                "num_train_utterances": 5788,
                "total_train_utterances": 5788,
                "num_train_speakers": 232,
                **ls_rel,
            },
            "univ_conformer_small_ge2e_9447utts": {
                "num_train_utterances": 9447,
                "total_train_utterances": 9447,
                "num_train_speakers": 381,
                **univ_rel,
            },
        },
        "own_schedule_horizon": own_horizon,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Assembled R2 controls -> {output_path}", flush=True)
    return results


def main(data_root: str = DEFAULT_DATA_ROOT, output_path: str = OUT_PATH):
    for tname in ALL_R2_TASKS:
        run_single_r2_task(tname, data_root=data_root)
    return assemble_r2_results(output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_path", type=str, default=OUT_PATH)
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional single R2 task name from ALL_R2_TASKS",
    )
    parser.add_argument(
        "--assemble_only",
        action="store_true",
        help="Assemble r2_seed_and_control_results.json from cached tasks and main benchmarks",
    )
    args = parser.parse_args()
    if args.assemble_only:
        assemble_r2_results(output_path=args.output_path)
    elif args.task:
        run_single_r2_task(args.task, data_root=args.data_root)
    else:
        main(data_root=args.data_root, output_path=args.output_path)
