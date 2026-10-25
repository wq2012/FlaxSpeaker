#!/usr/bin/env python3
"""Run converged empirical ablation studies for FlaxSpeaker (Phase 2).

Executes both:
- The **5 canonical paper/report ablation studies** at converged Phase-2 step budgets
  (`4,600` steps on Universal, `2,400` steps on LibriSpeech; seed 42):
  1. `domain_mixing_ablation` (`p_within_0.0`, `p_within_0.75`, `p_within_1.0`)
  2. `extended_set_and_dr_ablation` (`ext_softmax_pure_eq3`, `ext_softmax_blended`,
     `dr_vectors_3term_pure_eq7`, `dr_vectors_1term_blended`)
  3. `variable_length_ablation` (`variable_length_true`, `variable_length_false`,
     both full-sequence and 1.0 s / 100-frame evaluation)
  4. `training_horizon_ablation` (`steps_400`, `steps_800`, `steps_1200`,
     `steps_2400`, `steps_4600` on `univ_conformer_small_ge2e`)
  5. `mamba_directionality_ablation` (`ls_mamba_small_ge2e` unidirectional vs.
     bidirectional on LibriSpeech at `2,400` steps)
- Plus the **7 extended Universal architectural/frontend/preset studies** (`studies`
  section in `pretrained_models/ablation_results.json`) at `4,600` steps.

Saves all measured metrics to `pretrained_models/ablation_results.json`.
"""

import argparse
import copy
import datetime
import json
import os
import sys
import time
from typing import Any, Callable, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import configs  # noqa: E402
from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402
from flaxspeaker import neural_net  # noqa: E402
from scripts import run_research_benchmarks as rrb  # noqa: E402


DEFAULT_DATA_ROOT = rrb.DEFAULT_DATA_ROOT
DEFAULT_OUTPUT_ROOT = rrb.DEFAULT_OUTPUT_ROOT
DEFAULT_NUM_STEPS = 4600
DEFAULT_LS_NUM_STEPS = 2400
CACHE_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "_ablation_cache")


def run_single_config(
    exp_cfg: configs.ExperimentConfig,
    spk_train: dataset.SpkToUtts,
    frontend: feature_extraction.AudioFrontend,
    train_cache: feature_extraction.FeatureCache,
    caches: dict[str, feature_extraction.FeatureCache],
    trials_sets: dict[str, list[dataset.TrialPair]],
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]],
    label: str = "ablation",
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Train one `ExperimentConfig` and return `(eval_dict, state, shared_embs)`."""
    t0 = time.time()
    _, state = neural_net.get_speaker_encoder(exp_cfg, num_classes=0)
    step_fn = neural_net.make_modern_train_step(exp_cfg)
    rng = np.random.default_rng(exp_cfg.train.seed)
    losses: list[float] = []
    var_len = bool(getattr(exp_cfg.model, "variable_length_training", True))
    within_prob = float(getattr(exp_cfg.data, "within_domain_prob", 0.75))
    temp_alpha = float(getattr(exp_cfg.data, "domain_temp_alpha", 0.35))

    def _sample():
        return feature_extraction.get_batched_nm_input(
            spk_train,
            num_speakers=exp_cfg.train.loss.num_speakers_per_batch,
            num_utts_per_speaker=exp_cfg.train.loss.num_utts_per_speaker,
            seq_len=exp_cfg.model.seq_len,
            frontend=frontend,
            specaug_config=exp_cfg.train.specaug,
            waveform_aug_config=exp_cfg.train.waveform_aug,
            cache=train_cache,
            rng=rng,
            min_seq_len=exp_cfg.model.min_seq_len,
            variable_length=var_len,
            return_mask=True,
            within_domain_prob=within_prob,
            domain_temp_alpha=temp_alpha,
        )

    cur_batch = _sample()
    for step in range(exp_cfg.train.num_steps):
        batch_input, batch_labels, batch_mask = cur_batch
        state, loss_val = step_fn(state, batch_input, batch_labels, batch_mask)
        if step + 1 < exp_cfg.train.num_steps:
            cur_batch = _sample()
        losses.append(float(loss_val.item()))
        if (step + 1) % 400 == 0 or step == 0 or step == exp_cfg.train.num_steps - 1:
            mean100 = float(np.mean(losses[-min(len(losses), 100):]))
            print(
                f"  [{label}] step {step + 1}/{exp_cfg.train.num_steps} "
                f"loss={losses[-1]:.4f} (mean100={mean100:.4f})",
                flush=True,
            )

    train_time = round(time.time() - t0, 2)
    shared_embs: dict[str, Any] = {}
    md_eval = rrb.evaluate_all_domains(
        state, exp_cfg, caches, trials_sets, multi_trials_sets, shared_embs=shared_embs
    )
    return {
        "num_steps": int(exp_cfg.train.num_steps),
        "train_time_sec": train_time,
        "initial_loss": round(losses[0], 4),
        "final_loss": round(float(np.mean(losses[-min(len(losses), 100):])), 4),
        "final_loss_last_20": round(float(np.mean(losses[-min(len(losses), 20):])), 4),
        "multi_domain_eval": md_eval,
    }, state, shared_embs


def eval_short_duration(
    state: Any,
    exp_cfg: configs.ExperimentConfig,
    caches: dict[str, feature_extraction.FeatureCache],
    trials_sets: dict[str, list[dataset.TrialPair]],
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]],
    max_frames: int = 100,
) -> dict[str, Any]:
    """Evaluate all domains with utterance features truncated to `max_frames` (1.0 s)."""
    short_cfg = copy.deepcopy(exp_cfg)
    short_cfg.model.seq_len = max_frames
    short_cfg.model.max_eval_seq_len = max_frames
    short_cfg.eval.full_sequence_inference = False
    return rrb.evaluate_all_domains(
        state, short_cfg, caches, trials_sets, multi_trials_sets, shared_embs={}
    )


def _ensure_multi_domain_eval(rec: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Ensure an ablation record contains a full `multi_domain_eval` block."""
    if not isinstance(rec, dict):
        return rec
    if "multi_domain_eval" in rec:
        return rec
    if (
        "per_domain_1utt" in rec
        and "per_domain_3utt" in rec
        and "in_domain_1utt" in rec
        and "in_domain_3utt" in rec
    ):
        d1 = dict(rec["per_domain_1utt"])
        d1["universal_pooled"] = rec["in_domain_1utt"]
        d3 = dict(rec["per_domain_3utt"])
        d3["universal_pooled"] = rec["in_domain_3utt"]
        rec = dict(rec)
        rec["multi_domain_eval"] = {
            "domains_1utt": d1,
            "domains_3utt": d3,
            "universal_macro_avg_1utt": rec["macro_avg_1utt"],
            "universal_macro_avg_3utt": rec["macro_avg_3utt"],
        }
    return rec


def train_and_eval_ablation(
    exp_name: str,
    exp_cfg: configs.ExperimentConfig,
    spk_train: dataset.SpkToUtts,
    train_cache: feature_extraction.FeatureCache,
    eval_caches: dict[str, feature_extraction.FeatureCache],
    trials_sets: dict[str, list[dataset.TrialPair]],
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]],
    frontend: feature_extraction.AudioFrontend,
) -> dict[str, Any]:
    print(
        f"\n[Ablation] Running {exp_name} (steps={exp_cfg.train.num_steps}, "
        f"warmup={exp_cfg.train.warmup_steps})...",
        flush=True,
    )
    res_core, state, _ = run_single_config(
        exp_cfg,
        spk_train,
        frontend,
        train_cache,
        eval_caches,
        trials_sets,
        multi_trials_sets,
        label=exp_name,
    )
    num_params = neural_net.count_parameters(state.params)
    md_eval = res_core["multi_domain_eval"]
    pooled_1 = md_eval["domains_1utt"]["universal_pooled"]
    pooled_3 = md_eval["domains_3utt"]["universal_pooled"]
    macro_1 = md_eval["universal_macro_avg_1utt"]
    macro_3 = md_eval["universal_macro_avg_3utt"]
    n_train_utts = max(sum(len(u) for u in spk_train.values()), 1)
    print(
        f"  [{exp_name}] Done in {res_core['train_time_sec']:.1f}s | "
        f"Pooled 1-Utt EER={pooled_1['eer_percent']:.2f}%, "
        f"3-Utt EER={pooled_3['eer_percent']:.2f}%, "
        f"Macro 1-Utt EER={macro_1['eer_percent']:.2f}%",
        flush=True,
    )
    return {
        "experiment_name": exp_name,
        "num_parameters": num_params,
        "num_steps": int(exp_cfg.train.num_steps),
        "train_epochs": round(
            (exp_cfg.train.num_steps * exp_cfg.train.batch_size) / n_train_utts, 2
        ),
        "train_time_sec": res_core["train_time_sec"],
        "initial_loss": res_core["initial_loss"],
        "final_loss": res_core["final_loss"],
        "multi_domain_eval": md_eval,
        "in_domain_1utt": pooled_1,
        "in_domain_3utt": pooled_3,
        "macro_avg_1utt": macro_1,
        "macro_avg_3utt": macro_3,
        "per_domain_1utt": {
            k: md_eval["domains_1utt"][k]
            for k in ("librispeech", "cnceleb2", "voxceleb1")
        },
        "per_domain_3utt": {
            k: md_eval["domains_3utt"][k]
            for k in ("librispeech", "cnceleb2", "voxceleb1")
        },
    }


def _load_main_eval(model_id: str) -> Optional[dict[str, Any]]:
    eval_path = os.path.join(DEFAULT_OUTPUT_ROOT, model_id, "eval_results.json")
    if not os.path.exists(eval_path):
        return None
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_from_main_benchmark(
    model_id: str, exp_name_override: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Extract an ablation record from a completed Phase-2 main benchmark model."""
    rec = _load_main_eval(model_id)
    if rec is None:
        return None
    md = rec.get("multi_domain_eval", {})
    if "domains_1utt" not in md or "domains_3utt" not in md:
        return None
    return {
        "experiment_name": exp_name_override or model_id,
        "source_checkpoint": f"pretrained_models/{model_id}",
        "num_parameters": rec["num_parameters"],
        "num_steps": rec["num_steps"],
        "train_epochs": rec.get(
            "train_epochs", round((rec["num_steps"] * 32) / 9447.0, 2)
        ),
        "train_time_sec": rec["train_time_sec"],
        "initial_loss": rec["initial_loss"],
        "final_loss": rec["final_loss"],
        "multi_domain_eval": md,
        "in_domain_1utt": md["domains_1utt"]["universal_pooled"],
        "in_domain_3utt": md["domains_3utt"]["universal_pooled"],
        "macro_avg_1utt": md["universal_macro_avg_1utt"],
        "macro_avg_3utt": md["universal_macro_avg_3utt"],
        "per_domain_1utt": {
            k: md["domains_1utt"][k]
            for k in ("librispeech", "cnceleb2", "voxceleb1")
        },
        "per_domain_3utt": {
            k: md["domains_3utt"][k]
            for k in ("librispeech", "cnceleb2", "voxceleb1")
        },
    }


def _load_data_and_caches(data_root: str = DEFAULT_DATA_ROOT):
    splits_summary_path = os.path.join(data_root, "splits", "dataset_summary.json")
    with open(splits_summary_path, "r", encoding="utf-8") as f:
        ds_summary = json.load(f)

    frontend_cfg = configs.FrontendConfig(
        feature_type=configs.FeatureType.LOG_MEL.value,
        sample_rate=16000,
        n_mels=80,
        apply_cmvn=True,
    )
    vad_cfg = configs.VadConfig(mode=configs.VadMode.ENERGY.value)
    frontend = feature_extraction.AudioFrontend(frontend_cfg, vad_cfg)

    caches: dict[str, feature_extraction.FeatureCache] = {}
    spk_train_sets: dict[str, dataset.SpkToUtts] = {}
    spk_eval_sets: dict[str, dataset.SpkToUtts] = {}
    trials_sets: dict[str, list[dataset.TrialPair]] = {}
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]] = {}

    for dname in ("librispeech", "cnceleb2", "voxceleb1"):
        info = ds_summary[dname]
        spk_train_sets[dname] = dataset.get_csv_spk_to_utts(info["train_csv"])
        spk_eval_sets[dname] = dataset.get_csv_spk_to_utts(info["eval_csv"])
        trials_sets[dname] = dataset.load_trials_file(info["trials_file"])
        multi_trials_sets[dname] = rrb.load_multi_enroll_trials(
            info["multi_trials_file"]
        )
        c_dir = os.path.join(data_root, "feature_cache", dname)
        cache = feature_extraction.FeatureCache(frontend, cache_dir=c_dir)
        cache.precompute_dataset(spk_eval_sets[dname], num_workers=8)
        cache.precompute_dataset(spk_train_sets[dname], num_workers=8)
        caches[dname] = cache

    univ_info = ds_summary["universal"]
    spk_train_sets["universal"] = dataset.get_csv_spk_to_utts(univ_info["train_csv"])
    spk_eval_sets["universal"] = dataset.get_csv_spk_to_utts(univ_info["eval_csv"])
    trials_sets["universal"] = dataset.load_trials_file(univ_info["trials_file"])
    multi_trials_sets["universal"] = rrb.load_multi_enroll_trials(
        univ_info["multi_trials_file"]
    )

    univ_cache = feature_extraction.FeatureCache.merged(
        frontend,
        [caches[d] for d in ("librispeech", "cnceleb2", "voxceleb1")],
        cache_dir=os.path.join(data_root, "feature_cache", "universal"),
    )
    caches["universal"] = univ_cache
    return (
        ds_summary,
        frontend_cfg,
        frontend,
        caches,
        spk_train_sets,
        spk_eval_sets,
        trials_sets,
        multi_trials_sets,
    )


def _expected_steps_for_task(task_name: str, default_univ_steps: int) -> int:
    if task_name == "ls_mamba_bidirectional":
        return DEFAULT_LS_NUM_STEPS
    if task_name == "horizon_snapshots_full":
        return 2400
    return default_univ_steps


def run_single_ablation_task(
    task_name: str,
    data_root: str = DEFAULT_DATA_ROOT,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> dict[str, Any]:
    """Run one atomic R1 ablation task and cache its JSON result."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"r1_{task_name}.json")
    expected_steps = _expected_steps_for_task(task_name, num_steps)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if int(cached.get("num_steps", 0)) == int(expected_steps):
            print(
                f"[R1 Ablation] Skipping {task_name} "
                f"(already cached at {expected_steps} steps)"
            )
            return cached

    (
        ds_summary,
        frontend_cfg,
        frontend,
        caches,
        spk_train_sets,
        spk_eval_sets,
        trials_sets,
        multi_trials_sets,
    ) = _load_data_and_caches(data_root)
    univ_info = ds_summary["universal"]
    ls_info = ds_summary["librispeech"]
    spk_univ_train = spk_train_sets["universal"]

    if task_name in ("domain_mixing_p_within_0.0", "domain_mixing_p_within_1.0"):
        p_val = 0.0 if task_name.endswith("0.0") else 1.0
        cfg = rrb.make_exp_config(
            dataset_name="universal",
            train_csv=univ_info["train_csv"],
            eval_csv=univ_info["eval_csv"],
            trials_file=univ_info["trials_file"],
            backbone="conformer",
            size_variant="small",
            loss_type="ge2e_softmax",
            scoring_type="cosine",
            pooling_type="atp",
            num_steps=num_steps,
        )
        cfg.data.within_domain_prob = p_val
        res, _, _ = run_single_config(
            cfg,
            spk_univ_train,
            frontend,
            caches["universal"],
            caches,
            trials_sets,
            multi_trials_sets,
            label=task_name,
        )
        res["within_domain_prob"] = p_val
    elif task_name == "ext_softmax_pure_eq3":
        cfg = rrb.make_exp_config(
            dataset_name="universal",
            train_csv=univ_info["train_csv"],
            eval_csv=univ_info["eval_csv"],
            trials_file=univ_info["trials_file"],
            backbone="conformer",
            size_variant="small",
            loss_type="extended_set_softmax",
            scoring_type="cosine",
            pooling_type="atp",
            num_steps=num_steps,
        )
        cfg.train.loss.pure_extended_set = True
        res, _, _ = run_single_config(
            cfg,
            spk_univ_train,
            frontend,
            caches["universal"],
            caches,
            trials_sets,
            multi_trials_sets,
            label=task_name,
        )
        res["pure_extended_set"] = True
    elif task_name == "dr_vectors_3term_pure_eq7":
        cfg = rrb.make_exp_config(
            dataset_name="universal",
            train_csv=univ_info["train_csv"],
            eval_csv=univ_info["eval_csv"],
            trials_file=univ_info["trials_file"],
            backbone="conformer",
            size_variant="small",
            loss_type="extended_set_softmax",
            scoring_type="dr_vector",
            pooling_type="atp",
            num_steps=num_steps,
        )
        cfg.model.dr_vector.num_cosine_terms = 3
        cfg.train.loss.pure_extended_set = True
        res, _, _ = run_single_config(
            cfg,
            spk_univ_train,
            frontend,
            caches["universal"],
            caches,
            trials_sets,
            multi_trials_sets,
            label=task_name,
        )
        res["num_cosine_terms"] = 3
        res["pure_extended_set"] = True
    elif task_name == "variable_length_true_short_eval":
        model_dir = os.path.join(DEFAULT_OUTPUT_ROOT, "univ_conformer_small_ge2e")
        loaded_var = hf_compat.FlaxSpeakerModel.from_pretrained(model_dir)
        rec_075 = _load_main_eval("univ_conformer_small_ge2e")
        if rec_075 is None:
            raise FileNotFoundError(f"Missing eval_results.json in {model_dir}")
        short_var = eval_short_duration(
            loaded_var.state,
            loaded_var.exp_config,
            caches,
            trials_sets,
            multi_trials_sets,
            max_frames=100,
        )
        res = {
            "num_steps": int(rec_075["num_steps"]),
            "variable_length_training": True,
            "full_seq_eval": rec_075["multi_domain_eval"],
            "short_1s_eval": short_var,
        }
    elif task_name == "variable_length_false":
        cfg = rrb.make_exp_config(
            dataset_name="universal",
            train_csv=univ_info["train_csv"],
            eval_csv=univ_info["eval_csv"],
            trials_file=univ_info["trials_file"],
            backbone="conformer",
            size_variant="small",
            loss_type="ge2e_softmax",
            scoring_type="cosine",
            pooling_type="atp",
            num_steps=num_steps,
        )
        cfg.model.variable_length_training = False
        res_fixed, state_fixed, _ = run_single_config(
            cfg,
            spk_univ_train,
            frontend,
            caches["universal"],
            caches,
            trials_sets,
            multi_trials_sets,
            label=task_name,
        )
        short_fixed = eval_short_duration(
            state_fixed,
            cfg,
            caches,
            trials_sets,
            multi_trials_sets,
            max_frames=100,
        )
        res = {
            "num_steps": int(num_steps),
            "variable_length_training": False,
            "train_time_sec": res_fixed["train_time_sec"],
            "initial_loss": res_fixed["initial_loss"],
            "final_loss": res_fixed["final_loss"],
            "multi_domain_eval": res_fixed["multi_domain_eval"],
            "full_seq_eval": res_fixed["multi_domain_eval"],
            "short_1s_eval": short_fixed,
        }
    elif task_name == "horizon_snapshots_full":
        cfg = rrb.make_exp_config(
            dataset_name="universal",
            train_csv=univ_info["train_csv"],
            eval_csv=univ_info["eval_csv"],
            trials_file=univ_info["trials_file"],
            backbone="conformer",
            size_variant="small",
            loss_type="ge2e_softmax",
            scoring_type="cosine",
            pooling_type="atp",
            num_steps=num_steps,
        )
        t0 = time.time()
        _, state = neural_net.get_speaker_encoder(cfg, num_classes=0)
        step_fn = neural_net.make_modern_train_step(cfg)
        rng = np.random.default_rng(cfg.train.seed)
        losses: list[float] = []
        snaps: dict[str, Any] = {}
        max_snap_step = 2400

        def _sample_h():
            return feature_extraction.get_batched_nm_input(
                spk_univ_train,
                num_speakers=cfg.train.loss.num_speakers_per_batch,
                num_utts_per_speaker=cfg.train.loss.num_utts_per_speaker,
                seq_len=cfg.model.seq_len,
                frontend=frontend,
                specaug_config=cfg.train.specaug,
                waveform_aug_config=cfg.train.waveform_aug,
                cache=caches["universal"],
                rng=rng,
                min_seq_len=cfg.model.min_seq_len,
                variable_length=True,
                return_mask=True,
                within_domain_prob=cfg.data.within_domain_prob,
                domain_temp_alpha=cfg.data.domain_temp_alpha,
            )

        cur_batch = _sample_h()
        for step in range(max_snap_step):
            batch_input, batch_labels, batch_mask = cur_batch
            state, loss_val = step_fn(state, batch_input, batch_labels, batch_mask)
            if step + 1 < max_snap_step:
                cur_batch = _sample_h()
            losses.append(float(loss_val.item()))
            s_num = step + 1
            if s_num in (400, 800, 1200, 2400):
                elapsed = round(time.time() - t0, 2)
                md_h = rrb.evaluate_all_domains(
                    state, cfg, caches, trials_sets, multi_trials_sets, shared_embs={}
                )
                mean100 = round(
                    float(np.mean(losses[-min(len(losses), 100):])), 4
                )
                snaps[f"steps_{s_num}"] = {
                    "num_steps": s_num,
                    "epochs": round((s_num * cfg.train.batch_size) / 9447.0, 2),
                    "train_time_sec": elapsed,
                    "initial_loss": round(losses[0], 4),
                    "final_loss": mean100,
                    "loss_at_horizon": mean100,
                    "multi_domain_eval": md_h,
                    "in_domain_1utt": md_h["domains_1utt"]["universal_pooled"],
                    "in_domain_3utt": md_h["domains_3utt"]["universal_pooled"],
                    "macro_avg_1utt": md_h["universal_macro_avg_1utt"],
                    "macro_avg_3utt": md_h["universal_macro_avg_3utt"],
                }
                print(
                    f"  [horizon_snapshots_full] step {s_num} "
                    f"Pooled 1-Utt EER={md_h['domains_1utt']['universal_pooled']['eer_percent']:.2f}%",
                    flush=True,
                )
        res = {"num_steps": max_snap_step, "snapshots": snaps}
    elif task_name == "ls_mamba_bidirectional":
        cfg = rrb.make_exp_config(
            dataset_name="librispeech",
            train_csv=ls_info["train_csv"],
            eval_csv=ls_info["eval_csv"],
            trials_file=ls_info["trials_file"],
            backbone="mamba",
            size_variant="small",
            loss_type="ge2e_softmax",
            scoring_type="cosine",
            pooling_type="asp",
            num_steps=DEFAULT_LS_NUM_STEPS,
        )
        cfg.model.mamba.bidirectional = True
        res_mb, state_mb, _ = run_single_config(
            cfg,
            spk_train_sets["librispeech"],
            frontend,
            caches["librispeech"],
            caches,
            trials_sets,
            multi_trials_sets,
            label=task_name,
        )
        res = {
            "bidirectional": True,
            "num_parameters": neural_net.count_parameters(state_mb.params),
            "num_steps": DEFAULT_LS_NUM_STEPS,
            "train_epochs": round((DEFAULT_LS_NUM_STEPS * 32) / 5788.0, 2),
            "train_time_sec": res_mb["train_time_sec"],
            "initial_loss": res_mb["initial_loss"],
            "final_loss": res_mb["final_loss"],
            "multi_domain_eval": res_mb["multi_domain_eval"],
        }
    elif task_name == "mamba_bidirectional":
        cfg = rrb.make_exp_config(
            dataset_name="universal",
            train_csv=univ_info["train_csv"],
            eval_csv=univ_info["eval_csv"],
            trials_file=univ_info["trials_file"],
            backbone="mamba",
            size_variant="small",
            loss_type="ge2e_softmax",
            scoring_type="cosine",
            pooling_type="asp",
            num_steps=num_steps,
        )
        cfg.model.mamba.bidirectional = True
        res = train_and_eval_ablation(
            task_name,
            cfg,
            spk_univ_train,
            caches["universal"],
            caches,
            trials_sets,
            multi_trials_sets,
            frontend,
        )
    elif task_name == "ecapa_bn_style_after_act":
        cfg = rrb.make_exp_config(
            dataset_name="universal",
            train_csv=univ_info["train_csv"],
            eval_csv=univ_info["eval_csv"],
            trials_file=univ_info["trials_file"],
            backbone="ecapa_tdnn",
            size_variant="small",
            loss_type="ge2e_softmax",
            scoring_type="cosine",
            pooling_type="asp",
            num_steps=num_steps,
        )
        cfg.model.ecapa_tdnn.norm_after_activation = True
        res = train_and_eval_ablation(
            task_name,
            cfg,
            spk_univ_train,
            caches["universal"],
            caches,
            trials_sets,
            multi_trials_sets,
            frontend,
        )
    elif task_name in ("vad_26db_no_agc", "vad_20db_no_agc", "vad_26db_with_agc"):
        top_db = 20.0 if task_name == "vad_20db_no_agc" else 26.0
        use_agc = task_name == "vad_26db_with_agc"
        cfg = rrb.make_exp_config(
            dataset_name="universal",
            train_csv=univ_info["train_csv"],
            eval_csv=univ_info["eval_csv"],
            trials_file=univ_info["trials_file"],
            backbone="conformer",
            size_variant="small",
            loss_type="ge2e_softmax",
            scoring_type="cosine",
            pooling_type="atp",
            num_steps=num_steps,
        )
        cfg.vad.energy_db_threshold = top_db
        cfg.vad.apply_agc = use_agc
        custom_fe = feature_extraction.AudioFrontend(frontend_cfg, cfg.vad)
        custom_caches: dict[str, feature_extraction.FeatureCache] = {}
        for dname in ("librispeech", "cnceleb2", "voxceleb1"):
            c_dir = os.path.join(data_root, "feature_cache", f"{dname}_{task_name}")
            c_obj = feature_extraction.FeatureCache(custom_fe, cache_dir=c_dir)
            c_obj.precompute_dataset(spk_eval_sets[dname], num_workers=8)
            c_obj.precompute_dataset(spk_train_sets[dname], num_workers=8)
            custom_caches[dname] = c_obj
        custom_univ_cache = feature_extraction.FeatureCache.merged(
            custom_fe,
            [custom_caches[d] for d in ("librispeech", "cnceleb2", "voxceleb1")],
            cache_dir=os.path.join(
                data_root, "feature_cache", f"universal_{task_name}"
            ),
        )
        custom_caches["universal"] = custom_univ_cache
        res = train_and_eval_ablation(
            task_name,
            cfg,
            spk_univ_train,
            custom_univ_cache,
            custom_caches,
            trials_sets,
            multi_trials_sets,
            custom_fe,
        )
    elif task_name in (
        "pfas_sn_mlp_no_anchor",
        "pfas_sn_dot_with_anchor",
        "pfas_sn_maxpool_with_anchor",
    ):
        attn_map = {
            "pfas_sn_mlp_no_anchor": ("mlp", False),
            "pfas_sn_dot_with_anchor": ("dot", True),
            "pfas_sn_maxpool_with_anchor": ("maxpool", True),
        }
        attn_type, use_anchor = attn_map[task_name]
        cfg = rrb.make_exp_config(
            dataset_name="universal",
            train_csv=univ_info["train_csv"],
            eval_csv=univ_info["eval_csv"],
            trials_file=univ_info["trials_file"],
            backbone="conformer",
            size_variant="small",
            loss_type="extended_set_softmax",
            scoring_type="pfas",
            pooling_type="pfas",
            num_steps=num_steps,
        )
        cfg.model.pfas.attention_type = attn_type
        cfg.model.pfas.use_global_anchor = use_anchor
        res = train_and_eval_ablation(
            task_name,
            cfg,
            spk_univ_train,
            caches["universal"],
            caches,
            trials_sets,
            multi_trials_sets,
            frontend,
        )
    elif task_name == "dr_vector_3term_sym_skip":
        cfg = rrb.make_exp_config(
            dataset_name="universal",
            train_csv=univ_info["train_csv"],
            eval_csv=univ_info["eval_csv"],
            trials_file=univ_info["trials_file"],
            backbone="conformer",
            size_variant="small",
            loss_type="extended_set_softmax",
            scoring_type="dr_vector",
            pooling_type="atp",
            num_steps=num_steps,
        )
        cfg.model.dr_vector.num_cosine_terms = 3
        cfg.model.dr_vector.symmetric_input = True
        cfg.model.dr_vector.append_cosine_to_final_hidden = True
        res = train_and_eval_ablation(
            task_name,
            cfg,
            spk_univ_train,
            caches["universal"],
            caches,
            trials_sets,
            multi_trials_sets,
            frontend,
        )
    elif task_name in (
        "icassp2018_ge2e_pure_paper",
        "interspeech2021_dr_vectors_pure_paper",
        "odyssey2022_conformer_atp_pure_paper",
        "odyssey2022_pfas_pure_paper",
    ):
        preset_map: dict[str, Callable[[], configs.ExperimentConfig]] = {
            "icassp2018_ge2e_pure_paper": lambda: configs.icassp2018_ge2e_config(
                size="small", pure_paper_mode=True
            ),
            "interspeech2021_dr_vectors_pure_paper": (
                lambda: configs.interspeech2021_dr_vectors_config(
                    size="small", pure_paper_mode=True
                )
            ),
            "odyssey2022_conformer_atp_pure_paper": (
                lambda: configs.odyssey2022_conformer_atp_config(
                    size="small", pure_paper_mode=True
                )
            ),
            "odyssey2022_pfas_pure_paper": lambda: configs.odyssey2022_pfas_config(
                size="small", pure_paper_mode=True
            ),
        }
        cfg = preset_map[task_name]()
        cfg.model.seq_len = 180
        cfg.model.min_seq_len = 90
        cfg.model.max_seq_len = 180
        cfg.model.variable_length_training = True
        cfg.train.num_steps = num_steps
        cfg.train.learning_rate = 1.5e-3
        cfg.train.min_learning_rate = 1.5e-5
        cfg.train.warmup_steps = max(10, num_steps // 10)
        cfg.train.batch_size = 32
        cfg.train.loss.num_speakers_per_batch = 8
        cfg.train.loss.num_utts_per_speaker = 4
        res = train_and_eval_ablation(
            task_name,
            cfg,
            spk_univ_train,
            caches["universal"],
            caches,
            trials_sets,
            multi_trials_sets,
            frontend,
        )
    else:
        raise ValueError(f"Unknown R1 ablation task: {task_name}")

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


CANONICAL_R1_TASKS = [
    "domain_mixing_p_within_0.0",
    "domain_mixing_p_within_1.0",
    "ext_softmax_pure_eq3",
    "dr_vectors_3term_pure_eq7",
    "variable_length_true_short_eval",
    "variable_length_false",
    "horizon_snapshots_full",
    "ls_mamba_bidirectional",
]

EXTENDED_R1_TASKS = [
    "mamba_bidirectional",
    "ecapa_bn_style_after_act",
    "vad_26db_no_agc",
    "vad_20db_no_agc",
    "pfas_sn_mlp_no_anchor",
    "pfas_sn_dot_with_anchor",
    "pfas_sn_maxpool_with_anchor",
    "dr_vector_3term_sym_skip",
    "icassp2018_ge2e_pure_paper",
    "interspeech2021_dr_vectors_pure_paper",
    "odyssey2022_conformer_atp_pure_paper",
    "odyssey2022_pfas_pure_paper",
]

ALL_R1_DISTINCT_TASKS = CANONICAL_R1_TASKS + EXTENDED_R1_TASKS


def assemble_ablation_results(
    output_path: str = os.path.join(DEFAULT_OUTPUT_ROOT, "ablation_results.json"),
    num_steps: int = DEFAULT_NUM_STEPS,
) -> dict[str, Any]:
    """Assemble `pretrained_models/ablation_results.json` from main models + R1 cache."""
    cached_tasks: dict[str, Any] = {}
    for tname in ALL_R1_DISTINCT_TASKS + ["vad_26db_with_agc"]:
        cpath = os.path.join(CACHE_DIR, f"r1_{tname}.json")
        if os.path.exists(cpath):
            with open(cpath, "r", encoding="utf-8") as f:
                cached_tasks[tname] = _ensure_multi_domain_eval(json.load(f))

    rec_075 = _load_main_eval("univ_conformer_small_ge2e")
    rec_ext_blend = _load_main_eval("univ_conformer_small_ext_softmax")
    rec_dr1 = _load_main_eval("univ_conformer_small_dr_vectors")
    rec_ls_mamba_uni = _load_main_eval("ls_mamba_small_ge2e")

    domain_mixing: dict[str, Any] = {}
    if "domain_mixing_p_within_0.0" in cached_tasks:
        domain_mixing["p_within_0.0"] = cached_tasks["domain_mixing_p_within_0.0"]
    if rec_075 is not None:
        domain_mixing["p_within_0.75"] = {
            "within_domain_prob": 0.75,
            "num_steps": rec_075["num_steps"],
            "train_time_sec": rec_075["train_time_sec"],
            "initial_loss": rec_075["initial_loss"],
            "final_loss": rec_075["final_loss"],
            "multi_domain_eval": rec_075["multi_domain_eval"],
        }
    if "domain_mixing_p_within_1.0" in cached_tasks:
        domain_mixing["p_within_1.0"] = cached_tasks["domain_mixing_p_within_1.0"]

    ext_and_dr: dict[str, Any] = {}
    if "ext_softmax_pure_eq3" in cached_tasks:
        ext_and_dr["ext_softmax_pure_eq3"] = cached_tasks["ext_softmax_pure_eq3"]
    if rec_ext_blend is not None:
        ext_and_dr["ext_softmax_blended"] = {
            "pure_extended_set": False,
            "num_steps": rec_ext_blend["num_steps"],
            "train_time_sec": rec_ext_blend["train_time_sec"],
            "initial_loss": rec_ext_blend["initial_loss"],
            "final_loss": rec_ext_blend["final_loss"],
            "multi_domain_eval": rec_ext_blend["multi_domain_eval"],
        }
    if "dr_vectors_3term_pure_eq7" in cached_tasks:
        ext_and_dr["dr_vectors_3term_pure_eq7"] = cached_tasks[
            "dr_vectors_3term_pure_eq7"
        ]
    if rec_dr1 is not None:
        ext_and_dr["dr_vectors_1term_blended"] = {
            "num_cosine_terms": 1,
            "pure_extended_set": False,
            "num_steps": rec_dr1["num_steps"],
            "train_time_sec": rec_dr1["train_time_sec"],
            "initial_loss": rec_dr1["initial_loss"],
            "final_loss": rec_dr1["final_loss"],
            "multi_domain_eval": rec_dr1["multi_domain_eval"],
        }

    var_len_ablation: dict[str, Any] = {}
    if "variable_length_true_short_eval" in cached_tasks:
        var_len_ablation["variable_length_true"] = cached_tasks[
            "variable_length_true_short_eval"
        ]
    if "variable_length_false" in cached_tasks:
        var_len_ablation["variable_length_false"] = cached_tasks[
            "variable_length_false"
        ]

    horizon_ablation: dict[str, Any] = {}
    if "horizon_snapshots_full" in cached_tasks:
        snaps = cached_tasks["horizon_snapshots_full"].get("snapshots", {})
        for s_num in (400, 800, 1200, 2400):
            s_key = f"steps_{s_num}"
            if s_key in snaps:
                horizon_ablation[s_key] = snaps[s_key]
    if rec_075 is not None:
        md_4600 = rec_075["multi_domain_eval"]
        horizon_ablation[f"steps_{num_steps}"] = {
            "num_steps": int(rec_075["num_steps"]),
            "epochs": rec_075.get("train_epochs", round((num_steps * 32) / 9447.0, 2)),
            "train_time_sec": rec_075["train_time_sec"],
            "initial_loss": rec_075["initial_loss"],
            "final_loss": rec_075["final_loss"],
            "loss_at_horizon": rec_075["final_loss"],
            "multi_domain_eval": md_4600,
            "in_domain_1utt": md_4600["domains_1utt"]["universal_pooled"],
            "in_domain_3utt": md_4600["domains_3utt"]["universal_pooled"],
            "macro_avg_1utt": md_4600["universal_macro_avg_1utt"],
            "macro_avg_3utt": md_4600["universal_macro_avg_3utt"],
        }

    mamba_ls_ablation: dict[str, Any] = {}
    if rec_ls_mamba_uni is not None:
        mamba_ls_ablation["unidirectional"] = {
            "bidirectional": False,
            "num_parameters": rec_ls_mamba_uni["num_parameters"],
            "num_steps": rec_ls_mamba_uni["num_steps"],
            "train_epochs": rec_ls_mamba_uni.get("train_epochs", 13.27),
            "train_time_sec": rec_ls_mamba_uni["train_time_sec"],
            "initial_loss": rec_ls_mamba_uni["initial_loss"],
            "final_loss": rec_ls_mamba_uni["final_loss"],
            "multi_domain_eval": rec_ls_mamba_uni["multi_domain_eval"],
        }
    if "ls_mamba_bidirectional" in cached_tasks:
        mamba_ls_ablation["bidirectional"] = cached_tasks["ls_mamba_bidirectional"]

    mamba_uni = _extract_from_main_benchmark(
        "univ_mamba_small_ge2e", "mamba_unidirectional"
    )
    ecapa_before = _extract_from_main_benchmark(
        "univ_ecapa_tdnn_small_ge2e", "ecapa_ln_before_act"
    )
    vad_26_with_agc = _extract_from_main_benchmark(
        "univ_conformer_small_ge2e", "vad_26db_with_agc"
    )
    pfas_mlp_anchor = _extract_from_main_benchmark(
        "univ_conformer_small_pfas", "pfas_sn_mlp_with_anchor"
    )
    dr_1term = _extract_from_main_benchmark(
        "univ_conformer_small_dr_vectors", "dr_vector_1term_asym"
    )
    ge2e_contrast = _extract_from_main_benchmark(
        "univ_conformer_small_ge2e_contrast", "ge2e_contrast_stabilized"
    )

    studies: dict[str, Any] = {
        "mamba_direction_ablation": {
            "mamba_unidirectional": mamba_uni,
            "mamba_bidirectional": cached_tasks.get("mamba_bidirectional"),
        },
        "ecapa_norm_order_ablation": {
            "ecapa_ln_before_act": ecapa_before,
            "ecapa_bn_style_after_act": cached_tasks.get("ecapa_bn_style_after_act"),
        },
        "vad_and_agc_ablation": {
            "vad_26db_no_agc": cached_tasks.get("vad_26db_no_agc"),
            "vad_20db_no_agc": cached_tasks.get("vad_20db_no_agc"),
            "vad_26db_with_agc": (
                vad_26_with_agc or cached_tasks.get("vad_26db_with_agc")
            ),
        },
        "training_horizon_ablation": horizon_ablation,
        "pfas_sn_attention_and_anchor_ablation": {
            "pfas_sn_mlp_with_anchor": pfas_mlp_anchor,
            "pfas_sn_mlp_no_anchor": cached_tasks.get("pfas_sn_mlp_no_anchor"),
            "pfas_sn_dot_with_anchor": cached_tasks.get("pfas_sn_dot_with_anchor"),
            "pfas_sn_maxpool_with_anchor": cached_tasks.get(
                "pfas_sn_maxpool_with_anchor"
            ),
        },
        "dr_vector_terms_and_contrast_ablation": {
            "dr_vector_1term_asym": dr_1term,
            "dr_vector_3term_sym_skip": cached_tasks.get("dr_vector_3term_sym_skip"),
            "ge2e_contrast_stabilized": ge2e_contrast,
        },
        "paper_faithful_presets_smoke_test": {
            "icassp2018_ge2e_pure_paper": cached_tasks.get(
                "icassp2018_ge2e_pure_paper"
            ),
            "interspeech2021_dr_vectors_pure_paper": cached_tasks.get(
                "interspeech2021_dr_vectors_pure_paper"
            ),
            "odyssey2022_conformer_atp_pure_paper": cached_tasks.get(
                "odyssey2022_conformer_atp_pure_paper"
            ),
            "odyssey2022_pfas_pure_paper": cached_tasks.get(
                "odyssey2022_pfas_pure_paper"
            ),
        },
    }

    ablation_results: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "default_num_steps": num_steps,
        "default_librispeech_num_steps": DEFAULT_LS_NUM_STEPS,
        "default_epochs": round((num_steps * 32) / 9447.0, 2),
        "domain_mixing_ablation": domain_mixing,
        "extended_set_and_dr_ablation": ext_and_dr,
        "variable_length_ablation": var_len_ablation,
        "training_horizon_ablation": horizon_ablation,
        "mamba_directionality_ablation": mamba_ls_ablation,
        "studies": studies,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Assembled R1 ablations -> {output_path}", flush=True)
    return ablation_results


def run_ablations(
    data_root: str = DEFAULT_DATA_ROOT,
    output_path: str = os.path.join(DEFAULT_OUTPUT_ROOT, "ablation_results.json"),
    num_steps: int = DEFAULT_NUM_STEPS,
) -> dict[str, Any]:
    for tname in ALL_R1_DISTINCT_TASKS:
        run_single_ablation_task(tname, data_root=data_root, num_steps=num_steps)
    return assemble_ablation_results(output_path=output_path, num_steps=num_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(DEFAULT_OUTPUT_ROOT, "ablation_results.json"),
    )
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional single R1 task name from ALL_R1_DISTINCT_TASKS",
    )
    parser.add_argument(
        "--assemble_only",
        action="store_true",
        help="Assemble ablation_results.json from cached tasks and main benchmark results",
    )
    args = parser.parse_args()
    if args.assemble_only:
        assemble_ablation_results(
            output_path=args.output_path, num_steps=args.num_steps
        )
    elif args.task:
        run_single_ablation_task(
            args.task, data_root=args.data_root, num_steps=args.num_steps
        )
    else:
        run_ablations(
            data_root=args.data_root,
            output_path=args.output_path,
            num_steps=args.num_steps,
        )
