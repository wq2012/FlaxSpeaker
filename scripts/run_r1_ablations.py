#!/usr/bin/env python3
"""Run the 5 controlled ablation studies reported in Table 7 / Appendix C.

Studies:
1. Domain Mixing Probability (`within_domain_prob` in {0.0, 0.75, 1.0}) on
   `univ_conformer_small_ge2e`.
2. Pure Extended-Set Softmax (Pelecanos et al. 2021 Eq. 2, `pure_extended_set=True`)
   vs. Blended Extended-Set Softmax (FlaxSpeaker variant), and 3-Term Dr-Vectors
   (`[c, c^2, sqrt(max(c, 0))]`, a FlaxSpeaker extension NOT in the paper) vs. 1-Term
   Blended Dr-Vectors.

   NOTE: the baseline rows of Studies 1-5 (`p_within_0.75`, `ext_softmax_blended`,
   `dr_vectors_1term_blended`, `variable_length_true`, `steps_400`, `unidirectional`)
   are *copied* from the main benchmark's `eval_results.json`, not re-trained here; the
   re-trained variants therefore start from a different RNG stream (`initial_loss`
   differs). Study 4's `steps_800` entry is a mid-schedule snapshot of the single
   1,200-step run (LR still high), not an 800-step budget run; see
   `scripts/run_r2_controls.py` for an honest 800-step run with its own schedule.
3. Variable-Length Training (`variable_length_training=True`) vs. Fixed-Length
   Training (`variable_length_training=False`) evaluated on Full-Sequence and
   Short-Duration (1.0 s / 100-frame) utterances.
4. Training Horizon Scaling (400 vs. 800 vs. 1,200 steps) on
   `univ_conformer_small_ge2e`.
5. Unidirectional vs. Bidirectional Mamba (`bidirectional=True`) on LibriSpeech
   (`ls_mamba_small_ge2e`).

Outputs `pretrained_models/ablation_results.json`.
"""

import argparse
import copy
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
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402
from flaxspeaker import neural_net  # noqa: E402
from scripts import run_research_benchmarks as rrb  # noqa: E402

DEFAULT_DATA_ROOT = os.environ.get(
    "FLAXSPEAKER_DATA_ROOT", os.path.expanduser("~/Data/speaker_datasets")
)
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "pretrained_models")


def run_single_config(
    exp_cfg: Any,
    spk_train: dataset.SpkToUtts,
    frontend: feature_extraction.AudioFrontend,
    train_cache: feature_extraction.FeatureCache,
    caches: dict[str, feature_extraction.FeatureCache],
    trials_sets: dict[str, list[dataset.TrialPair]],
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]],
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    t0 = time.time()
    _, state = neural_net.get_speaker_encoder(exp_cfg, num_classes=0)
    step_fn = neural_net.make_modern_train_step(exp_cfg)
    rng = np.random.default_rng(exp_cfg.train.seed)
    losses: list[float] = []
    for step in range(exp_cfg.train.num_steps):
        batch_input, batch_labels, batch_mask = feature_extraction.get_batched_nm_input(
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
            variable_length=exp_cfg.model.variable_length_training,
            return_mask=True,
            within_domain_prob=exp_cfg.data.within_domain_prob,
            domain_temp_alpha=exp_cfg.data.domain_temp_alpha,
        )
        state, loss_val = step_fn(state, batch_input, batch_labels, batch_mask)
        losses.append(float(loss_val.item()))
        if (step + 1) % 100 == 0 or step == 0 or step == exp_cfg.train.num_steps - 1:
            print(f"    step {step + 1}/{exp_cfg.train.num_steps} loss={losses[-1]:.4f}")
    train_time = round(time.time() - t0, 2)
    shared_embs: dict[str, Any] = {}
    md_eval = rrb.evaluate_all_domains(
        state, exp_cfg, caches, trials_sets, multi_trials_sets, shared_embs=shared_embs
    )
    return {
        "train_time_sec": train_time,
        "initial_loss": round(losses[0], 4),
        "final_loss": round(float(np.mean(losses[-20:])), 4),
        "multi_domain_eval": md_eval,
    }, state, shared_embs


def eval_short_duration(
    state: Any,
    exp_cfg: Any,
    caches: dict[str, feature_extraction.FeatureCache],
    trials_sets: dict[str, list[dataset.TrialPair]],
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]],
    max_frames: int = 100,
) -> dict[str, Any]:
    short_cfg = copy.deepcopy(exp_cfg)
    short_cfg.model.seq_len = max_frames
    short_cfg.model.max_eval_seq_len = max_frames
    short_cfg.eval.full_sequence_inference = False
    return rrb.evaluate_all_domains(
        state, short_cfg, caches, trials_sets, multi_trials_sets, shared_embs={}
    )


def run_all_ablations(
    data_root: str = DEFAULT_DATA_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    output_json: str = "",
) -> dict[str, Any]:
    if not output_json:
        output_json = os.path.join(output_root, "ablation_results.json")

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
    spk_eval_sets: dict[str, dataset.SpkToUtts] = {}
    trials_sets: dict[str, list[dataset.TrialPair]] = {}
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]] = {}

    for dname in ("librispeech", "cnceleb2", "voxceleb1"):
        info = ds_summary[dname]
        spk_train_sets[dname] = dataset.get_csv_spk_to_utts(info["train_csv"])
        spk_eval_sets[dname] = dataset.get_csv_spk_to_utts(info["eval_csv"])
        trials_sets[dname] = dataset.load_trials_file(info["trials_file"])
        multi_trials_sets[dname] = rrb.load_multi_enroll_trials(info["multi_trials_file"])
        c_dir = os.path.join(data_root, "feature_cache", dname)
        cache = feature_extraction.FeatureCache(frontend, cache_dir=c_dir)
        cache.precompute_dataset(spk_eval_sets[dname], num_workers=16)
        cache.precompute_dataset(spk_train_sets[dname], num_workers=16)
        caches[dname] = cache

    univ_info = ds_summary["universal"]
    spk_train_sets["universal"] = dataset.get_csv_spk_to_utts(univ_info["train_csv"])
    spk_eval_sets["universal"] = dataset.get_csv_spk_to_utts(univ_info["eval_csv"])
    trials_sets["universal"] = dataset.load_trials_file(univ_info["trials_file"])
    multi_trials_sets["universal"] = rrb.load_multi_enroll_trials(univ_info["multi_trials_file"])
    univ_cache = feature_extraction.FeatureCache.merged(
        frontend,
        [caches[d] for d in ("librispeech", "cnceleb2", "voxceleb1")],
        cache_dir=os.path.join(data_root, "feature_cache", "universal"),
    )
    caches["universal"] = univ_cache

    results: dict[str, Any] = {
        "domain_mixing_ablation": {},
        "extended_set_and_dr_ablation": {},
        "variable_length_ablation": {},
        "training_horizon_ablation": {},
        "mamba_directionality_ablation": {},
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Study 1: Domain Mixing (within_domain_prob in {0.0, 0.75, 1.0})
    with open(
        os.path.join(output_root, "univ_conformer_small_ge2e", "eval_results.json"),
        "r",
        encoding="utf-8",
    ) as f:
        rec_075 = json.load(f)
    results["domain_mixing_ablation"]["p_within_0.75"] = {
        "within_domain_prob": 0.75,
        "train_time_sec": rec_075["train_time_sec"],
        "initial_loss": rec_075["initial_loss"],
        "final_loss": rec_075["final_loss"],
        "multi_domain_eval": rec_075["multi_domain_eval"],
    }
    for p_val in (0.0, 1.0):
        print(f"\n=== Study 1: univ_conformer_small_ge2e with within_domain_prob={p_val} ===")
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
            num_steps=400,
        )
        cfg.data.within_domain_prob = p_val
        res, _, _ = run_single_config(
            cfg,
            spk_train_sets["universal"],
            frontend,
            caches["universal"],
            caches,
            trials_sets,
            multi_trials_sets,
        )
        res["within_domain_prob"] = p_val
        results["domain_mixing_ablation"][f"p_within_{p_val}"] = res

    # Study 2: Pure vs Blended Extended-Set Softmax & 3-term vs 1-term Dr-Vectors
    with open(
        os.path.join(output_root, "univ_conformer_small_ext_softmax", "eval_results.json"),
        "r",
        encoding="utf-8",
    ) as f:
        rec_ext_blend = json.load(f)
    results["extended_set_and_dr_ablation"]["ext_softmax_blended"] = {
        "pure_extended_set": False,
        "train_time_sec": rec_ext_blend["train_time_sec"],
        "initial_loss": rec_ext_blend["initial_loss"],
        "final_loss": rec_ext_blend["final_loss"],
        "multi_domain_eval": rec_ext_blend["multi_domain_eval"],
    }
    print("\n=== Study 2a: Pure Extended-Set Softmax (paper Eq. 2, pure_extended_set=True) ===")
    cfg_pure = rrb.make_exp_config(
        dataset_name="universal",
        train_csv=univ_info["train_csv"],
        eval_csv=univ_info["eval_csv"],
        trials_file=univ_info["trials_file"],
        backbone="conformer",
        size_variant="small",
        loss_type="extended_set_softmax",
        scoring_type="cosine",
        pooling_type="atp",
        num_steps=400,
    )
    cfg_pure.train.loss.pure_extended_set = True
    res_pure, _, _ = run_single_config(
        cfg_pure,
        spk_train_sets["universal"],
        frontend,
        caches["universal"],
        caches,
        trials_sets,
        multi_trials_sets,
    )
    res_pure["pure_extended_set"] = True
    # Key name kept for backward compatibility with the archived JSON ("eq3" is historical;
    # the pure extended-set softmax is Eq. 2 of Pelecanos et al. 2021).
    results["extended_set_and_dr_ablation"]["ext_softmax_pure_eq3"] = res_pure

    with open(
        os.path.join(output_root, "univ_conformer_small_dr_vectors", "eval_results.json"),
        "r",
        encoding="utf-8",
    ) as f:
        rec_dr1 = json.load(f)
    results["extended_set_and_dr_ablation"]["dr_vectors_1term_blended"] = {
        "num_cosine_terms": 1,
        "pure_extended_set": False,
        "train_time_sec": rec_dr1["train_time_sec"],
        "initial_loss": rec_dr1["initial_loss"],
        "final_loss": rec_dr1["final_loss"],
        "multi_domain_eval": rec_dr1["multi_domain_eval"],
    }
    print("\n=== Study 2b: 3-Term Dr-Vectors (FlaxSpeaker extension) + Pure Extended-Set Softmax ===")
    cfg_dr3 = rrb.make_exp_config(
        dataset_name="universal",
        train_csv=univ_info["train_csv"],
        eval_csv=univ_info["eval_csv"],
        trials_file=univ_info["trials_file"],
        backbone="conformer",
        size_variant="small",
        loss_type="extended_set_softmax",
        scoring_type="dr_vector",
        pooling_type="atp",
        num_steps=400,
    )
    cfg_dr3.model.dr_vector.num_cosine_terms = 3
    cfg_dr3.train.loss.pure_extended_set = True
    res_dr3, _, _ = run_single_config(
        cfg_dr3,
        spk_train_sets["universal"],
        frontend,
        caches["universal"],
        caches,
        trials_sets,
        multi_trials_sets,
    )
    res_dr3["num_cosine_terms"] = 3
    res_dr3["pure_extended_set"] = True
    # Key name kept for backward compatibility with the archived JSON; the "eq7" suffix
    # is historical -- the 3-term expansion is NOT an equation of Pelecanos et al. 2021.
    results["extended_set_and_dr_ablation"]["dr_vectors_3term_pure_eq7"] = res_dr3

    # Study 3: Variable-Length vs Fixed-Length Training
    loaded_var = hf_compat.FlaxSpeakerModel.from_pretrained(
        os.path.join(output_root, "univ_conformer_small_ge2e")
    )
    short_var_eval = eval_short_duration(
        loaded_var.state,
        loaded_var.exp_config,
        caches,
        trials_sets,
        multi_trials_sets,
        max_frames=100,
    )
    results["variable_length_ablation"]["variable_length_true"] = {
        "variable_length_training": True,
        "full_seq_eval": rec_075["multi_domain_eval"],
        "short_1s_eval": short_var_eval,
    }
    print("\n=== Study 3: Fixed-Length Training (variable_length_training=False) ===")
    cfg_fixed = rrb.make_exp_config(
        dataset_name="universal",
        train_csv=univ_info["train_csv"],
        eval_csv=univ_info["eval_csv"],
        trials_file=univ_info["trials_file"],
        backbone="conformer",
        size_variant="small",
        loss_type="ge2e_softmax",
        scoring_type="cosine",
        pooling_type="atp",
        num_steps=400,
    )
    cfg_fixed.model.variable_length_training = False
    res_fixed, state_fixed, _ = run_single_config(
        cfg_fixed,
        spk_train_sets["universal"],
        frontend,
        caches["universal"],
        caches,
        trials_sets,
        multi_trials_sets,
    )
    short_fixed_eval = eval_short_duration(
        state_fixed, cfg_fixed, caches, trials_sets, multi_trials_sets, max_frames=100
    )
    results["variable_length_ablation"]["variable_length_false"] = {
        "variable_length_training": False,
        "train_time_sec": res_fixed["train_time_sec"],
        "initial_loss": res_fixed["initial_loss"],
        "final_loss": res_fixed["final_loss"],
        "full_seq_eval": res_fixed["multi_domain_eval"],
        "short_1s_eval": short_fixed_eval,
    }

    # Study 4: Training Horizon Scaling (400 vs 800 vs 1200 steps)
    print("\n=== Study 4: Training Horizon Scaling (400 -> 800 -> 1200 steps) ===")
    results["training_horizon_ablation"]["steps_400"] = {
        "num_steps": 400,
        "train_time_sec": rec_075["train_time_sec"],
        "initial_loss": rec_075["initial_loss"],
        "final_loss": rec_075["final_loss"],
        "multi_domain_eval": rec_075["multi_domain_eval"],
    }
    cfg_1200 = rrb.make_exp_config(
        dataset_name="universal",
        train_csv=univ_info["train_csv"],
        eval_csv=univ_info["eval_csv"],
        trials_file=univ_info["trials_file"],
        backbone="conformer",
        size_variant="small",
        loss_type="ge2e_softmax",
        scoring_type="cosine",
        pooling_type="atp",
        num_steps=1200,
    )
    t_h0 = time.time()
    _, state_h = neural_net.get_speaker_encoder(cfg_1200, num_classes=0)
    step_fn_h = neural_net.make_modern_train_step(cfg_1200)
    rng_h = np.random.default_rng(cfg_1200.train.seed)
    losses_h: list[float] = []
    for step in range(1200):
        batch_input, batch_labels, batch_mask = feature_extraction.get_batched_nm_input(
            spk_train_sets["universal"],
            num_speakers=cfg_1200.train.loss.num_speakers_per_batch,
            num_utts_per_speaker=cfg_1200.train.loss.num_utts_per_speaker,
            seq_len=cfg_1200.model.seq_len,
            frontend=frontend,
            specaug_config=cfg_1200.train.specaug,
            waveform_aug_config=cfg_1200.train.waveform_aug,
            cache=caches["universal"],
            rng=rng_h,
            min_seq_len=cfg_1200.model.min_seq_len,
            variable_length=True,
            return_mask=True,
            within_domain_prob=cfg_1200.data.within_domain_prob,
            domain_temp_alpha=cfg_1200.data.domain_temp_alpha,
        )
        state_h, loss_val = step_fn_h(state_h, batch_input, batch_labels, batch_mask)
        losses_h.append(float(loss_val.item()))
        if (step + 1) in (800, 1200):
            elapsed_h = round(time.time() - t_h0, 2)
            md_h = rrb.evaluate_all_domains(
                state_h, cfg_1200, caches, trials_sets, multi_trials_sets, shared_embs={}
            )
            results["training_horizon_ablation"][f"steps_{step + 1}"] = {
                "num_steps": step + 1,
                "train_time_sec": elapsed_h,
                "initial_loss": round(losses_h[0], 4),
                "final_loss": round(float(np.mean(losses_h[-20:])), 4),
                "multi_domain_eval": md_h,
            }

    # Study 5: Unidirectional vs Bidirectional Mamba
    with open(
        os.path.join(output_root, "ls_mamba_small_ge2e", "eval_results.json"),
        "r",
        encoding="utf-8",
    ) as f:
        rec_mamba_uni = json.load(f)
    results["mamba_directionality_ablation"]["unidirectional"] = {
        "bidirectional": False,
        "num_parameters": rec_mamba_uni["num_parameters"],
        "train_time_sec": rec_mamba_uni["train_time_sec"],
        "initial_loss": rec_mamba_uni["initial_loss"],
        "final_loss": rec_mamba_uni["final_loss"],
        "multi_domain_eval": rec_mamba_uni["multi_domain_eval"],
    }
    print("\n=== Study 5: Bidirectional Mamba (ls_mamba_small_ge2e, bidirectional=True) ===")
    ls_info = ds_summary["librispeech"]
    cfg_mamba_bi = rrb.make_exp_config(
        dataset_name="librispeech",
        train_csv=ls_info["train_csv"],
        eval_csv=ls_info["eval_csv"],
        trials_file=ls_info["trials_file"],
        backbone="mamba",
        size_variant="small",
        loss_type="ge2e_softmax",
        scoring_type="cosine",
        pooling_type="asp",
        num_steps=400,
    )
    cfg_mamba_bi.model.mamba.bidirectional = True
    res_mb, state_mb, _ = run_single_config(
        cfg_mamba_bi,
        spk_train_sets["librispeech"],
        frontend,
        caches["librispeech"],
        caches,
        trials_sets,
        multi_trials_sets,
    )
    results["mamba_directionality_ablation"]["bidirectional"] = {
        "bidirectional": True,
        "num_parameters": neural_net.count_parameters(state_mb.params),
        "train_time_sec": res_mb["train_time_sec"],
        "initial_loss": res_mb["initial_loss"],
        "final_loss": res_mb["final_loss"],
        "multi_domain_eval": res_mb["multi_domain_eval"],
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ablation results to {output_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the 5 controlled ablation studies in Table 7 / Appendix C."
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()
    run_all_ablations(
        data_root=args.data_root,
        output_root=args.output_root,
        output_json=args.output_json,
    )
