#!/usr/bin/env python3
"""Wide 16-configuration PFAS hyperparameter sweep on the warm-started Conformer-Large trunk.

Sweeps across all core architectural and objective hyperparameters of Parameter-Free
Attentive Scoring (Pelecanos et al., Odyssey 2022) and FlaxSpeaker PFAS extensions:
- `num_keys` in `{4, 8, 16, 32}`
- `key_dim` in `{16, 32, 64}`
- `value_dim` in `{48, 64, 96, 128}`
- `attention_type` in `{"shared_nonlinear", "atp_ffn", "unshared_nonlinear"}`
- `attention_weight_mode` in `{"exp", "outer"}`
- `use_global_anchor` in `{True, False}` and `local_value_scale` in `{0.15, 0.25, 0.35, 0.50}`
- `per_trial_softmax` vs. `apply_softmax_per_test_key`
- `scale_factor` in `{12.0, 16.0, 24.0}`
- `loss_type` in `{"extended_set_softmax" (blended & pure Eq. 2), "ge2e_softmax"}`

All 16 candidates warm-start from the converged `Conformer-Large` trunk checkpoint, run a
2-phase adaptation schedule (head warmup with frozen trunk -> joint low-LR trunk + head
fine-tuning) over `sharded_caches/universal_full_v050` (`base_speakers_only=False`), and are
evaluated strictly on the speaker-disjoint validation split (`splits/universal_full_val/`
1-utt and 3-utt trials; zero evaluation-set exposure).

Saves full sweep results and the winning hyperparameter dictionary to
`pretrained_models/pfas_hparam_sweep.json`.
"""

from __future__ import annotations

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
from flaxspeaker import evaluation  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import neural_net  # noqa: E402
from flaxspeaker import resources  # noqa: E402
from flaxspeaker import shard_cache  # noqa: E402


PFAS_SWEEP_GRID: list[dict[str, Any]] = [
    {
        "name": "pfas_sn_k4_d32_v128_anchor025",
        "num_keys": 4,
        "key_dim": 32,
        "value_dim": 128,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.25,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_sn_k8_d32_v96_anchor025",
        "num_keys": 8,
        "key_dim": 32,
        "value_dim": 96,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.25,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_sn_k8_d32_v128_anchor025",
        "num_keys": 8,
        "key_dim": 32,
        "value_dim": 128,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.25,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_sn_k16_d16_v64_anchor025",
        "num_keys": 16,
        "key_dim": 16,
        "value_dim": 64,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.25,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_sn_k16_d32_v64_anchor035",
        "num_keys": 16,
        "key_dim": 32,
        "value_dim": 64,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.35,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_sn_k32_d16_v48_anchor025",
        "num_keys": 32,
        "key_dim": 16,
        "value_dim": 48,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.25,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_paper_atpffn_k32_d16_v48_pure",
        "num_keys": 32,
        "key_dim": 16,
        "value_dim": 48,
        "attention_type": "atp_ffn",
        "attention_weight_mode": "exp",
        "use_global_anchor": False,
        "local_value_scale": 0.25,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": True,
    },
    {
        "name": "pfas_paper_atpffn_k16_d32_v64_ln",
        "num_keys": 16,
        "key_dim": 32,
        "value_dim": 64,
        "attention_type": "atp_ffn",
        "attention_weight_mode": "exp",
        "use_global_anchor": False,
        "local_value_scale": 0.25,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_paper_atpffn_k8_d32_v128_ln",
        "num_keys": 8,
        "key_dim": 32,
        "value_dim": 128,
        "attention_type": "atp_ffn",
        "attention_weight_mode": "exp",
        "use_global_anchor": False,
        "local_value_scale": 0.25,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_unshared_k8_d32_v96_anchor035",
        "num_keys": 8,
        "key_dim": 32,
        "value_dim": 96,
        "attention_type": "unshared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.35,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_outer_k8_d32_v96_anchor025",
        "num_keys": 8,
        "key_dim": 32,
        "value_dim": 96,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "outer",
        "use_global_anchor": True,
        "local_value_scale": 0.25,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_noanchor_k8_d32_v96_normkv",
        "num_keys": 8,
        "key_dim": 32,
        "value_dim": 96,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": False,
        "local_value_scale": 1.0,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": True,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_pertestkey_k8_d32_v96_anchor025",
        "num_keys": 8,
        "key_dim": 32,
        "value_dim": 96,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.25,
        "per_trial_softmax": False,
        "apply_softmax_per_test_key": True,
        "apply_l2_norm_to_values": False,
        "scale_factor": 12.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_scale24_k8_d64_v96_anchor015",
        "num_keys": 8,
        "key_dim": 64,
        "value_dim": 96,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.15,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 24.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": False,
    },
    {
        "name": "pfas_local050_k8_d32_v96_pureext",
        "num_keys": 8,
        "key_dim": 32,
        "value_dim": 96,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.50,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "extended_set_softmax",
        "pure_extended_set": True,
    },
    {
        "name": "pfas_ge2esoftmax_k8_d32_v96_anchor025",
        "num_keys": 8,
        "key_dim": 32,
        "value_dim": 96,
        "attention_type": "shared_nonlinear",
        "attention_weight_mode": "exp",
        "use_global_anchor": True,
        "local_value_scale": 0.25,
        "per_trial_softmax": True,
        "apply_softmax_per_test_key": False,
        "apply_l2_norm_to_values": False,
        "scale_factor": 16.0,
        "loss_type": "ge2e_softmax",
        "pure_extended_set": False,
    },
]


def _load_multi_enroll_tsv(tsv_path: str) -> list[dataset.MultiEnrollTrial]:
    trials: list[dataset.MultiEnrollTrial] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            lbl = int(parts[0])
            e_list = [dataset.resolve_data_path(p) for p in parts[1].split(",") if p]
            t_u = dataset.resolve_data_path(parts[2])
            trials.append((lbl, e_list, t_u))
    return trials


def apply_pfas_spec_to_config(
    cfg: configs.ExperimentConfig, spec: dict[str, Any]
) -> None:
    """Applies a `PFAS_SWEEP_GRID` entry onto `cfg` in-place."""
    cfg.model.pooling.pooling_type = "pfas"
    cfg.model.scoring_type = "pfas"
    cfg.train.loss.scoring_type = "pfas"
    cfg.train.loss.loss_type = str(spec["loss_type"])
    cfg.train.loss.pure_extended_set = bool(spec.get("pure_extended_set", False))

    pf = cfg.model.pfas
    pf.num_keys = int(spec["num_keys"])
    pf.key_dim = int(spec["key_dim"])
    pf.value_dim = int(spec["value_dim"])
    pf.attention_dim = 128
    pf.attention_type = str(spec["attention_type"])
    pf.attention_weight_mode = str(spec["attention_weight_mode"])
    pf.atp_ffn_use_layer_norm = True
    pf.use_global_anchor = bool(spec["use_global_anchor"])
    pf.local_value_scale = float(spec["local_value_scale"])
    pf.per_trial_softmax = bool(spec["per_trial_softmax"])
    pf.apply_softmax_per_test_key = bool(spec["apply_softmax_per_test_key"])
    pf.apply_l2_norm_to_keys = True
    pf.apply_l2_norm_to_values = bool(spec["apply_l2_norm_to_values"])
    pf.apply_global_l2_norm_to_concat_form = True
    pf.scale_factor = float(spec["scale_factor"])
    pf.use_trainable_scale_factor = True
    cfg.loss = cfg.train.loss


def build_base_pfas_conformer_large_config(
    trunk_ckpt: str,
    shard_dir: str,
    spec: dict[str, Any],
    seq_len: int = 180,
    seed: int = 42,
) -> configs.ExperimentConfig:
    cfg = configs.ExperimentConfig()
    cfg.train.seed = int(seed)
    cfg.data.dataset_name = "universal_full"
    cfg.data.sharded_cache_dir = shard_dir
    cfg.data.within_domain_prob = 0.75
    cfg.data.domain_temp_alpha = 0.35

    cfg.frontend.feature_type = configs.FeatureType.LOG_MEL.value
    cfg.frontend.n_mels = 80
    cfg.frontend.n_mfcc = 80
    cfg.frontend.apply_cmvn = True
    cfg.vad.enabled = True
    cfg.vad.mode = configs.VadMode.ENERGY.value

    cfg.model.backbone = configs.BackboneType.CONFORMER.value
    cfg.model.apply_size_variant(configs.ModelSize.LARGE)
    cfg.model.seq_len = int(seq_len)
    cfg.model.min_seq_len = max(60, int(seq_len) // 2)
    cfg.model.max_seq_len = int(seq_len)
    cfg.model.max_eval_seq_len = 0
    cfg.model.variable_length_training = True
    cfg.eval.full_sequence_inference = True
    cfg.eval.max_eval_frames_cap = 3000

    cfg.train.init_from_checkpoint = trunk_ckpt
    cfg.train.optimizer = "adamw"
    cfg.train.weight_decay = 1e-4
    cfg.train.grad_clip_norm = 3.0
    cfg.train.lr_schedule_cosine = True
    cfg.train.loss.num_speakers_per_batch = 16
    cfg.train.loss.num_utts_per_speaker = 4
    cfg.train.batch_size = 64
    cfg.export.sequence_length = 180

    apply_pfas_spec_to_config(cfg, spec)
    return cfg


def run_pfas_sweep(
    trunk_ckpt: str,
    data_root: str,
    shard_dir: str,
    output_json: str,
    head_warmup_steps: int = 250,
    joint_finetune_steps: int = 150,
    max_threads: int = 28,
    overwrite: bool = False,
) -> dict[str, Any]:
    resources.configure_polite_resources(max_threads=max_threads, nice_level=10)
    if not overwrite and os.path.isfile(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("num_completed") == len(PFAS_SWEEP_GRID) and "winner" in prev:
            winner_eer = prev["winner"]["composite_val_eer_percent"]
            print(
                f"[PFAS Sweep] Loaded completed 16-config sweep from {output_json}: "
                f"winner={prev['winner']['name']} (composite val EER={winner_eer:.2f}%)",
                flush=True,
            )
            return prev

    val_dir = os.path.join(data_root, "splits", "universal_full_val")
    val_trials_1 = dataset.load_trials_file(os.path.join(val_dir, "trials.tsv"))
    val_trials_3 = _load_multi_enroll_tsv(
        os.path.join(val_dir, "multi_enroll_trials.tsv")
    )

    fe = feature_extraction.AudioFrontend(
        configs.FrontendConfig(), configs.VadConfig()
    )
    val_cache = feature_extraction.FeatureCache(
        fe, cache_dir=os.path.join(data_root, "feature_cache", "universal_full_val")
    )
    val_utts = sorted(
        {u for _, u1, u2 in val_trials_1 for u in (u1, u2)}
        | {u for _, e_l, t_u in val_trials_3 for u in list(e_l) + [t_u]}
    )
    val_cache.precompute_dataset({"val": val_utts}, num_workers=8)

    store = shard_cache.ShardedFeatureStore(shard_dir, base_speakers_only=False)
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

    results: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "trunk_checkpoint": trunk_ckpt,
        "shard_dir": shard_dir,
        "head_warmup_steps": int(head_warmup_steps),
        "joint_finetune_steps": int(joint_finetune_steps),
        "selection_metric": "0.5 * val_1utt_eer_percent + 0.5 * val_3utt_eer_percent",
        "candidates": {},
    }
    if os.path.isfile(output_json) and not overwrite:
        with open(output_json, "r", encoding="utf-8") as f:
            old = json.load(f)
        results["candidates"] = old.get("candidates", {})

    for idx, spec in enumerate(PFAS_SWEEP_GRID, 1):
        name = str(spec["name"])
        if not overwrite and name in results["candidates"]:
            c_rec = results["candidates"][name]
            print(
                f"[PFAS Sweep {idx:02d}/{len(PFAS_SWEEP_GRID)}] Cached {name}: "
                f"1-utt={c_rec['val_1utt_eer_percent']:.2f}%, 3-utt={c_rec['val_3utt_eer_percent']:.2f}%",
                flush=True,
            )
            continue

        t0 = time.time()
        cfg = build_base_pfas_conformer_large_config(
            trunk_ckpt=trunk_ckpt,
            shard_dir=shard_dir,
            spec=spec,
            seq_len=180,
            seed=42 + idx,
        )

        # Phase A: Freeze trunk, warm up PFAS head
        cfg.train.freeze_trunk = True
        cfg.train.trunk_lr_mult = 0.0
        cfg.train.learning_rate = 2.0e-3
        cfg.train.min_learning_rate = 2.0e-4
        cfg.train.num_steps = int(head_warmup_steps)
        cfg.train.warmup_steps = max(20, head_warmup_steps // 10)
        _, state = neural_net.get_speaker_encoder(cfg, num_classes=0)
        step_fn_a = neural_net.make_modern_train_step(cfg)
        rng = np.random.default_rng(cfg.train.seed)

        losses_a: list[float] = []
        for _ in range(head_warmup_steps):
            b_in, b_lbl, b_m = store.sample_nm_batch(
                num_speakers=cfg.train.loss.num_speakers_per_batch,
                num_utts_per_speaker=cfg.train.loss.num_utts_per_speaker,
                max_seq_len=cfg.model.seq_len,
                min_seq_len=cfg.model.min_seq_len,
                variable_length=True,
                specaug_config=cfg.train.specaug,
                rng=rng,
                within_domain_prob=cfg.data.within_domain_prob,
                domain_temp_alpha=cfg.data.domain_temp_alpha,
            )
            state, l_val = step_fn_a(state, b_in, b_lbl, b_m)
            losses_a.append(float(l_val.item()))

        # Phase B: Unfreeze trunk with 0.05x LR multiplier for joint alignment
        cfg.train.freeze_trunk = False
        cfg.train.trunk_lr_mult = 0.05
        cfg.train.learning_rate = 6.0e-4
        cfg.train.min_learning_rate = 3.0e-5
        cfg.train.num_steps = int(joint_finetune_steps)
        cfg.train.warmup_steps = max(20, joint_finetune_steps // 10)
        tx_b = neural_net.build_optimizer(cfg)
        state = neural_net.SpeakerTrainState.create(
            apply_fn=state.apply_fn,
            params=state.params,
            tx=tx_b,
            batch_stats=state.batch_stats,
        )
        step_fn_b = neural_net.make_modern_train_step(cfg)
        losses_b: list[float] = []
        for _ in range(joint_finetune_steps):
            b_in, b_lbl, b_m = store.sample_nm_batch(
                num_speakers=cfg.train.loss.num_speakers_per_batch,
                num_utts_per_speaker=cfg.train.loss.num_utts_per_speaker,
                max_seq_len=cfg.model.seq_len,
                min_seq_len=cfg.model.min_seq_len,
                variable_length=True,
                specaug_config=cfg.train.specaug,
                rng=rng,
                within_domain_prob=cfg.data.within_domain_prob,
                domain_temp_alpha=cfg.data.domain_temp_alpha,
            )
            state, l_val = step_fn_b(state, b_in, b_lbl, b_m)
            losses_b.append(float(l_val.item()))

        shared_val_embs: dict[str, np.ndarray] = {}
        m1 = evaluation.evaluate_verification_trials(
            state,
            val_trials_1,
            cfg,
            feature_cache=val_cache,
            precomputed_embeddings=shared_val_embs,
        )
        m3 = evaluation.evaluate_multi_enroll_trials(
            state,
            val_trials_3,
            cfg,
            feature_cache=val_cache,
            precomputed_embeddings=shared_val_embs,
        )
        comp_eer = 0.5 * float(m1["eer_percent"]) + 0.5 * float(m3["eer_percent"])
        dt = time.time() - t0
        rec = {
            "name": name,
            "spec": spec,
            "representation_dim": int(cfg.model.output_embedding_dim),
            "num_parameters": neural_net.count_parameters(state.params),
            "val_1utt_eer_percent": float(m1["eer_percent"]),
            "val_3utt_eer_percent": float(m3["eer_percent"]),
            "composite_val_eer_percent": round(comp_eer, 4),
            "val_1utt_metrics": m1,
            "val_3utt_metrics": m3,
            "phase_a_final_loss": round(float(np.mean(losses_a[-50:])), 4),
            "phase_b_final_loss": round(float(np.mean(losses_b[-50:])), 4),
            "elapsed_sec": round(dt, 2),
        }
        results["candidates"][name] = rec
        results["num_completed"] = len(results["candidates"])
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(
            f"[PFAS Sweep {idx:02d}/{len(PFAS_SWEEP_GRID)}] {name}: "
            f"val 1-utt={m1['eer_percent']:.2f}%, 3-utt={m3['eer_percent']:.2f}%, "
            f"composite={comp_eer:.2f}% ({dt:.1f}s)",
            flush=True,
        )

    ranked = sorted(
        results["candidates"].values(),
        key=lambda r: (r["composite_val_eer_percent"], r["val_3utt_eer_percent"]),
    )
    results["ranking"] = [
        {
            "rank": i + 1,
            "name": r["name"],
            "val_1utt_eer_percent": r["val_1utt_eer_percent"],
            "val_3utt_eer_percent": r["val_3utt_eer_percent"],
            "composite_val_eer_percent": r["composite_val_eer_percent"],
            "representation_dim": r["representation_dim"],
        }
        for i, r in enumerate(ranked)
    ]
    results["winner"] = ranked[0]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(
        f"\n[PFAS Sweep] WINNER: {ranked[0]['name']} "
        f"(val 1-utt={ranked[0]['val_1utt_eer_percent']:.2f}%, "
        f"3-utt={ranked[0]['val_3utt_eer_percent']:.2f}%, "
        f"composite={ranked[0]['composite_val_eer_percent']:.2f}%)",
        flush=True,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run wide 16-config PFAS hyperparameter sweep on Conformer-Large."
    )
    parser.add_argument(
        "--trunk_ckpt",
        type=str,
        default=os.path.join(
            REPO_ROOT, "pretrained_models", "univ_full_conformer_large_arcface"
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get(
            "FLAXSPEAKER_DATA_ROOT", os.path.expanduser("~/Data/speaker_datasets")
        ),
    )
    parser.add_argument(
        "--shard_dir",
        type=str,
        default=os.path.expanduser(
            "~/Data/speaker_datasets/sharded_caches/universal_full_v050"
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.path.join(
            REPO_ROOT, "pretrained_models", "pfas_hparam_sweep.json"
        ),
    )
    parser.add_argument("--head_warmup_steps", type=int, default=250)
    parser.add_argument("--joint_finetune_steps", type=int, default=150)
    parser.add_argument("--max_threads", type=int, default=28)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_pfas_sweep(
        trunk_ckpt=args.trunk_ckpt,
        data_root=args.data_root,
        shard_dir=args.shard_dir,
        output_json=args.output_json,
        head_warmup_steps=args.head_warmup_steps,
        joint_finetune_steps=args.joint_finetune_steps,
        max_threads=args.max_threads,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
