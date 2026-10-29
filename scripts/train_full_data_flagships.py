#!/usr/bin/env python3
"""Train & evaluate the 4 Full-Data Universal Flagships (`FlaxSpeaker v0.5.0`, Tier 2).

Trains the 4 full-corpus flagship checkpoints (`FULL_DATA_FLAGSHIP_IDS`) on
`sharded_caches/universal_full_v050` (`VoxCeleb2-dev + VoxCeleb1-dev + CN-Celeb1/2 +
LibriSpeech 960h`, strictly zero speaker overlap with any evaluation benchmark):
1. `univ_full_conformer_large_arcface`:
   - `Conformer-Large` (`6.98M` encoder params, `subsample_factor=1`) + `ArcFace (K=1)` + `3.6s` `LMFT` (`m=0.25`).
   - Saves pre-LMFT converged trunk (`trunk_pre_lmft.msgpack`) for warm-starting `#3` and `#4`.
2. `univ_full_ecapa_c1024_arcface`:
   - `ECAPA-Large` (`c=1024`, `BatchNorm`, `14.66M` encoder params) + `ArcFace (K=1)` + `3.6s` `LMFT` (`m=0.25`).
3. `univ_full_conformer_large_pfas`:
   - Runs wide 16-config `PFAS` sweep (`scripts/run_pfas_hparam_sweep.py`) on `trunk_pre_lmft.msgpack`,
     then trains the winning `PFAS` head + `Conformer-Large` trunk to convergence (`base_speakers_only=False`).
4. `univ_full_conformer_large_dr_vectors`:
   - Warm-starts from `trunk_pre_lmft.msgpack` and trains a `DecisionResidualNetwork` (`Dr-Vectors`)
     under `Extended-Set Softmax` (`base_speakers_only=False`).

Evaluates all 4 flagships + calibrated heterogeneous score fusion (`Conformer-Large + ECAPA-c1024`
and 3-way multi-enroll fusion with `PFAS`, calibrated strictly on `splits/universal_full_val/`) across:
- Official `VoxCeleb1-O` (`veri_test2.txt`, 40 speakers, 37,611 trials — 100% zero-shot)
- Official `CN-Celeb-E` (`eval/lists/trials.lst`, 200 speakers, 3,484,292 trials — 100% zero-shot,
  both concatenated single-enroll `enroll.lst` and multi-utterance `enroll.map`)
- Official `LibriSpeech test-clean` (40 speakers, 10,000 1-utt + 3,000 3-utt trials — 100% zero-shot)
- Controlled 3-domain benchmark suite (`librispeech`, `cnceleb2`, `voxceleb1`, `universal_pooled`
  1-utt & 3-utt + `vox1_o_10spk_subset`)

Strictly enforces polite shared-host resource caps (`<= 28` vCPUs, `nice -n 10`, `< 28 GiB` RSS).
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import sys
import time
from typing import Any, Optional, Sequence

import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import configs  # noqa: E402
from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import evaluation  # noqa: E402
from flaxspeaker import export  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402
from flaxspeaker import neural_net  # noqa: E402
from flaxspeaker import resources  # noqa: E402
from flaxspeaker import scoring  # noqa: E402
from flaxspeaker import shard_cache  # noqa: E402
from scripts import run_pfas_hparam_sweep  # noqa: E402
from scripts import run_research_benchmarks  # noqa: E402


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts} UTC] {msg}", flush=True)


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


def _load_official_vox1_o_trials(data_root: str) -> list[dataset.TrialPair]:
    veri_path = os.path.join(data_root, "voxceleb1", "veri_test2.txt")
    wav_root = os.path.join(data_root, "voxceleb1", "wav")
    trials: list[dataset.TrialPair] = []
    with open(veri_path, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) != 3:
                continue
            trials.append(
                (
                    int(toks[0]),
                    os.path.join(wav_root, toks[1]),
                    os.path.join(wav_root, toks[2]),
                )
            )
    return trials


def _compute_cohort_centroids_from_shard_store(
    state: neural_net.SpeakerTrainState,
    exp_cfg: configs.ExperimentConfig,
    store: shard_cache.ShardedFeatureStore,
    shard_dir: str,
) -> Optional[np.ndarray]:
    """Computes `600` L2-normalized speaker-centroid cohort embeddings from `universal_full_v050`."""
    scoring_t = str(
        getattr(exp_cfg.model.scoring_type, "value", exp_cfg.model.scoring_type)
    )
    if scoring_t != "cosine":
        return None
    cohort_json = os.path.join(shard_dir, "cohort_clean_item_indices.json")
    if not os.path.isfile(cohort_json):
        return None
    with open(cohort_json, "r", encoding="utf-8") as f:
        spk_to_indices: dict[str, list[int]] = json.load(f)

    variables: dict[str, Any] = {"params": state.params}
    if getattr(state, "batch_stats", None):
        variables["batch_stats"] = state.batch_stats

    centroids: list[np.ndarray] = []
    for spk in sorted(spk_to_indices.keys()):
        idxs = spk_to_indices[spk][:5]
        u_embs: list[np.ndarray] = []
        for item_idx in idxs:
            raw_feat = store.get_item(int(item_idx))
            padded, mask = evaluation.prepare_eval_input(raw_feat, exp_cfg)
            emb = np.asarray(
                evaluation._jit_encode_eval_batch(
                    state.apply_fn,
                    variables,
                    jnp.asarray(padded[None, :, :], dtype=jnp.float32),
                    jnp.asarray(mask[None, :], dtype=jnp.float32),
                )[0]
            )
            u_embs.append(emb / (np.linalg.norm(emb) + 1e-6))
        if u_embs:
            c = np.mean(u_embs, axis=0)
            c = (c / (np.linalg.norm(c) + 1e-6)).astype(np.float32)
            centroids.append(c)
    if not centroids:
        return None
    return np.stack(centroids, axis=0)


def _score_single_trials_array(
    state: neural_net.SpeakerTrainState,
    trials: Sequence[dataset.TrialPair],
    exp_cfg: configs.ExperimentConfig,
    utt_to_emb: dict[str, np.ndarray],
    cohort_embs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Computes `(raw_scores, as_norm_scores)` for arbitrary-scale single-utterance trial lists."""
    scoring_t = str(
        getattr(exp_cfg.model.scoring_type, "value", exp_cfg.model.scoring_type)
    )
    unique_utts = sorted({u for _, u1, u2 in trials for u in (u1, u2)})
    u_idx = {u: i for i, u in enumerate(unique_utts)}
    u_mat = np.stack([utt_to_emb[u] for u in unique_utts], axis=0).astype(np.float32)
    n_tr = len(trials)
    raw_scores = np.empty((n_tr,), dtype=np.float64)

    if scoring_t == "pfas":
        import functools
        import jax

        pfas_cfg = exp_cfg.model.pfas
        pfas_scale = (
            float(np.exp(float(state.params["_aux_pfas_log_scale"])))
            if "_aux_pfas_log_scale" in state.params
            else float(pfas_cfg.scale_factor)
        )
        pair_fn = jax.jit(
            jax.vmap(
                functools.partial(
                    scoring.pfas_score_pair,
                    scale_factor=pfas_scale,
                    per_trial_softmax=pfas_cfg.per_trial_softmax,
                    apply_softmax_per_test_key=getattr(
                        pfas_cfg, "apply_softmax_per_test_key", False
                    ),
                    apply_l2_norm_to_keys=getattr(
                        pfas_cfg, "apply_l2_norm_to_keys", True
                    ),
                    apply_l2_norm_to_values=getattr(
                        pfas_cfg, "apply_l2_norm_to_values", False
                    ),
                    apply_global_l2_norm_to_concat_form=getattr(
                        pfas_cfg, "apply_global_l2_norm_to_concat_form", True
                    ),
                    attention_weight_mode=getattr(
                        pfas_cfg, "attention_weight_mode", "exp"
                    ),
                )
            )
        )
        u_keys_j, u_vals_j = scoring.unpack_pfas_representation(
            jnp.asarray(u_mat),
            pfas_cfg.num_keys,
            pfas_cfg.key_dim,
            pfas_cfg.value_dim,
        )
        u_keys = np.asarray(u_keys_j, dtype=np.float32)
        u_vals = np.asarray(u_vals_j, dtype=np.float32)
        for s_i in range(0, n_tr, 16384):
            e_i = min(n_tr, s_i + 16384)
            sub = trials[s_i:e_i]
            ei = np.fromiter((u_idx[u1] for _, u1, _ in sub), dtype=np.int64, count=len(sub))
            ti = np.fromiter((u_idx[u2] for _, _, u2 in sub), dtype=np.int64, count=len(sub))
            raw_scores[s_i:e_i] = np.asarray(
                pair_fn(
                    jnp.asarray(u_keys[ti]),
                    jnp.asarray(u_vals[ti]),
                    jnp.asarray(u_keys[ei]),
                    jnp.asarray(u_vals[ei]),
                )
            )
        return raw_scores, None

    if scoring_t == "dr_vector" and "_aux_dr_net" in state.params:
        import jax

        dr_net = scoring.build_decision_residual_network(exp_cfg.model.dr_vector)
        dr_params = state.params["_aux_dr_net"]
        dr_pair_fn = jax.jit(
            lambda te, ee: dr_net.apply(
                {"params": dr_params}, te, ee, method=dr_net.score_pairs
            )
        )
        for s_i in range(0, n_tr, 16384):
            e_i = min(n_tr, s_i + 16384)
            sub = trials[s_i:e_i]
            ei = np.fromiter((u_idx[u1] for _, u1, _ in sub), dtype=np.int64, count=len(sub))
            ti = np.fromiter((u_idx[u2] for _, _, u2 in sub), dtype=np.int64, count=len(sub))
            raw_scores[s_i:e_i] = np.asarray(
                dr_pair_fn(jnp.asarray(u_mat[ti]), jnp.asarray(u_mat[ei]))
            )
        return raw_scores, None

    u_norm = u_mat / (np.linalg.norm(u_mat, axis=-1, keepdims=True) + 1e-6)
    for s_i in range(0, n_tr, 131072):
        e_i = min(n_tr, s_i + 131072)
        sub = trials[s_i:e_i]
        ei = np.fromiter((u_idx[u1] for _, u1, _ in sub), dtype=np.int64, count=len(sub))
        ti = np.fromiter((u_idx[u2] for _, _, u2 in sub), dtype=np.int64, count=len(sub))
        raw_scores[s_i:e_i] = np.sum(u_norm[ei] * u_norm[ti], axis=-1, dtype=np.float64)

    asn_scores: Optional[np.ndarray] = None
    if cohort_embs is not None:
        cohort = evaluation.require_disjoint_cohort(
            cohort_embs, {u: utt_to_emb[u] for u in unique_utts}
        )
        asn_scores = evaluation._apply_unique_utt_as_norm(
            raw_scores,
            trials,
            {u: utt_to_emb[u] for u in unique_utts},
            cohort,
            top_k=300,
        )
    return raw_scores, asn_scores


def _extract_embs_and_frame_lengths(
    utts: Sequence[str],
    state: neural_net.SpeakerTrainState,
    exp_cfg: configs.ExperimentConfig,
    cache: feature_extraction.FeatureCache,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Extracts full-utterance bucketed embeddings AND active VAD frame counts."""
    fe = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)
    cache.precompute_dataset({"eval": list(utts)}, num_workers=8)
    utt_to_emb, _ = evaluation._extract_eval_embeddings(
        list(utts), state, exp_cfg, fe, feature_cache=cache
    )
    utt_to_frames: dict[str, float] = {}
    for u in utts:
        utt_to_frames[u] = float(cache.get(u).shape[0])
    return utt_to_emb, utt_to_frames


class EvaluationSuiteContext:
    """Preloads and caches all validation, official, and controlled evaluation protocols once."""

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.fe = feature_extraction.AudioFrontend(
            configs.FrontendConfig(), configs.VadConfig()
        )

        # 1. Held-out validation split (`splits/universal_full_val/`)
        val_dir = os.path.join(data_root, "splits", "universal_full_val")
        self.val_trials_1 = dataset.load_trials_file(
            os.path.join(val_dir, "trials.tsv")
        )
        self.val_trials_3 = _load_multi_enroll_tsv(
            os.path.join(val_dir, "multi_enroll_trials.tsv")
        )
        self.val_cache = feature_extraction.FeatureCache(
            self.fe,
            cache_dir=os.path.join(data_root, "feature_cache", "universal_full_val"),
        )
        val_utts = sorted(
            {u for _, u1, u2 in self.val_trials_1 for u in (u1, u2)}
            | {u for _, el, tu in self.val_trials_3 for u in list(el) + [tu]}
        )
        self.val_cache.precompute_dataset({"val": val_utts}, num_workers=8)

        # 2. Controlled 3-domain benchmark suite (`librispeech`, `cnceleb2`, `voxceleb1`, `universal`)
        with open(
            os.path.join(data_root, "splits", "dataset_summary.json"),
            "r",
            encoding="utf-8",
        ) as f:
            ds_sum = json.load(f)
        self.ctrl_caches: dict[str, feature_extraction.FeatureCache] = {}
        self.ctrl_trials_1: dict[str, list[dataset.TrialPair]] = {}
        self.ctrl_trials_3: dict[str, list[dataset.MultiEnrollTrial]] = {}
        for dom in ("librispeech", "cnceleb2", "voxceleb1"):
            self.ctrl_caches[dom] = feature_extraction.FeatureCache(
                self.fe, cache_dir=os.path.join(data_root, "feature_cache", dom)
            )
            eval_map = dataset.get_csv_spk_to_utts(ds_sum[dom]["eval_csv"])
            self.ctrl_caches[dom].precompute_dataset(eval_map, num_workers=8)
            self.ctrl_trials_1[dom] = dataset.load_trials_file(ds_sum[dom]["trials_file"])
            self.ctrl_trials_3[dom] = _load_multi_enroll_tsv(
                dataset.resolve_data_path(ds_sum[dom]["multi_trials_file"])
            )
        self.ctrl_caches["universal"] = feature_extraction.FeatureCache.merged(
            self.fe,
            [self.ctrl_caches[d] for d in ("librispeech", "cnceleb2", "voxceleb1")],
            cache_dir=os.path.join(data_root, "feature_cache", "universal"),
        )
        self.ctrl_trials_1["universal"] = dataset.load_trials_file(
            ds_sum["universal"]["trials_file"]
        )
        self.ctrl_trials_3["universal"] = _load_multi_enroll_tsv(
            dataset.resolve_data_path(ds_sum["universal"]["multi_trials_file"])
        )
        self.vox_subset_trials = dataset.load_trials_file(
            ds_sum["voxceleb1"].get(
                "vox1_o_10spk_subset_trials_file",
                ds_sum["voxceleb1"]["official_trials_file"],
            )
        )

        # 3. Official VoxCeleb1-O (`37,611` trials across all 40 test speakers; lazy-loaded on demand)
        self.vox1_o_trials = _load_official_vox1_o_trials(data_root)
        self.vox1_o_cache = feature_extraction.FeatureCache(
            self.fe,
            cache_dir=os.path.join(data_root, "feature_cache", "voxceleb1_o_full"),
        )

        # 4. Official LibriSpeech `test-clean` (`10,000` 1-utt + `3,000` 3-utt trials; lazy-loaded)
        off_ls_dir = os.path.join(data_root, "splits", "official_eval")
        self.ls_off_trials_1 = dataset.load_trials_file(
            os.path.join(off_ls_dir, "librispeech_test_clean_trials.tsv")
        )
        self.ls_off_trials_3 = _load_multi_enroll_tsv(
            os.path.join(off_ls_dir, "librispeech_test_clean_3utt_trials.tsv")
        )
        self.ls_off_cache = feature_extraction.FeatureCache(
            self.fe,
            cache_dir=os.path.join(
                data_root, "feature_cache", "librispeech_test_clean_full"
            ),
        )

        # 5. Official CN-Celeb-E (`3,484,292` trials across 196 enrolls / 200 eval speakers; lazy-loaded)
        cn1_root = os.path.join(data_root, "full_corpora", "CN-Celeb_flac")
        _log("Loading official CN-Celeb-E trial lists (single-concat + multi-utt enroll.map)...")
        self.cn_e_multi_trials, self.cn_e_single_trials = (
            dataset.load_cnceleb_eval_trials(cn1_root)
        )
        self.cn_e_cache = feature_extraction.FeatureCache(
            self.fe,
            cache_dir=os.path.join(data_root, "feature_cache", "cnceleb_e_official"),
        )


def evaluate_flagship_comprehensive(
    model_id: str,
    state: neural_net.SpeakerTrainState,
    exp_cfg: configs.ExperimentConfig,
    eval_ctx: EvaluationSuiteContext,
    store_for_cohort: shard_cache.ShardedFeatureStore,
    shard_dir: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluates one flagship checkpoint across all official & controlled benchmarks.

    Returns `(eval_bundle, score_vectors_for_fusion)`.
    """
    t0 = time.time()
    cohort_embs = _compute_cohort_centroids_from_shard_store(
        state, exp_cfg, store_for_cohort, shard_dir
    )

    # Fit QMF calibrator strictly on held-out `universal_full_val`
    val_utts = sorted({u for _, u1, u2 in eval_ctx.val_trials_1 for u in (u1, u2)})
    val_emb, val_frames = _extract_embs_and_frame_lengths(
        val_utts, state, exp_cfg, eval_ctx.val_cache
    )
    val_labels = [int(lbl) for lbl, _, _ in eval_ctx.val_trials_1]
    val_raw, val_asn = _score_single_trials_array(
        state, eval_ctx.val_trials_1, exp_cfg, val_emb, cohort_embs=cohort_embs
    )
    val_base_scores = val_asn if val_asn is not None else val_raw
    val_ef = np.array([val_frames[u1] for _, u1, _ in eval_ctx.val_trials_1])
    val_tf = np.array([val_frames[u2] for _, _, u2 in eval_ctx.val_trials_1])
    qmf = scoring.QualityMeasureCalibrator(c_reg=1.0).fit(
        val_base_scores, val_labels, enroll_frames=val_ef, test_frames=val_tf
    )

    _, val_3_scores = evaluation.evaluate_multi_enroll_trials(
        state,
        eval_ctx.val_trials_3,
        exp_cfg,
        feature_cache=eval_ctx.val_cache,
        cohort_embeddings=cohort_embs,
        include_as_norm_breakdown=(cohort_embs is not None),
        return_scores=True,
    )

    score_Bank: dict[str, np.ndarray] = {
        "val_1utt": val_base_scores,
        "val_1utt_qmf": qmf.predict(
            val_base_scores, enroll_frames=val_ef, test_frames=val_tf
        ),
        "val_3utt": val_3_scores,
    }

    def _eval_official_single(
        b_name: str,
        trials: list[dataset.TrialPair],
        cache: feature_extraction.FeatureCache,
        with_ci: bool,
    ) -> dict[str, Any]:
        utts = sorted({u for _, u1, u2 in trials for u in (u1, u2)})
        u_emb, u_frames = _extract_embs_and_frame_lengths(utts, state, exp_cfg, cache)
        labels = [int(lbl) for lbl, _, _ in trials]
        raw_s, asn_s = _score_single_trials_array(
            state, trials, exp_cfg, u_emb, cohort_embs=cohort_embs
        )
        m_dict = evaluation._compute_metrics_dict(
            labels, raw_s, compute_ci=with_ci
        )
        m_dict["num_trials"] = len(trials)
        m_dict["num_unique_utterances"] = len(utts)
        ef = np.array([u_frames[u1] for _, u1, _ in trials])
        tf = np.array([u_frames[u2] for _, _, u2 in trials])
        best_s = raw_s
        if asn_s is not None:
            asn_m = evaluation._compute_metrics_dict(
                labels, asn_s, compute_ci=with_ci
            )
            asn_m["top_k"] = 300
            m_dict["as_norm"] = asn_m
            best_s = asn_s
        qmf_s = qmf.predict(best_s, enroll_frames=ef, test_frames=tf)
        m_dict["qmf"] = evaluation._compute_metrics_dict(
            labels, qmf_s, compute_ci=with_ci
        )
        score_Bank[b_name] = best_s
        score_Bank[f"{b_name}_qmf"] = qmf_s
        return m_dict

    _log(f"[{model_id}] Scoring official VoxCeleb1-O (37,611 trials)...")
    vox1_o_res = _eval_official_single(
        "voxceleb1_o", eval_ctx.vox1_o_trials, eval_ctx.vox1_o_cache, with_ci=True
    )
    eval_ctx.vox1_o_cache._cache.clear()
    gc.collect()

    _log(f"[{model_id}] Scoring official CN-Celeb-E (3,484,292 trials)...")
    cn_e_res = _eval_official_single(
        "cnceleb_e",
        eval_ctx.cn_e_single_trials,
        eval_ctx.cn_e_cache,
        with_ci=False,
    )
    # Also evaluate multi-utterance CN-Celeb-E (`enroll.map`)
    cn_e_all_utts = sorted(
        {
            u
            for _, el, tu in eval_ctx.cn_e_multi_trials
            for u in list(el) + [tu]
        }
    )
    cn_e_m_emb, _ = evaluation._extract_eval_embeddings(
        cn_e_all_utts, state, exp_cfg, eval_ctx.fe, feature_cache=eval_ctx.cn_e_cache
    )
    cn_e_multi_res, cn_e_multi_scores = evaluation.evaluate_multi_enroll_trials(
        state,
        eval_ctx.cn_e_multi_trials,
        exp_cfg,
        feature_cache=eval_ctx.cn_e_cache,
        precomputed_embeddings=cn_e_m_emb,
        cohort_embeddings=cohort_embs,
        include_as_norm_breakdown=(cohort_embs is not None),
        return_scores=True,
    )
    cn_e_res["multi_enroll_map"] = cn_e_multi_res
    score_Bank["cnceleb_e_multi"] = cn_e_multi_scores
    del cn_e_m_emb
    eval_ctx.cn_e_cache._cache.clear()
    gc.collect()

    _log(f"[{model_id}] Scoring official LibriSpeech test-clean (10,000 1-utt + 3,000 3-utt)...")
    ls_off_res = _eval_official_single(
        "librispeech_test_clean",
        eval_ctx.ls_off_trials_1,
        eval_ctx.ls_off_cache,
        with_ci=True,
    )
    ls_off_3_res, ls_off_3_scores = evaluation.evaluate_multi_enroll_trials(
        state,
        eval_ctx.ls_off_trials_3,
        exp_cfg,
        feature_cache=eval_ctx.ls_off_cache,
        cohort_embeddings=cohort_embs,
        include_as_norm_breakdown=(cohort_embs is not None),
        return_scores=True,
    )
    ls_off_res["multi_enroll_3utt"] = ls_off_3_res
    score_Bank["librispeech_3utt"] = ls_off_3_scores
    eval_ctx.ls_off_cache._cache.clear()
    gc.collect()

    _log(f"[{model_id}] Scoring controlled 3-domain benchmark suite...")
    shared_ctrl_embs: dict[str, np.ndarray] = {}
    md_eval = run_research_benchmarks.evaluate_all_domains(
        state,
        exp_cfg,
        eval_ctx.ctrl_caches,
        eval_ctx.ctrl_trials_1,
        eval_ctx.ctrl_trials_3,
        shared_embs=shared_ctrl_embs,
        cohort_embeddings=cohort_embs,
    )
    vox_sub_metrics = evaluation.evaluate_verification_trials(
        state,
        eval_ctx.vox_subset_trials,
        exp_cfg,
        feature_cache=eval_ctx.ctrl_caches["voxceleb1"],
        precomputed_embeddings=shared_ctrl_embs,
        cohort_embeddings=cohort_embs,
        include_as_norm_breakdown=(cohort_embs is not None),
    )
    md_eval["domains_1utt"]["vox1_o_10spk_subset"] = vox_sub_metrics

    official_benchmarks = {
        "voxceleb1_o": vox1_o_res,
        "cnceleb_e": cn_e_res,
        "librispeech_test_clean": ls_off_res,
    }
    vox_asn_eer = (vox1_o_res.get("as_norm") or vox1_o_res)["eer_percent"]
    cn_asn_eer = (cn_e_res.get("as_norm") or cn_e_res)["eer_percent"]
    ls_asn_eer = (ls_off_res.get("as_norm") or ls_off_res)["eer_percent"]
    _log(
        f"[{model_id}] Zero-shot summary ({time.time() - t0:.1f}s): "
        f"Vox1-O Raw={vox1_o_res['eer_percent']:.2f}% / "
        f"AS-Norm={vox_asn_eer:.2f}% / QMF={vox1_o_res['qmf']['eer_percent']:.2f}% | "
        f"CN-Celeb-E Raw={cn_e_res['eer_percent']:.2f}% / "
        f"AS-Norm={cn_asn_eer:.2f}% / MultiEnroll={cn_e_multi_res['eer_percent']:.2f}% | "
        f"LS-test Raw={ls_off_res['eer_percent']:.2f}% / AS-Norm={ls_asn_eer:.2f}%"
    )
    return {
        "official_benchmarks": official_benchmarks,
        "multi_domain_eval": md_eval,
        "vox1_o_10spk_subset": vox_sub_metrics,
        "val_1utt_raw_eer_percent": round(
            evaluation.compute_eer_exact(val_labels, val_raw)[0] * 100.0, 4
        ),
        "val_1utt_calibrated_eer_percent": round(
            evaluation.compute_eer_exact(
                val_labels, score_Bank["val_1utt_qmf"]
            )[0]
            * 100.0,
            4,
        ),
    }, score_Bank


def _save_step_resume_ckpt(
    ckpt_dir: str,
    step_done: int,
    state: neural_net.SpeakerTrainState,
    loss_curve: list[float],
    val_history: list[dict[str, Any]],
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    neural_net.save_model(
        os.path.join(ckpt_dir, "resume_weights.msgpack"),
        state.params,
        batch_stats=state.batch_stats,
    )
    with open(os.path.join(ckpt_dir, "resume_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "step_done": int(step_done),
                "loss_curve": [round(float(x), 5) for x in loss_curve],
                "val_history": val_history,
            },
            f,
        )


def _try_load_step_resume_ckpt(
    ckpt_dir: str, state: neural_net.SpeakerTrainState
) -> tuple[int, neural_net.SpeakerTrainState, list[float], list[dict[str, Any]]]:
    meta_p = os.path.join(ckpt_dir, "resume_meta.json")
    w_p = os.path.join(ckpt_dir, "resume_weights.msgpack")
    if not (os.path.isfile(meta_p) and os.path.isfile(w_p)):
        return 0, state, [], []
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    template = (
        {"params": state.params, "batch_stats": state.batch_stats}
        if state.batch_stats
        else {"params": state.params}
    )
    loaded = neural_net.load_model(w_p, template)
    if isinstance(loaded, dict) and "params" in loaded:
        state = state.replace(
            params=loaded["params"],
            batch_stats=loaded.get("batch_stats", state.batch_stats) or state.batch_stats,
        )
    else:
        state = state.replace(params=loaded)
    step_done = int(meta["step_done"])
    _log(f"Resumed training checkpoint from {ckpt_dir} at step {step_done}.")
    return (
        step_done,
        state,
        [float(x) for x in meta.get("loss_curve", [])],
        list(meta.get("val_history", [])),
    )


class _AnchorSubsetStore:
    """Zero-copy wrapper restricting `sample_classification_batch` to `anchor_spks` (`0..N-1`)."""

    def __init__(
        self,
        parent: shard_cache.ShardedFeatureStore,
        anchor_spks: Sequence[int],
    ):
        self.parent = parent
        self.anchor_spks = [int(s) for s in anchor_spks]
        self.num_speakers = len(self.anchor_spks)
        self.feat_dim = parent.feat_dim
        dom_map: dict[int, list[int]] = {}
        for local_idx, orig_spk in enumerate(self.anchor_spks):
            d_id = int(parent.spk_domain_id[orig_spk])
            dom_map.setdefault(d_id, []).append(local_idx)
        self.domain_speakers: dict[int, np.ndarray] = {
            d: np.asarray(v, dtype=np.int32) for d, v in dom_map.items()
        }

    def get_speaker_items(self, local_spk_id: int) -> np.ndarray:
        return self.parent.get_speaker_items(self.anchor_spks[int(local_spk_id)])

    def sample_classification_batch(
        self,
        batch_size: int,
        max_seq_len: int,
        min_seq_len: Optional[int] = None,
        variable_length: bool = True,
        specaug_config: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        domain_temp_alpha: float = 0.35,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        gen = rng if rng is not None else np.random.default_rng()
        min_len = (
            max(16, min(int(min_seq_len), max_seq_len))
            if (variable_length and min_seq_len is not None)
            else max_seq_len
        )
        dom_keys = sorted(self.domain_speakers.keys())
        raw_w = np.array(
            [len(self.domain_speakers[d]) ** domain_temp_alpha for d in dom_keys],
            dtype=np.float64,
        )
        dom_probs = raw_w / np.sum(raw_w)

        feats = np.empty((batch_size, max_seq_len, self.feat_dim), dtype=np.float32)
        masks = np.empty((batch_size, max_seq_len), dtype=np.float32)
        labels = np.empty((batch_size,), dtype=np.int32)

        for b in range(batch_size):
            d_choice = dom_keys[int(gen.choice(len(dom_keys), p=dom_probs))]
            spks_d = self.domain_speakers[d_choice]
            local_spk = int(spks_d[int(gen.integers(0, spks_d.size))])
            items = self.get_speaker_items(local_spk)
            item_idx = int(items[int(gen.integers(0, items.size))])
            t_len = (
                int(gen.integers(min_len, max_seq_len + 1))
                if (variable_length and min_len < max_seq_len)
                else max_seq_len
            )
            padded, mask = self.parent._read_cropped_padded(
                item_idx, t_len, max_seq_len, min_len, specaug_config, gen
            )
            feats[b] = padded
            masks[b] = mask
            labels[b] = local_spk

        return (
            jnp.asarray(feats, dtype=jnp.float32),
            jnp.asarray(labels, dtype=jnp.int32),
            jnp.asarray(masks, dtype=jnp.float32),
        )


def _select_balanced_anchor_speakers(
    store_base: shard_cache.ShardedFeatureStore,
    total_base_speakers: int = 400,
) -> list[int]:
    """Selects `total_base_speakers` uniformly strided multi-corpus base speakers x 3 speeds."""
    id_to_spk = {v: k for k, v in store_base.speaker_to_id.items()}
    anchor_spks: list[int] = []
    dom_keys = sorted(store_base.domain_speakers.keys())
    for d_idx, d_id in enumerate(dom_keys):
        cands = [
            int(s)
            for s in store_base.domain_speakers[d_id]
            if "@sp" not in id_to_spk[int(s)] and store_base.spk_utt_counts[int(s)] >= 60
        ]
        n_pick = (total_base_speakers // len(dom_keys)) + (
            1 if d_idx == len(dom_keys) - 1 and total_base_speakers % len(dom_keys) != 0 else 0
        )
        idx = np.linspace(0, len(cands) - 1, min(len(cands), n_pick), dtype=int)
        for i in idx:
            base_name = id_to_spk[cands[int(i)]]
            for suf in ("", "@sp0.9", "@sp1.1"):
                full_name = f"{base_name}{suf}"
                if full_name in store_base.speaker_to_id:
                    anchor_spks.append(int(store_base.speaker_to_id[full_name]))
    return anchor_spks


def _compute_subcenter_init_prototypes(
    state: neural_net.SpeakerTrainState,
    anchor_store: _AnchorSubsetStore,
    utts_per_centroid: int = 8,
    emb_dim: int = 256,
    crop_frames: int = 180,
) -> np.ndarray:
    """Initializes `(emb_dim, num_anchors)` L2-normalized multi-utterance speaker centroids from encoder."""
    import jax

    variables: dict[str, Any] = {
        "params": {
            k: v for k, v in state.params.items() if not k.startswith("_aux_")
        }
    }
    if getattr(state, "batch_stats", None):
        variables["batch_stats"] = state.batch_stats

    enc_fn = jax.jit(
        lambda v, x, m: state.apply_fn(v, x, deterministic=True, mask=m)
    )
    rng_init = np.random.default_rng(777)
    n_spks = anchor_store.num_speakers
    w_init = np.zeros((emb_dim, n_spks), dtype=np.float32)

    batch_f: list[np.ndarray] = []
    batch_m: list[np.ndarray] = []
    batch_col: list[int] = []

    def _flush() -> None:
        if not batch_f:
            return
        b_sz = len(batch_f)
        pad_sz = 64 if b_sz <= 64 else b_sz
        fx = np.zeros((pad_sz, crop_frames, anchor_store.feat_dim), dtype=np.float32)
        mx = np.ones((pad_sz, crop_frames), dtype=np.float32)
        fx[:b_sz] = np.stack(batch_f, axis=0)
        mx[:b_sz] = np.stack(batch_m, axis=0)
        embs = np.asarray(
            enc_fn(
                variables,
                jnp.asarray(fx, dtype=jnp.float32),
                jnp.asarray(mx, dtype=jnp.float32),
            )
        )[:b_sz]
        embs = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-6)
        for row_i, col_idx in enumerate(batch_col):
            w_init[:, col_idx] += embs[row_i]
        batch_f.clear()
        batch_m.clear()
        batch_col.clear()

    for s_idx in range(n_spks):
        items = anchor_store.get_speaker_items(s_idx)
        chosen = items[
            np.linspace(0, len(items) - 1, min(len(items), utts_per_centroid), dtype=int)
        ]
        for it_idx in chosen:
            p, m = anchor_store.parent._read_cropped_padded(
                int(it_idx), crop_frames, crop_frames, crop_frames, None, rng_init
            )
            batch_f.append(p)
            batch_m.append(m)
            batch_col.append(s_idx)
            if len(batch_f) == 64:
                _flush()
    _flush()
    w_init = w_init / (np.linalg.norm(w_init, axis=0, keepdims=True) + 1e-6)
    return w_init


def _train_arcface_plus_lmft_flagship(
    model_id: str,
    backbone: str,
    shard_dir: str,
    store_base: shard_cache.ShardedFeatureStore,
    eval_ctx: EvaluationSuiteContext,
    stage1a_steps: int,
    stage1b_steps: int,
    lmft_steps: int,
    save_pre_lmft_trunk_path: str = "",
    resume: bool = True,
) -> tuple[
    neural_net.SpeakerTrainState,
    configs.ExperimentConfig,
    list[float],
    list[dict[str, Any]],
    float,
    dict[str, Any],
]:
    """3-stage training: Stage 1A Metric Pretrain -> Stage 1B Centroid ArcFace (K=1) + Open-Set Reg -> Stage 2 LMFT."""
    cfg = configs.ExperimentConfig()
    cfg.train.seed = 42
    cfg.data.dataset_name = "universal_full"
    cfg.data.sharded_cache_dir = "data/speaker_datasets/sharded_caches/universal_full_v050"
    cfg.data.within_domain_prob = 0.75
    cfg.data.domain_temp_alpha = 0.35

    cfg.frontend.feature_type = configs.FeatureType.LOG_MEL.value
    cfg.frontend.n_mels = 80
    cfg.frontend.n_mfcc = 80
    cfg.frontend.apply_cmvn = True
    cfg.vad.enabled = True
    cfg.vad.mode = configs.VadMode.ENERGY.value

    cfg.model.backbone = backbone
    cfg.model.apply_size_variant(configs.ModelSize.LARGE)
    cfg.model.seq_len = 180
    cfg.model.min_seq_len = 90
    cfg.model.max_seq_len = 180
    cfg.model.max_eval_seq_len = 0
    cfg.model.variable_length_training = True
    cfg.model.pooling.pooling_type = (
        configs.PoolingType.ASP.value
        if backbone == "ecapa_tdnn"
        else configs.PoolingType.ATP.value
    )
    cfg.model.scoring_type = configs.ScoringType.COSINE.value
    cfg.eval.full_sequence_inference = True
    cfg.eval.max_eval_frames_cap = 3000
    cfg.export.sequence_length = 180

    # Stage 1A: Metric Pretraining across all 36,579 speed-perturbed classes (`extended_set_softmax`, 16 x 4 = 64)
    s1a_warmup = min(120, max(50, stage1a_steps // 8))
    step0_overrides: dict[str, Any] = {
        "loss_type": "extended_set_softmax",
        "scoring_type": "cosine",
        "num_speakers_per_batch": 16,
        "num_utts_per_speaker": 4,
        "batch_size": 64,
        "seq_len": 180,
        "min_seq_len": 90,
        "max_seq_len": 180,
        "optimizer": "adamw",
        "learning_rate": 1.2e-3,
        "min_learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "grad_clip_norm": 3.0,
        "warmup_steps": int(s1a_warmup),
        "num_steps": int(stage1a_steps),
    }
    cfg.train.optimizer = "adamw"
    cfg.train.learning_rate = 1.2e-3
    cfg.train.min_learning_rate = 2.0e-4
    cfg.train.weight_decay = 1.0e-4
    cfg.train.grad_clip_norm = 3.0
    cfg.train.lr_schedule_cosine = True
    cfg.train.warmup_steps = int(s1a_warmup)
    cfg.train.num_steps = int(stage1a_steps)
    cfg.train.loss.loss_type = "extended_set_softmax"
    cfg.train.loss.scoring_type = "cosine"
    cfg.train.loss.num_speakers_per_batch = 16
    cfg.train.loss.num_utts_per_speaker = 4
    cfg.train.batch_size = 64
    cfg.loss = cfg.train.loss

    _, state = neural_net.get_speaker_encoder(cfg, num_classes=0)
    resume_dir = os.path.join(REPO_ROOT, "pretrained_models", model_id, "_resume")
    start_step = 0
    loss_curve: list[float] = []
    val_history: list[dict[str, Any]] = []
    if resume:
        start_step, state, loss_curve, val_history = _try_load_step_resume_ckpt(
            resume_dir, state
        )

    best_val_eer = 100.0
    best_enc_params = {
        k: v for k, v in state.params.items() if not k.startswith("_aux_")
    }
    best_batch_stats = state.batch_stats
    for vr in val_history:
        if float(vr.get("val_1utt_eer_percent", 100.0)) < best_val_eer:
            best_val_eer = float(vr["val_1utt_eer_percent"])

    t_train0 = time.time()
    if start_step < stage1a_steps:
        step_fn_1a = neural_net.make_modern_train_step(cfg)
        rng_1a = np.random.default_rng(cfg.train.seed + start_step)
        for step in range(start_step, stage1a_steps):
            b_in, b_lbl, b_m = store_base.sample_nm_batch(
                num_speakers=cfg.train.loss.num_speakers_per_batch,
                num_utts_per_speaker=cfg.train.loss.num_utts_per_speaker,
                max_seq_len=cfg.model.seq_len,
                min_seq_len=cfg.model.min_seq_len,
                variable_length=True,
                specaug_config=cfg.train.specaug,
                rng=rng_1a,
                within_domain_prob=cfg.data.within_domain_prob,
                domain_temp_alpha=cfg.data.domain_temp_alpha,
            )
            state, l_val = step_fn_1a(state, b_in, b_lbl, b_m)
            loss_s = float(l_val.item())
            loss_curve.append(loss_s)
            step_num = step + 1

            if step_num == 1 or step_num % 200 == 0 or step_num == stage1a_steps:
                mean100 = float(np.mean(loss_curve[-min(100, len(loss_curve)):]))
                _log(
                    f"[{model_id}] Stage-1A (Extended-Set Metric across 36,579 classes) "
                    f"step {step_num}/{stage1a_steps} loss={loss_s:.4f} "
                    f"(mean100={mean100:.4f}, elapsed={time.time() - t_train0:.1f}s)"
                )

            if step_num % 300 == 0 or step_num == stage1a_steps:
                t_v0 = time.time()
                v_m1 = evaluation.evaluate_verification_trials(
                    state, eval_ctx.val_trials_1, cfg, feature_cache=eval_ctx.val_cache
                )
                v_eer = float(v_m1["eer_percent"])
                val_history.append(
                    {
                        "stage": "stage1a_extended_set_metric",
                        "step": step_num,
                        "val_1utt_eer_percent": v_eer,
                        "mean100_loss": round(
                            float(np.mean(loss_curve[-min(100, len(loss_curve)):])), 4
                        ),
                    }
                )
                if v_eer <= best_val_eer:
                    best_val_eer = v_eer
                    best_enc_params = {
                        k: v for k, v in state.params.items() if not k.startswith("_aux_")
                    }
                    best_batch_stats = state.batch_stats
                _save_step_resume_ckpt(
                    resume_dir, step_num, state, loss_curve, val_history
                )
                t_v_dt = time.time() - t_v0
                t_train0 += t_v_dt
                _log(
                    f"[{model_id} @ Stage-1A step {step_num}] Held-out Val 1-Utt EER={v_eer:.2f}% "
                    f"(best={best_val_eer:.2f}%, eval {t_v_dt:.1f}s)"
                )

    aux_w_1a = state.params.get("_aux_w", jnp.array(10.0, dtype=jnp.float32))
    aux_b_1a = state.params.get("_aux_b", jnp.array(-5.0, dtype=jnp.float32))
    state = state.replace(params=dict(best_enc_params), batch_stats=best_batch_stats)

    # Stage 1B: 8-Utt Centroid-Initialized ArcFace (K=1) across 1,200 stratified speed-perturbed classes + Open-Set Reg
    anchor_spks = _select_balanced_anchor_speakers(store_base, total_base_speakers=400)
    anchor_store = _AnchorSubsetStore(store_base, anchor_spks)
    _log(
        f"[{model_id}] Initializing Stage-1B 8-utt speaker-centroid ArcFace (K=1) prototypes across "
        f"{anchor_store.num_speakers} stratified speed-perturbed classes..."
    )
    t_init0 = time.time()
    w_init = _compute_subcenter_init_prototypes(
        state, anchor_store, utts_per_centroid=8, emb_dim=cfg.model.output_embedding_dim
    )
    _log(
        f"[{model_id}] Computed {w_init.shape} ArcFace (K=1) speaker centroids in {time.time() - t_init0:.1f}s."
    )

    cfg.train.batch_size = 64
    cfg.train.num_steps = int(stage1b_steps)
    cfg.train.lr_schedule_cosine = True
    cfg.train.optimizer = "adamw"
    cfg.train.learning_rate = 1.0e-3 if backbone == "conformer" else 1.2e-3
    cfg.train.min_learning_rate = 1.5e-5
    cfg.train.weight_decay = 1.0e-4
    cfg.train.grad_clip_norm = 3.0
    cfg.train.warmup_steps = min(150, max(50, stage1b_steps // 16))

    cfg.train.loss.loss_type = "arcface"
    cfg.train.loss.scoring_type = "cosine"
    cfg.train.loss.subcenter_k = 1
    cfg.train.loss.scale = 30.0
    cfg.train.loss.margin = 0.20
    cfg.train.loss.margin_warmup_steps = min(300, max(50, stage1b_steps // 8))
    cfg.loss = cfg.train.loss

    # Also build open-set metric step across all 36,579 speed-perturbed classes (run every 6th step)
    import copy

    cfg_ext = copy.deepcopy(cfg)
    cfg_ext.train.loss.loss_type = "extended_set_softmax"
    cfg_ext.train.loss.scoring_type = "cosine"
    cfg_ext.train.loss.num_speakers_per_batch = 16
    cfg_ext.train.loss.num_utts_per_speaker = 4
    cfg_ext.loss = cfg_ext.train.loss

    params_1b = dict(best_enc_params)
    params_1b["_aux_class_weights"] = jnp.asarray(w_init, dtype=jnp.float32)
    params_1b["_aux_w"] = aux_w_1a
    params_1b["_aux_b"] = aux_b_1a
    tx_1b = neural_net.build_optimizer(cfg)
    state = neural_net.SpeakerTrainState.create(
        apply_fn=state.apply_fn,
        params=params_1b,
        tx=tx_1b,
        batch_stats=best_batch_stats,
    )
    if resume and start_step > stage1a_steps:
        _, state, loss_curve, val_history = _try_load_step_resume_ckpt(
            resume_dir, state
        )
    best_aux_w = state.params["_aux_class_weights"]
    step_fn_1b = neural_net.make_modern_train_step(cfg)
    step_fn_1b_ext = neural_net.make_modern_train_step(cfg_ext)
    s1b_start = max(0, start_step - stage1a_steps)
    rng_1b = np.random.default_rng(cfg.train.seed + stage1a_steps + s1b_start)

    for step in range(s1b_start, stage1b_steps):
        step_num = step + 1
        if step_num % 6 == 0:
            prev_aux_w = state.params["_aux_class_weights"]
            b_in, b_lbl, b_m = store_base.sample_nm_batch(
                num_speakers=16,
                num_utts_per_speaker=4,
                max_seq_len=cfg.model.seq_len,
                min_seq_len=cfg.model.min_seq_len,
                variable_length=True,
                specaug_config=cfg.train.specaug,
                rng=rng_1b,
                within_domain_prob=cfg.data.within_domain_prob,
                domain_temp_alpha=cfg.data.domain_temp_alpha,
            )
            state, l_val = step_fn_1b_ext(state, b_in, b_lbl, b_m)
            new_p = dict(state.params)
            new_p["_aux_class_weights"] = prev_aux_w
            state = state.replace(params=new_p)
        else:
            prev_w = state.params["_aux_w"]
            prev_b = state.params["_aux_b"]
            b_in, b_lbl, b_m = anchor_store.sample_classification_batch(
                batch_size=cfg.train.batch_size,
                max_seq_len=cfg.model.seq_len,
                min_seq_len=cfg.model.min_seq_len,
                variable_length=True,
                specaug_config=cfg.train.specaug,
                rng=rng_1b,
                domain_temp_alpha=cfg.data.domain_temp_alpha,
            )
            state, l_val = step_fn_1b(state, b_in, b_lbl, b_m)
            new_p = dict(state.params)
            new_p["_aux_w"] = prev_w
            new_p["_aux_b"] = prev_b
            state = state.replace(params=new_p)
        loss_s = float(l_val.item())
        loss_curve.append(loss_s)

        if step_num == 1 or step_num % 200 == 0 or step_num == stage1b_steps:
            mean100 = float(np.mean(loss_curve[-min(100, len(loss_curve)):]))
            _log(
                f"[{model_id}] Stage-1B (ArcFace K=1 + Open-Set 36,579-cls Reg) "
                f"step {step_num}/{stage1b_steps} loss={loss_s:.4f} "
                f"(mean100={mean100:.4f}, elapsed={time.time() - t_train0:.1f}s)"
            )

        if step_num % 400 == 0 or step_num == stage1b_steps:
            t_v0 = time.time()
            v_m1 = evaluation.evaluate_verification_trials(
                state, eval_ctx.val_trials_1, cfg, feature_cache=eval_ctx.val_cache
            )
            v_eer = float(v_m1["eer_percent"])
            val_history.append(
                {
                    "stage": "stage1b_arcface_plus_openset",
                    "step": stage1a_steps + step_num,
                    "val_1utt_eer_percent": v_eer,
                    "mean100_loss": round(
                        float(np.mean(loss_curve[-min(100, len(loss_curve)):])), 4
                    ),
                }
            )
            if v_eer <= best_val_eer:
                best_val_eer = v_eer
                best_enc_params = {
                    k: v for k, v in state.params.items() if not k.startswith("_aux_")
                }
                best_batch_stats = state.batch_stats
                best_aux_w = state.params["_aux_class_weights"]
            _save_step_resume_ckpt(
                resume_dir, stage1a_steps + step_num, state, loss_curve, val_history
            )
            t_v_dt = time.time() - t_v0
            t_train0 += t_v_dt
            _log(
                f"[{model_id} @ Stage-1B step {step_num}] Held-out Val 1-Utt EER={v_eer:.2f}% "
                f"(best={best_val_eer:.2f}%, eval {t_v_dt:.1f}s)"
            )

    restored_p = dict(best_enc_params)
    restored_p["_aux_class_weights"] = best_aux_w
    state = state.replace(params=restored_p, batch_stats=best_batch_stats)

    actual_stage1_steps = len(loss_curve)
    if save_pre_lmft_trunk_path:
        os.makedirs(os.path.dirname(save_pre_lmft_trunk_path), exist_ok=True)
        neural_net.save_model(
            save_pre_lmft_trunk_path, best_enc_params, batch_stats=best_batch_stats
        )
        _log(f"[{model_id}] Saved pre-LMFT converged trunk -> {save_pre_lmft_trunk_path}")

    if lmft_steps > 0:
        lmft_lr = 2.5e-5
        _log(
            f"[{model_id}] Starting Stage-2 3.6s LMFT ({lmft_steps} steps, seq_len=360, m=0.25, lr={lmft_lr})..."
        )
        state, lmft_losses = neural_net.run_lmft_stage(
            state=state,
            base_cfg=cfg,
            shard_store=anchor_store,
            lmft_steps=lmft_steps,
            lmft_seq_len=360,
            lmft_batch_size=64,
            lmft_lr=lmft_lr,
            lmft_margin=0.25,
        )
        loss_curve.extend(lmft_losses)
        v_lmft = evaluation.evaluate_verification_trials(
            state, eval_ctx.val_trials_1, cfg, feature_cache=eval_ctx.val_cache
        )
        v_lmft_eer = float(v_lmft["eer_percent"])
        val_history.append(
            {
                "stage": "stage2_lmft_3p6s",
                "step": actual_stage1_steps + lmft_steps,
                "val_1utt_eer_percent": v_lmft_eer,
                "mean100_loss": round(
                    float(np.mean(lmft_losses[-min(100, len(lmft_losses)):])), 4
                ),
            }
        )
        _log(
            f"[{model_id}] Post-LMFT Held-out Val 1-Utt EER={v_lmft_eer:.2f}% "
            f"(pre-LMFT best={best_val_eer:.2f}%)"
        )
        if v_lmft_eer <= best_val_eer:
            best_val_eer = v_lmft_eer
            best_enc_params = {
                k: v for k, v in state.params.items() if not k.startswith("_aux_")
            }
            best_batch_stats = state.batch_stats
            best_aux_w = state.params["_aux_class_weights"]
        else:
            restored_final = dict(best_enc_params)
            restored_final["_aux_class_weights"] = best_aux_w
            state = state.replace(params=restored_final, batch_stats=best_batch_stats)

    cfg.train.num_steps = len(loss_curve)
    train_sec = round(time.time() - t_train0, 2)
    return state, cfg, loss_curve, val_history, train_sec, step0_overrides


def _train_warm_started_head_flagship(
    model_id: str,
    cfg: configs.ExperimentConfig,
    store_base: shard_cache.ShardedFeatureStore,
    eval_ctx: EvaluationSuiteContext,
    head_warmup_steps: int,
    joint_steps: int,
    joint_lr: float,
    resume: bool = True,
) -> tuple[neural_net.SpeakerTrainState, list[float], list[dict[str, Any]], float]:
    """Two-phase warm-started training (`freeze_trunk` warmup -> joint `trunk_lr_mult=0.05`)."""
    t0 = time.time()
    loss_curve: list[float] = []
    val_history: list[dict[str, Any]] = []
    resume_dir = os.path.join(REPO_ROOT, "pretrained_models", model_id, "_resume")

    # Phase 1: Frozen trunk head warmup
    cfg.train.freeze_trunk = True
    cfg.train.trunk_lr_mult = 0.0
    cfg.train.learning_rate = 2.0e-3
    cfg.train.min_learning_rate = 2.0e-4
    cfg.train.num_steps = int(head_warmup_steps)
    cfg.train.warmup_steps = max(50, head_warmup_steps // 10)
    _, state = neural_net.get_speaker_encoder(cfg, num_classes=0)
    start_step = 0
    if resume:
        start_step, state, loss_curve, val_history = _try_load_step_resume_ckpt(
            resume_dir, state
        )

    if start_step < head_warmup_steps:
        step_fn_1 = neural_net.make_modern_train_step(cfg)
        rng = np.random.default_rng(cfg.train.seed + start_step)
        for step in range(start_step, head_warmup_steps):
            b_in, b_lbl, b_m = store_base.sample_nm_batch(
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
            state, l_val = step_fn_1(state, b_in, b_lbl, b_m)
            loss_curve.append(float(l_val.item()))
            if (step + 1) % 250 == 0 or step == 0:
                _log(
                    f"[{model_id}] Head-warmup step {step + 1}/{head_warmup_steps} "
                    f"loss={loss_curve[-1]:.4f}"
                )
        _save_step_resume_ckpt(
            resume_dir, head_warmup_steps, state, loss_curve, val_history
        )

    # Evaluate post-head-warmup checkpoint before unfreezing trunk
    t_v0 = time.time()
    shared_v0: dict[str, np.ndarray] = {}
    v1_0 = evaluation.evaluate_verification_trials(
        state,
        eval_ctx.val_trials_1,
        cfg,
        feature_cache=eval_ctx.val_cache,
        precomputed_embeddings=shared_v0,
    )
    v3_0 = evaluation.evaluate_multi_enroll_trials(
        state,
        eval_ctx.val_trials_3,
        cfg,
        feature_cache=eval_ctx.val_cache,
        precomputed_embeddings=shared_v0,
    )
    best_comp_eer = 0.5 * float(v1_0["eer_percent"]) + 0.5 * float(v3_0["eer_percent"])
    best_params = state.params
    best_batch_stats = state.batch_stats
    val_history.append(
        {
            "stage": "head_warmup_done",
            "step": head_warmup_steps,
            "val_1utt_eer_percent": float(v1_0["eer_percent"]),
            "val_3utt_eer_percent": float(v3_0["eer_percent"]),
        }
    )
    t0 += time.time() - t_v0
    _log(
        f"[{model_id} @ head-warmup done] Val 1-utt={v1_0['eer_percent']:.2f}%, "
        f"3-utt={v3_0['eer_percent']:.2f}% (comp={best_comp_eer:.2f}%)"
    )

    # Phase 2: Joint trunk + head fine-tuning (`trunk_lr_mult=0.05`)
    cfg.train.freeze_trunk = False
    cfg.train.trunk_lr_mult = 0.05
    cfg.train.learning_rate = float(joint_lr)
    cfg.train.min_learning_rate = float(joint_lr) * 0.02
    cfg.train.num_steps = int(joint_steps)
    cfg.train.warmup_steps = max(100, joint_steps // 12)
    tx_2 = neural_net.build_optimizer(cfg)
    state = neural_net.SpeakerTrainState.create(
        apply_fn=state.apply_fn,
        params=state.params,
        tx=tx_2,
        batch_stats=state.batch_stats,
    )
    step_fn_2 = neural_net.make_modern_train_step(cfg)
    joint_start = max(0, start_step - head_warmup_steps)
    rng_joint = np.random.default_rng(cfg.train.seed + head_warmup_steps + joint_start)

    for step in range(joint_start, joint_steps):
        b_in, b_lbl, b_m = store_base.sample_nm_batch(
            num_speakers=cfg.train.loss.num_speakers_per_batch,
            num_utts_per_speaker=cfg.train.loss.num_utts_per_speaker,
            max_seq_len=cfg.model.seq_len,
            min_seq_len=cfg.model.min_seq_len,
            variable_length=True,
            specaug_config=cfg.train.specaug,
            rng=rng_joint,
            within_domain_prob=cfg.data.within_domain_prob,
            domain_temp_alpha=cfg.data.domain_temp_alpha,
        )
        state, l_val = step_fn_2(state, b_in, b_lbl, b_m)
        loss_curve.append(float(l_val.item()))
        step_num = step + 1
        if step_num % 200 == 0 or step_num == joint_steps:
            mean100 = float(np.mean(loss_curve[-min(100, len(loss_curve)):]))
            _log(
                f"[{model_id}] Joint step {step_num}/{joint_steps} "
                f"loss={loss_curve[-1]:.4f} (mean100={mean100:.4f})"
            )
        if step_num % 450 == 0 or step_num == joint_steps:
            t_v0 = time.time()
            shared_v: dict[str, np.ndarray] = {}
            v1 = evaluation.evaluate_verification_trials(
                state,
                eval_ctx.val_trials_1,
                cfg,
                feature_cache=eval_ctx.val_cache,
                precomputed_embeddings=shared_v,
            )
            v3 = evaluation.evaluate_multi_enroll_trials(
                state,
                eval_ctx.val_trials_3,
                cfg,
                feature_cache=eval_ctx.val_cache,
                precomputed_embeddings=shared_v,
            )
            comp_eer = 0.5 * float(v1["eer_percent"]) + 0.5 * float(v3["eer_percent"])
            val_history.append(
                {
                    "stage": "joint_finetune",
                    "step": head_warmup_steps + step_num,
                    "val_1utt_eer_percent": float(v1["eer_percent"]),
                    "val_3utt_eer_percent": float(v3["eer_percent"]),
                }
            )
            if comp_eer <= best_comp_eer:
                best_comp_eer = comp_eer
                best_params = state.params
                best_batch_stats = state.batch_stats
            _save_step_resume_ckpt(
                resume_dir, head_warmup_steps + step_num, state, loss_curve, val_history
            )
            t0 += time.time() - t_v0
            _log(
                f"[{model_id} @ joint step {step_num}] Val 1-utt={v1['eer_percent']:.2f}%, "
                f"3-utt={v3['eer_percent']:.2f}% (best comp={best_comp_eer:.2f}%)"
            )

    state = state.replace(params=best_params, batch_stats=best_batch_stats)
    cfg.train.num_steps = int(head_warmup_steps + joint_steps)
    return state, loss_curve, val_history, round(time.time() - t0, 2)


def _export_and_eval_tflite_flagship(
    model_id: str,
    hf_dir: str,
    state: neural_net.SpeakerTrainState,
    exp_cfg: configs.ExperimentConfig,
    eval_ctx: EvaluationSuiteContext,
    jax_full_seq_eer: float,
) -> Optional[dict[str, Any]]:
    """Exports FP32 & INT8 `.tflite` models and measures fixed-window `VoxCeleb1-O` EER."""
    fp32_path = os.path.join(hf_dir, "model_fp32.tflite")
    int8_path = os.path.join(hf_dir, "model_int8.tflite")
    try:
        os.makedirs(hf_dir, exist_ok=True)
        fp32_info = export.export_to_tflite(
            state, exp_cfg, fp32_path, quantize_int8=False, seq_len=180
        )
        int8_info = export.export_to_tflite(
            state, exp_cfg, int8_path, quantize_int8=True, seq_len=180
        )
        r_fp32 = export.TFLiteSpeakerRunner(fp32_path)
        r_int8 = export.TFLiteSpeakerRunner(int8_path)
        vox_utts = sorted(
            {u for _, u1, u2 in eval_ctx.vox1_o_trials for u in (u1, u2)}
        )
        emb_fp32: dict[str, np.ndarray] = {}
        emb_int8: dict[str, np.ndarray] = {}
        cos_list: list[float] = []
        for u in vox_utts:
            feat = eval_ctx.vox1_o_cache.get(u)
            e32 = r_fp32(feat)
            e8 = r_int8(feat)
            e32_n = e32 / (np.linalg.norm(e32) + 1e-6)
            e8_n = e8 / (np.linalg.norm(e8) + 1e-6)
            emb_fp32[u] = e32_n
            emb_int8[u] = e8_n
            cos_list.append(float(np.dot(e32_n, e8_n)))
        labels = [int(lbl) for lbl, _, _ in eval_ctx.vox1_o_trials]
        s_32 = np.array(
            [float(np.dot(emb_fp32[u1], emb_fp32[u2])) for _, u1, u2 in eval_ctx.vox1_o_trials],
            dtype=np.float64,
        )
        s_8 = np.array(
            [float(np.dot(emb_int8[u1], emb_int8[u2])) for _, u1, u2 in eval_ctx.vox1_o_trials],
            dtype=np.float64,
        )
        eer_32, _ = evaluation.compute_eer_exact(labels, s_32)
        eer_8, _ = evaluation.compute_eer_exact(labels, s_8)
        tfl_rec = {
            "protocol": "VoxCeleb1-O official (37,611 trials, 180-frame center crop)",
            "eer_percent": round(eer_32 * 100.0, 4),
            "eer_percent_int8": round(eer_8 * 100.0, 4),
            "jax_full_seq_eer_percent": round(float(jax_full_seq_eer), 4),
            "cos_fp32_int8_mean": round(float(np.mean(cos_list)), 6),
            "fp32_size_kb": fp32_info["size_kb"],
            "int8_size_kb": int8_info["size_kb"],
        }
        _log(
            f"[{model_id}] Exported TFLite (FP32={fp32_info['size_kb']} KB, "
            f"INT8={int8_info['size_kb']} KB): Vox1-O EER FP32={tfl_rec['eer_percent']:.2f}%, "
            f"INT8={tfl_rec['eer_percent_int8']:.2f}%, cos={tfl_rec['cos_fp32_int8_mean']:.6f}"
        )
        eval_ctx.vox1_o_cache._cache.clear()
        gc.collect()
        return tfl_rec
    except Exception as exc:
        eval_ctx.vox1_o_cache._cache.clear()
        gc.collect()
        _log(f"[{model_id}] TFLite export skipped ({type(exc).__name__}: {exc})")
        return None


def _save_flagship_artifacts(
    model_id: str,
    state: neural_net.SpeakerTrainState,
    exp_cfg: configs.ExperimentConfig,
    shard_meta: dict[str, Any],
    loss_curve: list[float],
    val_history: list[dict[str, Any]],
    train_time_sec: float,
    eval_bundle: dict[str, Any],
    extra_fields: Optional[dict[str, Any]] = None,
    eval_ctx: Optional[EvaluationSuiteContext] = None,
    export_tflite: bool = False,
) -> dict[str, Any]:
    """Writes `configs/<id>.yml` and `pretrained_models/<id>/{config,safetensors,msgpack,eval_results,README}`."""
    enc_only_params = {
        k: v for k, v in state.params.items() if k != "_aux_class_weights"
    }
    num_params = neural_net.count_parameters(enc_only_params)
    md_eval = eval_bundle["multi_domain_eval"]
    in_dom = md_eval["domains_1utt"]["universal_pooled"]
    multi_3 = md_eval["domains_3utt"]["universal_pooled"]

    actual_steps = len(loss_curve) if loss_curve else int(exp_cfg.train.num_steps)
    loss_curve_100 = [
        {"step": idx + 1, "step_loss": round(float(val), 4)}
        for idx, val in enumerate(loss_curve)
        if idx == 0 or (idx + 1) % 100 == 0 or idx == len(loss_curve) - 1
    ]
    base_spks = int(shard_meta.get("num_base_speakers", 12256))
    clean_utts = int(shard_meta.get("clean_utterances", 2150000))
    total_epochs = round(
        (actual_steps * exp_cfg.train.batch_size) / max(1, clean_utts), 3
    )

    hf_dir = os.path.join(REPO_ROOT, "pretrained_models", model_id)
    tflite_rec: Optional[dict[str, Any]] = None
    if export_tflite and eval_ctx is not None:
        jax_vox_eer = float(
            eval_bundle["official_benchmarks"]["voxceleb1_o"]["eer_percent"]
        )
        tflite_rec = _export_and_eval_tflite_flagship(
            model_id, hf_dir, state, exp_cfg, eval_ctx, jax_vox_eer
        )

    exp_record: dict[str, Any] = {
        "experiment_name": model_id,
        "flaxspeaker_version": "0.5.0",
        "tier": "tier2_full_data_flagship",
        "train_dataset": "universal_full",
        "dataset_stats": {
            "num_speakers": base_spks,
            "num_speed_perturbed_classes": int(shard_meta["num_speakers"]),
            "total_utterances": clean_utts,
            "total_augmented_shard_items": int(shard_meta["num_items"]),
        },
        "backbone": str(exp_cfg.model.backbone),
        "size_variant": str(exp_cfg.model.size_variant),
        "pooling_type": str(exp_cfg.model.pooling.pooling_type),
        "loss_type": str(exp_cfg.train.loss.loss_type),
        "scoring_type": str(exp_cfg.model.scoring_type),
        "embedding_dim": int(exp_cfg.model.output_embedding_dim),
        "num_parameters": int(num_params),
        "num_steps": int(actual_steps),
        "batch_size": int(exp_cfg.train.batch_size),
        "warmup_steps": int(exp_cfg.train.warmup_steps),
        "margin_warmup_steps": int(exp_cfg.train.loss.margin_warmup_steps),
        "train_epochs": total_epochs,
        "train_time_sec": train_time_sec,
        "initial_loss": round(float(loss_curve[0]), 4) if loss_curve else 0.0,
        "final_loss": round(
            float(np.mean(loss_curve[-min(100, len(loss_curve)):])), 4
        )
        if loss_curve
        else 0.0,
        "loss_curve_100": loss_curve_100,
        "validation_trajectory": val_history,
        "official_benchmarks": eval_bundle["official_benchmarks"],
        "in_domain_eval": in_dom,
        "multi_enroll_3utt_eval": multi_3,
        "multi_domain_eval": md_eval,
        "cross_dataset_eval": {
            "vox1_o_10spk_subset": eval_bundle["vox1_o_10spk_subset"]
        },
        "config_yaml": f"configs/{model_id}.yml",
        "hf_pretrained_dir": f"pretrained_models/{model_id}",
        "hf_repo_id": f"{hf_compat.HF_DEFAULT_NAMESPACE}/{model_id}",
        "hf_url": f"https://huggingface.co/{hf_compat.HF_DEFAULT_NAMESPACE}/{model_id}",
        "hf_collection_url": hf_compat.HF_COLLECTION_URL,
    }
    if tflite_rec is not None:
        exp_record["tflite_eval"] = tflite_rec
    if extra_fields:
        exp_record.update(extra_fields)

    yml_path = os.path.join(REPO_ROOT, "configs", f"{model_id}.yml")
    exp_cfg.to_yaml(yml_path)

    hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(exp_cfg)
    hf_model = hf_compat.FlaxSpeakerModel(
        config=hf_cfg,
        params=enc_only_params,
        batch_stats=getattr(state, "batch_stats", None),
    )
    hf_model.save_pretrained(hf_dir, metrics=exp_record)

    resume_dir = os.path.join(hf_dir, "_resume")
    if os.path.isdir(resume_dir):
        import shutil

        shutil.rmtree(resume_dir, ignore_errors=True)

    # Also merge into benchmark_results.json, vox1_o_10spk_subset_results.json, and tflite_eval_results.json
    bench_json = os.path.join(REPO_ROOT, "pretrained_models", "benchmark_results.json")
    if os.path.isfile(bench_json):
        with open(bench_json, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        b_data.setdefault("experiments", {})[model_id] = exp_record
        if tflite_rec is not None:
            b_data.setdefault("tflite_exports", {})[model_id] = tflite_rec
        b_data["num_experiments"] = len(b_data["experiments"])
        with open(bench_json, "w", encoding="utf-8") as f:
            json.dump(b_data, f, indent=2)

    vox_sub_json = os.path.join(
        REPO_ROOT, "pretrained_models", "vox1_o_10spk_subset_results.json"
    )
    if os.path.isfile(vox_sub_json):
        with open(vox_sub_json, "r", encoding="utf-8") as f:
            v_data = json.load(f)
        v_data.setdefault("models", {})[model_id] = eval_bundle["vox1_o_10spk_subset"]
        with open(vox_sub_json, "w", encoding="utf-8") as f:
            json.dump(v_data, f, indent=2)

    if tflite_rec is not None:
        tfl_json = os.path.join(
            REPO_ROOT, "pretrained_models", "tflite_eval_results.json"
        )
        if os.path.isfile(tfl_json):
            with open(tfl_json, "r", encoding="utf-8") as f:
                t_data = json.load(f)
            t_data.setdefault("models", {})[model_id] = tflite_rec
            with open(tfl_json, "w", encoding="utf-8") as f:
                json.dump(t_data, f, indent=2)

    _log(f"Saved complete Hugging Face release directory -> {hf_dir}")
    return exp_record


def _compute_heterogeneous_fusion(
    score_banks: dict[str, dict[str, np.ndarray]],
    eval_ctx: EvaluationSuiteContext,
) -> dict[str, Any]:
    """Selects optimal convex fusion weight on `universal_full_val` and evaluates zero-shot."""
    if (
        "univ_full_conformer_large_arcface" not in score_banks
        or "univ_full_ecapa_c1024_arcface" not in score_banks
    ):
        return {}

    sb_conf = score_banks["univ_full_conformer_large_arcface"]
    sb_ecapa = score_banks["univ_full_ecapa_c1024_arcface"]
    val_labels = [int(lbl) for lbl, _, _ in eval_ctx.val_trials_1]

    best_w = 0.5
    best_val_eer = 1.0
    for w_cand in np.linspace(0.1, 0.9, 17):
        fused_val = scoring.fuse_scores(
            [sb_conf["val_1utt_qmf"], sb_ecapa["val_1utt_qmf"]],
            weights=[float(w_cand), float(1.0 - w_cand)],
            z_normalize=True,
        )
        eer_v, _ = evaluation.compute_eer_exact(val_labels, list(fused_val))
        if eer_v < best_val_eer:
            best_val_eer = eer_v
            best_w = round(float(w_cand), 3)

    weights_2way = [best_w, round(1.0 - best_w, 3)]
    out_fusion: dict[str, Any] = {
        "models": [
            "univ_full_conformer_large_arcface",
            "univ_full_ecapa_c1024_arcface",
        ],
        "weights_selected_on_val": weights_2way,
        "val_1utt_eer_percent": round(best_val_eer * 100.0, 4),
        "official_benchmarks": {},
    }

    for b_key, trials, with_ci in (
        ("voxceleb1_o", eval_ctx.vox1_o_trials, True),
        ("cnceleb_e", eval_ctx.cn_e_single_trials, False),
        ("librispeech_test_clean", eval_ctx.ls_off_trials_1, True),
    ):
        labels = [int(lbl) for lbl, _, _ in trials]
        fused_asn = scoring.fuse_scores(
            [sb_conf[b_key], sb_ecapa[b_key]],
            weights=weights_2way,
            z_normalize=True,
        )
        fused_qmf = scoring.fuse_scores(
            [sb_conf[f"{b_key}_qmf"], sb_ecapa[f"{b_key}_qmf"]],
            weights=weights_2way,
            z_normalize=True,
        )
        m_asn = evaluation._compute_metrics_dict(labels, fused_asn, compute_ci=with_ci)
        m_qmf = evaluation._compute_metrics_dict(labels, fused_qmf, compute_ci=with_ci)
        m_asn["num_trials"] = len(trials)
        m_asn["qmf"] = m_qmf
        out_fusion["official_benchmarks"][b_key] = m_asn

    _log(
        f"[Fusion Conformer-Large + ECAPA-c1024 (w={weights_2way})]: "
        f"Vox1-O AS-Norm={out_fusion['official_benchmarks']['voxceleb1_o']['eer_percent']:.2f}% "
        f"(QMF={out_fusion['official_benchmarks']['voxceleb1_o']['qmf']['eer_percent']:.2f}%), "
        f"CN-Celeb-E={out_fusion['official_benchmarks']['cnceleb_e']['eer_percent']:.2f}%, "
        f"LS-test={out_fusion['official_benchmarks']['librispeech_test_clean']['eer_percent']:.2f}%"
    )

    if "univ_full_conformer_large_pfas" in score_banks:
        sb_pfas = score_banks["univ_full_conformer_large_pfas"]
        val_3_labels = [int(lbl) for lbl, _, _ in eval_ctx.val_trials_3]
        best_3w = [0.4, 0.35, 0.25]
        best_3_eer = 1.0
        for w1 in (0.30, 0.40, 0.50):
            for w2 in (0.20, 0.30, 0.40):
                w3 = round(1.0 - w1 - w2, 3)
                if w3 <= 0.05:
                    continue
                f_v3 = scoring.fuse_scores(
                    [sb_conf["val_3utt"], sb_ecapa["val_3utt"], sb_pfas["val_3utt"]],
                    weights=[w1, w2, w3],
                    z_normalize=True,
                )
                eer_3, _ = evaluation.compute_eer_exact(val_3_labels, f_v3)
                if eer_3 < best_3_eer:
                    best_3_eer = eer_3
                    best_3w = [round(w1, 3), round(w2, 3), round(w3, 3)]

        cn_m_labels = [int(lbl) for lbl, _, _ in eval_ctx.cn_e_multi_trials]
        f_cn_m = scoring.fuse_scores(
            [
                sb_conf["cnceleb_e_multi"],
                sb_ecapa["cnceleb_e_multi"],
                sb_pfas["cnceleb_e_multi"],
            ],
            weights=best_3w,
            z_normalize=True,
        )
        ls_3_labels = [int(lbl) for lbl, _, _ in eval_ctx.ls_off_trials_3]
        f_ls_3 = scoring.fuse_scores(
            [
                sb_conf["librispeech_3utt"],
                sb_ecapa["librispeech_3utt"],
                sb_pfas["librispeech_3utt"],
            ],
            weights=best_3w,
            z_normalize=True,
        )
        m_cn_3w = evaluation._compute_metrics_dict(
            cn_m_labels, f_cn_m, compute_ci=False
        )
        m_cn_3w["num_trials"] = len(cn_m_labels)
        m_ls_3w = evaluation._compute_metrics_dict(
            ls_3_labels, f_ls_3, compute_ci=True
        )
        m_ls_3w["num_trials"] = len(ls_3_labels)
        out_fusion["multi_enroll_3way_fusion"] = {
            "models": [
                "univ_full_conformer_large_arcface",
                "univ_full_ecapa_c1024_arcface",
                "univ_full_conformer_large_pfas",
            ],
            "weights_selected_on_val_3utt": best_3w,
            "val_3utt_eer_percent": round(best_3_eer * 100.0, 4),
            "cnceleb_e_multi_enroll_map": m_cn_3w,
            "librispeech_test_clean_3utt": m_ls_3w,
        }
        _log(
            f"[3-Way Multi-Enroll Fusion + PFAS (w={best_3w})]: "
            f"CN-Celeb-E MultiEnroll={m_cn_3w['eer_percent']:.2f}%, "
            f"LS-test 3-Utt={m_ls_3w['eer_percent']:.2f}%"
        )
    return out_fusion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train & evaluate the 4 Full-Data Universal Flagships (`FlaxSpeaker v0.5.0`)."
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
    parser.add_argument("--conformer_steps", type=int, default=600)
    parser.add_argument("--ecapa_steps", type=int, default=600)
    parser.add_argument("--stage1b_steps", type=int, default=3600)
    parser.add_argument("--lmft_steps", type=int, default=300)
    parser.add_argument("--pfas_warmup_steps", type=int, default=600)
    parser.add_argument("--pfas_joint_steps", type=int, default=1800)
    parser.add_argument("--dr_warmup_steps", type=int, default=600)
    parser.add_argument("--dr_joint_steps", type=int, default=1800)
    parser.add_argument("--max_threads", type=int, default=28)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force_retrain", action="store_true")
    args = parser.parse_args()

    resources.configure_polite_resources(max_threads=args.max_threads, nice_level=10)
    with open(os.path.join(args.shard_dir, "metadata.json"), "r", encoding="utf-8") as f:
        shard_meta = json.load(f)

    store_base = shard_cache.ShardedFeatureStore(args.shard_dir, base_speakers_only=False)
    eval_ctx = EvaluationSuiteContext(args.data_root)

    target_ids = (
        set(args.only) if args.only else set(hf_compat.FULL_DATA_FLAGSHIP_IDS)
    )
    summary_path = os.path.join(
        REPO_ROOT, "pretrained_models", "full_data_flagship_results.json"
    )
    summary: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "flaxspeaker_version": "0.5.0",
        "shard_metadata": {
            k: shard_meta[k]
            for k in (
                "num_base_speakers",
                "num_speakers",
                "clean_utterances",
                "num_items",
                "num_shards",
                "zero_eval_speaker_overlap_verified",
            )
            if k in shard_meta
        },
        "models": {},
    }
    if os.path.isfile(summary_path) and not args.force_retrain:
        with open(summary_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        summary["models"] = prev.get("models", {})

    score_banks: dict[str, dict[str, np.ndarray]] = {}
    pre_lmft_trunk = os.path.join(
        REPO_ROOT,
        "pretrained_models",
        "univ_full_conformer_large_arcface",
        "trunk_pre_lmft.msgpack",
    )

    # 1. Flagship #1: univ_full_conformer_large_arcface
    m1_id = "univ_full_conformer_large_arcface"
    m1_dir = os.path.join(REPO_ROOT, "pretrained_models", m1_id)
    if m1_id in target_ids:
        if (
            not args.force_retrain
            and os.path.isfile(os.path.join(m1_dir, "model.safetensors"))
            and os.path.isfile(os.path.join(m1_dir, "eval_results.json"))
        ):
            _log(f"[{m1_id}] Loading completed checkpoint for fusion/summary...")
            loaded_m1 = hf_compat.FlaxSpeakerModel.from_pretrained(m1_dir)
            ev1, sb1 = evaluate_flagship_comprehensive(
                m1_id, loaded_m1.state, loaded_m1.exp_config, eval_ctx, store_base, args.shard_dir
            )
            score_banks[m1_id] = sb1
            with open(os.path.join(m1_dir, "eval_results.json"), "r", encoding="utf-8") as f:
                summary["models"][m1_id] = json.load(f)
        else:
            st1, cfg1, lc1, vh1, dt1, s0_ov1 = _train_arcface_plus_lmft_flagship(
                model_id=m1_id,
                backbone="conformer",
                shard_dir=args.shard_dir,
                store_base=store_base,
                eval_ctx=eval_ctx,
                stage1a_steps=args.conformer_steps,
                stage1b_steps=args.stage1b_steps,
                lmft_steps=args.lmft_steps,
                save_pre_lmft_trunk_path=pre_lmft_trunk,
                resume=True,
            )
            ev1, sb1 = evaluate_flagship_comprehensive(
                m1_id, st1, cfg1, eval_ctx, store_base, args.shard_dir
            )
            score_banks[m1_id] = sb1
            summary["models"][m1_id] = _save_flagship_artifacts(
                m1_id,
                st1,
                cfg1,
                shard_meta,
                lc1,
                vh1,
                dt1,
                ev1,
                extra_fields={
                    "stage1a_steps": args.conformer_steps,
                    "stage1b_steps": args.stage1b_steps,
                    "lmft_steps": args.lmft_steps,
                    "lmft_seq_len": 360,
                    "lmft_margin": 0.25,
                    "step0_phase_overrides": s0_ov1,
                },
                eval_ctx=eval_ctx,
                export_tflite=True,
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    trunk_for_warmstart = (
        pre_lmft_trunk
        if os.path.isfile(pre_lmft_trunk)
        else m1_dir
    )
    rel_trunk_ckpt = os.path.relpath(trunk_for_warmstart, REPO_ROOT)

    # 2. Flagship #2: univ_full_ecapa_c1024_arcface
    m2_id = "univ_full_ecapa_c1024_arcface"
    m2_dir = os.path.join(REPO_ROOT, "pretrained_models", m2_id)
    if m2_id in target_ids:
        if (
            not args.force_retrain
            and os.path.isfile(os.path.join(m2_dir, "model.safetensors"))
            and os.path.isfile(os.path.join(m2_dir, "eval_results.json"))
        ):
            _log(f"[{m2_id}] Loading completed checkpoint for fusion/summary...")
            loaded_m2 = hf_compat.FlaxSpeakerModel.from_pretrained(m2_dir)
            ev2, sb2 = evaluate_flagship_comprehensive(
                m2_id, loaded_m2.state, loaded_m2.exp_config, eval_ctx, store_base, args.shard_dir
            )
            score_banks[m2_id] = sb2
            with open(os.path.join(m2_dir, "eval_results.json"), "r", encoding="utf-8") as f:
                summary["models"][m2_id] = json.load(f)
        else:
            st2, cfg2, lc2, vh2, dt2, s0_ov2 = _train_arcface_plus_lmft_flagship(
                model_id=m2_id,
                backbone="ecapa_tdnn",
                shard_dir=args.shard_dir,
                store_base=store_base,
                eval_ctx=eval_ctx,
                stage1a_steps=args.ecapa_steps,
                stage1b_steps=args.stage1b_steps,
                lmft_steps=args.lmft_steps,
                resume=True,
            )
            ev2, sb2 = evaluate_flagship_comprehensive(
                m2_id, st2, cfg2, eval_ctx, store_base, args.shard_dir
            )
            score_banks[m2_id] = sb2
            summary["models"][m2_id] = _save_flagship_artifacts(
                m2_id,
                st2,
                cfg2,
                shard_meta,
                lc2,
                vh2,
                dt2,
                ev2,
                extra_fields={
                    "stage1a_steps": args.ecapa_steps,
                    "stage1b_steps": args.stage1b_steps,
                    "lmft_steps": args.lmft_steps,
                    "lmft_seq_len": 360,
                    "lmft_margin": 0.25,
                    "step0_phase_overrides": s0_ov2,
                },
                eval_ctx=eval_ctx,
                export_tflite=True,
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    # 3. Wide 16-config PFAS Hyperparameter Sweep -> Flagship #3: univ_full_conformer_large_pfas
    m3_id = "univ_full_conformer_large_pfas"
    m3_dir = os.path.join(REPO_ROOT, "pretrained_models", m3_id)
    if m3_id in target_ids:
        sweep_json = os.path.join(
            REPO_ROOT, "pretrained_models", "pfas_hparam_sweep.json"
        )
        sweep_res = run_pfas_hparam_sweep.run_pfas_sweep(
            trunk_ckpt=trunk_for_warmstart,
            data_root=args.data_root,
            shard_dir=args.shard_dir,
            output_json=sweep_json,
            max_threads=args.max_threads,
            overwrite=args.force_retrain,
        )
        summary["pfas_hparam_sweep_winner"] = sweep_res["winner"]
        winner_spec = sweep_res["candidates"][sweep_res["winner"]["name"]]["spec"]

        if (
            not args.force_retrain
            and os.path.isfile(os.path.join(m3_dir, "model.safetensors"))
            and os.path.isfile(os.path.join(m3_dir, "eval_results.json"))
        ):
            _log(f"[{m3_id}] Loading completed checkpoint for fusion/summary...")
            loaded_m3 = hf_compat.FlaxSpeakerModel.from_pretrained(m3_dir)
            ev3, sb3 = evaluate_flagship_comprehensive(
                m3_id, loaded_m3.state, loaded_m3.exp_config, eval_ctx, store_base, args.shard_dir
            )
            score_banks[m3_id] = sb3
            with open(os.path.join(m3_dir, "eval_results.json"), "r", encoding="utf-8") as f:
                summary["models"][m3_id] = json.load(f)
        else:
            cfg3 = run_pfas_hparam_sweep.build_base_pfas_conformer_large_config(
                trunk_ckpt=trunk_for_warmstart,
                shard_dir="data/speaker_datasets/sharded_caches/universal_full_v050",
                spec=winner_spec,
                seq_len=180,
                seed=42,
            )
            cfg3.model.min_seq_len = 90
            cfg3.export.sequence_length = 180
            st3, lc3, vh3, dt3 = _train_warm_started_head_flagship(
                model_id=m3_id,
                cfg=cfg3,
                store_base=store_base,
                eval_ctx=eval_ctx,
                head_warmup_steps=args.pfas_warmup_steps,
                joint_steps=args.pfas_joint_steps,
                joint_lr=4.0e-4,
                resume=True,
            )
            cfg3.train.init_from_checkpoint = ""
            ev3, sb3 = evaluate_flagship_comprehensive(
                m3_id, st3, cfg3, eval_ctx, store_base, args.shard_dir
            )
            score_banks[m3_id] = sb3
            summary["models"][m3_id] = _save_flagship_artifacts(
                m3_id,
                st3,
                cfg3,
                shard_meta,
                lc3,
                vh3,
                dt3,
                ev3,
                extra_fields={
                    "warm_started_from": "univ_full_conformer_large_arcface (pre-LMFT trunk)",
                    "pfas_winning_spec": winner_spec,
                    "step0_phase_overrides": {
                        "init_from_checkpoint": rel_trunk_ckpt,
                        "freeze_trunk": True,
                        "trunk_lr_mult": 0.0,
                        "learning_rate": 2.0e-3,
                        "min_learning_rate": 2.0e-4,
                        "num_steps": int(args.pfas_warmup_steps),
                        "warmup_steps": max(50, args.pfas_warmup_steps // 10),
                    },
                },
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    # 4. Flagship #4: univ_full_conformer_large_dr_vectors
    m4_id = "univ_full_conformer_large_dr_vectors"
    m4_dir = os.path.join(REPO_ROOT, "pretrained_models", m4_id)
    if m4_id in target_ids:
        if (
            not args.force_retrain
            and os.path.isfile(os.path.join(m4_dir, "model.safetensors"))
            and os.path.isfile(os.path.join(m4_dir, "eval_results.json"))
        ):
            _log(f"[{m4_id}] Loading completed checkpoint for fusion/summary...")
            loaded_m4 = hf_compat.FlaxSpeakerModel.from_pretrained(m4_dir)
            ev4, sb4 = evaluate_flagship_comprehensive(
                m4_id, loaded_m4.state, loaded_m4.exp_config, eval_ctx, store_base, args.shard_dir
            )
            score_banks[m4_id] = sb4
            with open(os.path.join(m4_dir, "eval_results.json"), "r", encoding="utf-8") as f:
                summary["models"][m4_id] = json.load(f)
        else:
            cfg4 = configs.ExperimentConfig()
            cfg4.train.seed = 42
            cfg4.data.dataset_name = "universal_full"
            cfg4.data.sharded_cache_dir = (
                "data/speaker_datasets/sharded_caches/universal_full_v050"
            )
            cfg4.data.within_domain_prob = 0.75
            cfg4.data.domain_temp_alpha = 0.35
            cfg4.frontend.feature_type = configs.FeatureType.LOG_MEL.value
            cfg4.frontend.n_mels = 80
            cfg4.frontend.n_mfcc = 80
            cfg4.frontend.apply_cmvn = True
            cfg4.vad.enabled = True
            cfg4.vad.mode = configs.VadMode.ENERGY.value

            cfg4.model.backbone = configs.BackboneType.CONFORMER.value
            cfg4.model.apply_size_variant(configs.ModelSize.LARGE)
            cfg4.model.seq_len = 180
            cfg4.model.min_seq_len = 90
            cfg4.model.max_seq_len = 180
            cfg4.model.max_eval_seq_len = 0
            cfg4.model.variable_length_training = True
            cfg4.model.pooling.pooling_type = configs.PoolingType.ATP.value
            cfg4.model.scoring_type = configs.ScoringType.DR_VECTOR.value
            cfg4.model.dr_vector.hidden_dims = (256, 128)
            cfg4.model.dr_vector.activation = "leaky_relu"
            cfg4.model.dr_vector.leaky_relu_slope = 0.2
            cfg4.model.dr_vector.use_layer_norm = True
            cfg4.model.dr_vector.include_cosine_in_input = True
            cfg4.eval.full_sequence_inference = True
            cfg4.eval.max_eval_frames_cap = 3000

            cfg4.train.init_from_checkpoint = trunk_for_warmstart
            cfg4.train.optimizer = "adamw"
            cfg4.train.weight_decay = 1.0e-4
            cfg4.train.grad_clip_norm = 3.0
            cfg4.train.lr_schedule_cosine = True
            cfg4.train.loss.loss_type = "extended_set_softmax"
            cfg4.train.loss.scoring_type = "dr_vector"
            cfg4.train.loss.num_speakers_per_batch = 16
            cfg4.train.loss.num_utts_per_speaker = 4
            cfg4.train.batch_size = 64
            cfg4.loss = cfg4.train.loss
            cfg4.export.sequence_length = 180

            st4, lc4, vh4, dt4 = _train_warm_started_head_flagship(
                model_id=m4_id,
                cfg=cfg4,
                store_base=store_base,
                eval_ctx=eval_ctx,
                head_warmup_steps=args.dr_warmup_steps,
                joint_steps=args.dr_joint_steps,
                joint_lr=3.5e-4,
                resume=True,
            )
            cfg4.train.init_from_checkpoint = ""
            ev4, sb4 = evaluate_flagship_comprehensive(
                m4_id, st4, cfg4, eval_ctx, store_base, args.shard_dir
            )
            score_banks[m4_id] = sb4
            summary["models"][m4_id] = _save_flagship_artifacts(
                m4_id,
                st4,
                cfg4,
                shard_meta,
                lc4,
                vh4,
                dt4,
                ev4,
                extra_fields={
                    "warm_started_from": "univ_full_conformer_large_arcface (pre-LMFT trunk)",
                    "step0_phase_overrides": {
                        "init_from_checkpoint": rel_trunk_ckpt,
                        "freeze_trunk": True,
                        "trunk_lr_mult": 0.0,
                        "learning_rate": 2.0e-3,
                        "min_learning_rate": 2.0e-4,
                        "num_steps": int(args.dr_warmup_steps),
                        "warmup_steps": max(50, args.dr_warmup_steps // 10),
                    },
                },
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    # 5. Calibrated Heterogeneous Score Fusion (Conformer-Large + ECAPA-c1024)
    fusion_res = _compute_heterogeneous_fusion(score_banks, eval_ctx)
    if fusion_res:
        summary["heterogeneous_fusion"] = fusion_res
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    _log(f"All requested full-data flagships finished! Summary -> {summary_path}")


if __name__ == "__main__":
    main()
