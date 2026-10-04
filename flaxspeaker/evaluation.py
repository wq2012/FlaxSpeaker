"""Speaker verification evaluation metrics and trial scoring pipelines.

Supports:
1. Legacy API (`run_inference`, `compute_single_triplet_scores`, `compute_scores`,
   `compute_eer`, `run_eval`) for 100% backward compatibility.
2. Modernized verification evaluation:
   - Exact Equal Error Rate (`compute_eer_exact`)
   - NIST Minimum Detection Cost Function (`compute_min_dcf` at p_target=0.01, 0.05)
   - Area Under ROC Curve (`compute_roc_auc`)
   - Single-utterance and multi-utterance enrollment trial scoring with Cosine,
     Parameter-Free Attentive Scoring (PFAS), and Decision Residual Networks (Dr-Vectors)
   - Optional Adaptive Symmetric Score Normalization (AS-Norm)
"""

import functools
from multiprocessing.pool import ThreadPool
import sys
import time
from typing import Any, Optional, Sequence

from flax.training import train_state
import jax
import jax.numpy as jnp
import munch
import numpy as np
from sklearn import metrics as sk_metrics

from flaxspeaker import configs
from flaxspeaker import dataset
from flaxspeaker import feature_extraction
from flaxspeaker import neural_net
from flaxspeaker import scoring


def _get_full_seq_flag(myconfig: Any) -> bool:
    eval_cfg = getattr(myconfig, "eval", None)
    model_cfg = getattr(myconfig, "model", None)
    if eval_cfg is not None and hasattr(eval_cfg, "full_sequence_inference"):
        val = getattr(eval_cfg, "full_sequence_inference")
        if val is not None:
            return bool(val)
    if model_cfg is not None and hasattr(model_cfg, "full_sequence_inference"):
        return bool(getattr(model_cfg, "full_sequence_inference"))
    return True


def run_inference(
    features: jax.Array,
    state: train_state.TrainState,
    myconfig: Any,
) -> Optional[jax.Array]:
    """Get the embedding of an utterance using the encoder."""
    if _get_full_seq_flag(myconfig):
        batch_input = jnp.expand_dims(features, axis=0)
        batch_output = state.apply_fn({"params": state.params}, batch_input)
        return batch_output[0, :]
    else:
        sliding_windows = feature_extraction.extract_sliding_windows(
            features, myconfig
        )
        if not sliding_windows:
            return None
        batch_input = jnp.stack(sliding_windows)
        batch_output = state.apply_fn({"params": state.params}, batch_input)
        aggregated_output = jnp.mean(batch_output, axis=0, keepdims=False)
        return aggregated_output


def compute_single_triplet_scores(
    i: int,
    spk_to_utts: dataset.SpkToUtts,
    state: train_state.TrainState,
    config: Any,
) -> tuple[list[int], list[float]]:
    """Get the labels and scores from a single triplet."""
    anchor, pos, neg = feature_extraction.get_triplet_features(
        spk_to_utts, config.model.n_mfcc
    )
    anchor_embedding = run_inference(anchor, state, config)
    pos_embedding = run_inference(pos, state, config)
    neg_embedding = run_inference(neg, state, config)
    if (
        (anchor_embedding is None)
        or (pos_embedding is None)
        or (neg_embedding is None)
    ):
        return ([], [])
    triplet_labels = [1, 0]
    triplet_scores = [
        float(neural_net.cosine_similarity(anchor_embedding, pos_embedding)),
        float(neural_net.cosine_similarity(anchor_embedding, neg_embedding)),
    ]
    print("triplets evaluated:", i, "/", config.eval.num_triplets)
    return (triplet_labels, triplet_scores)


def compute_scores(
    state: train_state.TrainState,
    spk_to_utts: dataset.SpkToUtts,
    myconfig: Any,
) -> tuple[list[int], list[float]]:
    """Compute cosine similarity scores from testing data."""
    labels: list[int] = []
    scores: list[float] = []
    score_fetcher = functools.partial(
        compute_single_triplet_scores,
        spk_to_utts=spk_to_utts,
        state=state,
        config=myconfig,
    )
    with ThreadPool(myconfig.train.num_processes) as pool:
        while myconfig.eval.num_triplets > len(labels) // 2:
            label_score_pairs = pool.map(
                score_fetcher,
                range(len(labels) // 2, myconfig.eval.num_triplets),
            )
            for triplet_labels, triplet_scores in label_score_pairs:
                labels += triplet_labels
                scores += triplet_scores
    print("Evaluated", len(labels) // 2, "triplets in total")
    return (labels, scores)


def compute_eer(
    labels: Sequence[int],
    scores: Sequence[float],
    eval_threshold_step: float = 0.001,
) -> tuple[Optional[float], Optional[float]]:
    """Compute the Equal Error Rate (EER) via threshold sweep."""
    if len(labels) != len(scores):
        raise ValueError("Length of labels and scored must match")
    eer_threshold = None
    eer = None
    min_delta = 1.0
    threshold = 0.0
    while threshold < 1.0:
        accept = [score >= threshold for score in scores]
        fa = [a and (1 - l) for a, l in zip(accept, labels)]
        fr = [(1 - a) and l for a, l in zip(accept, labels)]
        far = sum(fa) / (len(labels) - sum(labels))
        frr = sum(fr) / sum(labels)
        delta = abs(far - frr)
        if delta < min_delta:
            min_delta = delta
            eer = (far + frr) / 2
            eer_threshold = threshold
        threshold += eval_threshold_step

    return eer, eer_threshold


def compute_eer_exact(
    labels: Sequence[int],
    scores: Sequence[float],
) -> tuple[float, float]:
    """Compute exact Equal Error Rate (EER) and threshold using ROC curve."""
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float(0.5 * (fpr[idx] + fnr[idx]))
    thresh = float(thresholds[idx])
    return eer, thresh


def compute_min_dcf(
    labels: Sequence[int],
    scores: Sequence[float],
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Compute NIST Minimum Detection Cost Function (minDCF).

    `DCF(t) = C_miss * P_miss(t) * P_target + C_fa * P_fa(t) * (1 - P_target)`
    normalized by `min(C_miss * P_target, C_fa * (1 - P_target))`.
    """
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    c_det = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(np.min(c_det) / max(c_def, 1e-12))


def compute_roc_auc(
    labels: Sequence[int],
    scores: Sequence[float],
) -> float:
    """Compute Area Under the ROC Curve (ROC-AUC)."""
    return float(sk_metrics.roc_auc_score(np.asarray(labels), np.asarray(scores)))


def evaluate_verification_trials(
    state: train_state.TrainState,
    trials: Sequence[dataset.TrialPair],
    myconfig: Any,
    feature_cache: Optional[feature_extraction.FeatureCache] = None,
    scoring_override: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate speaker verification on a list of `(label, enroll_utt, test_utt)` trials.

    Pre-computes embeddings for all unique utterances referenced in `trials` once,
    then scores all trial pairs using Cosine, PFAS, or Decision Residual Network.
    """
    t0 = time.time()
    exp_cfg = (
        myconfig
        if isinstance(myconfig, configs.ExperimentConfig)
        else configs.ExperimentConfig.from_munch(myconfig)
    )
    frontend = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)
    seq_len = int(exp_cfg.model.seq_len)

    # Collect unique utterances
    unique_utts = sorted({u for _, u1, u2 in trials for u in (u1, u2)})

    # Extract fixed-length center crops for deterministic batch inference
    utt_to_emb: dict[str, np.ndarray] = {}
    batch_size = 64
    infer_start = time.time()
    for start in range(0, len(unique_utts), batch_size):
        batch_utts = unique_utts[start:start + batch_size]
        feats = []
        for u in batch_utts:
            raw = (
                feature_cache.get(u)
                if feature_cache is not None
                else frontend.extract_from_file(u)
            )
            # Center crop or pad to seq_len*2 (or seq_len) for consistent evaluation
            eval_len = seq_len
            if raw.shape[0] >= eval_len:
                offset = (raw.shape[0] - eval_len) // 2
                crop = raw[offset:offset + eval_len]
            else:
                reps = (eval_len // max(1, raw.shape[0])) + 1
                crop = np.tile(raw, (reps, 1))[:eval_len]
            feats.append(crop)
        batch_arr = jnp.asarray(np.stack(feats, axis=0), dtype=jnp.float32)
        embs = np.asarray(state.apply_fn({"params": state.params}, batch_arr))
        for u, e in zip(batch_utts, embs):
            utt_to_emb[u] = e
    infer_elapsed = time.time() - infer_start
    latency_ms = (infer_elapsed * 1000.0) / max(1, len(unique_utts))

    scoring_type = scoring_override or str(
        getattr(exp_cfg.train.loss.scoring_type, "value", exp_cfg.train.loss.scoring_type)
    )

    labels = [int(lbl) for lbl, _, _ in trials]
    enroll_arr = jnp.asarray(np.stack([utt_to_emb[u1] for _, u1, _ in trials], axis=0))
    test_arr = jnp.asarray(np.stack([utt_to_emb[u2] for _, _, u2 in trials], axis=0))

    if scoring_type == "pfas":
        pfas_cfg = exp_cfg.model.pfas
        t_keys, t_vals = scoring.unpack_pfas_representation(
            test_arr, pfas_cfg.num_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
        )
        e_keys, e_vals = scoring.unpack_pfas_representation(
            enroll_arr, pfas_cfg.num_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
        )
        pair_fn = jax.jit(
            jax.vmap(
                functools.partial(
                    scoring.pfas_score_pair,
                    scale_factor=pfas_cfg.scale_factor,
                    per_trial_softmax=pfas_cfg.per_trial_softmax,
                )
            )
        )
        scores_arr = np.asarray(pair_fn(t_keys, t_vals, e_keys, e_vals))
    elif scoring_type == "dr_vector" and "_aux_dr_net" in state.params:
        dr_cfg = exp_cfg.model.dr_vector
        dr_net = scoring.DecisionResidualNetwork(
            hidden_dims=tuple(dr_cfg.hidden_dims),
            activation=dr_cfg.activation,
            use_layer_norm=dr_cfg.use_layer_norm,
            include_cosine_in_input=dr_cfg.include_cosine_in_input,
            residual_weight=dr_cfg.residual_weight,
        )
        scores_arr = np.asarray(
            dr_net.apply(
                {"params": state.params["_aux_dr_net"]},
                test_arr,
                enroll_arr,
                method=dr_net.score_pairs,
            )
        )
    else:
        # Vectorized cosine similarity
        e_norm = enroll_arr / (jnp.linalg.norm(enroll_arr, axis=-1, keepdims=True) + 1e-6)
        t_norm = test_arr / (jnp.linalg.norm(test_arr, axis=-1, keepdims=True) + 1e-6)
        scores_arr = np.asarray(jnp.sum(e_norm * t_norm, axis=-1))

    if exp_cfg.eval.use_as_norm and len(utt_to_emb) >= 10:
        cohort = np.stack(list(utt_to_emb.values()), axis=0)
        scores_arr = scoring.apply_as_norm(
            scores_arr,
            np.asarray(enroll_arr),
            np.asarray(test_arr),
            cohort,
            top_k=exp_cfg.eval.as_norm_top_k,
        )

    scores_list = [float(s) for s in scores_arr]
    eer, eer_thresh = compute_eer_exact(labels, scores_list)
    min_dcf_001 = compute_min_dcf(labels, scores_list, p_target=0.01)
    min_dcf_005 = compute_min_dcf(labels, scores_list, p_target=0.05)
    roc_auc = compute_roc_auc(labels, scores_list)
    total_time = time.time() - t0

    return {
        "eer": eer,
        "eer_percent": round(eer * 100.0, 4),
        "eer_threshold": eer_thresh,
        "min_dcf_001": round(min_dcf_001, 4),
        "min_dcf_005": round(min_dcf_005, 4),
        "roc_auc": round(roc_auc, 4),
        "num_trials": len(trials),
        "num_unique_utterances": len(unique_utts),
        "latency_ms_per_utterance": round(latency_ms, 3),
        "eval_time_sec": round(total_time, 3),
        "scoring_type": scoring_type,
    }


def evaluate_multi_enroll_trials(
    state: train_state.TrainState,
    trials: Sequence[dataset.MultiEnrollTrial],
    myconfig: Any,
    feature_cache: Optional[feature_extraction.FeatureCache] = None,
) -> dict[str, Any]:
    """Evaluate multi-utterance enrollment trials `(label, [enroll_utts], test_utt)`."""
    exp_cfg = (
        myconfig
        if isinstance(myconfig, configs.ExperimentConfig)
        else configs.ExperimentConfig.from_munch(myconfig)
    )
    frontend = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)
    seq_len = int(exp_cfg.model.seq_len)

    unique_utts = sorted(
        {u for _, e_list, t_u in trials for u in list(e_list) + [t_u]}
    )
    utt_to_emb: dict[str, np.ndarray] = {}
    for start in range(0, len(unique_utts), 64):
        batch_utts = unique_utts[start:start + 64]
        feats = []
        for u in batch_utts:
            raw = (
                feature_cache.get(u)
                if feature_cache is not None
                else frontend.extract_from_file(u)
            )
            if raw.shape[0] >= seq_len:
                offset = (raw.shape[0] - seq_len) // 2
                crop = raw[offset:offset + seq_len]
            else:
                reps = (seq_len // max(1, raw.shape[0])) + 1
                crop = np.tile(raw, (reps, 1))[:seq_len]
            feats.append(crop)
        batch_arr = jnp.asarray(np.stack(feats, axis=0), dtype=jnp.float32)
        embs = np.asarray(state.apply_fn({"params": state.params}, batch_arr))
        for u, e in zip(batch_utts, embs):
            utt_to_emb[u] = e

    scoring_type = str(
        getattr(exp_cfg.train.loss.scoring_type, "value", exp_cfg.train.loss.scoring_type)
    )
    labels = [int(lbl) for lbl, _, _ in trials]
    scores_list = []

    if scoring_type == "pfas":
        pfas_cfg = exp_cfg.model.pfas
        for _, e_utts, t_utt in trials:
            stacked_e = scoring.stack_multi_enroll_pfas(
                [utt_to_emb[u] for u in e_utts],
                pfas_cfg.num_keys,
                pfas_cfg.key_dim,
                pfas_cfg.value_dim,
            )
            num_e_keys = len(e_utts) * pfas_cfg.num_keys
            e_keys, e_vals = scoring.unpack_pfas_representation(
                stacked_e, num_e_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
            )
            t_keys, t_vals = scoring.unpack_pfas_representation(
                jnp.asarray(utt_to_emb[t_utt]),
                pfas_cfg.num_keys,
                pfas_cfg.key_dim,
                pfas_cfg.value_dim,
            )
            s = scoring.pfas_score_pair(
                t_keys,
                t_vals,
                e_keys,
                e_vals,
                scale_factor=pfas_cfg.scale_factor,
                per_trial_softmax=pfas_cfg.per_trial_softmax,
            )
            scores_list.append(float(s))
    else:
        for _, e_utts, t_utt in trials:
            centroid = np.mean([utt_to_emb[u] for u in e_utts], axis=0)
            t_emb = utt_to_emb[t_utt]
            sim = float(
                np.dot(centroid, t_emb)
                / (np.linalg.norm(centroid) * np.linalg.norm(t_emb) + 1e-6)
            )
            scores_list.append(sim)

    eer, eer_thresh = compute_eer_exact(labels, scores_list)
    min_dcf_001 = compute_min_dcf(labels, scores_list, p_target=0.01)
    roc_auc = compute_roc_auc(labels, scores_list)
    return {
        "eer": eer,
        "eer_percent": round(eer * 100.0, 4),
        "eer_threshold": eer_thresh,
        "min_dcf_001": round(min_dcf_001, 4),
        "roc_auc": round(roc_auc, 4),
        "num_trials": len(trials),
    }


def run_eval(myconfig: Any) -> dict[str, Any]:
    """Run evaluation of the saved model on test data."""
    start_time = time.time()
    spk_to_utts = dataset.load_dataset_from_config(myconfig, split="eval")
    _, state = neural_net.get_speaker_encoder(
        myconfig, myconfig.model.saved_model_path
    )
    if not neural_net._is_modern_config(myconfig):
        labels, scores = compute_scores(state, spk_to_utts, myconfig)
        eer, eer_threshold = compute_eer(
            labels, scores, myconfig.eval.threshold_step
        )
        eval_time = time.time() - start_time
        print("Finished evaluation in", eval_time, "seconds")
        print("eer_threshold =", eer_threshold, "eer =", eer)
        return {"eer": eer, "eer_threshold": eer_threshold}

    trials_file = getattr(myconfig.data, "trials_file", None)
    if trials_file:
        trials = dataset.load_trials_file(trials_file)
    else:
        trials = dataset.generate_verification_trials(
            spk_to_utts, num_trials=myconfig.eval.num_triplets * 2
        )
    metrics = evaluate_verification_trials(state, trials, myconfig)
    print("Evaluation metrics:", metrics)
    return metrics


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        config_file = "myconfig.yml"
    elif len(args) == 1:
        config_file = args[0]
    else:
        raise ValueError("Expecting a single argument: config file path")
    with open(config_file, "r", encoding="utf-8") as f:
        myconfig = munch.Munch.fromYAML(f.read())
    run_eval(myconfig)
