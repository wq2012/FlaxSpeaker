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


@functools.partial(jax.jit, static_argnums=(0,))
def _jit_apply_unmasked(
    apply_fn: Any, params: Any, batch_input: jax.Array
) -> jax.Array:
    return apply_fn({"params": params}, batch_input)


def run_inference(
    features: jax.Array,
    state: train_state.TrainState,
    myconfig: Any,
) -> Optional[jax.Array]:
    """Get the embedding of an utterance using the encoder."""
    is_modern = neural_net._is_modern_config(myconfig)
    seq_len = int(getattr(getattr(myconfig, "model", None), "seq_len", 160))
    if is_modern and features.shape[0] < min(32, seq_len):
        min_len = min(32, seq_len)
        reps = (min_len // max(1, int(features.shape[0]))) + 1
        features = jnp.tile(features, (reps, 1))[:min_len]
    if _get_full_seq_flag(myconfig):
        if is_modern:
            max_eval_len = int(
                getattr(
                    getattr(myconfig, "model", None),
                    "max_eval_seq_len",
                    max(400, seq_len * 2),
                )
            )
            if features.shape[0] > max_eval_len:
                offset = (int(features.shape[0]) - max_eval_len) // 2
                features = features[offset:offset + max_eval_len]
        batch_input = jnp.expand_dims(features, axis=0)
        batch_output = _jit_apply_unmasked(
            state.apply_fn, state.params, batch_input
        )
        return batch_output[0, :]
    else:
        sliding_windows = feature_extraction.extract_sliding_windows(
            features, myconfig
        )
        if not sliding_windows:
            if is_modern and features.shape[0] > 0:
                reps = (seq_len // max(1, int(features.shape[0]))) + 1
                padded = jnp.tile(features, (reps, 1))[:seq_len]
                batch_output = _jit_apply_unmasked(
                    state.apply_fn,
                    state.params,
                    jnp.expand_dims(padded, axis=0),
                )
                return batch_output[0, :]
            return None
        batch_input = jnp.stack(sliding_windows)
        batch_output = _jit_apply_unmasked(
            state.apply_fn, state.params, batch_input
        )
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


def compute_bootstrap_eer_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    num_resamples: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval for EER."""
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    n = len(y_true)
    if n < 10 or np.all(y_true == y_true[0]):
        eer, _ = compute_eer_exact(labels, scores)
        return eer, eer
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    pos_scores = y_score[pos_idx]
    neg_scores = y_score[neg_idx]
    boot_eers = np.empty(num_resamples, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, 201)
    for b in range(num_resamples):
        b_pos = pos_scores[rng.integers(0, n_pos, size=n_pos)]
        b_neg = neg_scores[rng.integers(0, n_neg, size=n_neg)]
        thresh = np.quantile(np.concatenate([b_pos, b_neg]), quantiles)
        fnr = np.mean(b_pos[:, None] < thresh[None, :], axis=0)
        fpr = np.mean(b_neg[:, None] >= thresh[None, :], axis=0)
        idx = int(np.argmin(np.abs(fnr - fpr)))
        boot_eers[b] = 0.5 * (fnr[idx] + fpr[idx])
    alpha = 0.5 * (1.0 - confidence)
    low = float(np.quantile(boot_eers, alpha))
    high = float(np.quantile(boot_eers, 1.0 - alpha))
    return low, high


@functools.partial(jax.jit, static_argnums=(0,))
def _jit_encode_eval_batch(
    apply_fn: Any, params: Any, x_batch: jax.Array, m_batch: jax.Array
) -> jax.Array:
    return apply_fn({"params": params}, x_batch, mask=m_batch)


def _extract_eval_embeddings(
    unique_utts: Sequence[str],
    state: train_state.TrainState,
    exp_cfg: configs.ExperimentConfig,
    frontend: feature_extraction.AudioFrontend,
    feature_cache: Optional[feature_extraction.FeatureCache] = None,
    batch_size: int = 64,
    precomputed_embeddings: Optional[dict[str, np.ndarray]] = None,
) -> tuple[dict[str, np.ndarray], float]:
    """Extracts utterance embeddings with full/variable-length mask support."""
    utt_to_emb: dict[str, np.ndarray] = (
        precomputed_embeddings if precomputed_embeddings is not None else {}
    )
    missing_utts = [u for u in unique_utts if u not in utt_to_emb]
    if not missing_utts:
        return utt_to_emb, 0.0

    seq_len = int(exp_cfg.model.seq_len)
    max_eval_len = int(
        getattr(exp_cfg.model, "max_eval_seq_len", max(400, seq_len * 2))
    )
    full_seq = bool(getattr(exp_cfg.eval, "full_sequence_inference", True))
    target_pad_len = max_eval_len if full_seq else seq_len
    min_valid_len = min(32, target_pad_len)

    infer_start = time.time()
    for start in range(0, len(missing_utts), batch_size):
        batch_utts = missing_utts[start:start + batch_size]
        feats = []
        masks = []
        for u in batch_utts:
            raw = (
                feature_cache.get(u)
                if feature_cache is not None
                else frontend.extract_from_file(u)
            )
            if raw.shape[0] < min_valid_len:
                reps = (min_valid_len // max(1, raw.shape[0])) + 1
                raw = np.tile(raw, (reps, 1))
            if full_seq:
                if raw.shape[0] >= target_pad_len:
                    offset = (raw.shape[0] - target_pad_len) // 2
                    seg = raw[offset:offset + target_pad_len]
                    valid_len = target_pad_len
                else:
                    seg = raw
                    valid_len = raw.shape[0]
                padded = np.zeros(
                    (target_pad_len, raw.shape[1]), dtype=np.float32
                )
                padded[:valid_len, :] = seg
                mask = np.zeros((target_pad_len,), dtype=np.float32)
                mask[:valid_len] = 1.0
            else:
                if raw.shape[0] >= seq_len:
                    offset = (raw.shape[0] - seq_len) // 2
                    padded = raw[offset:offset + seq_len].astype(np.float32)
                else:
                    reps = (seq_len // max(1, raw.shape[0])) + 1
                    padded = np.tile(raw, (reps, 1))[:seq_len].astype(
                        np.float32
                    )
                mask = np.ones((seq_len,), dtype=np.float32)
            feats.append(padded)
            masks.append(mask)

        batch_arr = jnp.asarray(np.stack(feats, axis=0), dtype=jnp.float32)
        mask_arr = jnp.asarray(np.stack(masks, axis=0), dtype=jnp.float32)
        embs = np.asarray(
            _jit_encode_eval_batch(
                state.apply_fn, state.params, batch_arr, mask_arr
            )
        )
        for u, e in zip(batch_utts, embs):
            utt_to_emb[u] = e

    infer_elapsed = time.time() - infer_start
    latency_ms = (infer_elapsed * 1000.0) / max(1, len(missing_utts))
    return utt_to_emb, latency_ms


def evaluate_verification_trials(
    state: train_state.TrainState,
    trials: Sequence[dataset.TrialPair],
    myconfig: Any,
    feature_cache: Optional[feature_extraction.FeatureCache] = None,
    scoring_override: Optional[str] = None,
    precomputed_embeddings: Optional[dict[str, np.ndarray]] = None,
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

    # Collect unique utterances
    unique_utts = sorted({u for _, u1, u2 in trials for u in (u1, u2)})
    utt_to_emb, latency_ms = _extract_eval_embeddings(
        unique_utts,
        state,
        exp_cfg,
        frontend,
        feature_cache,
        precomputed_embeddings=precomputed_embeddings,
    )

    scoring_type = scoring_override or str(
        getattr(
            exp_cfg.train.loss.scoring_type,
            "value",
            exp_cfg.train.loss.scoring_type,
        )
    )

    labels = [int(lbl) for lbl, _, _ in trials]
    enroll_arr = jnp.asarray(
        np.stack([utt_to_emb[u1] for _, u1, _ in trials], axis=0)
    )
    test_arr = jnp.asarray(
        np.stack([utt_to_emb[u2] for _, _, u2 in trials], axis=0)
    )

    if scoring_type == "pfas":
        pfas_cfg = exp_cfg.model.pfas
        pfas_scale = (
            float(np.exp(float(state.params["_aux_pfas_log_scale"])))
            if "_aux_pfas_log_scale" in state.params
            else float(pfas_cfg.scale_factor)
        )
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
        scores_arr = np.asarray(pair_fn(t_keys, t_vals, e_keys, e_vals))
    elif scoring_type == "dr_vector" and "_aux_dr_net" in state.params:
        dr_cfg = exp_cfg.model.dr_vector
        dr_net = scoring.DecisionResidualNetwork(
            hidden_dims=tuple(dr_cfg.hidden_dims),
            activation=dr_cfg.activation,
            use_layer_norm=dr_cfg.use_layer_norm,
            include_cosine_in_input=dr_cfg.include_cosine_in_input,
            num_cosine_terms=getattr(dr_cfg, "num_cosine_terms", 1),
            append_cosine_as_feature_to_network_input=getattr(
                dr_cfg, "append_cosine_as_feature_to_network_input", True
            ),
            append_cosine_to_final_hidden=bool(
                getattr(dr_cfg, "append_cosine_to_final_hidden", False)
            ),
            add_cosine_to_network_output=getattr(
                dr_cfg, "add_cosine_to_network_output", True
            ),
            symmetric_input=getattr(dr_cfg, "symmetric_input", False),
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
        e_norm = enroll_arr / (
            jnp.linalg.norm(enroll_arr, axis=-1, keepdims=True) + 1e-6
        )
        t_norm = test_arr / (
            jnp.linalg.norm(test_arr, axis=-1, keepdims=True) + 1e-6
        )
        scores_arr = np.asarray(jnp.sum(e_norm * t_norm, axis=-1))

    use_as_norm = bool(
        getattr(exp_cfg.eval, "use_as_norm", False)
        or getattr(exp_cfg.eval, "apply_as_norm", False)
    )
    if use_as_norm and scoring_type == "cosine" and len(utt_to_emb) >= 10:
        cohort = np.stack(list(utt_to_emb.values()), axis=0)
        top_k = int(
            getattr(
                exp_cfg.eval,
                "as_norm_top_k",
                getattr(exp_cfg.eval, "as_norm_cohort_size", 100),
            )
        )
        scores_arr = scoring.apply_as_norm(
            scores_arr,
            np.asarray(enroll_arr),
            np.asarray(test_arr),
            cohort,
            top_k=top_k,
        )

    c_miss = float(getattr(exp_cfg.eval, "dcf_c_miss", 1.0))
    c_fa = float(getattr(exp_cfg.eval, "dcf_c_fa", 1.0))
    scores_list = [float(s) for s in scores_arr]
    eer, eer_thresh = compute_eer_exact(labels, scores_list)
    ci_low, ci_high = compute_bootstrap_eer_ci(labels, scores_list)
    min_dcf_001 = compute_min_dcf(
        labels, scores_list, p_target=0.01, c_miss=c_miss, c_fa=c_fa
    )
    min_dcf_005 = compute_min_dcf(
        labels, scores_list, p_target=0.05, c_miss=c_miss, c_fa=c_fa
    )
    min_dcf_01 = compute_min_dcf(
        labels, scores_list, p_target=0.10, c_miss=c_miss, c_fa=c_fa
    )
    roc_auc = compute_roc_auc(labels, scores_list)
    total_time = time.time() - t0

    return {
        "eer": eer,
        "eer_percent": round(eer * 100.0, 4),
        "eer_ci95_low_percent": round(ci_low * 100.0, 4),
        "eer_ci95_high_percent": round(ci_high * 100.0, 4),
        "eer_threshold": eer_thresh,
        "min_dcf_001": round(min_dcf_001, 4),
        "min_dcf_005": round(min_dcf_005, 4),
        "min_dcf_01": round(min_dcf_01, 4),
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
    precomputed_embeddings: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, Any]:
    """Evaluate multi-utterance enrollment trials `(label, [enroll_utts], test_utt)`."""
    exp_cfg = (
        myconfig
        if isinstance(myconfig, configs.ExperimentConfig)
        else configs.ExperimentConfig.from_munch(myconfig)
    )
    frontend = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)

    unique_utts = sorted(
        {u for _, e_list, t_u in trials for u in list(e_list) + [t_u]}
    )
    utt_to_emb, _ = _extract_eval_embeddings(
        unique_utts,
        state,
        exp_cfg,
        frontend,
        feature_cache,
        precomputed_embeddings=precomputed_embeddings,
    )

    scoring_type = str(
        getattr(
            exp_cfg.train.loss.scoring_type,
            "value",
            exp_cfg.train.loss.scoring_type,
        )
    )
    labels = [int(lbl) for lbl, _, _ in trials]
    scores_list = []

    if scoring_type == "pfas":
        pfas_cfg = exp_cfg.model.pfas
        pfas_scale = (
            float(np.exp(float(state.params["_aux_pfas_log_scale"])))
            if "_aux_pfas_log_scale" in state.params
            else float(pfas_cfg.scale_factor)
        )
        stacked_e_list = [
            scoring.stack_multi_enroll_pfas(
                [utt_to_emb[u] for u in e_utts],
                pfas_cfg.num_keys,
                pfas_cfg.key_dim,
                pfas_cfg.value_dim,
            )
            for _, e_utts, _ in trials
        ]
        num_e_keys = len(trials[0][1]) * pfas_cfg.num_keys
        e_arr = jnp.asarray(np.stack(stacked_e_list, axis=0))
        t_arr = jnp.asarray(
            np.stack([utt_to_emb[t_utt] for _, _, t_utt in trials], axis=0)
        )
        e_keys, e_vals = scoring.unpack_pfas_representation(
            e_arr, num_e_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
        )
        t_keys, t_vals = scoring.unpack_pfas_representation(
            t_arr, pfas_cfg.num_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
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
        scores_arr = np.asarray(pair_fn(t_keys, t_vals, e_keys, e_vals))
        scores_list = [float(s) for s in scores_arr]
    elif scoring_type == "dr_vector" and "_aux_dr_net" in state.params:
        dr_cfg = exp_cfg.model.dr_vector
        dr_net = scoring.DecisionResidualNetwork(
            hidden_dims=tuple(dr_cfg.hidden_dims),
            activation=dr_cfg.activation,
            use_layer_norm=dr_cfg.use_layer_norm,
            include_cosine_in_input=dr_cfg.include_cosine_in_input,
            num_cosine_terms=getattr(dr_cfg, "num_cosine_terms", 1),
            append_cosine_as_feature_to_network_input=getattr(
                dr_cfg, "append_cosine_as_feature_to_network_input", True
            ),
            append_cosine_to_final_hidden=bool(
                getattr(dr_cfg, "append_cosine_to_final_hidden", False)
            ),
            add_cosine_to_network_output=getattr(
                dr_cfg, "add_cosine_to_network_output", True
            ),
            symmetric_input=getattr(dr_cfg, "symmetric_input", False),
            residual_weight=dr_cfg.residual_weight,
        )
        centroids = []
        tests = []
        for _, e_utts, t_utt in trials:
            c = np.mean([utt_to_emb[u] for u in e_utts], axis=0)
            c = c / (np.linalg.norm(c) + 1e-6)
            centroids.append(c)
            tests.append(utt_to_emb[t_utt])
        e_arr = jnp.asarray(np.stack(centroids, axis=0))
        t_arr = jnp.asarray(np.stack(tests, axis=0))
        scores_arr = np.asarray(
            dr_net.apply(
                {"params": state.params["_aux_dr_net"]},
                t_arr,
                e_arr,
                method=dr_net.score_pairs,
            )
        )
        scores_list = [float(s) for s in scores_arr]
    else:
        centroids = []
        tests = []
        for _, e_utts, t_utt in trials:
            centroid = np.mean([utt_to_emb[u] for u in e_utts], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-6)
            t_emb = utt_to_emb[t_utt]
            sim = float(
                np.dot(centroid, t_emb)
                / (np.linalg.norm(centroid) * np.linalg.norm(t_emb) + 1e-6)
            )
            scores_list.append(sim)
            centroids.append(centroid)
            tests.append(t_emb)
        e_arr = jnp.asarray(np.stack(centroids, axis=0))
        t_arr = jnp.asarray(np.stack(tests, axis=0))

    use_as_norm = bool(
        getattr(exp_cfg.eval, "use_as_norm", False)
        or getattr(exp_cfg.eval, "apply_as_norm", False)
    )
    if (
        use_as_norm
        and scoring_type == "cosine"
        and len(utt_to_emb) >= 10
    ):
        cohort = np.stack(list(utt_to_emb.values()), axis=0)
        top_k = int(
            getattr(
                exp_cfg.eval,
                "as_norm_top_k",
                getattr(exp_cfg.eval, "as_norm_cohort_size", 100),
            )
        )
        norm_scores = scoring.apply_as_norm(
            np.asarray(scores_list, dtype=np.float64),
            np.asarray(e_arr),
            np.asarray(t_arr),
            cohort,
            top_k=top_k,
        )
        scores_list = [float(s) for s in norm_scores]

    c_miss = float(getattr(exp_cfg.eval, "dcf_c_miss", 1.0))
    c_fa = float(getattr(exp_cfg.eval, "dcf_c_fa", 1.0))
    eer, eer_thresh = compute_eer_exact(labels, scores_list)
    ci_low, ci_high = compute_bootstrap_eer_ci(labels, scores_list)
    min_dcf_001 = compute_min_dcf(
        labels, scores_list, p_target=0.01, c_miss=c_miss, c_fa=c_fa
    )
    min_dcf_005 = compute_min_dcf(
        labels, scores_list, p_target=0.05, c_miss=c_miss, c_fa=c_fa
    )
    min_dcf_01 = compute_min_dcf(
        labels, scores_list, p_target=0.10, c_miss=c_miss, c_fa=c_fa
    )
    roc_auc = compute_roc_auc(labels, scores_list)
    return {
        "eer": eer,
        "eer_percent": round(eer * 100.0, 4),
        "eer_ci95_low_percent": round(ci_low * 100.0, 4),
        "eer_ci95_high_percent": round(ci_high * 100.0, 4),
        "eer_threshold": eer_thresh,
        "min_dcf_001": round(min_dcf_001, 4),
        "min_dcf_005": round(min_dcf_005, 4),
        "min_dcf_01": round(min_dcf_01, 4),
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
