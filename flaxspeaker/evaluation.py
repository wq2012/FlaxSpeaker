"""Speaker verification evaluation metrics and trial scoring pipelines for FlaxSpeaker v0.5.0.

Provides:
- Exact Equal Error Rate (`compute_eer_exact`) and threshold sweep (`compute_eer`)
- NIST Minimum Detection Cost Function (`compute_min_dcf` at p_target=0.01, 0.05, 0.10)
- Area Under ROC Curve (`compute_roc_auc`) and 95% bootstrap CIs (`compute_bootstrap_eer_ci`)
- Full-utterance bucketed masked JAX inference (up to `eval.max_eval_frames_cap = 3000` frames = 30s)
- Single-utterance and multi-utterance enrollment trial scoring with Cosine,
  Parameter-Free Attentive Scoring (PFAS), and Decision Residual Networks (Dr-Vectors)
- Vectorized `O(U * C)` Top-300 Adaptive Symmetric Score Normalization (AS-Norm)
  and optional Quality Measure Function (QMF) calibration
"""

import functools
import sys
import time
from typing import Any, Optional, Sequence

from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import metrics as sk_metrics

from flaxspeaker import configs
from flaxspeaker import dataset
from flaxspeaker import feature_extraction
from flaxspeaker import neural_net
from flaxspeaker import scoring

_EVAL_LENGTH_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048)


def _get_full_seq_flag(exp_cfg: configs.ExperimentConfig) -> bool:
    return bool(
        exp_cfg.eval.full_sequence_inference
        and exp_cfg.model.full_sequence_inference
    )


def _resolve_max_eval_len(exp_cfg: configs.ExperimentConfig) -> int:
    raw_max = int(getattr(exp_cfg.model, "max_eval_seq_len", 0))
    if raw_max > 0:
        return raw_max
    return max(64, int(getattr(exp_cfg.eval, "max_eval_frames_cap", 3000)))


def _resolve_bucket_pad_len(
    raw_frames: int, exp_cfg: configs.ExperimentConfig
) -> int:
    raw_max = int(getattr(exp_cfg.model, "max_eval_seq_len", 0))
    if raw_max > 0:
        return raw_max
    cap = _resolve_max_eval_len(exp_cfg)
    effective = min(max(32, int(raw_frames)), cap)
    for b in _EVAL_LENGTH_BUCKETS:
        if effective <= b and b <= cap:
            return b
    return cap


@functools.partial(jax.jit, static_argnums=(0,))
def _jit_apply_unmasked(
    apply_fn: Any, variables: dict[str, Any], batch_input: jax.Array
) -> jax.Array:
    return apply_fn(variables, batch_input)


@functools.partial(jax.jit, static_argnums=(0,))
def _jit_encode_eval_batch(
    apply_fn: Any,
    variables: dict[str, Any],
    x_batch: jax.Array,
    m_batch: jax.Array,
) -> jax.Array:
    return apply_fn(variables, x_batch, mask=m_batch)


def _state_variables(state: train_state.TrainState) -> dict[str, Any]:
    bs = getattr(state, "batch_stats", None) or {}
    if bs:
        return {"params": state.params, "batch_stats": bs}
    return {"params": state.params}


def run_inference(
    features: jax.Array,
    state: train_state.TrainState,
    myconfig: Any,
) -> Optional[jax.Array]:
    """Get the embedding of an utterance using the encoder."""
    exp_cfg = configs.ExperimentConfig.from_any(myconfig)
    seq_len = int(exp_cfg.model.seq_len)
    min_len = min(32, seq_len)
    if features.shape[0] < min_len:
        reps = (min_len // max(1, int(features.shape[0]))) + 1
        features = jnp.tile(features, (reps, 1))[:min_len]

    variables = _state_variables(state)
    if _get_full_seq_flag(exp_cfg):
        max_eval_len = _resolve_max_eval_len(exp_cfg)
        if features.shape[0] > max_eval_len:
            offset = (int(features.shape[0]) - max_eval_len) // 2
            features = features[offset:offset + max_eval_len]
        batch_input = jnp.expand_dims(features, axis=0)
        batch_output = _jit_apply_unmasked(
            state.apply_fn, variables, batch_input
        )
        return batch_output[0, :]

    sliding_windows = feature_extraction.extract_sliding_windows(
        np.asarray(features), exp_cfg
    )
    if not sliding_windows:
        if features.shape[0] > 0:
            reps = (seq_len // max(1, int(features.shape[0]))) + 1
            padded = jnp.tile(features, (reps, 1))[:seq_len]
            batch_output = _jit_apply_unmasked(
                state.apply_fn,
                variables,
                jnp.expand_dims(padded, axis=0),
            )
            return batch_output[0, :]
        return None
    batch_input = jnp.stack([jnp.asarray(w) for w in sliding_windows])
    batch_output = _jit_apply_unmasked(state.apply_fn, variables, batch_input)
    return jnp.mean(batch_output, axis=0, keepdims=False)


def _validate_binary_labels(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Validates that `labels` and `scores` have equal length and contain both classes."""
    if len(labels) != len(scores):
        raise ValueError("Length of labels and scores must match")
    if len(labels) == 0:
        raise ValueError("Trial labels and scores must be non-empty")
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "Trial labels must contain both positive (1) and negative (0) examples."
        )
    return y_true, y_score


def compute_eer(
    labels: Sequence[int],
    scores: Sequence[float],
    eval_threshold_step: float = 0.001,
) -> tuple[Optional[float], Optional[float]]:
    """Compute the Equal Error Rate (EER) via threshold sweep."""
    _validate_binary_labels(labels, scores)
    eer_threshold = None
    eer = None
    min_delta = 1.0
    threshold = 0.0
    while threshold < 1.0:
        accept = [score >= threshold for score in scores]
        fa = [a and (1 - lbl) for a, lbl in zip(accept, labels)]
        fr = [(1 - a) and lbl for a, lbl in zip(accept, labels)]
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
    y_true, y_score = _validate_binary_labels(labels, scores)
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
    y_true, y_score = _validate_binary_labels(labels, scores)
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
    y_true, y_score = _validate_binary_labels(labels, scores)
    return float(sk_metrics.roc_auc_score(y_true, y_score))


def compute_bootstrap_eer_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    num_resamples: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval for EER."""
    y_true, y_score = _validate_binary_labels(labels, scores)
    n = len(y_true)
    if n < 10:
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


def prepare_eval_input(
    raw: np.ndarray,
    exp_cfg: configs.ExperimentConfig,
    target_pad_override: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pads / centre-crops one feature matrix to the evaluation window.

    When `model.max_eval_seq_len > 0`, pads/centre-crops to that exact frame count.
    When `model.max_eval_seq_len <= 0` (`v0.5.0` full-utterance default), evaluates up
    to `eval.max_eval_frames_cap` (`3000` frames = `30 s`) and pads to a power-of-two
    length bucket (`256, 512, 1024, 2048, 3000`) so XLA compiles at most 5 shapes while
    preserving exact padding-invariant full-utterance embeddings.
    """
    seq_len = int(exp_cfg.model.seq_len)
    full_seq = _get_full_seq_flag(exp_cfg)
    raw = np.asarray(raw, dtype=np.float32)

    if full_seq:
        max_eval_len = _resolve_max_eval_len(exp_cfg)
        target_pad_len = (
            int(target_pad_override)
            if target_pad_override is not None
            else _resolve_bucket_pad_len(raw.shape[0], exp_cfg)
        )
        min_valid_len = min(32, seq_len, target_pad_len)
        if raw.shape[0] < min_valid_len:
            reps = (min_valid_len // max(1, raw.shape[0])) + 1
            raw = np.tile(raw, (reps, 1))[:min_valid_len, :]
        crop_cap = min(max_eval_len, target_pad_len)
        if raw.shape[0] > crop_cap:
            offset = (raw.shape[0] - crop_cap) // 2
            seg = raw[offset:offset + crop_cap]
            valid_len = crop_cap
        else:
            seg = raw
            valid_len = raw.shape[0]
        padded = np.zeros((target_pad_len, raw.shape[1]), dtype=np.float32)
        padded[:valid_len, :] = seg
        mask = np.zeros((target_pad_len,), dtype=np.float32)
        mask[:valid_len] = 1.0
        return padded, mask

    min_valid_len = min(32, seq_len)
    if raw.shape[0] < min_valid_len:
        reps = (min_valid_len // max(1, raw.shape[0])) + 1
        raw = np.tile(raw, (reps, 1))[:min_valid_len, :]
    if raw.shape[0] >= seq_len:
        offset = (raw.shape[0] - seq_len) // 2
        padded = raw[offset:offset + seq_len].astype(np.float32)
    else:
        reps = (seq_len // max(1, raw.shape[0])) + 1
        padded = np.tile(raw, (reps, 1))[:seq_len].astype(np.float32)
    mask = np.ones((seq_len,), dtype=np.float32)
    return padded, mask


def encode_features(
    features: np.ndarray,
    state: train_state.TrainState,
    exp_cfg: configs.ExperimentConfig,
) -> np.ndarray:
    """Embeds one utterance through the masked bucketed evaluation path."""
    padded, mask = prepare_eval_input(np.asarray(features), exp_cfg)
    variables = _state_variables(state)
    emb = _jit_encode_eval_batch(
        state.apply_fn,
        variables,
        jnp.asarray(padded[None, :, :], dtype=jnp.float32),
        jnp.asarray(mask[None, :], dtype=jnp.float32),
    )
    return np.asarray(emb[0], dtype=np.float32)


def _extract_eval_embeddings(
    unique_utts: Sequence[str],
    state: train_state.TrainState,
    exp_cfg: configs.ExperimentConfig,
    frontend: feature_extraction.AudioFrontend,
    feature_cache: Optional[feature_extraction.FeatureCache] = None,
    batch_size: int = 64,
    precomputed_embeddings: Optional[dict[str, np.ndarray]] = None,
) -> tuple[dict[str, np.ndarray], float]:
    """Extracts utterance embeddings grouped by length bucket for fast masked JAX inference."""
    utt_to_emb: dict[str, np.ndarray] = (
        precomputed_embeddings if precomputed_embeddings is not None else {}
    )
    missing_utts = [u for u in unique_utts if u not in utt_to_emb]
    if not missing_utts:
        return utt_to_emb, 0.0

    variables = _state_variables(state)
    full_seq = _get_full_seq_flag(exp_cfg)
    infer_start = time.time()

    # Load raw features and group by bucket length so every batch shares a single pad shape
    buckets: dict[int, list[tuple[str, np.ndarray]]] = {}
    for u in missing_utts:
        raw = (
            feature_cache.get(u)
            if feature_cache is not None
            else frontend.extract_from_file(u)
        )
        b_len = (
            _resolve_bucket_pad_len(raw.shape[0], exp_cfg)
            if full_seq
            else int(exp_cfg.model.seq_len)
        )
        buckets.setdefault(b_len, []).append((u, raw))

    for b_len in sorted(buckets.keys()):
        items = buckets[b_len]
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            feats = []
            masks = []
            for _, raw in chunk:
                padded, mask = prepare_eval_input(
                    raw, exp_cfg, target_pad_override=b_len if full_seq else None
                )
                feats.append(padded)
                masks.append(mask)
            batch_arr = jnp.asarray(np.stack(feats, axis=0), dtype=jnp.float32)
            mask_arr = jnp.asarray(np.stack(masks, axis=0), dtype=jnp.float32)
            embs = np.asarray(
                _jit_encode_eval_batch(
                    state.apply_fn, variables, batch_arr, mask_arr
                )
            )
            for (u, _), e in zip(chunk, embs):
                utt_to_emb[u] = e

    infer_elapsed = time.time() - infer_start
    latency_ms = (infer_elapsed * 1000.0) / max(1, len(missing_utts))
    return utt_to_emb, latency_ms


def compute_speaker_centroid_cohort(
    state: train_state.TrainState,
    spk_to_utts: dataset.SpkToUtts,
    myconfig: Any,
    feature_cache: Optional[feature_extraction.FeatureCache] = None,
    max_utts_per_speaker: int = 5,
    max_speakers: Optional[int] = None,
    seed: int = 42,
    precomputed_embeddings: Optional[dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Computes L2-normalized per-speaker centroid cohort `(N_spk, D)` from disjoint speakers."""
    exp_cfg = configs.ExperimentConfig.from_any(myconfig)
    frontend = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)
    spks = sorted(s for s, u in spk_to_utts.items() if u)
    if max_speakers is not None and len(spks) > max_speakers:
        rng = np.random.default_rng(seed)
        chosen = sorted(
            rng.choice(len(spks), size=max_speakers, replace=False).tolist()
        )
        spks = [spks[i] for i in chosen]
    selected_map: dict[str, list[str]] = {
        s: sorted(spk_to_utts[s])[: max(1, int(max_utts_per_speaker))]
        for s in spks
    }
    unique_utts = sorted({u for u_list in selected_map.values() for u in u_list})
    utt_to_emb, _ = _extract_eval_embeddings(
        unique_utts,
        state,
        exp_cfg,
        frontend,
        feature_cache=feature_cache,
        precomputed_embeddings=precomputed_embeddings,
    )
    scoring_t = str(
        getattr(exp_cfg.model.scoring_type, "value", exp_cfg.model.scoring_type)
    )
    centroids: list[np.ndarray] = []
    for s in spks:
        if scoring_t == "pfas":
            u_embs = [
                utt_to_emb[u] for u in selected_map[s] if u in utt_to_emb
            ]
            if not u_embs:
                continue
            centroids.append(np.mean(u_embs, axis=0).astype(np.float32))
        else:
            u_embs = [
                utt_to_emb[u] / (np.linalg.norm(utt_to_emb[u]) + 1e-6)
                for u in selected_map[s]
                if u in utt_to_emb
            ]
            if not u_embs:
                continue
            c = np.mean(u_embs, axis=0)
            c = (c / (np.linalg.norm(c) + 1e-6)).astype(np.float32)
            centroids.append(c)
    if not centroids:
        raise ValueError("No valid cohort speaker centroids could be computed.")
    return np.stack(centroids, axis=0)


def require_disjoint_cohort(
    cohort_embeddings: Optional[np.ndarray],
    utt_to_emb: dict[str, np.ndarray],
    min_cohort_size: int = 10,
    max_abs_cosine: float = 1.0 - 1e-5,
) -> np.ndarray:
    """Validates an AS-Norm impostor cohort for strict disjointness from trial embeddings."""
    if cohort_embeddings is None:
        raise ValueError(
            "AS-Norm is enabled but no `cohort_embeddings` were provided. Pass a "
            "(N_cohort, D) array of impostor embeddings disjoint from the trial "
            "utterances (e.g. embeddings of training speakers)."
        )
    cohort = np.asarray(cohort_embeddings, dtype=np.float32)
    if cohort.ndim != 2 or cohort.shape[0] < min_cohort_size:
        raise ValueError(
            f"AS-Norm cohort must be a 2-D array with >= {min_cohort_size} rows; "
            f"got shape {cohort.shape}."
        )
    if utt_to_emb:
        trial_embs = np.stack(list(utt_to_emb.values()), axis=0).astype(
            np.float32
        )
        c_norm = cohort / (
            np.linalg.norm(cohort, axis=-1, keepdims=True) + 1e-6
        )
        t_norm = trial_embs / (
            np.linalg.norm(trial_embs, axis=-1, keepdims=True) + 1e-6
        )
        max_sim = 0.0
        for s in range(0, t_norm.shape[0], 4096):
            blk = np.matmul(t_norm[s:s + 4096], c_norm.T)
            max_sim = max(max_sim, float(np.max(blk)))
        if max_sim > max_abs_cosine:
            raise ValueError(
                "AS-Norm cohort overlaps the evaluation trials (a cohort embedding is "
                f"identical to a trial embedding, cosine={max_sim:.6f}). The cohort must "
                "be disjoint from the enrollment/test utterances."
            )
    return cohort


def _compute_metrics_dict(
    labels: Sequence[int],
    scores_arr: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    compute_ci: bool = True,
) -> dict[str, Any]:
    y_true, y_score = _validate_binary_labels(labels, scores_arr)
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float(0.5 * (fpr[idx] + fnr[idx]))
    eer_thresh = float(thresholds[idx])
    if compute_ci:
        ci_low, ci_high = compute_bootstrap_eer_ci(y_true, y_score)
    else:
        ci_low, ci_high = eer, eer

    def _dcf(p_tgt: float) -> float:
        c_det = c_miss * fnr * p_tgt + c_fa * fpr * (1.0 - p_tgt)
        c_def = min(c_miss * p_tgt, c_fa * (1.0 - p_tgt))
        return float(np.min(c_det) / max(c_def, 1e-12))

    min_dcf_001 = _dcf(0.01)
    min_dcf_005 = _dcf(0.05)
    min_dcf_01 = _dcf(0.10)
    roc_auc = float(sk_metrics.roc_auc_score(y_true, y_score))
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
    }


def _apply_unique_utt_as_norm(
    raw_scores: np.ndarray,
    trials: Sequence[dataset.TrialPair],
    utt_to_emb: dict[str, np.ndarray],
    cohort: np.ndarray,
    top_k: int,
    scoring_type: str = "cosine",
    pfas_cfg: Any = None,
    pfas_scale: float = 16.0,
) -> np.ndarray:
    """Computes exact AS-Norm in `O(U * C + N_trials)` via unique-utterance cohort stats."""
    u_keys = list(utt_to_emb.keys())
    u_idx = {u: i for i, u in enumerate(u_keys)}
    u_mat = np.stack([utt_to_emb[u] for u in u_keys], axis=0)
    if scoring_type == "pfas" and pfas_cfg is not None:
        mu_u, std_u = scoring.compute_pfas_cohort_topk_stats(
            u_mat,
            cohort,
            pfas_cfg=pfas_cfg,
            scale_factor=pfas_scale,
            top_k=top_k,
        )
    else:
        mu_u, std_u = scoring.compute_cohort_topk_stats(
            u_mat, cohort, top_k=top_k
        )

    e_indices = np.fromiter(
        (u_idx[u1] for _, u1, _ in trials), dtype=np.int64, count=len(trials)
    )
    t_indices = np.fromiter(
        (u_idx[u2] for _, _, u2 in trials), dtype=np.int64, count=len(trials)
    )
    scores_f64 = np.asarray(raw_scores, dtype=np.float64)
    return 0.5 * (
        (scores_f64 - mu_u[e_indices]) / std_u[e_indices]
        + (scores_f64 - mu_u[t_indices]) / std_u[t_indices]
    )


def evaluate_verification_trials(
    state: train_state.TrainState,
    trials: Sequence[dataset.TrialPair],
    myconfig: Any,
    feature_cache: Optional[feature_extraction.FeatureCache] = None,
    scoring_override: Optional[str] = None,
    precomputed_embeddings: Optional[dict[str, np.ndarray]] = None,
    cohort_embeddings: Optional[np.ndarray] = None,
    include_as_norm_breakdown: bool = False,
) -> dict[str, Any]:
    """Evaluate speaker verification on a list of `(label, enroll_utt, test_utt)` trials."""
    t0 = time.time()
    exp_cfg = configs.ExperimentConfig.from_any(myconfig)
    frontend = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)

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
    if scoring_type == "pfas":
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
        # Score in chunks of 65,536 trials so million-trial benchmarks stay < 200 MB
        n_tr = len(trials)
        scores_arr = np.empty((n_tr,), dtype=np.float64)
        for s_i in range(0, n_tr, 65536):
            e_i = min(n_tr, s_i + 65536)
            sub_tr = trials[s_i:e_i]
            sub_e = jnp.asarray(
                np.stack([utt_to_emb[u1] for _, u1, _ in sub_tr], axis=0)
            )
            sub_t = jnp.asarray(
                np.stack([utt_to_emb[u2] for _, _, u2 in sub_tr], axis=0)
            )
            t_keys, t_vals = scoring.unpack_pfas_representation(
                sub_t, pfas_cfg.num_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
            )
            e_keys, e_vals = scoring.unpack_pfas_representation(
                sub_e, pfas_cfg.num_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
            )
            scores_arr[s_i:e_i] = np.asarray(
                pair_fn(t_keys, t_vals, e_keys, e_vals)
            )
    elif scoring_type == "dr_vector" and "_aux_dr_net" in state.params:
        dr_net = scoring.build_decision_residual_network(exp_cfg.model.dr_vector)
        n_tr = len(trials)
        scores_arr = np.empty((n_tr,), dtype=np.float64)
        for s_i in range(0, n_tr, 65536):
            e_i = min(n_tr, s_i + 65536)
            sub_e = jnp.asarray(
                np.stack([utt_to_emb[u1] for _, u1, _ in trials[s_i:e_i]], axis=0)
            )
            sub_t = jnp.asarray(
                np.stack([utt_to_emb[u2] for _, _, u2 in trials[s_i:e_i]], axis=0)
            )
            scores_arr[s_i:e_i] = np.asarray(
                dr_net.apply(
                    {"params": state.params["_aux_dr_net"]},
                    sub_t,
                    sub_e,
                    method=dr_net.score_pairs,
                )
            )
    else:
        # Memory-efficient unique-utterance normalized dot products
        u_keys = unique_utts
        u_idx = {u: i for i, u in enumerate(u_keys)}
        u_mat = np.stack([utt_to_emb[u] for u in u_keys], axis=0).astype(
            np.float32
        )
        u_norm = u_mat / (np.linalg.norm(u_mat, axis=-1, keepdims=True) + 1e-6)
        n_tr = len(trials)
        scores_arr = np.empty((n_tr,), dtype=np.float64)
        for s_i in range(0, n_tr, 131072):
            e_i = min(n_tr, s_i + 131072)
            sub = trials[s_i:e_i]
            ei = np.fromiter(
                (u_idx[u1] for _, u1, _ in sub), dtype=np.int64, count=len(sub)
            )
            ti = np.fromiter(
                (u_idx[u2] for _, _, u2 in sub), dtype=np.int64, count=len(sub)
            )
            scores_arr[s_i:e_i] = np.sum(
                u_norm[ei] * u_norm[ti], axis=-1, dtype=np.float64
            )

    use_as_norm = bool(
        getattr(exp_cfg.eval, "use_as_norm", False)
        or getattr(exp_cfg.eval, "apply_as_norm", False)
    )
    top_k = int(
        getattr(
            exp_cfg.eval,
            "as_norm_top_k",
            getattr(exp_cfg.eval, "as_norm_cohort_size", 300),
        )
    )
    c_miss = float(getattr(exp_cfg.eval, "dcf_c_miss", 1.0))
    c_fa = float(getattr(exp_cfg.eval, "dcf_c_fa", 1.0))

    as_norm_subdict: Optional[dict[str, Any]] = None
    pfas_cfg_opt = exp_cfg.model.pfas if scoring_type == "pfas" else None
    pfas_scale_opt = (
        float(np.exp(float(state.params["_aux_pfas_log_scale"])))
        if (scoring_type == "pfas" and "_aux_pfas_log_scale" in state.params)
        else float(exp_cfg.model.pfas.scale_factor)
    )
    if use_as_norm and scoring_type in ("cosine", "dr_vector", "pfas"):
        cohort = require_disjoint_cohort(
            cohort_embeddings, {u: utt_to_emb[u] for u in unique_utts}
        )
        scores_arr = _apply_unique_utt_as_norm(
            scores_arr,
            trials,
            {u: utt_to_emb[u] for u in unique_utts},
            cohort,
            top_k=top_k,
            scoring_type=scoring_type,
            pfas_cfg=pfas_cfg_opt,
            pfas_scale=pfas_scale_opt,
        )
    elif (
        include_as_norm_breakdown
        and cohort_embeddings is not None
        and scoring_type in ("cosine", "dr_vector", "pfas")
    ):
        cohort = require_disjoint_cohort(
            cohort_embeddings, {u: utt_to_emb[u] for u in unique_utts}
        )
        asn_scores = _apply_unique_utt_as_norm(
            scores_arr,
            trials,
            {u: utt_to_emb[u] for u in unique_utts},
            cohort,
            top_k=top_k,
            scoring_type=scoring_type,
            pfas_cfg=pfas_cfg_opt,
            pfas_scale=pfas_scale_opt,
        )
        as_norm_subdict = _compute_metrics_dict(
            labels, asn_scores, c_miss=c_miss, c_fa=c_fa, compute_ci=len(trials) <= 50000
        )
        as_norm_subdict["top_k"] = top_k

    out = _compute_metrics_dict(
        labels, scores_arr, c_miss=c_miss, c_fa=c_fa, compute_ci=len(trials) <= 50000
    )
    total_time = time.time() - t0
    out.update(
        {
            "num_trials": len(trials),
            "num_unique_utterances": len(unique_utts),
            "latency_ms_per_utterance": round(latency_ms, 3),
            "eval_time_sec": round(total_time, 3),
            "scoring_type": scoring_type,
        }
    )
    if as_norm_subdict is not None:
        out["as_norm"] = as_norm_subdict
    return out


def evaluate_multi_enroll_trials(
    state: train_state.TrainState,
    trials: Sequence[dataset.MultiEnrollTrial],
    myconfig: Any,
    feature_cache: Optional[feature_extraction.FeatureCache] = None,
    precomputed_embeddings: Optional[dict[str, np.ndarray]] = None,
    cohort_embeddings: Optional[np.ndarray] = None,
    include_as_norm_breakdown: bool = False,
    return_scores: bool = False,
) -> Any:
    """Evaluate multi-utterance enrollment trials `(label, [enroll_utts], test_utt)`."""
    exp_cfg = configs.ExperimentConfig.from_any(myconfig)
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
    e_arr_np: Optional[np.ndarray] = None
    t_arr_np: Optional[np.ndarray] = None

    if scoring_type == "pfas":
        pfas_cfg = exp_cfg.model.pfas
        pfas_scale = (
            float(np.exp(float(state.params["_aux_pfas_log_scale"])))
            if "_aux_pfas_log_scale" in state.params
            else float(pfas_cfg.scale_factor)
        )
        # Memoize stacked multi-enrollment representations per unique enrollment tuple
        enroll_key_to_stacked: dict[tuple[str, ...], np.ndarray] = {}
        by_count: dict[int, list[int]] = {}
        for idx_t, (_, e_utts, _) in enumerate(trials):
            e_key_tup = tuple(e_utts)
            if e_key_tup not in enroll_key_to_stacked:
                enroll_key_to_stacked[e_key_tup] = np.asarray(
                    scoring.stack_multi_enroll_pfas(
                        [utt_to_emb[u] for u in e_key_tup],
                        pfas_cfg.num_keys,
                        pfas_cfg.key_dim,
                        pfas_cfg.value_dim,
                    ),
                    dtype=np.float32,
                )
            by_count.setdefault(len(e_key_tup), []).append(idx_t)

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
        scores_arr = np.empty((len(trials),), dtype=np.float64)
        for k_cnt, idxs in by_count.items():
            num_e_keys = k_cnt * pfas_cfg.num_keys
            stacked_dim = num_e_keys * (pfas_cfg.key_dim + pfas_cfg.value_dim)
            chunk_sz = max(
                256, min(8192, (64 * 1024 * 1024) // max(1, stacked_dim * 4))
            )
            for s_i in range(0, len(idxs), chunk_sz):
                sub_idxs = idxs[s_i:s_i + chunk_sz]
                e_sub = jnp.asarray(
                    np.stack(
                        [enroll_key_to_stacked[tuple(trials[j][1])] for j in sub_idxs],
                        axis=0,
                    )
                )
                t_sub = jnp.asarray(
                    np.stack([utt_to_emb[trials[j][2]] for j in sub_idxs], axis=0)
                )
                e_keys, e_vals = scoring.unpack_pfas_representation(
                    e_sub, num_e_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
                )
                t_keys, t_vals = scoring.unpack_pfas_representation(
                    t_sub, pfas_cfg.num_keys, pfas_cfg.key_dim, pfas_cfg.value_dim
                )
                sub_scores = np.asarray(
                    pair_fn(t_keys, t_vals, e_keys, e_vals), dtype=np.float64
                )
                scores_arr[np.asarray(sub_idxs, dtype=np.int64)] = sub_scores
    else:
        # Deduplicate enrollment speaker tuples for million-trial multi-enroll sets (e.g. CN-Celeb-E)
        enroll_key_to_centroid: dict[tuple[str, ...], np.ndarray] = {}
        for _, e_utts, _ in trials:
            key = tuple(e_utts)
            if key not in enroll_key_to_centroid:
                normed_list = [
                    utt_to_emb[u] / (np.linalg.norm(utt_to_emb[u]) + 1e-6)
                    for u in key
                ]
                c = np.mean(normed_list, axis=0)
                c = (c / (np.linalg.norm(c) + 1e-6)).astype(np.float32)
                enroll_key_to_centroid[key] = c

        c_keys = list(enroll_key_to_centroid.keys())
        c_idx = {k: i for i, k in enumerate(c_keys)}
        c_mat = np.stack([enroll_key_to_centroid[k] for k in c_keys], axis=0)

        u_keys = unique_utts
        u_idx = {u: i for i, u in enumerate(u_keys)}
        u_mat = np.stack([utt_to_emb[u] for u in u_keys], axis=0).astype(
            np.float32
        )
        u_norm = u_mat / (
            np.linalg.norm(u_mat, axis=-1, keepdims=True) + 1e-6
        )
        e_arr_np = c_mat
        t_arr_np = u_norm

        if scoring_type == "dr_vector" and "_aux_dr_net" in state.params:
            dr_net = scoring.build_decision_residual_network(
                exp_cfg.model.dr_vector
            )
            scores_arr = np.empty((len(trials),), dtype=np.float64)
            for s_i in range(0, len(trials), 65536):
                e_i = min(len(trials), s_i + 65536)
                sub = trials[s_i:e_i]
                sub_e = jnp.asarray(
                    np.stack(
                        [enroll_key_to_centroid[tuple(eu)] for _, eu, _ in sub],
                        axis=0,
                    )
                )
                sub_t = jnp.asarray(
                    np.stack([utt_to_emb[tu] for _, _, tu in sub], axis=0)
                )
                scores_arr[s_i:e_i] = np.asarray(
                    dr_net.apply(
                        {"params": state.params["_aux_dr_net"]},
                        sub_t,
                        sub_e,
                        method=dr_net.score_pairs,
                    )
                )
        else:
            n_tr = len(trials)
            scores_arr = np.empty((n_tr,), dtype=np.float64)
            for s_i in range(0, n_tr, 131072):
                e_i = min(n_tr, s_i + 131072)
                sub = trials[s_i:e_i]
                ci = np.fromiter(
                    (c_idx[tuple(eu)] for _, eu, _ in sub),
                    dtype=np.int64,
                    count=len(sub),
                )
                ti = np.fromiter(
                    (u_idx[tu] for _, _, tu in sub),
                    dtype=np.int64,
                    count=len(sub),
                )
                scores_arr[s_i:e_i] = np.sum(
                    c_mat[ci] * u_norm[ti], axis=-1, dtype=np.float64
                )

    use_as_norm = bool(
        getattr(exp_cfg.eval, "use_as_norm", False)
        or getattr(exp_cfg.eval, "apply_as_norm", False)
    )
    top_k = int(
        getattr(
            exp_cfg.eval,
            "as_norm_top_k",
            getattr(exp_cfg.eval, "as_norm_cohort_size", 300),
        )
    )
    c_miss = float(getattr(exp_cfg.eval, "dcf_c_miss", 1.0))
    c_fa = float(getattr(exp_cfg.eval, "dcf_c_fa", 1.0))

    best_scores_out = scores_arr
    as_norm_subdict: Optional[dict[str, Any]] = None
    if (
        (use_as_norm or (include_as_norm_breakdown and cohort_embeddings is not None))
        and scoring_type in ("cosine", "dr_vector")
        and e_arr_np is not None
        and t_arr_np is not None
    ):
        cohort = require_disjoint_cohort(
            cohort_embeddings, {u: utt_to_emb[u] for u in unique_utts}
        )
        mu_c, std_c = scoring.compute_cohort_topk_stats(c_mat, cohort, top_k=top_k)
        mu_u, std_u = scoring.compute_cohort_topk_stats(u_norm, cohort, top_k=top_k)
        c_keys = list(enroll_key_to_centroid.keys())
        c_idx = {k: i for i, k in enumerate(c_keys)}
        u_idx = {u: i for i, u in enumerate(unique_utts)}
        ci_all = np.fromiter(
            (c_idx[tuple(eu)] for _, eu, _ in trials),
            dtype=np.int64,
            count=len(trials),
        )
        ti_all = np.fromiter(
            (u_idx[tu] for _, _, tu in trials),
            dtype=np.int64,
            count=len(trials),
        )
        norm_scores = 0.5 * (
            (scores_arr - mu_c[ci_all]) / std_c[ci_all]
            + (scores_arr - mu_u[ti_all]) / std_u[ti_all]
        )
        best_scores_out = norm_scores
        if use_as_norm:
            scores_arr = norm_scores
        else:
            as_norm_subdict = _compute_metrics_dict(
                labels,
                norm_scores,
                c_miss=c_miss,
                c_fa=c_fa,
                compute_ci=len(trials) <= 50000,
            )
            as_norm_subdict["top_k"] = top_k
    elif (
        (use_as_norm or (include_as_norm_breakdown and cohort_embeddings is not None))
        and scoring_type == "pfas"
    ):
        cohort = require_disjoint_cohort(
            cohort_embeddings, {u: utt_to_emb[u] for u in unique_utts}
        )
        u_mat_pf = np.stack([utt_to_emb[u] for u in unique_utts], axis=0).astype(
            np.float32
        )
        mu_u, std_u = scoring.compute_pfas_cohort_topk_stats(
            u_mat_pf,
            cohort,
            pfas_cfg=pfas_cfg,
            scale_factor=pfas_scale,
            top_k=top_k,
        )
        u_idx = {u: i for i, u in enumerate(unique_utts)}
        c_keys = list(enroll_key_to_stacked.keys())
        c_idx = {k: i for i, k in enumerate(c_keys)}
        mu_c = np.empty((len(c_keys),), dtype=np.float64)
        std_c = np.empty((len(c_keys),), dtype=np.float64)
        c_by_len: dict[int, list[int]] = {}
        for i_c, k_tup in enumerate(c_keys):
            c_by_len.setdefault(len(k_tup), []).append(i_c)
        for _, c_idxs in c_by_len.items():
            c_sub = np.stack(
                [enroll_key_to_stacked[c_keys[i_c]] for i_c in c_idxs], axis=0
            )
            m_sub, s_sub = scoring.compute_pfas_cohort_topk_stats(
                c_sub,
                cohort,
                pfas_cfg=pfas_cfg,
                scale_factor=pfas_scale,
                top_k=top_k,
            )
            mu_c[np.asarray(c_idxs, dtype=np.int64)] = m_sub
            std_c[np.asarray(c_idxs, dtype=np.int64)] = s_sub
        ci_all = np.fromiter(
            (c_idx[tuple(eu)] for _, eu, _ in trials),
            dtype=np.int64,
            count=len(trials),
        )
        ti_all = np.fromiter(
            (u_idx[tu] for _, _, tu in trials),
            dtype=np.int64,
            count=len(trials),
        )
        norm_scores = 0.5 * (
            (scores_arr - mu_c[ci_all]) / std_c[ci_all]
            + (scores_arr - mu_u[ti_all]) / std_u[ti_all]
        )
        best_scores_out = norm_scores
        if use_as_norm:
            scores_arr = norm_scores
        else:
            as_norm_subdict = _compute_metrics_dict(
                labels,
                norm_scores,
                c_miss=c_miss,
                c_fa=c_fa,
                compute_ci=len(trials) <= 50000,
            )
            as_norm_subdict["top_k"] = top_k

    out = _compute_metrics_dict(
        labels,
        scores_arr,
        c_miss=c_miss,
        c_fa=c_fa,
        compute_ci=len(trials) <= 50000,
    )
    out["num_trials"] = len(trials)
    if as_norm_subdict is not None:
        out["as_norm"] = as_norm_subdict
    if return_scores:
        return out, best_scores_out
    return out


def run_eval(myconfig: Any) -> dict[str, Any]:
    """Run evaluation of the saved model on test data."""
    exp_cfg = configs.ExperimentConfig.from_any(myconfig)
    spk_to_utts = dataset.load_dataset_from_config(exp_cfg, split="eval")
    _, state = neural_net.get_speaker_encoder(
        exp_cfg, exp_cfg.model.saved_model_path
    )
    trials_file = getattr(exp_cfg.data, "trials_file", None)
    if trials_file:
        trials = dataset.load_trials_file(trials_file)
    else:
        trials = dataset.generate_verification_trials(
            spk_to_utts, num_trials=exp_cfg.eval.num_triplets * 2
        )
    metrics = evaluate_verification_trials(state, trials, exp_cfg)
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
    run_eval(configs.ExperimentConfig.from_yaml(config_file))
