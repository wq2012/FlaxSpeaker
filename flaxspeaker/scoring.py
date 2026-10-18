"""Speaker verification scoring functions and networks.

Implements:
- Cosine similarity scoring (pairwise and matrix)
- Parameter-Free Attentive Scoring (PFAS, Pelecanos et al., Odyssey 2022,
  https://arxiv.org/pdf/2203.05642v3) for both single-utterance and
  multi-utterance enrollment without parameter overhead
- Decision Residual Networks (Dr-Vectors, Pelecanos et al., Interspeech 2021,
  https://arxiv.org/pdf/2104.01989) combining cosine similarity with a learned
  nonlinear residual scoring branch
- Adaptive Symmetric Score Normalization (AS-Norm)
"""

from typing import Sequence, Union
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def cosine_similarity(a: jax.Array, b: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Compute cosine similarity between two 1D vectors `a` and `b`."""
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + eps)


@jax.jit
def compute_speaker_centroid(
    embeddings: jax.Array, eps: float = 1e-6
) -> jax.Array:
    """Compute L2-normalized centroid of `(K, D)` enrollment embeddings."""
    normed = embeddings / (
        jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + eps
    )
    mean_vec = jnp.mean(normed, axis=0)
    return mean_vec / (jnp.linalg.norm(mean_vec, axis=-1, keepdims=True) + eps)


@jax.jit
def cosine_similarity_matrix(
    test_embs: jax.Array,
    enroll_embs: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    """Compute `(N_test, N_enroll)` cosine similarity matrix."""
    test_norm = test_embs / (jnp.linalg.norm(test_embs, axis=-1, keepdims=True) + eps)
    enroll_norm = enroll_embs / (jnp.linalg.norm(enroll_embs, axis=-1, keepdims=True) + eps)
    return jnp.matmul(test_norm, enroll_norm.T)


def unpack_pfas_representation(
    packed: jax.Array,
    num_keys: int = 4,
    key_dim: int = 32,
    value_dim: int = 128,
) -> tuple[jax.Array, jax.Array]:
    """Unpack flat PFAS embedding `(..., M*(key_dim + value_dim))` into `(keys, values)`.

    Returns:
        `keys` of shape `(..., M, key_dim)` and `values` of shape `(..., M, value_dim)`.
    """
    prefix_shape = packed.shape[:-1]
    key_total = num_keys * key_dim
    val_total = num_keys * value_dim
    if packed.shape[-1] != key_total + val_total:
        raise ValueError(
            f"Expected last dim {key_total + val_total} for num_keys={num_keys}, "
            f"key_dim={key_dim}, value_dim={value_dim}, got {packed.shape[-1]}."
        )
    keys = packed[..., :key_total].reshape(*prefix_shape, num_keys, key_dim)
    values = packed[..., key_total:].reshape(*prefix_shape, num_keys, value_dim)
    return keys, values


def _compute_pfas_weights(
    key_dots: jax.Array,
    scale_factor: Union[jax.Array, float] = 16.0,
    per_trial_softmax: bool = True,
    apply_softmax_per_test_key: bool = False,
    attention_weight_mode: str = "exp",
    eps: float = 1e-6,
) -> jax.Array:
    """Computes normalized PFAS key-key attention weights over `(..., M_t, M_e)`.

    Implements Eq. 5 of Pelecanos et al.
    (Odyssey 2022, https://arxiv.org/pdf/2203.05642v3).
    """
    mode = attention_weight_mode.lower()
    m_t = key_dots.shape[-2]
    m_e = key_dots.shape[-1]
    prefix_shape = key_dots.shape[:-2]
    use_per_trial = bool(per_trial_softmax and not apply_softmax_per_test_key)

    if mode == "outer":
        logits = scale_factor * key_dots
        w_row = jax.nn.softmax(logits, axis=-2)
        w_col = jax.nn.softmax(logits, axis=-1)
        weights = 0.5 * (w_row + w_col) + eps
    elif mode == "square":
        weights = jnp.square(key_dots) + eps
    elif mode == "abs":
        weights = jnp.abs(key_dots) + eps
    else:
        # Default "exp" (Full-Matrix Softmax over M_t * M_e when per_trial_softmax=True)
        logits = scale_factor * key_dots
        if use_per_trial:
            flat_logits = logits.reshape(*prefix_shape, m_t * m_e)
            weights = (
                jax.nn.softmax(flat_logits, axis=-1).reshape(
                    *prefix_shape, m_t, m_e
                )
                + eps
            )
        else:
            weights = (jax.nn.softmax(logits, axis=-1) / float(m_t)) + eps

    weights = weights / (
        jnp.sum(weights, axis=(-2, -1), keepdims=True) + eps
    )
    return weights


def pfas_score_pair(
    test_keys: jax.Array,
    test_values: jax.Array,
    enroll_keys: jax.Array,
    enroll_values: jax.Array,
    scale_factor: Union[jax.Array, float] = 16.0,
    per_trial_softmax: bool = True,
    apply_softmax_per_test_key: bool = False,
    apply_l2_norm_to_keys: bool = True,
    apply_l2_norm_to_values: bool = False,
    apply_global_l2_norm_to_concat_form: bool = True,
    attention_weight_mode: str = "exp",
    eps: float = 1e-6,
) -> jax.Array:
    """Compute Parameter-Free Attentive Score between one test and one enrollment set.

    Implements the "Key & Global L2-Norm" formulation from Pelecanos et al.
    (Odyssey 2022, https://arxiv.org/pdf/2203.05642v3, Eq. 5-6 & Table 2).

    Supports both single-utterance enrollment (`enroll_keys` has `M` keys) and
    multi-utterance enrollment (`enroll_keys` has `U * M` keys stacked across `U`
    enrollment utterances).
    """
    t_k = (
        test_keys / (jnp.linalg.norm(test_keys, axis=-1, keepdims=True) + eps)
        if apply_l2_norm_to_keys
        else test_keys
    )
    e_k = (
        enroll_keys
        / (jnp.linalg.norm(enroll_keys, axis=-1, keepdims=True) + eps)
        if apply_l2_norm_to_keys
        else enroll_keys
    )
    t_v = (
        test_values
        / (jnp.linalg.norm(test_values, axis=-1, keepdims=True) + eps)
        if apply_l2_norm_to_values
        else test_values
    )
    e_v = (
        enroll_values
        / (jnp.linalg.norm(enroll_values, axis=-1, keepdims=True) + eps)
        if apply_l2_norm_to_values
        else enroll_values
    )

    # Key-key dot products: (M_t, M_e)
    key_dots = jnp.matmul(t_k, e_k.T)
    weights = _compute_pfas_weights(
        key_dots,
        scale_factor=scale_factor,
        per_trial_softmax=per_trial_softmax,
        apply_softmax_per_test_key=apply_softmax_per_test_key,
        attention_weight_mode=attention_weight_mode,
        eps=eps,
    )

    # Value-value dot products: (M_t, M_e)
    val_dots = jnp.matmul(t_v, e_v.T)
    score = jnp.sum(weights * val_dots)

    if apply_global_l2_norm_to_concat_form:
        t_val_sq = jnp.sum(jnp.square(t_v), axis=-1)  # (M_t,)
        e_val_sq = jnp.sum(jnp.square(e_v), axis=-1)  # (M_e,)
        w_t = jnp.sum(weights, axis=-1)  # (M_t,)
        w_e = jnp.sum(weights, axis=-2)  # (M_e,)
        norm_t = jnp.sqrt(jnp.sum(w_t * t_val_sq) + eps)
        norm_e = jnp.sqrt(jnp.sum(w_e * e_val_sq) + eps)
        score = score / jnp.maximum(norm_t * norm_e, eps)

    return score


def pfas_score_matrix(
    test_packed: jax.Array,
    enroll_packed: jax.Array,
    num_keys: int = 4,
    key_dim: int = 32,
    value_dim: int = 128,
    scale_factor: Union[jax.Array, float] = 16.0,
    per_trial_softmax: bool = True,
    apply_softmax_per_test_key: bool = False,
    apply_l2_norm_to_keys: bool = True,
    apply_l2_norm_to_values: bool = False,
    apply_global_l2_norm_to_concat_form: bool = True,
    attention_weight_mode: str = "exp",
    eps: float = 1e-6,
) -> jax.Array:
    """Compute `(N_test, N_enroll)` PFAS similarity matrix for batched training/eval.

    Implements Eq. 5-6 of Pelecanos et al. (`arXiv:2203.05642v3`).
    """
    kv_unit = key_dim + value_dim
    num_enroll_keys = enroll_packed.shape[-1] // kv_unit
    t_keys, t_vals = unpack_pfas_representation(
        test_packed, num_keys, key_dim, value_dim
    )
    e_keys, e_vals = unpack_pfas_representation(
        enroll_packed, num_enroll_keys, key_dim, value_dim
    )

    if apply_l2_norm_to_keys:
        t_keys = t_keys / (
            jnp.linalg.norm(t_keys, axis=-1, keepdims=True) + eps
        )
        e_keys = e_keys / (
            jnp.linalg.norm(e_keys, axis=-1, keepdims=True) + eps
        )
    if apply_l2_norm_to_values:
        t_vals = t_vals / (
            jnp.linalg.norm(t_vals, axis=-1, keepdims=True) + eps
        )
        e_vals = e_vals / (
            jnp.linalg.norm(e_vals, axis=-1, keepdims=True) + eps
        )

    # t_keys: (N_t, M_t, d_k), e_keys: (N_e, M_e, d_k)
    # key_dots: (N_t, N_e, M_t, M_e)
    key_dots = jnp.einsum("tpd,eqd->tepq", t_keys, e_keys)
    weights = _compute_pfas_weights(
        key_dots,
        scale_factor=scale_factor,
        per_trial_softmax=per_trial_softmax,
        apply_softmax_per_test_key=apply_softmax_per_test_key,
        attention_weight_mode=attention_weight_mode,
        eps=eps,
    )

    val_dots = jnp.einsum("tpd,eqd->tepq", t_vals, e_vals)
    scores = jnp.sum(weights * val_dots, axis=(-2, -1))  # (N_t, N_e)

    if apply_global_l2_norm_to_concat_form:
        # _calc_l2_norm_for_concat_form:
        # norm_t(t, e) = sqrt(sum_{p, q} w(t,e,p,q) ||t_vals(t,p)||^2)
        # norm_e(t, e) = sqrt(sum_{p, q} w(t,e,p,q) ||e_vals(e,q)||^2)
        t_val_sq = jnp.sum(jnp.square(t_vals), axis=-1)  # (N_t, M_t)
        e_val_sq = jnp.sum(jnp.square(e_vals), axis=-1)  # (N_e, M_e)
        w_t = jnp.sum(weights, axis=-1)  # (N_t, N_e, M_t)
        w_e = jnp.sum(weights, axis=-2)  # (N_t, N_e, M_e)
        norm_t = jnp.sqrt(jnp.einsum("tep,tp->te", w_t, t_val_sq) + eps)
        norm_e = jnp.sqrt(jnp.einsum("teq,eq->te", w_e, e_val_sq) + eps)
        scores = scores / jnp.maximum(norm_t * norm_e, eps)

    return scores


def stack_multi_enroll_pfas(
    enroll_embeddings: Sequence[jax.Array],
    num_keys: int = 4,
    key_dim: int = 32,
    value_dim: int = 128,
) -> jax.Array:
    """Stack multiple enrollment utterance PFAS embeddings into `(U*M, d_k)` & `(U*M, d_v)`.

    Returns a single packed vector of length `U * M * (key_dim + value_dim)` so
    multi-utterance enrollment can be scored directly by `pfas_score_pair` or
    `pfas_score_matrix`.
    """
    keys_list = []
    vals_list = []
    for emb in enroll_embeddings:
        k, v = unpack_pfas_representation(jnp.asarray(emb), num_keys, key_dim, value_dim)
        keys_list.append(k)
        vals_list.append(v)
    all_keys = jnp.concatenate(keys_list, axis=0)  # (U*M, d_k)
    all_vals = jnp.concatenate(vals_list, axis=0)  # (U*M, d_v)
    return jnp.concatenate([all_keys.reshape(-1), all_vals.reshape(-1)], axis=-1)


class DecisionResidualNetwork(nn.Module):
    """Decision Residual Network (Dr-Vectors) scoring module.

    Reference:
    "Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker
    Recognition" (Pelecanos, Wang, Moreno, Interspeech 2021,
    https://arxiv.org/pdf/2104.01989).

    Given a test embedding `e_t` and enrollment embedding `e_e`, computes:
    `score(e_t, e_e) = cos(e_t, e_e) + w_res * MLP([e_t_norm, e_e_norm, cos_feats])`
    Supports non-linear cosine features (`num_cosine_terms = 3` for
    `[cos, cos^2, sqrt(max(cos, 0))]` as in Eq. 7 of Pelecanos et al., 2021, or
    `num_cosine_terms = 1` default), optional skip-concatenation of cosine features
    to the final hidden representation (`append_cosine_to_final_hidden = True`,
    Figure 1 of arXiv:2104.01989), and optional symmetric pair features
    (`symmetric_input = True`).
    """

    hidden_dims: Sequence[int] = (128, 64)
    activation: str = "relu"
    use_layer_norm: bool = True
    include_cosine_in_input: bool = True
    append_cosine_as_feature_to_network_input: bool = True
    append_cosine_to_final_hidden: bool = False
    add_cosine_to_network_output: bool = True
    num_cosine_terms: int = 1
    symmetric_input: bool = False
    residual_weight: float = 0.25
    eps: float = 1e-6

    def _act_fn(self, x: jax.Array) -> jax.Array:
        act = self.activation.lower()
        if act == "elu":
            return nn.elu(x)
        elif act == "gelu":
            return nn.gelu(x)
        elif act == "tanh":
            return jnp.tanh(x)
        return nn.relu(x)

    @nn.compact
    def score_pairs(self, test_embs: jax.Array, enroll_embs: jax.Array) -> jax.Array:
        """Score aligned `(B, D)` pairs of `(test_embs, enroll_embs)` -> `(B,)`."""
        t_norm = test_embs / (jnp.linalg.norm(test_embs, axis=-1, keepdims=True) + self.eps)
        e_norm = enroll_embs / (jnp.linalg.norm(enroll_embs, axis=-1, keepdims=True) + self.eps)
        cos_sim = jnp.sum(t_norm * e_norm, axis=-1, keepdims=True)  # (B, 1)

        if self.symmetric_input:
            pair_feats = [0.5 * (t_norm + e_norm), jnp.abs(t_norm - e_norm)]
        else:
            pair_feats = [t_norm, e_norm]

        cos_feats: list[jax.Array] = []
        n_terms = max(1, int(self.num_cosine_terms))
        if n_terms == 1:
            cos_feats = [cos_sim]
        elif n_terms == 3:
            # Pelecanos et al. (Interspeech 2021, Eq. 7): [c, c^2, sqrt(max(c, 0))]
            cos_feats = [
                cos_sim,
                jnp.square(cos_sim),
                jnp.sqrt(jnp.maximum(cos_sim, 0.0) + self.eps),
            ]
        else:
            cos_feats = [jnp.power(cos_sim, p) for p in range(1, n_terms + 1)]

        use_cos_in_input = bool(
            self.include_cosine_in_input and self.append_cosine_as_feature_to_network_input
        )
        if use_cos_in_input:
            pair_feats.extend(cos_feats)

        h = jnp.concatenate(pair_feats, axis=-1)

        for idx, h_dim in enumerate(self.hidden_dims):
            h = nn.Dense(h_dim, name=f"dr_fc_{idx}")(h)
            if self.use_layer_norm:
                h = nn.LayerNorm(name=f"dr_ln_{idx}")(h)
            h = self._act_fn(h)

        if self.append_cosine_to_final_hidden:
            h = jnp.concatenate([h] + cos_feats, axis=-1)

        res = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="dr_out",
        )(h)
        if self.add_cosine_to_network_output:
            return (cos_sim + self.residual_weight * res).squeeze(-1)
        return (self.residual_weight * res).squeeze(-1)

    @nn.compact
    def __call__(self, test_embs: jax.Array, enroll_embs: jax.Array) -> jax.Array:
        """Compute `(N_test, N_enroll)` Dr-Vector similarity matrix."""
        n_test, dim = test_embs.shape
        n_enroll = enroll_embs.shape[0]
        t_rep = jnp.broadcast_to(test_embs[:, None, :], (n_test, n_enroll, dim)).reshape(
            n_test * n_enroll, dim
        )
        e_rep = jnp.broadcast_to(enroll_embs[None, :, :], (n_test, n_enroll, dim)).reshape(
            n_test * n_enroll, dim
        )
        flat_scores = self.score_pairs(t_rep, e_rep)
        return flat_scores.reshape(n_test, n_enroll)


def apply_as_norm(
    raw_scores: np.ndarray,
    enroll_embs: np.ndarray,
    test_embs: np.ndarray,
    cohort_embs: np.ndarray,
    top_k: int = 100,
    eps: float = 1e-6,
) -> np.ndarray:
    """Adaptive Symmetric Score Normalization (AS-Norm).

    Normalizes raw cosine verification scores for pairs `(enroll_embs[i], test_embs[i])`
    against an imposter cohort `cohort_embs` of shape `(N_cohort, D)` using the
    top-`top_k` highest-scoring cohort imposters for each side.
    """
    e_norm = enroll_embs / (np.linalg.norm(enroll_embs, axis=-1, keepdims=True) + eps)
    t_norm = test_embs / (np.linalg.norm(test_embs, axis=-1, keepdims=True) + eps)
    c_norm = cohort_embs / (np.linalg.norm(cohort_embs, axis=-1, keepdims=True) + eps)

    k = min(top_k, c_norm.shape[0])
    e_cohort = np.matmul(e_norm, c_norm.T)  # (N_trials, N_cohort)
    t_cohort = np.matmul(t_norm, c_norm.T)  # (N_trials, N_cohort)

    e_topk = np.sort(e_cohort, axis=-1)[:, -k:]
    t_topk = np.sort(t_cohort, axis=-1)[:, -k:]

    mu_e = np.mean(e_topk, axis=-1)
    std_e = np.maximum(np.std(e_topk, axis=-1), eps)
    mu_t = np.mean(t_topk, axis=-1)
    std_t = np.maximum(np.std(t_topk, axis=-1), eps)

    return 0.5 * ((raw_scores - mu_e) / std_e + (raw_scores - mu_t) / std_t)
