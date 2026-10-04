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

from typing import Sequence
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def cosine_similarity(a: jax.Array, b: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Compute cosine similarity between two 1D vectors `a` and `b`."""
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + eps)


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
    num_keys: int,
    key_dim: int,
    value_dim: int,
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


def pfas_score_pair(
    test_keys: jax.Array,
    test_values: jax.Array,
    enroll_keys: jax.Array,
    enroll_values: jax.Array,
    scale_factor: float = 5.0,
    per_trial_softmax: bool = True,
) -> jax.Array:
    """Compute Parameter-Free Attentive Score between one test and one enrollment set.

    Supports both single-utterance enrollment (`enroll_keys` has `M` keys) and
    multi-utterance enrollment (`enroll_keys` has `U * M` keys stacked across `U`
    enrollment utterances).

    Args:
        test_keys: `(M_t, d_k)`
        test_values: `(M_t, d_v)`
        enroll_keys: `(M_e, d_k)`
        enroll_values: `(M_e, d_v)`
        scale_factor: Softmax temperature scaling factor `s`.
        per_trial_softmax: If True, softmax is computed over all `M_t * M_e` pairs
            jointly (Full-Matrix Softmax). If False, softmax is computed across
            enrollment keys `M_e` per test key and averaged over `M_t`.

    Returns:
        Scalar similarity score.
    """
    # Key-key dot products: (M_t, M_e)
    key_sim = scale_factor * jnp.matmul(test_keys, enroll_keys.T)
    # Value-value dot products: (M_t, M_e)
    val_sim = jnp.matmul(test_values, enroll_values.T)

    if per_trial_softmax:
        weights = jax.nn.softmax(key_sim.reshape(-1)).reshape(key_sim.shape)
    else:
        weights = jax.nn.softmax(key_sim, axis=-1) / float(test_keys.shape[0])

    return jnp.sum(weights * val_sim)


def pfas_score_matrix(
    test_packed: jax.Array,
    enroll_packed: jax.Array,
    num_keys: int = 4,
    key_dim: int = 64,
    value_dim: int = 64,
    scale_factor: float = 5.0,
    per_trial_softmax: bool = True,
) -> jax.Array:
    """Compute `(N_test, N_enroll)` PFAS similarity matrix for batched training/eval.

    Args:
        test_packed: `(N_test, M * (key_dim + value_dim))`
        enroll_packed: `(N_enroll, M_e * (key_dim + value_dim))`
        num_keys: Number of keys `M` per test utterance.
        key_dim: Dimension of each key vector.
        value_dim: Dimension of each value vector.
        scale_factor: Temperature scaling factor for key-key attention logits.
        per_trial_softmax: Full-matrix trial softmax vs per-test-key softmax.

    Returns:
        Similarity matrix of shape `(N_test, N_enroll)`.
    """
    kv_unit = key_dim + value_dim
    num_enroll_keys = enroll_packed.shape[-1] // kv_unit
    t_keys, t_vals = unpack_pfas_representation(test_packed, num_keys, key_dim, value_dim)
    e_keys, e_vals = unpack_pfas_representation(
        enroll_packed, num_enroll_keys, key_dim, value_dim
    )

    # t_keys: (N_t, M_t, d_k), e_keys: (N_e, M_e, d_k)
    # key_logits: (N_t, N_e, M_t, M_e)
    key_logits = scale_factor * jnp.einsum("tpd,eqd->tepq", t_keys, e_keys)
    val_dots = jnp.einsum("tpd,eqd->tepq", t_vals, e_vals)

    n_t, n_e, m_t, m_e = key_logits.shape
    if per_trial_softmax:
        flat_logits = key_logits.reshape(n_t, n_e, m_t * m_e)
        weights = jax.nn.softmax(flat_logits, axis=-1).reshape(n_t, n_e, m_t, m_e)
    else:
        weights = jax.nn.softmax(key_logits, axis=-1) / float(m_t)

    return jnp.sum(weights * val_dots, axis=(-2, -1))


def stack_multi_enroll_pfas(
    enroll_embeddings: Sequence[jax.Array],
    num_keys: int,
    key_dim: int,
    value_dim: int,
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
    `score(e_t, e_e) = cos(e_t, e_e) + w_res * MLP([e_t_norm, e_e_norm, cos(e_t, e_e)])`
    """

    hidden_dims: Sequence[int] = (128, 64)
    activation: str = "relu"
    use_layer_norm: bool = True
    include_cosine_in_input: bool = True
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

        if self.include_cosine_in_input:
            h = jnp.concatenate([t_norm, e_norm, cos_sim], axis=-1)
        else:
            h = jnp.concatenate([t_norm, e_norm], axis=-1)

        for idx, h_dim in enumerate(self.hidden_dims):
            h = nn.Dense(h_dim, name=f"dr_fc_{idx}")(h)
            if self.use_layer_norm:
                h = nn.LayerNorm(name=f"dr_ln_{idx}")(h)
            h = self._act_fn(h)

        res = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="dr_out",
        )(h)
        return (cos_sim + self.residual_weight * res).squeeze(-1)

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
