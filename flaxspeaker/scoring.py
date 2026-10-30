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

from typing import Any, Optional, Sequence, Union
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
    num_test_keys = (
        test_packed.shape[-1] // kv_unit
        if test_packed.shape[-1] % kv_unit == 0
        else num_keys
    )
    num_enroll_keys = enroll_packed.shape[-1] // kv_unit
    t_keys, t_vals = unpack_pfas_representation(
        test_packed, num_test_keys, key_dim, value_dim
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
    https://arxiv.org/pdf/2104.01989). The paper contains two numbered
    equations only (Eq. 1 GE2E softmax, Eq. 2 GE2E extended-set softmax); the
    decision network itself is described in its Sec. 2 / Sec. 4 and Fig. 1.

    Given a test embedding `e_t` and enrollment embedding `e_e`:

    - `cosine_dims > 0` (paper: `d = 200` base dims of a 256-D embedding) restricts
      the cosine score to the first `cosine_dims` dimensions; the remaining
      side-information dimensions are only seen by the decision network.
      `cosine_dims = 0` (FlaxSpeaker default) uses all dimensions.
    - The decision network input is `[e_t_norm, e_e_norm, phi(c)]` (switch "B" of the
      paper when `append_cosine_as_feature_to_network_input=True`).
    - Hidden layers: `hidden_dims` Dense layers with `activation`
      (paper: 3 x 256, leaky-ReLU slope 0.2, no normalisation; FlaxSpeaker default:
      (128, 64) ReLU + LayerNorm).
    - `readout="weighted_sum"` (paper) forms the intermediate score as a learned
      weighted sum over *all* hidden-layer activations; `readout="linear"`
      (FlaxSpeaker default) uses a linear layer on the last hidden layer.
    - `score = c + residual_weight * net` when `add_cosine_to_network_output`
      (switch "A"/"C"); `trainable_affine=True` (paper) then applies a trainable
      `scale * score + offset`.
    - `num_cosine_terms = 3` (`phi(c) = [c, c^2, sqrt(max(c, 0))]`) is a FlaxSpeaker
      extension that does **not** appear in the paper.
    """

    hidden_dims: Sequence[int] = (128, 64)
    activation: str = "relu"
    leaky_relu_slope: float = 0.2
    use_layer_norm: bool = True
    include_cosine_in_input: bool = True
    append_cosine_as_feature_to_network_input: bool = True
    append_cosine_to_final_hidden: bool = False
    skip_connect_cosine_to_readout: bool = False
    add_cosine_to_network_output: bool = True
    num_cosine_terms: int = 1
    symmetric_input: bool = False
    residual_weight: float = 0.25
    cosine_dims: int = 0
    readout: str = "linear"
    trainable_affine: bool = False
    eps: float = 1e-6

    def _act_fn(self, x: jax.Array) -> jax.Array:
        act = self.activation.lower()
        if act == "elu":
            return nn.elu(x)
        elif act == "gelu":
            return nn.gelu(x)
        elif act == "tanh":
            return jnp.tanh(x)
        elif act == "leaky_relu":
            return nn.leaky_relu(x, negative_slope=float(self.leaky_relu_slope))
        return nn.relu(x)

    @nn.compact
    def score_pairs(self, test_embs: jax.Array, enroll_embs: jax.Array) -> jax.Array:
        """Score aligned `(B, D)` pairs of `(test_embs, enroll_embs)` -> `(B,)`."""
        t_norm = test_embs / (jnp.linalg.norm(test_embs, axis=-1, keepdims=True) + self.eps)
        e_norm = enroll_embs / (jnp.linalg.norm(enroll_embs, axis=-1, keepdims=True) + self.eps)
        d_cos = int(self.cosine_dims)
        if d_cos > test_embs.shape[-1]:
            raise ValueError(
                f"cosine_dims ({d_cos}) cannot exceed embedding dimension ({test_embs.shape[-1]})."
            )
        if d_cos > 0 and d_cos < test_embs.shape[-1]:
            # Paper: cosine over the `d` base dimensions only (re-normalised).
            t_base = test_embs[..., :d_cos]
            e_base = enroll_embs[..., :d_cos]
            t_base = t_base / (jnp.linalg.norm(t_base, axis=-1, keepdims=True) + self.eps)
            e_base = e_base / (jnp.linalg.norm(e_base, axis=-1, keepdims=True) + self.eps)
            cos_sim = jnp.sum(t_base * e_base, axis=-1, keepdims=True)  # (B, 1)
        else:
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
            # FlaxSpeaker extension (not in Pelecanos et al. 2021): [c, c^2, sqrt(max(c, 0))]
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

        hidden_outs: list[jax.Array] = []
        for idx, h_dim in enumerate(self.hidden_dims):
            h = nn.Dense(h_dim, name=f"dr_fc_{idx}")(h)
            if self.use_layer_norm:
                h = nn.LayerNorm(name=f"dr_ln_{idx}")(h)
            h = self._act_fn(h)
            hidden_outs.append(h)

        readout = self.readout.lower()
        if readout == "weighted_sum":
            # Paper: "The 3 layers are followed by a weighted summation operation to
            # give a single intermediate score" -- a learned linear read-out over the
            # concatenation of all hidden-layer activations.
            h = jnp.concatenate(hidden_outs, axis=-1)
        elif readout != "linear":
            raise ValueError(f"Unknown Dr-Vector readout: {self.readout}")

        if self.append_cosine_to_final_hidden or self.skip_connect_cosine_to_readout:
            h = jnp.concatenate([h] + cos_feats, axis=-1)

        res = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="dr_out",
        )(h)
        if self.add_cosine_to_network_output:
            score = cos_sim + self.residual_weight * res
        else:
            score = self.residual_weight * res
        if self.trainable_affine:
            # Paper: "This score is scaled and an offset is applied and is optimized
            # as part of the end-to-end training loss."
            affine_scale = self.param("dr_affine_scale", nn.initializers.ones, (1,))
            affine_offset = self.param("dr_affine_offset", nn.initializers.zeros, (1,))
            score = score * affine_scale + affine_offset
        return score.squeeze(-1)

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


def build_decision_residual_network(dr_cfg: Any) -> DecisionResidualNetwork:
    """Builds a `DecisionResidualNetwork` from a `configs.DrVectorConfig` (single source of truth)."""
    return DecisionResidualNetwork(
        hidden_dims=tuple(dr_cfg.hidden_dims),
        activation=str(dr_cfg.activation),
        leaky_relu_slope=float(getattr(dr_cfg, "leaky_relu_slope", 0.2)),
        use_layer_norm=bool(dr_cfg.use_layer_norm),
        include_cosine_in_input=bool(dr_cfg.include_cosine_in_input),
        append_cosine_as_feature_to_network_input=bool(
            getattr(dr_cfg, "append_cosine_as_feature_to_network_input", True)
        ),
        append_cosine_to_final_hidden=bool(
            getattr(dr_cfg, "append_cosine_to_final_hidden", False)
        ),
        add_cosine_to_network_output=bool(
            getattr(dr_cfg, "add_cosine_to_network_output", True)
        ),
        num_cosine_terms=int(getattr(dr_cfg, "num_cosine_terms", 1)),
        symmetric_input=bool(getattr(dr_cfg, "symmetric_input", False)),
        residual_weight=float(dr_cfg.residual_weight),
        cosine_dims=int(getattr(dr_cfg, "cosine_dims", 0)),
        readout=str(getattr(dr_cfg, "readout", "linear")),
        trainable_affine=bool(getattr(dr_cfg, "trainable_affine", False)),
    )


def compute_cohort_topk_stats(
    embs: np.ndarray,
    cohort_embs: np.ndarray,
    top_k: int = 300,
    chunk_size: int = 4096,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-`top_k` cohort mean and std for `(N, D)` embeddings in bounded memory.

    Processes `embs` in chunks of `chunk_size` rows and uses `O(C)` `np.partition`
    rather than full `O(C log C)` sorting so even million-trial cohorts require
    `< 50 MB` peak RAM.
    """
    x = np.asarray(embs, dtype=np.float32)
    c = np.asarray(cohort_embs, dtype=np.float32)
    x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    c_norm = c / (np.linalg.norm(c, axis=-1, keepdims=True) + eps)

    n = x_norm.shape[0]
    k = max(1, min(int(top_k), c_norm.shape[0]))
    mu = np.empty((n,), dtype=np.float64)
    std = np.empty((n,), dtype=np.float64)

    c_t = c_norm.T
    step = max(1, int(chunk_size))
    for start in range(0, n, step):
        end = min(n, start + step)
        sims = np.matmul(x_norm[start:end], c_t)  # (chunk, N_cohort)
        if k < sims.shape[1]:
            topk = np.partition(sims, sims.shape[1] - k, axis=-1)[:, -k:]
        else:
            topk = sims
        mu[start:end] = np.mean(topk, axis=-1, dtype=np.float64)
        std[start:end] = np.maximum(
            np.std(topk, axis=-1, dtype=np.float64), eps
        )
    return mu, std


def compute_pfas_cohort_topk_stats(
    embs: np.ndarray,
    cohort_embs: np.ndarray,
    pfas_cfg: Any,
    scale_factor: float = 16.0,
    top_k: int = 100,
    chunk_size: int = 256,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-`top_k` PFAS cohort mean and std for `(N, D)` packed embeddings."""
    import functools

    x = np.asarray(embs, dtype=np.float32)
    c_j = jnp.asarray(cohort_embs, dtype=jnp.float32)
    score_mat_fn = jax.jit(
        functools.partial(
            pfas_score_matrix,
            num_keys=int(pfas_cfg.num_keys),
            key_dim=int(pfas_cfg.key_dim),
            value_dim=int(pfas_cfg.value_dim),
            scale_factor=float(scale_factor),
            per_trial_softmax=bool(pfas_cfg.per_trial_softmax),
            apply_softmax_per_test_key=bool(
                getattr(pfas_cfg, "apply_softmax_per_test_key", False)
            ),
            apply_l2_norm_to_keys=bool(
                getattr(pfas_cfg, "apply_l2_norm_to_keys", True)
            ),
            apply_l2_norm_to_values=bool(
                getattr(pfas_cfg, "apply_l2_norm_to_values", False)
            ),
            apply_global_l2_norm_to_concat_form=bool(
                getattr(pfas_cfg, "apply_global_l2_norm_to_concat_form", True)
            ),
            attention_weight_mode=str(
                getattr(pfas_cfg, "attention_weight_mode", "exp")
            ),
            eps=eps,
        )
    )
    n = x.shape[0]
    k = max(1, min(int(top_k), cohort_embs.shape[0]))
    mu = np.empty((n,), dtype=np.float64)
    std = np.empty((n,), dtype=np.float64)
    step = max(1, int(chunk_size))
    for start in range(0, n, step):
        end = min(n, start + step)
        sims = np.asarray(
            score_mat_fn(jnp.asarray(x[start:end]), c_j), dtype=np.float64
        )
        if k < sims.shape[1]:
            topk = np.partition(sims, sims.shape[1] - k, axis=-1)[:, -k:]
        else:
            topk = sims
        mu[start:end] = np.mean(topk, axis=-1, dtype=np.float64)
        std[start:end] = np.maximum(
            np.std(topk, axis=-1, dtype=np.float64), eps
        )
    return mu, std


def apply_as_norm(
    raw_scores: np.ndarray,
    enroll_embs: np.ndarray,
    test_embs: np.ndarray,
    cohort_embs: np.ndarray,
    top_k: int = 300,
    chunk_size: int = 4096,
    eps: float = 1e-6,
) -> np.ndarray:
    """Adaptive Symmetric Score Normalization (AS-Norm).

    Normalizes raw cosine verification scores for pairs `(enroll_embs[i], test_embs[i])`
    against an imposter cohort `cohort_embs` of shape `(N_cohort, D)` using the
    top-`top_k` highest-scoring cohort imposters for each side.
    """
    scores = np.asarray(raw_scores, dtype=np.float64)
    mu_e, std_e = compute_cohort_topk_stats(
        enroll_embs, cohort_embs, top_k=top_k, chunk_size=chunk_size, eps=eps
    )
    mu_t, std_t = compute_cohort_topk_stats(
        test_embs, cohort_embs, top_k=top_k, chunk_size=chunk_size, eps=eps
    )
    return 0.5 * ((scores - mu_e) / std_e + (scores - mu_t) / std_t)


class QualityMeasureCalibrator:
    """Logistic Quality Measure Function (QMF) score calibrator (Thienpondt et al., 2021).

    Fits an L2-regularized logistic regression model on a speaker-disjoint validation
    trial set combining the verification score (raw or AS-Norm) with symmetric duration
    and pre-norm embedding magnitude quality features:
    `[score, log(1 + T_min), log(1 + T_max), |log(1 + T_e) - log(1 + T_t)|, min(norm_e, norm_t), |norm_e - norm_t|]`.
    """

    def __init__(self, c_reg: float = 1.0):
        self.c_reg = float(c_reg)
        self._clf: Any = None

    @staticmethod
    def build_features(
        scores: np.ndarray,
        enroll_frames: Optional[np.ndarray] = None,
        test_frames: Optional[np.ndarray] = None,
        enroll_norms: Optional[np.ndarray] = None,
        test_norms: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        cols = [s]
        n = s.shape[0]
        if enroll_frames is not None and test_frames is not None:
            le = np.log1p(
                np.maximum(np.asarray(enroll_frames, dtype=np.float64), 0.0)
            ).reshape(n, 1)
            lt = np.log1p(
                np.maximum(np.asarray(test_frames, dtype=np.float64), 0.0)
            ).reshape(n, 1)
            cols.extend(
                [np.minimum(le, lt), np.maximum(le, lt), np.abs(le - lt)]
            )
        if enroll_norms is not None and test_norms is not None:
            ne = np.asarray(enroll_norms, dtype=np.float64).reshape(n, 1)
            nt = np.asarray(test_norms, dtype=np.float64).reshape(n, 1)
            cols.extend([np.minimum(ne, nt), np.abs(ne - nt)])
        return np.concatenate(cols, axis=1)

    def fit(
        self,
        scores: np.ndarray,
        labels: Sequence[int],
        enroll_frames: Optional[np.ndarray] = None,
        test_frames: Optional[np.ndarray] = None,
        enroll_norms: Optional[np.ndarray] = None,
        test_norms: Optional[np.ndarray] = None,
    ) -> "QualityMeasureCalibrator":
        from sklearn.linear_model import LogisticRegression

        feats = self.build_features(
            scores, enroll_frames, test_frames, enroll_norms, test_norms
        )
        y = np.asarray(labels, dtype=np.int32)
        self._clf = LogisticRegression(
            C=self.c_reg, class_weight="balanced", max_iter=500
        )
        self._clf.fit(feats, y)
        return self

    def predict(
        self,
        scores: np.ndarray,
        enroll_frames: Optional[np.ndarray] = None,
        test_frames: Optional[np.ndarray] = None,
        enroll_norms: Optional[np.ndarray] = None,
        test_norms: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns calibrated log-likelihood-ratio scores."""
        if self._clf is None:
            raise RuntimeError("QualityMeasureCalibrator must be fit before predict().")
        feats = self.build_features(
            scores, enroll_frames, test_frames, enroll_norms, test_norms
        )
        return np.asarray(self._clf.decision_function(feats), dtype=np.float64)


def fuse_scores(
    score_arrays: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
    z_normalize: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """Convex linear fusion of complementary verification score arrays."""
    if not score_arrays:
        raise ValueError("score_arrays must contain at least one array.")
    arrs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in score_arrays]
    n = arrs[0].shape[0]
    for a in arrs[1:]:
        if a.shape[0] != n:
            raise ValueError("All score_arrays must have identical length.")
    if z_normalize:
        arrs = [(a - np.mean(a)) / max(float(np.std(a)), eps) for a in arrs]
    if weights is None:
        w = np.full((len(arrs),), 1.0 / len(arrs), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != len(arrs):
            raise ValueError("weights length must match score_arrays length.")
        w_sum = float(np.sum(w))
        if abs(w_sum) > eps:
            w = w / w_sum
    stacked = np.stack(arrs, axis=0)  # (M, N)
    return np.sum(w[:, None] * stacked, axis=0)
