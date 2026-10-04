"""Loss functions for speaker verification and identification in JAX/Flax.

Implements:
1. Generalized End-to-End (GE2E) Loss (Wan et al., ICASSP 2018,
   https://arxiv.org/pdf/1710.10467):
   - `ge2e_softmax_loss` (both leave-one-out and split-batch centroids)
   - `ge2e_contrast_loss`
2. Extended-Set (Multi-Row) Softmax Loss from Dr-Vectors (Pelecanos et al.,
   Interspeech 2021, https://arxiv.org/pdf/2104.01989):
   - `extended_set_softmax_loss`
3. Pairwise Balanced Logistic Loss:
   - `pairwise_loss`
4. Angular & Additive Margin Classification Losses:
   - `arcface_loss` (AAM-Softmax, Deng et al., CVPR 2019)
   - `cosface_loss` (AM-Softmax, Wang et al., CVPR 2018)
   - `sphereface_loss` (A-Softmax, Liu et al., CVPR 2017)
   - `softmax_loss`
5. Triplet Loss (FaceNet / bredin2017):
   - `triplet_loss` & `triplet_loss_from_batch`
"""

from functools import partial
from typing import Callable, Optional
import jax
import jax.numpy as jnp

from flaxspeaker import scoring


@jax.jit
def get_triplet_loss(
    anchor: jax.Array,
    pos: jax.Array,
    neg: jax.Array,
    alpha: float,
) -> jax.Array:
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""
    return jnp.maximum(
        jax.vmap(scoring.cosine_similarity, in_axes=[0, 0])(anchor, neg)
        - jax.vmap(scoring.cosine_similarity, in_axes=[0, 0])(anchor, pos)
        + alpha,
        0.0,
    )


@partial(jax.jit, static_argnames=["batch_size", "triplet_alpha"])
def get_triplet_loss_from_batch_output(
    batch_output: jax.Array,
    batch_size: int,
    triplet_alpha: float,
) -> jax.Array:
    """Triplet loss from N*(a|p|n) batch output."""
    batch_output_reshaped = jnp.reshape(
        batch_output, (batch_size, 3, batch_output.shape[1])
    )
    batch_loss = get_triplet_loss(
        batch_output_reshaped[:, 0, :],
        batch_output_reshaped[:, 1, :],
        batch_output_reshaped[:, 2, :],
        triplet_alpha,
    )
    return jnp.mean(batch_loss)


def compute_ge2e_similarity_matrix_leave_one_out(
    embeddings: jax.Array,
    num_speakers: int,
    num_utts_per_speaker: int,
    w: jax.Array,
    b: jax.Array,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """Compute exact Leave-One-Out GE2E similarity matrix (Wan et al., 2018 Eq. 1 & 4).

    Args:
        embeddings: `(N * M, D)` speaker embeddings ordered as `N` speakers
            with `M` utterances each.
        num_speakers: `N` speakers in batch.
        num_utts_per_speaker: `M` utterances per speaker (`M >= 2`).
        w: Positive scale parameter (`w > 0`).
        b: Bias parameter.

    Returns:
        `(sim_matrix, labels)` where `sim_matrix` has shape `(N * M, N)` and
        `labels` has shape `(N * M,)` with values in `0..N-1`.
    """
    dim = embeddings.shape[-1]
    norm_embs = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + eps)
    # Shape: (N, M, D)
    emb_grid = norm_embs.reshape(num_speakers, num_utts_per_speaker, dim)

    # Inclusive centroids: (N, D)
    sum_all = jnp.sum(emb_grid, axis=1)  # (N, D)
    centroids_incl = sum_all / float(num_utts_per_speaker)
    centroids_incl = centroids_incl / (
        jnp.linalg.norm(centroids_incl, axis=-1, keepdims=True) + eps
    )

    # Exclusive centroids: c_j^{(-i)} = (sum_m e_jm - e_ji) / (M - 1) -> (N, M, D)
    centroids_excl = (sum_all[:, None, :] - emb_grid) / float(
        max(1, num_utts_per_speaker - 1)
    )
    centroids_excl = centroids_excl / (
        jnp.linalg.norm(centroids_excl, axis=-1, keepdims=True) + eps
    )

    # Cosine similarity of all (N, M) embeddings against all N inclusive centroids: (N, M, N)
    sim_incl = jnp.einsum("jid,kd->jik", emb_grid, centroids_incl)
    # Cosine similarity of (j, i) against its own exclusive centroid c_j^{(-i)}: (N, M)
    sim_excl_diag = jnp.sum(emb_grid * centroids_excl, axis=-1)

    # Replace diagonal k == j with exclusive centroid similarity
    eye_mask = jnp.eye(num_speakers, dtype=embeddings.dtype)[:, None, :]  # (N, 1, N)
    sim_combined = sim_incl * (1.0 - eye_mask) + sim_excl_diag[:, :, None] * eye_mask

    w_pos = jnp.abs(w) + eps
    sim_matrix = w_pos * sim_combined.reshape(num_speakers * num_utts_per_speaker, num_speakers) + b
    labels = jnp.repeat(jnp.arange(num_speakers, dtype=jnp.int32), num_utts_per_speaker)
    return sim_matrix, labels


def compute_split_batch_similarity_matrix(
    embeddings: jax.Array,
    num_speakers: int,
    num_utts_per_speaker: int,
    w: jax.Array,
    b: jax.Array,
    score_matrix_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    is_pfas: bool = False,
    pfas_num_keys: int = 4,
    pfas_key_dim: int = 64,
    pfas_value_dim: int = 64,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, int]:
    """Compute Split-Batch Test-vs-Enrollment similarity matrix.

    Splits the `M` utterances per speaker into `M_enroll = M // 2` enrollment
    utterances and `M_test = M - M_enroll` test utterances, arranged in `M_test`
    blocks of `N` speakers each `(M_test * N, N)`.
    Compatible with Cosine, PFAS, and Decision Residual Network (Dr-Vectors) scoring.

    Returns:
        `(sim_matrix, labels, num_test_utts)` where `sim_matrix` has shape
        `(M_test * N, N)` (each block of `N` rows has diagonal targets `0..N-1`).
    """
    dim = embeddings.shape[-1]
    emb_grid = embeddings.reshape(num_speakers, num_utts_per_speaker, dim)
    m_enroll = max(1, num_utts_per_speaker // 2)
    m_test = num_utts_per_speaker - m_enroll

    enroll_utts = emb_grid[:, :m_enroll, :]  # (N, M_enroll, D)
    test_utts = emb_grid[:, m_enroll:, :]  # (N, M_test, D)

    if is_pfas:
        # Stack keys and values across M_enroll utterances for each speaker
        key_total = pfas_num_keys * pfas_key_dim
        e_keys = enroll_utts[:, :, :key_total].reshape(
            num_speakers, m_enroll * pfas_num_keys, pfas_key_dim
        )
        e_vals = enroll_utts[:, :, key_total:].reshape(
            num_speakers, m_enroll * pfas_num_keys, pfas_value_dim
        )
        enroll_repr = jnp.concatenate(
            [
                e_keys.reshape(num_speakers, -1),
                e_vals.reshape(num_speakers, -1),
            ],
            axis=-1,
        )
    else:
        enroll_repr = jnp.mean(enroll_utts, axis=1)  # (N, D)
        enroll_repr = enroll_repr / (
            jnp.linalg.norm(enroll_repr, axis=-1, keepdims=True) + eps
        )

    # Transpose test utterances to (M_test, N, D) -> (M_test * N, D) so each N-row
    # block corresponds to one test utterance per speaker (diagonal targets!)
    test_blocks = jnp.transpose(test_utts, (1, 0, 2)).reshape(m_test * num_speakers, dim)

    if score_matrix_fn is not None:
        raw_sim = score_matrix_fn(test_blocks, enroll_repr)
    else:
        raw_sim = scoring.cosine_similarity_matrix(test_blocks, enroll_repr, eps=eps)

    w_pos = jnp.abs(w) + eps
    sim_matrix = w_pos * raw_sim + b
    labels = jnp.tile(jnp.arange(num_speakers, dtype=jnp.int32), m_test)
    return sim_matrix, labels, m_test


def ge2e_softmax_loss(
    embeddings: jax.Array,
    num_speakers: int,
    num_utts_per_speaker: int,
    w: jax.Array,
    b: jax.Array,
    split_batch: bool = False,
    score_matrix_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    is_pfas: bool = False,
    pfas_num_keys: int = 4,
    pfas_key_dim: int = 64,
    pfas_value_dim: int = 64,
) -> jax.Array:
    """Generalized End-to-End (GE2E) Softmax Loss (Wan et al., ICASSP 2018, Eq. 6)."""
    if split_batch or score_matrix_fn is not None or is_pfas:
        sim_matrix, labels, _ = compute_split_batch_similarity_matrix(
            embeddings,
            num_speakers,
            num_utts_per_speaker,
            w,
            b,
            score_matrix_fn=score_matrix_fn,
            is_pfas=is_pfas,
            pfas_num_keys=pfas_num_keys,
            pfas_key_dim=pfas_key_dim,
            pfas_value_dim=pfas_value_dim,
        )
    else:
        sim_matrix, labels = compute_ge2e_similarity_matrix_leave_one_out(
            embeddings, num_speakers, num_utts_per_speaker, w, b
        )

    log_probs = jax.nn.log_softmax(sim_matrix, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, labels[:, None], axis=-1).squeeze(-1)
    return -jnp.mean(target_log_probs)


def ge2e_contrast_loss(
    embeddings: jax.Array,
    num_speakers: int,
    num_utts_per_speaker: int,
    w: jax.Array,
    b: jax.Array,
    split_batch: bool = False,
    score_matrix_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
) -> jax.Array:
    """Generalized End-to-End (GE2E) Contrast Loss (Wan et al., ICASSP 2018, Eq. 7).

    `L_C(e_ji) = 1 - sigmoid(S_{ji, j}) + max_{k != j} sigmoid(S_{ji, k})`
    """
    w_clamped = jnp.maximum(jnp.abs(w), 5.0)
    if split_batch or score_matrix_fn is not None:
        sim_matrix, labels, _ = compute_split_batch_similarity_matrix(
            embeddings,
            num_speakers,
            num_utts_per_speaker,
            w_clamped,
            b,
            score_matrix_fn=score_matrix_fn,
        )
    else:
        sim_matrix, labels = compute_ge2e_similarity_matrix_leave_one_out(
            embeddings, num_speakers, num_utts_per_speaker, w_clamped, b
        )

    sig_scores = jax.nn.sigmoid(sim_matrix)
    target_mask = jax.nn.one_hot(labels, num_speakers, dtype=sim_matrix.dtype)
    pos_sig = jnp.sum(sig_scores * target_mask, axis=-1)
    neg_sig = jnp.max(sig_scores * (1.0 - target_mask) - 1e9 * target_mask, axis=-1)
    return jnp.mean(1.0 - pos_sig + neg_sig)


def extended_set_softmax_loss(
    embeddings: jax.Array,
    num_speakers: int,
    num_utts_per_speaker: int,
    w: jax.Array,
    b: jax.Array,
    multi_row_window: int = 0,
    score_matrix_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    is_pfas: bool = False,
    pfas_num_keys: int = 4,
    pfas_key_dim: int = 64,
    pfas_value_dim: int = 64,
) -> jax.Array:
    """Extended-Set (Multi-Row) Softmax Loss from Dr-Vectors (Pelecanos et al., 2021).

    Reference:
    "Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker
    Recognition" (Interspeech 2021, https://arxiv.org/pdf/2104.01989).

    In each `N x N` test-vs-enrollment score block `S`, each target score `S_{j,j}`
    is normalized against ALL off-diagonal non-target scores across `multi_row_window`
    rows (defaults to all `N` rows in the `N x N` block, i.e., `N*(N-1)` non-targets):
    `L(j) = -log( exp(S_{j,j}) / (exp(S_{j,j}) + sum_{p in W, q != p} exp(S_{p,q})) )`
    """
    sim_matrix, _, m_test = compute_split_batch_similarity_matrix(
        embeddings,
        num_speakers,
        num_utts_per_speaker,
        w,
        b,
        score_matrix_fn=score_matrix_fn,
        is_pfas=is_pfas,
        pfas_num_keys=pfas_num_keys,
        pfas_key_dim=pfas_key_dim,
        pfas_value_dim=pfas_value_dim,
    )
    # Reshape into M_test blocks of (N, N)
    blocks = sim_matrix.reshape(m_test, num_speakers, num_speakers)
    eye = jnp.eye(num_speakers, dtype=jnp.bool_)
    off_diag_mask = ~eye  # (N, N)

    window = num_speakers if (multi_row_window <= 0 or multi_row_window > num_speakers) else multi_row_window

    def _block_loss(block: jax.Array) -> jax.Array:
        # block: (N, N)
        diag_targets = jnp.diag(block)  # (N,)
        if window == num_speakers:
            # All N*(N-1) off-diagonal entries in the block share the same logsumexp
            off_diag_logits = jnp.where(off_diag_mask, block, -1e9)
            log_sum_non_target = jax.nn.logsumexp(off_diag_logits)
            return jnp.mean(jax.nn.softplus(log_sum_non_target - diag_targets))
        else:
            # Group rows into windows of size `window`
            losses = []
            for start in range(0, num_speakers, window):
                end = min(num_speakers, start + window)
                sub_block = block[start:end, :]
                sub_mask = off_diag_mask[start:end, :]
                sub_off_logits = jnp.where(sub_mask, sub_block, -1e9)
                log_sum_neg = jax.nn.logsumexp(sub_off_logits)
                sub_targets = diag_targets[start:end]
                losses.append(jnp.mean(jax.nn.softplus(log_sum_neg - sub_targets)))
            return jnp.mean(jnp.stack(losses))

    block_losses = jax.vmap(_block_loss)(blocks)
    return jnp.mean(block_losses)


def pairwise_loss(
    embeddings: jax.Array,
    num_speakers: int,
    num_utts_per_speaker: int,
    w: jax.Array,
    b: jax.Array,
    score_matrix_fn: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    is_pfas: bool = False,
    pfas_num_keys: int = 4,
    pfas_key_dim: int = 64,
    pfas_value_dim: int = 64,
) -> jax.Array:
    """Balanced pairwise binary cross-entropy loss over test-vs-enrollment score matrix."""
    sim_matrix, labels, _ = compute_split_batch_similarity_matrix(
        embeddings,
        num_speakers,
        num_utts_per_speaker,
        w,
        b,
        score_matrix_fn=score_matrix_fn,
        is_pfas=is_pfas,
        pfas_num_keys=pfas_num_keys,
        pfas_key_dim=pfas_key_dim,
        pfas_value_dim=pfas_value_dim,
    )
    target_mask = jax.nn.one_hot(labels, num_speakers, dtype=sim_matrix.dtype)
    # Binary cross-entropy with logits: max(x, 0) - x*y + log(1 + exp(-|x|))
    bce = (
        jnp.maximum(sim_matrix, 0.0)
        - sim_matrix * target_mask
        + jnp.log1p(jnp.exp(-jnp.abs(sim_matrix)))
    )
    n_pos = jnp.maximum(jnp.sum(target_mask), 1.0)
    n_neg = jnp.maximum(jnp.sum(1.0 - target_mask), 1.0)
    weights = 0.5 * target_mask / n_pos + 0.5 * (1.0 - target_mask) / n_neg
    return jnp.sum(bce * weights)


def arcface_loss(
    embeddings: jax.Array,
    class_weights: jax.Array,
    labels: jax.Array,
    scale: float = 30.0,
    margin: float = 0.2,
    eps: float = 1e-6,
) -> jax.Array:
    """ArcFace / Additive Angular Margin Softmax (AAM-Softmax) Loss.

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
    Recognition", CVPR 2019.

    `L = -log( exp(s * cos(theta_y + m)) / (exp(s * cos(theta_y + m)) + sum_{c != y} exp(s * cos(theta_c))) )`
    """
    x_norm = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + eps)
    w_norm = class_weights / (jnp.linalg.norm(class_weights, axis=0, keepdims=True) + eps)

    cosine = jnp.clip(jnp.matmul(x_norm, w_norm), -1.0 + eps, 1.0 - eps)
    sine = jnp.sqrt(jnp.maximum(1.0 - cosine ** 2, eps))

    cos_m = jnp.cos(margin)
    sin_m = jnp.sin(margin)
    # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
    phi = cosine * cos_m - sine * sin_m

    # Monotonicity guard when theta + m > pi
    threshold = jnp.cos(jnp.pi - margin)
    mm = jnp.sin(jnp.pi - margin) * margin
    phi = jnp.where(cosine > threshold, phi, cosine - mm)

    one_hot = jax.nn.one_hot(labels, class_weights.shape[1], dtype=embeddings.dtype)
    logits = scale * (one_hot * phi + (1.0 - one_hot) * cosine)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def cosface_loss(
    embeddings: jax.Array,
    class_weights: jax.Array,
    labels: jax.Array,
    scale: float = 30.0,
    margin: float = 0.2,
    eps: float = 1e-6,
) -> jax.Array:
    """CosFace / Large Margin Cosine (AM-Softmax) Loss.

    Reference: Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face
    Recognition", CVPR 2018.

    `L = -log( exp(s * (cos(theta_y) - m)) / (exp(s * (cos(theta_y) - m)) + sum_{c != y} exp(s * cos(theta_c))) )`
    """
    x_norm = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + eps)
    w_norm = class_weights / (jnp.linalg.norm(class_weights, axis=0, keepdims=True) + eps)

    cosine = jnp.clip(jnp.matmul(x_norm, w_norm), -1.0 + eps, 1.0 - eps)
    phi = cosine - margin

    one_hot = jax.nn.one_hot(labels, class_weights.shape[1], dtype=embeddings.dtype)
    logits = scale * (one_hot * phi + (1.0 - one_hot) * cosine)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def sphereface_loss(
    embeddings: jax.Array,
    class_weights: jax.Array,
    labels: jax.Array,
    scale: float = 30.0,
    margin: float = 1.35,
    anneal_lambda: float = 5.0,
    eps: float = 1e-6,
) -> jax.Array:
    """SphereFace / Multiplicative Angular Margin (A-Softmax) Loss.

    Reference: Liu et al., "SphereFace: Deep Hypersphere Embedding for Face
    Recognition", CVPR 2017.

    Supports both real-valued multiplicative angular margin `m >= 1.0` with
    generalized monotonic extension `psi(theta) = (-1)^k cos(m * theta) - 2*k`
    for `theta in [k*pi/m, (k+1)*pi/m]`, stabilized with annealing factor `anneal_lambda`.
    """
    x_norm = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + eps)
    w_norm = class_weights / (jnp.linalg.norm(class_weights, axis=0, keepdims=True) + eps)

    cosine = jnp.clip(jnp.matmul(x_norm, w_norm), -1.0 + eps, 1.0 - eps)
    theta = jnp.arccos(cosine)
    m_val = jnp.maximum(margin, 1.0)
    m_theta = m_val * theta
    k = jnp.floor(m_theta / jnp.pi)
    sign = jnp.where((k.astype(jnp.int32) % 2) == 0, 1.0, -1.0)
    psi = sign * jnp.cos(m_theta) - 2.0 * k

    # Stabilized combination: (anneal_lambda * cos(theta) + psi(theta)) / (1 + anneal_lambda)
    phi = (anneal_lambda * cosine + psi) / (1.0 + anneal_lambda)

    one_hot = jax.nn.one_hot(labels, class_weights.shape[1], dtype=embeddings.dtype)
    logits = scale * (one_hot * phi + (1.0 - one_hot) * cosine)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def softmax_loss(
    embeddings: jax.Array,
    class_weights: jax.Array,
    labels: jax.Array,
    scale: float = 16.0,
    eps: float = 1e-6,
) -> jax.Array:
    """Normalized Cosine Softmax classification loss."""
    x_norm = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + eps)
    w_norm = class_weights / (jnp.linalg.norm(class_weights, axis=0, keepdims=True) + eps)
    logits = scale * jnp.matmul(x_norm, w_norm)
    one_hot = jax.nn.one_hot(labels, class_weights.shape[1], dtype=embeddings.dtype)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
