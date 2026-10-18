"""Temporal frame aggregation and pooling layers for speaker recognition.

Includes:
- MeanPooling (`mean`)
- LastFramePooling (`last`)
- StatsPooling (`stats`: temporal mean + standard deviation)
- SelfAttentivePooling (`sap`)
- AttentiveStatsPooling (`asp`: channel- & context-dependent attentive mean + std)
- AttentiveTemporalPooling (`atp`: single-head & multi-head sigmoid frame-weighted
  cumulative statistics pooling from https://arxiv.org/pdf/2202.12163)
- CumulativeStatsPooling (`cumulative_stats`)
- PfasRepresentationHead (`pfas`: multi-key/value utterance representation for
  Parameter-Free Attentive Scoring, https://arxiv.org/pdf/2203.05642v3)
"""

from typing import Optional
from flax import linen as nn
import jax
import jax.numpy as jnp


class MeanPooling(nn.Module):
    """Temporal mean pooling over frames `(B, T, D) -> (B, D)`."""

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        if mask is not None:
            mask_f = (mask > 0).astype(x.dtype)[..., None]
            denom = jnp.maximum(jnp.sum(mask_f, axis=1), 1.0)
            return jnp.sum(x * mask_f, axis=1) / denom
        return jnp.mean(x, axis=1)


class LastFramePooling(nn.Module):
    """Select the final valid frame `(B, T, D) -> (B, D)`."""

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        if mask is not None:
            valid_lengths = jnp.sum(mask > 0, axis=1).astype(jnp.int32)
            last_idx = jnp.clip(valid_lengths - 1, 0, x.shape[1] - 1)
            batch_idx = jnp.arange(x.shape[0])
            return x[batch_idx, last_idx, :]
        return x[:, -1, :]


class StatsPooling(nn.Module):
    """Temporal statistics pooling (mean + standard deviation) `(B, T, D) -> (B, 2*D)`."""

    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        if mask is not None:
            mask_f = (mask > 0).astype(x.dtype)[..., None]
            denom = jnp.maximum(jnp.sum(mask_f, axis=1), 1.0)
            mean = jnp.sum(x * mask_f, axis=1) / denom
            var = jnp.sum(((x - mean[:, None, :]) ** 2) * mask_f, axis=1) / denom
        else:
            mean = jnp.mean(x, axis=1)
            var = jnp.var(x, axis=1)
        std = jnp.sqrt(jnp.maximum(var, self.eps))
        return jnp.concatenate([mean, std], axis=-1)


class SelfAttentivePooling(nn.Module):
    """Self-Attentive Pooling (SAP) over temporal frames `(B, T, D) -> (B, D)`."""

    attention_dim: int = 128

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        # x: (B, T, D)
        h = jnp.tanh(nn.Dense(self.attention_dim, name="att_proj")(x))
        logits = nn.Dense(1, use_bias=False, name="att_score")(h)  # (B, T, 1)
        if mask is not None:
            logits = jnp.where(mask[..., None] > 0, logits, -1e9)
        weights = jax.nn.softmax(logits, axis=1)
        return jnp.sum(x * weights, axis=1)


class AttentiveStatsPooling(nn.Module):
    """Channel- and context-dependent Attentive Statistics Pooling (ASP).

    Follows Desplanques et al. (2020) ECAPA-TDNN: concatenates frame features
    with global temporal mean and standard deviation before computing channel-wise
    attention weights, then outputs weighted mean and weighted standard deviation
    `(B, T, D) -> (B, 2*D)`.
    """

    attention_dim: int = 128
    use_global_context: bool = True
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        # x: (B, T, D)
        bsz, seq_len, feat_dim = x.shape
        if self.use_global_context:
            if mask is not None:
                mask_f = (mask > 0).astype(x.dtype)[..., None]
                denom = jnp.maximum(jnp.sum(mask_f, axis=1, keepdims=True), 1.0)
                mean_ctx = jnp.sum(x * mask_f, axis=1, keepdims=True) / denom
                var_ctx = (
                    jnp.sum(((x - mean_ctx) ** 2) * mask_f, axis=1, keepdims=True)
                    / denom
                )
            else:
                mean_ctx = jnp.mean(x, axis=1, keepdims=True)
                var_ctx = jnp.var(x, axis=1, keepdims=True)
            std_ctx = jnp.sqrt(jnp.maximum(var_ctx, self.eps))
            mean_rep = jnp.broadcast_to(mean_ctx, (bsz, seq_len, feat_dim))
            std_rep = jnp.broadcast_to(std_ctx, (bsz, seq_len, feat_dim))
            att_in = jnp.concatenate([x, mean_rep, std_rep], axis=-1)
        else:
            att_in = x

        h = jnp.tanh(nn.Dense(self.attention_dim, name="asp_hidden")(att_in))
        logits = nn.Dense(feat_dim, name="asp_out")(h)  # (B, T, D)
        if mask is not None:
            logits = jnp.where(mask[..., None] > 0, logits, -1e9)
        alpha = jax.nn.softmax(logits, axis=1)  # (B, T, D)

        mean = jnp.sum(alpha * x, axis=1)  # (B, D)
        second_moment = jnp.sum(alpha * (x ** 2), axis=1)
        var = jnp.maximum(second_moment - (mean ** 2), self.eps)
        std = jnp.sqrt(var)
        return jnp.concatenate([mean, std], axis=-1)


class AttentiveTemporalPooling(nn.Module):
    """Attentive Temporal Pooling (ATP) for Conformer/Transformer Speaker Recognition.

    Implements single-head and multi-head Attentive Temporal Pooling as described in:
    "Attentive Temporal Pooling for Conformer-based Streaming Language Identification
    in Long-Form Speech" (Wang et al., Odyssey 2022, https://arxiv.org/pdf/2202.12163).

    For each frame $t$ and head $h \\in \\{1, \\dots, H\\}$:
    - Computes unnormalized positive importance weight via a feed-forward network
      with sigmoid activation:
      $$w_{t,h} = (1 - \\text{padding}_t) \\cdot \\sigma(\\text{FFN}(\\mathbf{x}_t))_h + \\epsilon$$
      with $\\epsilon = 10^{-4}$ for numerical stability.
    - Accumulates weighted zero-th, first, and second moments across all valid frames:
      $$\\eta_h = \\sum_t w_{t,h}, \\quad \\mu_h = \\eta_h^{-1}\\sum_t w_{t,h} x_{t,h}$$
      $$\\sigma_h = \\sqrt{\\max(\\eta_h^{-1}\\sum_t w_{t,h} x_{t,h}^2 - \\mu_h^2, \\epsilon)}$$
    - When `subspace_heads=True` (default) and `D % H == 0`, splits the $D$ feature
      channels across the $H$ heads ($\\mathbf{x}_{t,h} \\in \\mathbb{R}^{D/H}$) so each
      head attends to distinct temporal frames for its feature subspace while keeping
      the output dimension compact at `(B, 2 * D)`. When `subspace_heads=False`,
      pools all $D$ channels with each head to produce `(B, 2 * H * D)`.
    """

    num_heads: int = 4
    attention_dim: int = 128
    pooling_stats: str = "mean_std"
    subspace_heads: bool = True
    eps: float = 1e-4

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        # x: (B, T, D)
        bsz, seq_len, feat_dim = x.shape
        n_heads = max(1, int(self.num_heads))

        if self.attention_dim > 0:
            h = nn.tanh(nn.Dense(self.attention_dim, name="atp_hid")(x))
            logits = nn.Dense(n_heads, name="atp_ffn")(h)
        else:
            logits = nn.Dense(n_heads, name="atp_ffn")(x)

        raw_weights = jax.nn.sigmoid(logits)  # (B, T, H)
        if mask is not None:
            mask_f = (mask > 0).astype(x.dtype)[..., None]  # (B, T, 1)
            weights = (raw_weights + self.eps) * mask_f
        else:
            weights = raw_weights + self.eps

        eta = jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), self.eps)  # (B, 1, H)
        norm_weights = weights / eta  # (B, T, H)

        if self.subspace_heads and n_heads > 1 and (feat_dim % n_heads == 0):
            head_dim = feat_dim // n_heads
            x_heads = x.reshape(bsz, seq_len, n_heads, head_dim)
            w_heads = norm_weights[..., None]  # (B, T, H, 1)
            mean = jnp.sum(w_heads * x_heads, axis=1)  # (B, H, head_dim)
            second_moment = jnp.sum(w_heads * jnp.square(x_heads), axis=1)
            var = jnp.maximum(second_moment - jnp.square(mean), self.eps)
            std = jnp.sqrt(var)
            mean_flat = mean.reshape(bsz, feat_dim)
            if self.pooling_stats.lower() == "mean":
                return mean_flat
            std_flat = std.reshape(bsz, feat_dim)
            return jnp.concatenate([mean_flat, std_flat], axis=-1)

        # Weighted mean and second moment per head across full D: (B, H, D)
        mean = jnp.einsum("bth,btd->bhd", norm_weights, x)
        second_moment = jnp.einsum("bth,btd->bhd", norm_weights, jnp.square(x))
        var = jnp.maximum(second_moment - jnp.square(mean), self.eps)
        std = jnp.sqrt(var)

        mean_flat = mean.reshape(bsz, n_heads * feat_dim)
        if self.pooling_stats.lower() == "mean":
            return mean_flat
        std_flat = std.reshape(bsz, n_heads * feat_dim)
        return jnp.concatenate([mean_flat, std_flat], axis=-1)


class CumulativeStatsPooling(nn.Module):
    """Cumulative statistics pooling across time `(B, T, D) -> (B, 2*D)`."""

    eps: float = 1e-4

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        if mask is not None:
            mask_f = (mask > 0).astype(x.dtype)[..., None]
            weights = mask_f + self.eps * mask_f
            cum_eta = jnp.maximum(jnp.cumsum(weights, axis=1), self.eps)
            cum_mean = jnp.cumsum(x * weights, axis=1) / cum_eta
            cum_sq_mean = jnp.cumsum((x ** 2) * weights, axis=1) / cum_eta
            cum_var = jnp.maximum(cum_sq_mean - cum_mean ** 2, self.eps)
            cum_std = jnp.sqrt(cum_var)
            valid_lengths = jnp.sum(mask > 0, axis=1).astype(jnp.int32)
            last_idx = jnp.clip(valid_lengths - 1, 0, x.shape[1] - 1)
            batch_idx = jnp.arange(x.shape[0])
            return jnp.concatenate(
                [cum_mean[batch_idx, last_idx, :], cum_std[batch_idx, last_idx, :]],
                axis=-1,
            )
        seq_len = x.shape[1]
        steps = jnp.arange(1, seq_len + 1, dtype=x.dtype)[None, :, None]
        cum_mean = jnp.cumsum(x, axis=1) / steps
        cum_sq_mean = jnp.cumsum(x ** 2, axis=1) / steps
        cum_var = jnp.maximum(cum_sq_mean - cum_mean ** 2, self.eps)
        cum_std = jnp.sqrt(cum_var)
        return jnp.concatenate([cum_mean[:, -1, :], cum_std[:, -1, :]], axis=-1)


class PfasRepresentationHead(nn.Module):
    """Utterance Key-Value representation head for Parameter-Free Attentive Scoring.

    Implements the utterance summarization mechanisms from:
    "Parameter-Free Attentive Scoring for Speaker Verification"
    (Pelecanos et al., Odyssey 2022, https://arxiv.org/pdf/2203.05642v3).

    Produces `M = num_keys` key vectors of dimension `key_dim` and `M` value
    vectors of dimension `value_dim` per utterance, packed as a flat vector
    `[key_1, ..., key_M, value_1, ..., value_M]` of shape
    `(B, M * (key_dim + value_dim))`.

    Supported `attention_type` modes:
    - `"shared_nonlinear"` (Default): Combines a global `AttentiveTemporalPooling`
      (`MEAN_STD`) bottleneck value anchor with multi-head tanh frame attention
      shared between keys & values across temporal frames, producing diverse
      per-head phonetic/acoustic subspace summaries (`vals = global_val + 0.25 * local_vals`).
    - `"unshared_nonlinear"`: Separate multi-head tanh attention for keys and values
    - `"shared_linear"`: Multi-head linear attention shared between keys & values
    - `"maxpool"`: Temporal windowed max-pooling into M segments
    - `"atp_ffn"`: Pools frames with `AttentiveTemporalPooling` (`MEAN_STD`),
      passes through a nonlinear bottleneck FFN (`Dense(bottleneck_dim) -> LayerNorm -> ReLU`),
      and projects linearly to `M * (key_dim + value_dim)`.
    """

    num_keys: int = 4
    key_dim: int = 32
    value_dim: int = 128
    attention_dim: int = 128
    bottleneck_dim: int = 256
    atp_num_heads: int = 4
    attention_type: str = "shared_nonlinear"
    apply_l2_norm_to_keys: bool = True
    apply_l2_norm_to_values: bool = False
    apply_global_l2_norm_to_concat_form: bool = True
    use_global_anchor: bool = True
    local_value_scale: float = 0.25
    normalize_keys_and_values: Optional[bool] = None
    normalize_overall_output: bool = False
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        # x: (B, T, D)
        bsz, seq_len, _ = x.shape
        att_mode = self.attention_type.lower()
        total_kv_dim = self.num_keys * (self.key_dim + self.value_dim)

        if att_mode == "atp_ffn":
            pooled = AttentiveTemporalPooling(
                num_heads=self.atp_num_heads,
                pooling_stats="mean_std",
                name="pfas_atp",
            )(x, mask=mask)
            h = nn.LayerNorm(name="pfas_pool_ln")(pooled)
            h = nn.Dense(self.bottleneck_dim, name="pfas_ffn_hid")(h)
            h = nn.LayerNorm(name="pfas_ffn_ln")(h)
            h = nn.relu(h)
            packed = nn.Dense(total_kv_dim, name="pfas_kv_out")(h)
            key_size = self.num_keys * self.key_dim
            keys = packed[:, :key_size].reshape(bsz, self.num_keys, self.key_dim)
            vals = packed[:, key_size:].reshape(bsz, self.num_keys, self.value_dim)
        else:
            if self.use_global_anchor:
                pooled = AttentiveTemporalPooling(
                    num_heads=self.atp_num_heads,
                    attention_dim=self.attention_dim,
                    pooling_stats="mean_std",
                    name="pfas_global_atp",
                )(x, mask=mask)
                g = nn.LayerNorm(name="pfas_g_ln1")(pooled)
                g = nn.relu(
                    nn.LayerNorm(name="pfas_g_ln2")(
                        nn.Dense(self.bottleneck_dim, name="pfas_g_fc")(g)
                    )
                )
                global_val = nn.Dense(self.value_dim, name="pfas_g_val")(g)
            else:
                global_val = None

            frame_keys = nn.Dense(
                self.num_keys * self.key_dim, name="pfas_key_proj"
            )(x).reshape(bsz, seq_len, self.num_keys, self.key_dim)
            frame_values = nn.Dense(
                self.num_keys * self.value_dim, name="pfas_val_proj"
            )(x).reshape(bsz, seq_len, self.num_keys, self.value_dim)

            if att_mode == "maxpool":
                if mask is not None:
                    valid_mask = mask > 0
                    valid_len = jnp.maximum(
                        jnp.sum(valid_mask.astype(jnp.float32), axis=1, keepdims=True),
                        1.0,
                    )
                else:
                    valid_mask = jnp.ones((bsz, seq_len), dtype=jnp.bool_)
                    valid_len = jnp.full((bsz, 1), float(seq_len), dtype=jnp.float32)

                t_idx = jnp.arange(seq_len, dtype=jnp.float32)[None, :]
                k_segs = []
                v_segs = []
                for idx in range(self.num_keys):
                    seg_start = jnp.floor(idx * valid_len / float(self.num_keys))
                    if idx == self.num_keys - 1:
                        seg_end = valid_len
                    else:
                        seg_end = jnp.maximum(
                            seg_start + 1.0,
                            jnp.floor((idx + 1) * valid_len / float(self.num_keys)),
                        )
                    in_seg = (t_idx >= seg_start) & (t_idx < seg_end) & valid_mask
                    in_seg = jnp.where(
                        jnp.any(in_seg, axis=1, keepdims=True), in_seg, valid_mask
                    )
                    k_m = jnp.where(
                        in_seg[:, :, None], frame_keys[:, :, idx, :], -1e9
                    )
                    v_m = jnp.where(
                        in_seg[:, :, None], frame_values[:, :, idx, :], -1e9
                    )
                    k_segs.append(jnp.max(k_m, axis=1))
                    v_segs.append(jnp.max(v_m, axis=1))
                keys = jnp.stack(k_segs, axis=1)
                local_vals = jnp.stack(v_segs, axis=1)
            elif att_mode == "unshared_nonlinear":
                hk = jnp.tanh(nn.Dense(self.attention_dim, name="pfas_att_k_hid")(x))
                logits_k = nn.Dense(self.num_keys, name="pfas_att_k_out")(hk)
                hv = jnp.tanh(nn.Dense(self.attention_dim, name="pfas_att_v_hid")(x))
                logits_v = nn.Dense(self.num_keys, name="pfas_att_v_out")(hv)
                if mask is not None:
                    logits_k = jnp.where(mask[..., None] > 0, logits_k, -1e9)
                    logits_v = jnp.where(mask[..., None] > 0, logits_v, -1e9)
                alpha_k = jax.nn.softmax(logits_k, axis=1)
                alpha_v = jax.nn.softmax(logits_v, axis=1)
                keys = jnp.einsum("btm,btmd->bmd", alpha_k, frame_keys)
                local_vals = jnp.einsum("btm,btmd->bmd", alpha_v, frame_values)
            elif att_mode == "shared_linear":
                logits = nn.Dense(self.num_keys, name="pfas_att_lin")(x)
                if mask is not None:
                    logits = jnp.where(mask[..., None] > 0, logits, -1e9)
                alpha = jax.nn.softmax(logits, axis=1)
                keys = jnp.einsum("btm,btmd->bmd", alpha, frame_keys)
                local_vals = jnp.einsum("btm,btmd->bmd", alpha, frame_values)
            else:
                h = jnp.tanh(nn.Dense(self.attention_dim, name="pfas_att_hid")(x))
                logits = nn.Dense(self.num_keys, name="pfas_att_out")(h)
                if mask is not None:
                    logits = jnp.where(mask[..., None] > 0, logits, -1e9)
                alpha = jax.nn.softmax(logits, axis=1)
                keys = jnp.einsum("btm,btmd->bmd", alpha, frame_keys)
                local_vals = jnp.einsum("btm,btmd->bmd", alpha, frame_values)

            if global_val is not None:
                vals = global_val[:, None, :] + self.local_value_scale * local_vals
            else:
                vals = local_vals

        norm_k = bool(self.apply_l2_norm_to_keys or self.normalize_keys_and_values)
        norm_v = bool(self.apply_l2_norm_to_values or self.normalize_keys_and_values)
        if norm_k:
            keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + self.eps)
        if norm_v:
            vals = vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + self.eps)

        flat_keys = keys.reshape(bsz, self.num_keys * self.key_dim)
        flat_vals = vals.reshape(bsz, self.num_keys * self.value_dim)
        if self.apply_global_l2_norm_to_concat_form and not norm_v:
            flat_vals = flat_vals / (
                jnp.linalg.norm(flat_vals, axis=-1, keepdims=True) + self.eps
            )
        packed = jnp.concatenate([flat_keys, flat_vals], axis=-1)

        if self.normalize_overall_output:
            packed = packed / (
                jnp.linalg.norm(packed, axis=-1, keepdims=True) + self.eps
            )

        return packed
