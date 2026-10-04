"""Temporal frame aggregation and pooling layers for speaker recognition.

Includes:
- MeanPooling (`mean`)
- LastFramePooling (`last`)
- StatsPooling (`stats`: temporal mean + standard deviation)
- SelfAttentivePooling (`sap`)
- AttentiveStatsPooling (`asp`: channel- & context-dependent attentive mean + std)
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
            mask_f = mask[..., None].astype(x.dtype)
            denom = jnp.maximum(jnp.sum(mask_f, axis=1), 1.0)
            return jnp.sum(x * mask_f, axis=1) / denom
        return jnp.mean(x, axis=1)


class LastFramePooling(nn.Module):
    """Select the final frame `(B, T, D) -> (B, D)`."""

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        return x[:, -1, :]


class StatsPooling(nn.Module):
    """Temporal statistics pooling (mean + standard deviation) `(B, T, D) -> (B, 2*D)`."""

    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        if mask is not None:
            mask_f = mask[..., None].astype(x.dtype)
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
            mean_ctx = jnp.mean(x, axis=1, keepdims=True)
            std_ctx = jnp.sqrt(jnp.maximum(jnp.var(x, axis=1, keepdims=True), self.eps))
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


class CumulativeStatsPooling(nn.Module):
    """Cumulative statistics pooling across time `(B, T, D) -> (B, 2*D)`."""

    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        seq_len = x.shape[1]
        steps = jnp.arange(1, seq_len + 1, dtype=x.dtype)[None, :, None]
        cum_mean = jnp.cumsum(x, axis=1) / steps
        cum_sq_mean = jnp.cumsum(x ** 2, axis=1) / steps
        cum_var = jnp.maximum(cum_sq_mean - cum_mean ** 2, self.eps)
        cum_std = jnp.sqrt(cum_var)
        # Aggregate cumulative trajectory with mean + final state
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
    - `"shared_nonlinear"`: Multi-head tanh attention shared between keys & values
    - `"unshared_nonlinear"`: Separate multi-head tanh attention for keys and values
    - `"shared_linear"`: Multi-head linear attention shared between keys & values
    - `"maxpool"`: Temporal windowed max-pooling into M segments
    """

    num_keys: int = 4
    key_dim: int = 64
    value_dim: int = 64
    attention_dim: int = 128
    attention_type: str = "shared_nonlinear"
    normalize_keys_and_values: bool = True
    normalize_overall_output: bool = False
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        # Project frame features to key space and value space
        # x: (B, T, D)
        bsz, seq_len, _ = x.shape
        frame_keys = nn.Dense(
            self.num_keys * self.key_dim, name="pfas_key_proj"
        )(x).reshape(bsz, seq_len, self.num_keys, self.key_dim)
        frame_values = nn.Dense(
            self.num_keys * self.value_dim, name="pfas_val_proj"
        )(x).reshape(bsz, seq_len, self.num_keys, self.value_dim)

        att_mode = self.attention_type.lower()
        if att_mode == "maxpool":
            # Partition sequence into M segments and max-pool each segment
            pad_len = (self.num_keys - (seq_len % self.num_keys)) % self.num_keys
            if pad_len > 0:
                frame_keys = jnp.pad(frame_keys, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
                frame_values = jnp.pad(frame_values, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            seg_len = frame_keys.shape[1] // self.num_keys
            keys = jnp.max(
                frame_keys.reshape(bsz, self.num_keys, seg_len, self.num_keys, self.key_dim),
                axis=2,
            )
            # Take diagonal across (segment_idx, head_idx) -> (B, M, key_dim)
            keys = jnp.diagonal(keys, axis1=1, axis2=2).transpose(0, 2, 1)
            vals = jnp.max(
                frame_values.reshape(bsz, self.num_keys, seg_len, self.num_keys, self.value_dim),
                axis=2,
            )
            vals = jnp.diagonal(vals, axis1=1, axis2=2).transpose(0, 2, 1)
        elif att_mode == "unshared_nonlinear":
            hk = jnp.tanh(nn.Dense(self.attention_dim, name="pfas_att_k_hid")(x))
            logits_k = nn.Dense(self.num_keys, name="pfas_att_k_out")(hk)  # (B, T, M)
            hv = jnp.tanh(nn.Dense(self.attention_dim, name="pfas_att_v_hid")(x))
            logits_v = nn.Dense(self.num_keys, name="pfas_att_v_out")(hv)  # (B, T, M)
            if mask is not None:
                logits_k = jnp.where(mask[..., None] > 0, logits_k, -1e9)
                logits_v = jnp.where(mask[..., None] > 0, logits_v, -1e9)
            alpha_k = jax.nn.softmax(logits_k, axis=1)  # (B, T, M)
            alpha_v = jax.nn.softmax(logits_v, axis=1)  # (B, T, M)
            keys = jnp.einsum("btm,btmd->bmd", alpha_k, frame_keys)
            vals = jnp.einsum("btm,btmd->bmd", alpha_v, frame_values)
        elif att_mode == "shared_linear":
            logits = nn.Dense(self.num_keys, name="pfas_att_lin")(x)  # (B, T, M)
            if mask is not None:
                logits = jnp.where(mask[..., None] > 0, logits, -1e9)
            alpha = jax.nn.softmax(logits, axis=1)
            keys = jnp.einsum("btm,btmd->bmd", alpha, frame_keys)
            vals = jnp.einsum("btm,btmd->bmd", alpha, frame_values)
        else:
            # Default: shared_nonlinear
            h = jnp.tanh(nn.Dense(self.attention_dim, name="pfas_att_hid")(x))
            logits = nn.Dense(self.num_keys, name="pfas_att_out")(h)  # (B, T, M)
            if mask is not None:
                logits = jnp.where(mask[..., None] > 0, logits, -1e9)
            alpha = jax.nn.softmax(logits, axis=1)
            keys = jnp.einsum("btm,btmd->bmd", alpha, frame_keys)
            vals = jnp.einsum("btm,btmd->bmd", alpha, frame_values)

        if self.normalize_keys_and_values:
            keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + self.eps)
            vals = vals / (jnp.linalg.norm(vals, axis=-1, keepdims=True) + self.eps)

        flat_keys = keys.reshape(bsz, self.num_keys * self.key_dim)
        flat_vals = vals.reshape(bsz, self.num_keys * self.value_dim)
        packed = jnp.concatenate([flat_keys, flat_vals], axis=-1)

        if self.normalize_overall_output:
            packed = packed / (jnp.linalg.norm(packed, axis=-1, keepdims=True) + self.eps)

        return packed
