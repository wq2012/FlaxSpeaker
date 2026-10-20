"""Neural network backbones and unified speaker encoder in Flax Linen.

Supported backbones:
- `LstmBackbone`: Multi-layer uni/bidirectional LSTM with optional projections
- `TransformerBackbone`: Pre-LN Transformer encoder with local conv stem and mask support
- `ConformerBackbone`: Macaron-style Conformer (FFN + MHSA + Depthwise Conv + FFN; Gulati et al. 2020, arXiv:2005.08100)
- `MambaBackbone`: Selective State Space Model (SSM) using `jax.lax.associative_scan`
- `EcapaTdnnBackbone`: Emphasized Channel Attention, Propagation and Aggregation TDNN
  (SE-Res2Net blocks + Multi-layer Feature Aggregation)
- `ResNetBackbone`: 2D Time-Frequency SE-ResNet (WeSpeaker / Thin-ResNet style)
"""

from typing import Any, Optional, Sequence
from flax import linen as nn
import jax
import jax.numpy as jnp

from flaxspeaker import pooling


def sinusoidal_positional_encoding(
    seq_len: int, dim: int, dtype: Any = jnp.float32
) -> jax.Array:
    """Create sinusoidal positional encodings of shape `(1, seq_len, dim)`."""
    position = jnp.arange(seq_len, dtype=dtype)[:, None]
    div_term = jnp.exp(
        jnp.arange(0, dim, 2, dtype=dtype) * (-jnp.log(10000.0) / max(dim, 2))
    )
    pe_even = jnp.sin(position * div_term)
    pe_odd = jnp.cos(position * div_term)
    pe = jnp.zeros((seq_len, dim), dtype=dtype)
    pe = pe.at[:, 0::2].set(pe_even)
    pe = pe.at[:, 1::2].set(pe_odd[:, : pe[:, 1::2].shape[1]])
    return pe[None, :, :]


def _pad_time_to_multiple(
    x: jax.Array, factor: int, mask: Optional[jax.Array]
) -> tuple[jax.Array, Optional[jax.Array]]:
    """Right-pads the time axis of `x` (B, T, ...) and `mask` (B, T) to a multiple of `factor`."""
    seq_len = x.shape[1]
    pad = (-seq_len) % factor
    if pad == 0:
        return x, mask
    x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
    if mask is not None:
        mask = jnp.pad(mask, [(0, 0), (0, pad)])
    return x, mask


def subsample_mask(mask: Optional[jax.Array], factor: int) -> Optional[jax.Array]:
    """Time-subsamples a binary mask (B, T) by `factor`: a group is valid if any frame is."""
    if mask is None or factor <= 1:
        return mask
    m, _ = _pad_time_to_multiple(mask[..., None], factor, None)
    m = m[..., 0]
    bsz, seq_len = m.shape
    return jnp.max(
        m.reshape(bsz, seq_len // factor, factor), axis=-1
    ).astype(mask.dtype)


def _stack_frames(
    x: jax.Array, factor: int, mask: Optional[jax.Array]
) -> tuple[jax.Array, Optional[jax.Array]]:
    """Stack-by-`factor` time reduction: (B, T, D) -> (B, ceil(T/factor), factor*D)."""
    x, mask = _pad_time_to_multiple(x, factor, mask)
    bsz, seq_len, dim = x.shape
    stacked = x.reshape(bsz, seq_len // factor, factor * dim)
    return stacked, subsample_mask(mask, factor) if mask is not None else None


def _average_frames(x: jax.Array, factor: int) -> jax.Array:
    """Parameter-free time reduction by averaging each group of `factor` frames."""
    x, _ = _pad_time_to_multiple(x, factor, None)
    bsz, seq_len, dim = x.shape
    return jnp.mean(x.reshape(bsz, seq_len // factor, factor, dim), axis=2)


class LstmBackbone(nn.Module):
    """Multi-layer LSTM backbone with optional projection and bidirectionality."""

    hidden_size: int = 256
    num_layers: int = 3
    bidirectional: bool = False
    proj_size: int = 0

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        # x: (B, T, F)
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        # Valid-frame lengths are required so that the *backward* recurrence
        # starts at the last valid frame instead of the zero-padded tail (which
        # would leak padding into every valid frame of a bidirectional LSTM).
        seq_lengths = (
            jnp.sum((mask > 0).astype(jnp.int32), axis=1)
            if mask is not None
            else None
        )
        h = x * mask_f if mask_f is not None else x
        for i in range(self.num_layers):
            fwd_rnn = nn.RNN(
                nn.OptimizedLSTMCell(features=self.hidden_size),
                name=f"lstm_fwd_{i}",
            )
            y_fwd = fwd_rnn(h, seq_lengths=seq_lengths)
            if self.bidirectional:
                # `keep_order=True` flips the backward outputs back into forward
                # time order so they are frame-aligned with `y_fwd`; Flax's default
                # (`keep_order=False`) would return them time-reversed.
                bwd_rnn = nn.RNN(
                    nn.OptimizedLSTMCell(features=self.hidden_size),
                    reverse=True,
                    keep_order=True,
                    name=f"lstm_bwd_{i}",
                )
                y_bwd = bwd_rnn(h, seq_lengths=seq_lengths)
                y = jnp.concatenate([y_fwd, y_bwd], axis=-1)
            else:
                y = y_fwd

            if self.proj_size > 0:
                y = nn.Dense(self.proj_size, name=f"lstm_proj_{i}")(y)

            if h.shape[-1] == y.shape[-1] and i > 0:
                h = nn.LayerNorm(name=f"lstm_ln_{i}")(h + y)
            else:
                h = nn.LayerNorm(name=f"lstm_ln_{i}")(y)
            if mask_f is not None:
                h = h * mask_f
        return h


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block."""

    dim: int = 256
    num_heads: int = 4
    mlp_dim: int = 512

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        attn_mask = None
        if mask is not None:
            attn_mask = nn.make_attention_mask(
                jnp.ones_like(mask, dtype=jnp.bool_),
                mask > 0,
                dtype=x.dtype,
            )
        h = nn.LayerNorm(name="ln1")(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            name="mhsa",
        )(h, h, mask=attn_mask)
        x = x + h
        if mask_f is not None:
            x = x * mask_f

        h = nn.LayerNorm(name="ln2")(x)
        h = nn.Dense(self.mlp_dim, name="ffn1")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.dim, name="ffn2")(h)
        x = x + h
        if mask_f is not None:
            x = x * mask_f
        return x


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone for speaker recognition."""

    dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    mlp_dim: int = 512
    use_positional_encoding: bool = False
    use_conv_stem: bool = True

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        if mask_f is not None:
            x = x * mask_f
        h = nn.Dense(self.dim, name="input_proj")(x)
        if self.use_conv_stem:
            c = nn.Conv(
                features=self.dim,
                kernel_size=(3,),
                padding="SAME",
                name="stem_conv",
            )(x)
            c = nn.silu(nn.LayerNorm(name="stem_ln")(c))
            h = h + c
        if self.use_positional_encoding:
            pe = sinusoidal_positional_encoding(h.shape[1], self.dim, h.dtype)
            h = h + pe
        if mask_f is not None:
            h = h * mask_f
        layer_outs = []
        for i in range(self.num_layers):
            h = TransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                name=f"block_{i}",
            )(h, deterministic=deterministic, mask=mask)
            layer_outs.append(h)
        cat = jnp.concatenate(layer_outs, axis=-1)
        out = nn.Dense(self.dim, name="mfa_proj")(nn.LayerNorm(name="final_ln")(cat))
        if mask_f is not None:
            out = out * mask_f
        return out


class ConformerConvModule(nn.Module):
    """Conformer depthwise separable convolution module with GLU gating and Mask-Aware GroupNorm.

    Uses a mask-aware GroupNorm with 2 groups in place of the BatchNorm of Gulati et al. 2020
    (FlaxSpeaker design choice; not taken from any of the cited speaker papers).
    """

    dim: int = 256
    kernel_size: int = 15
    use_group_norm: bool = True

    @nn.compact
    def __call__(
        self, x: jax.Array, mask: Optional[jax.Array] = None
    ) -> jax.Array:
        bsz, seq_len, _ = x.shape
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        h = nn.LayerNorm(name="ln")(x)
        h = nn.Dense(2 * self.dim, name="pw_conv1")(h)
        # Gated Linear Unit (GLU)
        h_a, h_b = jnp.split(h, 2, axis=-1)
        h = h_a * jax.nn.sigmoid(h_b)
        if mask_f is not None:
            h = h * mask_f
        # 1D Depthwise Convolution
        h = nn.Conv(
            features=self.dim,
            kernel_size=(self.kernel_size,),
            feature_group_count=self.dim,
            padding="SAME",
            name="dw_conv",
        )(h)
        if mask_f is not None:
            h = h * mask_f
        if self.use_group_norm and (self.dim % 2 == 0):
            num_groups = 2
            group_dim = self.dim // num_groups
            h_g = h.reshape(bsz, seq_len, num_groups, group_dim)
            if mask_f is not None:
                m_g = mask_f[..., None]  # (B, T, 1, 1)
                denom = jnp.maximum(
                    jnp.sum(m_g, axis=(1, 3), keepdims=True) * float(group_dim),
                    1.0,
                )
                mean = jnp.sum(h_g * m_g, axis=(1, 3), keepdims=True) / denom
                var = (
                    jnp.sum(jnp.square(h_g - mean) * m_g, axis=(1, 3), keepdims=True)
                    / denom
                )
            else:
                mean = jnp.mean(h_g, axis=(1, 3), keepdims=True)
                var = jnp.var(h_g, axis=(1, 3), keepdims=True)
            h = ((h_g - mean) * jax.lax.rsqrt(var + 1e-5)).reshape(
                bsz, seq_len, self.dim
            )
            gn_scale = self.param(
                "conv_gn_scale", nn.initializers.ones, (self.dim,)
            )
            gn_bias = self.param(
                "conv_gn_bias", nn.initializers.zeros, (self.dim,)
            )
            h = h * gn_scale[None, None, :] + gn_bias[None, None, :]
        else:
            h = nn.LayerNorm(name="conv_ln")(h)
        if mask_f is not None:
            h = h * mask_f
        h = nn.silu(h)
        h = nn.Dense(self.dim, name="pw_conv2")(h)
        if mask_f is not None:
            h = h * mask_f
        return h


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block (FFN/2 + MHSA + Conv + FFN/2 + optional LayerNorm)."""

    dim: int = 256
    num_heads: int = 4
    ffn_dim: int = 512
    conv_kernel_size: int = 15
    use_group_norm: bool = True
    normalize_output: bool = True
    layer_order: str = "mhsa_before_conv"

    def _mhsa(self, x: jax.Array, mask: Optional[jax.Array], mask_f: Optional[jax.Array]) -> jax.Array:
        attn_mask = None
        if mask is not None:
            attn_mask = nn.make_attention_mask(
                jnp.ones_like(mask, dtype=jnp.bool_),
                mask > 0,
                dtype=x.dtype,
            )
        h = nn.LayerNorm(name="mhsa_ln")(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            name="mhsa",
        )(h, h, mask=attn_mask)
        x = x + h
        if mask_f is not None:
            x = x * mask_f
        return x

    def _conv(self, x: jax.Array, mask: Optional[jax.Array], mask_f: Optional[jax.Array]) -> jax.Array:
        x = x + ConformerConvModule(
            dim=self.dim,
            kernel_size=self.conv_kernel_size,
            use_group_norm=self.use_group_norm,
            name="conv_mod",
        )(x, mask=mask)
        if mask_f is not None:
            x = x * mask_f
        return x

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        order = self.layer_order.lower()
        if order not in ("mhsa_before_conv", "conv_before_mhsa"):
            raise ValueError(f"Unknown Conformer layer_order: {self.layer_order}")
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        # 1. First half-step FFN
        h = nn.LayerNorm(name="ffn1_ln")(x)
        h = nn.Dense(self.ffn_dim, name="ffn1_up")(h)
        h = nn.silu(h)
        h = nn.Dense(self.dim, name="ffn1_down")(h)
        x = x + 0.5 * h
        if mask_f is not None:
            x = x * mask_f

        # 2./3. Multi-Head Self-Attention and Convolution module (configurable order)
        if order == "mhsa_before_conv":
            x = self._mhsa(x, mask, mask_f)
            x = self._conv(x, mask, mask_f)
        else:
            x = self._conv(x, mask, mask_f)
            x = self._mhsa(x, mask, mask_f)

        # 4. Second half-step FFN
        h = nn.LayerNorm(name="ffn2_ln")(x)
        h = nn.Dense(self.ffn_dim, name="ffn2_up")(h)
        h = nn.silu(h)
        h = nn.Dense(self.dim, name="ffn2_down")(h)
        x = x + 0.5 * h
        if mask_f is not None:
            x = x * mask_f

        if self.normalize_output:
            x = nn.LayerNorm(name="out_ln")(x)
            if mask_f is not None:
                x = x * mask_f
        return x


class ConformerBackbone(nn.Module):
    """Conformer encoder backbone for speaker representation learning (Gulati et al. 2020).

    Key speaker recognition design choices:
    - Shift-invariant self-attention (`use_positional_encoding=False` by default)
      so randomly cropped segments and variable-length utterances do not suffer
      from absolute frame-index bias.
    - Local 1D convolutional input stem (`use_conv_stem=True`) for immediate
      short-range time-frequency context.
    - Mask-aware `GroupNorm(num_groups=2)` inside the depthwise convolution module.
    - `normalize_output=False` before Attentive Temporal Pooling (`MEAN_STD`) so
      frame-to-frame magnitude variations are preserved for sigmoid frame weighting
      and standard deviation computation.
    """

    dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 512
    conv_kernel_size: int = 15
    use_positional_encoding: bool = False
    normalize_output: bool = False
    use_conv_stem: bool = True
    use_group_norm: bool = True
    layer_order: str = "mhsa_before_conv"
    subsample_factor: int = 1
    subsample_after_layer: int = 3

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Returns frame representations `(B, T', D)`.

        `T' = T` unless `subsample_factor > 1`, in which case stack-by-`subsample_factor`
        time reduction is applied after `min(subsample_after_layer, num_layers)` blocks and
        `T' = ceil(T / subsample_factor)`; callers must then use
        `subsample_mask(mask, subsample_factor)` for downstream pooling.
        """
        factor = max(1, int(self.subsample_factor))
        sub_after = min(max(0, int(self.subsample_after_layer)), self.num_layers)
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        if mask_f is not None:
            x = x * mask_f
        h = nn.Dense(self.dim, name="input_proj")(x)
        if self.use_conv_stem:
            c = nn.Conv(
                features=self.dim,
                kernel_size=(3,),
                padding="SAME",
                name="stem_conv",
            )(x)
            c = nn.silu(nn.LayerNorm(name="stem_ln")(c))
            h = h + c
        if self.use_positional_encoding:
            pe = sinusoidal_positional_encoding(h.shape[1], self.dim, h.dtype)
            h = h + pe
        if mask_f is not None:
            h = h * mask_f
        layer_outs = []
        cur_mask = mask
        for i in range(self.num_layers):
            if factor > 1 and i == sub_after:
                # Stack-by-`factor` time reduction + linear projection back to `dim`.
                h, cur_mask = _stack_frames(h, factor, cur_mask)
                h = nn.Dense(self.dim, name="subsample_proj")(h)
                mask_f = (
                    (cur_mask > 0).astype(h.dtype)[..., None]
                    if cur_mask is not None
                    else None
                )
                if mask_f is not None:
                    h = h * mask_f
                # Earlier layer outputs are time-reduced parameter-free (group mean)
                # so Multi-Layer Feature Aggregation can concatenate them.
                layer_outs = [_average_frames(lo, factor) for lo in layer_outs]
            h = ConformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                conv_kernel_size=self.conv_kernel_size,
                use_group_norm=self.use_group_norm,
                normalize_output=False,
                layer_order=self.layer_order,
                name=f"conformer_{i}",
            )(h, deterministic=deterministic, mask=cur_mask)
            layer_outs.append(h)
        if factor > 1 and sub_after >= self.num_layers:
            h, cur_mask = _stack_frames(h, factor, cur_mask)
            h = nn.Dense(self.dim, name="subsample_proj")(h)
            mask_f = (
                (cur_mask > 0).astype(h.dtype)[..., None]
                if cur_mask is not None
                else None
            )
            if mask_f is not None:
                h = h * mask_f
            layer_outs = [_average_frames(lo, factor) for lo in layer_outs[:-1]] + [h]
        cat = jnp.concatenate(layer_outs, axis=-1)
        out = nn.Dense(self.dim, name="mfa_proj")(nn.LayerNorm(name="mfa_ln")(cat))
        if self.normalize_output:
            out = nn.LayerNorm(name="out_ln")(out)
        if mask_f is not None:
            out = out * mask_f
        return out


def _selective_ssm_scan(
    u: jax.Array,
    delta: jax.Array,
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    D: jax.Array,
) -> jax.Array:
    """Parallel Selective State Space scan using `jax.lax.associative_scan`."""
    # Discretize: A_bar = exp(delta * A) -> (B, T, d_inner, d_state)
    a_bar = jnp.exp(jnp.einsum("btd,dn->btdn", delta, A))
    # B_bar * u = delta * u * B -> (B, T, d_inner, d_state)
    bu_bar = jnp.einsum("btd,btn->btdn", delta * u, B)

    def _binary_assoc(
        left: tuple[jax.Array, jax.Array], right: tuple[jax.Array, jax.Array]
    ):
        a_l, b_l = left
        a_r, b_r = right
        return a_r * a_l, a_r * b_l + b_r

    _, h_states = jax.lax.associative_scan(
        _binary_assoc, (a_bar, bu_bar), axis=1
    )
    y = jnp.einsum("btdn,btn->btd", h_states, C) + u * D[None, None, :]
    return y


def _reverse_valid_frames(
    arr: jax.Array, mask: Optional[jax.Array] = None
) -> jax.Array:
    """Reverses only the valid prefix frames `[0 .. L_b - 1]` along `axis=1`."""
    if mask is None:
        return jnp.flip(arr, axis=1)
    seq_len = arr.shape[1]
    valid_len = jnp.maximum(
        jnp.sum((mask > 0).astype(jnp.int32), axis=1, keepdims=True), 1
    )
    t_idx = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    rev_idx = jnp.where(t_idx < valid_len, valid_len - 1 - t_idx, t_idx)
    return jnp.take_along_axis(arr, rev_idx[:, :, None], axis=1)


class MambaBlock(nn.Module):
    """Selective State Space Model (Mamba) block (Gu & Dao, 2023)."""

    dim: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: int = 16
    bidirectional: bool = False

    @nn.compact
    def __call__(
        self, x: jax.Array, mask: Optional[jax.Array] = None
    ) -> jax.Array:
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        d_inner = self.dim * self.expand_factor
        h = nn.LayerNorm(name="ln")(x)
        xz = nn.Dense(2 * d_inner, use_bias=False, name="in_proj")(h)
        u, z = jnp.split(xz, 2, axis=-1)
        if mask_f is not None:
            u = u * mask_f
            z = z * mask_f

        # Depthwise 1D convolution over u
        u_conv = nn.Conv(
            features=d_inner,
            kernel_size=(self.d_conv,),
            feature_group_count=d_inner,
            padding="SAME",
            name="conv1d",
        )(u)
        u_act = nn.silu(u_conv)
        if mask_f is not None:
            u_act = u_act * mask_f

        # Selective parameters (dt_raw, B, C) from u_act
        dbc = nn.Dense(
            self.dt_rank + 2 * self.d_state, use_bias=False, name="x_proj"
        )(u_act)
        dt_raw, B_mat, C_mat = jnp.split(
            dbc, [self.dt_rank, self.dt_rank + self.d_state], axis=-1
        )
        delta = jax.nn.softplus(nn.Dense(d_inner, name="dt_proj")(dt_raw))

        # Diagonal state matrix A initialized with log(1..d_state) (HiPPO-inspired)
        def _a_log_init(
            rng: jax.Array, shape: tuple[int, int], dtype: Any = jnp.float32
        ):
            base = jnp.log(jnp.arange(1, shape[1] + 1, dtype=dtype))
            return jnp.broadcast_to(base[None, :], shape)

        a_log = self.param("A_log", _a_log_init, (d_inner, self.d_state))
        A = -jnp.exp(a_log)
        D = self.param("D", nn.initializers.ones, (d_inner,))

        y = _selective_ssm_scan(u_act, delta, A, B_mat, C_mat, D)
        if self.bidirectional:
            u_rev = _reverse_valid_frames(u_act, mask)
            d_rev = _reverse_valid_frames(delta, mask)
            b_rev = _reverse_valid_frames(B_mat, mask)
            c_rev = _reverse_valid_frames(C_mat, mask)
            y_rev = _selective_ssm_scan(u_rev, d_rev, A, b_rev, c_rev, D)
            y = 0.5 * (y + _reverse_valid_frames(y_rev, mask))
        y = y * nn.silu(z)
        out = nn.Dense(self.dim, use_bias=False, name="out_proj")(y)
        res = x + out
        if mask_f is not None:
            res = res * mask_f
        return res


class MambaBackbone(nn.Module):
    """Stacked Mamba (Selective SSM) backbone for speaker recognition."""

    dim: int = 256
    num_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: int = 16
    bidirectional: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        if mask_f is not None:
            x = x * mask_f
        h = nn.Dense(self.dim, name="input_proj")(x)
        if mask_f is not None:
            h = h * mask_f
        for i in range(self.num_layers):
            h = MambaBlock(
                dim=self.dim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand_factor=self.expand_factor,
                dt_rank=self.dt_rank,
                bidirectional=self.bidirectional,
                name=f"mamba_{i}",
            )(h, mask=mask)
        out = nn.LayerNorm(name="final_ln")(h)
        if mask_f is not None:
            out = out * mask_f
        return out


class SqueezeExcitation1D(nn.Module):
    """1D Squeeze-and-Excitation channel attention block."""

    channels: int
    bottleneck_dim: int = 64

    @nn.compact
    def __call__(
        self, x: jax.Array, mask: Optional[jax.Array] = None
    ) -> jax.Array:
        # x: (B, T, C)
        if mask is not None:
            mask_f = (mask > 0).astype(x.dtype)[..., None]
            denom = jnp.maximum(jnp.sum(mask_f, axis=1, keepdims=True), 1.0)
            s = jnp.sum(x * mask_f, axis=1, keepdims=True) / denom
        else:
            s = jnp.mean(x, axis=1, keepdims=True)  # (B, 1, C)
        s = nn.relu(nn.Dense(self.bottleneck_dim, name="se_fc1")(s))
        s = jax.nn.sigmoid(nn.Dense(self.channels, name="se_fc2")(s))
        out = x * s
        if mask is not None:
            out = out * (mask > 0).astype(x.dtype)[..., None]
        return out


class SERes2Block(nn.Module):
    """ECAPA-TDNN Squeeze-and-Excitation Res2Net block."""

    channels: int = 256
    kernel_size: int = 3
    dilation: int = 2
    scale: int = 4
    se_bottleneck_dim: int = 64
    norm_after_activation: bool = False

    def _norm_act(self, h: jax.Array, name: str) -> jax.Array:
        if self.norm_after_activation:
            return nn.LayerNorm(name=name)(nn.relu(h))
        return nn.relu(nn.LayerNorm(name=name)(h))

    @nn.compact
    def __call__(
        self, x: jax.Array, mask: Optional[jax.Array] = None
    ) -> jax.Array:
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        if mask_f is not None:
            x = x * mask_f
        residual = x
        h = self._norm_act(nn.Dense(self.channels, name="pw1")(x), "ln1")
        if mask_f is not None:
            h = h * mask_f

        # Split channels into `scale` sub-bands for hierarchical multi-scale conv
        width = self.channels // self.scale
        splits = jnp.split(h, self.scale, axis=-1)
        outs = [splits[0]]
        prev = None
        for s_idx in range(1, self.scale):
            cur = splits[s_idx] if prev is None else splits[s_idx] + prev
            if mask_f is not None:
                cur = cur * mask_f
            cur = nn.Conv(
                features=width,
                kernel_size=(self.kernel_size,),
                kernel_dilation=(self.dilation,),
                padding="SAME",
                name=f"res2_conv_{s_idx}",
            )(cur)
            cur = self._norm_act(cur, f"res2_ln_{s_idx}")
            if mask_f is not None:
                cur = cur * mask_f
            outs.append(cur)
            prev = cur

        h = jnp.concatenate(outs, axis=-1)
        h = self._norm_act(nn.Dense(self.channels, name="pw2")(h), "ln2")
        h = SqueezeExcitation1D(
            channels=self.channels,
            bottleneck_dim=self.se_bottleneck_dim,
            name="se",
        )(h, mask=mask)
        res = residual + h
        if mask_f is not None:
            res = res * mask_f
        return res


class EcapaTdnnBackbone(nn.Module):
    """ECAPA-TDNN backbone (Desplanques et al., Interspeech 2020).

    Combines an initial 1D TDNN layer, multiple SE-Res2Net blocks with increasing
    dilations, and Multi-layer Feature Aggregation (MFA).
    """

    channels: int = 256
    scale: int = 4
    kernel_sizes: Sequence[int] = (5, 3, 3, 3)
    dilations: Sequence[int] = (1, 2, 3, 4)
    mfa_dim: int = 512
    se_bottleneck_dim: int = 64
    norm_after_activation: bool = False

    def _norm_act(self, h: jax.Array, name: str) -> jax.Array:
        if self.norm_after_activation:
            return nn.LayerNorm(name=name)(nn.relu(h))
        return nn.relu(nn.LayerNorm(name=name)(h))

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        if mask_f is not None:
            x = x * mask_f
        # Stem 1D Conv
        h = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_sizes[0],),
            padding="SAME",
            name="stem_conv",
        )(x)
        h = self._norm_act(h, "stem_ln")
        if mask_f is not None:
            h = h * mask_f

        block_outputs = []
        num_blocks = min(len(self.kernel_sizes) - 1, len(self.dilations) - 1)
        for i in range(max(1, num_blocks)):
            k_size = self.kernel_sizes[min(i + 1, len(self.kernel_sizes) - 1)]
            dil = self.dilations[min(i + 1, len(self.dilations) - 1)]
            h = SERes2Block(
                channels=self.channels,
                kernel_size=k_size,
                dilation=dil,
                scale=self.scale,
                se_bottleneck_dim=self.se_bottleneck_dim,
                norm_after_activation=self.norm_after_activation,
                name=f"se_res2_{i}",
            )(h, mask=mask)
            block_outputs.append(h)

        # Multi-layer Feature Aggregation (MFA)
        cat = jnp.concatenate(block_outputs, axis=-1)
        out = self._norm_act(
            nn.Dense(self.mfa_dim, name="mfa_proj")(cat), "mfa_ln"
        )
        if mask_f is not None:
            out = out * mask_f
        return out


class ResNetBasicBlock2D(nn.Module):
    """2D Residual Block with optional Squeeze-and-Excitation."""

    channels: int
    stride: tuple[int, int] = (1, 1)
    use_se: bool = True

    @nn.compact
    def __call__(
        self, x: jax.Array, mask: Optional[jax.Array] = None
    ) -> jax.Array:
        mask_f = (
            (mask > 0).astype(x.dtype)[:, :, None, None]
            if mask is not None
            else None
        )
        if mask_f is not None:
            x = x * mask_f
        residual = x
        h = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=self.stride,
            padding="SAME",
            use_bias=False,
            name="conv1",
        )(x)
        h = nn.relu(nn.LayerNorm(name="ln1")(h))
        if mask_f is not None:
            h = h * mask_f
        h = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="conv2",
        )(h)
        h = nn.LayerNorm(name="ln2")(h)
        if mask_f is not None:
            h = h * mask_f

        if self.use_se:
            if mask_f is not None:
                denom = jnp.maximum(
                    jnp.sum(mask_f, axis=(1, 2), keepdims=True)
                    * float(h.shape[2]),
                    1.0,
                )
                se = jnp.sum(h * mask_f, axis=(1, 2), keepdims=True) / denom
            else:
                se = jnp.mean(h, axis=(1, 2), keepdims=True)
            bottleneck = max(8, self.channels // 4)
            se = nn.relu(nn.Dense(bottleneck, name="se1")(se))
            se = jax.nn.sigmoid(nn.Dense(self.channels, name="se2")(se))
            h = h * se

        if residual.shape != h.shape:
            residual = nn.Conv(
                features=self.channels,
                kernel_size=(1, 1),
                strides=self.stride,
                padding="SAME",
                use_bias=False,
                name="downsample",
            )(residual)
            residual = nn.LayerNorm(name="downsample_ln")(residual)

        out = nn.relu(residual + h)
        if mask_f is not None:
            out = out * mask_f
        return out


class ResNetBackbone(nn.Module):
    """2D Time-Frequency SE-ResNet backbone (Thin-ResNet / r-vector style)."""

    base_channels: int = 32
    num_blocks: Sequence[int] = (2, 2, 2, 2)
    use_se: bool = True
    out_dim: int = 256

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        mask_f = (
            (mask > 0).astype(x.dtype)[..., None] if mask is not None else None
        )
        if mask_f is not None:
            x = x * mask_f
        # x: (B, T, F) -> (B, T, F, 1)
        h = x[..., None]
        h = nn.Conv(
            features=self.base_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="stem_conv",
        )(h)
        h = nn.relu(nn.LayerNorm(name="stem_ln")(h))
        if mask_f is not None:
            h = h * mask_f[..., None]

        for stage_idx, n_blocks in enumerate(self.num_blocks):
            ch = self.base_channels * (2 ** stage_idx)
            for b_idx in range(n_blocks):
                # Downsample frequency axis at start of stages 1..3, preserve time resolution
                stride = (1, 2) if (stage_idx > 0 and b_idx == 0) else (1, 1)
                h = ResNetBasicBlock2D(
                    channels=ch,
                    stride=stride,
                    use_se=self.use_se,
                    name=f"stage{stage_idx}_block{b_idx}",
                )(h, mask=mask)

        # Collapse frequency & channel dimensions: (B, T, F', C) -> (B, T, F'*C)
        bsz, seq_len, freq_bins, ch = h.shape
        h = h.reshape(bsz, seq_len, freq_bins * ch)
        h = nn.relu(
            nn.LayerNorm(name="out_ln")(
                nn.Dense(self.out_dim, name="out_proj")(h)
            )
        )
        if mask_f is not None:
            h = h * mask_f
        return h


class ModernSpeakerEncoder(nn.Module):
    """Unified configurable speaker encoder supporting all backbones and pooling heads.

    Architecture flow (backbone -> pooling -> bottleneck -> embedding, cf. arXiv:2203.05642 Sec. 3):
    `Input (B, T, F) -> Backbone -> Temporal Pooling -> Bottleneck FFN (ReLU) -> Linear Proj -> Output (B, D)`
    (When `pooling_type == "pfas"`, the output is the packed Key-Value representation
    of shape `(B, num_keys * (key_dim + value_dim))` directly from `PfasRepresentationHead`.)
    """

    backbone_type: str = "conformer"
    pooling_type: str = "atp"
    embedding_dim: int = 192
    normalize_embedding: bool = True
    backbone_kwargs: Optional[dict[str, Any]] = None
    pooling_kwargs: Optional[dict[str, Any]] = None

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        b_kwargs = dict(self.backbone_kwargs or {})
        p_kwargs = dict(self.pooling_kwargs or {})
        b_type = self.backbone_type.lower()

        if b_type == "lstm":
            frames = LstmBackbone(**b_kwargs, name="backbone")(
                x, deterministic=deterministic, mask=mask
            )
        elif b_type == "transformer":
            frames = TransformerBackbone(**b_kwargs, name="backbone")(
                x, deterministic=deterministic, mask=mask
            )
        elif b_type == "conformer":
            frames = ConformerBackbone(**b_kwargs, name="backbone")(
                x, deterministic=deterministic, mask=mask
            )
            mask = subsample_mask(mask, int(b_kwargs.get("subsample_factor", 1)))
        elif b_type == "mamba":
            frames = MambaBackbone(**b_kwargs, name="backbone")(
                x, deterministic=deterministic, mask=mask
            )
        elif b_type == "ecapa_tdnn":
            frames = EcapaTdnnBackbone(**b_kwargs, name="backbone")(
                x, deterministic=deterministic, mask=mask
            )
        elif b_type == "resnet":
            frames = ResNetBackbone(**b_kwargs, name="backbone")(
                x, deterministic=deterministic, mask=mask
            )
        else:
            raise ValueError(
                f"Unsupported backbone_type: {self.backbone_type}"
            )

        p_type = self.pooling_type.lower()
        if p_type == "pfas":
            return pooling.PfasRepresentationHead(
                **p_kwargs, name="pfas_head"
            )(frames, mask=mask)
        elif p_type == "atp":
            pooled = pooling.AttentiveTemporalPooling(
                num_heads=int(p_kwargs.get("num_heads", 4)),
                attention_dim=int(p_kwargs.get("attention_dim", 128)),
                pooling_stats=str(p_kwargs.get("pooling_stats", "mean_std")),
                subspace_heads=bool(p_kwargs.get("subspace_heads", True)),
                eps=float(p_kwargs.get("eps", 1e-4)),
                name="pool",
            )(frames, mask=mask)
        elif p_type == "mean":
            pooled = pooling.MeanPooling(name="pool")(frames, mask=mask)
        elif p_type == "last":
            pooled = pooling.LastFramePooling(name="pool")(frames, mask=mask)
        elif p_type == "stats":
            pooled = pooling.StatsPooling(name="pool")(frames, mask=mask)
        elif p_type == "sap":
            pooled = pooling.SelfAttentivePooling(
                attention_dim=int(p_kwargs.get("attention_dim", 128)),
                name="pool",
            )(frames, mask=mask)
        elif p_type == "asp":
            pooled = pooling.AttentiveStatsPooling(
                attention_dim=int(p_kwargs.get("attention_dim", 128)),
                use_global_context=bool(
                    p_kwargs.get("use_global_context", True)
                ),
                name="pool",
            )(frames, mask=mask)
        elif p_type == "cumulative_stats":
            pooled = pooling.CumulativeStatsPooling(name="pool")(
                frames, mask=mask
            )
        else:
            raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")

        # Post-pooling bottleneck FFN:
        # LayerNorm -> [Dense(bottleneck_dim) -> LayerNorm -> ReLU] -> Dense(embedding_dim)
        emb = nn.LayerNorm(name="bottleneck_ln")(pooled)
        use_nonlinear = bool(p_kwargs.get("use_nonlinear_bottleneck", True))
        if use_nonlinear:
            bottleneck_dim = int(p_kwargs.get("bottleneck_dim", 256))
            emb = nn.Dense(bottleneck_dim, name="bottleneck_fc")(emb)
            emb = nn.LayerNorm(name="bottleneck_act_ln")(emb)
            emb = nn.relu(emb)

        emb = nn.Dense(self.embedding_dim, name="embedding_proj")(emb)
        if self.normalize_embedding:
            emb = emb / (jnp.linalg.norm(emb, axis=-1, keepdims=True) + 1e-6)
        return emb
