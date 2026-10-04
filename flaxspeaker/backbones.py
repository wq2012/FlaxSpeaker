"""Neural network backbones and unified speaker encoder in Flax Linen.

Supported backbones:
- `LstmBackbone`: Multi-layer uni/bidirectional LSTM with optional projections
- `TransformerBackbone`: Pre-LN Transformer encoder with sinusoidal positional encoding
- `ConformerBackbone`: Macaron-style Conformer (FFN + MHSA + Depthwise Conv + FFN)
- `MambaBackbone`: Selective State Space Model (SSM) using `jax.lax.associative_scan`
- `EcapaTdnnBackbone`: Emphasized Channel Attention, Propagation and Aggregation TDNN
  (SE-Res2Net blocks + Multi-layer Feature Aggregation)
- `ResNetBackbone`: 2D Time-Frequency SE-ResNet (WeSpeaker / Thin-ResNet style)
"""

from typing import Any, Sequence
from flax import linen as nn
import jax
import jax.numpy as jnp

from flaxspeaker import pooling


def sinusoidal_positional_encoding(seq_len: int, dim: int, dtype: Any = jnp.float32) -> jax.Array:
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


class LstmBackbone(nn.Module):
    """Multi-layer LSTM backbone with optional projection and bidirectionality."""

    hidden_size: int = 256
    num_layers: int = 3
    bidirectional: bool = False
    proj_size: int = 0

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        # x: (B, T, F)
        h = x
        for i in range(self.num_layers):
            fwd_rnn = nn.RNN(
                nn.OptimizedLSTMCell(features=self.hidden_size),
                name=f"lstm_fwd_{i}",
            )
            y_fwd = fwd_rnn(h)
            if self.bidirectional:
                bwd_rnn = nn.RNN(
                    nn.OptimizedLSTMCell(features=self.hidden_size),
                    reverse=True,
                    name=f"lstm_bwd_{i}",
                )
                y_bwd = bwd_rnn(h)
                y = jnp.concatenate([y_fwd, y_bwd], axis=-1)
            else:
                y = y_fwd

            if self.proj_size > 0:
                y = nn.Dense(self.proj_size, name=f"lstm_proj_{i}")(y)

            if h.shape[-1] == y.shape[-1] and i > 0:
                h = nn.LayerNorm(name=f"lstm_ln_{i}")(h + y)
            else:
                h = nn.LayerNorm(name=f"lstm_ln_{i}")(y)
        return h


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block."""

    dim: int = 256
    num_heads: int = 4
    mlp_dim: int = 512

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        h = nn.LayerNorm(name="ln1")(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            name="mhsa",
        )(h, h)
        x = x + h

        h = nn.LayerNorm(name="ln2")(x)
        h = nn.Dense(self.mlp_dim, name="ffn1")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.dim, name="ffn2")(h)
        return x + h


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone with sinusoidal positional encoding."""

    dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    mlp_dim: int = 512
    use_positional_encoding: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        h = nn.Dense(self.dim, name="input_proj")(x)
        if self.use_positional_encoding:
            pe = sinusoidal_positional_encoding(h.shape[1], self.dim, h.dtype)
            h = h + pe
        for i in range(self.num_layers):
            h = TransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                name=f"block_{i}",
            )(h, deterministic=deterministic)
        return nn.LayerNorm(name="final_ln")(h)


class ConformerConvModule(nn.Module):
    """Conformer depthwise separable convolution module with GLU gating."""

    dim: int = 256
    kernel_size: int = 15

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h = nn.LayerNorm(name="ln")(x)
        h = nn.Dense(2 * self.dim, name="pw_conv1")(h)
        # Gated Linear Unit (GLU)
        h_a, h_b = jnp.split(h, 2, axis=-1)
        h = h_a * jax.nn.sigmoid(h_b)
        # 1D Depthwise Convolution
        h = nn.Conv(
            features=self.dim,
            kernel_size=(self.kernel_size,),
            feature_group_count=self.dim,
            padding="SAME",
            name="dw_conv",
        )(h)
        h = nn.LayerNorm(name="conv_ln")(h)
        h = nn.silu(h)
        h = nn.Dense(self.dim, name="pw_conv2")(h)
        return h


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block (FFN/2 + MHSA + Conv + FFN/2 + LayerNorm)."""

    dim: int = 256
    num_heads: int = 4
    ffn_dim: int = 512
    conv_kernel_size: int = 15

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        # 1. First half-step FFN
        h = nn.LayerNorm(name="ffn1_ln")(x)
        h = nn.Dense(self.ffn_dim, name="ffn1_up")(h)
        h = nn.silu(h)
        h = nn.Dense(self.dim, name="ffn1_down")(h)
        x = x + 0.5 * h

        # 2. Multi-Head Self-Attention
        h = nn.LayerNorm(name="mhsa_ln")(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            name="mhsa",
        )(h, h)
        x = x + h

        # 3. Convolution module
        x = x + ConformerConvModule(
            dim=self.dim,
            kernel_size=self.conv_kernel_size,
            name="conv_mod",
        )(x)

        # 4. Second half-step FFN
        h = nn.LayerNorm(name="ffn2_ln")(x)
        h = nn.Dense(self.ffn_dim, name="ffn2_up")(h)
        h = nn.silu(h)
        h = nn.Dense(self.dim, name="ffn2_down")(h)
        x = x + 0.5 * h

        return nn.LayerNorm(name="out_ln")(x)


class ConformerBackbone(nn.Module):
    """Conformer encoder backbone for speaker representation learning."""

    dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 512
    conv_kernel_size: int = 15

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        h = nn.Dense(self.dim, name="input_proj")(x)
        pe = sinusoidal_positional_encoding(h.shape[1], self.dim, h.dtype)
        h = h + pe
        for i in range(self.num_layers):
            h = ConformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                conv_kernel_size=self.conv_kernel_size,
                name=f"conformer_{i}",
            )(h, deterministic=deterministic)
        return h


def _selective_ssm_scan(
    u: jax.Array,
    delta: jax.Array,
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    D: jax.Array,
) -> jax.Array:
    """Parallel Selective State Space scan using `jax.lax.associative_scan`.

    Args:
        u: `(B, T, d_inner)` input sequence
        delta: `(B, T, d_inner)` step sizes (positive after softplus)
        A: `(d_inner, d_state)` negative state transition rates
        B: `(B, T, d_state)` input-dependent B matrix
        C: `(B, T, d_state)` input-dependent C matrix
        D: `(d_inner,)` skip connection weights

    Returns:
        Output sequence of shape `(B, T, d_inner)`.
    """
    # Discretize: A_bar = exp(delta * A) -> (B, T, d_inner, d_state)
    a_bar = jnp.exp(jnp.einsum("btd,dn->btdn", delta, A))
    # B_bar * u = delta * u * B -> (B, T, d_inner, d_state)
    bu_bar = jnp.einsum("btd,btn->btdn", delta * u, B)

    def _binary_assoc(left: tuple[jax.Array, jax.Array], right: tuple[jax.Array, jax.Array]):
        a_l, b_l = left
        a_r, b_r = right
        return a_r * a_l, a_r * b_l + b_r

    _, h_states = jax.lax.associative_scan(_binary_assoc, (a_bar, bu_bar), axis=1)
    y = jnp.einsum("btdn,btn->btd", h_states, C) + u * D[None, None, :]
    return y


class MambaBlock(nn.Module):
    """Selective State Space Model (Mamba) block (Gu & Dao, 2023)."""

    dim: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: int = 16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        d_inner = self.dim * self.expand_factor
        h = nn.LayerNorm(name="ln")(x)
        xz = nn.Dense(2 * d_inner, use_bias=False, name="in_proj")(h)
        u, z = jnp.split(xz, 2, axis=-1)

        # Depthwise 1D convolution over u
        u_conv = nn.Conv(
            features=d_inner,
            kernel_size=(self.d_conv,),
            feature_group_count=d_inner,
            padding="SAME",
            name="conv1d",
        )(u)
        u_act = nn.silu(u_conv)

        # Selective parameters (dt_raw, B, C) from u_act
        dbc = nn.Dense(
            self.dt_rank + 2 * self.d_state, use_bias=False, name="x_proj"
        )(u_act)
        dt_raw, B_mat, C_mat = jnp.split(
            dbc, [self.dt_rank, self.dt_rank + self.d_state], axis=-1
        )
        delta = jax.nn.softplus(nn.Dense(d_inner, name="dt_proj")(dt_raw))

        # Diagonal state matrix A initialized with log(1..d_state) (HiPPO-inspired)
        def _a_log_init(rng: jax.Array, shape: tuple[int, int], dtype: Any = jnp.float32):
            base = jnp.log(jnp.arange(1, shape[1] + 1, dtype=dtype))
            return jnp.broadcast_to(base[None, :], shape)

        a_log = self.param("A_log", _a_log_init, (d_inner, self.d_state))
        A = -jnp.exp(a_log)
        D = self.param("D", nn.initializers.ones, (d_inner,))

        y = _selective_ssm_scan(u_act, delta, A, B_mat, C_mat, D)
        y = y * nn.silu(z)
        out = nn.Dense(self.dim, use_bias=False, name="out_proj")(y)
        return x + out


class MambaBackbone(nn.Module):
    """Stacked Mamba (Selective SSM) backbone for speaker recognition."""

    dim: int = 256
    num_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: int = 16

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        h = nn.Dense(self.dim, name="input_proj")(x)
        for i in range(self.num_layers):
            h = MambaBlock(
                dim=self.dim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand_factor=self.expand_factor,
                dt_rank=self.dt_rank,
                name=f"mamba_{i}",
            )(h)
        return nn.LayerNorm(name="final_ln")(h)


class SqueezeExcitation1D(nn.Module):
    """1D Squeeze-and-Excitation channel attention block."""

    channels: int
    bottleneck_dim: int = 64

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, T, C)
        s = jnp.mean(x, axis=1, keepdims=True)  # (B, 1, C)
        s = nn.relu(nn.Dense(self.bottleneck_dim, name="se_fc1")(s))
        s = jax.nn.sigmoid(nn.Dense(self.channels, name="se_fc2")(s))
        return x * s


class SERes2Block(nn.Module):
    """ECAPA-TDNN Squeeze-and-Excitation Res2Net block."""

    channels: int = 256
    kernel_size: int = 3
    dilation: int = 2
    scale: int = 4
    se_bottleneck_dim: int = 64

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        h = nn.relu(nn.LayerNorm(name="ln1")(nn.Dense(self.channels, name="pw1")(x)))

        # Split channels into `scale` sub-bands for hierarchical multi-scale conv
        width = self.channels // self.scale
        splits = jnp.split(h, self.scale, axis=-1)
        outs = [splits[0]]
        prev = None
        for s_idx in range(1, self.scale):
            cur = splits[s_idx] if prev is None else splits[s_idx] + prev
            cur = nn.Conv(
                features=width,
                kernel_size=(self.kernel_size,),
                kernel_dilation=(self.dilation,),
                padding="SAME",
                name=f"res2_conv_{s_idx}",
            )(cur)
            cur = nn.relu(nn.LayerNorm(name=f"res2_ln_{s_idx}")(cur))
            outs.append(cur)
            prev = cur

        h = jnp.concatenate(outs, axis=-1)
        h = nn.relu(nn.LayerNorm(name="ln2")(nn.Dense(self.channels, name="pw2")(h)))
        h = SqueezeExcitation1D(
            channels=self.channels,
            bottleneck_dim=self.se_bottleneck_dim,
            name="se",
        )(h)
        return residual + h


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

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        # Stem 1D Conv
        h = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_sizes[0],),
            padding="SAME",
            name="stem_conv",
        )(x)
        h = nn.relu(nn.LayerNorm(name="stem_ln")(h))

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
                name=f"se_res2_{i}",
            )(h)
            block_outputs.append(h)

        # Multi-layer Feature Aggregation (MFA)
        cat = jnp.concatenate(block_outputs, axis=-1)
        out = nn.relu(nn.LayerNorm(name="mfa_ln")(nn.Dense(self.mfa_dim, name="mfa_proj")(cat)))
        return out


class ResNetBasicBlock2D(nn.Module):
    """2D Residual Block with optional Squeeze-and-Excitation."""

    channels: int
    stride: tuple[int, int] = (1, 1)
    use_se: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
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
        h = nn.Conv(
            features=self.channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="conv2",
        )(h)
        h = nn.LayerNorm(name="ln2")(h)

        if self.use_se:
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

        return nn.relu(residual + h)


class ResNetBackbone(nn.Module):
    """2D Time-Frequency SE-ResNet backbone (Thin-ResNet / r-vector style)."""

    base_channels: int = 32
    num_blocks: Sequence[int] = (2, 2, 2, 2)
    use_se: bool = True
    out_dim: int = 256

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
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
                )(h)

        # Collapse frequency & channel dimensions: (B, T, F', C) -> (B, T, F'*C)
        bsz, seq_len, freq_bins, ch = h.shape
        h = h.reshape(bsz, seq_len, freq_bins * ch)
        h = nn.relu(nn.LayerNorm(name="out_ln")(nn.Dense(self.out_dim, name="out_proj")(h)))
        return h


class ModernSpeakerEncoder(nn.Module):
    """Unified configurable speaker encoder supporting all backbones and pooling heads.

    Architecture flow:
    `Input (B, T, F) -> Backbone -> Temporal Pooling -> Bottleneck Projection -> Output (B, D)`
    (When `pooling_type == "pfas"`, the output is the packed Key-Value representation
    of shape `(B, num_keys * (key_dim + value_dim))` directly from `PfasRepresentationHead`.)
    """

    backbone_type: str = "conformer"
    pooling_type: str = "asp"
    embedding_dim: int = 192
    normalize_embedding: bool = True
    backbone_kwargs: dict[str, Any] = None
    pooling_kwargs: dict[str, Any] = None

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        b_kwargs = dict(self.backbone_kwargs or {})
        p_kwargs = dict(self.pooling_kwargs or {})
        b_type = self.backbone_type.lower()

        if b_type == "lstm":
            frames = LstmBackbone(**b_kwargs, name="backbone")(x, deterministic=deterministic)
        elif b_type == "transformer":
            frames = TransformerBackbone(**b_kwargs, name="backbone")(x, deterministic=deterministic)
        elif b_type == "conformer":
            frames = ConformerBackbone(**b_kwargs, name="backbone")(x, deterministic=deterministic)
        elif b_type == "mamba":
            frames = MambaBackbone(**b_kwargs, name="backbone")(x, deterministic=deterministic)
        elif b_type == "ecapa_tdnn":
            frames = EcapaTdnnBackbone(**b_kwargs, name="backbone")(x, deterministic=deterministic)
        elif b_type == "resnet":
            frames = ResNetBackbone(**b_kwargs, name="backbone")(x, deterministic=deterministic)
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")

        p_type = self.pooling_type.lower()
        if p_type == "pfas":
            return pooling.PfasRepresentationHead(**p_kwargs, name="pfas_head")(frames)
        elif p_type == "mean":
            pooled = pooling.MeanPooling(name="pool")(frames)
        elif p_type == "last":
            pooled = pooling.LastFramePooling(name="pool")(frames)
        elif p_type == "stats":
            pooled = pooling.StatsPooling(name="pool")(frames)
        elif p_type == "sap":
            pooled = pooling.SelfAttentivePooling(
                attention_dim=int(p_kwargs.get("attention_dim", 128)),
                name="pool",
            )(frames)
        elif p_type == "asp":
            pooled = pooling.AttentiveStatsPooling(
                attention_dim=int(p_kwargs.get("attention_dim", 128)),
                use_global_context=bool(p_kwargs.get("use_global_context", True)),
                name="pool",
            )(frames)
        elif p_type == "cumulative_stats":
            pooled = pooling.CumulativeStatsPooling(name="pool")(frames)
        else:
            raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")

        emb = nn.LayerNorm(name="bottleneck_ln")(pooled)
        emb = nn.Dense(self.embedding_dim, name="embedding_proj")(emb)
        if self.normalize_embedding:
            emb = emb / (jnp.linalg.norm(emb, axis=-1, keepdims=True) + 1e-6)
        return emb
