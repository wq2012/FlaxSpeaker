"""Spectrum augmentation (SpecAugment) for speaker recognition."""

from __future__ import annotations

import random
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np


def _get_attr(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def apply_specaug(
    features: np.ndarray,
    specaug_config: Any,
    rng: Any = None,
) -> np.ndarray:
    """Applies SpecAugment (frequency masking, time masking, optional noise) to features.

    Args:
        features: 2D array of shape (seq_len, feat_dim).
        specaug_config: SpecAugConfig, Munch, or dict.

    Returns:
        Augmented 2D array of shape (seq_len, feat_dim).
    """
    outputs = np.array(features, copy=True)
    if not _get_attr(specaug_config, "use_specaug", True):
        return outputs

    seq_len, feat_dim = outputs.shape
    mean_feature = float(np.mean(outputs))

    freq_mask_prob = float(_get_attr(specaug_config, "freq_mask_prob", 0.3))
    time_mask_prob = float(_get_attr(specaug_config, "time_mask_prob", 0.3))
    freq_mask_max_width = int(
        _get_attr(specaug_config, "freq_mask_max_width", max(1, feat_dim // 8))
    )
    time_mask_max_width = int(
        _get_attr(specaug_config, "time_mask_max_width", max(1, seq_len // 10))
    )
    num_freq_masks = int(_get_attr(specaug_config, "num_freq_masks", 1))
    num_time_masks = int(_get_attr(specaug_config, "num_time_masks", 1))
    noise_std = float(_get_attr(specaug_config, "gaussian_noise_std", 0.0))

    # Frequency masking
    max_fw = min(freq_mask_max_width, max(1, feat_dim - 1))
    if max_fw >= 1 and feat_dim > 1:
        for _ in range(num_freq_masks):
            if random.random() < freq_mask_prob:
                width = random.randint(1, max_fw)
                start = random.randint(0, feat_dim - width)
                outputs[:, start:start + width] = mean_feature

    # Time masking
    max_tw = min(time_mask_max_width, max(1, seq_len - 1))
    if max_tw >= 1 and seq_len > 1:
        for _ in range(num_time_masks):
            if random.random() < time_mask_prob:
                width = random.randint(1, max_tw)
                start = random.randint(0, seq_len - width)
                outputs[start:start + width, :] = mean_feature

    if noise_std > 0.0:
        outputs = outputs + np.random.normal(
            0.0, noise_std, size=outputs.shape
        ).astype(outputs.dtype)

    return outputs


def apply_specaug_batch_jax(
    batch_features: jax.Array,
    rng: jax.Array,
    freq_mask_max_width: int = 10,
    time_mask_max_width: int = 10,
    mask_prob: float = 0.5,
) -> jax.Array:
    """Pure JAX vectorized SpecAugment on a 3D batch of shape (B, T, D)."""
    bsz, seq_len, feat_dim = batch_features.shape
    rng_f1, rng_f2, rng_fp, rng_t1, rng_t2, rng_tp = jax.random.split(rng, 6)

    mean_vals = jnp.mean(batch_features, axis=(1, 2), keepdims=True)

    # Frequency mask
    fw = jax.random.randint(
        rng_f1, (bsz, 1, 1), 1, max(2, min(freq_mask_max_width + 1, feat_dim))
    )
    f0 = jax.random.randint(
        rng_f2, (bsz, 1, 1), 0, jnp.maximum(1, feat_dim - fw)
    )
    do_f = jax.random.uniform(rng_fp, (bsz, 1, 1)) < mask_prob
    f_idx = jnp.arange(feat_dim)[None, None, :]
    f_mask = do_f & (f_idx >= f0) & (f_idx < f0 + fw)
    out = jnp.where(f_mask, mean_vals, batch_features)

    # Time mask
    tw = jax.random.randint(
        rng_t1, (bsz, 1, 1), 1, max(2, min(time_mask_max_width + 1, seq_len))
    )
    t0 = jax.random.randint(
        rng_t2, (bsz, 1, 1), 0, jnp.maximum(1, seq_len - tw)
    )
    do_t = jax.random.uniform(rng_tp, (bsz, 1, 1)) < mask_prob
    t_idx = jnp.arange(seq_len)[None, :, None]
    t_mask = do_t & (t_idx >= t0) & (t_idx < t0 + tw)
    out = jnp.where(t_mask, mean_vals, out)
    return out
