"""Spectrum augmentation (SpecAugment) and Multi-Style Training (MTR) waveform/feature augmentation.

Enforces mutual exclusion between feature-domain noise augmentation and SpecAugment:
- Additive noise SNR is kept moderate (e.g., 8 to 20 dB; never too high).
- SpecAugment and Multi-Style Training (MTR waveform/noise/reverb augmentation)
  are NEVER applied to the same utterance simultaneously; SpecAugment is applied
  exclusively to clean (unaugmented) utterances.
"""

from __future__ import annotations

import random
from typing import Any, Optional
import jax
import jax.numpy as jnp
import numpy as np


def _get_attr(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _rand_uniform(
    low: float, high: float, rng: Optional[np.random.Generator] = None
) -> float:
    if rng is not None:
        return float(rng.uniform(low, high))
    return random.uniform(low, high)


def _rand_int(
    low: int, high_inclusive: int, rng: Optional[np.random.Generator] = None
) -> int:
    if high_inclusive <= low:
        return low
    if rng is not None:
        return int(rng.integers(low, high_inclusive + 1))
    return random.randint(low, high_inclusive)


def apply_specaug(
    features: np.ndarray,
    specaug_config: Any,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Applies SpecAugment (frequency masking, time masking) to clean features.

    Args:
        features: 2D array of shape (seq_len, feat_dim).
        specaug_config: SpecAugConfig, Munch, or dict.
        rng: Optional NumPy random Generator.

    Returns:
        Augmented 2D array of shape (seq_len, feat_dim).
    """
    outputs = np.array(features, copy=True)
    if not _get_attr(specaug_config, "use_specaug", True):
        return outputs

    seq_len, feat_dim = outputs.shape
    if seq_len <= 2 or feat_dim <= 2:
        return outputs

    mean_feature = float(np.mean(outputs))

    freq_mask_prob = float(_get_attr(specaug_config, "freq_mask_prob", 0.3))
    time_mask_prob = float(_get_attr(specaug_config, "time_mask_prob", 0.3))
    freq_mask_max_width = int(
        _get_attr(specaug_config, "freq_mask_max_width", min(6, max(1, feat_dim // 12)))
    )
    time_mask_max_width = int(
        _get_attr(specaug_config, "time_mask_max_width", min(10, max(1, seq_len // 12)))
    )
    time_mask_max_ratio = float(
        _get_attr(specaug_config, "time_mask_max_ratio", 0.08)
    )
    num_freq_masks = int(_get_attr(specaug_config, "num_freq_masks", 2))
    num_time_masks = int(_get_attr(specaug_config, "num_time_masks", 2))
    noise_std = float(_get_attr(specaug_config, "gaussian_noise_std", 0.0))

    # Frequency masking (cap width so we never mask too much of the spectrum)
    max_fw = min(freq_mask_max_width, max(1, feat_dim // 8))
    if max_fw >= 1 and feat_dim > 1:
        for _ in range(num_freq_masks):
            if _rand_uniform(0.0, 1.0, rng) < freq_mask_prob:
                width = _rand_int(1, max_fw, rng)
                start = _rand_int(0, feat_dim - width, rng)
                outputs[:, start:start + width] = mean_feature

    # Time masking (cap width by both time_mask_max_width and time_mask_max_ratio)
    ratio_cap = max(1, int(round(seq_len * time_mask_max_ratio)))
    max_tw = min(time_mask_max_width, ratio_cap, max(1, seq_len - 1))
    if max_tw >= 1 and seq_len > 1:
        for _ in range(num_time_masks):
            if _rand_uniform(0.0, 1.0, rng) < time_mask_prob:
                width = _rand_int(1, max_tw, rng)
                start = _rand_int(0, seq_len - width, rng)
                outputs[start:start + width, :] = mean_feature

    if noise_std > 0.0:
        if rng is not None:
            noise = rng.normal(0.0, noise_std, size=outputs.shape)
        else:
            noise = np.random.normal(0.0, noise_std, size=outputs.shape)
        outputs = outputs + noise.astype(outputs.dtype)

    return outputs


def _synthesize_noise_waveform(
    length: int,
    sample_rate: int,
    noise_type: str,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Synthesizes realistic environmental/babble/pink/white noise of given length."""
    if rng is not None:
        white = rng.standard_normal(length).astype(np.float32)
    else:
        white = np.random.randn(length).astype(np.float32)

    ntype = noise_type.lower()
    if ntype == "white":
        return white

    freqs = np.fft.rfftfreq(length, d=1.0 / sample_rate)
    spec = np.fft.rfft(white)

    if ntype == "pink":
        scale = 1.0 / np.sqrt(np.maximum(freqs, 20.0))
        colored = np.fft.irfft(spec * scale, n=length).astype(np.float32)
        return colored

    if ntype == "babble":
        # Speech-shaped bandpass (250-3600 Hz) with slow syllabic (~4 Hz) modulation
        band = np.exp(-0.5 * ((freqs - 900.0) / 850.0) ** 2) + 0.35 * np.exp(
            -0.5 * ((freqs - 2200.0) / 700.0) ** 2
        )
        shaped = np.fft.irfft(spec * band, n=length).astype(np.float32)
        t = np.arange(length, dtype=np.float32) / float(sample_rate)
        mod_freq = _rand_uniform(3.0, 5.5, rng)
        phase = _rand_uniform(0.0, 2.0 * np.pi, rng)
        env = 0.65 + 0.35 * np.sin(2.0 * np.pi * mod_freq * t + phase)
        return (shaped * env).astype(np.float32)

    # "hum" / low-frequency room HVAC noise
    band = 1.0 / (1.0 + (freqs / 250.0) ** 2)
    return np.fft.irfft(spec * band, n=length).astype(np.float32)


def apply_waveform_augmentation(
    waveform: np.ndarray,
    waveform_aug_config: Any,
    sample_rate: int = 16000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, bool]:
    """Applies Multi-Style Training (MTR) waveform augmentation (room reverb + moderate SNR noise).

    Strictly enforces a safe SNR floor (>= 8.0 dB) so noise is never too high.

    Returns:
        (augmented_waveform, was_augmented)
    """
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    w_enabled = _get_attr(
        waveform_aug_config,
        "use_waveform_aug",
        _get_attr(waveform_aug_config, "enabled", True),
    )
    if not w_enabled or waveform.size < 160:
        return waveform, False

    aug_prob = float(_get_attr(waveform_aug_config, "aug_prob", 0.4))
    if _rand_uniform(0.0, 1.0, rng) >= aug_prob:
        return waveform, False

    out = waveform.copy()
    # 1. Optional mild room reverberation (synthetic RIR)
    reverb_prob = float(_get_attr(waveform_aug_config, "reverb_prob", 0.35))
    if _rand_uniform(0.0, 1.0, rng) < reverb_prob:
        rt60_min = max(0.05, float(_get_attr(waveform_aug_config, "reverb_rt60_min", 0.08)))
        rt60_max = min(0.40, float(_get_attr(waveform_aug_config, "reverb_rt60_max", 0.28)))
        rt60 = _rand_uniform(rt60_min, max(rt60_min, rt60_max), rng)
        rir_len = min(int(sample_rate * rt60), max(64, waveform.size // 2))
        t_rir = np.arange(rir_len, dtype=np.float32) / float(sample_rate)
        decay = np.exp(-6.9078 * t_rir / max(rt60, 0.05))
        if rng is not None:
            rir_tail = rng.standard_normal(rir_len).astype(np.float32) * decay * 0.12
        else:
            rir_tail = np.random.randn(rir_len).astype(np.float32) * decay * 0.12
        rir = rir_tail
        rir[0] = 1.0
        rir = rir / (np.linalg.norm(rir) + 1e-8)
        n_fft = 1 << int(np.ceil(np.log2(out.size + rir_len - 1)))
        conv = np.fft.irfft(
            np.fft.rfft(out, n=n_fft) * np.fft.rfft(rir, n=n_fft),
            n=n_fft,
        )[: out.size].astype(np.float32)
        out = conv

    # 2. Additive noise at moderate SNR in [max(8.0, snr_min_db), snr_max_db]
    snr_min_db = max(8.0, float(_get_attr(waveform_aug_config, "snr_min_db", 10.0)))
    snr_max_db = max(snr_min_db + 1.0, float(_get_attr(waveform_aug_config, "snr_max_db", 20.0)))
    target_snr_db = _rand_uniform(snr_min_db, snr_max_db, rng)

    noise_types = tuple(
        _get_attr(waveform_aug_config, "noise_types", ("babble", "pink", "white", "hum"))
    )
    chosen_type = noise_types[_rand_int(0, len(noise_types) - 1, rng)]
    noise = _synthesize_noise_waveform(out.size, sample_rate, chosen_type, rng=rng)

    sig_rms = float(np.sqrt(np.mean(np.square(out)) + 1e-10))
    noise_rms = float(np.sqrt(np.mean(np.square(noise)) + 1e-10))
    desired_noise_rms = sig_rms * (10.0 ** (-target_snr_db / 20.0))
    out = out + noise * (desired_noise_rms / noise_rms)

    return out.astype(np.float32), True


def apply_feature_mtr_augmentation(
    features: np.ndarray,
    waveform_aug_config: Any,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, bool]:
    """Feature-domain "MTR" noise augmentation in the standardized log-Mel domain.

    This is NOT waveform-domain multi-style training with recorded noises and room
    impulse responses. It applies (i) with probability `reverb_prob` a one-tap
    temporal smoothing of adjacent frames and (ii) frequency-shaped Gaussian noise
    with amplitude `0.45 * 10^(-SNR/20)` for `SNR ~ U(snr_min_db, snr_max_db)`,
    sigmoid-gated towards low-energy time-frequency bins, followed by per-utterance
    mean re-centering.

    Returns:
        (augmented_features, was_augmented)
    """
    features = np.asarray(features, dtype=np.float32)
    w_enabled = _get_attr(
        waveform_aug_config,
        "use_waveform_aug",
        _get_attr(waveform_aug_config, "enabled", True),
    )
    if not w_enabled or features.shape[0] < 4:
        return features, False

    aug_prob = float(_get_attr(waveform_aug_config, "aug_prob", 0.35))
    if _rand_uniform(0.0, 1.0, rng) >= aug_prob:
        return features, False

    seq_len, feat_dim = features.shape
    out = features.copy()

    # 1. Optional mild room reverberation (temporal RIR smoothing)
    reverb_prob = float(_get_attr(waveform_aug_config, "reverb_prob", 0.35))
    if _rand_uniform(0.0, 1.0, rng) < reverb_prob:
        alpha_rev = _rand_uniform(0.04, 0.12, rng)
        reverb = out.copy()
        reverb[1:] = (1.0 - alpha_rev) * out[1:] + alpha_rev * out[:-1]
        out = reverb

    # 2. Additive noise at an SNR sampled from the *configured* range (no hidden clamp).
    snr_min_db = float(_get_attr(waveform_aug_config, "snr_min_db", 12.0))
    snr_max_db = max(
        snr_min_db, float(_get_attr(waveform_aug_config, "snr_max_db", 20.0))
    )
    target_snr_db = _rand_uniform(snr_min_db, snr_max_db, rng)
    noise_scale = 0.45 * (10.0 ** (-target_snr_db / 20.0))

    # Generate frequency-shaped Mel noise profile (babble / pink / white)
    mel_idx = np.linspace(0.0, 1.0, feat_dim, dtype=np.float32)
    mode_idx = _rand_int(0, 2, rng)
    if mode_idx == 0:
        freq_shape = np.exp(-0.5 * ((mel_idx - 0.35) / 0.25) ** 2) + 0.2
    elif mode_idx == 1:
        freq_shape = 1.0 / (0.3 + mel_idx)
    else:
        freq_shape = np.ones_like(mel_idx)
    freq_shape = freq_shape / (np.mean(freq_shape) + 1e-6)

    if rng is not None:
        raw_noise = rng.standard_normal((seq_len, feat_dim)).astype(np.float32)
    else:
        raw_noise = np.random.randn(seq_len, feat_dim).astype(np.float32)
    noise = raw_noise * freq_shape[None, :]

    # Additive background noise affects lower-energy time-frequency bins more strongly
    gate = 1.0 / (1.0 + np.exp(np.clip(out, -4.0, 4.0)))
    out = out + noise_scale * gate * noise
    out = out - np.mean(out, axis=0, keepdims=True)
    return out.astype(np.float32), True


def augment_features_mutually_exclusive(
    features: np.ndarray,
    specaug_config: Optional[Any] = None,
    waveform_aug_config: Optional[Any] = None,
    is_already_mtr: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, str]:
    """Applies either MTR augmentation OR SpecAugment (never both on the same utterance).

    Mutually exclusive augmentation (a FlaxSpeaker design choice):
    1. If the utterance is already MTR-augmented (`is_already_mtr=True`), SpecAugment is
       strictly skipped and `(features, "mtr")` is returned.
    2. Otherwise, if `waveform_aug_config` is enabled, MTR is sampled with probability
       `waveform_aug_config.aug_prob`. If applied, SpecAugment is strictly skipped and
       `(mtr_features, "mtr")` is returned.
    3. Otherwise (the utterance is clean), if `specaug_config` is enabled, SpecAugment
       is applied with probability `specaug_config.apply_prob` (returning `"specaug"`).
    4. Otherwise `(features, "clean")` is returned.
    """
    if is_already_mtr:
        return np.asarray(features, dtype=np.float32), "mtr"

    w_enabled = _get_attr(
        waveform_aug_config,
        "use_waveform_aug",
        _get_attr(waveform_aug_config, "enabled", False),
    )
    if waveform_aug_config is not None and w_enabled:
        mtr_feats, did_mtr = apply_feature_mtr_augmentation(
            features, waveform_aug_config, rng=rng
        )
        if did_mtr:
            return mtr_feats, "mtr"

    if specaug_config is not None and _get_attr(
        specaug_config, "use_specaug", False
    ):
        apply_prob = float(_get_attr(specaug_config, "apply_prob", 0.5))
        if _rand_uniform(0.0, 1.0, rng) < apply_prob:
            return apply_specaug(features, specaug_config, rng=rng), "specaug"

    return np.asarray(features, dtype=np.float32), "clean"


def apply_mutually_exclusive_augmentation(
    features: np.ndarray,
    waveform_aug_config: Optional[Any] = None,
    specaug_config: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, bool, bool]:
    """Convenience wrapper returning `(augmented_features, applied_mtr, applied_specaug)`."""
    out, mode = augment_features_mutually_exclusive(
        features,
        specaug_config=specaug_config,
        waveform_aug_config=waveform_aug_config,
        rng=rng,
    )
    return out, (mode == "mtr"), (mode == "specaug")


def apply_specaug_batch_jax(
    batch_features: jax.Array,
    rng: jax.Array,
    freq_mask_max_width: int = 6,
    time_mask_max_width: int = 10,
    mask_prob: float = 0.3,
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
