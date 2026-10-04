"""Voice Activity Detection (VAD) for audio waveforms and acoustic features."""

from __future__ import annotations

from typing import Optional
import numpy as np

from flaxspeaker import configs


def compute_frame_energy_db(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    frame_size_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """Computes frame-level log RMS energy in dB from a 1D waveform."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    frame_length = max(1, int(round(sample_rate * frame_size_ms / 1000.0)))
    frame_step = max(1, int(round(sample_rate * frame_step_ms / 1000.0)))

    if waveform.shape[0] < frame_length:
        rms = np.sqrt(np.mean(np.square(waveform)) + eps)
        return np.array([20.0 * np.log10(max(float(rms), eps))], dtype=np.float32)

    num_frames = 1 + (waveform.shape[0] - frame_length) // frame_step
    strides = (waveform.strides[0] * frame_step, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        waveform, shape=(num_frames, frame_length), strides=strides
    )
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + eps)
    return 20.0 * np.log10(np.maximum(rms, eps)).astype(np.float32)


def compute_spectral_flatness_and_band_ratio(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    frame_size_ms: float = 25.0,
    frame_step_ms: float = 10.0,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes frame-level spectral flatness and speech-band (300-3400Hz) energy ratio."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    frame_length = max(1, int(round(sample_rate * frame_size_ms / 1000.0)))
    frame_step = max(1, int(round(sample_rate * frame_step_ms / 1000.0)))

    if waveform.shape[0] < frame_length:
        waveform = np.pad(waveform, (0, frame_length - waveform.shape[0]))

    num_frames = 1 + (waveform.shape[0] - frame_length) // frame_step
    strides = (waveform.strides[0] * frame_step, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        waveform, shape=(num_frames, frame_length), strides=strides
    )
    window = np.hanning(frame_length).astype(np.float32)
    windowed = frames * window[None, :]
    n_fft = 512
    while n_fft < frame_length:
        n_fft *= 2
    power_spec = np.square(np.abs(np.fft.rfft(windowed, n=n_fft, axis=1))) + eps

    geom_mean = np.exp(np.mean(np.log(power_spec), axis=1))
    arith_mean = np.mean(power_spec, axis=1)
    flatness = (geom_mean / (arith_mean + eps)).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    speech_bins = (freqs >= 300.0) & (freqs <= 3400.0)
    band_ratio = (
        np.sum(power_spec[:, speech_bins], axis=1) / (np.sum(power_spec, axis=1) + eps)
    ).astype(np.float32)
    return flatness, band_ratio


def smooth_vad_mask(
    raw_mask: np.ndarray,
    min_speech_frames: int = 3,
    hangover_frames: int = 5,
) -> np.ndarray:
    """Applies minimum speech duration filtering and hangover smoothing to a boolean VAD mask."""
    mask = np.asarray(raw_mask, dtype=bool).copy()
    n = mask.shape[0]
    if n == 0:
        return mask

    # 1. Remove isolated short spikes (< min_speech_frames)
    if min_speech_frames > 1:
        i = 0
        while i < n:
            if mask[i]:
                j = i
                while j < n and mask[j]:
                    j += 1
                if (j - i) < min_speech_frames:
                    mask[i:j] = False
                i = j
            else:
                i += 1

    # 2. Apply hangover smoothing after active speech segments
    if hangover_frames > 0:
        out = mask.copy()
        countdown = 0
        for t in range(n):
            if mask[t]:
                out[t] = True
                countdown = hangover_frames
            elif countdown > 0:
                out[t] = True
                countdown -= 1
        mask = out

    return mask


class VadProcessor:
    """Configurable Voice Activity Detector operating on waveforms or feature matrices."""

    def __init__(
        self,
        config: Optional[configs.VadConfig] = None,
        sample_rate: Optional[int] = None,
    ):
        self.config = config or configs.VadConfig(enabled=True)
        if sample_rate is not None:
            self.config.sample_rate = sample_rate

    def detect_frames_from_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Returns a boolean speech mask of shape (num_frames,) for a 1D waveform."""
        cfg = self.config
        energy_db = compute_frame_energy_db(
            waveform,
            sample_rate=cfg.sample_rate,
            frame_size_ms=cfg.frame_size_ms,
            frame_step_ms=cfg.frame_step_ms,
        )
        mode_str = str(cfg.mode).lower()
        if mode_str == "none":
            return np.ones(energy_db.shape[0], dtype=bool)

        noise_floor = np.percentile(energy_db, cfg.adaptive_percentile)
        dynamic_thresh = max(
            cfg.energy_threshold_db, float(noise_floor + cfg.snr_margin_db)
        )
        energy_mask = energy_db >= dynamic_thresh

        if mode_str == "energy":
            raw_mask = energy_mask
        elif mode_str in ("spectral", "hybrid"):
            flatness, band_ratio = compute_spectral_flatness_and_band_ratio(
                waveform,
                sample_rate=cfg.sample_rate,
                frame_size_ms=cfg.frame_size_ms,
                frame_step_ms=cfg.frame_step_ms,
            )
            spectral_mask = (flatness <= cfg.spectral_flatness_threshold) & (
                band_ratio >= 0.25
            )
            raw_mask = (
                (energy_mask & spectral_mask)
                if cfg.mode == "hybrid"
                else (energy_mask | spectral_mask)
            )
        else:
            raise ValueError(f"Unsupported VAD mode: {cfg.mode}")

        mask = smooth_vad_mask(
            raw_mask,
            min_speech_frames=cfg.min_speech_frames,
            hangover_frames=cfg.hangover_frames,
        )

        # Ensure at least a few frames survive even for very quiet recordings
        if not np.any(mask):
            top_k = min(max(cfg.min_speech_frames, 5), energy_db.shape[0])
            top_indices = np.argsort(energy_db)[-top_k:]
            mask[top_indices] = True

        return mask

    def filter_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Removes non-speech regions from a 1D waveform."""
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not self.config.enabled or waveform.size == 0:
            return waveform

        cfg = self.config
        mask = self.detect_frames_from_waveform(waveform)
        frame_step = max(1, int(round(cfg.sample_rate * cfg.frame_step_ms / 1000.0)))
        frame_length = max(1, int(round(cfg.sample_rate * cfg.frame_size_ms / 1000.0)))

        sample_mask = np.zeros(waveform.shape[0], dtype=bool)
        for idx, is_speech in enumerate(mask):
            if is_speech:
                start = idx * frame_step
                end = min(start + frame_length, waveform.shape[0])
                sample_mask[start:end] = True

        if not np.any(sample_mask):
            return waveform
        return waveform[sample_mask]

    def filter_features(
        self,
        features: np.ndarray,
        waveform: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Filters frame-level acoustic features of shape (T, D) using VAD."""
        features = np.asarray(features, dtype=np.float32)
        if not self.config.enabled or features.shape[0] <= 4:
            return features

        if waveform is not None:
            mask = self.detect_frames_from_waveform(waveform)
            if mask.shape[0] != features.shape[0]:
                # Align mask length to feature length via nearest-neighbor interpolation
                indices = np.linspace(
                    0, mask.shape[0] - 1, features.shape[0]
                ).astype(int)
                mask = mask[indices]
        else:
            # Estimate frame energy directly from feature vectors (mean across bins)
            frame_energy = np.mean(features, axis=1)
            noise_floor = np.percentile(
                frame_energy, self.config.adaptive_percentile
            )
            std_energy = float(np.std(frame_energy)) + 1e-6
            raw_mask = frame_energy >= (noise_floor + 0.25 * std_energy)
            mask = smooth_vad_mask(
                raw_mask,
                min_speech_frames=self.config.min_speech_frames,
                hangover_frames=self.config.hangover_frames,
            )
            if not np.any(mask):
                return features

        filtered = features[mask]
        return filtered if filtered.shape[0] > 0 else features
