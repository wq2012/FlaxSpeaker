"""Corpus-backed waveform augmentation (MUSAN, RIRS_NOISES, and speed perturbation).

Implements standard speaker recognition Multi-Style Training (MTR) augmentation
(Snyder et al., x-vectors, ICASSP 2018; Desplanques et al., ECAPA-TDNN, Interspeech 2020;
Wang et al., WeSpeaker, ICASSP 2023):
1. Reverberation via FFT convolution with recorded/simulated room impulse responses
   (`RIRS_NOISES`, Ko et al., ICASSP 2017), peak-aligned to prevent latency drift.
2. Additive environmental noise (`MUSAN/noise`, `0-15` dB SNR).
3. Additive background music (`MUSAN/music`, `5-15` dB SNR).
4. Multi-speaker babble (`MUSAN/speech`, `3-7` speakers summed at `13-20` dB SNR).
5. Polyphase 3-way speed perturbation (`0.9x`, `1.0x`, `1.1x`) expanding speaker
   identities `3x` for margin-based classification objectives.
"""

from __future__ import annotations

from fractions import Fraction
import os
from typing import Any, Optional

import numpy as np
from scipy import signal
import soundfile as sf


def _rms_db(waveform: np.ndarray, eps: float = 1e-8) -> float:
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)) + eps))
    return 20.0 * float(np.log10(max(rms, eps)))


def apply_speed_perturbation(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    speed_factor: float = 1.0,
) -> np.ndarray:
    """Applies polyphase resampling speed perturbation to a 1D float32 waveform.

    A `speed_factor < 1.0` (e.g. `0.9`) stretches duration and lowers pitch;
    `speed_factor > 1.0` (e.g. `1.1`) compresses duration and raises pitch.
    Following WeSpeaker / Kaldi conventions, speed-perturbed copies with factors
    `!= 1.0` are treated as distinct speaker classes during classification training.
    """
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wav.size == 0 or abs(float(speed_factor) - 1.0) < 1e-6:
        return wav
    # Output length is len(wav) / speed_factor -> resample_poly(up, down) has ratio up/down = 1/speed_factor
    frac = Fraction(1.0 / float(speed_factor)).limit_denominator(20)
    resampled = signal.resample_poly(wav, up=frac.numerator, down=frac.denominator)
    return np.asarray(resampled, dtype=np.float32)


def _scan_wav_files(root_dir: str) -> list[str]:
    if not root_dir or not os.path.isdir(root_dir):
        return []
    found: list[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith((".wav", ".flac")):
                found.append(os.path.join(dirpath, fn))
    found.sort()
    return found


def _load_mono_16k(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        frac = Fraction(target_sr, sr).limit_denominator(200)
        wav = signal.resample_poly(wav, up=frac.numerator, down=frac.denominator)
    return np.asarray(wav, dtype=np.float32)


def _load_random_segment_16k(
    path: str,
    target_len: int,
    rng: np.random.Generator,
    target_sr: int = 16000,
) -> np.ndarray:
    """Reads only a random `target_len` segment via `libsndfile` frame seeking."""
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    info = sf.info(path)
    sr = int(info.samplerate)
    total_frames = int(info.frames)
    need_src = int(np.ceil(target_len * float(sr) / float(target_sr))) + 32
    if total_frames > need_src:
        start = int(rng.integers(0, total_frames - need_src + 1))
        wav, _ = sf.read(path, start=start, frames=need_src, dtype="float32")
    else:
        wav, _ = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    if sr != target_sr and wav.size > 0:
        frac = Fraction(target_sr, sr).limit_denominator(200)
        wav = signal.resample_poly(wav, up=frac.numerator, down=frac.denominator)
    return _fit_clip_to_length(np.asarray(wav, dtype=np.float32), target_len, rng)


def _fit_clip_to_length(
    clip: np.ndarray, target_len: int, rng: np.random.Generator
) -> np.ndarray:
    if clip.size == 0:
        return np.zeros((target_len,), dtype=np.float32)
    if clip.shape[0] < target_len:
        reps = (target_len // clip.shape[0]) + 1
        clip = np.tile(clip, reps)
    if clip.shape[0] > target_len:
        start = int(rng.integers(0, clip.shape[0] - target_len + 1))
        return clip[start:start + target_len].astype(np.float32, copy=False)
    return clip[:target_len].astype(np.float32, copy=False)


def mix_at_snr_db(
    clean: np.ndarray,
    interferer: np.ndarray,
    snr_db: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Mixes `interferer` into `clean` at exact `snr_db` dB."""
    clean_db = _rms_db(clean, eps=eps)
    noise_db = _rms_db(interferer, eps=eps)
    scale = 10.0 ** ((clean_db - noise_db - float(snr_db)) / 20.0)
    mixed = clean + float(scale) * interferer
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.999:
        mixed = mixed * (0.999 / peak)
    return np.asarray(mixed, dtype=np.float32)


def apply_rir_convolution(
    waveform: np.ndarray, rir: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Convolves `waveform` with room impulse response `rir`, aligned to direct-path peak."""
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    h = np.asarray(rir, dtype=np.float32).reshape(-1)
    if wav.size == 0 or h.size == 0:
        return wav
    peak_idx = int(np.argmax(np.abs(h)))
    # Keep from just before direct-path impulse so output stays frame-aligned
    lead = max(0, peak_idx - 4)
    h = h[lead:]
    norm = float(np.sqrt(np.sum(np.square(h, dtype=np.float64)) + eps))
    h = h / norm
    rev = signal.fftconvolve(wav, h, mode="full")[: wav.shape[0]]
    # Match clean RMS level
    orig_rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64)) + eps))
    rev_rms = float(np.sqrt(np.mean(np.square(rev, dtype=np.float64)) + eps))
    rev = rev * (orig_rms / max(rev_rms, eps))
    return np.asarray(rev, dtype=np.float32)


class MusanRirAugmenter:
    """Corpus-backed waveform augmenter indexing MUSAN and RIRS_NOISES."""

    def __init__(
        self,
        musan_dir: str = "",
        rirs_dir: str = "",
        sample_rate: int = 16000,
        noise_snr_range: tuple[float, float] = (0.0, 15.0),
        music_snr_range: tuple[float, float] = (5.0, 15.0),
        babble_snr_range: tuple[float, float] = (13.0, 20.0),
        babble_spk_range: tuple[int, int] = (3, 7),
    ):
        self.sample_rate = int(sample_rate)
        self.noise_snr_range = (float(noise_snr_range[0]), float(noise_snr_range[1]))
        self.music_snr_range = (float(music_snr_range[0]), float(music_snr_range[1]))
        self.babble_snr_range = (float(babble_snr_range[0]), float(babble_snr_range[1]))
        self.babble_spk_range = (int(babble_spk_range[0]), int(babble_spk_range[1]))

        musan_root = os.path.expanduser(musan_dir) if musan_dir else ""
        if musan_root and os.path.isdir(os.path.join(musan_root, "musan")):
            musan_root = os.path.join(musan_root, "musan")
        self.noise_files = _scan_wav_files(os.path.join(musan_root, "noise")) if musan_root else []
        self.music_files = _scan_wav_files(os.path.join(musan_root, "music")) if musan_root else []
        self.speech_files = _scan_wav_files(os.path.join(musan_root, "speech")) if musan_root else []

        rirs_root = os.path.expanduser(rirs_dir) if rirs_dir else ""
        if rirs_root and not os.path.isdir(rirs_root):
            alt_upper = os.path.join(os.path.dirname(rirs_root), "RIRS_NOISES")
            if os.path.isdir(alt_upper):
                rirs_root = alt_upper
        if rirs_root and os.path.isdir(os.path.join(rirs_root, "RIRS_NOISES")):
            rirs_root = os.path.join(rirs_root, "RIRS_NOISES")
        self.rir_files: list[str] = []
        if rirs_root:
            for sub in ("simulated_rirs", "real_rirs_isotropic_noises"):
                sub_p = os.path.join(rirs_root, sub)
                if os.path.isdir(sub_p):
                    for p in _scan_wav_files(sub_p):
                        # Exclude multi-channel isotropic background noise files from RIR list
                        bname = os.path.basename(p).lower()
                        if "noise" not in bname:
                            self.rir_files.append(p)
            if not self.rir_files:
                self.rir_files = _scan_wav_files(rirs_root)

        self._cached_rirs: list[np.ndarray] = []
        self._cached_noise: list[np.ndarray] = []
        self._cached_music: list[np.ndarray] = []
        self._cached_speech: list[np.ndarray] = []

    def preload_cache(
        self,
        max_rirs: int = 256,
        max_noise: int = 128,
        max_music: int = 128,
        max_speech: int = 256,
        max_clip_sec: float = 6.5,
        max_rir_sec: float = 0.5,
        seed: int = 202605,
    ) -> None:
        """Preloads a stratified bank of RIR and MUSAN clips into RAM for zero-I/O augmentation."""
        rng = np.random.default_rng(seed)
        clip_samples = int(round(max_clip_sec * self.sample_rate))
        rir_samples = int(round(max_rir_sec * self.sample_rate))

        def _pick(files: list[str], k: int) -> list[str]:
            if len(files) <= k:
                return list(files)
            idxs = np.linspace(0, len(files) - 1, num=k, dtype=np.int64)
            return [files[int(i)] for i in idxs]

        self._cached_rirs = []
        for p in _pick(self.rir_files, max_rirs):
            try:
                h = _load_mono_16k(p, self.sample_rate).reshape(-1)
                if h.size > 0:
                    peak_idx = int(np.argmax(np.abs(h)))
                    lead = max(0, peak_idx - 4)
                    h = h[lead:lead + rir_samples]
                    norm = float(np.sqrt(np.sum(np.square(h, dtype=np.float64)) + 1e-8))
                    self._cached_rirs.append(np.asarray(h / norm, dtype=np.float32))
            except Exception:
                continue

        self._cached_noise = []
        for p in _pick(self.noise_files, max_noise):
            try:
                c = _load_random_segment_16k(p, clip_samples, rng, self.sample_rate)
                if c.size > 0:
                    self._cached_noise.append(c)
            except Exception:
                continue

        self._cached_music = []
        for p in _pick(self.music_files, max_music):
            try:
                c = _load_random_segment_16k(p, clip_samples, rng, self.sample_rate)
                if c.size > 0:
                    self._cached_music.append(c)
            except Exception:
                continue

        self._cached_speech = []
        for p in _pick(self.speech_files, max_speech):
            try:
                c = _load_random_segment_16k(p, clip_samples, rng, self.sample_rate)
                if c.size > 0:
                    std = max(float(np.std(c)), 1e-6)
                    self._cached_speech.append(np.asarray(c / std, dtype=np.float32))
            except Exception:
                continue

    @property
    def available_modes(self) -> list[str]:
        modes: list[str] = []
        has_rir = bool(self.rir_files or self._cached_rirs)
        has_noise = bool(self.noise_files or self._cached_noise)
        has_music = bool(self.music_files or self._cached_music)
        n_speech = max(len(self.speech_files), len(self._cached_speech))
        if has_rir:
            modes.append("reverb")
        if has_noise:
            modes.append("noise")
        if has_music:
            modes.append("music")
        if n_speech >= max(1, self.babble_spk_range[0]):
            modes.append("babble")
        if has_rir and (has_noise or has_music):
            modes.append("reverb_noise")
        return modes

    @property
    def is_ready(self) -> bool:
        return bool(self.available_modes)

    def _sample_rir(self, gen: np.random.Generator) -> np.ndarray:
        if self._cached_rirs:
            return self._cached_rirs[int(gen.integers(0, len(self._cached_rirs)))]
        rir_p = self.rir_files[int(gen.integers(0, len(self.rir_files)))]
        return _load_mono_16k(rir_p, self.sample_rate)

    def _sample_clip(
        self,
        cached_list: list[np.ndarray],
        file_list: list[str],
        target_len: int,
        gen: np.random.Generator,
    ) -> np.ndarray:
        if cached_list:
            clip = cached_list[int(gen.integers(0, len(cached_list)))]
            return _fit_clip_to_length(clip, target_len, gen)
        p = file_list[int(gen.integers(0, len(file_list)))]
        return _load_random_segment_16k(p, target_len, gen, self.sample_rate)

    def augment(
        self,
        waveform: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        mode: Optional[str] = None,
    ) -> tuple[np.ndarray, str]:
        """Applies one randomly chosen (or explicit) corpus augmentation mode."""
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if wav.size == 0 or not self.is_ready:
            return wav, "clean"
        gen = rng if rng is not None else np.random.default_rng()
        modes = self.available_modes
        chosen = mode if (mode in modes) else modes[int(gen.integers(0, len(modes)))]
        target_len = wav.shape[0]

        if chosen == "reverb":
            rir = self._sample_rir(gen)
            return apply_rir_convolution(wav, rir), "reverb"

        if chosen == "noise":
            noise = self._sample_clip(
                self._cached_noise, self.noise_files, target_len, gen
            )
            snr = float(gen.uniform(*self.noise_snr_range))
            return mix_at_snr_db(wav, noise, snr), "noise"

        if chosen == "music":
            music = self._sample_clip(
                self._cached_music, self.music_files, target_len, gen
            )
            snr = float(gen.uniform(*self.music_snr_range))
            return mix_at_snr_db(wav, music, snr), "music"

        if chosen == "babble":
            n_avail = (
                len(self._cached_speech)
                if self._cached_speech
                else len(self.speech_files)
            )
            k_min = min(self.babble_spk_range[0], n_avail)
            k_max = min(self.babble_spk_range[1], n_avail)
            n_babble = int(gen.integers(k_min, k_max + 1))
            idxs = gen.choice(n_avail, size=n_babble, replace=False)
            babble = np.zeros((target_len,), dtype=np.float32)
            for idx in idxs:
                if self._cached_speech:
                    babble += _fit_clip_to_length(
                        self._cached_speech[int(idx)], target_len, gen
                    )
                else:
                    clip = _load_random_segment_16k(
                        self.speech_files[int(idx)], target_len, gen, self.sample_rate
                    )
                    babble += clip / max(float(np.std(clip)), 1e-6)
            snr = float(gen.uniform(*self.babble_snr_range))
            return mix_at_snr_db(wav, babble, snr), "babble"

        if chosen == "reverb_noise":
            rev = apply_rir_convolution(wav, self._sample_rir(gen))
            use_noise = bool(self._cached_noise or self.noise_files)
            c_pool = self._cached_noise if use_noise else self._cached_music
            f_pool = self.noise_files if use_noise else self.music_files
            snr_range = self.noise_snr_range if use_noise else self.music_snr_range
            bg = self._sample_clip(c_pool, f_pool, target_len, gen)
            snr = float(gen.uniform(*snr_range))
            return mix_at_snr_db(rev, bg, snr), "reverb_noise"

        return wav, "clean"

    @classmethod
    def from_config(cls, aug_cfg: Any, sample_rate: int = 16000) -> "MusanRirAugmenter":
        return cls(
            musan_dir=str(getattr(aug_cfg, "musan_dir", "")),
            rirs_dir=str(getattr(aug_cfg, "rirs_dir", "")),
            sample_rate=sample_rate,
            noise_snr_range=tuple(getattr(aug_cfg, "noise_snr_range", (0.0, 15.0))),
            music_snr_range=tuple(getattr(aug_cfg, "music_snr_range", (5.0, 15.0))),
            babble_snr_range=tuple(getattr(aug_cfg, "babble_snr_range", (13.0, 20.0))),
            babble_spk_range=tuple(getattr(aug_cfg, "babble_spk_range", (3, 7))),
        )
