"""Audio frontend, feature extraction, and batch builders for FlaxSpeaker."""

from __future__ import annotations

import functools
from multiprocessing import pool as mp_pool
import os
import random
from typing import Any, Optional
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import soundfile as sf

from flaxspeaker import configs
from flaxspeaker import dataset
from flaxspeaker import specaug
from flaxspeaker import vad

SAMPLE_RATE = 16000


def load_waveform(audio_file: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Loads an audio file as a mono float32 waveform at `target_sr` Hz."""
    waveform, sample_rate = sf.read(audio_file, dtype="float32")
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    if sample_rate != target_sr:
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=target_sr
        )
    return np.asarray(waveform, dtype=np.float32)


def stack_and_subsample_frames(
    features: np.ndarray,
    left_context: int = 0,
    right_context: int = 0,
    frame_stride: int = 1,
) -> np.ndarray:
    """Stacks neighboring context frames and subsamples along the time axis."""
    if left_context == 0 and right_context == 0 and frame_stride == 1:
        return features
    context_size = 1 + left_context + right_context
    padded = np.pad(
        features,
        pad_width=((left_context, right_context), (0, 0)),
        mode="edge",
    )
    num_frames = features.shape[0]
    slices = [
        padded[offset:offset + num_frames, :] for offset in range(context_size)
    ]
    stacked = np.concatenate(slices, axis=-1)
    if frame_stride > 1:
        stacked = stacked[::frame_stride, :]
    return stacked


class AudioFrontend:
    """Modernized audio feature extractor compatible with HuggingFace Transformers and WeSpeaker."""

    def __init__(
        self,
        frontend_config: Optional[configs.FrontendConfig] = None,
        vad_config: Optional[configs.VadConfig] = None,
    ):
        self.config = frontend_config or configs.FrontendConfig()
        self.vad_config = vad_config
        self.vad_processor = (
            vad.VadProcessor(vad_config)
            if (vad_config and vad_config.enabled)
            else None
        )
        self._frame_length = max(
            1,
            int(
                round(
                    self.config.sample_rate * self.config.frame_size_ms / 1000.0
                )
            ),
        )
        self._frame_step = max(
            1,
            int(
                round(
                    self.config.sample_rate * self.config.frame_step_ms / 1000.0
                )
            ),
        )
        self._n_fft = max(self.config.n_fft, self._frame_length)
        self._mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self._n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            htk=False,
            norm="slaney",
        ).astype(np.float32)
        self._hf_extractor = None

    def _get_hf_extractor(self):
        if self._hf_extractor is None:
            from transformers import AutoFeatureExtractor

            self._hf_extractor = AutoFeatureExtractor.from_pretrained(
                self.config.hf_extractor_name
            )
        return self._hf_extractor

    def extract_from_waveform(
        self,
        waveform: np.ndarray,
        waveform_aug_config: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Extracts 2D acoustic features (T, D) from a 1D waveform.

        Applies AGC peak normalization first, extracts STFT/filterbank features
        on the continuous waveform (avoiding splice boundary clicks), filters
        non-speech frames via frame-level VAD, and then applies utterance CMVN
        strictly over active speech frames.
        """
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size > 0:
            vad_cfg = (
                self.vad_processor.config
                if self.vad_processor is not None
                else self.vad_config
            )
            apply_agc = (
                True
                if vad_cfg is None
                else bool(getattr(vad_cfg, "apply_agc", True))
            )
            if apply_agc:
                peak = float(np.max(np.abs(waveform)))
                if peak > 1e-8:
                    waveform = waveform / peak

        clean_waveform = waveform
        if waveform_aug_config is not None:
            waveform, _ = specaug.apply_waveform_augmentation(
                waveform,
                waveform_aug_config,
                sample_rate=self.config.sample_rate,
                rng=rng,
            )

        cfg = self.config
        ftype = str(cfg.feature_type).lower()

        if ftype == configs.FeatureType.MFCC.value:
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=cfg.sample_rate,
                n_mfcc=cfg.n_mfcc,
            )
            features = mfcc.T.astype(np.float32)

        elif ftype in (
            configs.FeatureType.LOG_MEL.value,
            configs.FeatureType.WHISPER_MEL.value,
        ):
            proc_wav = waveform
            if cfg.preemph > 0.0 and proc_wav.shape[0] > 1:
                proc_wav = np.concatenate(
                    [
                        proc_wav[:1],
                        proc_wav[1:] - cfg.preemph * proc_wav[:-1],
                    ]
                )
            if proc_wav.shape[0] < self._frame_length:
                proc_wav = np.pad(
                    proc_wav, (0, self._frame_length - proc_wav.shape[0])
                )

            stft = librosa.stft(
                proc_wav,
                n_fft=self._n_fft,
                hop_length=self._frame_step,
                win_length=self._frame_length,
                window="hann",
                center=True,
            )
            power_spec = np.square(np.abs(stft)).astype(np.float32)
            mel_spec = np.matmul(self._mel_basis, power_spec)

            if ftype == configs.FeatureType.WHISPER_MEL.value:
                # HuggingFace WhisperFeatureExtractor compatible log10 scaling
                log_spec = np.log10(np.maximum(mel_spec, 1e-10))
                log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
                features = ((log_spec + 4.0) / 4.0).T.astype(np.float32)
            else:
                features = np.log(mel_spec + cfg.log_offset).T.astype(np.float32)

        elif ftype == configs.FeatureType.HF_COMPAT.value:
            extractor = self._get_hf_extractor()
            out = extractor(
                waveform,
                sampling_rate=cfg.sample_rate,
                return_tensors="np",
            )
            key = (
                "input_features"
                if "input_features" in out
                else list(out.keys())[0]
            )
            arr = np.asarray(out[key], dtype=np.float32)[0]
            # HF Whisper outputs (n_mels, T); transpose to (T, n_mels)
            if arr.shape[0] == cfg.n_mels and arr.shape[1] != cfg.n_mels:
                arr = arr.T
            features = arr
        else:
            raise ValueError(f"Unsupported feature_type: {ftype}")

        # Apply frame-level VAD after STFT extraction and before CMVN
        if self.vad_processor is not None and self.vad_processor.config.enabled:
            features = self.vad_processor.filter_features(
                features, waveform=clean_waveform, pre_normalized=True
            )

        if cfg.apply_cmvn and ftype != configs.FeatureType.WHISPER_MEL.value:
            mean = np.mean(features, axis=0, keepdims=True)
            features = features - mean
            if cfg.cmvn_norm_vars:
                std = np.std(features, axis=0, keepdims=True) + 1e-6
                features = features / std

        features = stack_and_subsample_frames(
            features,
            left_context=cfg.stack_left_context,
            right_context=cfg.stack_right_context,
            frame_stride=cfg.frame_stride,
        )
        return features.astype(np.float32)

    def extract_from_file(self, audio_file: str) -> np.ndarray:
        """Extracts 2D features (T, D) from an audio file path."""
        waveform = load_waveform(audio_file, target_sr=self.config.sample_rate)
        return self.extract_from_waveform(waveform)

    def to_hf_feature_dict(
        self, waveform: np.ndarray, max_frames: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Returns a HuggingFace Transformers compatible dict with `input_features` and `attention_mask`."""
        feats = self.extract_from_waveform(waveform)
        num_frames, dim = feats.shape
        if max_frames is not None:
            if num_frames >= max_frames:
                feats = feats[:max_frames]
                mask = np.ones((1, max_frames), dtype=np.int32)
            else:
                pad_len = max_frames - num_frames
                feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="constant")
                mask = np.zeros((1, max_frames), dtype=np.int32)
                mask[0, :num_frames] = 1
        else:
            mask = np.ones((1, num_frames), dtype=np.int32)
        return {
            "input_features": np.expand_dims(feats, axis=0),
            "attention_mask": mask,
        }


def extract_features(
    audio_file: str,
    n_mfcc: int = 128,
    frontend: Optional[AudioFrontend] = None,
) -> np.ndarray:
    """Extract acoustic features from an audio file, shape=(TIME, FEAT_DIM).

    Preserves exact backward compatibility with legacy MFCC extraction when
    `frontend` is not passed, while supporting `AudioFrontend` for log-Mel /
    Whisper-Mel / VAD pipelines.
    """
    if frontend is not None:
        return frontend.extract_from_file(audio_file)

    waveform, sample_rate = sf.read(audio_file)
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE
        )
    features = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
    return features.transpose().astype(np.float32)


def extract_sliding_windows(
    features: np.ndarray,
    myconfig: Any,
) -> list[np.ndarray]:
    """Extract sliding windows from features."""
    seq_len = int(myconfig.model.seq_len)
    step = int(myconfig.model.sliding_window_step)
    sliding_windows = []
    start = 0
    while start + seq_len <= features.shape[0]:
        sliding_windows.append(features[start:start + seq_len, :])
        start += step
    return sliding_windows


def get_triplet_features(
    spk_to_utts: dataset.SpkToUtts,
    n_mfcc: int,
    frontend: Optional[AudioFrontend] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get a triplet of anchor/pos/neg features."""
    anchor_utt, pos_utt, neg_utt = dataset.get_triplet(spk_to_utts)
    return (
        extract_features(anchor_utt, n_mfcc, frontend=frontend),
        extract_features(pos_utt, n_mfcc, frontend=frontend),
        extract_features(neg_utt, n_mfcc, frontend=frontend),
    )


def pad_or_trim_features(
    features: np.ndarray,
    seq_len: int,
    specaug_config: Optional[Any] = None,
    random_crop: bool = True,
    waveform_aug_config: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Pads (via tiling/edge) or crops features to exact `seq_len` frames."""
    full_length = features.shape[0]
    if full_length < seq_len:
        repeats = (seq_len // max(1, full_length)) + 1
        features = np.tile(features, (repeats, 1))
        full_length = features.shape[0]

    if random_crop and full_length > seq_len:
        if rng is not None:
            start = int(rng.integers(0, full_length - seq_len + 1))
        else:
            start = random.randint(0, full_length - seq_len)
    else:
        start = (full_length - seq_len) // 2
    trimmed = features[start:start + seq_len, :].copy()
    trimmed, _ = specaug.augment_features_mutually_exclusive(
        trimmed,
        specaug_config=specaug_config,
        waveform_aug_config=waveform_aug_config,
        rng=rng,
    )
    return trimmed


def sample_variable_length_features(
    features: np.ndarray,
    max_seq_len: int,
    min_seq_len: Optional[int] = None,
    variable_length: bool = True,
    specaug_config: Optional[Any] = None,
    waveform_aug_config: Optional[Any] = None,
    random_crop: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Samples a variable-length segment in `[min_seq_len, max_seq_len]`,
    applies mutually exclusive augmentation, and pads with mask.

    Returns:
        padded_features: np.ndarray of shape (max_seq_len, feat_dim)
        mask: np.ndarray of shape (max_seq_len,) with 1.0 on valid frames and 0.0 on padding
        aug_mode: one of `"clean"`, `"specaug"`, or `"mtr"`
    """
    features = np.asarray(features, dtype=np.float32)
    full_length, feat_dim = features.shape
    min_len = (
        max(16, min(int(min_seq_len), max_seq_len))
        if (variable_length and min_seq_len is not None)
        else max_seq_len
    )

    if variable_length and min_len < max_seq_len:
        if rng is not None:
            target_len = int(rng.integers(min_len, max_seq_len + 1))
        else:
            target_len = random.randint(min_len, max_seq_len)
    else:
        target_len = max_seq_len

    # Only tile-pad if shorter than min_len; if between min_len and target_len, keep natural length
    if full_length < min_len:
        repeats = (min_len // max(1, full_length)) + 1
        features = np.tile(features, (repeats, 1))
        full_length = features.shape[0]

    valid_len = min(full_length, target_len)
    if full_length > valid_len:
        if random_crop:
            if rng is not None:
                start = int(rng.integers(0, full_length - valid_len + 1))
            else:
                start = random.randint(0, full_length - valid_len)
        else:
            start = (full_length - valid_len) // 2
    else:
        start = 0

    segment = features[start:start + valid_len, :].copy()
    aug_segment, aug_mode = specaug.augment_features_mutually_exclusive(
        segment,
        specaug_config=specaug_config,
        waveform_aug_config=waveform_aug_config,
        rng=rng,
    )

    padded = np.zeros((max_seq_len, feat_dim), dtype=np.float32)
    padded[:valid_len, :] = aug_segment
    mask = np.zeros((max_seq_len,), dtype=np.float32)
    mask[:valid_len] = 1.0
    return padded, mask, aug_mode


def trim_features(
    features: np.ndarray,
    seq_len: int,
    specaug_config: Any,
) -> np.ndarray:
    """Trim features to SEQ_LEN."""
    full_length = features.shape[0]
    if full_length < seq_len:
        return pad_or_trim_features(features, seq_len, specaug_config)
    start = random.randint(0, full_length - seq_len)
    trimmed_features = features[start:start + seq_len, :]
    if getattr(specaug_config, "use_specaug", False):
        trimmed_features = specaug.apply_specaug(
            trimmed_features, specaug_config
        )
    return trimmed_features


def get_trimmed_triplet_features(
    _: Any,
    spk_to_utts: dataset.SpkToUtts,
    config: Any,
) -> np.ndarray:
    """Get a triplet of trimmed anchor/pos/neg features."""
    seq_len = config.model.seq_len
    specaug_config = config.train.specaug
    n_mfcc = config.model.n_mfcc

    anchor, pos, neg = get_triplet_features(spk_to_utts, n_mfcc)
    retries = 0
    while (
        anchor.shape[0] < seq_len
        or pos.shape[0] < seq_len
        or neg.shape[0] < seq_len
    ) and retries < 10:
        anchor, pos, neg = get_triplet_features(spk_to_utts, n_mfcc)
        retries += 1
    return np.stack(
        [
            pad_or_trim_features(anchor, seq_len, specaug_config),
            pad_or_trim_features(pos, seq_len, specaug_config),
            pad_or_trim_features(neg, seq_len, specaug_config),
        ]
    )


def get_batched_triplet_input(
    spk_to_utts: dataset.SpkToUtts,
    myconfig: Any,
    pool: Optional[mp_pool.Pool] = None,
) -> jax.Array:
    """Get batched triplet input for JAX, shape=(3 * batch_size, seq_len, feat_dim)."""
    feature_fetcher = functools.partial(
        get_trimmed_triplet_features,
        spk_to_utts=spk_to_utts,
        config=myconfig,
    )
    if pool is None:
        input_arrays = list(
            map(feature_fetcher, range(myconfig.train.batch_size))
        )
    else:
        input_arrays = pool.map(
            feature_fetcher, range(myconfig.train.batch_size)
        )
    batch_input = np.concatenate(input_arrays, axis=0)
    return jnp.asarray(batch_input)


def random_crop_or_pad(
    features: np.ndarray,
    seq_len: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomly crop or tile-pad 2D features `(T, D)` to `(seq_len, D)`."""
    full_length = features.shape[0]
    if full_length < seq_len:
        repeats = (seq_len // max(1, full_length)) + 1
        features = np.tile(features, (repeats, 1))
        full_length = features.shape[0]
    if full_length > seq_len:
        if rng is not None:
            start = int(rng.integers(0, full_length - seq_len + 1))
        else:
            start = random.randint(0, full_length - seq_len)
    else:
        start = 0
    return features[start:start + seq_len, :].copy()


class FeatureCache:
    """In-memory and optional on-disk feature cache for fast training and evaluation."""

    def __init__(
        self,
        frontend: Optional[AudioFrontend] = None,
        n_mfcc: int = 80,
        cache_dir: Optional[str] = None,
    ):
        self.frontend = frontend
        self.n_mfcc = n_mfcc
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self._cache: dict[str, np.ndarray] = {}
        self._cfg_signature = self._build_signature()

    def _build_signature(self) -> str:
        if self.frontend is None:
            return f"legacy_mfcc_{self.n_mfcc}"
        fc = self.frontend.config
        vp = self.frontend.vad_processor
        vc_str = "novad"
        if vp is not None and vp.config.enabled:
            vc = vp.config
            vc_str = (
                f"vad_{vc.mode}_{vc.energy_threshold_db}_"
                f"{getattr(vc, 'energy_db_threshold', 26.0)}_"
                f"{getattr(vc, 'apply_agc', True)}_{vc.hangover_frames}"
            )
        return (
            f"v2_{fc.feature_type}_{fc.n_mels}_{fc.apply_cmvn}_"
            f"{fc.stack_left_context}_{fc.stack_right_context}_{fc.frame_stride}_{vc_str}"
        )

    def _disk_path(self, audio_file: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        import hashlib

        key = f"{self._cfg_signature}::{audio_file}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        fname = f"{digest}.npy"
        primary = os.path.join(self.cache_dir, fname)
        if os.path.exists(primary):
            return primary
        parent_dir = os.path.dirname(self.cache_dir.rstrip(os.sep))
        if parent_dir and os.path.isdir(parent_dir):
            for sub in ("librispeech", "cnceleb2", "voxceleb1"):
                cand = os.path.join(parent_dir, sub, fname)
                if os.path.exists(cand):
                    return cand
        return primary

    def get(self, audio_file: str) -> np.ndarray:
        if audio_file in self._cache:
            return self._cache[audio_file]
        disk_p = self._disk_path(audio_file)
        if disk_p and os.path.exists(disk_p):
            try:
                arr = np.load(disk_p)
                self._cache[audio_file] = arr
                return arr
            except Exception:
                pass
        arr = extract_features(
            audio_file, n_mfcc=self.n_mfcc, frontend=self.frontend
        )
        if disk_p:
            try:
                np.save(disk_p, arr)
            except Exception:
                pass
        self._cache[audio_file] = arr
        return arr

    def preload(self, audio_files: list[str], num_workers: int = 8) -> None:
        missing = [f for f in audio_files if f not in self._cache]
        if not missing:
            return
        with mp_pool.ThreadPool(num_workers) as pool:
            pool.map(self.get, missing)

    def precompute_dataset(
        self, spk_to_utts: dataset.SpkToUtts, num_workers: int = 8
    ) -> None:
        all_files = [u for utts in spk_to_utts.values() for u in utts]
        self.preload(all_files, num_workers=num_workers)

    def __len__(self) -> int:
        return len(self._cache)


def _group_speakers_by_domain(
    speakers: list[str],
) -> dict[str, list[str]]:
    """Groups speaker IDs by `<domain>:<spk>` prefix (or `'default'` if unprefixed)."""
    groups: dict[str, list[str]] = {}
    for spk in speakers:
        dom = spk.split(":", 1)[0] if ":" in spk else "default"
        groups.setdefault(dom, []).append(spk)
    return groups


def _sample_speakers_domain_balanced(
    eligible_spks: list[str],
    n_spk: int,
    rng: Optional[np.random.Generator] = None,
    within_domain_prob: float = 0.75,
    domain_temp_alpha: float = 0.35,
) -> list[str]:
    """Samples `n_spk` speakers with domain-balanced stratified + cross-domain mixing."""
    if len(eligible_spks) < n_spk:
        return [random.choice(eligible_spks) for _ in range(n_spk)]

    domain_groups = _group_speakers_by_domain(eligible_spks)
    if len(domain_groups) <= 1:
        if rng is not None:
            idxs = rng.choice(len(eligible_spks), size=n_spk, replace=False)
            return [eligible_spks[int(i)] for i in idxs]
        return random.sample(eligible_spks, n_spk)

    domains = sorted(domain_groups.keys())
    raw_weights = np.array(
        [len(domain_groups[d]) ** domain_temp_alpha for d in domains],
        dtype=np.float64,
    )
    dom_probs = raw_weights / np.sum(raw_weights)
    u_draw = float(rng.random()) if rng is not None else random.random()

    valid_within = [
        i for i, d in enumerate(domains) if len(domain_groups[d]) >= n_spk
    ]
    if u_draw < within_domain_prob and valid_within:
        sub_probs = dom_probs[valid_within]
        sub_probs = sub_probs / np.sum(sub_probs)
        if rng is not None:
            chosen_idx = int(rng.choice(valid_within, p=sub_probs))
            dom_spks = domain_groups[domains[chosen_idx]]
            s_idxs = rng.choice(len(dom_spks), size=n_spk, replace=False)
            return [dom_spks[int(i)] for i in s_idxs]
        chosen_idx = random.choices(valid_within, weights=sub_probs.tolist(), k=1)[0]
        return random.sample(domain_groups[domains[chosen_idx]], n_spk)

    # Cross-domain balanced speaker sampling
    spk_weights = np.zeros(len(eligible_spks), dtype=np.float64)
    dom_to_prob = {d: dom_probs[i] for i, d in enumerate(domains)}
    for idx, spk in enumerate(eligible_spks):
        dom = spk.split(":", 1)[0] if ":" in spk else "default"
        spk_weights[idx] = dom_to_prob[dom] / len(domain_groups[dom])
    spk_weights /= np.sum(spk_weights)
    if rng is not None:
        idxs = rng.choice(
            len(eligible_spks), size=n_spk, replace=False, p=spk_weights
        )
        return [eligible_spks[int(i)] for i in idxs]
    return random.sample(eligible_spks, n_spk)


def get_batched_nm_input(
    spk_to_utts: dataset.SpkToUtts,
    num_spks_per_batch: Optional[int] = None,
    num_utts_per_spk: Optional[int] = None,
    seq_len: int = 160,
    specaug_config: Optional[Any] = None,
    feature_cache: Optional[FeatureCache] = None,
    frontend: Optional[AudioFrontend] = None,
    n_mfcc: int = 80,
    num_speakers: Optional[int] = None,
    num_utts_per_speaker: Optional[int] = None,
    cache: Optional[FeatureCache] = None,
    rng: Optional[np.random.Generator] = None,
    min_seq_len: Optional[int] = None,
    variable_length: bool = True,
    waveform_aug_config: Optional[Any] = None,
    return_mask: bool = False,
    within_domain_prob: float = 0.75,
    domain_temp_alpha: float = 0.35,
) -> Any:
    """Builds an N x M speaker-utterance batch for GE2E, Extended-Set Softmax, PFAS, and Dr-Vectors.

    Returns:
        `(batch_features, batch_labels)` by default, or
        `(batch_features, batch_labels, batch_mask)` when `return_mask=True`.
    """
    n_spk = num_speakers or num_spks_per_batch or 8
    n_utt = num_utts_per_speaker or num_utts_per_spk or 4
    active_cache = cache if cache is not None else feature_cache

    eligible_spks = sorted(
        [spk for spk, utts in spk_to_utts.items() if len(utts) >= 2]
    )
    selected_spks = _sample_speakers_domain_balanced(
        eligible_spks,
        n_spk=n_spk,
        rng=rng,
        within_domain_prob=within_domain_prob,
        domain_temp_alpha=domain_temp_alpha,
    )

    features_list = []
    masks_list = []
    labels_list = []
    for spk_idx, spk in enumerate(selected_spks):
        utts = spk_to_utts[spk]
        if len(utts) >= n_utt:
            if rng is not None:
                u_idxs = rng.choice(len(utts), size=n_utt, replace=False)
                chosen_utts = [utts[int(i)] for i in u_idxs]
            else:
                chosen_utts = random.sample(utts, n_utt)
        else:
            chosen_utts = [random.choice(utts) for _ in range(n_utt)]
        for utt in chosen_utts:
            if active_cache is not None:
                raw_feat = active_cache.get(utt)
            else:
                raw_feat = extract_features(
                    utt, n_mfcc=n_mfcc, frontend=frontend
                )
            padded, mask, _ = sample_variable_length_features(
                raw_feat,
                max_seq_len=seq_len,
                min_seq_len=min_seq_len,
                variable_length=variable_length,
                specaug_config=specaug_config,
                waveform_aug_config=waveform_aug_config,
                random_crop=True,
                rng=rng,
            )
            features_list.append(padded)
            masks_list.append(mask)
            labels_list.append(spk_idx)

    batch_features = jnp.asarray(
        np.stack(features_list, axis=0), dtype=jnp.float32
    )
    batch_labels = jnp.asarray(np.array(labels_list, dtype=np.int32))
    if return_mask:
        batch_mask = jnp.asarray(
            np.stack(masks_list, axis=0), dtype=jnp.float32
        )
        return batch_features, batch_labels, batch_mask
    return batch_features, batch_labels


def get_batched_classification_input(
    spk_to_utts: dataset.SpkToUtts,
    spk_to_id: dict[str, int],
    batch_size: int,
    seq_len: int,
    specaug_config: Optional[Any] = None,
    feature_cache: Optional[FeatureCache] = None,
    frontend: Optional[AudioFrontend] = None,
    n_mfcc: int = 80,
    cache: Optional[FeatureCache] = None,
    rng: Optional[np.random.Generator] = None,
    min_seq_len: Optional[int] = None,
    variable_length: bool = True,
    waveform_aug_config: Optional[Any] = None,
    return_mask: bool = False,
    domain_temp_alpha: float = 0.35,
) -> Any:
    """Builds a classification batch (features, global_speaker_ids[, mask]) for ArcFace/CosFace/SphereFace/Softmax."""
    active_cache = cache if cache is not None else feature_cache
    all_spks = sorted(spk_to_utts.keys())
    domain_groups = _group_speakers_by_domain(all_spks)
    domains = sorted(domain_groups.keys())
    raw_weights = np.array(
        [len(domain_groups[d]) ** domain_temp_alpha for d in domains],
        dtype=np.float64,
    )
    dom_probs = raw_weights / np.sum(raw_weights)

    features_list = []
    masks_list = []
    labels_list = []
    for _ in range(batch_size):
        if len(domains) > 1 and rng is not None:
            d_idx = int(rng.choice(len(domains), p=dom_probs))
            dom_spks = domain_groups[domains[d_idx]]
            spk = dom_spks[int(rng.integers(0, len(dom_spks)))]
            utts = spk_to_utts[spk]
            utt = utts[int(rng.integers(0, len(utts)))]
        elif rng is not None:
            spk = all_spks[int(rng.integers(0, len(all_spks)))]
            utts = spk_to_utts[spk]
            utt = utts[int(rng.integers(0, len(utts)))]
        else:
            spk = random.choice(all_spks)
            utt = random.choice(spk_to_utts[spk])
        if active_cache is not None:
            raw_feat = active_cache.get(utt)
        else:
            raw_feat = extract_features(utt, n_mfcc=n_mfcc, frontend=frontend)
        padded, mask, _ = sample_variable_length_features(
            raw_feat,
            max_seq_len=seq_len,
            min_seq_len=min_seq_len,
            variable_length=variable_length,
            specaug_config=specaug_config,
            waveform_aug_config=waveform_aug_config,
            random_crop=True,
            rng=rng,
        )
        features_list.append(padded)
        masks_list.append(mask)
        labels_list.append(spk_to_id[spk])

    batch_features = jnp.asarray(
        np.stack(features_list, axis=0), dtype=jnp.float32
    )
    batch_labels = jnp.asarray(np.array(labels_list, dtype=np.int32))
    if return_mask:
        batch_mask = jnp.asarray(
            np.stack(masks_list, axis=0), dtype=jnp.float32
        )
        return batch_features, batch_labels, batch_mask
    return batch_features, batch_labels
