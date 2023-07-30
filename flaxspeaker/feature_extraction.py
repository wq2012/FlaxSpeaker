import librosa
import soundfile as sf
import random
import numpy as np
import jax
import jax.numpy as jnp
from multiprocessing import pool
from typing import Optional, Any
import munch
import functools

from flaxspeaker import dataset
from flaxspeaker import specaug


def extract_features(audio_file: str, n_mfcc: int) -> np.ndarray:
    """Extract MFCC features from an audio file, shape=(TIME, MFCC)."""
    waveform, sample_rate = sf.read(audio_file)

    # Convert to mono-channel.
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())

    # Convert to 16kHz.
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, sample_rate, 16000)

    features = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    return features.transpose()


def extract_sliding_windows(features: np.ndarray,
                            myconfig: munch.Munch) -> list[np.ndarray]:
    """Extract sliding windows from features."""
    sliding_windows = []
    start = 0
    while start + myconfig.model.seq_len <= features.shape[0]:
        sliding_windows.append(
            features[start: start + myconfig.model.seq_len, :])
        start += myconfig.model.sliding_window_step
    return sliding_windows


def get_triplet_features(spk_to_utts: dataset.SpkToUtts,
                         n_mfcc: int
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get a triplet of anchor/pos/neg features."""
    anchor_utt, pos_utt, neg_utt = dataset.get_triplet(spk_to_utts)
    return (extract_features(anchor_utt, n_mfcc),
            extract_features(pos_utt, n_mfcc),
            extract_features(neg_utt, n_mfcc))


def trim_features(features: np.ndarray,
                  seq_len: int,
                  specaug_config: munch.Munch
                  ) -> np.ndarray:
    """Trim features to SEQ_LEN."""
    full_length = features.shape[0]
    start = random.randint(0, full_length - seq_len)
    trimmed_features = features[start: start + seq_len, :]
    if specaug_config.use_specaug:
        trimmed_features = specaug.apply_specaug(
            trimmed_features, specaug_config)
    return trimmed_features


def get_trimmed_triplet_features(_: Any, spk_to_utts: dataset.SpkToUtts,
                                 myconfig: munch.Munch) -> np.ndarray:
    """Get a triplet of trimmed anchor/pos/neg features."""
    seq_len = myconfig.model.seq_len
    specaug_config = myconfig.train.specaug
    n_mfcc = myconfig.model.n_mfcc

    anchor, pos, neg = get_triplet_features(spk_to_utts, n_mfcc)
    while (anchor.shape[0] < seq_len or
            pos.shape[0] < seq_len or
            neg.shape[0] < seq_len):
        anchor, pos, neg = get_triplet_features(
            spk_to_utts, n_mfcc)
    return np.stack([
        trim_features(anchor, seq_len, specaug_config),
        trim_features(pos, seq_len, specaug_config),
        trim_features(neg, seq_len, specaug_config)])


def get_batched_triplet_input(
        spk_to_utts: dataset.SpkToUtts,
        myconfig: munch.Munch,
        pool: Optional[pool.Pool] = None) -> jax.Array:
    """Get batched triplet input for Jax."""
    feature_fetcher = functools.partial(
        get_trimmed_triplet_features,
        spk_to_utts=spk_to_utts,
        myconfig=myconfig)
    if pool is None:
        input_arrays = list(
            map(feature_fetcher, range(myconfig.train.batch_size)))
    else:
        input_arrays = pool.map(
            feature_fetcher, range(myconfig.train.batch_size))
    batch_input = np.concatenate(input_arrays)
    return jnp.asarray(batch_input)
