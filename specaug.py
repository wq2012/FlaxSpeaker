import numpy as np
import random
import munch


def apply_specaug(
        features: np.ndarray,
        specaug_config: munch.Munch) -> np.ndarray:
    """Apply SpecAugment to features."""
    seq_len, n_mfcc = features.shape
    outputs = features
    mean_feature = np.mean(features)

    # Frequancy masking.
    if random.random() < specaug_config.freq_mask_prob:
        width = random.randint(1, specaug_config.freq_mask_max_width)
        start = random.randint(0, n_mfcc - width)
        outputs[:, start: start + width] = mean_feature

    # Time masking.
    if random.random() < specaug_config.time_mask_prob:
        width = random.randint(1, specaug_config.time_mask_max_width)
        start = random.randint(0, seq_len - width)
        outputs[start: start + width, :] = mean_feature

    return outputs
