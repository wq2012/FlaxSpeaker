"""Dataset loaders, academic split generators, and trial list builders.

Supports:
- LibriSpeech (`<root>/<spk_id>/<chapter_id>/<spk_id>-<chapter_id>-<utt_id>.flac`)
- VoxCeleb1 & VoxCeleb2 (`<root>/idXXXXX/<video_id>/<utt_id>.wav`) + official `veri_test2.txt`
- CN-Celeb1 & CN-Celeb2 (`<root>/idXXXXX/<genre>-<utt_id>.flac`)
- Generic folder hierarchies and CSV manifests (`<speaker_id>,<audio_path>`)
- Academic speaker-disjoint train/eval split generation
- Single-utterance and multi-utterance verification trial generation
"""

import csv
import glob
import os
import random
from typing import Any, Optional, Sequence

import numpy as np


SpkToUtts = dict[str, list[str]]
TrialPair = tuple[int, str, str]  # (label, enroll_utt, test_utt)
MultiEnrollTrial = tuple[int, list[str], str]  # (label, [enroll_utts], test_utt)


def get_librispeech_spk_to_utts(data_dir: str) -> SpkToUtts:
    """Get the dict from speaker to list of utterances for LibriSpeech."""
    flac_files = sorted(glob.glob(os.path.join(data_dir, "*", "*", "*.flac")))
    spk_to_utts: SpkToUtts = {}
    for flac_file in flac_files:
        basename = os.path.basename(flac_file)
        split_name = basename.split("-")
        spk = split_name[0]
        if spk not in spk_to_utts:
            spk_to_utts[spk] = [flac_file]
        else:
            spk_to_utts[spk].append(flac_file)
    return spk_to_utts


def get_voxceleb_spk_to_utts(data_dir: str) -> SpkToUtts:
    """Get speaker-to-utterances dictionary for VoxCeleb1 / VoxCeleb2.

    Supports both `<data_dir>/idXXXXX/<video_id>/<utt>.wav` and
    `<data_dir>/wav/idXXXXX/<video_id>/<utt>.wav` layouts.
    """
    search_roots = [data_dir]
    for sub in ("wav", "vox1_test_wav/wav", "vox1_dev_wav/wav", "dev/aac"):
        candidate = os.path.join(data_dir, sub)
        if os.path.isdir(candidate):
            search_roots.append(candidate)

    spk_to_utts: SpkToUtts = {}
    for root in search_roots:
        for ext in ("*.wav", "*.flac"):
            pattern = os.path.join(root, "id*", "*", ext)
            for path in sorted(glob.glob(pattern)):
                parts = path.split(os.sep)
                spk = parts[-3]
                spk_to_utts.setdefault(spk, []).append(path)
        if spk_to_utts:
            break
    return spk_to_utts


def get_cnceleb_spk_to_utts(data_dir: str) -> SpkToUtts:
    """Get speaker-to-utterances dictionary for CN-Celeb1 / CN-Celeb2.

    Supports `<data_dir>/idXXXXX/<utt>.flac`, `<data_dir>/data/idXXXXX/<utt>.flac`,
    and `<data_dir>/CN-Celeb2_flac/data/idXXXXX/<utt>.flac`.
    """
    search_roots = [data_dir]
    for sub in ("data", "CN-Celeb_flac/data", "CN-Celeb2_flac/data", "eval/enroll", "eval/test"):
        candidate = os.path.join(data_dir, sub)
        if os.path.isdir(candidate):
            search_roots.append(candidate)

    spk_to_utts: SpkToUtts = {}
    for root in search_roots:
        for ext in ("*.flac", "*.wav"):
            pattern = os.path.join(root, "id*", ext)
            for path in sorted(glob.glob(pattern)):
                parts = path.split(os.sep)
                spk = parts[-2]
                spk_to_utts.setdefault(spk, []).append(path)
        if spk_to_utts:
            break
    return spk_to_utts


def get_folder_spk_to_utts(
    data_dir: str,
    audio_extensions: Sequence[str] = (".flac", ".wav", ".ogg", ".mp3"),
    speaker_label_index: int = -2,
) -> SpkToUtts:
    """Scan an arbitrary directory tree and group audio files by speaker ID."""
    spk_to_utts: SpkToUtts = {}
    for dirpath, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if any(fname.lower().endswith(ext) for ext in audio_extensions):
                full_path = os.path.join(dirpath, fname)
                parts = full_path.split(os.sep)
                if abs(speaker_label_index) <= len(parts):
                    spk = parts[speaker_label_index]
                else:
                    spk = os.path.basename(dirpath)
                spk_to_utts.setdefault(spk, []).append(full_path)
    return spk_to_utts


def get_csv_spk_to_utts(csv_file: str) -> SpkToUtts:
    """Get the dict from speaker to list of utterances from CSV file."""
    spk_to_utts: SpkToUtts = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            spk = row[0].strip()
            utt = row[1].strip()
            if not spk or not utt or (spk == "speaker_id" and utt == "audio_path"):
                continue
            if spk not in spk_to_utts:
                spk_to_utts[spk] = [utt]
            else:
                spk_to_utts[spk].append(utt)
    return spk_to_utts


def save_spk_to_utts_csv(spk_to_utts: SpkToUtts, csv_file: str) -> None:
    """Save a speaker-to-utterances dictionary to a CSV manifest file."""
    parent = os.path.dirname(os.path.abspath(csv_file))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for spk in sorted(spk_to_utts.keys()):
            for utt in spk_to_utts[spk]:
                writer.writerow([spk, utt])


def filter_speakers_by_min_utts(
    spk_to_utts: SpkToUtts,
    min_utts: int = 2,
    max_utts_per_spk: Optional[int] = None,
    seed: int = 42,
) -> SpkToUtts:
    """Filter out speakers with fewer than `min_utts` utterances."""
    rng = random.Random(seed)
    filtered: SpkToUtts = {}
    for spk, utts in sorted(spk_to_utts.items()):
        if len(utts) < min_utts:
            continue
        if max_utts_per_spk is not None and len(utts) > max_utts_per_spk:
            utts = rng.sample(list(utts), max_utts_per_spk)
        filtered[spk] = list(utts)
    return filtered


def split_speakers_train_eval(
    spk_to_utts: SpkToUtts,
    train_ratio: float = 0.8,
    min_utts_per_spk: int = 2,
    seed: int = 42,
) -> tuple[SpkToUtts, SpkToUtts]:
    """Create a speaker-disjoint train/eval split (open-set verification protocol).

    In speaker verification research, evaluation speakers must be completely
    disjoint from training speakers so that embedding generalization to unseen
    speakers is measured faithfully.
    """
    filtered = filter_speakers_by_min_utts(spk_to_utts, min_utts=min_utts_per_spk, seed=seed)
    speakers = sorted(filtered.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_train = max(2, int(round(len(speakers) * train_ratio)))
    if n_train >= len(speakers):
        n_train = max(1, len(speakers) - 2)
    train_spks = sorted(speakers[:n_train])
    eval_spks = sorted(speakers[n_train:])
    train_dict = {spk: filtered[spk] for spk in train_spks}
    eval_dict = {spk: filtered[spk] for spk in eval_spks}
    return train_dict, eval_dict


def get_triplet(spk_to_utts: SpkToUtts) -> tuple[str, str, str]:
    """Get a triplet of anchor/pos/neg samples."""
    pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    # Retry if too few positive utterances.
    while len(spk_to_utts[pos_spk]) < 2:
        pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    anchor_utt, pos_utt = random.sample(spk_to_utts[pos_spk], 2)
    neg_utt = random.sample(spk_to_utts[neg_spk], 1)[0]
    return (anchor_utt, pos_utt, neg_utt)


def generate_verification_trials(
    spk_to_utts: SpkToUtts,
    num_trials: int = 2000,
    positive_ratio: float = 0.5,
    seed: int = 42,
) -> list[TrialPair]:
    """Generate a balanced deterministic list of verification trial pairs.

    Returns:
        List of `(label, enroll_utt, test_utt)` where `label` is 1 for target
        (same speaker) and 0 for non-target (different speaker).
    """
    rng = random.Random(seed)
    valid_spks = [s for s, u in sorted(spk_to_utts.items()) if len(u) >= 2]
    all_spks = sorted(spk_to_utts.keys())
    if len(valid_spks) < 1 or len(all_spks) < 2:
        raise ValueError("Need at least 2 speakers and >=2 utterances to build trials.")

    num_pos = int(round(num_trials * positive_ratio))
    num_neg = num_trials - num_pos
    trials: list[TrialPair] = []

    for _ in range(num_pos):
        spk = rng.choice(valid_spks)
        u1, u2 = rng.sample(spk_to_utts[spk], 2)
        trials.append((1, u1, u2))

    for _ in range(num_neg):
        s1, s2 = rng.sample(all_spks, 2)
        u1 = rng.choice(spk_to_utts[s1])
        u2 = rng.choice(spk_to_utts[s2])
        trials.append((0, u1, u2))

    rng.shuffle(trials)
    return trials


def generate_multi_enroll_trials(
    spk_to_utts: SpkToUtts,
    num_trials: int = 1000,
    num_enroll_utts: int = 3,
    positive_ratio: float = 0.5,
    seed: int = 42,
) -> list[MultiEnrollTrial]:
    """Generate multi-utterance enrollment verification trials.

    Each trial consists of `(label, [enroll_utt_1, ..., enroll_utt_K], test_utt)`,
    ideal for evaluating Parameter-Free Attentive Scoring (PFAS) vs centroid
    averaging across multiple enrollment utterances.
    """
    rng = random.Random(seed)
    valid_pos_spks = [
        s for s, u in sorted(spk_to_utts.items()) if len(u) >= num_enroll_utts + 1
    ]
    valid_enroll_spks = [
        s for s, u in sorted(spk_to_utts.items()) if len(u) >= num_enroll_utts
    ]
    all_spks = sorted(spk_to_utts.keys())
    if not valid_pos_spks or len(all_spks) < 2:
        raise ValueError(
            f"Need speakers with at least {num_enroll_utts + 1} utterances."
        )

    num_pos = int(round(num_trials * positive_ratio))
    num_neg = num_trials - num_pos
    trials: list[MultiEnrollTrial] = []

    for _ in range(num_pos):
        spk = rng.choice(valid_pos_spks)
        sampled = rng.sample(spk_to_utts[spk], num_enroll_utts + 1)
        trials.append((1, sampled[:-1], sampled[-1]))

    for _ in range(num_neg):
        enroll_spk = rng.choice(valid_enroll_spks)
        other_spks = [s for s in all_spks if s != enroll_spk]
        test_spk = rng.choice(other_spks)
        enroll_utts = rng.sample(spk_to_utts[enroll_spk], num_enroll_utts)
        test_utt = rng.choice(spk_to_utts[test_spk])
        trials.append((0, enroll_utts, test_utt))

    rng.shuffle(trials)
    return trials


def load_voxceleb_veri_test(
    veri_test_path: str,
    wav_root: str,
    max_trials: Optional[int] = None,
    seed: int = 42,
) -> list[TrialPair]:
    """Load official VoxCeleb1 verification trials (`veri_test.txt` / `veri_test2.txt`)."""
    # Resolve wav_root if nested under wav/
    for sub in ("", "wav", "vox1_test_wav/wav"):
        candidate = os.path.join(wav_root, sub) if sub else wav_root
        if os.path.isdir(os.path.join(candidate, "id10270")):
            wav_root = candidate
            break

    trials: list[TrialPair] = []
    with open(veri_test_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            label = int(parts[0])
            u1 = os.path.join(wav_root, parts[1])
            u2 = os.path.join(wav_root, parts[2])
            if os.path.exists(u1) and os.path.exists(u2):
                trials.append((label, u1, u2))

    if max_trials is not None and len(trials) > max_trials:
        rng = random.Random(seed)
        pos_trials = [t for t in trials if t[0] == 1]
        neg_trials = [t for t in trials if t[0] == 0]
        n_pos = max_trials // 2
        n_neg = max_trials - n_pos
        trials = rng.sample(pos_trials, min(n_pos, len(pos_trials))) + rng.sample(
            neg_trials, min(n_neg, len(neg_trials))
        )
        rng.shuffle(trials)
    return trials


def save_trials_file(trials: Sequence[TrialPair], output_path: str) -> None:
    """Save verification trials (`label enroll_path test_path`) to disk."""
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for label, u1, u2 in trials:
            f.write(f"{int(label)}\t{u1}\t{u2}\n")


def load_trials_file(trials_path: str) -> list[TrialPair]:
    """Load verification trials from TSV or whitespace-separated file."""
    trials: list[TrialPair] = []
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) != 3:
                continue
            trials.append((int(parts[0]), parts[1], parts[2]))
    return trials


def load_dataset_from_config(config: Any, split: str = "train") -> SpkToUtts:
    """Unified dataset loader supporting LibriSpeech, VoxCeleb, CN-Celeb, and CSV."""
    data_cfg = getattr(config, "data", config)
    dataset_type = str(getattr(data_cfg, "dataset_type", "librispeech")).lower()

    if split == "train":
        csv_path = getattr(data_cfg, "train_csv", None)
        dir_path = getattr(data_cfg, "train_dir", None) or getattr(
            data_cfg, "train_librispeech_dir", None
        )
    else:
        csv_path = getattr(data_cfg, "test_csv", None)
        dir_path = getattr(data_cfg, "test_dir", None) or getattr(
            data_cfg, "test_librispeech_dir", None
        )

    if csv_path:
        return get_csv_spk_to_utts(csv_path)
    if not dir_path:
        raise ValueError(f"No data path configured for split={split}")

    if dataset_type == "librispeech":
        spk_to_utts = get_librispeech_spk_to_utts(dir_path)
        if not spk_to_utts:
            spk_to_utts = get_folder_spk_to_utts(dir_path)
        return spk_to_utts
    elif dataset_type == "voxceleb":
        return get_voxceleb_spk_to_utts(dir_path)
    elif dataset_type == "cnceleb":
        return get_cnceleb_spk_to_utts(dir_path)
    else:
        return get_folder_spk_to_utts(
            dir_path,
            speaker_label_index=int(getattr(data_cfg, "speaker_label_index", -2)),
        )


def compute_dataset_stats(spk_to_utts: SpkToUtts) -> dict[str, Any]:
    """Compute summary statistics for a speaker-to-utterances dictionary."""
    num_speakers = len(spk_to_utts)
    utt_counts = [len(u) for u in spk_to_utts.values()]
    total_utts = sum(utt_counts)
    return {
        "num_speakers": num_speakers,
        "total_utterances": total_utts,
        "min_utts_per_speaker": int(min(utt_counts)) if utt_counts else 0,
        "max_utts_per_speaker": int(max(utt_counts)) if utt_counts else 0,
        "mean_utts_per_speaker": float(np.mean(utt_counts)) if utt_counts else 0.0,
        "median_utts_per_speaker": float(np.median(utt_counts)) if utt_counts else 0.0,
    }
