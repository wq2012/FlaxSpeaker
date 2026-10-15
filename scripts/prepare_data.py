#!/usr/bin/env python3
"""Prepare speaker recognition datasets (LibriSpeech, CN-Celeb2, VoxCeleb1, Universal Multi-Domain, or Synthetic).

Usage:
    # Prepare real downloaded datasets + Universal Multi-Domain training/eval splits:
    python scripts/prepare_data.py --data_root ~/Data/speaker_datasets

    # Generate a self-contained synthetic speaker dataset for smoke testing:
    python scripts/prepare_data.py --generate_synthetic --output_dir /tmp/synthetic_speaker_data
"""

import argparse
import csv
import json
import os
import sys
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import dataset  # noqa: E402
from scripts import prepare_datasets  # noqa: E402


def generate_synthetic_dataset(
    output_dir: str,
    num_train_speakers: int = 16,
    num_eval_speakers: int = 8,
    utts_per_speaker: int = 6,
    sample_rate: int = 16000,
    duration_sec: float = 2.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a synthetic speaker verification dataset with harmonic formants."""
    rng = np.random.default_rng(seed)
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    def _synth_speaker_utts(spk_prefix: str, num_spks: int) -> dict[str, list[str]]:
        spk_map: dict[str, list[str]] = {}
        for s_idx in range(num_spks):
            spk_id = f"{spk_prefix}_spk_{s_idx:03d}"
            f0 = float(rng.uniform(95.0, 255.0))
            formants = [f0 * k for k in (1.0, 2.0, 3.0, 4.5)]
            amps = rng.uniform(0.15, 0.45, size=len(formants))
            utts: list[str] = []
            for u_idx in range(utts_per_speaker):
                jitter = float(rng.normal(1.0, 0.015))
                sig = np.zeros_like(t)
                for freq, amp in zip(formants, amps):
                    phase = float(rng.uniform(0, 2 * np.pi))
                    sig += float(amp) * np.sin(2.0 * np.pi * freq * jitter * t + phase)
                sig += rng.normal(0.0, 0.02, size=n_samples).astype(np.float32)
                sig = sig / (np.max(np.abs(sig)) + 1e-6) * 0.8
                wav_path = os.path.abspath(
                    os.path.join(wav_dir, f"{spk_id}_utt_{u_idx:02d}.wav")
                )
                sf.write(wav_path, sig, sample_rate)
                utts.append(wav_path)
            spk_map[spk_id] = utts
        return spk_map

    train_spks = _synth_speaker_utts("train", num_train_speakers)
    eval_spks = _synth_speaker_utts("eval", num_eval_speakers)

    train_csv = os.path.join(output_dir, "train.csv")
    eval_csv = os.path.join(output_dir, "eval.csv")
    trials_file = os.path.join(output_dir, "trials.tsv")

    for path, spk_dict in ((train_csv, train_spks), (eval_csv, eval_spks)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for spk in sorted(spk_dict.keys()):
                for u in spk_dict[spk]:
                    writer.writerow([spk, u])

    trials = dataset.generate_verification_trials(
        eval_spks, num_trials=200, positive_ratio=0.5, seed=seed
    )
    dataset.save_trials_file(trials, trials_file)

    summary = {
        "train_csv": os.path.abspath(train_csv),
        "eval_csv": os.path.abspath(eval_csv),
        "trials_file": os.path.abspath(trials_file),
        "train_speakers": len(train_spks),
        "eval_speakers": len(eval_spks),
        "num_trials": len(trials),
    }
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare real or synthetic speaker recognition datasets."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=prepare_datasets.DEFAULT_DATA_ROOT,
        help="Root directory containing raw/ and splits/ for LibriSpeech, CN-Celeb2, VoxCeleb1.",
    )
    parser.add_argument(
        "--generate_synthetic",
        action="store_true",
        help="Generate a synthetic speaker verification dataset for fast smoke testing.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/flaxspeaker_synthetic",
        help="Output directory when --generate_synthetic is enabled.",
    )
    args = parser.parse_args()

    if args.generate_synthetic:
        summary = generate_synthetic_dataset(args.output_dir)
        print(json.dumps(summary, indent=2))
    else:
        summary = prepare_datasets.prepare_all_datasets(data_root=args.data_root)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
