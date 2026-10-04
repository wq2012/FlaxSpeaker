"""Generate CSV manifests and train/eval splits for audio datasets."""

import argparse
import os
from typing import Optional

from flaxspeaker import dataset

# Default constants kept for backward compatibility.
PATH_TO_DATASET = os.path.join(
    os.path.expanduser("~"), "Downloads/CN-Celeb_flac/data"
)
AUDIO_FORMAT = ".flac"
SPEAKER_LABEL_INDEX = -2
OUTPUT_CSV = "CN-Celeb.csv"


def generate_csv(
    path_to_dataset: str,
    audio_format: str,
    speaker_label_index: int,
    output_csv: str,
) -> None:
    """Generate a CSV file from the audio dataset."""
    all_files = sorted(
        [
            os.path.join(dirpath, filename)
            for dirpath, _, files in os.walk(path_to_dataset)
            for filename in files
            if filename.endswith(audio_format)
        ]
    )

    content = []
    for filename in all_files:
        speaker = filename.split(os.sep)[speaker_label_index]
        content.append(",".join([speaker, filename]))

    parent = os.path.dirname(os.path.abspath(output_csv))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(content))


def prepare_dataset_splits(
    dataset_type: str,
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    min_utts_per_spk: int = 4,
    num_eval_trials: int = 2000,
    seed: int = 42,
    eval_data_dir: Optional[str] = None,
) -> dict[str, str]:
    """Prepare academic speaker-disjoint train/eval CSV manifests and trial files.

    Args:
        dataset_type: One of 'librispeech', 'cnceleb', 'voxceleb', or 'folder'.
        data_dir: Root directory of the dataset (or training subset).
        output_dir: Directory where `train.csv`, `eval.csv`, `trials.tsv` are written.
        train_ratio: Ratio of speakers assigned to train if `eval_data_dir` is None.
        min_utts_per_spk: Minimum utterances per speaker.
        num_eval_trials: Number of verification trials to generate.
        seed: Random seed for reproducible splits and trials.
        eval_data_dir: Optional separate directory for evaluation speakers.

    Returns:
        Dictionary with paths to `train_csv`, `eval_csv`, and `trials_file`.
    """
    os.makedirs(output_dir, exist_ok=True)
    dtype = dataset_type.lower()

    def _load(d: str) -> dataset.SpkToUtts:
        if dtype == "librispeech":
            return dataset.get_librispeech_spk_to_utts(d)
        elif dtype == "cnceleb":
            return dataset.get_cnceleb_spk_to_utts(d)
        elif dtype == "voxceleb":
            return dataset.get_voxceleb_spk_to_utts(d)
        else:
            return dataset.get_folder_spk_to_utts(d)

    if eval_data_dir:
        train_spk = dataset.filter_speakers_by_min_utts(
            _load(data_dir), min_utts=min_utts_per_spk, seed=seed
        )
        eval_spk = dataset.filter_speakers_by_min_utts(
            _load(eval_data_dir), min_utts=min_utts_per_spk, seed=seed
        )
    else:
        all_spk = _load(data_dir)
        train_spk, eval_spk = dataset.split_speakers_train_eval(
            all_spk,
            train_ratio=train_ratio,
            min_utts_per_spk=min_utts_per_spk,
            seed=seed,
        )

    train_csv = os.path.join(output_dir, "train.csv")
    eval_csv = os.path.join(output_dir, "eval.csv")
    trials_file = os.path.join(output_dir, "trials.tsv")

    dataset.save_spk_to_utts_csv(train_spk, train_csv)
    dataset.save_spk_to_utts_csv(eval_spk, eval_csv)
    trials = dataset.generate_verification_trials(
        eval_spk, num_trials=num_eval_trials, seed=seed
    )
    dataset.save_trials_file(trials, trials_file)

    return {
        "train_csv": train_csv,
        "eval_csv": eval_csv,
        "trials_file": trials_file,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV manifests for FlaxSpeaker.")
    parser.add_argument("--path_to_dataset", type=str, default=PATH_TO_DATASET)
    parser.add_argument("--audio_format", type=str, default=AUDIO_FORMAT)
    parser.add_argument("--speaker_label_index", type=int, default=SPEAKER_LABEL_INDEX)
    parser.add_argument("--output_csv", type=str, default=OUTPUT_CSV)
    args = parser.parse_args()
    generate_csv(
        args.path_to_dataset,
        args.audio_format,
        args.speaker_label_index,
        args.output_csv,
    )
