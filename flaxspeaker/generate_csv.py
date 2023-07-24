import os

# Path to the directory containing the audio dataset.
PATH_TO_DATASET = os.path.join(
    os.path.expanduser("~"), "Downloads/CN-Celeb_flac/data")

# Extention name of the audio files.
AUDIO_FORMAT = ".flac"

# After we split the full audio path, the index indicating which part is
# the speaker label.
SPEAKER_LABEL_INDEX = -2

# Path to the output CSV file.
OUTPUT_CSV = "CN-Celeb.csv"


def generate_csv(path_to_dataset: str,
                 audio_format: str,
                 speaker_label_index: int,
                 output_csv: str):
    """Generate a CSV file from the audio dataset."""
    # Find all files in path_to_dataset with the extenion of audio_format.
    all_files = [os.path.join(dirpath, filename)
                 for dirpath, _, files in os.walk(path_to_dataset)
                 for filename in files if filename.endswith(audio_format)]

    # Prepare CSV text content.
    content = []
    for filename in all_files:
        speaker = filename.split(os.sep)[speaker_label_index]
        content.append(",".join([speaker, filename]))

    # Write CSV.
    with open(output_csv, "w") as f:
        f.write("\n".join(content))


if __name__ == "__main__":
    generate_csv(
        PATH_TO_DATASET,
        AUDIO_FORMAT,
        SPEAKER_LABEL_INDEX,
        OUTPUT_CSV)
