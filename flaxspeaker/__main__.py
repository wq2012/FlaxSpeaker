import argparse
import munch

from flaxspeaker import neural_net
from flaxspeaker import evaluation
from flaxspeaker import generate_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flaxspeaker",
        description="Model configurations.", add_help=True)

    parser.add_argument(
        "-m", "--mode",
        default="train",
        choices=["train", "eval", "generate_csv"],
        type=str,
        help="What mode to run the program in.")

    parser.add_argument(
        "-c", "--config",
        default="myconfig.yml",
        type=str,
        help="Path of the config file in YAML format.")

    parser.add_argument(
        "--path_to_dataset",
        type=str,
        help="Path to the directory containing the audio dataset. "
        "Only used by mode generate_csv.")

    parser.add_argument(
        "--audio_format",
        type=str,
        help="Extention name of the audio files. "
        "Only used by mode generate_csv.")

    parser.add_argument(
        "--speaker_label_index",
        default=-2,
        type=int,
        help="After we split the full audio path, the index indicating "
        "which part is the speaker label. "
        "Only used by mode generate_csv.")

    parser.add_argument(
        "--output_csv",
        type=str,
        help="Path to the output CSV file. "
        "Only used by mode generate_csv.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        myconfig = munch.Munch.fromYAML(f.read())

    if args.mode == "train":
        neural_net.run_training(myconfig)
    elif args.mode == "eval":
        evaluation.run_eval(myconfig)
    elif args.mode == "generate_csv":
        generate_csv.generate_csv(
            args.path_to_dataset,
            args.audio_format,
            args.speaker_label_index,
            args.output_csv)
    else:
        raise ValueError("Unsupported mode.")


if __name__ == "__main__":
    main()
