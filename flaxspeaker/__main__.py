"""Command-line interface for FlaxSpeaker."""

import argparse
import os
import munch

from flaxspeaker import configs
from flaxspeaker import evaluation
from flaxspeaker import export
from flaxspeaker import generate_csv
from flaxspeaker import hf_compat
from flaxspeaker import neural_net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flaxspeaker",
        description="FlaxSpeaker: JAX/Flax Speaker Recognition & Verification CLI.",
        add_help=True,
    )

    parser.add_argument(
        "-m",
        "--mode",
        default="train",
        choices=[
            "train",
            "eval",
            "generate_csv",
            "prepare_splits",
            "export_tflite",
            "save_hf",
        ],
        type=str,
        help="What mode to run the program in.",
    )

    parser.add_argument(
        "-c",
        "--config",
        default="myconfig.yml",
        type=str,
        help="Path of the config file in YAML format.",
    )

    parser.add_argument(
        "--path_to_dataset",
        type=str,
        help="Path to the directory containing the audio dataset.",
    )

    parser.add_argument(
        "--dataset_type",
        default="librispeech",
        type=str,
        help="Dataset type (librispeech, cnceleb, voxceleb, folder).",
    )

    parser.add_argument(
        "--audio_format",
        default=".flac",
        type=str,
        help="Extension name of the audio files (used by generate_csv).",
    )

    parser.add_argument(
        "--speaker_label_index",
        default=-2,
        type=int,
        help="Index of speaker label when splitting full audio path.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        help="Path to the output CSV file (used by generate_csv).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for splits, TFLite export, or HuggingFace model.",
    )

    parser.add_argument(
        "--quantize_int8",
        action="store_true",
        help="Apply 8-bit dynamic range quantization when exporting TFLite.",
    )

    return parser.parse_args()


def load_config_any(config_path: str):
    """Load YAML config as ExperimentConfig if modern, else legacy Munch."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    m = munch.Munch.fromYAML(raw_text)
    if neural_net._is_modern_config(m):
        return configs.ExperimentConfig.from_yaml(config_path)
    return m


def main() -> None:
    args = parse_args()

    if args.mode == "generate_csv":
        generate_csv.generate_csv(
            args.path_to_dataset,
            args.audio_format,
            args.speaker_label_index,
            args.output_csv,
        )
        return

    if args.mode == "prepare_splits":
        paths = generate_csv.prepare_dataset_splits(
            dataset_type=args.dataset_type,
            data_dir=args.path_to_dataset,
            output_dir=args.output_dir,
        )
        print("Prepared dataset splits:", paths)
        return

    myconfig = load_config_any(args.config)

    if args.mode == "train":
        neural_net.run_training(myconfig)
    elif args.mode == "eval":
        evaluation.run_eval(myconfig)
    elif args.mode == "export_tflite":
        _, state = neural_net.get_speaker_encoder(
            myconfig, load_from=myconfig.model.saved_model_path
        )
        out_path = os.path.join(args.output_dir, "speaker_encoder.tflite")
        info = export.export_to_tflite(
            state, myconfig, out_path, quantize_int8=args.quantize_int8
        )
        print("Exported TFLite model:", info)
    elif args.mode == "save_hf":
        exp_cfg = (
            myconfig
            if isinstance(myconfig, configs.ExperimentConfig)
            else configs.ExperimentConfig.from_munch(myconfig)
        )
        _, state = neural_net.get_speaker_encoder(
            exp_cfg, load_from=exp_cfg.model.saved_model_path
        )
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(exp_cfg)
        hf_model = hf_compat.FlaxSpeakerModel(config=hf_cfg, params=state.params)
        hf_model.save_pretrained(args.output_dir)
        print("Saved HuggingFace model to:", args.output_dir)
    else:
        raise ValueError("Unsupported mode.")


if __name__ == "__main__":
    main()
