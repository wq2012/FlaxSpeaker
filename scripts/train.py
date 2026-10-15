#!/usr/bin/env python3
"""Train a FlaxSpeaker model from a YAML config or paper preset.

Usage:
    python scripts/train.py --config configs/univ_conformer_small_ge2e.yml \\
        --output_dir pretrained_models/my_experiment

    python scripts/train.py --preset universal_multi_domain \\
        --train_csv /path/to/train.csv --eval_csv /path/to/eval.csv \\
        --trials_file /path/to/trials.tsv --output_dir /tmp/my_model
"""

import argparse
import json
import os
import sys
import time

import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import configs  # noqa: E402
from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import evaluation  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402
from flaxspeaker import neural_net  # noqa: E402


PRESET_BUILDERS = {
    "icassp2018_ge2e": configs.icassp2018_ge2e_config,
    "interspeech2021_dr_vectors": configs.interspeech2021_dr_vectors_config,
    "odyssey2022_conformer_atp": configs.odyssey2022_conformer_atp_config,
    "odyssey2022_pfas": configs.odyssey2022_pfas_config,
    "universal_multi_domain": configs.universal_multi_domain_config,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a FlaxSpeaker speaker encoder model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML experiment configuration file.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="",
        choices=sorted(PRESET_BUILDERS.keys()),
        help="Optional pre-packaged paper preset name if --config is not provided.",
    )
    parser.add_argument("--train_csv", type=str, default="", help="Override train CSV path.")
    parser.add_argument("--eval_csv", type=str, default="", help="Override eval CSV path.")
    parser.add_argument("--trials_file", type=str, default="", help="Override trials TSV path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save HuggingFace-compatible pretrained model.",
    )
    parser.add_argument("--num_steps", type=int, default=0, help="Override training steps.")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0, help="Override learning rate."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional directory for caching acoustic features.",
    )
    args = parser.parse_args()

    if args.config:
        exp_cfg = configs.ExperimentConfig.from_yaml(args.config)
    elif args.preset:
        exp_cfg = PRESET_BUILDERS[args.preset]()
    else:
        exp_cfg = configs.universal_multi_domain_config()

    if args.train_csv:
        exp_cfg.data.train_csv = args.train_csv
    if args.eval_csv:
        exp_cfg.data.test_csv = args.eval_csv
    if args.trials_file:
        exp_cfg.data.trials_file = args.trials_file
    if args.num_steps > 0:
        exp_cfg.train.num_steps = args.num_steps
    if args.learning_rate > 0.0:
        exp_cfg.train.learning_rate = args.learning_rate
    if args.output_dir:
        exp_cfg.train.saved_model_dir = args.output_dir

    if not exp_cfg.data.train_csv or not os.path.exists(exp_cfg.data.train_csv):
        raise FileNotFoundError(
            f"Training CSV not found: '{exp_cfg.data.train_csv}'. Pass --train_csv or configure data.train_csv."
        )

    spk_train = dataset.get_csv_spk_to_utts(exp_cfg.data.train_csv)
    frontend = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)
    cache = feature_extraction.FeatureCache(
        frontend, cache_dir=args.cache_dir if args.cache_dir else None
    )
    cache.precompute_dataset(spk_train, num_workers=8)

    loss_type = str(exp_cfg.train.loss.loss_type).lower()
    is_cls = loss_type in ("arcface", "cosface", "sphereface", "softmax")
    speaker_to_id = {s: i for i, s in enumerate(sorted(spk_train.keys()))}
    num_classes = len(speaker_to_id) if is_cls else 0

    _, state = neural_net.get_speaker_encoder(exp_cfg, num_classes=num_classes)
    step_fn = neural_net.make_modern_train_step(exp_cfg)
    rng = np.random.default_rng(exp_cfg.train.seed)

    t0 = time.time()
    losses: list[float] = []
    for step in range(exp_cfg.train.num_steps):
        if loss_type == "triplet":
            batch_list, mask_list = [], []
            for _ in range(exp_cfg.train.batch_size):
                a_u, p_u, n_u = dataset.get_triplet(spk_train)
                for u in (a_u, p_u, n_u):
                    feat = cache.get(u)
                    crop_aug, mask, _ = feature_extraction.sample_variable_length_features(
                        feat,
                        max_seq_len=exp_cfg.model.seq_len,
                        min_seq_len=exp_cfg.model.min_seq_len,
                        variable_length=True,
                        specaug_config=exp_cfg.train.specaug,
                        waveform_aug_config=exp_cfg.train.waveform_aug,
                        rng=rng,
                    )
                    batch_list.append(crop_aug)
                    mask_list.append(mask)
            batch_input = jnp.asarray(np.stack(batch_list, axis=0), dtype=jnp.float32)
            batch_mask = jnp.asarray(np.stack(mask_list, axis=0), dtype=jnp.float32)
            batch_labels = None
        elif is_cls:
            batch_input, batch_labels, batch_mask = (
                feature_extraction.get_batched_classification_input(
                    spk_train,
                    speaker_to_id,
                    batch_size=exp_cfg.train.batch_size,
                    seq_len=exp_cfg.model.seq_len,
                    frontend=frontend,
                    specaug_config=exp_cfg.train.specaug,
                    waveform_aug_config=exp_cfg.train.waveform_aug,
                    cache=cache,
                    rng=rng,
                    min_seq_len=exp_cfg.model.min_seq_len,
                    return_mask=True,
                )
            )
        else:
            batch_input, batch_labels, batch_mask = (
                feature_extraction.get_batched_nm_input(
                    spk_train,
                    num_speakers=exp_cfg.train.loss.num_speakers_per_batch,
                    num_utts_per_speaker=exp_cfg.train.loss.num_utts_per_speaker,
                    seq_len=exp_cfg.model.seq_len,
                    frontend=frontend,
                    specaug_config=exp_cfg.train.specaug,
                    waveform_aug_config=exp_cfg.train.waveform_aug,
                    cache=cache,
                    rng=rng,
                    min_seq_len=exp_cfg.model.min_seq_len,
                    return_mask=True,
                )
            )
        state, loss_val = step_fn(state, batch_input, batch_labels, batch_mask)
        loss_scalar = float(loss_val.item())
        losses.append(loss_scalar)
        if (step + 1) % 50 == 0 or step == 0 or step == exp_cfg.train.num_steps - 1:
            print(f"[step {step + 1}/{exp_cfg.train.num_steps}] loss={loss_scalar:.4f}")

    train_time_sec = round(time.time() - t0, 2)
    metrics_record = {
        "backbone": str(exp_cfg.model.backbone),
        "size_variant": str(exp_cfg.model.size_variant),
        "loss_type": loss_type,
        "scoring_type": str(exp_cfg.train.loss.scoring_type),
        "num_steps": exp_cfg.train.num_steps,
        "train_time_sec": train_time_sec,
        "initial_loss": round(losses[0], 4) if losses else None,
        "final_loss": round(float(np.mean(losses[-20:])), 4) if losses else None,
    }

    if exp_cfg.data.trials_file and os.path.exists(exp_cfg.data.trials_file):
        trials = dataset.load_trials_file(exp_cfg.data.trials_file)
        eval_metrics = evaluation.evaluate_verification_trials(
            state, trials, exp_cfg, feature_cache=cache
        )
        metrics_record["in_domain_eval"] = eval_metrics
        print("Evaluation Results:", json.dumps(eval_metrics, indent=2))

    out_dir = args.output_dir or exp_cfg.train.saved_model_dir
    if out_dir:
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(exp_cfg)
        hf_model = hf_compat.FlaxSpeakerModel(config=hf_cfg, params=state.params)
        hf_model.save_pretrained(out_dir, metrics=metrics_record)
        print(f"Saved HuggingFace-compatible pretrained model to: {out_dir}")


if __name__ == "__main__":
    main()
