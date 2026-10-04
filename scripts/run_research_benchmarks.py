#!/usr/bin/env python3
"""Execute the end-to-end speaker recognition research benchmark suite.

Trains and evaluates models across:
1. Neural Backbones: LSTM, Transformer, Conformer, Mamba, ECAPA-TDNN, ResNet
2. Losses & Scoring: GE2E-Softmax, GE2E-Contrast, Extended-Set Softmax,
   Dr-Vectors (Decision Residual Network), Parameter-Free Attentive Scoring (PFAS),
   ArcFace, CosFace, SphereFace, and Triplet Loss
3. Size Variants: Tiny, Small, Base (all < 100M parameters)
4. Datasets: LibriSpeech, CN-Celeb2, VoxCeleb1 (in-domain + zero-shot cross-dataset)
5. Production Export: HuggingFace Hub format (`.safetensors` + `config.json`)
   and TFLite (`.tflite` FP32 & INT8)
"""

import argparse
import datetime
import json
import os
import time
from typing import Any

import numpy as np

from flaxspeaker import configs
from flaxspeaker import dataset
from flaxspeaker import evaluation
from flaxspeaker import export
from flaxspeaker import feature_extraction
from flaxspeaker import hf_compat
from flaxspeaker import neural_net
from scripts import prepare_datasets


DEFAULT_DATA_ROOT = "/usr/local/google/home/quanw/Data/speaker_datasets"
DEFAULT_OUTPUT_ROOT = "/usr/local/google/home/quanw/Code/github/FlaxSpeaker/pretrained_models"
DEFAULT_CONFIGS_DIR = "/usr/local/google/home/quanw/Code/github/FlaxSpeaker/configs"


def load_multi_enroll_trials(path: str) -> list[dataset.MultiEnrollTrial]:
    trials: list[dataset.MultiEnrollTrial] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            lbl = int(parts[0])
            e_list = [x for x in parts[1].split(",") if x]
            t_u = parts[2]
            trials.append((lbl, e_list, t_u))
    return trials


def make_exp_config(
    dataset_name: str,
    train_csv: str,
    eval_csv: str,
    trials_file: str,
    backbone: str,
    size_variant: str,
    loss_type: str,
    scoring_type: str = "cosine",
    pooling_type: str = "asp",
    num_steps: int = 300,
    learning_rate: float = 1.5e-3,
    seq_len: int = 160,
) -> configs.ExperimentConfig:
    cfg = configs.ExperimentConfig()
    cfg.data.dataset_type = dataset_name
    cfg.data.dataset_name = dataset_name
    cfg.data.train_csv = train_csv
    cfg.data.test_csv = eval_csv
    cfg.data.trials_file = trials_file

    cfg.frontend.feature_type = configs.FeatureType.LOG_MEL.value
    cfg.frontend.n_mels = 80
    cfg.frontend.n_mfcc = 80
    cfg.frontend.apply_cmvn = True
    cfg.vad.enabled = True
    cfg.vad.mode = configs.VadMode.ENERGY.value

    cfg.model.backbone = backbone
    cfg.model.apply_size_variant(size_variant)
    cfg.model.seq_len = seq_len
    cfg.model.pooling.pooling_type = "pfas" if scoring_type == "pfas" else pooling_type
    cfg.model.scoring_type = scoring_type

    cfg.train.num_steps = num_steps
    cfg.train.learning_rate = learning_rate
    cfg.train.warmup_steps = max(10, num_steps // 10)
    cfg.train.batch_size = 32
    cfg.train.save_model_frequency = 0
    cfg.train.loss.loss_type = loss_type
    cfg.train.loss.scoring_type = scoring_type
    cfg.train.loss.num_speakers_per_batch = 8
    cfg.train.loss.num_utts_per_speaker = 4
    cfg.loss = cfg.train.loss
    return cfg


def run_all_benchmarks(
    data_root: str = DEFAULT_DATA_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    configs_dir: str = DEFAULT_CONFIGS_DIR,
    num_steps: int = 300,
    start_idx: int = 0,
    end_idx: int = 100,
) -> dict[str, Any]:
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    splits_summary_path = os.path.join(data_root, "splits", "dataset_summary.json")
    if not os.path.exists(splits_summary_path):
        ds_summary = prepare_datasets.prepare_all_datasets(data_root=data_root)
    else:
        with open(splits_summary_path, "r", encoding="utf-8") as f:
            ds_summary = json.load(f)

    # Load feature caches into memory for fast training & evaluation
    frontend_cfg = configs.FrontendConfig(
        feature_type=configs.FeatureType.LOG_MEL.value,
        sample_rate=16000,
        n_mels=80,
        apply_cmvn=True,
    )
    vad_cfg = configs.VadConfig(mode=configs.VadMode.ENERGY.value)
    frontend = feature_extraction.AudioFrontend(frontend_cfg, vad_cfg)

    caches: dict[str, feature_extraction.FeatureCache] = {}
    spk_train_sets: dict[str, dataset.SpkToUtts] = {}
    spk_eval_sets: dict[str, dataset.SpkToUtts] = {}
    trials_sets: dict[str, list[dataset.TrialPair]] = {}
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]] = {}

    for dname in ("librispeech", "cnceleb2", "voxceleb1"):
        info = ds_summary[dname]
        spk_train_sets[dname] = dataset.get_csv_spk_to_utts(info["train_csv"])
        spk_eval_sets[dname] = dataset.get_csv_spk_to_utts(info["eval_csv"])
        trials_sets[dname] = dataset.load_trials_file(info["trials_file"])
        if "multi_trials_file" in info and os.path.exists(info["multi_trials_file"]):
            multi_trials_sets[dname] = load_multi_enroll_trials(info["multi_trials_file"])

        c_dir = os.path.join(data_root, "feature_cache", dname)
        cache = feature_extraction.FeatureCache(frontend, cache_dir=c_dir)
        cache.precompute_dataset(spk_train_sets[dname], num_workers=16)
        cache.precompute_dataset(spk_eval_sets[dname], num_workers=16)
        caches[dname] = cache

    # Also load official VoxCeleb1-O trials
    vox_official_trials = dataset.load_trials_file(
        ds_summary["voxceleb1"]["official_trials_file"]
    )
    vox_official_utts = {
        "official": sorted({u for _, u1, u2 in vox_official_trials for u in (u1, u2)})
    }
    caches["voxceleb1"].precompute_dataset(vox_official_utts, num_workers=16)

    # Define experiment suite
    experiments = [
        # Study 1: Neural Backbones on LibriSpeech (evaluated in-domain + zero-shot CN-Celeb2 & VoxCeleb1-O)
        ("ls_lstm_small_ge2e", "librispeech", "lstm", "small", "ge2e_softmax", "cosine", "asp"),
        ("ls_transformer_small_ge2e", "librispeech", "transformer", "small", "ge2e_softmax", "cosine", "asp"),
        ("ls_conformer_small_ge2e", "librispeech", "conformer", "small", "ge2e_softmax", "cosine", "asp"),
        ("ls_mamba_small_ge2e", "librispeech", "mamba", "small", "ge2e_softmax", "cosine", "asp"),
        ("ls_ecapa_tdnn_small_ge2e", "librispeech", "ecapa_tdnn", "small", "ge2e_softmax", "cosine", "asp"),
        ("ls_resnet_small_ge2e", "librispeech", "resnet", "small", "ge2e_softmax", "cosine", "asp"),
        # Study 2: Loss Functions & Scoring Mechanisms (Conformer-Small on LibriSpeech)
        ("ls_conformer_small_triplet", "librispeech", "conformer", "small", "triplet", "cosine", "asp"),
        ("ls_conformer_small_ge2e_contrast", "librispeech", "conformer", "small", "ge2e_contrast", "cosine", "asp"),
        (
            "ls_conformer_small_ext_softmax",
            "librispeech",
            "conformer",
            "small",
            "extended_set_softmax",
            "cosine",
            "asp",
        ),
        (
            "ls_conformer_small_dr_vectors",
            "librispeech",
            "conformer",
            "small",
            "extended_set_softmax",
            "dr_vector",
            "asp",
        ),
        ("ls_conformer_small_pfas", "librispeech", "conformer", "small", "extended_set_softmax", "pfas", "pfas"),
        ("ls_conformer_small_arcface", "librispeech", "conformer", "small", "arcface", "cosine", "asp"),
        ("ls_conformer_small_cosface", "librispeech", "conformer", "small", "cosface", "cosine", "asp"),
        ("ls_conformer_small_sphereface", "librispeech", "conformer", "small", "sphereface", "cosine", "asp"),
        # Study 3: Size Variants (Tiny vs Small vs Base, all < 100M parameters)
        ("ls_conformer_tiny_ge2e", "librispeech", "conformer", "tiny", "ge2e_softmax", "cosine", "asp"),
        ("ls_conformer_base_ge2e", "librispeech", "conformer", "base", "ge2e_softmax", "cosine", "asp"),
        ("ls_ecapa_tdnn_tiny_ge2e", "librispeech", "ecapa_tdnn", "tiny", "ge2e_softmax", "cosine", "asp"),
        ("ls_ecapa_tdnn_base_ge2e", "librispeech", "ecapa_tdnn", "base", "ge2e_softmax", "cosine", "asp"),
        # Study 4: CN-Celeb2 & VoxCeleb1 In-Domain Training
        ("cn_conformer_small_ge2e", "cnceleb2", "conformer", "small", "ge2e_softmax", "cosine", "asp"),
        ("cn_ecapa_tdnn_small_arcface", "cnceleb2", "ecapa_tdnn", "small", "arcface", "cosine", "asp"),
        ("cn_conformer_small_dr_vectors", "cnceleb2", "conformer", "small", "extended_set_softmax", "dr_vector", "asp"),
        ("cn_conformer_small_pfas", "cnceleb2", "conformer", "small", "extended_set_softmax", "pfas", "pfas"),
        ("vox_conformer_small_ge2e", "voxceleb1", "conformer", "small", "ge2e_softmax", "cosine", "asp"),
        (
            "vox_ecapa_tdnn_small_ext_softmax",
            "voxceleb1",
            "ecapa_tdnn",
            "small",
            "extended_set_softmax",
            "cosine",
            "asp",
        ),
    ]

    results: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "datasets": ds_summary,
        "experiments": {},
        "tflite_exports": {},
    }
    report_json_path = os.path.join(
        output_root, f"benchmark_results_{start_idx}_{end_idx}.json"
    )

    for idx, (exp_name, dname, backbone, size_var, loss_type, scoring_type, pool_type) in enumerate(experiments):
        if idx < start_idx or idx >= end_idx:
            continue
        hf_dir = os.path.join(output_root, exp_name)
        existing_eval = os.path.join(hf_dir, "eval_results.json")
        if os.path.exists(existing_eval):
            print(f"\n[{idx + 1}/{len(experiments)}] Skipping {exp_name} (already completed)")
            with open(existing_eval, "r", encoding="utf-8") as f:
                results["experiments"][exp_name] = json.load(f)
            continue

        print(
            f"\n[{idx + 1}/{len(experiments)}] Running {exp_name} "
            f"({backbone}-{size_var}, loss={loss_type}, scoring={scoring_type})..."
        )
        d_info = ds_summary[dname]
        exp_cfg = make_exp_config(
            dataset_name=dname,
            train_csv=d_info["train_csv"],
            eval_csv=d_info["eval_csv"],
            trials_file=d_info["trials_file"],
            backbone=backbone,
            size_variant=size_var,
            loss_type=loss_type,
            scoring_type=scoring_type,
            pooling_type=pool_type,
            num_steps=num_steps,
        )
        # Save YAML preset config
        yaml_path = os.path.join(configs_dir, f"{exp_name}.yml")
        exp_cfg.to_yaml(yaml_path)

        # Train
        t_start = time.time()
        is_cls = loss_type in ("arcface", "cosface", "sphereface", "softmax")
        spk_train = spk_train_sets[dname]
        speaker_to_id = {s: i for i, s in enumerate(sorted(spk_train.keys()))}
        num_classes = len(speaker_to_id) if is_cls else 0

        _, state = neural_net.get_speaker_encoder(exp_cfg, num_classes=num_classes)
        step_fn = neural_net.make_modern_train_step(exp_cfg)
        rng = np.random.default_rng(exp_cfg.train.seed)
        train_cache = caches[dname]

        loss_curve: list[float] = []
        for step in range(exp_cfg.train.num_steps):
            if loss_type == "triplet":
                from flaxspeaker import specaug

                batch_list = []
                for _ in range(exp_cfg.train.batch_size):
                    a_u, p_u, n_u = dataset.get_triplet(spk_train)
                    for u in (a_u, p_u, n_u):
                        feat = train_cache.get(u)
                        crop = feature_extraction.random_crop_or_pad(
                            feat, exp_cfg.model.seq_len, rng=rng
                        )
                        crop = specaug.apply_specaug(crop, exp_cfg.train.specaug, rng=rng)
                        batch_list.append(crop)
                import jax.numpy as jnp

                batch_input = jnp.asarray(np.stack(batch_list, axis=0), dtype=jnp.float32)
                batch_labels = None
            elif is_cls:
                batch_input, batch_labels = feature_extraction.get_batched_classification_input(
                    spk_train,
                    speaker_to_id,
                    batch_size=exp_cfg.train.batch_size,
                    seq_len=exp_cfg.model.seq_len,
                    frontend=frontend,
                    specaug_config=exp_cfg.train.specaug,
                    cache=train_cache,
                    rng=rng,
                )
            else:
                batch_input, batch_labels = feature_extraction.get_batched_nm_input(
                    spk_train,
                    num_speakers=exp_cfg.train.loss.num_speakers_per_batch,
                    num_utts_per_speaker=exp_cfg.train.loss.num_utts_per_speaker,
                    seq_len=exp_cfg.model.seq_len,
                    frontend=frontend,
                    specaug_config=exp_cfg.train.specaug,
                    cache=train_cache,
                    rng=rng,
                )

            state, loss_val = step_fn(state, batch_input, batch_labels)
            loss_scalar = float(loss_val.item())
            loss_curve.append(loss_scalar)
            if (step + 1) % 100 == 0 or step == 0 or step == exp_cfg.train.num_steps - 1:
                print(f"  [{exp_name}] step {step + 1}/{exp_cfg.train.num_steps} loss={loss_scalar:.4f}")

        train_time_sec = round(time.time() - t_start, 2)

        # Count encoder parameters (excluding training-only class prototype matrix)
        enc_only_params = {
            k: v for k, v in state.params.items() if k != "_aux_class_weights"
        }
        num_params = neural_net.count_parameters(enc_only_params)

        # Evaluate in-domain
        in_domain_eval = evaluation.evaluate_verification_trials(
            state, trials_sets[dname], exp_cfg, feature_cache=caches[dname]
        )
        print(
            f"  [{exp_name}] In-domain ({dname}) EER={in_domain_eval['eer_percent']:.2f}%, "
            f"minDCF={in_domain_eval['min_dcf_001']:.4f}, AUC={in_domain_eval['roc_auc']:.4f}"
        )

        # Evaluate 3-utterance multi-enrollment if available
        multi_enroll_eval = None
        if dname in multi_trials_sets:
            multi_enroll_eval = evaluation.evaluate_multi_enroll_trials(
                state, multi_trials_sets[dname], exp_cfg, feature_cache=caches[dname]
            )
            print(
                f"  [{exp_name}] 3-Utt Multi-Enroll ({dname}) EER={multi_enroll_eval['eer_percent']:.2f}%, "
                f"minDCF={multi_enroll_eval['min_dcf_001']:.4f}"
            )

        # Cross-dataset zero-shot evaluation on official VoxCeleb1-O and CN-Celeb2
        cross_eval = {}
        if dname == "librispeech":
            vox_o_metrics = evaluation.evaluate_verification_trials(
                state, vox_official_trials, exp_cfg, feature_cache=caches["voxceleb1"]
            )
            cn_metrics = evaluation.evaluate_verification_trials(
                state, trials_sets["cnceleb2"], exp_cfg, feature_cache=caches["cnceleb2"]
            )
            cross_eval["voxceleb1_o_official"] = vox_o_metrics
            cross_eval["cnceleb2_zero_shot"] = cn_metrics
            print(
                f"  [{exp_name}] Zero-shot VoxCeleb1-O EER={vox_o_metrics['eer_percent']:.2f}%, "
                f"CN-Celeb2 EER={cn_metrics['eer_percent']:.2f}%"
            )

        exp_record = {
            "experiment_name": exp_name,
            "train_dataset": dname,
            "backbone": backbone,
            "size_variant": size_var,
            "pooling_type": exp_cfg.model.pooling.pooling_type,
            "loss_type": loss_type,
            "scoring_type": scoring_type,
            "embedding_dim": exp_cfg.model.output_embedding_dim,
            "num_parameters": num_params,
            "num_steps": exp_cfg.train.num_steps,
            "train_time_sec": train_time_sec,
            "initial_loss": round(loss_curve[0], 4),
            "final_loss": round(float(np.mean(loss_curve[-20:])), 4),
            "in_domain_eval": in_domain_eval,
            "multi_enroll_3utt_eval": multi_enroll_eval,
            "cross_dataset_eval": cross_eval,
            "config_yaml": yaml_path,
        }

        # Save HuggingFace-compatible pretrained directory
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(exp_cfg)
        hf_model = hf_compat.FlaxSpeakerModel(config=hf_cfg, params=state.params)
        hf_model.save_pretrained(hf_dir, metrics=exp_record)
        exp_record["hf_pretrained_dir"] = hf_dir

        # Export TFLite for representative models
        if exp_name in ("ls_ecapa_tdnn_small_ge2e", "ls_conformer_small_ge2e", "ls_lstm_small_ge2e"):
            for quant in (False, True):
                suffix = "int8" if quant else "fp32"
                tflite_path = os.path.join(hf_dir, f"model_{suffix}.tflite")
                tf_info = export.export_to_tflite(
                    state, exp_cfg, tflite_path, quantize_int8=quant
                )
                runner = export.TFLiteSpeakerRunner(tflite_path)
                sample_utt = trials_sets[dname][0][1]
                sample_feat = caches[dname].get(sample_utt)[: exp_cfg.model.seq_len]
                t_tfl0 = time.time()
                tfl_emb = runner(sample_feat)
                tfl_latency_ms = (time.time() - t_tfl0) * 1000.0
                flax_emb = np.asarray(
                    state.apply_fn(
                        {"params": state.params}, sample_feat[None, :, :]
                    )[0]
                )
                cos_fidelity = float(
                    np.dot(tfl_emb, flax_emb)
                    / (np.linalg.norm(tfl_emb) * np.linalg.norm(flax_emb) + 1e-6)
                )
                tf_info["latency_ms"] = round(tfl_latency_ms, 3)
                tf_info["cosine_fidelity_vs_flax"] = round(cos_fidelity, 6)
                results["tflite_exports"][f"{exp_name}_{suffix}"] = tf_info

        results["experiments"][exp_name] = exp_record
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== SHARD [{start_idx}..{end_idx}] COMPLETED ===")
    print(f"Results saved to: {report_json_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=100)
    args = parser.parse_args()
    run_all_benchmarks(
        data_root=args.data_root,
        output_root=args.output_root,
        num_steps=args.num_steps,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
