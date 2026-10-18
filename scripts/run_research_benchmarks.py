#!/usr/bin/env python3
"""Execute the end-to-end speaker recognition research benchmark suite.

Trains and evaluates models across:
1. Neural Backbones: LSTM, Transformer, Conformer, Mamba, ECAPA-TDNN, ResNet
2. Losses & Scoring: GE2E-Softmax, GE2E-Contrast, Extended-Set Softmax,
   Dr-Vectors (Decision Residual Network), Parameter-Free Attentive Scoring (PFAS),
   ArcFace, CosFace, SphereFace, and Triplet Loss
3. Size Variants: Tiny, Small, Base (all < 100M parameters)
4. Training Regimes:
   - Universal Multi-Domain Training (`LibriSpeech + CN-Celeb2 + VoxCeleb1`,
     381 speakers, 9,447 utterances with domain-balanced sampling)
   - Single-Domain Baselines (`LibriSpeech`, `CN-Celeb2`, `VoxCeleb1`)
5. Multi-Domain Evaluation:
   - Evaluates every model on all 3 speaker-disjoint evaluation domains
     (`LibriSpeech`, `CN-Celeb2`, `VoxCeleb1`) for both 1-utterance and
     3-utterance enrollment, plus Universal Macro-Average and Pooled 8,000-trial
     metrics with 95% bootstrap confidence intervals.
6. Production Export: HuggingFace Hub format (`.safetensors` + `config.json` +
   Model Card) and TFLite (`.tflite` FP32 & INT8)
"""

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import configs  # noqa: E402
from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import evaluation  # noqa: E402
from flaxspeaker import export  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402
from flaxspeaker import neural_net  # noqa: E402
from scripts import prepare_datasets  # noqa: E402


DEFAULT_DATA_ROOT = os.environ.get(
    "FLAXSPEAKER_DATA_ROOT", os.path.expanduser("~/Data/speaker_datasets")
)
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "pretrained_models")
DEFAULT_CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")


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
    pooling_type: str = "atp",
    num_steps: int = 400,
    learning_rate: float = 1.5e-3,
    seq_len: int = 180,
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
    cfg.model.min_seq_len = max(60, seq_len // 2)
    cfg.model.max_seq_len = seq_len
    cfg.model.variable_length_training = True
    cfg.model.pooling.pooling_type = "pfas" if scoring_type == "pfas" else pooling_type
    cfg.model.scoring_type = scoring_type

    lr_val = 1.2e-3 if str(size_variant).lower() == "base" else learning_rate
    cfg.train.num_steps = num_steps
    cfg.train.learning_rate = lr_val
    cfg.train.warmup_steps = max(10, num_steps // 10)
    cfg.train.batch_size = 32
    cfg.train.save_model_frequency = 0
    cfg.train.loss.loss_type = loss_type
    cfg.train.loss.scoring_type = scoring_type
    cfg.train.loss.num_speakers_per_batch = 8
    cfg.train.loss.num_utts_per_speaker = 4
    cfg.loss = cfg.train.loss
    return cfg


def compute_macro_average(domain_metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Compute unweighted macro-average across ('librispeech', 'cnceleb2', 'voxceleb1')."""
    keys = ("librispeech", "cnceleb2", "voxceleb1")
    present = [domain_metrics[k] for k in keys if k in domain_metrics]
    if not present:
        return {}
    out: dict[str, float] = {}
    for metric_key in ("eer_percent", "min_dcf_001", "min_dcf_01", "roc_auc"):
        vals = [float(m[metric_key]) for m in present if metric_key in m]
        if vals:
            digits = 2 if metric_key == "eer_percent" else 4
            out[metric_key] = round(float(np.mean(vals)), digits)
    return out


def evaluate_all_domains(
    state: Any,
    exp_cfg: configs.ExperimentConfig,
    caches: dict[str, feature_extraction.FeatureCache],
    trials_sets: dict[str, list[dataset.TrialPair]],
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]],
    shared_embs: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, Any]:
    """Evaluate a trained model across all 3 evaluation domains + universal pooled & macro-avg."""
    if shared_embs is None:
        shared_embs = {}
    domains_1utt: dict[str, dict[str, Any]] = {}
    domains_3utt: dict[str, dict[str, Any]] = {}

    for dom in ("librispeech", "cnceleb2", "voxceleb1"):
        domains_1utt[dom] = evaluation.evaluate_verification_trials(
            state,
            trials_sets[dom],
            exp_cfg,
            feature_cache=caches[dom],
            precomputed_embeddings=shared_embs,
        )
        if dom in multi_trials_sets:
            domains_3utt[dom] = evaluation.evaluate_multi_enroll_trials(
                state,
                multi_trials_sets[dom],
                exp_cfg,
                feature_cache=caches[dom],
                precomputed_embeddings=shared_embs,
            )

    if "universal" in trials_sets:
        domains_1utt["universal_pooled"] = evaluation.evaluate_verification_trials(
            state,
            trials_sets["universal"],
            exp_cfg,
            feature_cache=caches["universal"],
            precomputed_embeddings=shared_embs,
        )
    if "universal" in multi_trials_sets:
        domains_3utt["universal_pooled"] = evaluation.evaluate_multi_enroll_trials(
            state,
            multi_trials_sets["universal"],
            exp_cfg,
            feature_cache=caches["universal"],
            precomputed_embeddings=shared_embs,
        )

    macro_1utt = compute_macro_average(domains_1utt)
    macro_3utt = compute_macro_average(domains_3utt)
    return {
        "domains_1utt": domains_1utt,
        "universal_macro_avg_1utt": macro_1utt,
        "domains_3utt": domains_3utt,
        "universal_macro_avg_3utt": macro_3utt,
    }


def run_all_benchmarks(
    data_root: str = DEFAULT_DATA_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    configs_dir: str = DEFAULT_CONFIGS_DIR,
    num_steps: int = 400,
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
        if "universal" not in ds_summary:
            ds_summary = prepare_datasets.prepare_all_datasets(data_root=data_root)

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
        cache.precompute_dataset(spk_eval_sets[dname], num_workers=16)
        caches[dname] = cache

    # Universal multi-domain dataset & unified in-memory cache
    univ_info = ds_summary["universal"]
    spk_train_sets["universal"] = dataset.get_csv_spk_to_utts(univ_info["train_csv"])
    spk_eval_sets["universal"] = dataset.get_csv_spk_to_utts(univ_info["eval_csv"])
    trials_sets["universal"] = dataset.load_trials_file(univ_info["trials_file"])
    if "multi_trials_file" in univ_info and os.path.exists(univ_info["multi_trials_file"]):
        multi_trials_sets["universal"] = load_multi_enroll_trials(
            univ_info["multi_trials_file"]
        )
    univ_cache = feature_extraction.FeatureCache(
        frontend, cache_dir=os.path.join(data_root, "feature_cache", "universal")
    )
    for dname in ("librispeech", "cnceleb2", "voxceleb1"):
        univ_cache._cache.update(caches[dname]._cache)
    caches["universal"] = univ_cache
    preloaded_train_domains: set[str] = set()

    # Also load official VoxCeleb1-O trials (4,000 trials on 40 test speakers, for models not trained on VoxCeleb1)
    vox_official_trials = dataset.load_trials_file(
        ds_summary["voxceleb1"]["official_trials_file"]
    )

    # Define experiment suite (25 single-domain + 19 universal multi-domain = 44 models)
    experiments = [
        # Study 1: Single-Domain Neural Backbones on LibriSpeech
        ("ls_lstm_small_ge2e", "librispeech", "lstm", "small", "ge2e_softmax", "cosine", "asp"),
        ("ls_transformer_small_ge2e", "librispeech", "transformer", "small", "ge2e_softmax", "cosine", "atp"),
        ("ls_conformer_small_ge2e", "librispeech", "conformer", "small", "ge2e_softmax", "cosine", "atp"),
        ("ls_mamba_small_ge2e", "librispeech", "mamba", "small", "ge2e_softmax", "cosine", "asp"),
        ("ls_ecapa_tdnn_small_ge2e", "librispeech", "ecapa_tdnn", "small", "ge2e_softmax", "cosine", "asp"),
        ("ls_resnet_small_ge2e", "librispeech", "resnet", "small", "ge2e_softmax", "cosine", "asp"),
        # Study 2: Single-Domain Loss Functions & Scoring Mechanisms (Conformer-Small on LibriSpeech)
        ("ls_conformer_small_triplet", "librispeech", "conformer", "small", "triplet", "cosine", "atp"),
        ("ls_conformer_small_ge2e_contrast", "librispeech", "conformer", "small", "ge2e_contrast", "cosine", "atp"),
        (
            "ls_conformer_small_ext_softmax",
            "librispeech",
            "conformer",
            "small",
            "extended_set_softmax",
            "cosine",
            "atp",
        ),
        (
            "ls_conformer_small_dr_vectors",
            "librispeech",
            "conformer",
            "small",
            "extended_set_softmax",
            "dr_vector",
            "atp",
        ),
        ("ls_conformer_small_pfas", "librispeech", "conformer", "small", "extended_set_softmax", "pfas", "pfas"),
        ("ls_conformer_small_pfas_ge2e", "librispeech", "conformer", "small", "ge2e_softmax", "pfas", "pfas"),
        ("ls_conformer_small_arcface", "librispeech", "conformer", "small", "arcface", "cosine", "atp"),
        ("ls_conformer_small_cosface", "librispeech", "conformer", "small", "cosface", "cosine", "atp"),
        ("ls_conformer_small_sphereface", "librispeech", "conformer", "small", "sphereface", "cosine", "atp"),
        # Study 3: Single-Domain Size Variants (Tiny vs Small vs Base)
        ("ls_conformer_tiny_ge2e", "librispeech", "conformer", "tiny", "ge2e_softmax", "cosine", "atp"),
        ("ls_conformer_base_ge2e", "librispeech", "conformer", "base", "ge2e_softmax", "cosine", "atp"),
        ("ls_ecapa_tdnn_tiny_ge2e", "librispeech", "ecapa_tdnn", "tiny", "ge2e_softmax", "cosine", "asp"),
        ("ls_ecapa_tdnn_base_ge2e", "librispeech", "ecapa_tdnn", "base", "ge2e_softmax", "cosine", "asp"),
        # Study 4: CN-Celeb2 & VoxCeleb1 Single-Domain Baselines
        ("cn_conformer_small_ge2e", "cnceleb2", "conformer", "small", "ge2e_softmax", "cosine", "atp"),
        ("cn_ecapa_tdnn_small_arcface", "cnceleb2", "ecapa_tdnn", "small", "arcface", "cosine", "asp"),
        ("cn_conformer_small_dr_vectors", "cnceleb2", "conformer", "small", "extended_set_softmax", "dr_vector", "atp"),
        ("cn_conformer_small_pfas", "cnceleb2", "conformer", "small", "extended_set_softmax", "pfas", "pfas"),
        ("vox_conformer_small_ge2e", "voxceleb1", "conformer", "small", "ge2e_softmax", "cosine", "atp"),
        (
            "vox_ecapa_tdnn_small_ext_softmax",
            "voxceleb1",
            "ecapa_tdnn",
            "small",
            "extended_set_softmax",
            "cosine",
            "asp",
        ),
        # Study 5: Universal Multi-Domain Neural Backbones (Trained on LibriSpeech + CN-Celeb2 + VoxCeleb1)
        ("univ_conformer_small_ge2e", "universal", "conformer", "small", "ge2e_softmax", "cosine", "atp"),
        ("univ_transformer_small_ge2e", "universal", "transformer", "small", "ge2e_softmax", "cosine", "atp"),
        ("univ_mamba_small_ge2e", "universal", "mamba", "small", "ge2e_softmax", "cosine", "asp"),
        ("univ_lstm_small_ge2e", "universal", "lstm", "small", "ge2e_softmax", "cosine", "asp"),
        ("univ_ecapa_tdnn_small_ge2e", "universal", "ecapa_tdnn", "small", "ge2e_softmax", "cosine", "asp"),
        ("univ_resnet_small_ge2e", "universal", "resnet", "small", "ge2e_softmax", "cosine", "asp"),
        # Study 6: Universal Multi-Domain Loss Functions & Scoring (Conformer-Small on Universal 3-Dataset Mix)
        (
            "univ_conformer_small_ext_softmax",
            "universal",
            "conformer",
            "small",
            "extended_set_softmax",
            "cosine",
            "atp",
        ),
        (
            "univ_conformer_small_dr_vectors",
            "universal",
            "conformer",
            "small",
            "extended_set_softmax",
            "dr_vector",
            "atp",
        ),
        ("univ_conformer_small_pfas", "universal", "conformer", "small", "extended_set_softmax", "pfas", "pfas"),
        ("univ_conformer_small_pfas_ge2e", "universal", "conformer", "small", "ge2e_softmax", "pfas", "pfas"),
        ("univ_conformer_small_arcface", "universal", "conformer", "small", "arcface", "cosine", "atp"),
        ("univ_conformer_small_cosface", "universal", "conformer", "small", "cosface", "cosine", "atp"),
        ("univ_conformer_small_sphereface", "universal", "conformer", "small", "sphereface", "cosine", "atp"),
        ("univ_conformer_small_triplet", "universal", "conformer", "small", "triplet", "cosine", "atp"),
        ("univ_conformer_small_ge2e_contrast", "universal", "conformer", "small", "ge2e_contrast", "cosine", "atp"),
        # Study 7: Universal Multi-Domain Model Scaling (Tiny vs Small vs Base)
        ("univ_conformer_tiny_ge2e", "universal", "conformer", "tiny", "ge2e_softmax", "cosine", "atp"),
        ("univ_conformer_base_ge2e", "universal", "conformer", "base", "ge2e_softmax", "cosine", "atp"),
        ("univ_ecapa_tdnn_tiny_ge2e", "universal", "ecapa_tdnn", "tiny", "ge2e_softmax", "cosine", "asp"),
        ("univ_ecapa_tdnn_base_ge2e", "universal", "ecapa_tdnn", "base", "ge2e_softmax", "cosine", "asp"),
    ]

    results: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hf_collection_url": hf_compat.HF_COLLECTION_URL,
        "datasets": ds_summary,
        "experiments": {},
        "tflite_exports": {},
    }
    report_json_path = os.path.join(
        output_root, f"benchmark_results_{start_idx}_{end_idx}.json"
    )

    tflite_targets = {
        "ls_ecapa_tdnn_small_ge2e",
        "ls_conformer_small_ge2e",
        "ls_lstm_small_ge2e",
        "univ_conformer_small_ge2e",
        "univ_ecapa_tdnn_small_ge2e",
        "univ_lstm_small_ge2e",
    }

    for idx, (exp_name, dname, backbone, size_var, loss_type, scoring_type, pool_type) in enumerate(experiments):
        if idx < start_idx or idx >= end_idx:
            continue
        hf_dir = os.path.join(output_root, exp_name)
        existing_eval = os.path.join(hf_dir, "eval_results.json")
        st_path = os.path.join(hf_dir, "model.safetensors")

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
        yaml_path = os.path.join(configs_dir, f"{exp_name}.yml")
        exp_cfg.to_yaml(yaml_path)

        # If already trained and multi_domain_eval is already present, skip
        if os.path.exists(existing_eval) and os.path.exists(st_path):
            with open(existing_eval, "r", encoding="utf-8") as f:
                cached_rec = json.load(f)
            if "multi_domain_eval" in cached_rec and "domains_1utt" in cached_rec["multi_domain_eval"]:
                print(f"\n[{idx + 1}/{len(experiments)}] Skipping {exp_name} (multi-domain eval already completed)")
                results["experiments"][exp_name] = cached_rec
                continue
            else:
                # Model weights exist! Load weights and run complete multi-domain evaluation without retraining
                print(
                    f"\n[{idx + 1}/{len(experiments)}] Upgrading {exp_name} with full multi-domain evaluation..."
                )
                loaded_hf = hf_compat.FlaxSpeakerModel.from_pretrained(hf_dir)
                state = loaded_hf.state
                shared_embs: dict[str, np.ndarray] = {}
                md_eval = evaluate_all_domains(
                    state, exp_cfg, caches, trials_sets, multi_trials_sets, shared_embs=shared_embs
                )
                primary_dom = "universal_pooled" if dname == "universal" else dname
                cached_rec["in_domain_eval"] = md_eval["domains_1utt"][primary_dom]
                cached_rec["multi_enroll_3utt_eval"] = md_eval["domains_3utt"].get(primary_dom)
                cached_rec["multi_domain_eval"] = md_eval
                if dname == "librispeech":
                    existing_vox_o = cached_rec.get("cross_dataset_eval", {}).get("voxceleb1_o_official")
                    if not existing_vox_o:
                        vox_o_metrics = evaluation.evaluate_verification_trials(
                            state,
                            vox_official_trials,
                            exp_cfg,
                            feature_cache=caches["voxceleb1"],
                            precomputed_embeddings=shared_embs,
                        )
                        cached_rec.setdefault("cross_dataset_eval", {})["voxceleb1_o_official"] = vox_o_metrics
                    cached_rec.setdefault("cross_dataset_eval", {})[
                        "cnceleb2_zero_shot"
                    ] = md_eval["domains_1utt"]["cnceleb2"]
                cached_rec["config_yaml"] = f"configs/{exp_name}.yml"
                cached_rec["hf_pretrained_dir"] = f"pretrained_models/{exp_name}"
                cached_rec["hf_repo_id"] = f"{hf_compat.HF_DEFAULT_NAMESPACE}/{exp_name}"
                cached_rec["hf_url"] = f"https://huggingface.co/{hf_compat.HF_DEFAULT_NAMESPACE}/{exp_name}"
                cached_rec["hf_collection_url"] = hf_compat.HF_COLLECTION_URL
                loaded_hf.save_pretrained(hf_dir, metrics=cached_rec)
                print(
                    f"  [{exp_name}] Macro-Avg 1-Utt EER={md_eval['universal_macro_avg_1utt']['eer_percent']:.2f}% "
                    f"(LS={md_eval['domains_1utt']['librispeech']['eer_percent']:.2f}%, "
                    f"CN={md_eval['domains_1utt']['cnceleb2']['eer_percent']:.2f}%, "
                    f"VOX={md_eval['domains_1utt']['voxceleb1']['eer_percent']:.2f}%)"
                )
                results["experiments"][exp_name] = cached_rec
                with open(report_json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                continue

        print(
            f"\n[{idx + 1}/{len(experiments)}] Training & evaluating {exp_name} "
            f"(dataset={dname}, {backbone}-{size_var}, loss={loss_type}, "
            f"scoring={scoring_type}, pooling={pool_type})..."
        )

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
        if dname not in preloaded_train_domains:
            train_cache.precompute_dataset(spk_train, num_workers=16)
            preloaded_train_domains.add(dname)

        loss_curve: list[float] = []
        var_len = bool(getattr(exp_cfg.model, "variable_length_training", True))
        within_prob = float(getattr(exp_cfg.data, "within_domain_prob", 0.75))
        temp_alpha = float(getattr(exp_cfg.data, "domain_temp_alpha", 0.35))
        for step in range(exp_cfg.train.num_steps):
            if loss_type == "triplet":
                batch_list = []
                mask_list = []
                for _ in range(exp_cfg.train.batch_size):
                    a_u, p_u, n_u = dataset.get_triplet(spk_train)
                    for u in (a_u, p_u, n_u):
                        feat = train_cache.get(u)
                        crop_aug, mask, _ = (
                            feature_extraction.sample_variable_length_features(
                                feat,
                                max_seq_len=exp_cfg.model.seq_len,
                                min_seq_len=exp_cfg.model.min_seq_len,
                                variable_length=var_len,
                                specaug_config=exp_cfg.train.specaug,
                                waveform_aug_config=exp_cfg.train.waveform_aug,
                                rng=rng,
                            )
                        )
                        batch_list.append(crop_aug)
                        mask_list.append(mask)
                import jax.numpy as jnp

                batch_input = jnp.asarray(np.stack(batch_list, axis=0), dtype=jnp.float32)
                batch_mask = jnp.asarray(np.stack(mask_list, axis=0), dtype=jnp.float32)
                batch_labels = None
            elif is_cls:
                batch_input, batch_labels, batch_mask = feature_extraction.get_batched_classification_input(
                    spk_train,
                    speaker_to_id,
                    batch_size=exp_cfg.train.batch_size,
                    seq_len=exp_cfg.model.seq_len,
                    frontend=frontend,
                    specaug_config=exp_cfg.train.specaug,
                    waveform_aug_config=exp_cfg.train.waveform_aug,
                    cache=train_cache,
                    rng=rng,
                    min_seq_len=exp_cfg.model.min_seq_len,
                    variable_length=var_len,
                    return_mask=True,
                    domain_temp_alpha=temp_alpha,
                )
            else:
                batch_input, batch_labels, batch_mask = feature_extraction.get_batched_nm_input(
                    spk_train,
                    num_speakers=exp_cfg.train.loss.num_speakers_per_batch,
                    num_utts_per_speaker=exp_cfg.train.loss.num_utts_per_speaker,
                    seq_len=exp_cfg.model.seq_len,
                    frontend=frontend,
                    specaug_config=exp_cfg.train.specaug,
                    waveform_aug_config=exp_cfg.train.waveform_aug,
                    cache=train_cache,
                    rng=rng,
                    min_seq_len=exp_cfg.model.min_seq_len,
                    variable_length=var_len,
                    return_mask=True,
                    within_domain_prob=within_prob,
                    domain_temp_alpha=temp_alpha,
                )

            state, loss_val = step_fn(state, batch_input, batch_labels, batch_mask)
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

        # Evaluate across all 3 domains + universal pooled & macro-avg
        shared_embs = {}
        md_eval = evaluate_all_domains(
            state, exp_cfg, caches, trials_sets, multi_trials_sets, shared_embs=shared_embs
        )
        primary_dom = "universal_pooled" if dname == "universal" else dname
        in_domain_eval = md_eval["domains_1utt"][primary_dom]
        multi_enroll_eval = md_eval["domains_3utt"].get(primary_dom)

        print(
            f"  [{exp_name}] Multi-Domain 1-Utt EER: "
            f"Macro-Avg={md_eval['universal_macro_avg_1utt']['eer_percent']:.2f}%, "
            f"Pooled={md_eval['domains_1utt']['universal_pooled']['eer_percent']:.2f}% "
            f"(LS={md_eval['domains_1utt']['librispeech']['eer_percent']:.2f}%, "
            f"CN={md_eval['domains_1utt']['cnceleb2']['eer_percent']:.2f}%, "
            f"VOX={md_eval['domains_1utt']['voxceleb1']['eer_percent']:.2f}%)"
        )
        if md_eval.get("universal_macro_avg_3utt"):
            print(
                f"  [{exp_name}] Multi-Domain 3-Utt EER: "
                f"Macro-Avg={md_eval['universal_macro_avg_3utt']['eer_percent']:.2f}%, "
                f"Pooled={md_eval['domains_3utt']['universal_pooled']['eer_percent']:.2f}%"
            )

        cross_eval = {}
        if dname == "librispeech":
            vox_o_metrics = evaluation.evaluate_verification_trials(
                state,
                vox_official_trials,
                exp_cfg,
                feature_cache=caches["voxceleb1"],
                precomputed_embeddings=shared_embs,
            )
            cross_eval["voxceleb1_o_official"] = vox_o_metrics
            cross_eval["cnceleb2_zero_shot"] = md_eval["domains_1utt"]["cnceleb2"]

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
            "multi_domain_eval": md_eval,
            "cross_dataset_eval": cross_eval,
            "config_yaml": f"configs/{exp_name}.yml",
            "hf_pretrained_dir": f"pretrained_models/{exp_name}",
            "hf_repo_id": f"{hf_compat.HF_DEFAULT_NAMESPACE}/{exp_name}",
            "hf_url": f"https://huggingface.co/{hf_compat.HF_DEFAULT_NAMESPACE}/{exp_name}",
            "hf_collection_url": hf_compat.HF_COLLECTION_URL,
        }

        # Save HuggingFace-compatible pretrained directory
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(exp_cfg)
        hf_model = hf_compat.FlaxSpeakerModel(config=hf_cfg, params=state.params)
        hf_model.save_pretrained(hf_dir, metrics=exp_record)

        # Export TFLite for representative models
        if exp_name in tflite_targets:
            for quant in (False, True):
                suffix = "int8" if quant else "fp32"
                tflite_path = os.path.join(hf_dir, f"model_{suffix}.tflite")
                tf_info = export.export_to_tflite(
                    state, exp_cfg, tflite_path, quantize_int8=quant
                )
                runner = export.TFLiteSpeakerRunner(tflite_path)
                sample_utt = trials_sets["librispeech"][0][1]
                sample_feat = caches["librispeech"].get(sample_utt)[: exp_cfg.model.seq_len]
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
                tf_info["tflite_path"] = f"pretrained_models/{exp_name}/model_{suffix}.tflite"
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
    parser.add_argument("--num_steps", type=int, default=400)
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
