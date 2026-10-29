#!/usr/bin/env python3
"""Execute the end-to-end speaker recognition research benchmark suite.

Trains and evaluates all 44 models to convergence under the Full Convergence +
2-Epoch Safety Margin (`S_safe = X + 2 epochs`) schedule across:
1. Neural Backbones: LSTM, Transformer, Conformer, Mamba, ECAPA-TDNN, ResNet
2. Losses & Scoring: GE2E-Softmax, GE2E-Contrast, Extended-Set Softmax,
   Dr-Vectors (Decision Residual Network), Parameter-Free Attentive Scoring (PFAS),
   ArcFace, CosFace, SphereFace, and Triplet Loss
3. Size Variants: Tiny, Small, Base (all < 100M parameters)
4. Training Regimes:
   - Universal Multi-Domain Training (`LibriSpeech + CN-Celeb2 + VoxCeleb1`,
     381 speakers, 9,447 utterances with domain-balanced sampling):
     `S_safe = 4,600` steps (`X = 4,000 + 600`, `15.58` epochs)
   - Single-Domain Baselines:
     - `LibriSpeech` (232 speakers, 5,788 utterances):
       `S_safe = 2,400` steps (`13.27` epochs) for 16 Tiny/Small 1D models;
       `S_safe = 2,600` steps (`14.37` epochs) for 3 Base / 2D ResNet models
     - `CN-Celeb2` (119 speakers, 2,909 utterances):
       `S_safe = 1,600` steps (`17.60` epochs) for metric/scoring losses;
       `S_safe = 1,800` steps (`19.80` epochs) for ArcFace
     - `VoxCeleb1` (30 speakers, 750 utterances):
       `S_safe = 1,000` steps (`42.67` epochs)
5. Multi-Domain Evaluation & Convergence Tracking:
   - Evaluates every model on all 3 speaker-disjoint evaluation domains
     (`LibriSpeech`, `CN-Celeb2`, `VoxCeleb1`) for both 1-utterance and
     3-utterance enrollment, plus Universal Macro-Average and Pooled 8,000-trial
     metrics with 95% bootstrap confidence intervals.
   - Records intermediate validation checkpoints at step `400`, step `X` (knee),
     and step `S_safe` (final) in `eval_results.json["convergence_checkpoints"]`,
     plus 100-step loss curve points (`loss_curve_100`).
6. Production Export: HuggingFace Hub format (`.safetensors` + `config.json` +
   Model Card) and TFLite (`.tflite` FP32 & INT8)
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from typing import Any, Optional

import jax.numpy as jnp
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
from flaxspeaker import resources  # noqa: E402
from flaxspeaker import shard_cache  # noqa: E402
from scripts import prepare_datasets  # noqa: E402


DEFAULT_DATA_ROOT = os.environ.get(
    "FLAXSPEAKER_DATA_ROOT", os.path.expanduser("~/Data/speaker_datasets")
)
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "pretrained_models")
DEFAULT_CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")

DOMAIN_UTT_COUNTS = {
    "universal": 9447,
    "librispeech": 5788,
    "cnceleb2": 2909,
    "voxceleb1": 750,
}

# Define experiment suite (25 single-domain + 19 universal multi-domain = 44 models)
EXPERIMENTS = [
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

TFLITE_TARGETS = {
    "ls_ecapa_tdnn_small_ge2e",
    "ls_conformer_small_ge2e",
    "ls_lstm_small_ge2e",
    "univ_conformer_small_ge2e",
    "univ_ecapa_tdnn_small_ge2e",
    "univ_lstm_small_ge2e",
}


def get_model_schedule(
    exp_name: str,
    dataset_name: str,
    backbone: str,
    size_variant: str,
    loss_type: str,
) -> dict[str, int]:
    """Return the Phase-2 converged training schedule (`S_safe = X + 2 epochs`)."""
    dname = str(dataset_name).lower()
    bb = str(backbone).lower()
    sz = str(size_variant).lower()
    lt = str(loss_type).lower()

    if dname == "universal":
        num_steps = 4600
        knee_step = 4000
        safety_margin_steps = 600
        warmup_steps = 460
        margin_warmup_steps = 500 if lt in ("arcface", "cosface") else 0
    elif dname == "librispeech":
        if sz == "base" or bb == "resnet":
            num_steps = 2600
            knee_step = 2200
            safety_margin_steps = 400
            warmup_steps = 260
            margin_warmup_steps = 0
        else:
            num_steps = 2400
            knee_step = 2000
            safety_margin_steps = 400
            warmup_steps = 240
            margin_warmup_steps = 300 if lt in ("arcface", "cosface") else 0
    elif dname == "cnceleb2":
        if lt in ("arcface", "cosface"):
            num_steps = 1800
            knee_step = 1600
            safety_margin_steps = 200
            warmup_steps = 180
            margin_warmup_steps = 250
        else:
            num_steps = 1600
            knee_step = 1400
            safety_margin_steps = 200
            warmup_steps = 160
            margin_warmup_steps = 0
    elif dname == "voxceleb1":
        num_steps = 1000
        knee_step = 900
        safety_margin_steps = 100
        warmup_steps = 100
        margin_warmup_steps = 0
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name} for {exp_name}")

    return {
        "num_steps": num_steps,
        "knee_step": knee_step,
        "safety_margin_steps": safety_margin_steps,
        "warmup_steps": warmup_steps,
        "margin_warmup_steps": margin_warmup_steps,
    }


def load_multi_enroll_trials(path: str) -> list[dataset.MultiEnrollTrial]:
    path = dataset.resolve_data_path(path)
    trials: list[dataset.MultiEnrollTrial] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            lbl = int(parts[0])
            e_list = [dataset.resolve_data_path(x) for x in parts[1].split(",") if x]
            t_u = dataset.resolve_data_path(parts[2])
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
    num_steps: int = 4600,
    warmup_steps: Optional[int] = None,
    margin_warmup_steps: int = 0,
    learning_rate: float = 1.5e-3,
    seq_len: int = 180,
) -> configs.ExperimentConfig:
    cfg = configs.ExperimentConfig()
    cfg.data.dataset_type = dataset_name
    cfg.data.dataset_name = dataset_name
    # Store portable relative paths in configs/*.yml & config.json;
    # dataset.resolve_data_path() resolves them to FLAXSPEAKER_DATA_ROOT.
    cfg.data.train_csv = f"data/speaker_datasets/splits/{dataset_name}/train.csv"
    cfg.data.test_csv = f"data/speaker_datasets/splits/{dataset_name}/eval.csv"
    cfg.data.trials_file = f"data/speaker_datasets/splits/{dataset_name}/trials.tsv"
    cfg.data.sharded_cache_dir = (
        f"data/speaker_datasets/sharded_caches/controlled_v050/{dataset_name}"
    )

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
    cfg.model.max_eval_seq_len = 0
    cfg.model.variable_length_training = True
    cfg.model.ecapa_tdnn.use_batch_norm = True
    cfg.model.ecapa_tdnn.norm_after_activation = True
    cfg.model.resnet.use_batch_norm = True
    cfg.model.pooling.pooling_type = "pfas" if scoring_type == "pfas" else pooling_type
    cfg.model.scoring_type = scoring_type
    cfg.eval.full_sequence_inference = True
    cfg.eval.max_eval_frames_cap = 3000

    is_cls = str(loss_type).lower() in ("arcface", "cosface", "sphereface", "softmax")
    if is_cls:
        lr_val = 1.5e-3 if str(size_variant).lower() == "base" else 2.0e-3
        b_size = 64
    else:
        lr_val = 1.2e-3 if str(size_variant).lower() == "base" else learning_rate
        b_size = 32
    cfg.train.num_steps = int(num_steps)
    cfg.train.learning_rate = lr_val
    # End value of the cosine schedule used for every released checkpoint
    cfg.train.min_learning_rate = max(1e-5, 0.01 * lr_val)
    cfg.train.warmup_steps = (
        int(warmup_steps) if warmup_steps is not None else max(10, int(num_steps) // 10)
    )
    cfg.train.batch_size = b_size
    cfg.train.save_model_frequency = 0
    cfg.train.loss.loss_type = loss_type
    cfg.train.loss.scoring_type = scoring_type
    cfg.train.loss.num_speakers_per_batch = 8
    cfg.train.loss.num_utts_per_speaker = 4
    cfg.train.loss.margin_warmup_steps = int(margin_warmup_steps)
    if loss_type == "ge2e_contrast":
        cfg.train.loss.ge2e_init_b = -1.0
        cfg.train.loss.ge2e_contrast_surrogate = True
        cfg.train.loss.ge2e_contrast_b_grad_scale = 0.5
    cfg.loss = cfg.train.loss
    cfg.export.sequence_length = int(seq_len)
    return cfg


def compute_macro_average(domain_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute unweighted macro-average across ('librispeech', 'cnceleb2', 'voxceleb1')."""
    keys = ("librispeech", "cnceleb2", "voxceleb1")
    present = [domain_metrics[k] for k in keys if k in domain_metrics]
    if not present:
        return {}
    out: dict[str, Any] = {}
    for metric_key in ("eer_percent", "min_dcf_001", "min_dcf_01", "roc_auc"):
        vals = [float(m[metric_key]) for m in present if metric_key in m]
        if vals:
            digits = 2 if metric_key == "eer_percent" else 4
            out[metric_key] = round(float(np.mean(vals)), digits)
    asn_present = [m["as_norm"] for m in present if isinstance(m.get("as_norm"), dict)]
    if len(asn_present) == len(present):
        asn_out: dict[str, float] = {}
        for metric_key in ("eer_percent", "min_dcf_001", "min_dcf_01", "roc_auc"):
            vals = [float(m[metric_key]) for m in asn_present if metric_key in m]
            if vals:
                digits = 2 if metric_key == "eer_percent" else 4
                asn_out[metric_key] = round(float(np.mean(vals)), digits)
        if asn_out:
            out["as_norm"] = asn_out
    return out


def compute_training_cohort_embeddings(
    state: Any,
    exp_cfg: configs.ExperimentConfig,
    dname: str,
    spk_train: dataset.SpkToUtts,
    cache: feature_extraction.FeatureCache,
) -> Optional[np.ndarray]:
    """Compute speaker-centroid imposter cohort from speaker-disjoint training data."""
    scoring_t = str(
        getattr(exp_cfg.model.scoring_type, "value", exp_cfg.model.scoring_type)
    )
    if scoring_t != "cosine":
        return None
    cohort_spk_map: Optional[dataset.SpkToUtts] = None
    if exp_cfg.data.sharded_cache_dir:
        resolved_shard = dataset.resolve_data_path(exp_cfg.data.sharded_cache_dir)
        cohort_json = os.path.join(resolved_shard, "clean_cohort_utts.json")
        if os.path.isfile(cohort_json):
            with open(cohort_json, "r", encoding="utf-8") as cf:
                raw_map = json.load(cf)
            cohort_spk_map = {
                s: [dataset.resolve_data_path(u) for u in utts]
                for s, utts in raw_map.items()
            }
    if not cohort_spk_map:
        cohort_spk_map = {s: sorted(utts)[:5] for s, utts in spk_train.items()}
    return evaluation.compute_speaker_centroid_cohort(
        state,
        cohort_spk_map,
        exp_cfg,
        feature_cache=cache,
        max_utts_per_speaker=5,
    )


def evaluate_all_domains(
    state: Any,
    exp_cfg: configs.ExperimentConfig,
    caches: dict[str, feature_extraction.FeatureCache],
    trials_sets: dict[str, list[dataset.TrialPair]],
    multi_trials_sets: dict[str, list[dataset.MultiEnrollTrial]],
    shared_embs: Optional[dict[str, np.ndarray]] = None,
    cohort_embeddings: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Evaluate a trained model across all 3 evaluation domains + universal pooled & macro-avg."""
    if shared_embs is None:
        shared_embs = {}
    with_asn = cohort_embeddings is not None
    domains_1utt: dict[str, dict[str, Any]] = {}
    domains_3utt: dict[str, dict[str, Any]] = {}

    for dom in ("librispeech", "cnceleb2", "voxceleb1"):
        domains_1utt[dom] = evaluation.evaluate_verification_trials(
            state,
            trials_sets[dom],
            exp_cfg,
            feature_cache=caches[dom],
            precomputed_embeddings=shared_embs,
            cohort_embeddings=cohort_embeddings,
            include_as_norm_breakdown=with_asn,
        )
        if dom in multi_trials_sets:
            domains_3utt[dom] = evaluation.evaluate_multi_enroll_trials(
                state,
                multi_trials_sets[dom],
                exp_cfg,
                feature_cache=caches[dom],
                precomputed_embeddings=shared_embs,
                cohort_embeddings=cohort_embeddings,
                include_as_norm_breakdown=with_asn,
            )

    if "universal" in trials_sets:
        domains_1utt["universal_pooled"] = evaluation.evaluate_verification_trials(
            state,
            trials_sets["universal"],
            exp_cfg,
            feature_cache=caches["universal"],
            precomputed_embeddings=shared_embs,
            cohort_embeddings=cohort_embeddings,
            include_as_norm_breakdown=with_asn,
        )
    if "universal" in multi_trials_sets:
        domains_3utt["universal_pooled"] = evaluation.evaluate_multi_enroll_trials(
            state,
            multi_trials_sets["universal"],
            exp_cfg,
            feature_cache=caches["universal"],
            precomputed_embeddings=shared_embs,
            cohort_embeddings=cohort_embeddings,
            include_as_norm_breakdown=with_asn,
        )

    macro_1utt = compute_macro_average(domains_1utt)
    macro_3utt = compute_macro_average(domains_3utt)
    return {
        "domains_1utt": domains_1utt,
        "universal_macro_avg_1utt": macro_1utt,
        "domains_3utt": domains_3utt,
        "universal_macro_avg_3utt": macro_3utt,
    }


def summarize_checkpoint_eval(
    step_num: int,
    dname: str,
    loss_curve: list[float],
    md_eval: dict[str, Any],
    batch_size: int = 32,
) -> dict[str, Any]:
    """Create a compact convergence checkpoint summary for `eval_results.json`."""
    train_utts = DOMAIN_UTT_COUNTS.get(dname, 9447)
    primary_dom = "universal_pooled" if dname == "universal" else dname
    d1 = md_eval["domains_1utt"]
    d3 = md_eval.get("domains_3utt", {})
    return {
        "step": int(step_num),
        "epochs": round((step_num * batch_size) / max(train_utts, 1), 2),
        "mean_loss_last_100": round(
            float(np.mean(loss_curve[-min(len(loss_curve), 100):])), 4
        ),
        "in_domain_1utt_eer_percent": d1[primary_dom]["eer_percent"],
        "in_domain_3utt_eer_percent": (
            d3[primary_dom]["eer_percent"] if primary_dom in d3 else None
        ),
        "pooled_1utt_eer_percent": d1["universal_pooled"]["eer_percent"],
        "pooled_3utt_eer_percent": (
            d3["universal_pooled"]["eer_percent"] if "universal_pooled" in d3 else None
        ),
        "macro_avg_1utt_eer_percent": md_eval["universal_macro_avg_1utt"]["eer_percent"],
        "macro_avg_3utt_eer_percent": md_eval.get("universal_macro_avg_3utt", {}).get(
            "eer_percent"
        ),
        "domains_1utt_eer_percent": {
            k: d1[k]["eer_percent"]
            for k in ("librispeech", "cnceleb2", "voxceleb1")
            if k in d1
        },
        "domains_3utt_eer_percent": {
            k: d3[k]["eer_percent"]
            for k in ("librispeech", "cnceleb2", "voxceleb1")
            if k in d3
        },
        "in_domain_1utt_full": d1[primary_dom],
        "in_domain_3utt_full": d3.get(primary_dom),
        "universal_macro_avg_1utt": md_eval["universal_macro_avg_1utt"],
        "universal_macro_avg_3utt": md_eval.get("universal_macro_avg_3utt"),
    }


def _prepare_fixed_window_sample(feat: np.ndarray, seq_len: int) -> np.ndarray:
    """Centre-crop or tile a 2-D feature matrix `(T, F)` to `(seq_len, F)`."""
    arr = np.asarray(feat, dtype=np.float32)
    if arr.shape[0] >= seq_len:
        offset = (arr.shape[0] - seq_len) // 2
        return arr[offset:offset + seq_len]
    reps = (seq_len // max(1, arr.shape[0])) + 1
    return np.tile(arr, (reps, 1))[:seq_len]


def assemble_benchmark_results(
    data_root: str = DEFAULT_DATA_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Assemble `pretrained_models/benchmark_results.json` from all 44 model directories."""
    splits_summary_path = os.path.join(data_root, "splits", "dataset_summary.json")
    with open(splits_summary_path, "r", encoding="utf-8") as f:
        ds_summary = json.load(f)

    if isinstance(ds_summary.get("voxceleb1"), dict):
        vx_entry = ds_summary["voxceleb1"]
        vx_entry.pop("num_official_veri_test2_trials", None)
        vx_entry.pop("official_trials_file", None)
        if "vox1_o_10spk_subset_trials_file" in vx_entry and "vox1_o_10spk_subset_note" not in vx_entry:
            vx_entry["vox1_o_10spk_subset_note"] = (
                "VoxCeleb1-O 10-speaker subset: official veri_test2.txt filtered to trials "
                "whose enrollment AND test speakers are both among the 10 held-out eval "
                "speakers, capped at 2,500 trials (~72% target). NOT comparable to published "
                "VoxCeleb1-O numbers; 30/40 official test speakers are used for training by "
                "vox_* and univ_* models."
            )

    results: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hf_collection_url": hf_compat.HF_COLLECTION_URL,
        "datasets": ds_summary,
        "experiments": {},
        "tflite_exports": {},
    }

    for exp_name, *_ in EXPERIMENTS:
        eval_path = os.path.join(output_root, exp_name, "eval_results.json")
        if not os.path.exists(eval_path):
            print(f"WARNING: Missing {eval_path}")
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        results["experiments"][exp_name] = rec
        if isinstance(rec.get("tflite_exports"), dict):
            results["tflite_exports"].update(rec["tflite_exports"])

    out_path = os.path.join(output_root, "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(
        f"Assembled {len(results['experiments'])}/{len(EXPERIMENTS)} models "
        f"and {len(results['tflite_exports'])} TFLite exports -> {out_path}"
    )
    return results


def run_all_benchmarks(
    data_root: str = DEFAULT_DATA_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    configs_dir: str = DEFAULT_CONFIGS_DIR,
    num_steps_override: Optional[int] = None,
    start_idx: int = 0,
    end_idx: int = 100,
    only_models: Optional[set[str]] = None,
    force_retrain: bool = False,
) -> dict[str, Any]:
    resources.configure_polite_resources(
        max_threads=int(os.environ.get("OMP_NUM_THREADS", "8")), nice_level=10
    )
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
        if "multi_trials_file" in info and os.path.exists(
            dataset.resolve_data_path(info["multi_trials_file"])
        ):
            multi_trials_sets[dname] = load_multi_enroll_trials(info["multi_trials_file"])

        c_dir = os.path.join(data_root, "feature_cache", dname)
        cache = feature_extraction.FeatureCache(frontend, cache_dir=c_dir)
        cache.precompute_dataset(spk_eval_sets[dname], num_workers=8)
        caches[dname] = cache

    # Universal multi-domain dataset & unified in-memory cache
    univ_info = ds_summary["universal"]
    spk_train_sets["universal"] = dataset.get_csv_spk_to_utts(univ_info["train_csv"])
    spk_eval_sets["universal"] = dataset.get_csv_spk_to_utts(univ_info["eval_csv"])
    trials_sets["universal"] = dataset.load_trials_file(univ_info["trials_file"])
    if "multi_trials_file" in univ_info and os.path.exists(
        dataset.resolve_data_path(univ_info["multi_trials_file"])
    ):
        multi_trials_sets["universal"] = load_multi_enroll_trials(
            univ_info["multi_trials_file"]
        )
    univ_cache = feature_extraction.FeatureCache.merged(
        frontend,
        [caches[d] for d in ("librispeech", "cnceleb2", "voxceleb1")],
        cache_dir=os.path.join(data_root, "feature_cache", "universal"),
    )
    caches["universal"] = univ_cache
    preloaded_train_domains: set[str] = set()

    vox_official_trials = dataset.load_trials_file(
        ds_summary["voxceleb1"].get(
            "vox1_o_10spk_subset_trials_file",
            ds_summary["voxceleb1"].get("official_trials_file", ""),
        )
    )

    results: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hf_collection_url": hf_compat.HF_COLLECTION_URL,
        "datasets": ds_summary,
        "experiments": {},
        "tflite_exports": {},
    }
    if only_models and len(only_models) == 1:
        single_name = next(iter(only_models))
        report_json_path = os.path.join(
            output_root, f"benchmark_results_{single_name}.json"
        )
    else:
        report_json_path = os.path.join(
            output_root, f"benchmark_results_{start_idx}_{end_idx}.json"
        )

    for idx, (exp_name, dname, backbone, size_var, loss_type, scoring_type, pool_type) in enumerate(EXPERIMENTS):
        if only_models is not None:
            if exp_name not in only_models:
                continue
        elif idx < start_idx or idx >= end_idx:
            continue

        sched = get_model_schedule(exp_name, dname, backbone, size_var, loss_type)
        target_steps = (
            int(num_steps_override)
            if num_steps_override is not None and num_steps_override > 0
            else sched["num_steps"]
        )
        knee_step = min(sched["knee_step"], target_steps)
        safety_margin_steps = sched["safety_margin_steps"]
        warmup_steps = (
            max(10, target_steps // 10)
            if num_steps_override is not None and num_steps_override > 0
            else sched["warmup_steps"]
        )
        margin_warmup_steps = sched["margin_warmup_steps"]

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
            num_steps=target_steps,
            warmup_steps=warmup_steps,
            margin_warmup_steps=margin_warmup_steps,
        )
        yaml_path = os.path.join(configs_dir, f"{exp_name}.yml")
        exp_cfg.to_yaml(yaml_path)

        # Skip only if already trained under v0.5.0 at target_steps AND convergence_checkpoints are recorded
        if not force_retrain and os.path.exists(existing_eval) and os.path.exists(st_path):
            with open(existing_eval, "r", encoding="utf-8") as f:
                cached_rec = json.load(f)
            if (
                cached_rec.get("flaxspeaker_version") == "0.5.0"
                and int(cached_rec.get("num_steps", 0)) == target_steps
                and "multi_domain_eval" in cached_rec
                and "domains_1utt" in cached_rec["multi_domain_eval"]
                and "convergence_checkpoints" in cached_rec
            ):
                print(
                    f"\n[{idx + 1}/{len(EXPERIMENTS)}] Skipping {exp_name} "
                    f"(already trained in v0.5.0 to {target_steps} steps)"
                )
                results["experiments"][exp_name] = cached_rec
                if isinstance(cached_rec.get("tflite_exports"), dict):
                    results["tflite_exports"].update(cached_rec["tflite_exports"])
                continue

        train_utts = DOMAIN_UTT_COUNTS.get(dname, 9447)
        total_epochs = round((target_steps * exp_cfg.train.batch_size) / max(train_utts, 1), 2)
        print(
            f"\n[{idx + 1}/{len(EXPERIMENTS)}] Training & evaluating {exp_name} "
            f"(dataset={dname}, {backbone}-{size_var}, loss={loss_type}, "
            f"scoring={scoring_type}, pooling={pool_type}, "
            f"steps={target_steps} [X={knee_step} + {safety_margin_steps}], "
            f"epochs={total_epochs}, warmup={warmup_steps}, margin_warmup={margin_warmup_steps})...",
            flush=True,
        )

        # Train
        t_start = time.time()
        is_cls = loss_type in (
            "arcface",
            "subcenter_arcface",
            "cosface",
            "sphereface",
            "softmax",
        )
        spk_train = spk_train_sets[dname]
        resolved_shard_dir = (
            dataset.resolve_data_path(exp_cfg.data.sharded_cache_dir)
            if exp_cfg.data.sharded_cache_dir
            else ""
        )
        use_shards = bool(
            resolved_shard_dir
            and os.path.isfile(os.path.join(resolved_shard_dir, "metadata.json"))
        )
        shard_store: Optional[shard_cache.ShardedFeatureStore] = None
        if use_shards:
            shard_store = shard_cache.ShardedFeatureStore(
                resolved_shard_dir, base_speakers_only=not is_cls
            )
            num_classes = shard_store.num_speakers if is_cls else 0
            speaker_to_id = {}
        else:
            speaker_to_id = {s: i for i, s in enumerate(sorted(spk_train.keys()))}
            num_classes = len(speaker_to_id) if is_cls else 0

        _, state = neural_net.get_speaker_encoder(exp_cfg, num_classes=num_classes)
        step_fn = neural_net.make_modern_train_step(exp_cfg)
        rng = np.random.default_rng(exp_cfg.train.seed)
        train_cache = caches[dname]
        if dname not in preloaded_train_domains:
            if use_shards:
                # Only preload the <=5 clean cohort utterances per speaker for AS-Norm
                for sub_d in (
                    ("librispeech", "cnceleb2", "voxceleb1")
                    if dname == "universal"
                    else (dname,)
                ):
                    sub_cohort = {
                        s: sorted(u_list)[:5]
                        for s, u_list in spk_train_sets[sub_d].items()
                    }
                    caches[sub_d].precompute_dataset(sub_cohort, num_workers=4)
                if dname == "universal":
                    train_cache = feature_extraction.FeatureCache.merged(
                        frontend,
                        [caches[d] for d in ("librispeech", "cnceleb2", "voxceleb1")],
                        cache_dir=os.path.join(data_root, "feature_cache", "universal"),
                    )
                    caches["universal"] = train_cache
            elif dname == "universal":
                for sub_d in ("librispeech", "cnceleb2", "voxceleb1"):
                    caches[sub_d].precompute_dataset(spk_train_sets[sub_d], num_workers=8)
                train_cache = feature_extraction.FeatureCache.merged(
                    frontend,
                    [caches[d] for d in ("librispeech", "cnceleb2", "voxceleb1")],
                    cache_dir=os.path.join(data_root, "feature_cache", "universal"),
                )
                caches["universal"] = train_cache
            else:
                train_cache.precompute_dataset(spk_train, num_workers=8)
            preloaded_train_domains.add(dname)

        loss_curve: list[float] = []
        loss_curve_100: list[dict[str, Any]] = []
        convergence_checkpoints: dict[str, Any] = {}
        horizon_snapshots: dict[str, Any] = {}

        var_len = bool(getattr(exp_cfg.model, "variable_length_training", True))
        within_prob = float(getattr(exp_cfg.data, "within_domain_prob", 0.75))
        temp_alpha = float(getattr(exp_cfg.data, "domain_temp_alpha", 0.35))
        triplet_rng = random.Random(int(exp_cfg.train.seed))

        # Checkpoints to evaluate during the training trajectory
        intermediate_eval_steps = {400, knee_step}
        if exp_name == "univ_conformer_small_ge2e":
            intermediate_eval_steps.update({400, 800, 1200, 2400, 4000})

        def _sample_one_batch():
            if shard_store is not None:
                if loss_type == "triplet":
                    return shard_store.sample_triplet_batch(
                        batch_size=exp_cfg.train.batch_size,
                        max_seq_len=exp_cfg.model.seq_len,
                        min_seq_len=exp_cfg.model.min_seq_len,
                        variable_length=var_len,
                        specaug_config=exp_cfg.train.specaug,
                        rng=rng,
                    )
                if is_cls:
                    return shard_store.sample_classification_batch(
                        batch_size=exp_cfg.train.batch_size,
                        max_seq_len=exp_cfg.model.seq_len,
                        min_seq_len=exp_cfg.model.min_seq_len,
                        variable_length=var_len,
                        specaug_config=exp_cfg.train.specaug,
                        rng=rng,
                        domain_temp_alpha=temp_alpha,
                    )
                return shard_store.sample_nm_batch(
                    num_speakers=exp_cfg.train.loss.num_speakers_per_batch,
                    num_utts_per_speaker=exp_cfg.train.loss.num_utts_per_speaker,
                    max_seq_len=exp_cfg.model.seq_len,
                    min_seq_len=exp_cfg.model.min_seq_len,
                    variable_length=var_len,
                    specaug_config=exp_cfg.train.specaug,
                    rng=rng,
                    within_domain_prob=within_prob,
                    domain_temp_alpha=temp_alpha,
                )
            if loss_type == "triplet":
                batch_list = []
                mask_list = []
                for _ in range(exp_cfg.train.batch_size):
                    a_u, p_u, n_u = dataset.get_triplet(spk_train, rng=triplet_rng)
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
                b_in = jnp.asarray(np.stack(batch_list, axis=0), dtype=jnp.float32)
                b_msk = jnp.asarray(np.stack(mask_list, axis=0), dtype=jnp.float32)
                return b_in, None, b_msk
            if is_cls:
                return feature_extraction.get_batched_classification_input(
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
            return feature_extraction.get_batched_nm_input(
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

        cur_batch = _sample_one_batch()
        for step in range(exp_cfg.train.num_steps):
            batch_input, batch_labels, batch_mask = cur_batch
            state, loss_val = step_fn(state, batch_input, batch_labels, batch_mask)
            if step + 1 < exp_cfg.train.num_steps:
                cur_batch = _sample_one_batch()
            loss_scalar = float(loss_val.item())
            loss_curve.append(loss_scalar)
            step_num = step + 1

            if step_num % 100 == 0 or step_num == exp_cfg.train.num_steps:
                mean_100 = round(
                    float(np.mean(loss_curve[-min(len(loss_curve), 100):])), 4
                )
                loss_curve_100.append(
                    {
                        "step": step_num,
                        "epochs": round(
                            (step_num * exp_cfg.train.batch_size) / max(train_utts, 1), 2
                        ),
                        "mean_loss_last_100": mean_100,
                        "step_loss": round(loss_scalar, 4),
                    }
                )

            if step_num % 200 == 0 or step == 0 or step_num == exp_cfg.train.num_steps:
                elapsed = time.time() - t_start
                print(
                    f"  [{exp_name}] step {step_num}/{exp_cfg.train.num_steps} "
                    f"loss={loss_scalar:.4f} (mean100={np.mean(loss_curve[-min(len(loss_curve), 100):]):.4f}, "
                    f"elapsed={elapsed:.1f}s)",
                    flush=True,
                )

            if step_num in intermediate_eval_steps and step_num < exp_cfg.train.num_steps:
                t_ev0 = time.time()
                inter_embs: dict[str, np.ndarray] = {}
                inter_md = evaluate_all_domains(
                    state,
                    exp_cfg,
                    caches,
                    trials_sets,
                    multi_trials_sets,
                    shared_embs=inter_embs,
                )
                snap = summarize_checkpoint_eval(
                    step_num,
                    dname,
                    loss_curve,
                    inter_md,
                    batch_size=exp_cfg.train.batch_size,
                )
                if step_num == 400:
                    convergence_checkpoints["step_400"] = snap
                if step_num == knee_step:
                    convergence_checkpoints["step_X_knee"] = snap
                if exp_name == "univ_conformer_small_ge2e":
                    horizon_snapshots[f"steps_{step_num}"] = snap
                t_ev_sec = time.time() - t_ev0
                # Exclude intermediate evaluation wall-clock from pure training time
                t_start += t_ev_sec
                p3_val = snap.get("pooled_3utt_eer_percent")
                p3_str = f"{p3_val:.2f}%" if p3_val is not None else "n/a"
                print(
                    f"  [{exp_name} @ step {step_num}] Pooled 1-Utt EER={snap['pooled_1utt_eer_percent']:.2f}%, "
                    f"3-Utt EER={p3_str}, "
                    f"In-Dom 1-Utt EER={snap['in_domain_1utt_eer_percent']:.2f}% (eval {t_ev_sec:.1f}s)",
                    flush=True,
                )

        train_time_sec = round(time.time() - t_start, 2)

        # Count encoder parameters (excluding training-only class prototype matrix)
        enc_only_params = {
            k: v for k, v in state.params.items() if k != "_aux_class_weights"
        }
        num_params = neural_net.count_parameters(enc_only_params)

        # Evaluate final S_safe model across all 3 domains + universal pooled & macro-avg
        cohort_embs = compute_training_cohort_embeddings(
            state, exp_cfg, dname, spk_train, train_cache
        )
        shared_embs: dict[str, np.ndarray] = {}
        md_eval = evaluate_all_domains(
            state,
            exp_cfg,
            caches,
            trials_sets,
            multi_trials_sets,
            shared_embs=shared_embs,
            cohort_embeddings=cohort_embs,
        )
        primary_dom = "universal_pooled" if dname == "universal" else dname
        in_domain_eval = md_eval["domains_1utt"][primary_dom]
        multi_enroll_eval = md_eval["domains_3utt"].get(primary_dom)

        final_snap = summarize_checkpoint_eval(
            exp_cfg.train.num_steps,
            dname,
            loss_curve,
            md_eval,
            batch_size=exp_cfg.train.batch_size,
        )
        if exp_cfg.train.num_steps == 400 and "step_400" not in convergence_checkpoints:
            convergence_checkpoints["step_400"] = final_snap
        if exp_cfg.train.num_steps == knee_step and "step_X_knee" not in convergence_checkpoints:
            convergence_checkpoints["step_X_knee"] = final_snap
        convergence_checkpoints["step_S_safe_final"] = final_snap
        if exp_name == "univ_conformer_small_ge2e":
            horizon_snapshots[f"steps_{exp_cfg.train.num_steps}"] = final_snap
            convergence_checkpoints["horizon_snapshots"] = horizon_snapshots

        asn_tag = ""
        if isinstance(md_eval["domains_1utt"]["universal_pooled"].get("as_norm"), dict):
            asn_tag = (
                f" | AS-Norm Pooled={md_eval['domains_1utt']['universal_pooled']['as_norm']['eer_percent']:.2f}%"
            )
        print(
            f"  [{exp_name}] Final S_safe ({exp_cfg.train.num_steps} steps) Multi-Domain 1-Utt EER: "
            f"Macro-Avg={md_eval['universal_macro_avg_1utt']['eer_percent']:.2f}%, "
            f"Pooled={md_eval['domains_1utt']['universal_pooled']['eer_percent']:.2f}% "
            f"(LS={md_eval['domains_1utt']['librispeech']['eer_percent']:.2f}%, "
            f"CN={md_eval['domains_1utt']['cnceleb2']['eer_percent']:.2f}%, "
            f"VOX={md_eval['domains_1utt']['voxceleb1']['eer_percent']:.2f}%){asn_tag}",
            flush=True,
        )
        if md_eval.get("universal_macro_avg_3utt"):
            print(
                f"  [{exp_name}] Final S_safe Multi-Domain 3-Utt EER: "
                f"Macro-Avg={md_eval['universal_macro_avg_3utt']['eer_percent']:.2f}%, "
                f"Pooled={md_eval['domains_3utt']['universal_pooled']['eer_percent']:.2f}%",
                flush=True,
            )

        # Always evaluate VoxCeleb1-O 10-speaker subset using already-computed embeddings
        vox_o_metrics = evaluation.evaluate_verification_trials(
            state,
            vox_official_trials,
            exp_cfg,
            feature_cache=caches["voxceleb1"],
            precomputed_embeddings=shared_embs,
            cohort_embeddings=cohort_embs,
            include_as_norm_breakdown=(cohort_embs is not None),
        )
        md_eval["domains_1utt"]["vox1_o_10spk_subset"] = vox_o_metrics
        cross_eval: dict[str, Any] = {"vox1_o_10spk_subset": vox_o_metrics}
        if dname == "librispeech":
            cross_eval["cnceleb2_zero_shot"] = md_eval["domains_1utt"]["cnceleb2"]

        exp_record: dict[str, Any] = {
            "experiment_name": exp_name,
            "flaxspeaker_version": "0.5.0",
            "train_dataset": dname,
            "backbone": backbone,
            "size_variant": size_var,
            "pooling_type": exp_cfg.model.pooling.pooling_type,
            "loss_type": loss_type,
            "scoring_type": scoring_type,
            "embedding_dim": exp_cfg.model.output_embedding_dim,
            "num_parameters": num_params,
            "num_steps": exp_cfg.train.num_steps,
            "batch_size": exp_cfg.train.batch_size,
            "knee_step": knee_step,
            "safety_margin_steps": safety_margin_steps,
            "warmup_steps": exp_cfg.train.warmup_steps,
            "margin_warmup_steps": exp_cfg.train.loss.margin_warmup_steps,
            "train_epochs": total_epochs,
            "train_time_sec": train_time_sec,
            "initial_loss": round(loss_curve[0], 4),
            "final_loss": round(float(np.mean(loss_curve[-min(len(loss_curve), 100):])), 4),
            "final_loss_last_20": round(
                float(np.mean(loss_curve[-min(len(loss_curve), 20):])), 4
            ),
            "loss_curve_100": loss_curve_100,
            "convergence_checkpoints": convergence_checkpoints,
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

        # Export TFLite for the 6 representative models before saving model card
        if exp_name in TFLITE_TARGETS:
            model_tflite_exports: dict[str, Any] = {}
            enc_only_for_apply = {
                k: v for k, v in state.params.items() if not k.startswith("_aux_")
            }
            apply_vars: dict[str, Any] = {"params": enc_only_for_apply}
            if getattr(state, "batch_stats", None):
                apply_vars["batch_stats"] = state.batch_stats
            sample_utt = trials_sets["librispeech"][0][1]
            sample_feat = _prepare_fixed_window_sample(
                caches["librispeech"].get(sample_utt), exp_cfg.model.seq_len
            )
            flax_emb = np.asarray(
                state.apply_fn(
                    apply_vars, sample_feat[None, :, :], deterministic=True
                )[0]
            )
            for quant in (False, True):
                suffix = "int8" if quant else "fp32"
                tflite_path = os.path.join(hf_dir, f"model_{suffix}.tflite")
                tf_info = export.export_to_tflite(
                    state, exp_cfg, tflite_path, quantize_int8=quant
                )
                runner = export.TFLiteSpeakerRunner(tflite_path)
                t_tfl0 = time.time()
                tfl_emb = runner(sample_feat)
                tfl_latency_ms = (time.time() - t_tfl0) * 1000.0
                cos_fidelity = float(
                    np.dot(tfl_emb, flax_emb)
                    / (np.linalg.norm(tfl_emb) * np.linalg.norm(flax_emb) + 1e-6)
                )
                tf_info["tflite_path"] = f"pretrained_models/{exp_name}/model_{suffix}.tflite"
                tf_info["latency_ms"] = round(tfl_latency_ms, 3)
                tf_info["cosine_fidelity_vs_flax"] = round(cos_fidelity, 6)
                model_tflite_exports[f"{exp_name}_{suffix}"] = tf_info
                results["tflite_exports"][f"{exp_name}_{suffix}"] = tf_info
            exp_record["tflite_exports"] = model_tflite_exports

        # Save HuggingFace-compatible pretrained directory
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(exp_cfg)
        hf_model = hf_compat.FlaxSpeakerModel(
            config=hf_cfg,
            params=state.params,
            batch_stats=getattr(state, "batch_stats", None),
        )
        hf_model.save_pretrained(hf_dir, metrics=exp_record)

        results["experiments"][exp_name] = exp_record
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== COMPLETED ({len(results['experiments'])} models) ===", flush=True)
    print(f"Results saved to: {report_json_path}", flush=True)
    return results


def estimate_model_cost(exp_tuple: tuple[str, ...]) -> float:
    exp_name, dname, backbone, size_var, loss_type, scoring_type, _ = exp_tuple
    sched = get_model_schedule(exp_name, dname, backbone, size_var, loss_type)
    steps = sched["num_steps"]
    if backbone == "resnet":
        sec_per_step = 4.92
    elif backbone == "mamba":
        sec_per_step = 1.00
    elif loss_type == "triplet":
        sec_per_step = 0.85
    elif backbone == "ecapa_tdnn" and size_var == "base":
        sec_per_step = 0.62
    elif backbone == "conformer" and size_var == "base":
        sec_per_step = 0.50
    elif scoring_type == "pfas":
        sec_per_step = 0.40
    elif scoring_type == "dr_vector":
        sec_per_step = 0.37
    elif backbone == "ecapa_tdnn" and size_var == "small":
        sec_per_step = 0.36
    elif backbone == "conformer" and size_var == "small":
        sec_per_step = 0.365 if exp_name == "univ_conformer_small_ge2e" else 0.35
    elif backbone == "transformer":
        sec_per_step = 0.26
    elif backbone == "ecapa_tdnn" and size_var == "tiny":
        sec_per_step = 0.22
    elif backbone == "conformer" and size_var == "tiny":
        sec_per_step = 0.21
    elif backbone == "lstm":
        sec_per_step = 0.15
    else:
        sec_per_step = 0.35
    return round(steps * sec_per_step, 1)


def run_parallel_4ccd(
    data_root: str = DEFAULT_DATA_ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    include_ablations: bool = True,
    force_retrain: bool = False,
) -> None:
    """Execute all 44 models (+ R1/R2 ablations) across 4 CCD-pinned workers in LPT order."""
    import collections
    import subprocess
    import threading

    # First ensure all 44 configs/<model_id>.yml are written with S_safe budgets
    os.makedirs(DEFAULT_CONFIGS_DIR, exist_ok=True)
    splits_summary_path = os.path.join(data_root, "splits", "dataset_summary.json")
    with open(splits_summary_path, "r", encoding="utf-8") as f:
        ds_summary = json.load(f)
    for exp_name, dname, backbone, size_var, loss_type, scoring_type, pool_type in EXPERIMENTS:
        sched = get_model_schedule(exp_name, dname, backbone, size_var, loss_type)
        d_info = ds_summary[dname]
        cfg = make_exp_config(
            dataset_name=dname,
            train_csv=d_info["train_csv"],
            eval_csv=d_info["eval_csv"],
            trials_file=d_info["trials_file"],
            backbone=backbone,
            size_variant=size_var,
            loss_type=loss_type,
            scoring_type=scoring_type,
            pooling_type=pool_type,
            num_steps=sched["num_steps"],
            warmup_steps=sched["warmup_steps"],
            margin_warmup_steps=sched["margin_warmup_steps"],
        )
        cfg.to_yaml(os.path.join(DEFAULT_CONFIGS_DIR, f"{exp_name}.yml"))

    # Separate the two heavy 2D ResNet models (which benefit 2x from 2-CCD 16-core
    # execution on 0-15,32-47, matching Phase 1) from the 1D models + ablations.
    resnet_names = {"univ_resnet_small_ge2e", "ls_resnet_small_ge2e"}
    heavy_resnet_tasks = sorted(
        [
            (0, "main", exp[0], estimate_model_cost(exp))
            for exp in EXPERIMENTS
            if exp[0] in resnet_names
        ],
        key=lambda x: -x[3],
    )
    fast_main_tasks = sorted(
        [
            (0, "main", exp[0], estimate_model_cost(exp))
            for exp in EXPERIMENTS
            if exp[0] not in resnet_names
        ],
        key=lambda x: -x[3],
    )

    ablation_tasks: list[tuple[int, str, str, float]] = []
    if include_ablations:
        if force_retrain:
            ab_cache_dir = os.path.join(output_root, "_ablation_cache")
            if os.path.isdir(ab_cache_dir):
                import shutil

                shutil.rmtree(ab_cache_dir)
                os.makedirs(ab_cache_dir, exist_ok=True)
        raw_ablation_tasks = [
            (1, "r1", "mamba_bidirectional", 8510.0),
            (1, "r1", "ls_mamba_bidirectional", 4500.0),
            (1, "r1", "odyssey2022_pfas_pure_paper", 2070.0),
            (1, "r1", "pfas_sn_mlp_no_anchor", 1840.0),
            (1, "r1", "pfas_sn_dot_with_anchor", 1840.0),
            (1, "r1", "pfas_sn_maxpool_with_anchor", 1840.0),
            (1, "r1", "interspeech2021_dr_vectors_pure_paper", 1748.0),
            (1, "r1", "dr_vector_3term_sym_skip", 1748.0),
            (1, "r1", "dr_vectors_3term_pure_eq7", 1748.0),
            (1, "r1", "ecapa_ln_before_act", 1656.0),
            (1, "r2", "seed_univ_ecapa_tdnn_small_ge2e_s0", 1656.0),
            (1, "r2", "seed_univ_ecapa_tdnn_small_ge2e_s1", 1656.0),
            (1, "r2", "seed_univ_ecapa_tdnn_small_ge2e_s2", 1656.0),
            (1, "r1", "vad_20db_no_agc", 1640.0),
            (1, "r1", "vad_26db_no_agc", 1640.0),
            (1, "r1", "variable_length_false", 1630.0),
            (1, "r1", "domain_mixing_p_within_0.0", 1610.0),
            (1, "r1", "domain_mixing_p_within_1.0", 1610.0),
            (1, "r1", "ext_softmax_pure_eq3", 1610.0),
            (1, "r1", "odyssey2022_conformer_atp_pure_paper", 1610.0),
            (1, "r2", "seed_univ_conformer_small_ge2e_s0", 1610.0),
            (1, "r2", "seed_univ_conformer_small_ge2e_s1", 1610.0),
            (1, "r2", "seed_univ_conformer_small_ge2e_s2", 1610.0),
            (1, "r2", "matched_exact5788_4600steps", 1610.0),
            (1, "r1", "icassp2018_ge2e_pure_paper", 1472.0),
            (1, "r1", "horizon_snapshots_full", 900.0),
            (1, "r2", "seed_ls_conformer_small_ge2e_s0", 840.0),
            (1, "r2", "seed_ls_conformer_small_ge2e_s1", 840.0),
            (1, "r2", "seed_ls_conformer_small_ge2e_s2", 840.0),
            (1, "r2", "matched_exact5788_2400steps", 840.0),
            (1, "r2", "own_horizon_s42_2400", 840.0),
            (1, "r2", "own_horizon_s42_1200", 420.0),
            (1, "r2", "own_horizon_s42_800", 280.0),
            (1, "r2", "own_horizon_s42_400", 140.0),
            (1, "r1", "variable_length_true_short_eval", 25.0),
        ]
        ablation_tasks = sorted(raw_ablation_tasks, key=lambda x: -x[3])

    total_main_count = len(heavy_resnet_tasks) + len(fast_main_tasks)
    fast_all_tasks = fast_main_tasks + ablation_tasks
    queue = collections.deque(fast_all_tasks)
    heavy_queue = collections.deque(heavy_resnet_tasks)
    total_tasks = len(heavy_resnet_tasks) + len(fast_all_tasks)
    lock = threading.Lock()
    completed_count = 0
    completed_main = 0
    failures: list[str] = []
    t_global_start = time.time()

    py_exe = sys.executable

    print(
        f"=== Starting 4-CCD LPT Dynamic Queue ({total_main_count} main models + "
        f"{len(ablation_tasks)} ablation tasks = {total_tasks} total) ===",
        flush=True,
    )
    print(
        f"Heavy 2-CCD slot (0-15,32-47): {[f'{t[1]}:{t[2]} (~{t[3]}s)' for t in heavy_resnet_tasks]}",
        flush=True,
    )
    print(
        f"Fast 1-CCD queue top-4: {[f'{t[1]}:{t[2]} (~{t[3]}s)' for t in fast_all_tasks[:4]]}",
        flush=True,
    )

    def _run_one_task(
        w_label: str,
        core_range: str,
        n_threads: int,
        log_path: str,
        task_tuple: tuple[int, str, str, float],
    ) -> None:
        nonlocal completed_count, completed_main
        _, kind, name, est_s = task_tuple
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(n_threads)
        env["MKL_NUM_THREADS"] = str(n_threads)
        env["OPENBLAS_NUM_THREADS"] = str(n_threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
        env["NUMEXPR_NUM_THREADS"] = str(n_threads)
        env["XLA_FLAGS"] = (
            f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={n_threads}"
        )

        t0 = time.time()
        print(
            f"[{w_label} ({core_range})] START {kind}:{name} (est {est_s:.0f}s) "
            f"at +{t0 - t_global_start:.1f}s",
            flush=True,
        )

        if kind == "main":
            cmd = [
                "nice",
                "-n",
                "10",
                "taskset",
                "-c",
                core_range,
                py_exe,
                os.path.join(REPO_ROOT, "scripts", "run_research_benchmarks.py"),
                "--data_root",
                data_root,
                "--output_root",
                output_root,
                "--only",
                name,
            ]
            if force_retrain:
                cmd.append("--force_retrain")
        elif kind == "r1":
            cmd = [
                "nice",
                "-n",
                "10",
                "taskset",
                "-c",
                core_range,
                py_exe,
                os.path.join(REPO_ROOT, "scripts", "run_r1_ablations.py"),
                "--data_root",
                data_root,
                "--task",
                name,
            ]
        elif kind == "r2":
            cmd = [
                "nice",
                "-n",
                "10",
                "taskset",
                "-c",
                core_range,
                py_exe,
                os.path.join(REPO_ROOT, "scripts", "run_r2_controls.py"),
                "--data_root",
                data_root,
                "--task",
                name,
            ]
        else:
            raise ValueError(f"Unknown task kind: {kind}")

        success = False
        for attempt in (1, 2):
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(
                    f"\n--- [{datetime.datetime.now().isoformat()}] {kind}:{name} (attempt {attempt}) ---\n"
                )
                lf.flush()
                proc = subprocess.run(
                    cmd, cwd=REPO_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT
                )
            if proc.returncode == 0:
                success = True
                break
            print(
                f"[{w_label} ({core_range})] WARNING: {kind}:{name} failed (rc={proc.returncode}, attempt={attempt})",
                flush=True,
            )

        dt = time.time() - t0
        should_assemble_main = False
        with lock:
            completed_count += 1
            if kind == "main" and success:
                completed_main += 1
                if completed_main == total_main_count:
                    should_assemble_main = True
            if not success:
                failures.append(f"{kind}:{name}")
            done_now = completed_count
            main_now = completed_main

        status_str = "DONE" if success else "FAILED"
        print(
            f"[{w_label} ({core_range})] {status_str} [{done_now}/{total_tasks}, main {main_now}/{total_main_count}] "
            f"{kind}:{name} in {dt:.1f}s (total elapsed {(time.time() - t_global_start) / 60.0:.1f} min)",
            flush=True,
        )
        if should_assemble_main:
            try:
                assemble_benchmark_results(data_root=data_root, output_root=output_root)
            except Exception as exc:
                print(f"WARNING: assemble_benchmark_results error: {exc}", flush=True)

    def _fast_worker_loop(w_idx: int, core_range: str) -> None:
        log_path = f"/tmp/flaxspeaker_v050_ccd{w_idx}.log"
        while True:
            with lock:
                if not queue:
                    break
                task_tuple = queue.popleft()
            _run_one_task(f"CCD {w_idx}", core_range, 7, log_path, task_tuple)

    def _heavy_then_split_loop() -> None:
        heavy_log = "/tmp/flaxspeaker_v050_ccd0.log"
        while True:
            with lock:
                if not heavy_queue:
                    break
                h_task = heavy_queue.popleft()
            _run_one_task("CCD 0+1", "0-11", 12, heavy_log, h_task)

        print(
            "[Orchestrator] Both 2D ResNet models completed on CCD 0+1; splitting into CCD 0 (0-6) and CCD 1 (8-14)!",
            flush=True,
        )
        t_ccd0 = threading.Thread(target=_fast_worker_loop, args=(0, "0-6"), daemon=False)
        t_ccd1 = threading.Thread(target=_fast_worker_loop, args=(1, "8-14"), daemon=False)
        t_ccd0.start()
        t_ccd1.start()
        t_ccd0.join()
        t_ccd1.join()

    threads = [
        threading.Thread(target=_heavy_then_split_loop, daemon=False),
        threading.Thread(target=_fast_worker_loop, args=(2, "16-22"), daemon=False),
        threading.Thread(target=_fast_worker_loop, args=(3, "24-30"), daemon=False),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_min = (time.time() - t_global_start) / 60.0
    print(f"\n=== 4-CCD LPT Queue Finished in {total_min:.2f} min (failures={failures}) ===", flush=True)
    if failures:
        raise RuntimeError(f"Tasks failed: {failures}")

    # Final assembly & post-training verification scripts
    assemble_benchmark_results(data_root=data_root, output_root=output_root)
    if include_ablations:
        from scripts import run_r1_ablations
        from scripts import run_r2_controls

        run_r1_ablations.assemble_ablation_results()
        run_r2_controls.assemble_r2_results()

    post_scripts = [
        [py_exe, os.path.join(REPO_ROOT, "scripts", "eval_vox1_o_subset.py")],
        [py_exe, os.path.join(REPO_ROOT, "scripts", "eval_tflite_models.py")],
        [
            py_exe,
            os.path.join(REPO_ROOT, "scripts", "fingerprint_released_checkpoints.py"),
            "--overwrite",
        ],
        [py_exe, os.path.join(REPO_ROOT, "scripts", "write_training_provenance.py")],
        [py_exe, os.path.join(REPO_ROOT, "scripts", "update_hf_model_cards.py")],
    ]
    for sc_cmd in post_scripts:
        print(f"\n[Post-Processing] Running: {' '.join(sc_cmd)}", flush=True)
        subprocess.run(sc_cmd, cwd=REPO_ROOT, check=True)
    assemble_benchmark_results(data_root=data_root, output_root=output_root)
    print("\n=== ALL PHASE 2 TRAINING, ABLATIONS & POST-PROCESSING COMPLETE ===", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--configs_dir", type=str, default=DEFAULT_CONFIGS_DIR)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Optional override of num_steps (default: per-model S_safe converged schedule)",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=100)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of model IDs to run (e.g. univ_resnet_small_ge2e)",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Retrain even if eval_results.json already matches S_safe",
    )
    parser.add_argument(
        "--assemble_only",
        action="store_true",
        help="Assemble pretrained_models/benchmark_results.json from per-model eval_results.json",
    )
    parser.add_argument(
        "--parallel_4ccd",
        action="store_true",
        help="Run all 44 models across 4 CCD-pinned workers in LPT order",
    )
    parser.add_argument(
        "--include_ablations",
        action="store_true",
        help="When used with --parallel_4ccd, also queue R1 and R2 converged ablation tasks",
    )
    args = parser.parse_args()
    if args.assemble_only:
        assemble_benchmark_results(data_root=args.data_root, output_root=args.output_root)
    elif args.parallel_4ccd:
        run_parallel_4ccd(
            data_root=args.data_root,
            output_root=args.output_root,
            include_ablations=args.include_ablations,
            force_retrain=args.force_retrain,
        )
    else:
        only_set = (
            {m.strip() for m in args.only.split(",") if m.strip()}
            if args.only
            else None
        )
        run_all_benchmarks(
            data_root=args.data_root,
            output_root=args.output_root,
            configs_dir=args.configs_dir,
            num_steps_override=args.num_steps,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            only_models=only_set,
            force_retrain=args.force_retrain,
        )
