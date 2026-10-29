#!/usr/bin/env python3
"""Step-0 loss fingerprint of every released checkpoint (CPU, ~15-25 s per probe).

For each ``configs/<model_id>.yml`` this script loads the *released* YAML
(``ExperimentConfig.from_yaml``, so loss-specific fields such as ``ge2e_init_b`` and
``ge2e_contrast_surrogate`` match the released recipe), builds the first training
batch at the released seed exactly as ``scripts/run_research_benchmarks.py`` does,
runs one optimizer step, and compares the resulting loss with the archived
``eval_results.json -> initial_loss`` under two crop regimes:

* ``head_fixed``: ``variable_length_training=False`` (fixed ``seq_len`` = 180-frame
  crops, all-ones validity masks) -- the regime that the pre-``9accf95`` batcher
  defaults silently produced;
* ``head_variable``: ``variable_length_training=True`` (``T ~ U(min_seq_len, seq_len)``
  crops with an exact validity mask) -- the pipeline default since ``9accf95``.

The first batch (speaker/utterance draws, crop offsets, augmentation draws) and the
parameter initialisation are both deterministic functions of the seed, so a match of
the step-0 loss to four decimals identifies the crop regime the released weights were
trained with. ``match`` is one of

* ``"fixed"`` / ``"variable"``: exactly one regime reproduces the archived 4-decimal
  value (|diff| <= 5e-5);
* ``"both"``: not discriminable (e.g. GE2E-contrast, whose loss is ~1.0 at init);
* ``"fixed_approx"`` / ``"variable_approx"``: neither regime matches exactly, but one
  is within 1e-3 while the other is at least 10x farther away. This happens for the
  three ``*_pfas*`` checkpoints, whose PFAS scorer gained numerical changes after
  release (``9accf95`` / ``1797566``: global L2 norm, +epsilon in the attention
  softmax), so a sub-1e-3 drift of the step-0 loss is expected even at the right
  crop regime;
* ``"neither"``: expected for the two ``*_triplet`` checkpoints, whose triplets were
  drawn from the *unseeded* global ``random`` module at ``1cb90c7``.

Loss-schedule caveat: the two ``*_sphereface`` checkpoints were trained when
``sphereface_loss`` used a *constant* annealing factor ``lambda = 5`` (``1cb90c7``);
``9accf95`` introduced the schedule ``lambda = max(lambda_min, lambda_base *
(1 + gamma * step)^-power)`` (= 1000 at step 0 with the released YAML values). For
these two checkpoints the probe therefore also runs with ``sphereface_lambda_base``
set to ``lambda_min`` (constant lambda = 5, the pre-``9accf95`` behaviour) and records
the result under ``head_fixed_legacy_lambda`` / ``head_variable_legacy_lambda``; the
``match`` field uses the legacy-lambda arms for them (``legacy_lambda_used: true``).

Outputs ``pretrained_models/step0_fingerprints.json``. Nothing is estimated.
"""

import argparse
import datetime
import glob
import json
import os
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import configs  # noqa: E402
from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import neural_net  # noqa: E402
from flaxspeaker import resources  # noqa: E402
from flaxspeaker import shard_cache  # noqa: E402

DEFAULT_DATA_ROOT = os.environ.get(
    "FLAXSPEAKER_DATA_ROOT", os.path.expanduser("~/Data/speaker_datasets")
)
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "pretrained_models")
CLASSIFICATION_LOSSES = (
    "arcface",
    "subcenter_arcface",
    "cosface",
    "sphereface",
    "softmax",
)
TOLERANCE = 5e-5  # released initial_loss is stored with 4 decimals


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:  # pragma: no cover
        return "unknown"


def first_step_loss(
    cfg: Any,
    spk_train: dataset.SpkToUtts,
    frontend: feature_extraction.AudioFrontend,
    cache: feature_extraction.FeatureCache,
    variable_length: bool,
) -> tuple[float, list[int]]:
    """One training step at ``cfg.train.seed``; returns (loss, valid frames of first utts)."""
    loss_type = cfg.train.loss.loss_type
    is_cls = loss_type in CLASSIFICATION_LOSSES
    resolved_shard = (
        dataset.resolve_data_path(cfg.data.sharded_cache_dir)
        if getattr(cfg.data, "sharded_cache_dir", "")
        else ""
    )
    use_base_only = (not is_cls) and (cfg.data.dataset_name != "universal_full")
    shard_store = (
        shard_cache.ShardedFeatureStore(resolved_shard, base_speakers_only=use_base_only)
        if (
            resolved_shard
            and os.path.isfile(os.path.join(resolved_shard, "metadata.json"))
        )
        else None
    )
    if shard_store is not None:
        num_cls = shard_store.num_speakers if is_cls else 0
        speaker_to_id: dict[str, int] = {}
    else:
        speaker_to_id = {s: i for i, s in enumerate(sorted(spk_train.keys()))}
        num_cls = len(speaker_to_id) if is_cls else 0
    _, state = neural_net.get_speaker_encoder(cfg, num_classes=num_cls)
    step_fn = neural_net.make_modern_train_step(cfg)
    rng = np.random.default_rng(cfg.train.seed)
    if shard_store is not None:
        if loss_type == "triplet":
            bi, bl, bm = shard_store.sample_triplet_batch(
                batch_size=cfg.train.batch_size,
                max_seq_len=cfg.model.seq_len,
                min_seq_len=cfg.model.min_seq_len,
                variable_length=variable_length,
                specaug_config=cfg.train.specaug,
                rng=rng,
            )
        elif is_cls:
            bi, bl, bm = shard_store.sample_classification_batch(
                batch_size=cfg.train.batch_size,
                max_seq_len=cfg.model.seq_len,
                min_seq_len=cfg.model.min_seq_len,
                variable_length=variable_length,
                specaug_config=cfg.train.specaug,
                rng=rng,
                domain_temp_alpha=cfg.data.domain_temp_alpha,
            )
        else:
            bi, bl, bm = shard_store.sample_nm_batch(
                num_speakers=cfg.train.loss.num_speakers_per_batch,
                num_utts_per_speaker=cfg.train.loss.num_utts_per_speaker,
                max_seq_len=cfg.model.seq_len,
                min_seq_len=cfg.model.min_seq_len,
                variable_length=variable_length,
                specaug_config=cfg.train.specaug,
                rng=rng,
                within_domain_prob=cfg.data.within_domain_prob,
                domain_temp_alpha=cfg.data.domain_temp_alpha,
            )
    elif loss_type == "triplet":
        import jax.numpy as jnp

        triplet_rng = random.Random(int(cfg.train.seed))
        batch_list, mask_list = [], []
        for _ in range(cfg.train.batch_size):
            a_u, p_u, n_u = dataset.get_triplet(spk_train, rng=triplet_rng)
            for u in (a_u, p_u, n_u):
                crop_aug, mask, _ = feature_extraction.sample_variable_length_features(
                    cache.get(u),
                    max_seq_len=cfg.model.seq_len,
                    min_seq_len=cfg.model.min_seq_len,
                    variable_length=variable_length,
                    specaug_config=cfg.train.specaug,
                    waveform_aug_config=cfg.train.waveform_aug,
                    rng=rng,
                )
                batch_list.append(crop_aug)
                mask_list.append(mask)
        bi = jnp.asarray(np.stack(batch_list, axis=0), dtype=jnp.float32)
        bm = jnp.asarray(np.stack(mask_list, axis=0), dtype=jnp.float32)
        bl = None
    elif is_cls:
        bi, bl, bm = feature_extraction.get_batched_classification_input(
            spk_train, speaker_to_id, batch_size=cfg.train.batch_size,
            seq_len=cfg.model.seq_len, frontend=frontend,
            specaug_config=cfg.train.specaug, waveform_aug_config=cfg.train.waveform_aug,
            cache=cache, rng=rng, min_seq_len=cfg.model.min_seq_len,
            variable_length=variable_length, return_mask=True,
            domain_temp_alpha=cfg.data.domain_temp_alpha,
        )
    else:
        bi, bl, bm = feature_extraction.get_batched_nm_input(
            spk_train, num_speakers=cfg.train.loss.num_speakers_per_batch,
            num_utts_per_speaker=cfg.train.loss.num_utts_per_speaker,
            seq_len=cfg.model.seq_len, frontend=frontend,
            specaug_config=cfg.train.specaug, waveform_aug_config=cfg.train.waveform_aug,
            cache=cache, rng=rng, min_seq_len=cfg.model.min_seq_len,
            variable_length=variable_length, return_mask=True,
            within_domain_prob=cfg.data.within_domain_prob,
            domain_temp_alpha=cfg.data.domain_temp_alpha,
        )
    valid = [int(v) for v in np.asarray(bm).sum(axis=1)[:4]]
    _, loss_val = step_fn(state, bi, bl, bm)
    return float(loss_val), valid


APPROX_TOLERANCE = 1e-3
APPROX_RATIO = 10.0


def classify(released: float, fixed: float, variable: float, tol: float = TOLERANCE) -> str:
    d_fixed = abs(round(fixed, 4) - released)
    d_var = abs(round(variable, 4) - released)
    m_fixed = d_fixed <= tol
    m_var = d_var <= tol
    if m_fixed and m_var:
        return "both"
    if m_fixed:
        return "fixed"
    if m_var:
        return "variable"
    if d_fixed <= APPROX_TOLERANCE and d_var >= APPROX_RATIO * max(d_fixed, tol):
        return "fixed_approx"
    if d_var <= APPROX_TOLERANCE and d_fixed >= APPROX_RATIO * max(d_var, tol):
        return "variable_approx"
    return "neither"


def main(
    data_root: str,
    output_root: str,
    output_json: str,
    only: list[str],
    overwrite: bool = False,
) -> None:
    resources.configure_polite_resources(max_threads=8, nice_level=10)
    if not output_json:
        output_json = os.path.join(output_root, "step0_fingerprints.json")
    with open(os.path.join(data_root, "splits", "dataset_summary.json"), "r", encoding="utf-8") as f:
        ds = json.load(f)

    model_ids = sorted(
        os.path.basename(p)[:-4] for p in glob.glob(os.path.join(REPO_ROOT, "configs", "*.yml"))
    )
    if only:
        model_ids = [m for m in model_ids if m in set(only)]

    results: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "tolerance": TOLERANCE,
        "description": (
            "Step-0 training loss at the released seed from the released YAML, under "
            "fixed seq_len crops (head_fixed) and U(min_seq_len, seq_len) crops "
            "(head_variable), compared with eval_results.json initial_loss."
        ),
        "checkpoints": {},
        "histogram": {},
    }
    if os.path.exists(output_json) and not (overwrite and not only):
        with open(output_json, "r", encoding="utf-8") as f:
            prev = json.load(f)
        results["checkpoints"] = prev.get("checkpoints", {})

    frontend: feature_extraction.AudioFrontend | None = None
    caches: dict[str, feature_extraction.FeatureCache] = {}
    spk_sets: dict[str, dataset.SpkToUtts] = {}

    def _get_cache(dname: str, fe: feature_extraction.AudioFrontend):
        if dname in caches:
            return caches[dname], spk_sets[dname]
        if dname == "universal":
            parts = []
            for d in ("librispeech", "cnceleb2", "voxceleb1"):
                c, _ = _get_cache(d, fe)
                parts.append(c)
            cache = feature_extraction.FeatureCache.merged(
                fe, parts, cache_dir=os.path.join(data_root, "feature_cache", "universal")
            )
        else:
            cache = feature_extraction.FeatureCache(
                fe, cache_dir=os.path.join(data_root, "feature_cache", dname)
            )
        spk = dataset.get_csv_spk_to_utts(ds[dname]["train_csv"])
        cache.precompute_dataset(spk, num_workers=8)
        caches[dname] = cache
        spk_sets[dname] = spk
        return cache, spk

    def _save() -> None:
        hist: dict[str, int] = {}
        for rec in results["checkpoints"].values():
            hist[rec["match"]] = hist.get(rec["match"], 0) + 1
        results["histogram"] = dict(sorted(hist.items()))
        results["num_checkpoints"] = len(results["checkpoints"])
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    for i, mid in enumerate(model_ids):
        if not overwrite and mid in results["checkpoints"]:
            continue
        eval_res_path = os.path.join(output_root, mid, "eval_results.json")
        if not os.path.isfile(eval_res_path):
            print(f"\n[{i + 1}/{len(model_ids)}] Skipping {mid} (no eval_results.json yet)")
            continue
        cfg = configs.ExperimentConfig.from_yaml(os.path.join(REPO_ROOT, "configs", f"{mid}.yml"))
        with open(eval_res_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        phase_ov = rec.get("step0_phase_overrides")
        if isinstance(phase_ov, dict):
            for ov_k, ov_v in phase_ov.items():
                if hasattr(cfg.train, ov_k):
                    setattr(cfg.train, ov_k, ov_v)
                elif hasattr(cfg.train.loss, ov_k):
                    setattr(cfg.train.loss, ov_k, ov_v)
                elif hasattr(cfg.model, ov_k):
                    setattr(cfg.model, ov_k, ov_v)
            cfg.loss = cfg.train.loss
        dname = rec["train_dataset"]
        if frontend is None:
            frontend = feature_extraction.AudioFrontend(cfg.frontend, cfg.vad)
        resolved_shard = (
            dataset.resolve_data_path(cfg.data.sharded_cache_dir)
            if getattr(cfg.data, "sharded_cache_dir", "")
            else ""
        )
        has_shard = bool(
            resolved_shard
            and os.path.isfile(os.path.join(resolved_shard, "metadata.json"))
        )
        if has_shard:
            if dname in ds and "train_csv" in ds[dname]:
                cfg.data.train_csv = ds[dname]["train_csv"]
            cache, spk_train = None, {}
        else:
            cfg.data.train_csv = ds[dname]["train_csv"]
            cache, spk_train = _get_cache(dname, frontend)
        released = float(rec["initial_loss"])
        print(f"\n[{i + 1}/{len(model_ids)}] {mid} ({dname}, {cfg.train.loss.loss_type}, "
              f"seed {cfg.train.seed}) released initial_loss={released}")
        out: dict[str, Any] = {
            "train_dataset": dname,
            "loss_type": cfg.train.loss.loss_type,
            "seed": int(cfg.train.seed),
            "seq_len": int(cfg.model.seq_len),
            "min_seq_len": int(cfg.model.min_seq_len),
            "yaml_variable_length_training": bool(cfg.model.variable_length_training),
            "released_initial_loss": released,
        }
        probes = [("head_fixed", False, False), ("head_variable", True, False)]
        is_sphereface = cfg.train.loss.loss_type == "sphereface"
        if is_sphereface:
            probes += [
                ("head_fixed_legacy_lambda", False, True),
                ("head_variable_legacy_lambda", True, True),
            ]
        released_lambda_base = float(cfg.train.loss.sphereface_lambda_base)
        for key, var_len, legacy_lambda in probes:
            cfg.model.variable_length_training = var_len
            # Pre-9accf95 sphereface_loss used a constant anneal_lambda (= lambda_min).
            cfg.train.loss.sphereface_lambda_base = (
                float(cfg.train.loss.sphereface_lambda_min) if legacy_lambda else released_lambda_base
            )
            t0 = time.time()
            loss, valid = first_step_loss(cfg, spk_train, frontend, cache, var_len)
            out[key] = round(loss, 4)
            out[f"{key}_raw"] = round(loss, 6)
            out[f"{key}_valid_frames_first4"] = valid
            out[f"{key}_probe_sec"] = round(time.time() - t0, 1)
            print(f"    {key:28s} loss={loss:.4f}  valid frames {valid}  ({out[f'{key}_probe_sec']} s)")
        cfg.train.loss.sphereface_lambda_base = released_lambda_base
        m_head = classify(released, out["head_fixed"], out["head_variable"])
        use_legacy_lambda = bool(is_sphereface and m_head == "neither")
        out["legacy_lambda_used"] = use_legacy_lambda
        fixed_key = "head_fixed_legacy_lambda" if use_legacy_lambda else "head_fixed"
        var_key = "head_variable_legacy_lambda" if use_legacy_lambda else "head_variable"
        out["abs_diff_fixed"] = round(abs(out[fixed_key] - released), 4)
        out["abs_diff_variable"] = round(abs(out[var_key] - released), 4)
        out["match"] = classify(released, out[fixed_key], out[var_key])
        print(f"    -> match: {out['match']}")
        results["checkpoints"][mid] = out
        _save()

    _save()
    print(f"\nSaved {output_json}; histogram = {results['histogram']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-0 loss fingerprint of the released checkpoints.")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--only", type=str, nargs="*", default=[], help="Subset of model ids.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing fingerprints.")
    args = parser.parse_args()
    main(args.data_root, args.output_root, args.output_json, args.only, overwrite=args.overwrite)
