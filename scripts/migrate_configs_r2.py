#!/usr/bin/env python3
"""Round-2 config migration: make every YAML / `config.json` truthful about what was run.

- Drops config keys that were never consumed by any module (see
  `configs._DEPRECATED_CONFIG_KEYS`); loading them now emits a DeprecationWarning.
- Makes the previously *hidden* `ge2e_contrast` overrides explicit:
  `ge2e_init_b = -1.0`, `ge2e_contrast_surrogate = true`, `ge2e_contrast_b_grad_scale = 0.5`.
- `snr_min_db: 8.0` -> `12.0` (the code used to clamp 8 -> 12 silently; the clamp is gone).
- `min_learning_rate` -> the value actually used (`max(1e-5, 0.01 * lr)`), since
  `build_optimizer` now honours the field exactly.
- `export.sequence_length: 100` (ignored by a magic sentinel) -> `180`, the window the
  released `.tflite` files were actually traced with.

Run from the repository root:  python scripts/migrate_configs_r2.py
"""

import glob
import json
import os
import sys
import warnings

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import configs  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402


def _fix_semantics(cfg: configs.ExperimentConfig) -> None:
    loss = cfg.train.loss
    lt = str(getattr(loss.loss_type, "value", loss.loss_type))
    if lt == "ge2e_contrast":
        # Previously applied silently inside neural_net.py; now explicit.
        if float(loss.ge2e_init_b) == -5.0:
            loss.ge2e_init_b = -1.0
        loss.ge2e_contrast_surrogate = True
        loss.ge2e_contrast_b_grad_scale = 0.5
    if float(cfg.train.waveform_aug.snr_min_db) < 12.0:
        cfg.train.waveform_aug.snr_min_db = 12.0  # was clamped to 12 by the old code
    lr = float(cfg.train.learning_rate)
    cfg.train.min_learning_rate = max(1e-5, 0.01 * lr)  # old effective end-LR
    if int(cfg.export.sequence_length) in (0, 100):
        cfg.export.sequence_length = int(cfg.model.seq_len)  # 180 for all released models
    cfg.loss = cfg.train.loss


def migrate_yaml(path: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = configs.ExperimentConfig.from_yaml(path)
    _fix_semantics(cfg)
    with open(path, "r", encoding="utf-8") as f:
        header = [ln for ln in f.read().splitlines()[:5] if ln.startswith("#")]
    text = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write("\n".join(header) + "\n")
        f.write(text)


def migrate_config_json(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = configs.ExperimentConfig.from_dict(raw["experiment_dict"])
    _fix_semantics(cfg)
    new_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(cfg)
    new_cfg._explicit_embedding_dim = raw.get("_explicit_embedding_dim", True)
    new_cfg.save_pretrained(os.path.dirname(path))


def main() -> None:
    yml_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "*.yml")))
    for p in yml_paths:
        migrate_yaml(p)
    print(f"Migrated {len(yml_paths)} YAML configs")
    json_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "pretrained_models", "*", "config.json")))
    for p in json_paths:
        migrate_config_json(p)
        stale = os.path.join(os.path.dirname(p), "preprocessor_config.json")
        if os.path.exists(stale):
            os.remove(stale)
    print(f"Migrated {len(json_paths)} config.json files (and removed stale preprocessor_config.json)")
    # Verify: everything loads without any warning now.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for p in yml_paths:
            configs.ExperimentConfig.from_yaml(p)
        for p in json_paths:
            with open(p, "r", encoding="utf-8") as f:
                configs.ExperimentConfig.from_dict(json.load(f)["experiment_dict"])
    print("All migrated configs load with zero deprecation warnings.")


if __name__ == "__main__":
    main()
