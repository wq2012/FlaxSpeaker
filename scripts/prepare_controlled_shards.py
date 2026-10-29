#!/usr/bin/env python3
"""Build float16 MUSAN + RIR + speed-perturbed feature shards for the 44 controlled models.

Creates `sharded_caches/controlled_v050/{librispeech,cnceleb2,voxceleb1,universal}` from
the existing controlled `splits/*/train.csv` manifests (381 base speakers / 9,447 clean
utterances) combined with real `MUSAN` and `RIRS_NOISES` waveforms:
- 1x clean utterance (`spk`)
- 2x corpus-augmented `MUSAN/RIR` copies (`spk`)
- 1x `0.9x` speed-perturbed copy (`spk@sp0.9`, 50% `MUSAN/RIR`)
- 1x `1.1x` speed-perturbed copy (`spk@sp1.1`, 50% `MUSAN/RIR`)

Strictly enforces polite shared-machine CPU/memory caps (`<= 16` worker processes,
`nice -n 10`). Also writes `clean_cohort_utts.json` per domain (up to 5 clean utterances
per training speaker) for fast speaker-centroid imposter cohort extraction during AS-Norm.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import hashlib
import json
import os
import sys
import time
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import configs  # noqa: E402
from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import resources  # noqa: E402
from flaxspeaker import shard_cache  # noqa: E402
from flaxspeaker import waveform_aug  # noqa: E402


_WORKER_FE: Any = None
_WORKER_AUG: Any = None


def _init_worker(
    noise_files: list[str],
    music_files: list[str],
    speech_files: list[str],
    rir_files: list[str],
) -> None:
    global _WORKER_FE, _WORKER_AUG
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if hasattr(os, "nice"):
        try:
            cur = os.nice(0)
            if cur < 10:
                os.nice(10 - cur)
        except OSError:
            pass
    fe_cfg = configs.FrontendConfig(
        feature_type=configs.FeatureType.LOG_MEL.value,
        sample_rate=16000,
        n_mels=80,
        n_mfcc=80,
        apply_cmvn=True,
    )
    vad_cfg = configs.VadConfig(enabled=True, mode=configs.VadMode.ENERGY.value)
    _WORKER_FE = feature_extraction.AudioFrontend(fe_cfg, vad_cfg)
    aug = waveform_aug.MusanRirAugmenter(musan_dir="", rirs_dir="", sample_rate=16000)
    aug.noise_files = noise_files
    aug.music_files = music_files
    aug.speech_files = speech_files
    aug.rir_files = rir_files
    _WORKER_AUG = aug


def _stable_seed(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _process_one_utterance(
    task: tuple[str, str, str],
) -> list[tuple[str, str, bool, np.ndarray]]:
    """Extracts 5 float16 `(T, 80)` feature matrices for `(spk, dom, wav_path)`."""
    spk, dom, wav_path = task
    resolved = dataset.resolve_data_path(wav_path)
    wav, sr = sf.read(resolved, dtype="float32")
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    if sr != 16000:
        from fractions import Fraction
        from scipy import signal

        frac = Fraction(16000, sr).limit_denominator(200)
        wav = signal.resample_poly(wav, up=frac.numerator, down=frac.denominator)
    wav = np.asarray(wav, dtype=np.float32)

    rng = np.random.default_rng(_stable_seed(f"{dom}:{spk}:{os.path.basename(resolved)}"))
    outputs: list[tuple[str, str, bool, np.ndarray]] = []

    # 1. Clean 1.0x
    f_clean = np.asarray(_WORKER_FE.extract_from_waveform(wav), dtype=np.float16)
    outputs.append((spk, dom, False, f_clean))

    # 2 & 3. Two distinct MUSAN/RIR augmented copies at 1.0x (VAD anchored on clean wav)
    modes = _WORKER_AUG.available_modes
    chosen_modes = (
        rng.choice(modes, size=min(2, len(modes)), replace=False).tolist()
        if len(modes) >= 2
        else ["noise", "reverb"]
    )
    for m_name in chosen_modes:
        w_aug, _ = _WORKER_AUG.augment(wav, rng=rng, mode=str(m_name))
        f_aug = np.asarray(
            _WORKER_FE.extract_from_waveform(w_aug, clean_waveform=wav),
            dtype=np.float16,
        )
        outputs.append((spk, dom, True, f_aug))

    # 4 & 5. Speed-perturbed 0.9x and 1.1x (with 50% MUSAN/RIR; VAD anchored on clean w_sp)
    for sp_factor, suffix in ((0.9, "@sp0.9"), (1.1, "@sp1.1")):
        w_sp_clean = waveform_aug.apply_speed_perturbation(
            wav, sample_rate=16000, speed_factor=sp_factor
        )
        w_sp = w_sp_clean
        if _WORKER_AUG.is_ready and float(rng.random()) < 0.5:
            w_sp, _ = _WORKER_AUG.augment(w_sp_clean, rng=rng)
        f_sp = np.asarray(
            _WORKER_FE.extract_from_waveform(w_sp, clean_waveform=w_sp_clean),
            dtype=np.float16,
        )
        outputs.append((f"{spk}{suffix}", dom, True, f_sp))

    return outputs


def build_controlled_shards(
    data_root: str,
    shard_root: str,
    musan_dir: str,
    rirs_dir: str,
    num_workers: int = 16,
    overwrite: bool = False,
) -> dict[str, Any]:
    resources.configure_polite_resources(max_threads=max(4, num_workers + 2), nice_level=10)
    with open(
        os.path.join(data_root, "splits", "dataset_summary.json"), "r", encoding="utf-8"
    ) as f:
        ds_summary = json.load(f)

    # Check if already built
    all_done = not overwrite and all(
        os.path.isfile(os.path.join(shard_root, d, "metadata.json"))
        for d in ("librispeech", "cnceleb2", "voxceleb1", "universal")
    )
    if all_done:
        out_summary = {}
        for d in ("librispeech", "cnceleb2", "voxceleb1", "universal"):
            with open(os.path.join(shard_root, d, "metadata.json"), "r", encoding="utf-8") as f:
                out_summary[d] = json.load(f)
        print(f"Controlled v0.5.0 shards already exist at {shard_root}.")
        return out_summary

    os.makedirs(shard_root, exist_ok=True)
    univ_writer = shard_cache.ShardedFeatureWriter(
        os.path.join(shard_root, "universal"), feat_dim=80
    )
    univ_cohort_utts: dict[str, list[str]] = {}
    domain_metas: dict[str, Any] = {}

    t0_all = time.time()
    parent_aug = waveform_aug.MusanRirAugmenter(
        musan_dir=musan_dir, rirs_dir=rirs_dir, sample_rate=16000
    )
    print(
        f"[Shards] Indexed MUSAN/RIR once in parent ({time.time() - t0_all:.1f}s): "
        f"noise={len(parent_aug.noise_files)}, music={len(parent_aug.music_files)}, "
        f"speech={len(parent_aug.speech_files)}, rirs={len(parent_aug.rir_files)}",
        flush=True,
    )
    with futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(
            parent_aug.noise_files,
            parent_aug.music_files,
            parent_aug.speech_files,
            parent_aug.rir_files,
        ),
    ) as ex:
        for dname in ("librispeech", "cnceleb2", "voxceleb1"):
            t0 = time.time()
            spk_map = dataset.get_csv_spk_to_utts(ds_summary[dname]["train_csv"])
            dom_dir = os.path.join(shard_root, dname)
            dom_writer = shard_cache.ShardedFeatureWriter(dom_dir, feat_dim=80)
            dom_cohort: dict[str, list[str]] = {}

            tasks: list[tuple[str, str, str]] = []
            for spk in sorted(spk_map.keys()):
                utts = sorted(spk_map[spk])
                dom_cohort[spk] = utts[:5]
                univ_spk = f"{dname}:{spk}" if not spk.startswith(f"{dname}:") else spk
                univ_cohort_utts[univ_spk] = utts[:5]
                for u in utts:
                    tasks.append((spk, dname, u))

            print(
                f"[Shards] Extracting {dname}: {len(spk_map)} base speakers, "
                f"{len(tasks)} clean utterances -> {len(tasks) * 5} augmented items...",
                flush=True,
            )
            for idx, item_list in enumerate(ex.map(_process_one_utterance, tasks, chunksize=8), 1):
                for spk_label, dom_label, is_aug, feat_f16 in item_list:
                    dom_writer.add(
                        feat_f16,
                        speaker=spk_label,
                        domain=dom_label,
                        is_augmented=is_aug,
                    )
                    univ_spk_label = (
                        f"{dom_label}:{spk_label}"
                        if not spk_label.startswith(f"{dom_label}:")
                        else spk_label
                    )
                    univ_writer.add(
                        feat_f16,
                        speaker=univ_spk_label,
                        domain=dom_label,
                        is_augmented=is_aug,
                    )
                if idx % 1000 == 0 or idx == len(tasks):
                    print(
                        f"  [{dname}] {idx}/{len(tasks)} utterances processed "
                        f"({time.time() - t0:.1f}s)",
                        flush=True,
                    )

            meta_d = dom_writer.close(
                extra_metadata={
                    "dataset_name": dname,
                    "base_speakers": len(spk_map),
                    "clean_utterances": len(tasks),
                    "augmentation_recipe": "1x_clean + 2x_musan_rir + 2x_speed_0.9_1.1",
                }
            )
            with open(
                os.path.join(dom_dir, "clean_cohort_utts.json"), "w", encoding="utf-8"
            ) as cf:
                json.dump(dom_cohort, cf, indent=2)
            domain_metas[dname] = meta_d

    meta_u = univ_writer.close(
        extra_metadata={
            "dataset_name": "universal",
            "base_speakers": len(univ_cohort_utts),
            "clean_utterances": sum(
                domain_metas[d]["clean_utterances"]
                for d in ("librispeech", "cnceleb2", "voxceleb1")
            ),
            "augmentation_recipe": "1x_clean + 2x_musan_rir + 2x_speed_0.9_1.1",
        }
    )
    with open(
        os.path.join(shard_root, "universal", "clean_cohort_utts.json"),
        "w",
        encoding="utf-8",
    ) as ucf:
        json.dump(univ_cohort_utts, ucf, indent=2)
    domain_metas["universal"] = meta_u

    print(
        f"[Shards] Completed all 4 controlled v0.5.0 shard stores in "
        f"{time.time() - t0_all:.1f}s -> {shard_root}",
        flush=True,
    )
    return domain_metas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get(
            "FLAXSPEAKER_DATA_ROOT", os.path.expanduser("~/Data/speaker_datasets")
        ),
    )
    parser.add_argument(
        "--shard_root",
        type=str,
        default=os.path.expanduser(
            "~/Data/speaker_datasets/sharded_caches/controlled_v050"
        ),
    )
    parser.add_argument(
        "--musan_dir",
        type=str,
        default=os.path.expanduser("~/Data/speaker_datasets/full_corpora/musan"),
    )
    parser.add_argument(
        "--rirs_dir",
        type=str,
        default=os.path.expanduser(
            "~/Data/speaker_datasets/full_corpora/RIRS_NOISES"
        ),
    )
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_controlled_shards(
        data_root=args.data_root,
        shard_root=args.shard_root,
        musan_dir=args.musan_dir,
        rirs_dir=args.rirs_dir,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
