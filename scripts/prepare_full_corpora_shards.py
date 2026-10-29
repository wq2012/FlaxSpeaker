#!/usr/bin/env python3
"""Build full-data float16 MUSAN + RIR + speed-perturbed feature shards (`universal_full_v050`).

Pools all four open-source corpora for Tier-2 Universal Flagship training (`48`-model zoo):
1. `VoxCeleb2-dev` (`full_corpora/voxceleb2_dev_wds/voxceleb2-dev-*.tar`, streamed in-memory via PyAV)
2. `VoxCeleb1-dev` (`full_corpora/voxceleb1_dev_wav`, strictly disjoint from all 40 `VoxCeleb1-O` speakers)
3. `CN-Celeb1-dev` (`full_corpora/CN-Celeb_flac/dev/dev.lst`, strictly disjoint from `CN-Celeb-E`)
   + `CN-Celeb2` (`full_corpora/CN-Celeb2_flac/data`, strictly disjoint from `splits/cnceleb2/eval.csv`)
4. `LibriSpeech 960h + dev-clean` (`train-clean-100`, `train-clean-360`, `train-other-500`, `dev-clean`,
   strictly disjoint from `test-clean` and `splits/librispeech/eval.csv`)

Guarantees & Deliverables:
- Asserts **zero speaker overlap** between training speakers, held-out validation speakers, and
  every evaluation protocol (`VoxCeleb1-O`, `CN-Celeb-E`, `LibriSpeech test-clean`, and all
  controlled `splits/*/eval.csv` speaker sets).
- Holds out `90` speaker-disjoint validation speakers (`30` Vox + `30` CN + `30` LS) under
  `splits/universal_full_val/` (`trials.tsv` 3,000 1-utt + `multi_enroll_trials.tsv` 1,500 3-utt)
  for convergence tracking, wide `PFAS` hyperparameter selection, `QMF` calibration, and fusion weights.
- Generates deterministic official `LibriSpeech test-clean` trial lists under `splits/official_eval/`.
- Writes `1x` clean + `1x` `MUSAN/RIR` + `0.9x` speed + `1.1x` speed (`float16`, capped at `6.2s`
  so `6.0s` = `600`-frame `LMFT` is supported across all classes) into `2 GB` shards under
  `sharded_caches/universal_full_v050`, plus `cohort_clean_item_indices.json` (`600` disjoint
  cohort speakers) for fast Top-300 `AS-Norm` centroid computation.
- Enforces polite shared-host resource caps (`<= 22` worker processes, `OMP_NUM_THREADS=1`,
  `nice -n 10`, and `min_free_gb >= 1850.0` safety floor).
"""

from __future__ import annotations

import argparse
from concurrent import futures
import csv
import glob
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
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
from flaxspeaker import waveform_aug  # noqa: E402

VOX1_O_TEST_SPEAKERS = {f"id{spk}" for spk in range(10270, 10310)}
MAX_WAVEFORM_SAMPLES = 99200  # 6.2s @ 16kHz -> >= 600 frames for 6.0s LMFT
MAX_SP_WAVEFORM_SAMPLES = 51200  # 3.2s @ 16kHz -> >= 300 frames for Stage-1 2.0s training
TARGET_SHARD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB per final .f16 shard
_VOX_SPK_RE = re.compile(r"(id\d{5})")

_WORKER_FE: Any = None
_WORKER_AUG: Any = None
_SHARED_PARENT_AUG: Any = None
_HAS_PYAV: bool = False


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts} UTC] {msg}", flush=True)


def _check_free_disk_gb(path: str, min_free_gb: float) -> float:
    os.makedirs(path, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Disk safety abort: {free_gb:.1f} GB free (< {min_free_gb:.1f} GB floor)."
        )
    return free_gb


def _stable_seed(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _init_worker(
    noise_files: list[str],
    music_files: list[str],
    speech_files: list[str],
    rir_files: list[str],
) -> None:
    global _WORKER_FE, _WORKER_AUG, _HAS_PYAV
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
    try:
        import av  # noqa: F401

        _HAS_PYAV = True
    except ImportError:
        _HAS_PYAV = False

    fe_cfg = configs.FrontendConfig(
        feature_type=configs.FeatureType.LOG_MEL.value,
        sample_rate=16000,
        n_mels=80,
        n_mfcc=80,
        apply_cmvn=True,
    )
    vad_cfg = configs.VadConfig(enabled=True, mode=configs.VadMode.ENERGY.value)
    _WORKER_FE = feature_extraction.AudioFrontend(fe_cfg, vad_cfg)
    if _SHARED_PARENT_AUG is not None:
        _WORKER_AUG = _SHARED_PARENT_AUG
    else:
        aug = waveform_aug.MusanRirAugmenter(
            musan_dir="", rirs_dir="", sample_rate=16000
        )
        aug.noise_files = noise_files
        aug.music_files = music_files
        aug.speech_files = speech_files
        aug.rir_files = rir_files
        aug.preload_cache()
        _WORKER_AUG = aug


def _decode_m4a_bytes_16k(raw_bytes: bytes) -> np.ndarray:
    """Decodes AAC/M4A bytes to 16 kHz mono float32 waveform in memory."""
    if _HAS_PYAV:
        import av

        with av.open(io.BytesIO(raw_bytes)) as container:
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            chunks: list[np.ndarray] = []
            for frame in container.decode(audio=0):
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray()[0])
            for rf in resampler.resample(None):
                chunks.append(rf.to_ndarray()[0])
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        input=raw_bytes,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _read_file_16k(filepath: str) -> np.ndarray:
    wav, sr = sf.read(filepath, dtype="float32")
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    if sr != 16000:
        from fractions import Fraction
        from scipy import signal

        frac = Fraction(16000, sr).limit_denominator(200)
        wav = signal.resample_poly(wav, up=frac.numerator, down=frac.denominator)
    return np.asarray(wav, dtype=np.float32)


def _crop_waveform_max_len(
    wav: np.ndarray, rng: np.random.Generator, max_samples: int = MAX_WAVEFORM_SAMPLES
) -> np.ndarray:
    if wav.shape[0] <= max_samples:
        return wav
    start = int(rng.integers(0, wav.shape[0] - max_samples + 1))
    return wav[start:start + max_samples].copy()


def _extract_four_views(
    wav: np.ndarray, spk: str, dom: str, key_id: str
) -> list[tuple[str, str, int, np.ndarray]]:
    """Returns `[clean_1.0x, musan_rir_1.0x, sp0.9, sp1.1]` float16 `(T, 80)` matrices."""
    if wav.shape[0] < 3200:  # Skip corrupt < 0.2s clips
        return []
    rng = np.random.default_rng(_stable_seed(key_id))
    w_clean = _crop_waveform_max_len(wav, rng, MAX_WAVEFORM_SAMPLES)
    out: list[tuple[str, str, int, np.ndarray]] = []

    f_clean = np.asarray(_WORKER_FE.extract_from_waveform(w_clean), dtype=np.float16)
    if f_clean.shape[0] < 16:
        return []
    out.append((spk, dom, 0, f_clean))

    if _WORKER_AUG.is_ready:
        w_aug, _ = _WORKER_AUG.augment(w_clean, rng=rng)
    else:
        w_aug = w_clean
    f_aug = np.asarray(
        _WORKER_FE.extract_from_waveform(w_aug, clean_waveform=w_clean),
        dtype=np.float16,
    )
    if f_aug.shape[0] >= 16:
        out.append((spk, dom, 1, f_aug))

    w_sp_base = _crop_waveform_max_len(w_clean, rng, MAX_SP_WAVEFORM_SAMPLES)
    for sp_factor, suffix in ((0.9, "@sp0.9"), (1.1, "@sp1.1")):
        w_sp_clean = waveform_aug.apply_speed_perturbation(
            w_sp_base, sample_rate=16000, speed_factor=sp_factor
        )
        w_sp_clean = _crop_waveform_max_len(w_sp_clean, rng, MAX_SP_WAVEFORM_SAMPLES)
        w_sp = w_sp_clean
        if _WORKER_AUG.is_ready and float(rng.random()) < 0.5:
            w_sp, _ = _WORKER_AUG.augment(w_sp_clean, rng=rng)
        f_sp = np.asarray(
            _WORKER_FE.extract_from_waveform(w_sp, clean_waveform=w_sp_clean),
            dtype=np.float16,
        )
        if f_sp.shape[0] >= 16:
            out.append((f"{spk}{suffix}", dom, 1, f_sp))
    return out


def _run_work_unit(unit: dict[str, Any]) -> dict[str, Any]:
    """Processes one `.tar` shard or filesystem utterance batch into `part_XXXX.{f16,json}`."""
    unit_id = int(unit["unit_id"])
    parts_dir = str(unit["parts_dir"])
    done_marker = os.path.join(parts_dir, f"part_{unit_id:04d}.done")
    meta_json = os.path.join(parts_dir, f"part_{unit_id:04d}.json")
    f16_path = os.path.join(parts_dir, f"part_{unit_id:04d}.f16")

    if os.path.isfile(done_marker) and os.path.isfile(meta_json) and os.path.isfile(f16_path):
        with open(meta_json, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        return {
            "unit_id": unit_id,
            "num_clean": cached_meta["num_clean"],
            "num_items": len(cached_meta["records"]),
            "total_frames": cached_meta["total_frames"],
            "skipped_errors": cached_meta.get("skipped_errors", 0),
        }

    records: list[tuple[str, str, int, int]] = []  # (speaker, domain, aug_flag, num_frames)
    num_clean = 0
    skipped_errors = 0
    total_frames = 0

    with open(f16_path, "wb") as out_f:
        if unit["kind"] == "vox2_tar":
            tar_path = str(unit["tar_path"])
            excluded_raw_spks = set(unit["excluded_raw_spks"])
            with tarfile.open(tar_path, "r") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    lower_name = member.name.lower()
                    if not lower_name.endswith((".m4a", ".aac", ".wav", ".flac")):
                        continue
                    m_spk = _VOX_SPK_RE.search(member.name)
                    raw_spk = m_spk.group(1) if m_spk else None
                    if raw_spk is None or raw_spk in excluded_raw_spks:
                        continue
                    spk_label = f"voxceleb2:{raw_spk}"
                    try:
                        fobj = tf.extractfile(member)
                        if fobj is None:
                            continue
                        raw_bytes = fobj.read()
                        wav = _decode_m4a_bytes_16k(raw_bytes)
                        views = _extract_four_views(
                            wav, spk_label, "voxceleb", f"vox2:{member.name}"
                        )
                    except Exception:
                        skipped_errors += 1
                        continue
                    if not views:
                        continue
                    num_clean += 1
                    for v_spk, v_dom, v_aug, feat_f16 in views:
                        arr_c = np.ascontiguousarray(feat_f16, dtype=np.float16)
                        out_f.write(arr_c.tobytes())
                        nf = int(arr_c.shape[0])
                        records.append((v_spk, v_dom, v_aug, nf))
                        total_frames += nf
        else:
            for spk_label, dom_label, wav_path in unit["items"]:
                try:
                    wav = _read_file_16k(wav_path)
                    views = _extract_four_views(
                        wav, spk_label, dom_label, f"{spk_label}:{wav_path}"
                    )
                except Exception:
                    skipped_errors += 1
                    continue
                if not views:
                    continue
                num_clean += 1
                for v_spk, v_dom, v_aug, feat_f16 in views:
                    arr_c = np.ascontiguousarray(feat_f16, dtype=np.float16)
                    out_f.write(arr_c.tobytes())
                    nf = int(arr_c.shape[0])
                    records.append((v_spk, v_dom, v_aug, nf))
                    total_frames += nf

    payload = {
        "unit_id": unit_id,
        "num_clean": num_clean,
        "total_frames": total_frames,
        "skipped_errors": skipped_errors,
        "records": records,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    with open(done_marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    return {
        "unit_id": unit_id,
        "num_clean": num_clean,
        "num_items": len(records),
        "total_frames": total_frames,
        "skipped_errors": skipped_errors,
    }


def _read_csv_speakers(csv_path: str) -> set[str]:
    if not os.path.isfile(csv_path):
        return set()
    spk_map = dataset.get_csv_spk_to_utts(csv_path)
    return {s.split(":")[-1] for s in spk_map.keys() if s}


def _collect_eval_exclusion_sets(data_root: str) -> dict[str, set[str]]:
    """Collects all speaker IDs across official & controlled evaluation benchmarks."""
    full_root = os.path.join(data_root, "full_corpora")

    # 1. VoxCeleb1-O (40 official speakers) + controlled splits/voxceleb1/eval.csv
    vox_excl = set(VOX1_O_TEST_SPEAKERS)
    veri_path = os.path.join(data_root, "voxceleb1", "veri_test2.txt")
    if os.path.isfile(veri_path):
        with open(veri_path, "r", encoding="utf-8") as f:
            for line in f:
                toks = line.strip().split()
                if len(toks) == 3:
                    vox_excl.add(toks[1].split("/")[0])
                    vox_excl.add(toks[2].split("/")[0])
    vox_excl |= _read_csv_speakers(
        os.path.join(data_root, "splits", "voxceleb1", "eval.csv")
    )

    # 2. CN-Celeb-E (200 official speakers) + controlled splits/cnceleb2/eval.csv (30 speakers)
    cn_excl: set[str] = set()
    cn1_eval_lists = os.path.join(full_root, "CN-Celeb_flac", "eval", "lists")
    for lst_name in ("enroll.lst", "enroll.map", "test.lst"):
        p = os.path.join(cn1_eval_lists, lst_name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    toks = line.strip().split()
                    if toks:
                        base_id = os.path.basename(toks[0]).split("-")[0].split(".")[0]
                        if base_id.startswith("id"):
                            cn_excl.add(base_id)
    cn_excl |= _read_csv_speakers(
        os.path.join(data_root, "splits", "cnceleb2", "eval.csv")
    )

    # 3. LibriSpeech test-clean (40 official speakers) + controlled splits/librispeech/eval.csv
    ls_excl: set[str] = set()
    ls_test_dir = os.path.join(data_root, "librispeech", "LibriSpeech", "test-clean")
    if os.path.isdir(ls_test_dir):
        for entry in os.listdir(ls_test_dir):
            if os.path.isdir(os.path.join(ls_test_dir, entry)):
                ls_excl.add(entry)
    ls_excl |= _read_csv_speakers(
        os.path.join(data_root, "splits", "librispeech", "eval.csv")
    )

    # Also cross-check `splits/universal/eval.csv` domain-prefixed speaker IDs
    univ_eval_csv = os.path.join(data_root, "splits", "universal", "eval.csv")
    if os.path.isfile(univ_eval_csv):
        for dom_spk in dataset.get_csv_spk_to_utts(univ_eval_csv).keys():
            if ":" in dom_spk:
                pref, raw_id = dom_spk.split(":", 1)
                if "vox" in pref:
                    vox_excl.add(raw_id)
                elif "cn" in pref:
                    cn_excl.add(raw_id)
                elif "libri" in pref:
                    ls_excl.add(raw_id)

    if len(vox_excl) < 40 or len(cn_excl) < 220 or len(ls_excl) < 40:
        raise AssertionError(
            f"Incomplete eval exclusion sets: vox={len(vox_excl)}, "
            f"cn={len(cn_excl)}, ls={len(ls_excl)}"
        )
    return {"voxceleb": vox_excl, "cnceleb": cn_excl, "librispeech": ls_excl}


def _ensure_official_ls_test_clean_trials(data_root: str) -> dict[str, str]:
    """Generates deterministic official LibriSpeech `test-clean` 1-utt & 3-utt trial lists."""
    out_dir = os.path.join(data_root, "splits", "official_eval")
    os.makedirs(out_dir, exist_ok=True)
    t1_path = os.path.join(out_dir, "librispeech_test_clean_trials.tsv")
    t3_path = os.path.join(out_dir, "librispeech_test_clean_3utt_trials.tsv")

    ls_test_dir = os.path.join(data_root, "librispeech", "LibriSpeech", "test-clean")
    spk_map = dataset.get_librispeech_spk_to_utts(ls_test_dir)
    if not os.path.isfile(t1_path):
        trials_1 = dataset.generate_verification_trials(
            spk_map, num_trials=10000, positive_ratio=0.5, seed=202605
        )
        dataset.save_trials_file(trials_1, t1_path)
    if not os.path.isfile(t3_path):
        trials_3 = dataset.generate_multi_enroll_trials(
            spk_map,
            num_trials=3000,
            num_enroll_utts=3,
            positive_ratio=0.5,
            seed=202605,
        )
        with open(t3_path, "w", encoding="utf-8") as f:
            for lbl, e_list, t_u in trials_3:
                f.write(f"{int(lbl)}\t{','.join(e_list)}\t{t_u}\n")
    return {"trials_1utt": t1_path, "trials_3utt": t3_path}


def _collect_filesystem_corpora(
    data_root: str, excl: dict[str, set[str]]
) -> dict[str, dict[str, list[str]]]:
    """Scans VoxCeleb1-dev, CN-Celeb1-dev + CN-Celeb2, and LibriSpeech 960h+dev."""
    full_root = os.path.join(data_root, "full_corpora")

    # 1. VoxCeleb1-dev
    vox1_dir = os.path.join(full_root, "voxceleb1_dev_wav")
    if not os.path.isdir(vox1_dir):
        raise FileNotFoundError(f"Missing VoxCeleb1-dev directory: {vox1_dir}")
    vox1_raw = dataset.get_voxceleb_spk_to_utts(vox1_dir)
    vox1_clean = {
        f"voxceleb1:{spk}": sorted(utts)
        for spk, utts in sorted(vox1_raw.items())
        if spk not in excl["voxceleb"] and utts
    }

    # 2. CN-Celeb1-dev (strictly restricted to dev.lst) + CN-Celeb2
    cn1_root = os.path.join(full_root, "CN-Celeb_flac")
    dev_lst = os.path.join(cn1_root, "dev", "dev.lst")
    if not os.path.isfile(dev_lst):
        raise FileNotFoundError(f"Missing CN-Celeb1 dev.lst: {dev_lst}")
    with open(dev_lst, "r", encoding="utf-8") as f:
        cn1_dev_allowed = {line.strip() for line in f if line.strip()}
    cn_clean: dict[str, list[str]] = {}
    cn1_data_dir = os.path.join(cn1_root, "data")
    for spk in sorted(cn1_dev_allowed):
        if spk in excl["cnceleb"]:
            continue
        s_dir = os.path.join(cn1_data_dir, spk)
        if not os.path.isdir(s_dir):
            continue
        flacs = sorted(
            os.path.join(s_dir, fn)
            for fn in os.listdir(s_dir)
            if fn.endswith((".flac", ".wav"))
        )
        if flacs:
            cn_clean[f"cnceleb1:{spk}"] = flacs

    cn2_data_dir = os.path.join(full_root, "CN-Celeb2_flac", "data")
    if not os.path.isdir(cn2_data_dir):
        raise FileNotFoundError(f"Missing CN-Celeb2 data directory: {cn2_data_dir}")
    for spk in sorted(os.listdir(cn2_data_dir)):
        if spk in excl["cnceleb"]:
            continue
        s_dir = os.path.join(cn2_data_dir, spk)
        if not os.path.isdir(s_dir):
            continue
        flacs = sorted(
            os.path.join(s_dir, fn)
            for fn in os.listdir(s_dir)
            if fn.endswith((".flac", ".wav"))
        )
        if flacs:
            cn_clean[f"cnceleb2:{spk}"] = flacs

    # 3. LibriSpeech train-clean-100 + train-clean-360 + train-other-500 + dev-clean
    ls_base = os.path.join(data_root, "librispeech", "LibriSpeech")
    ls_clean: dict[str, list[str]] = {}
    for subset in (
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
    ):
        sub_dir = os.path.join(ls_base, subset)
        if not os.path.isdir(sub_dir):
            raise FileNotFoundError(f"Missing LibriSpeech subset: {sub_dir}")
        sub_map = dataset.get_librispeech_spk_to_utts(sub_dir)
        for spk, utts in sorted(sub_map.items()):
            if spk in excl["librispeech"]:
                continue
            ls_clean.setdefault(f"librispeech:{spk}", []).extend(sorted(utts))

    # Explicit zero-speaker-overlap assertions before proceeding
    assert not ({s.split(":", 1)[1] for s in vox1_clean} & excl["voxceleb"])
    assert not ({s.split(":", 1)[1] for s in cn_clean} & excl["cnceleb"])
    assert not ({s.split(":", 1)[1] for s in ls_clean} & excl["librispeech"])
    return {"voxceleb": vox1_clean, "cnceleb": cn_clean, "librispeech": ls_clean}


def _carve_out_validation_split(
    data_root: str,
    fs_corpora: dict[str, dict[str, list[str]]],
    spks_per_domain: int = 30,
    seed: int = 202605,
) -> dict[str, Any]:
    """Holds out 90 validation speakers (30/domain) and removes them from `fs_corpora`."""
    val_dir = os.path.join(data_root, "splits", "universal_full_val")
    os.makedirs(val_dir, exist_ok=True)
    meta_path = os.path.join(val_dir, "val_metadata.json")
    t1_path = os.path.join(val_dir, "trials.tsv")
    t3_path = os.path.join(val_dir, "multi_enroll_trials.tsv")

    rng = random.Random(seed)
    val_spk_to_utts: dict[str, list[str]] = {}
    domain_val_maps: dict[str, dict[str, list[str]]] = {}

    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        chosen_by_dom = saved["val_speakers_by_domain"]
        for dom, spk_list in chosen_by_dom.items():
            d_map: dict[str, list[str]] = {}
            for s in spk_list:
                if s in fs_corpora[dom]:
                    utts = fs_corpora[dom].pop(s)[:30]
                    val_spk_to_utts[s] = utts
                    d_map[s] = utts
            domain_val_maps[dom] = d_map
        return saved

    chosen_by_dom = {}
    all_t1: list[dataset.TrialPair] = []
    all_t3: list[dataset.MultiEnrollTrial] = []

    for dom in ("voxceleb", "cnceleb", "librispeech"):
        candidates = [
            s for s, u in sorted(fs_corpora[dom].items()) if len(u) >= 10
        ]
        chosen = sorted(rng.sample(candidates, spks_per_domain))
        chosen_by_dom[dom] = chosen
        d_map = {}
        for s in chosen:
            utts = sorted(fs_corpora[dom].pop(s))[:30]
            val_spk_to_utts[s] = utts
            d_map[s] = utts
        domain_val_maps[dom] = d_map

        dom_t1 = dataset.generate_verification_trials(
            d_map, num_trials=1000, positive_ratio=0.5, seed=seed + len(all_t1)
        )
        dom_t3 = dataset.generate_multi_enroll_trials(
            d_map,
            num_trials=500,
            num_enroll_utts=3,
            positive_ratio=0.5,
            seed=seed + len(all_t3) + 17,
        )
        dom_t1_file = os.path.join(val_dir, f"{dom}_trials.tsv")
        dom_t3_file = os.path.join(val_dir, f"{dom}_multi_enroll_trials.tsv")
        dataset.save_trials_file(dom_t1, dom_t1_file)
        with open(dom_t3_file, "w", encoding="utf-8") as f:
            for lbl, e_list, t_u in dom_t3:
                f.write(f"{int(lbl)}\t{','.join(e_list)}\t{t_u}\n")
        all_t1.extend(dom_t1)
        all_t3.extend(dom_t3)

    dataset.save_trials_file(all_t1, t1_path)
    with open(t3_path, "w", encoding="utf-8") as f:
        for lbl, e_list, t_u in all_t3:
            f.write(f"{int(lbl)}\t{','.join(e_list)}\t{t_u}\n")

    val_csv = os.path.join(val_dir, "val_utterances.csv")
    with open(val_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "filepath"])
        for s in sorted(val_spk_to_utts):
            for u in val_spk_to_utts[s]:
                w.writerow([s, u])

    val_meta = {
        "num_val_speakers": len(val_spk_to_utts),
        "val_speakers_by_domain": chosen_by_dom,
        "num_trials_1utt": len(all_t1),
        "num_trials_3utt": len(all_t3),
        "trials_1utt_path": t1_path,
        "trials_3utt_path": t3_path,
        "val_utterances_csv": val_csv,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(val_meta, f, indent=2)
    return val_meta


def _merge_parts_into_final_shards(
    shard_dir: str,
    parts_dir: str,
    num_units: int,
    extra_metadata: dict[str, Any],
    clean_cohort_utts: dict[str, list[str]],
) -> dict[str, Any]:
    """Streams `part_XXXX.f16` files into `2 GiB` `shard_XXXX.f16` files and deletes parts."""
    feat_dim = 80
    bytes_per_frame = feat_dim * 2
    max_frames_per_shard = TARGET_SHARD_BYTES // bytes_per_frame

    speaker_to_id: dict[str, int] = {}
    domain_to_id: dict[str, int] = {}
    shard_ids: list[int] = []
    frame_offsets: list[int] = []
    frame_lengths: list[int] = []
    speaker_ids: list[int] = []
    domain_ids: list[int] = []
    aug_flags: list[int] = []
    shard_frame_counts: list[int] = []

    # Track up to 5 clean item indices per base speaker by domain for Top-300 AS-Norm cohort
    clean_items_by_dom_spk: dict[str, dict[str, list[int]]] = {
        "voxceleb": {},
        "cnceleb": {},
        "librispeech": {},
    }

    cur_shard_id = 0
    cur_frame_offset = 0
    cur_shard_path = os.path.join(shard_dir, f"shard_{cur_shard_id:04d}.f16")
    cur_out = open(cur_shard_path, "wb")

    total_clean_utts = 0
    total_skipped_errors = 0

    for uid in range(num_units):
        p_meta_path = os.path.join(parts_dir, f"part_{uid:04d}.json")
        p_f16_path = os.path.join(parts_dir, f"part_{uid:04d}.f16")
        with open(p_meta_path, "r", encoding="utf-8") as f:
            p_meta = json.load(f)

        p_frames = int(p_meta["total_frames"])
        total_clean_utts += int(p_meta["num_clean"])
        total_skipped_errors += int(p_meta.get("skipped_errors", 0))

        if p_frames > 0:
            if cur_frame_offset > 0 and cur_frame_offset + p_frames > max_frames_per_shard:
                cur_out.flush()
                cur_out.close()
                shard_frame_counts.append(cur_frame_offset)
                cur_shard_id += 1
                cur_frame_offset = 0
                cur_shard_path = os.path.join(shard_dir, f"shard_{cur_shard_id:04d}.f16")
                cur_out = open(cur_shard_path, "wb")

            with open(p_f16_path, "rb") as src_f:
                shutil.copyfileobj(src_f, cur_out, length=16 * 1024 * 1024)

            for v_spk, v_dom, v_aug, nf in p_meta["records"]:
                spk_id = speaker_to_id.setdefault(v_spk, len(speaker_to_id))
                dom_id = domain_to_id.setdefault(v_dom, len(domain_to_id))
                item_idx = len(frame_lengths)
                shard_ids.append(cur_shard_id)
                frame_offsets.append(cur_frame_offset)
                frame_lengths.append(int(nf))
                speaker_ids.append(spk_id)
                domain_ids.append(dom_id)
                aug_flags.append(int(v_aug))
                cur_frame_offset += int(nf)

                if int(v_aug) == 0 and "@sp" not in v_spk and v_dom in clean_items_by_dom_spk:
                    lst = clean_items_by_dom_spk[v_dom].setdefault(v_spk, [])
                    if len(lst) < 5:
                        lst.append(item_idx)

        os.remove(p_f16_path)

    cur_out.flush()
    cur_out.close()
    if cur_frame_offset > 0:
        shard_frame_counts.append(cur_frame_offset)
    elif os.path.exists(cur_shard_path):
        os.remove(cur_shard_path)

    shutil.rmtree(parts_dir, ignore_errors=True)

    np.savez(
        os.path.join(shard_dir, "index.npz"),
        shard_ids=np.asarray(shard_ids, dtype=np.int32),
        frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
        frame_lengths=np.asarray(frame_lengths, dtype=np.int32),
        speaker_ids=np.asarray(speaker_ids, dtype=np.int32),
        domain_ids=np.asarray(domain_ids, dtype=np.int16),
        aug_flags=np.asarray(aug_flags, dtype=np.uint8),
    )
    with open(os.path.join(shard_dir, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"speaker_to_id": speaker_to_id, "domain_to_id": domain_to_id}, f
        )

    # Select 600 balanced disjoint cohort speakers (200 Vox + 200 CN + 200 LS)
    rng_cohort = random.Random(202605)
    cohort_indices: dict[str, list[int]] = {}
    for dom_k in ("voxceleb", "cnceleb", "librispeech"):
        elig = [
            s for s, idxs in sorted(clean_items_by_dom_spk[dom_k].items()) if len(idxs) >= 3
        ]
        chosen_c = sorted(rng_cohort.sample(elig, min(200, len(elig))))
        for s in chosen_c:
            cohort_indices[s] = clean_items_by_dom_spk[dom_k][s]

    with open(
        os.path.join(shard_dir, "cohort_clean_item_indices.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(cohort_indices, f, indent=2)
    with open(
        os.path.join(shard_dir, "clean_cohort_utts.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(clean_cohort_utts, f, indent=2)

    base_spks = {k.split("@sp", 1)[0] for k in speaker_to_id}
    meta = {
        "format_version": "flaxspeaker_f16_v1",
        "dtype": "float16",
        "feat_dim": feat_dim,
        "num_shards": len(shard_frame_counts),
        "shard_frame_counts": shard_frame_counts,
        "num_items": len(frame_lengths),
        "total_frames": int(sum(shard_frame_counts)),
        "num_speakers": len(speaker_to_id),
        "num_base_speakers": len(base_spks),
        "clean_utterances": total_clean_utts,
        "skipped_decode_errors": total_skipped_errors,
        "num_domains": len(domain_to_id),
        **extra_metadata,
    }
    with open(os.path.join(shard_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def _select_strided_utterances(utts: list[str], max_utts: int) -> list[str]:
    if max_utts <= 0 or len(utts) <= max_utts:
        return list(utts)
    idxs = np.linspace(0, len(utts) - 1, num=max_utts, dtype=np.int64)
    return [utts[int(i)] for i in idxs]


def build_full_corpora_shards(
    data_root: str,
    shard_dir: str,
    num_workers: int = 22,
    min_free_gb: float = 1850.0,
    max_vox2_tars: int = 160,
    max_utts_per_fs_spk: int = 40,
    cpu_offset: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    global _SHARED_PARENT_AUG
    resources.configure_polite_resources(
        max_threads=max(4, min(28, num_workers + 2)), nice_level=10
    )
    if cpu_offset > 0 and hasattr(os, "sched_setaffinity"):
        try:
            total_cpus = os.cpu_count() or 64
            end_cpu = min(total_cpus, cpu_offset + max(4, min(24, num_workers + 2)))
            os.sched_setaffinity(0, set(range(cpu_offset, end_cpu)))
        except OSError:
            pass
    meta_out = os.path.join(shard_dir, "metadata.json")
    if not overwrite and os.path.isfile(meta_out):
        with open(meta_out, "r", encoding="utf-8") as f:
            existing = json.load(f)
        _log(f"Full-corpora shards already exist at {shard_dir}: {existing['num_items']:,} items.")
        return existing

    free_gb = _check_free_disk_gb(data_root, min_free_gb)
    _log(f"Building universal_full_v050 shards (initial free disk: {free_gb:.1f} GB)...")

    ls_official_paths = _ensure_official_ls_test_clean_trials(data_root)
    excl = _collect_eval_exclusion_sets(data_root)
    fs_corpora = _collect_filesystem_corpora(data_root, excl)
    val_meta = _carve_out_validation_split(data_root, fs_corpora, spks_per_domain=30)

    # Verify zero overlap between remaining fs_corpora training speakers and val speakers
    val_spk_set = {
        s for lst in val_meta["val_speakers_by_domain"].values() for s in lst
    }
    train_fs_spk_set = {
        s for dom_map in fs_corpora.values() for s in dom_map.keys()
    }
    if train_fs_spk_set & val_spk_set:
        raise AssertionError("Leakage detected between training and validation speakers!")

    # Locate VoxCeleb2-dev WebDataset .tar shards (shuffled across all 5,994 speakers)
    full_root = os.path.join(data_root, "full_corpora")
    vox2_wds_dir = os.path.join(full_root, "voxceleb2_dev_wds")
    vox2_tars = sorted(glob.glob(os.path.join(vox2_wds_dir, "voxceleb2-dev-*.tar")))
    if not vox2_tars:
        raise FileNotFoundError(f"No voxceleb2-dev-*.tar shards found in {vox2_wds_dir}")
    if max_vox2_tars > 0:
        vox2_tars = vox2_tars[:max_vox2_tars]

    # Build clean_cohort_utts.json (200 filesystem speakers x 5 utts) as secondary backup
    clean_cohort_utts: dict[str, list[str]] = {}
    for dom_k in ("voxceleb", "cnceleb", "librispeech"):
        for spk in sorted(fs_corpora[dom_k].keys())[:200]:
            clean_cohort_utts[spk] = fs_corpora[dom_k][spk][:5]

    os.makedirs(shard_dir, exist_ok=True)
    parts_dir = os.path.join(shard_dir, "_parts_tmp")
    os.makedirs(parts_dir, exist_ok=True)

    # Assemble deterministic work units
    work_units: list[dict[str, Any]] = []
    vox2_excl_list = sorted(excl["voxceleb"])
    for tar_p in vox2_tars:
        work_units.append(
            {
                "unit_id": len(work_units),
                "kind": "vox2_tar",
                "tar_path": tar_p,
                "excluded_raw_spks": vox2_excl_list,
                "parts_dir": parts_dir,
            }
        )

    flat_fs_items: list[tuple[str, str, str]] = []
    for dom_k in ("voxceleb", "cnceleb", "librispeech"):
        for spk, utts in sorted(fs_corpora[dom_k].items()):
            for u in _select_strided_utterances(utts, max_utts_per_fs_spk):
                flat_fs_items.append((spk, dom_k, u))

    batch_sz = 2500
    for start_i in range(0, len(flat_fs_items), batch_sz):
        work_units.append(
            {
                "unit_id": len(work_units),
                "kind": "fs_batch",
                "items": flat_fs_items[start_i:start_i + batch_sz],
                "parts_dir": parts_dir,
            }
        )

    musan_dir = os.path.join(full_root, "musan")
    rirs_dir = os.path.join(full_root, "RIRS_NOISES")
    parent_aug = waveform_aug.MusanRirAugmenter(
        musan_dir=musan_dir, rirs_dir=rirs_dir, sample_rate=16000
    )
    parent_aug.preload_cache()
    _SHARED_PARENT_AUG = parent_aug
    _log(
        f"Prepared {len(work_units)} work units ({len(vox2_tars)} VoxCeleb2 .tar shards + "
        f"{len(flat_fs_items):,} filesystem clean utterances across "
        f"{len(train_fs_spk_set):,} filesystem speakers; {val_meta['num_val_speakers']} val speakers held out)."
    )

    t0 = time.time()
    done_units = 0
    running_clean = 0
    running_items = 0
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
        for res in ex.map(_run_work_unit, work_units, chunksize=1):
            done_units += 1
            running_clean += int(res["num_clean"])
            running_items += int(res["num_items"])
            if done_units % 25 == 0 or done_units == len(work_units):
                free_now = _check_free_disk_gb(shard_dir, min_free_gb)
                elapsed = time.time() - t0
                rate = running_clean / max(1.0, elapsed)
                _log(
                    f"[FullShards] {done_units}/{len(work_units)} units | "
                    f"clean_utts={running_clean:,} | shard_items={running_items:,} | "
                    f"{rate:.1f} clean_utts/s | free_disk={free_now:.1f} GB"
                )

    _log("Merging temporary part files into 2 GiB contiguous .f16 shards...")
    extra_meta = {
        "dataset_name": "universal_full",
        "augmentation_recipe": "1x_clean + 1x_musan_rir + 2x_speed_0.9_1.1",
        "max_waveform_seconds": round(MAX_WAVEFORM_SAMPLES / 16000.0, 2),
        "zero_eval_speaker_overlap_verified": True,
        "excluded_eval_counts": {k: len(v) for k, v in excl.items()},
        "val_split": val_meta,
        "official_librispeech_test_clean": ls_official_paths,
    }
    final_meta = _merge_parts_into_final_shards(
        shard_dir=shard_dir,
        parts_dir=parts_dir,
        num_units=len(work_units),
        extra_metadata=extra_meta,
        clean_cohort_utts=clean_cohort_utts,
    )
    free_final = shutil.disk_usage(shard_dir).free / (1024**3)
    _log(
        f"Completed universal_full_v050 in {(time.time() - t0) / 60.0:.1f} min: "
        f"{final_meta['num_base_speakers']:,} base speakers ({final_meta['num_speakers']:,} speed-perturbed classes), "
        f"{final_meta['clean_utterances']:,} clean utts -> {final_meta['num_items']:,} items across "
        f"{final_meta['num_shards']} shards (free disk: {free_final:.1f} GB)."
    )
    return final_meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full multi-corpus float16 shards (`universal_full_v050`)."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get(
            "FLAXSPEAKER_DATA_ROOT", os.path.expanduser("~/Data/speaker_datasets")
        ),
    )
    parser.add_argument(
        "--shard_dir",
        type=str,
        default=os.path.expanduser(
            "~/Data/speaker_datasets/sharded_caches/universal_full_v050"
        ),
    )
    parser.add_argument("--num_workers", type=int, default=22)
    parser.add_argument("--min_free_gb", type=float, default=1850.0)
    parser.add_argument("--max_vox2_tars", type=int, default=160)
    parser.add_argument("--max_utts_per_fs_spk", type=int, default=40)
    parser.add_argument("--cpu_offset", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_full_corpora_shards(
        data_root=args.data_root,
        shard_dir=args.shard_dir,
        num_workers=args.num_workers,
        min_free_gb=args.min_free_gb,
        max_vox2_tars=args.max_vox2_tars,
        max_utts_per_fs_spk=args.max_utts_per_fs_spk,
        cpu_offset=args.cpu_offset,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
