#!/usr/bin/env python3
"""Polite, resumable downloader and extractor for full open-source speaker corpora.

Downloads into `~/Data/speaker_datasets/full_corpora` in priority order:
1. MUSAN (`openslr.org/resources/17/musan.tar.gz`, ~11.1 GB) — needed by Stage 2 & Stage 4
2. RIRS_NOISES (`openslr.org/resources/28/rirs_noises.zip`, ~1.3 GB) — needed by Stage 2 & Stage 4
3. LibriSpeech `train-clean-360` + `train-other-500` (`openslr.org/resources/12/`, ~53.6 GB)
4. CN-Celeb1 v2 (`openslr.org/resources/82/cn-celeb_v2.tar.gz`, ~22.3 GB, includes official `CN-Celeb-E`)
5. CN-Celeb2 v2 (`openslr.org/resources/82/cn-celeb2_v2.tar.gza{a,b,c}`, ~77.6 GB)
6. VoxCeleb1-dev (`harsha-jk/vox1-dev-test-combined` on HF, ~33.6 GB, strictly excluding the
   40 `VoxCeleb1-O` test speakers `id10270`..`id10309`)
7. VoxCeleb2-dev (`gaunernst/voxceleb2-dev-wds` on HF, 779 WebDataset `.tar` shards, ~82.1 GB,
   kept directly as compact `.tar` shards so 1.09M M4A files never waste ext4 inodes)

Resource guardrails:
- Runs under `os.nice(15)` and at most 4 download workers.
- Deletes temporary `.tar.gz` / `.zip` archives immediately after extraction.
- Refuses to download any step if free disk on `/` drops below `min_free_gb` (default 1800 GB).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

from huggingface_hub import hf_hub_download, snapshot_download


VOX1_O_TEST_SPEAKERS = {f"id{spk}" for spk in range(10270, 10310)}


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts} UTC] {msg}", flush=True)


def _check_free_disk(path: str, min_free_gb: float) -> float:
    os.makedirs(path, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Safety abort: free disk is {free_gb:.1f} GB (< required {min_free_gb:.1f} GB reserve)."
        )
    return free_gb


def _wget_resumable(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cmd = [
        "nice",
        "-n",
        "15",
        "wget",
        "-c",
        "--tries=10",
        "--waitretry=5",
        "--progress=dot:giga",
        "-O",
        dest_path,
        url,
    ]
    _log(f"Downloading {url} -> {dest_path}")
    subprocess.run(cmd, check=True)


def step_musan(root: str, min_free_gb: float) -> dict:
    marker = os.path.join(root, ".done_musan")
    out_dir = os.path.join(root, "musan")
    if os.path.exists(marker) and os.path.isdir(out_dir):
        _log("MUSAN already ready.")
        return {"status": "cached", "path": out_dir}
    _check_free_disk(root, min_free_gb)
    archive = os.path.join(root, "_tmp_musan.tar.gz")
    _wget_resumable("https://www.openslr.org/resources/17/musan.tar.gz", archive)
    _log("Extracting MUSAN...")
    subprocess.run(["nice", "-n", "15", "tar", "-xzf", archive, "-C", root], check=True)
    os.remove(archive)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    _log("MUSAN extraction complete; archive removed.")
    return {"status": "downloaded", "path": out_dir}


def step_rirs(root: str, min_free_gb: float) -> dict:
    marker = os.path.join(root, ".done_rirs")
    out_dir = os.path.join(root, "RIRS_NOISES")
    if os.path.exists(marker) and os.path.isdir(out_dir):
        _log("RIRS_NOISES already ready.")
        return {"status": "cached", "path": out_dir}
    _check_free_disk(root, min_free_gb)
    archive = os.path.join(root, "_tmp_rirs_noises.zip")
    _wget_resumable("https://www.openslr.org/resources/28/rirs_noises.zip", archive)
    _log("Extracting RIRS_NOISES...")
    subprocess.run(["nice", "-n", "15", "unzip", "-q", "-o", archive, "-d", root], check=True)
    os.remove(archive)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    _log("RIRS_NOISES extraction complete; archive removed.")
    return {"status": "downloaded", "path": out_dir}


def step_librispeech_extra(data_root: str, root: str, min_free_gb: float) -> dict:
    ls_parent = os.path.join(data_root, "librispeech")
    os.makedirs(ls_parent, exist_ok=True)
    results = {}
    for subset in ("train-clean-360", "train-other-500"):
        marker = os.path.join(root, f".done_ls_{subset}")
        target_dir = os.path.join(ls_parent, "LibriSpeech", subset)
        if os.path.exists(marker) and os.path.isdir(target_dir):
            _log(f"LibriSpeech {subset} already ready.")
            results[subset] = "cached"
            continue
        _check_free_disk(root, min_free_gb)
        archive = os.path.join(root, f"_tmp_{subset}.tar.gz")
        _wget_resumable(
            f"https://www.openslr.org/resources/12/{subset}.tar.gz", archive
        )
        _log(f"Extracting LibriSpeech {subset} into {ls_parent}...")
        subprocess.run(
            ["nice", "-n", "15", "tar", "-xzf", archive, "-C", ls_parent], check=True
        )
        os.remove(archive)
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
        _log(f"LibriSpeech {subset} complete; archive removed.")
        results[subset] = "downloaded"
    return results


def step_cnceleb1(root: str, min_free_gb: float) -> dict:
    marker = os.path.join(root, ".done_cnceleb1")
    out_dir = os.path.join(root, "CN-Celeb_flac")
    if os.path.exists(marker) and os.path.isdir(out_dir):
        _log("CN-Celeb1 already ready.")
        return {"status": "cached", "path": out_dir}
    _check_free_disk(root, min_free_gb)
    archive = os.path.join(root, "_tmp_cn-celeb_v2.tar.gz")
    _wget_resumable("https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz", archive)
    _log("Extracting CN-Celeb1 (including official CN-Celeb-E eval set)...")
    subprocess.run(["nice", "-n", "15", "tar", "-xzf", archive, "-C", root], check=True)
    os.remove(archive)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    _log("CN-Celeb1 complete; archive removed.")
    return {"status": "downloaded", "path": out_dir}


def step_cnceleb2_full(root: str, min_free_gb: float) -> dict:
    marker = os.path.join(root, ".done_cnceleb2_full")
    out_dir = os.path.join(root, "CN-Celeb2_flac")
    if os.path.exists(marker) and os.path.isdir(out_dir):
        _log("CN-Celeb2 full already ready.")
        return {"status": "cached", "path": out_dir}
    _check_free_disk(root, min_free_gb)
    parts = []
    for suffix in ("aa", "ab", "ac"):
        p = os.path.join(root, f"_tmp_cn-celeb2_v2.tar.gz{suffix}")
        _wget_resumable(
            f"https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gz{suffix}", p
        )
        parts.append(p)
    _log("Streaming multi-part CN-Celeb2 archive into tar -xzf - ...")
    cat_cmd = (
        f"nice -n 15 cat {' '.join(parts)} | nice -n 15 tar -xzf - -C {root}"
    )
    subprocess.run(["bash", "-c", cat_cmd], check=True)
    for p in parts:
        if os.path.exists(p):
            os.remove(p)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    _log("CN-Celeb2 full complete; split archives removed.")
    return {"status": "downloaded", "path": out_dir}


def step_voxceleb1_dev(root: str, min_free_gb: float) -> dict:
    marker = os.path.join(root, ".done_voxceleb1_dev")
    out_dir = os.path.join(root, "voxceleb1_dev_wav")
    if os.path.exists(marker) and os.path.isdir(out_dir):
        _log("VoxCeleb1-dev already ready.")
        return {"status": "cached", "path": out_dir}
    _check_free_disk(root, min_free_gb)
    _log("Downloading harsha-jk/vox1-dev-test-combined (vox1_combined_wav.zip)...")
    zip_path = hf_hub_download(
        repo_id="harsha-jk/vox1-dev-test-combined",
        filename="vox1_combined_wav.zip",
        repo_type="dataset",
        local_dir=os.path.join(root, "_tmp_vox1_dl"),
    )
    os.makedirs(out_dir, exist_ok=True)
    kept_utts = 0
    skipped_test_utts = 0
    seen_spks: set[str] = set()
    _log("Extracting VoxCeleb1-dev while strictly excluding all 40 VoxCeleb1-O test speakers...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.endswith(".wav"):
                continue
            parts = member.filename.replace("\\", "/").split("/")
            spk_id = next((p for p in parts if p.startswith("id1")), None)
            if spk_id is None:
                continue
            if spk_id in VOX1_O_TEST_SPEAKERS:
                skipped_test_utts += 1
                continue
            idx = parts.index(spk_id)
            rel_path = os.path.join(*parts[idx:])
            dest = os.path.join(out_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            seen_spks.add(spk_id)
            kept_utts += 1
    shutil.rmtree(os.path.join(root, "_tmp_vox1_dl"), ignore_errors=True)
    meta = {
        "num_dev_speakers": len(seen_spks),
        "num_dev_utterances": kept_utts,
        "excluded_vox1_o_test_utterances": skipped_test_utts,
        "overlap_with_vox1_o": len(seen_spks & VOX1_O_TEST_SPEAKERS),
    }
    if meta["overlap_with_vox1_o"] != 0:
        raise AssertionError("Leakage check failed for VoxCeleb1-dev!")
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _log(f"VoxCeleb1-dev ready: {meta}")
    return {"status": "downloaded", "path": out_dir, **meta}


def step_voxceleb2_dev_wds(root: str, min_free_gb: float) -> dict:
    marker = os.path.join(root, ".done_voxceleb2_dev_wds")
    out_dir = os.path.join(root, "voxceleb2_dev_wds")
    if os.path.exists(marker) and os.path.isdir(out_dir):
        _log("VoxCeleb2-dev WebDataset shards already ready.")
        return {"status": "cached", "path": out_dir}
    _check_free_disk(root, min_free_gb)
    _log("Downloading gaunernst/voxceleb2-dev-wds (779 WebDataset shards, max_workers=4)...")
    snapshot_download(
        repo_id="gaunernst/voxceleb2-dev-wds",
        repo_type="dataset",
        local_dir=out_dir,
        max_workers=4,
    )
    tar_files = sorted(
        f for f in os.listdir(out_dir) if f.startswith("voxceleb2-dev-") and f.endswith(".tar")
    )
    meta = {"num_tar_shards": len(tar_files), "path": out_dir}
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _log(f"VoxCeleb2-dev WebDataset ready: {len(tar_files)} shards.")
    return {"status": "downloaded", **meta}


def main() -> None:
    try:
        os.nice(15)
    except OSError:
        pass
    parser = argparse.ArgumentParser(description="Polite full-corpora downloader.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.expanduser("~/Data/speaker_datasets"),
    )
    parser.add_argument(
        "--min_free_gb",
        type=float,
        default=1800.0,
        help="Minimum free GB that must remain untouched on `/`.",
    )
    parser.add_argument(
        "--only_augmentation",
        action="store_true",
        help="Only download MUSAN + RIRS_NOISES.",
    )
    args = parser.parse_args()

    full_root = os.path.join(args.data_root, "full_corpora")
    os.makedirs(full_root, exist_ok=True)
    free_start = _check_free_disk(full_root, args.min_free_gb)
    _log(f"Starting polite full-corpora download. Initial free disk: {free_start:.1f} GB.")

    report: dict[str, dict] = {}
    report["musan"] = step_musan(full_root, args.min_free_gb)
    report["rirs_noises"] = step_rirs(full_root, args.min_free_gb)
    if not args.only_augmentation:
        report["librispeech_extra"] = step_librispeech_extra(
            args.data_root, full_root, args.min_free_gb
        )
        report["cnceleb1"] = step_cnceleb1(full_root, args.min_free_gb)
        report["cnceleb2_full"] = step_cnceleb2_full(full_root, args.min_free_gb)
        report["voxceleb1_dev"] = step_voxceleb1_dev(full_root, args.min_free_gb)
        report["voxceleb2_dev_wds"] = step_voxceleb2_dev_wds(
            full_root, args.min_free_gb
        )

    free_end = shutil.disk_usage(full_root).free / (1024**3)
    report["free_disk_gb_end"] = round(free_end, 2)
    summary_path = os.path.join(full_root, "download_manifest.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _log(f"All requested downloads finished cleanly! Remaining free disk: {free_end:.1f} GB.")


if __name__ == "__main__":
    sys.exit(main())
