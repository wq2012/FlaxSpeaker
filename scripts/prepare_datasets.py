#!/usr/bin/env python3
"""Prepare academic train/eval splits, trial lists, and feature caches.

Processes:
1. LibriSpeech (`train-clean-100` + `dev-clean` -> train; `test-clean` -> eval)
2. CN-Celeb2 (speaker-disjoint 80% train / 20% eval split + single & multi-utt trials)
3. VoxCeleb1 (official `veri_test2.txt` zero-shot benchmark + speaker-disjoint split)
"""

import argparse
import json
import os
import time

from multiprocessing import pool as mp_pool
import soundfile as sf

from flaxspeaker import configs
from flaxspeaker import dataset
from flaxspeaker import feature_extraction


DEFAULT_DATA_ROOT = "/usr/local/google/home/quanw/Data/speaker_datasets"


def _is_valid_audio(path: str) -> bool:
    try:
        data, _ = sf.read(path, dtype="float32")
        return data.size >= 1600
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return False


def filter_valid_utts(spk_to_utts: dataset.SpkToUtts) -> dataset.SpkToUtts:
    all_utts = [u for utts in spk_to_utts.values() for u in utts]
    with mp_pool.ThreadPool(16) as pool:
        flags = pool.map(_is_valid_audio, all_utts)
    valid_set = {u for u, ok in zip(all_utts, flags) if ok}
    cleaned: dataset.SpkToUtts = {}
    for spk, utts in spk_to_utts.items():
        v_utts = [u for u in utts if u in valid_set]
        if v_utts:
            cleaned[spk] = v_utts
    return cleaned


def prepare_all_datasets(
    data_root: str = DEFAULT_DATA_ROOT,
    max_utts_per_train_spk: int = 25,
    max_utts_per_eval_spk: int = 25,
    seed: int = 42,
) -> dict:
    splits_root = os.path.join(data_root, "splits")
    cache_root = os.path.join(data_root, "feature_cache")
    os.makedirs(splits_root, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)

    frontend_cfg = configs.FrontendConfig(
        feature_type=configs.FeatureType.LOG_MEL,
        sample_rate=16000,
        n_mels=80,
        apply_cmvn=True,
    )
    vad_cfg = configs.VadConfig(mode=configs.VadMode.ENERGY, energy_db_threshold=30.0)
    frontend = feature_extraction.AudioFrontend(frontend_cfg, vad_cfg)

    summary = {}

    # -------------------------------------------------------------------------
    # 1. LibriSpeech
    # -------------------------------------------------------------------------
    ls_root = os.path.join(data_root, "librispeech", "LibriSpeech")
    if os.path.isdir(ls_root):
        print("=== Preparing LibriSpeech splits ===")
        train_100 = dataset.get_librispeech_spk_to_utts(
            os.path.join(ls_root, "train-clean-100")
        )
        dev_clean = dataset.get_librispeech_spk_to_utts(
            os.path.join(ls_root, "dev-clean")
        )
        test_clean = dataset.get_librispeech_spk_to_utts(
            os.path.join(ls_root, "test-clean")
        )

        ls_train = {}
        ls_train.update(train_100)
        ls_train.update(dev_clean)
        ls_train = filter_valid_utts(ls_train)
        ls_eval = filter_valid_utts(test_clean)
        ls_train = dataset.filter_speakers_by_min_utts(
            ls_train, min_utts=6, max_utts_per_spk=max_utts_per_train_spk, seed=seed
        )
        ls_eval = dataset.filter_speakers_by_min_utts(
            ls_eval, min_utts=6, max_utts_per_spk=max_utts_per_eval_spk, seed=seed
        )

        ls_dir = os.path.join(splits_root, "librispeech")
        os.makedirs(ls_dir, exist_ok=True)
        train_csv = os.path.join(ls_dir, "train.csv")
        eval_csv = os.path.join(ls_dir, "eval.csv")
        trials_path = os.path.join(ls_dir, "trials.tsv")
        multi_trials_path = os.path.join(ls_dir, "multi_enroll_trials.tsv")

        dataset.save_spk_to_utts_csv(ls_train, train_csv)
        dataset.save_spk_to_utts_csv(ls_eval, eval_csv)
        ls_trials = dataset.generate_verification_trials(
            ls_eval, num_trials=3000, seed=seed
        )
        dataset.save_trials_file(ls_trials, trials_path)
        ls_multi = dataset.generate_multi_enroll_trials(
            ls_eval, num_trials=1000, num_enroll_utts=3, seed=seed
        )
        with open(multi_trials_path, "w", encoding="utf-8") as f:
            for lbl, e_list, t_u in ls_multi:
                f.write(f"{lbl}\t{','.join(e_list)}\t{t_u}\n")

        # Precompute feature cache
        t0 = time.time()
        cache = feature_extraction.FeatureCache(
            frontend, cache_dir=os.path.join(cache_root, "librispeech")
        )
        cache.precompute_dataset(ls_train, num_workers=16)
        cache.precompute_dataset(ls_eval, num_workers=16)
        print(f"LibriSpeech feature cache ready in {time.time() - t0:.1f}s")

        summary["librispeech"] = {
            "train_stats": dataset.compute_dataset_stats(ls_train),
            "eval_stats": dataset.compute_dataset_stats(ls_eval),
            "num_trials": len(ls_trials),
            "num_multi_enroll_trials": len(ls_multi),
            "train_csv": train_csv,
            "eval_csv": eval_csv,
            "trials_file": trials_path,
            "multi_trials_file": multi_trials_path,
        }

    # -------------------------------------------------------------------------
    # 2. CN-Celeb2
    # -------------------------------------------------------------------------
    cn_root = os.path.join(data_root, "cnceleb2")
    if os.path.isdir(cn_root):
        print("=== Preparing CN-Celeb2 splits ===")
        cn_all = filter_valid_utts(dataset.get_cnceleb_spk_to_utts(cn_root))
        cn_train, cn_eval = dataset.split_speakers_train_eval(
            cn_all, train_ratio=0.8, min_utts_per_spk=6, seed=seed
        )
        cn_train = dataset.filter_speakers_by_min_utts(
            cn_train, min_utts=6, max_utts_per_spk=max_utts_per_train_spk, seed=seed
        )
        cn_eval = dataset.filter_speakers_by_min_utts(
            cn_eval, min_utts=6, max_utts_per_spk=max_utts_per_eval_spk, seed=seed
        )

        cn_dir = os.path.join(splits_root, "cnceleb2")
        os.makedirs(cn_dir, exist_ok=True)
        train_csv = os.path.join(cn_dir, "train.csv")
        eval_csv = os.path.join(cn_dir, "eval.csv")
        trials_path = os.path.join(cn_dir, "trials.tsv")
        multi_trials_path = os.path.join(cn_dir, "multi_enroll_trials.tsv")

        dataset.save_spk_to_utts_csv(cn_train, train_csv)
        dataset.save_spk_to_utts_csv(cn_eval, eval_csv)
        cn_trials = dataset.generate_verification_trials(
            cn_eval, num_trials=3000, seed=seed
        )
        dataset.save_trials_file(cn_trials, trials_path)
        cn_multi = dataset.generate_multi_enroll_trials(
            cn_eval, num_trials=1000, num_enroll_utts=3, seed=seed
        )
        with open(multi_trials_path, "w", encoding="utf-8") as f:
            for lbl, e_list, t_u in cn_multi:
                f.write(f"{lbl}\t{','.join(e_list)}\t{t_u}\n")

        t0 = time.time()
        cache = feature_extraction.FeatureCache(
            frontend, cache_dir=os.path.join(cache_root, "cnceleb2")
        )
        cache.precompute_dataset(cn_train, num_workers=16)
        cache.precompute_dataset(cn_eval, num_workers=16)
        print(f"CN-Celeb2 feature cache ready in {time.time() - t0:.1f}s")

        summary["cnceleb2"] = {
            "train_stats": dataset.compute_dataset_stats(cn_train),
            "eval_stats": dataset.compute_dataset_stats(cn_eval),
            "num_trials": len(cn_trials),
            "num_multi_enroll_trials": len(cn_multi),
            "train_csv": train_csv,
            "eval_csv": eval_csv,
            "trials_file": trials_path,
            "multi_trials_file": multi_trials_path,
        }

    # -------------------------------------------------------------------------
    # 3. VoxCeleb1
    # -------------------------------------------------------------------------
    vox_root = os.path.join(data_root, "voxceleb1")
    if os.path.isdir(vox_root):
        print("=== Preparing VoxCeleb1 splits & official veri_test2 trials ===")
        vox_all = dataset.get_voxceleb_spk_to_utts(vox_root)
        vox_train, vox_eval = dataset.split_speakers_train_eval(
            vox_all, train_ratio=0.75, min_utts_per_spk=6, seed=seed
        )
        vox_train = dataset.filter_speakers_by_min_utts(
            vox_train, min_utts=6, max_utts_per_spk=max_utts_per_train_spk, seed=seed
        )
        vox_eval = dataset.filter_speakers_by_min_utts(
            vox_eval, min_utts=6, max_utts_per_spk=max_utts_per_eval_spk, seed=seed
        )

        vox_dir = os.path.join(splits_root, "voxceleb1")
        os.makedirs(vox_dir, exist_ok=True)
        train_csv = os.path.join(vox_dir, "train.csv")
        eval_csv = os.path.join(vox_dir, "eval.csv")
        trials_path = os.path.join(vox_dir, "trials.tsv")
        official_trials_path = os.path.join(vox_dir, "vox1_o_official_4000.tsv")

        dataset.save_spk_to_utts_csv(vox_train, train_csv)
        dataset.save_spk_to_utts_csv(vox_eval, eval_csv)
        vox_trials = dataset.generate_verification_trials(
            vox_eval, num_trials=2000, seed=seed
        )
        dataset.save_trials_file(vox_trials, trials_path)

        veri_txt = os.path.join(vox_root, "veri_test2.txt")
        official_trials = []
        if os.path.exists(veri_txt):
            official_trials = dataset.load_voxceleb_veri_test(
                veri_txt, vox_root, max_trials=4000, seed=seed
            )
            dataset.save_trials_file(official_trials, official_trials_path)

        # Precompute feature cache for all utterances referenced in train, eval, and official trials
        t0 = time.time()
        cache = feature_extraction.FeatureCache(
            frontend, cache_dir=os.path.join(cache_root, "voxceleb1")
        )
        cache.precompute_dataset(vox_train, num_workers=16)
        cache.precompute_dataset(vox_eval, num_workers=16)
        official_utts = {"official": sorted({u for _, u1, u2 in official_trials for u in (u1, u2)})}
        cache.precompute_dataset(official_utts, num_workers=16)
        print(f"VoxCeleb1 feature cache ready in {time.time() - t0:.1f}s")

        summary["voxceleb1"] = {
            "total_stats": dataset.compute_dataset_stats(vox_all),
            "train_stats": dataset.compute_dataset_stats(vox_train),
            "eval_stats": dataset.compute_dataset_stats(vox_eval),
            "num_split_trials": len(vox_trials),
            "num_official_veri_test2_trials": len(official_trials),
            "train_csv": train_csv,
            "eval_csv": eval_csv,
            "trials_file": trials_path,
            "official_trials_file": official_trials_path,
        }

    summary_path = os.path.join(splits_root, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved dataset summary to {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    prepare_all_datasets(args.data_root)
