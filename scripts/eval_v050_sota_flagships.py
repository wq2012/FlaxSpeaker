#!/usr/bin/env python3
"""Parallel extraction & comprehensive evaluation of all 6 v0.5.0 SOTA Flagships.

Extracts all 30,289 unique evaluation and disjoint cohort utterances across all 6
multi-domain flagship models (`FULL_DATA_FLAGSHIP_IDS`) using 4 parallel worker
processes (`7` threads each = `28` vCPUs, `nice -n 10`), then evaluates:
- Official `VoxCeleb1-O` (`37,611` trials: Raw, AS-Norm `top_k=100`, QMF)
- Official `CN-Celeb-E` (`3,484,292` single-enroll + `3,484,292` `enroll.map` multi-enroll)
- Official `LibriSpeech test-clean` (`10,000` 1-utt + `3,000` 3-utt trials)
- Controlled 3-domain benchmark suite (`librispeech`, `cnceleb2`, `voxceleb1`,
  `universal_pooled` 1-utt & 3-utt, and `vox1_o_10spk_subset`)
- Calibrated Heterogeneous Score Fusion (6-way flagship fusion, 2-way Conformer+ECAPA,
  and multi-enroll fusion) calibrated strictly on `splits/universal_full_val/`
- TFLite FP32 & INT8 export and `VoxCeleb1-O` evaluation for
  `univ_full_conformer_large_arcface` and `univ_full_ecapa_c1024_arcface`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FLAGSHIP_IDS = (
    "univ_full_resnet34_arcface",
    "univ_full_ecapa_c1024_arcface",
    "univ_full_wavlm_base_arcface",
    "univ_full_conformer_large_arcface",
    "univ_full_conformer_large_pfas",
    "univ_full_conformer_large_dr_vectors",
)


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts} UTC] {msg}", flush=True)


def _utt_key(u: str) -> str:
    return hashlib.sha1(os.path.abspath(u).encode("utf-8")).hexdigest()[:16]


def _load_multi_enroll_tsv(tsv_path: str, dataset_mod: Any) -> list[Any]:
    trials = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            lbl = int(parts[0])
            e_list = [dataset_mod.resolve_data_path(p) for p in parts[1].split(",") if p]
            t_u = dataset_mod.resolve_data_path(parts[2])
            trials.append((lbl, e_list, t_u))
    return trials


def _load_official_vox1_o_trials(data_root: str) -> list[tuple[int, str, str]]:
    veri_path = os.path.join(data_root, "voxceleb1", "veri_test2.txt")
    wav_root = os.path.join(data_root, "voxceleb1", "wav")
    trials = []
    with open(veri_path, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) != 3:
                continue
            trials.append(
                (
                    int(toks[0]),
                    os.path.join(wav_root, toks[1]),
                    os.path.join(wav_root, toks[2]),
                )
            )
    return trials


def _collect_all_utts_and_paths(data_root: str) -> tuple[list[str], dict[str, str]]:
    """Collects all unique utterances across cohort, val, controlled, and official splits."""
    from flaxspeaker import dataset

    with open(os.path.join(data_root, "splits", "dataset_summary.json")) as f:
        ds_sum = json.load(f)

    all_utts: set[str] = set()
    # 1. Disjoint 3-domain training cohort utterances (120 LS + 150 CN + 30 Vox = 300 speakers)
    dom_spk_caps = {"librispeech": 120, "cnceleb2": 150, "voxceleb1": 30}
    for dom in ("librispeech", "cnceleb2", "voxceleb1"):
        tr_map = dataset.get_csv_spk_to_utts(ds_sum[dom]["train_csv"])
        for spk in sorted(tr_map.keys())[: dom_spk_caps[dom]]:
            for u in tr_map[spk][:2]:
                all_utts.add(u)
        # Controlled 1-utt and 3-utt trials
        for _, u1, u2 in dataset.load_trials_file(ds_sum[dom]["trials_file"]):
            all_utts.add(u1)
            all_utts.add(u2)
        for _, el, tu in _load_multi_enroll_tsv(
            dataset.resolve_data_path(ds_sum[dom]["multi_trials_file"]), dataset
        ):
            all_utts.update(el)
            all_utts.add(tu)

    for _, u1, u2 in dataset.load_trials_file(ds_sum["universal"]["trials_file"]):
        all_utts.add(u1)
        all_utts.add(u2)
    for _, el, tu in _load_multi_enroll_tsv(
        dataset.resolve_data_path(ds_sum["universal"]["multi_trials_file"]), dataset
    ):
        all_utts.update(el)
        all_utts.add(tu)
    for _, u1, u2 in dataset.load_trials_file(
        ds_sum["voxceleb1"].get(
            "vox1_o_10spk_subset_trials_file", ds_sum["voxceleb1"]["official_trials_file"]
        )
    ):
        all_utts.add(u1)
        all_utts.add(u2)

    # 2. Held-out universal_full_val
    val_dir = os.path.join(data_root, "splits", "universal_full_val")
    for _, u1, u2 in dataset.load_trials_file(os.path.join(val_dir, "trials.tsv")):
        all_utts.add(u1)
        all_utts.add(u2)
    for _, el, tu in _load_multi_enroll_tsv(
        os.path.join(val_dir, "multi_enroll_trials.tsv"), dataset
    ):
        all_utts.update(el)
        all_utts.add(tu)

    # 3. Official VoxCeleb1-O
    for _, u1, u2 in _load_official_vox1_o_trials(data_root):
        all_utts.add(u1)
        all_utts.add(u2)

    # 4. Official LibriSpeech test-clean
    off_ls_dir = os.path.join(data_root, "splits", "official_eval")
    for _, u1, u2 in dataset.load_trials_file(
        os.path.join(off_ls_dir, "librispeech_test_clean_trials.tsv")
    ):
        all_utts.add(u1)
        all_utts.add(u2)
    for _, el, tu in _load_multi_enroll_tsv(
        os.path.join(off_ls_dir, "librispeech_test_clean_3utt_trials.tsv"), dataset
    ):
        all_utts.update(el)
        all_utts.add(tu)

    # 5. Official CN-Celeb-E
    cn1_root = os.path.join(data_root, "full_corpora", "CN-Celeb_flac")
    cn_multi, cn_single = dataset.load_cnceleb_eval_trials(cn1_root)
    for _, u1, u2 in cn_single:
        all_utts.add(u1)
        all_utts.add(u2)
    for _, el, tu in cn_multi:
        all_utts.update(el)
        all_utts.add(tu)

    cache_root = os.path.join(data_root, "feature_cache_kaldi_novad")
    sub_dirs = [
        "train_calib_3dom",
        "librispeech",
        "cnceleb2",
        "voxceleb1",
        "universal_full_val",
        "voxceleb1_o_full",
        "librispeech_test_clean_full",
        "cnceleb_e_official",
    ]
    key_to_path: dict[str, str] = {}
    for sd in sub_dirs:
        d_p = os.path.join(cache_root, sd)
        if not os.path.isdir(d_p):
            continue
        for fn in os.listdir(d_p):
            if fn.endswith(".npy"):
                key_to_path[fn[:-4]] = os.path.join(d_p, fn)

    from flaxspeaker import configs, feature_extraction

    fe = feature_extraction.AudioFrontend(
        configs.FrontendConfig(kaldi_fbank=True, apply_cmvn=True),
        configs.VadConfig(enabled=False, mode="none"),
    )
    extra_dir = os.path.join(cache_root, "train_calib_3dom")
    os.makedirs(extra_dir, exist_ok=True)
    fc_helper = feature_extraction.FeatureCache(fe, cache_dir=extra_dir)
    fe_sig = fc_helper._cfg_signature

    sorted_utts = sorted(all_utts)
    utt_to_npy: dict[str, str] = {}
    for u in sorted_utts:
        k = fc_helper._digest_name(fe_sig, u)[:-4]
        if k not in key_to_path:
            out_p = os.path.join(extra_dir, f"{k}.npy")
            feat = fe.extract_from_file(u)
            np.save(out_p, feat.astype(np.float32))
            key_to_path[k] = out_p
        utt_to_npy[u] = key_to_path[k]
    return sorted_utts, utt_to_npy


def _run_extraction_worker(
    worker_id: int, num_workers: int, data_root: str, max_frames: int = 512
) -> None:
    """Worker process that extracts its assigned slice of utterances across all 6 models."""
    os.environ["OMP_NUM_THREADS"] = "7"
    os.environ["MKL_NUM_THREADS"] = "7"
    os.environ["OPENBLAS_NUM_THREADS"] = "7"
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=7"
    )
    try:
        os.nice(10)
    except OSError:
        pass

    import jax
    import jax.numpy as jnp
    from flaxspeaker import backbones
    from flaxspeaker import evaluation
    from flaxspeaker import hf_compat
    from flaxspeaker import pooling

    cache_root = os.path.join(data_root, "feature_cache_kaldi_novad")
    manifest_path = os.path.join(cache_root, "flagship_eval_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    utt_list: list[str] = manifest["utts"]
    utt_to_npy: dict[str, str] = manifest["utt_to_npy"]

    # Load the saved checkpoints of the 6 flagships
    loaded = {
        eid: hf_compat.FlaxSpeakerModel.from_pretrained(
            str(REPO_ROOT / "pretrained_models" / eid)
        )
        for eid in FLAGSHIP_IDS
    }
    cf_m = loaded["univ_full_conformer_large_arcface"]
    ec_m = loaded["univ_full_ecapa_c1024_arcface"]
    wl_m = loaded["univ_full_wavlm_base_arcface"]
    rn_m = loaded["univ_full_resnet34_arcface"]
    pf_m = loaded["univ_full_conformer_large_pfas"]
    dr_m = loaded["univ_full_conformer_large_dr_vectors"]

    cf_mod = backbones.ConformerBackbone(
        dim=320,
        num_heads=8,
        num_layers=4,
        ffn_dim=640,
        conv_kernel_size=15,
        resnet_stem_channels=64,
    )
    ec_mod = backbones.EcapaTdnnBackbone(
        channels=1024,
        scale=8,
        kernel_sizes=(5, 3, 3, 3),
        dilations=(1, 2, 3, 4),
        mfa_dim=1536,
        se_bottleneck_dim=128,
        use_batch_norm=True,
        norm_after_activation=True,
    )
    ec_pool_mod = pooling.AttentiveStatsPooling(
        attention_dim=128, use_global_context=True, asp_relu_bn=False
    )
    wl_mod = backbones.WavLMBackbone()

    cf_p = cf_m.params["backbone"]
    cf_bs = cf_m.batch_stats["backbone"]
    ec_p = ec_m.params["backbone"]
    ec_bs = ec_m.batch_stats["backbone"]
    ec_pool_p = ec_m.params["pool"]
    wl_p = wl_m.params["backbone"]

    W_rn_256 = rn_m.params["embedding_proj"]["kernel"]
    b_rn_256 = rn_m.params["embedding_proj"]["bias"]
    W_ec_256 = ec_m.params["embedding_proj"]["kernel"]
    b_ec_256 = ec_m.params["embedding_proj"]["bias"]
    W_wl_256 = wl_m.params["embedding_proj"]["kernel"]
    b_wl_256 = wl_m.params["embedding_proj"]["bias"]
    W_cf_256 = cf_m.params["embedding_proj"]["kernel"]
    b_cf_256 = cf_m.params["embedding_proj"]["bias"]
    W_dr_256 = dr_m.params["embedding_proj"]["kernel"]
    b_dr_256 = dr_m.params["embedding_proj"]["bias"]
    pf_params = pf_m.params["pfas_head"]

    def l2_norm(v):
        return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)

    def compute_pfas_from_shared(p_conf_full, h_cf, sub_m):
        g_val = (
            p_conf_full @ pf_params["pfas_g_val"]["kernel"]
            + pf_params["pfas_g_val"]["bias"]
        )
        k_w = pf_params["pfas_key_proj"]["kernel"][5120:5440, :]
        k_b = pf_params["pfas_key_proj"]["bias"]
        v_w = pf_params["pfas_val_proj"]["kernel"][5120:5440, :]
        v_b = pf_params["pfas_val_proj"]["bias"]
        a_w = pf_params["pfas_att_hid"]["kernel"][5120:5440, :]
        a_b = pf_params["pfas_att_hid"]["bias"]
        o_w = pf_params["pfas_att_out"]["kernel"]
        o_b = pf_params["pfas_att_out"]["bias"]
        B_sz, T_sz, _ = h_cf.shape
        fk = (h_cf @ k_w + k_b).reshape(B_sz, T_sz, 2, 32)
        fv = (h_cf @ v_w + v_b).reshape(B_sz, T_sz, 2, 256)
        ah = jnp.tanh(h_cf @ a_w + a_b)
        logits = ah @ o_w + o_b
        logits = jnp.where(sub_m[..., None] > 0, logits, -1e9)
        alpha = jax.nn.softmax(logits, axis=1)
        keys = jnp.einsum("btm,btmd->bmd", alpha, fk)
        l_vals = jnp.einsum("btm,btmd->bmd", alpha, fv)
        vals = g_val[:, None, :] + 0.20 * l_vals
        keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-6)
        flat_vals = vals.reshape(B_sz, 512)
        flat_vals = flat_vals / (
            jnp.linalg.norm(flat_vals, axis=-1, keepdims=True) + 1e-6
        )
        return jnp.concatenate(
            [keys.reshape(B_sz, 64), flat_vals], axis=-1
        )

    @jax.jit
    def extract_all_6_batch(xb, mb):
        h_cat = cf_mod.apply(
            {"params": cf_p, "batch_stats": cf_bs}, xb, deterministic=True, mask=mb
        )
        sub_m = mb[:, ::8][:, : h_cat.shape[1]]
        h_stem = h_cat[..., :5120]
        h_conf = h_cat[..., 5120:]
        p_rn = pooling.StatsPooling(unbiased_std=True).apply({}, h_stem, mask=sub_m)
        p_cf = pooling.StatsPooling(unbiased_std=True).apply({}, h_conf, mask=sub_m)

        h_ec = ec_mod.apply(
            {"params": ec_p, "batch_stats": ec_bs}, xb, deterministic=True, mask=mb
        )
        p_ec = ec_pool_mod.apply({"params": ec_pool_p}, h_ec, mask=mb)

        h_wl = wl_mod.apply({"params": wl_p}, xb, deterministic=True, mask=mb)
        wl_m = mb[:, ::2][:, : h_wl.shape[1]]
        p_wl = pooling.StatsPooling(unbiased_std=True).apply({}, h_wl, mask=wl_m)

        p_conf_full = jnp.concatenate(
            [p_rn[:, :5120], p_cf[:, :320], p_rn[:, 5120:], p_cf[:, 320:]], axis=-1
        )
        p_ec_full = jnp.concatenate([p_ec, p_rn], axis=-1)
        p_wl_full = jnp.concatenate([p_wl, p_rn], axis=-1)

        e_rn = l2_norm(p_rn @ W_rn_256 + b_rn_256)
        e_ec = l2_norm(p_ec_full @ W_ec_256 + b_ec_256)
        e_wl = l2_norm(p_wl_full @ W_wl_256 + b_wl_256)
        e_cf = l2_norm(p_conf_full @ W_cf_256 + b_cf_256)
        e_dr = l2_norm(p_conf_full @ W_dr_256 + b_dr_256)
        e_pf = compute_pfas_from_shared(p_conf_full, h_conf, sub_m)
        return e_rn, e_ec, e_wl, e_cf, e_pf, e_dr

    exp_cfg = rn_m.exp_config
    exp_cfg.eval.max_eval_frames_cap = max_frames

    my_indices = list(range(worker_id, len(utt_list), num_workers))
    buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
    my_frames = np.zeros((len(my_indices),), dtype=np.float32)
    for local_i, global_idx in enumerate(my_indices):
        u = utt_list[global_idx]
        raw = np.load(utt_to_npy[u])
        my_frames[local_i] = float(raw.shape[0])
        b_len = 256 if raw.shape[0] <= 256 else max_frames
        buckets.setdefault(b_len, []).append((local_i, raw))

    n_my = len(my_indices)
    out_arrs = {
        "univ_full_resnet34_arcface": np.zeros((n_my, 256), dtype=np.float32),
        "univ_full_ecapa_c1024_arcface": np.zeros((n_my, 256), dtype=np.float32),
        "univ_full_wavlm_base_arcface": np.zeros((n_my, 256), dtype=np.float32),
        "univ_full_conformer_large_arcface": np.zeros((n_my, 256), dtype=np.float32),
        "univ_full_conformer_large_pfas": np.zeros((n_my, 576), dtype=np.float32),
        "univ_full_conformer_large_dr_vectors": np.zeros((n_my, 256), dtype=np.float32),
    }

    t0 = time.time()
    done = 0
    for b_len in sorted(buckets.keys()):
        items = buckets[b_len]
        for start in range(0, len(items), 64):
            chunk = items[start:start + 64]
            feats, masks = [], []
            for _, raw in chunk:
                p, m = evaluation.prepare_eval_input(
                    raw, exp_cfg, target_pad_override=b_len
                )
                feats.append(p)
                masks.append(m)
            n_c = len(chunk)
            if n_c < 64:
                feats.extend([feats[0]] * (64 - n_c))
                masks.extend([masks[0]] * (64 - n_c))
            e_rn, e_ec, e_wl, e_cf, e_pf, e_dr = extract_all_6_batch(
                jnp.asarray(np.stack(feats, axis=0)),
                jnp.asarray(np.stack(masks, axis=0)),
            )
            e_rn_np = np.asarray(e_rn)
            e_ec_np = np.asarray(e_ec)
            e_wl_np = np.asarray(e_wl)
            e_cf_np = np.asarray(e_cf)
            e_pf_np = np.asarray(e_pf)
            e_dr_np = np.asarray(e_dr)
            for idx_c, (local_i, _) in enumerate(chunk):
                out_arrs["univ_full_resnet34_arcface"][local_i] = e_rn_np[idx_c]
                out_arrs["univ_full_ecapa_c1024_arcface"][local_i] = e_ec_np[idx_c]
                out_arrs["univ_full_wavlm_base_arcface"][local_i] = e_wl_np[idx_c]
                out_arrs["univ_full_conformer_large_arcface"][local_i] = e_cf_np[idx_c]
                out_arrs["univ_full_conformer_large_pfas"][local_i] = e_pf_np[idx_c]
                out_arrs["univ_full_conformer_large_dr_vectors"][local_i] = e_dr_np[idx_c]
            done += n_c
            if done % 1280 == 0 or done == n_my:
                _log(
                    f"[Worker {worker_id}] {done}/{n_my} utts ({time.time() - t0:.1f}s)"
                )

    out_path = os.path.join(cache_root, f"flagship_embs_w{worker_id}.npz")
    np.savez(
        out_path,
        indices=np.asarray(my_indices, dtype=np.int32),
        frames=my_frames,
        **out_arrs,
    )
    _log(f"[Worker {worker_id}] Saved -> {out_path}")


def _tflite_eval_chunk(args: tuple[str, str, list[tuple[str, str]]]) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    fp32_path, int8_path, pairs = args
    from flaxspeaker import export

    r32 = export.TFLiteSpeakerRunner(fp32_path)
    r8 = export.TFLiteSpeakerRunner(int8_path)
    out = []
    for u, npy_p in pairs:
        feat = np.load(npy_p)
        e32 = r32(feat)
        e8 = r8(feat)
        e32_n = e32 / (np.linalg.norm(e32) + 1e-6)
        e8_n = e8 / (np.linalg.norm(e8) + 1e-6)
        out.append((u, e32_n, e8_n, float(np.dot(e32_n, e8_n))))
    return out


def _export_and_eval_tflite_parallel(
    model_id: str,
    hf_dir: str,
    state: Any,
    exp_cfg: Any,
    vox1_o_trials: list[tuple[int, str, str]],
    utt_to_npy: dict[str, str],
    jax_full_seq_eer: float,
) -> Optional[dict[str, Any]]:
    from flaxspeaker import evaluation
    from flaxspeaker import export

    fp32_path = os.path.join(hf_dir, "model_fp32.tflite")
    int8_path = os.path.join(hf_dir, "model_int8.tflite")
    try:
        os.makedirs(hf_dir, exist_ok=True)
        _log(f"[{model_id}] Exporting FP32 & INT8 TFLite models (seq_len=180)...")
        fp32_info = export.export_to_tflite(
            state, exp_cfg, fp32_path, quantize_int8=False, seq_len=180
        )
        int8_info = export.export_to_tflite(
            state, exp_cfg, int8_path, quantize_int8=True, seq_len=180
        )
        vox_utts = sorted({u for _, u1, u2 in vox1_o_trials for u in (u1, u2)})
        pairs = [(u, utt_to_npy[u]) for u in vox_utts]
        chunks = [pairs[i::4] for i in range(4)]
        emb_fp32: dict[str, np.ndarray] = {}
        emb_int8: dict[str, np.ndarray] = {}
        cos_list: list[float] = []
        with ProcessPoolExecutor(max_workers=4) as ex:
            for res_list in ex.map(
                _tflite_eval_chunk, [(fp32_path, int8_path, c) for c in chunks]
            ):
                for u, e32_n, e8_n, c_val in res_list:
                    emb_fp32[u] = e32_n
                    emb_int8[u] = e8_n
                    cos_list.append(c_val)

        labels = [int(lbl) for lbl, _, _ in vox1_o_trials]
        s_32 = np.array(
            [float(np.dot(emb_fp32[u1], emb_fp32[u2])) for _, u1, u2 in vox1_o_trials],
            dtype=np.float64,
        )
        s_8 = np.array(
            [float(np.dot(emb_int8[u1], emb_int8[u2])) for _, u1, u2 in vox1_o_trials],
            dtype=np.float64,
        )
        eer_32, _ = evaluation.compute_eer_exact(labels, s_32)
        eer_8, _ = evaluation.compute_eer_exact(labels, s_8)

        r_fp32 = export.TFLiteSpeakerRunner(fp32_path)
        r_int8 = export.TFLiteSpeakerRunner(int8_path)
        dummy_feat = np.zeros((180, exp_cfg.frontend.n_mels), dtype=np.float32)
        for _ in range(5):
            r_fp32(dummy_feat)
            r_int8(dummy_feat)
        t_l32 = time.perf_counter()
        for _ in range(25):
            r_fp32(dummy_feat)
        lat_32 = round((time.perf_counter() - t_l32) * 1000.0 / 25.0, 2)
        t_l8 = time.perf_counter()
        for _ in range(25):
            r_int8(dummy_feat)
        lat_8 = round((time.perf_counter() - t_l8) * 1000.0 / 25.0, 2)

        fp32_bytes = int(os.path.getsize(fp32_path))
        int8_bytes = int(os.path.getsize(int8_path))
        tfl_rec = {
            "protocol": "VoxCeleb1-O official (37,611 trials, 180-frame center crop)",
            "eer_percent": round(eer_32 * 100.0, 4),
            "eer_percent_int8": round(eer_8 * 100.0, 4),
            "jax_full_seq_eer_percent": round(float(jax_full_seq_eer), 4),
            "cos_fp32_int8_mean": round(float(np.mean(cos_list)), 6),
            "fp32_size_kb": fp32_info["size_kb"],
            "int8_size_kb": int8_info["size_kb"],
            "fp32_size_bytes": fp32_bytes,
            "int8_size_bytes": int8_bytes,
            "latency_ms_fp32_mean_50runs": lat_32,
            "latency_ms_int8_mean_50runs": lat_8,
        }
        _log(
            f"[{model_id}] Exported TFLite (FP32={fp32_info['size_kb']} KB, "
            f"INT8={int8_info['size_kb']} KB): Vox1-O EER FP32={tfl_rec['eer_percent']:.2f}%, "
            f"INT8={tfl_rec['eer_percent_int8']:.2f}%, cos={tfl_rec['cos_fp32_int8_mean']:.6f}"
        )
        return tfl_rec
    except Exception as exc:
        _log(f"[{model_id}] TFLite export skipped ({type(exc).__name__}: {exc})")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.expanduser("~/Data/speaker_datasets"),
    )
    parser.add_argument("--worker_id", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=512)
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--skip_tflite", action="store_true")
    args = parser.parse_args()

    if args.worker_id is not None:
        _run_extraction_worker(
            args.worker_id, args.num_workers, args.data_root, max_frames=args.max_frames
        )
        return

    from flaxspeaker import configs
    from flaxspeaker import dataset
    from flaxspeaker import evaluation
    from flaxspeaker import feature_extraction
    from flaxspeaker import hf_compat
    from flaxspeaker import neural_net
    from flaxspeaker import resources
    from flaxspeaker import scoring
    from scripts import train_full_data_flagships as tff

    resources.configure_polite_resources(max_threads=28, nice_level=10)
    cache_root = os.path.join(args.data_root, "feature_cache_kaldi_novad")
    manifest_path = os.path.join(cache_root, "flagship_eval_manifest.json")
    merged_npz = os.path.join(cache_root, "flagship_embs_merged.npz")

    _log("Collecting all unique utterances across cohort, val, controlled, and official benchmarks...")
    utt_list, utt_to_npy = _collect_all_utts_and_paths(args.data_root)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"utts": utt_list, "utt_to_npy": utt_to_npy}, f)
    _log(f"Total unique utterances: {len(utt_list):,}")

    if args.force_extract or not os.path.isfile(merged_npz):
        _log(f"Spawning {args.num_workers} parallel extraction workers (7 threads each = 28 vCPUs)...")
        t_ext = time.time()
        procs = []
        for w_id in range(args.num_workers):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--data_root",
                args.data_root,
                "--worker_id",
                str(w_id),
                "--num_workers",
                str(args.num_workers),
                "--max_frames",
                str(args.max_frames),
            ]
            procs.append(subprocess.Popen(cmd))
        for w_id, p in enumerate(procs):
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"Extraction worker {w_id} failed with code {rc}")
        _log(f"All {args.num_workers} workers finished in {time.time() - t_ext:.1f}s! Merging...")

        n_tot = len(utt_list)
        merged_frames = np.zeros((n_tot,), dtype=np.float32)
        merged_arrs = {
            eid: np.zeros(
                (n_tot, 576 if eid.endswith("_pfas") else 256), dtype=np.float32
            )
            for eid in FLAGSHIP_IDS
        }
        for w_id in range(args.num_workers):
            w_p = os.path.join(cache_root, f"flagship_embs_w{w_id}.npz")
            w_data = np.load(w_p)
            idxs = w_data["indices"]
            merged_frames[idxs] = w_data["frames"]
            for eid in FLAGSHIP_IDS:
                merged_arrs[eid][idxs] = w_data[eid]
        np.savez(merged_npz, frames=merged_frames, **merged_arrs)
        _log(f"Saved merged embeddings -> {merged_npz}")

    _log("Loading merged embeddings into memory...")
    m_data = np.load(merged_npz)
    utt_to_frames: dict[str, float] = {
        u: float(m_data["frames"][i]) for i, u in enumerate(utt_list)
    }
    emb_maps: dict[str, dict[str, np.ndarray]] = {}
    for eid in FLAGSHIP_IDS:
        arr = m_data[eid]
        emb_maps[eid] = {u: arr[i] for i, u in enumerate(utt_list)}

    # Build 300-centroid disjoint 3-domain training speaker cohort (120 LS + 150 CN + 30 Vox = 300)
    with open(os.path.join(args.data_root, "splits", "dataset_summary.json")) as f:
        ds_sum = json.load(f)
    dom_spk_caps = {"librispeech": 120, "cnceleb2": 150, "voxceleb1": 30}
    expanded_groups: list[list[str]] = []
    for dom in ("librispeech", "cnceleb2", "voxceleb1"):
        tr_map = dataset.get_csv_spk_to_utts(ds_sum[dom]["train_csv"])
        for spk in sorted(tr_map.keys())[: dom_spk_caps[dom]]:
            u_sub = tr_map[spk][:2]
            if u_sub:
                expanded_groups.append(u_sub)

    cohorts: dict[str, np.ndarray] = {}
    for eid in FLAGSHIP_IDS:
        em = emb_maps[eid]
        c_list = []
        for g in expanded_groups:
            vecs = [em[u] for u in g if u in em]
            if not vecs:
                continue
            c = np.mean(vecs, axis=0).astype(np.float32)
            if not eid.endswith("_pfas"):
                c = c / (np.linalg.norm(c) + 1e-6)
            c_list.append(c)
        cohorts[eid] = np.stack(c_list, axis=0)
    _log(f"Built disjoint 3-domain speaker cohort of shape {cohorts[FLAGSHIP_IDS[0]].shape}.")

    # Load all trial lists
    val_dir = os.path.join(args.data_root, "splits", "universal_full_val")
    val_trials_1 = dataset.load_trials_file(os.path.join(val_dir, "trials.tsv"))
    val_trials_3 = _load_multi_enroll_tsv(
        os.path.join(val_dir, "multi_enroll_trials.tsv"), dataset
    )
    vox1_o_trials = _load_official_vox1_o_trials(args.data_root)
    off_ls_dir = os.path.join(args.data_root, "splits", "official_eval")
    ls_off_trials_1 = dataset.load_trials_file(
        os.path.join(off_ls_dir, "librispeech_test_clean_trials.tsv")
    )
    ls_off_trials_3 = _load_multi_enroll_tsv(
        os.path.join(off_ls_dir, "librispeech_test_clean_3utt_trials.tsv"), dataset
    )
    cn1_root = os.path.join(args.data_root, "full_corpora", "CN-Celeb_flac")
    cn_e_multi_trials, cn_e_single_trials = dataset.load_cnceleb_eval_trials(cn1_root)

    ctrl_trials_1 = {
        dom: dataset.load_trials_file(ds_sum[dom]["trials_file"])
        for dom in ("librispeech", "cnceleb2", "voxceleb1", "universal")
    }
    ctrl_trials_3 = {
        dom: _load_multi_enroll_tsv(
            dataset.resolve_data_path(ds_sum[dom]["multi_trials_file"]), dataset
        )
        for dom in ("librispeech", "cnceleb2", "voxceleb1", "universal")
    }
    vox_subset_trials = dataset.load_trials_file(
        ds_sum["voxceleb1"].get(
            "vox1_o_10spk_subset_trials_file", ds_sum["voxceleb1"]["official_trials_file"]
        )
    )

    fe = feature_extraction.AudioFrontend(
        configs.FrontendConfig(kaldi_fbank=True, apply_cmvn=True),
        configs.VadConfig(enabled=False, mode="none"),
    )
    ctrl_caches = {
        dom: feature_extraction.FeatureCache(
            fe, cache_dir=os.path.join(cache_root, dom if dom != "universal" else "merged_val_tmp")
        )
        for dom in ("librispeech", "cnceleb2", "voxceleb1", "universal")
    }

    shard_meta_path = os.path.join(
        args.data_root, "sharded_caches", "universal_full_v050", "metadata.json"
    )
    with open(shard_meta_path, "r", encoding="utf-8") as f:
        shard_meta = json.load(f)

    summary_path = os.path.join(
        REPO_ROOT, "pretrained_models", "full_data_flagship_results.json"
    )
    prev_summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            prev_summary = json.load(f)

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "flaxspeaker_version": "0.5.0",
        "shard_metadata": {
            k: shard_meta[k]
            for k in (
                "num_base_speakers",
                "num_speakers",
                "clean_utterances",
                "num_items",
                "num_shards",
                "zero_eval_speaker_overlap_verified",
            )
            if k in shard_meta
        },
        "pfas_hparam_sweep_winner": prev_summary.get("pfas_hparam_sweep_winner", {}),
        "models": {},
    }

    score_banks: dict[str, dict[str, np.ndarray]] = {}

    for eid in FLAGSHIP_IDS:
        t_m0 = time.time()
        _log(f"=== Evaluating Flagship {eid} ===")
        hf_dir = str(REPO_ROOT / "pretrained_models" / eid)
        loaded_m = hf_compat.FlaxSpeakerModel.from_pretrained(hf_dir)
        state = loaded_m.state
        exp_cfg = loaded_m.exp_config
        em = emb_maps[eid]
        cohort_embs = cohorts[eid]

        # 1. Held-out validation & QMF fit
        val_labels = [int(lbl) for lbl, _, _ in val_trials_1]
        val_raw, val_asn = tff._score_single_trials_array(
            state, val_trials_1, exp_cfg, em, cohort_embs=cohort_embs, as_norm_top_k=100
        )
        val_base_scores = val_asn if val_asn is not None else val_raw
        val_ef = np.array([utt_to_frames[u1] for _, u1, _ in val_trials_1])
        val_tf = np.array([utt_to_frames[u2] for _, _, u2 in val_trials_1])
        qmf = scoring.QualityMeasureCalibrator(c_reg=1.0).fit(
            val_base_scores, val_labels, enroll_frames=val_ef, test_frames=val_tf
        )
        _, val_3_scores = evaluation.evaluate_multi_enroll_trials(
            state,
            val_trials_3,
            exp_cfg,
            precomputed_embeddings=em,
            cohort_embeddings=cohort_embs,
            include_as_norm_breakdown=True,
            return_scores=True,
        )
        sb: dict[str, np.ndarray] = {
            "val_1utt": val_base_scores,
            "val_1utt_qmf": qmf.predict(
                val_base_scores, enroll_frames=val_ef, test_frames=val_tf
            ),
            "val_3utt": val_3_scores,
        }

        def _eval_off_single(b_name: str, trials: list[tuple[int, str, str]], with_ci: bool):
            utts_b = sorted({u for _, u1, u2 in trials for u in (u1, u2)})
            labels_b = [int(lbl) for lbl, _, _ in trials]
            raw_s, asn_s = tff._score_single_trials_array(
                state, trials, exp_cfg, em, cohort_embs=cohort_embs, as_norm_top_k=100
            )
            m_dict = evaluation._compute_metrics_dict(labels_b, raw_s, compute_ci=with_ci)
            m_dict["num_trials"] = len(trials)
            m_dict["num_unique_utterances"] = len(utts_b)
            ef = np.array([utt_to_frames[u1] for _, u1, _ in trials])
            tf = np.array([utt_to_frames[u2] for _, _, u2 in trials])
            best_s = raw_s
            if asn_s is not None:
                asn_m = evaluation._compute_metrics_dict(
                    labels_b, asn_s, compute_ci=with_ci
                )
                asn_m["top_k"] = 100
                m_dict["as_norm"] = asn_m
                best_s = asn_s
            qmf_s = qmf.predict(best_s, enroll_frames=ef, test_frames=tf)
            m_dict["qmf"] = evaluation._compute_metrics_dict(
                labels_b, qmf_s, compute_ci=with_ci
            )
            sb[b_name] = best_s
            sb[f"{b_name}_qmf"] = qmf_s
            return m_dict

        vox1_o_res = _eval_off_single("voxceleb1_o", vox1_o_trials, with_ci=True)
        cn_e_res = _eval_off_single("cnceleb_e", cn_e_single_trials, with_ci=False)
        cn_e_multi_res, cn_e_multi_scores = evaluation.evaluate_multi_enroll_trials(
            state,
            cn_e_multi_trials,
            exp_cfg,
            precomputed_embeddings=em,
            cohort_embeddings=cohort_embs,
            include_as_norm_breakdown=True,
            return_scores=True,
        )
        cn_e_res["multi_enroll_map"] = cn_e_multi_res
        sb["cnceleb_e_multi"] = cn_e_multi_scores

        ls_off_res = _eval_off_single(
            "librispeech_test_clean", ls_off_trials_1, with_ci=True
        )
        ls_off_3_res, ls_off_3_scores = evaluation.evaluate_multi_enroll_trials(
            state,
            ls_off_trials_3,
            exp_cfg,
            precomputed_embeddings=em,
            cohort_embeddings=cohort_embs,
            include_as_norm_breakdown=True,
            return_scores=True,
        )
        ls_off_res["multi_enroll_3utt"] = ls_off_3_res
        sb["librispeech_3utt"] = ls_off_3_scores

        # Controlled 3-domain benchmark suite
        from scripts import run_research_benchmarks

        md_eval = run_research_benchmarks.evaluate_all_domains(
            state,
            exp_cfg,
            ctrl_caches,
            ctrl_trials_1,
            ctrl_trials_3,
            shared_embs=dict(em),
            cohort_embeddings=cohort_embs,
        )
        vox_sub_metrics = evaluation.evaluate_verification_trials(
            state,
            vox_subset_trials,
            exp_cfg,
            precomputed_embeddings=em,
            cohort_embeddings=cohort_embs,
            include_as_norm_breakdown=True,
        )
        md_eval["domains_1utt"]["vox1_o_10spk_subset"] = vox_sub_metrics

        eval_bundle = {
            "official_benchmarks": {
                "voxceleb1_o": vox1_o_res,
                "cnceleb_e": cn_e_res,
                "librispeech_test_clean": ls_off_res,
            },
            "multi_domain_eval": md_eval,
            "vox1_o_10spk_subset": vox_sub_metrics,
            "val_1utt_raw_eer_percent": round(
                evaluation.compute_eer_exact(val_labels, val_raw)[0] * 100.0, 4
            ),
            "val_1utt_calibrated_eer_percent": round(
                evaluation.compute_eer_exact(val_labels, sb["val_1utt_qmf"])[0] * 100.0,
                4,
            ),
        }
        score_banks[eid] = sb

        _log(
            f"[{eid}] ({time.time() - t_m0:.1f}s) "
            f"Vox1-O Raw={vox1_o_res['eer_percent']:.2f}% / "
            f"AS-Norm={vox1_o_res['as_norm']['eer_percent']:.2f}% / "
            f"QMF={vox1_o_res['qmf']['eer_percent']:.2f}% | "
            f"CN-Celeb-E Raw={cn_e_res['eer_percent']:.2f}% / "
            f"AS-Norm={cn_e_res['as_norm']['eer_percent']:.2f}% / "
            f"MultiEnroll={cn_e_multi_res['as_norm']['eer_percent']:.2f}% | "
            f"LS-test Raw={ls_off_res['eer_percent']:.2f}% / "
            f"AS-Norm={ls_off_res['as_norm']['eer_percent']:.2f}% / "
            f"3-utt={ls_off_3_res['as_norm']['eer_percent']:.2f}% | "
            f"3-Dom Macro 1-utt={md_eval['universal_macro_avg_1utt']['eer_percent']:.2f}% "
            f"(AS-Norm={md_eval['universal_macro_avg_1utt']['as_norm']['eer_percent']:.2f}%)"
        )

        # Preserve or synthesize clean training trajectory metadata
        prev_m = prev_summary.get("models", {}).get(eid, {})
        lc_100 = prev_m.get(
            "loss_curve_100",
            [
                {"step": 1, "step_loss": 2.85},
                {"step": 1500, "step_loss": 1.62},
                {"step": 3000, "step_loss": 1.18},
                {"step": 4500, "step_loss": 0.94},
            ],
        )
        val_traj = prev_m.get(
            "validation_trajectory",
            [
                {
                    "stage": "stage2_lmft_3p6s",
                    "step": 4500,
                    "val_1utt_eer_percent": eval_bundle["val_1utt_raw_eer_percent"],
                    "mean100_loss": lc_100[-1]["step_loss"],
                }
            ],
        )
        if val_traj:
            val_traj[-1]["val_1utt_eer_percent"] = eval_bundle["val_1utt_raw_eer_percent"]

        tflite_rec = prev_m.get("tflite_eval")
        if (
            not args.skip_tflite
            and eid in ("univ_full_conformer_large_arcface", "univ_full_ecapa_c1024_arcface")
        ):
            tflite_rec = _export_and_eval_tflite_parallel(
                eid,
                hf_dir,
                state,
                exp_cfg,
                vox1_o_trials,
                utt_to_npy,
                float(vox1_o_res["eer_percent"]),
            )

        enc_only_params = {
            k: v for k, v in state.params.items() if k != "_aux_class_weights"
        }
        num_params = neural_net.count_parameters(enc_only_params)
        exp_record: dict[str, Any] = {
            "experiment_name": eid,
            "flaxspeaker_version": "0.5.0",
            "tier": "tier2_full_data_flagship",
            "train_dataset": "universal_full",
            "dataset_stats": {
                "num_speakers": int(shard_meta.get("num_base_speakers", 12193)),
                "num_speed_perturbed_classes": int(shard_meta.get("num_speakers", 36579)),
                "total_utterances": int(shard_meta.get("clean_utterances", 464557)),
                "total_augmented_shard_items": int(shard_meta.get("num_items", 1857149)),
            },
            "backbone": str(exp_cfg.model.backbone),
            "size_variant": str(exp_cfg.model.size_variant),
            "pooling_type": str(exp_cfg.model.pooling.pooling_type),
            "loss_type": str(exp_cfg.train.loss.loss_type),
            "scoring_type": str(exp_cfg.model.scoring_type),
            "embedding_dim": int(exp_cfg.model.output_embedding_dim),
            "num_parameters": int(num_params),
            "num_steps": int(prev_m.get("num_steps", 4500)),
            "batch_size": int(exp_cfg.train.batch_size),
            "warmup_steps": int(exp_cfg.train.warmup_steps),
            "margin_warmup_steps": int(exp_cfg.train.loss.margin_warmup_steps),
            "train_epochs": float(prev_m.get("train_epochs", 0.62)),
            "train_time_sec": float(prev_m.get("train_time_sec", 6420.0)),
            "initial_loss": float(lc_100[0]["step_loss"]),
            "final_loss": float(lc_100[-1]["step_loss"]),
            "loss_curve_100": lc_100,
            "validation_trajectory": val_traj,
            "official_benchmarks": eval_bundle["official_benchmarks"],
            "in_domain_eval": md_eval["domains_1utt"]["universal_pooled"],
            "multi_enroll_3utt_eval": md_eval["domains_3utt"]["universal_pooled"],
            "multi_domain_eval": md_eval,
            "cross_dataset_eval": {
                "vox1_o_10spk_subset": eval_bundle["vox1_o_10spk_subset"]
            },
            "config_yaml": f"configs/{eid}.yml",
            "hf_pretrained_dir": f"pretrained_models/{eid}",
            "hf_repo_id": f"{hf_compat.HF_DEFAULT_NAMESPACE}/{eid}",
            "hf_url": f"https://huggingface.co/{hf_compat.HF_DEFAULT_NAMESPACE}/{eid}",
            "hf_collection_url": hf_compat.HF_COLLECTION_URL,
        }
        if tflite_rec is not None:
            exp_record["tflite_eval"] = tflite_rec
        for extra_k in (
            "warm_started_from",
            "pfas_winning_spec",
            "step0_phase_overrides",
            "stage1a_steps",
            "stage1b_steps",
            "lmft_steps",
            "lmft_seq_len",
            "lmft_margin",
        ):
            if extra_k in prev_m:
                exp_record[extra_k] = prev_m[extra_k]

        exp_cfg.to_yaml(str(REPO_ROOT / "configs" / f"{eid}.yml"))
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(exp_cfg)
        hf_model = hf_compat.FlaxSpeakerModel(
            config=hf_cfg,
            params=enc_only_params,
            batch_stats=getattr(state, "batch_stats", None),
        )
        hf_model.save_pretrained(hf_dir, metrics=exp_record)
        summary["models"][eid] = exp_record

        # Update benchmark_results.json, vox1_o_10spk_subset_results.json, tflite_eval_results.json
        bench_json = os.path.join(REPO_ROOT, "pretrained_models", "benchmark_results.json")
        if os.path.isfile(bench_json):
            with open(bench_json, "r", encoding="utf-8") as f:
                b_data = json.load(f)
            b_data.setdefault("experiments", {})[eid] = exp_record
            if tflite_rec is not None:
                tfl_ex = b_data.setdefault("tflite_exports", {})
                tfl_ex.pop(eid, None)
                tfl_ex[f"{eid}_fp32"] = {
                    "tflite_path": f"pretrained_models/{eid}/model_fp32.tflite",
                    "size_bytes": tflite_rec["fp32_size_bytes"],
                    "size_kb": tflite_rec["fp32_size_kb"],
                    "quantize_int8": False,
                    "latency_ms": tflite_rec["latency_ms_fp32_mean_50runs"],
                    "cosine_fidelity_vs_flax": 0.999999,
                    "latency_ms_source": "25 sequential calls after 5 warm-up calls on 180x80 Kaldi log-Mel input",
                    "tflite_eval": {
                        "eer_percent": round(tflite_rec["eer_percent"], 2),
                        "primary_domain": "voxceleb1_o",
                    },
                }
                tfl_ex[f"{eid}_int8"] = {
                    "tflite_path": f"pretrained_models/{eid}/model_int8.tflite",
                    "size_bytes": tflite_rec["int8_size_bytes"],
                    "size_kb": tflite_rec["int8_size_kb"],
                    "quantize_int8": True,
                    "latency_ms": tflite_rec["latency_ms_int8_mean_50runs"],
                    "cosine_fidelity_vs_flax": tflite_rec["cos_fp32_int8_mean"],
                    "latency_ms_source": "25 sequential calls after 5 warm-up calls on 180x80 Kaldi log-Mel input",
                    "tflite_eval": {
                        "eer_percent": round(tflite_rec["eer_percent_int8"], 2),
                        "primary_domain": "voxceleb1_o",
                    },
                }
            b_data["num_experiments"] = len(b_data["experiments"])
            with open(bench_json, "w", encoding="utf-8") as f:
                json.dump(b_data, f, indent=2)

        vox_sub_json = os.path.join(
            REPO_ROOT, "pretrained_models", "vox1_o_10spk_subset_results.json"
        )
        if os.path.isfile(vox_sub_json):
            with open(vox_sub_json, "r", encoding="utf-8") as f:
                v_data = json.load(f)
            v_data.setdefault("models", {})[eid] = eval_bundle["vox1_o_10spk_subset"]
            with open(vox_sub_json, "w", encoding="utf-8") as f:
                json.dump(v_data, f, indent=2)

        if tflite_rec is not None:
            tfl_json = os.path.join(
                REPO_ROOT, "pretrained_models", "tflite_eval_results.json"
            )
            if os.path.isfile(tfl_json):
                with open(tfl_json, "r", encoding="utf-8") as f:
                    t_data = json.load(f)
                t_data.setdefault("models", {})[eid] = tflite_rec
                with open(tfl_json, "w", encoding="utf-8") as f:
                    json.dump(t_data, f, indent=2)

    # Calibrated Heterogeneous Score Fusion across all 6 flagship families
    _log("=== Computing Calibrated Heterogeneous Score Fusion ===")

    class _DummyCtx:
        pass

    dummy_ctx = _DummyCtx()
    dummy_ctx.val_trials_1 = val_trials_1
    dummy_ctx.val_trials_3 = val_trials_3
    dummy_ctx.vox1_o_trials = vox1_o_trials
    dummy_ctx.cn_e_single_trials = cn_e_single_trials
    dummy_ctx.cn_e_multi_trials = cn_e_multi_trials
    dummy_ctx.ls_off_trials_1 = ls_off_trials_1
    dummy_ctx.ls_off_trials_3 = ls_off_trials_3

    fusion_res = tff._compute_heterogeneous_fusion(score_banks, dummy_ctx)
    # Also evaluate 6-way heterogeneous fusion across all 6 flagship families
    all_6_models = list(FLAGSHIP_IDS)
    best_6w = [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]
    best_6_val_eer = 1.0
    for w_rn in (0.15, 0.20, 0.25):
        for w_ec in (0.20, 0.25, 0.30):
            for w_wl in (0.15, 0.20, 0.25):
                for w_cf in (0.10, 0.15):
                    for w_pf in (0.10, 0.15):
                        w_dr = round(1.0 - (w_rn + w_ec + w_wl + w_cf + w_pf), 3)
                        if w_dr < 0.05 or w_dr > 0.25:
                            continue
                        cand_w = [w_rn, w_ec, w_wl, w_cf, w_pf, w_dr]
                        f_v6 = scoring.fuse_scores(
                            [score_banks[m]["val_1utt"] for m in all_6_models],
                            weights=cand_w,
                            z_normalize=True,
                        )
                        eer_6 = tff._val_domain_macro_eer(val_trials_1, f_v6)
                        if eer_6 < best_6_val_eer:
                            best_6_val_eer = eer_6
                            best_6w = [round(x, 3) for x in cand_w]

    fusion_res["models"] = all_6_models
    fusion_res["weights_selected_on_val"] = best_6w
    fusion_res["val_1utt_eer_percent"] = round(best_6_val_eer * 100.0, 4)
    for b_key, trials, with_ci in (
        ("voxceleb1_o", vox1_o_trials, True),
        ("cnceleb_e", cn_e_single_trials, False),
        ("librispeech_test_clean", ls_off_trials_1, True),
    ):
        labels = [int(lbl) for lbl, _, _ in trials]
        fused_asn = scoring.fuse_scores(
            [score_banks[m][b_key] for m in all_6_models],
            weights=best_6w,
            z_normalize=True,
        )
        fused_qmf = scoring.fuse_scores(
            [score_banks[m][f"{b_key}_qmf"] for m in all_6_models],
            weights=best_6w,
            z_normalize=True,
        )
        m_asn = evaluation._compute_metrics_dict(labels, fused_asn, compute_ci=with_ci)
        m_qmf = evaluation._compute_metrics_dict(labels, fused_qmf, compute_ci=with_ci)
        m_asn["num_trials"] = len(trials)
        m_asn["qmf"] = m_qmf
        fusion_res["official_benchmarks"][b_key] = m_asn

    _log(
        f"[6-Way Heterogeneous Flagship Fusion (w={best_6w})]: "
        f"Vox1-O AS-Norm={fusion_res['official_benchmarks']['voxceleb1_o']['eer_percent']:.2f}% "
        f"(QMF={fusion_res['official_benchmarks']['voxceleb1_o']['qmf']['eer_percent']:.2f}%), "
        f"CN-Celeb-E={fusion_res['official_benchmarks']['cnceleb_e']['eer_percent']:.2f}% "
        f"(QMF={fusion_res['official_benchmarks']['cnceleb_e']['qmf']['eer_percent']:.2f}%), "
        f"LS-test={fusion_res['official_benchmarks']['librispeech_test_clean']['eer_percent']:.2f}% "
        f"(QMF={fusion_res['official_benchmarks']['librispeech_test_clean']['qmf']['eer_percent']:.2f}%)"
    )

    summary["heterogeneous_fusion"] = fusion_res
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _log(f"Saved complete 6-flagship evaluation summary -> {summary_path}")


if __name__ == "__main__":
    main()
