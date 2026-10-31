#!/usr/bin/env python3
"""Build, calibrate, verify, and save all 6 multi-domain v0.5.0 flagship models.

Implements both Path A (native Flax WavLM-Base+ SSL Transformer + X-Vector +
2D Acoustic ResNet-34) and Path B (Kaldi HTK 80-D Mel-filterbank, continuous
non-VAD frame sequences, 2D Dual-Stream ResNet-34 r-vector, Hybrid 2D-1D
ECAPA-TDNN c=1024, 2D-ResNet-Stem Conformer-Large, PFAS 576-D super-vectors,
and Dr-Vectors decision residual network) calibrated across all 3 domains
(VoxCeleb + CN-Celeb + LibriSpeech).
"""

import io
import json
import os
import pickle
import sys
import time
import zipfile
from pathlib import Path

import flax.core
from flax import linen as nn
from huggingface_hub import hf_hub_download
import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flaxspeaker import backbones  # noqa: E402
from flaxspeaker import configs  # noqa: E402
from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import evaluation  # noqa: E402
from flaxspeaker import feature_extraction  # noqa: E402
from flaxspeaker import hf_compat  # noqa: E402
from flaxspeaker import neural_net  # noqa: E402
from flaxspeaker import pooling  # noqa: E402
from flaxspeaker import resources  # noqa: E402
from flaxspeaker import scoring  # noqa: E402


def load_pt_state_dict(pt_path: str) -> dict[str, np.ndarray]:
    """Load a PyTorch zip checkpoint into a NumPy dictionary without requiring torch."""
    with zipfile.ZipFile(pt_path, "r") as zf:
        pkl_name = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
        prefix = pkl_name[: -len("data.pkl")]
        storage_cache: dict[str, np.ndarray] = {}

        class _DummyStorage:
            def __init__(self, name: str):
                self.__name__ = name

        class _CustomUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    def _rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                        arr = storage[storage_offset:storage_offset + int(np.prod(size))]
                        return arr.reshape(size)
                    return _rebuild
                if "Storage" in name:
                    return _DummyStorage(name)
                if module == "collections" and name == "OrderedDict":
                    from collections import OrderedDict
                    return OrderedDict
                return super().find_class(module, name)

            def persistent_load(self, pid):
                _, stype, key, _, _ = pid
                if key not in storage_cache:
                    raw = zf.read(f"{prefix}data/{key}")
                    sname = getattr(stype, "__name__", str(stype))
                    dt = np.int64 if "Long" in sname else np.float32
                    storage_cache[key] = np.frombuffer(raw, dtype=dt).copy()
                return storage_cache[key]

        return dict(_CustomUnpickler(io.BytesIO(zf.read(pkl_name))).load())


def populate_dual_resnet_module(bb_p, bb_bs, sd0, sd1):
    """Populate a Flax ResNetBackbone (base_channels=64, wespeaker_mode=True) from dual domains."""
    def copy_conv2d_dual(flax_conv, k_name):
        w0 = np.transpose(sd0[k_name], (2, 3, 1, 0))
        w1 = np.transpose(sd1[k_name], (2, 3, 1, 0))
        H, W, I0, O0 = w0.shape
        if flax_conv["kernel"].shape[2] == I0:
            w = np.concatenate([w0, w1], axis=3)
        else:
            w = np.zeros((H, W, 2 * I0, 2 * O0), dtype=np.float32)
            w[:, :, :I0, :O0] = w0
            w[:, :, I0:, O0:] = w1
        flax_conv["kernel"] = jnp.asarray(w)

    def copy_bn_dual(flax_bn_p, flax_bn_bs, prefix):
        flax_bn_p["scale"] = jnp.asarray(
            np.concatenate([sd0[f"{prefix}.weight"], sd1[f"{prefix}.weight"]])
        )
        flax_bn_p["bias"] = jnp.asarray(
            np.concatenate([sd0[f"{prefix}.bias"], sd1[f"{prefix}.bias"]])
        )
        flax_bn_bs["running_mean"] = jnp.asarray(
            np.concatenate([sd0[f"{prefix}.running_mean"], sd1[f"{prefix}.running_mean"]])
        )
        flax_bn_bs["running_var"] = jnp.asarray(
            np.concatenate([sd0[f"{prefix}.running_var"], sd1[f"{prefix}.running_var"]])
        )

    copy_conv2d_dual(bb_p["stem_conv"], "conv1.weight")
    copy_bn_dual(bb_p["stem_bn"], bb_bs["stem_bn"], "bn1")

    for s_idx, n_blocks in enumerate((3, 4, 6, 3)):
        pt_layer = f"layer{s_idx + 1}"
        for b_idx in range(n_blocks):
            fb_name = f"stage{s_idx}_block{b_idx}"
            pt_blk = f"{pt_layer}.{b_idx}"
            copy_conv2d_dual(bb_p[fb_name]["conv1"], f"{pt_blk}.conv1.weight")
            copy_bn_dual(bb_p[fb_name]["bn1"], bb_bs[fb_name]["bn1"], f"{pt_blk}.bn1")
            copy_conv2d_dual(bb_p[fb_name]["conv2"], f"{pt_blk}.conv2.weight")
            copy_bn_dual(bb_p[fb_name]["bn2"], bb_bs[fb_name]["bn2"], f"{pt_blk}.bn2")
            if "downsample" in bb_p[fb_name]:
                copy_conv2d_dual(
                    bb_p[fb_name]["downsample"], f"{pt_blk}.shortcut.0.weight"
                )
                copy_bn_dual(
                    bb_p[fb_name]["downsample_bn"],
                    bb_bs[fb_name]["downsample_bn"],
                    f"{pt_blk}.shortcut.1",
                )

    w_seg0, b_seg0 = sd0["seg_1.weight"].T, sd0["seg_1.bias"]
    w_seg1, b_seg1 = sd1["seg_1.weight"].T, sd1["seg_1.bias"]
    W_512 = np.zeros((10240, 512), dtype=np.float32)
    b_512 = np.zeros((512,), dtype=np.float32)
    W_512[0:2560, 0:256] = w_seg0[0:2560, :]
    W_512[5120:7680, 0:256] = w_seg0[2560:5120, :]
    b_512[0:256] = b_seg0
    W_512[2560:5120, 256:512] = w_seg1[0:2560, :]
    W_512[7680:10240, 256:512] = w_seg1[2560:5120, :]
    b_512[256:512] = b_seg1
    return W_512, b_512


def populate_ecapa_c1024_module(bb_p, bb_bs, pool_p, sd_ec):
    """Populate a Flax EcapaTdnnBackbone (c=1024) + AttentiveStatsPooling."""
    bb_p["stem_conv"]["kernel"] = jnp.asarray(
        np.transpose(sd_ec["layer1.conv.weight"], (2, 1, 0))
    )
    bb_p["stem_conv"]["bias"] = jnp.asarray(sd_ec["layer1.conv.bias"])
    bb_p["stem_bn"]["scale"] = jnp.asarray(sd_ec["layer1.bn.weight"])
    bb_p["stem_bn"]["bias"] = jnp.asarray(sd_ec["layer1.bn.bias"])
    bb_bs["stem_bn"]["running_mean"] = jnp.asarray(sd_ec["layer1.bn.running_mean"])
    bb_bs["stem_bn"]["running_var"] = jnp.asarray(sd_ec["layer1.bn.running_var"])

    perm = np.concatenate([np.arange(7 * 128, 8 * 128), np.arange(0, 7 * 128)])
    for i in range(3):
        pt_p = f"layer{i + 2}.se_res2block"
        fb = bb_p[f"se_res2_{i}"]
        fbs = bb_bs[f"se_res2_{i}"]
        fb["pw1"]["kernel"] = jnp.asarray(
            sd_ec[f"{pt_p}.0.conv.weight"][:, :, 0].T[:, perm]
        )
        fb["pw1"]["bias"] = jnp.asarray(sd_ec[f"{pt_p}.0.conv.bias"][perm])
        fb["bn1"]["scale"] = jnp.asarray(sd_ec[f"{pt_p}.0.bn.weight"][perm])
        fb["bn1"]["bias"] = jnp.asarray(sd_ec[f"{pt_p}.0.bn.bias"][perm])
        fbs["bn1"]["running_mean"] = jnp.asarray(
            sd_ec[f"{pt_p}.0.bn.running_mean"][perm]
        )
        fbs["bn1"]["running_var"] = jnp.asarray(
            sd_ec[f"{pt_p}.0.bn.running_var"][perm]
        )

        for s_idx in range(1, 8):
            c_idx = s_idx - 1
            fb[f"res2_conv_{s_idx}"]["kernel"] = jnp.asarray(
                np.transpose(sd_ec[f"{pt_p}.1.convs.{c_idx}.weight"], (2, 1, 0))
            )
            fb[f"res2_conv_{s_idx}"]["bias"] = jnp.asarray(
                sd_ec[f"{pt_p}.1.convs.{c_idx}.bias"]
            )
            fb[f"res2_bn_{s_idx}"]["scale"] = jnp.asarray(
                sd_ec[f"{pt_p}.1.bns.{c_idx}.weight"]
            )
            fb[f"res2_bn_{s_idx}"]["bias"] = jnp.asarray(
                sd_ec[f"{pt_p}.1.bns.{c_idx}.bias"]
            )
            fbs[f"res2_bn_{s_idx}"]["running_mean"] = jnp.asarray(
                sd_ec[f"{pt_p}.1.bns.{c_idx}.running_mean"]
            )
            fbs[f"res2_bn_{s_idx}"]["running_var"] = jnp.asarray(
                sd_ec[f"{pt_p}.1.bns.{c_idx}.running_var"]
            )

        fb["pw2"]["kernel"] = jnp.asarray(
            sd_ec[f"{pt_p}.2.conv.weight"][:, :, 0].T[perm, :]
        )
        fb["pw2"]["bias"] = jnp.asarray(sd_ec[f"{pt_p}.2.conv.bias"])
        fb["bn2"]["scale"] = jnp.asarray(sd_ec[f"{pt_p}.2.bn.weight"])
        fb["bn2"]["bias"] = jnp.asarray(sd_ec[f"{pt_p}.2.bn.bias"])
        fbs["bn2"]["running_mean"] = jnp.asarray(sd_ec[f"{pt_p}.2.bn.running_mean"])
        fbs["bn2"]["running_var"] = jnp.asarray(sd_ec[f"{pt_p}.2.bn.running_var"])

        fb["se"]["se_fc1"]["kernel"] = jnp.asarray(sd_ec[f"{pt_p}.3.linear1.weight"].T)
        fb["se"]["se_fc1"]["bias"] = jnp.asarray(sd_ec[f"{pt_p}.3.linear1.bias"])
        fb["se"]["se_fc2"]["kernel"] = jnp.asarray(sd_ec[f"{pt_p}.3.linear2.weight"].T)
        fb["se"]["se_fc2"]["bias"] = jnp.asarray(sd_ec[f"{pt_p}.3.linear2.bias"])

    bb_p["mfa_proj"]["kernel"] = jnp.asarray(sd_ec["conv.weight"][:, :, 0].T)
    bb_p["mfa_proj"]["bias"] = jnp.asarray(sd_ec["conv.bias"])
    bb_p["mfa_bn"]["scale"] = jnp.ones_like(bb_p["mfa_bn"]["scale"])
    bb_p["mfa_bn"]["bias"] = jnp.zeros_like(bb_p["mfa_bn"]["bias"])
    bb_bs["mfa_bn"]["running_mean"] = jnp.zeros_like(bb_bs["mfa_bn"]["running_mean"])
    bb_bs["mfa_bn"]["running_var"] = jnp.ones_like(bb_bs["mfa_bn"]["running_var"])

    pool_p["asp_hidden"]["kernel"] = jnp.asarray(
        sd_ec["pool.linear1.weight"][:, :, 0].T
    )
    pool_p["asp_hidden"]["bias"] = jnp.asarray(sd_ec["pool.linear1.bias"])
    pool_p["asp_out"]["kernel"] = jnp.asarray(
        sd_ec["pool.linear2.weight"][:, :, 0].T
    )
    pool_p["asp_out"]["bias"] = jnp.asarray(sd_ec["pool.linear2.bias"])

    bn_w = sd_ec["bn.weight"]
    bn_b = sd_ec["bn.bias"]
    bn_m = sd_ec["bn.running_mean"]
    bn_v = sd_ec["bn.running_var"]
    bn_scale = bn_w / np.sqrt(bn_v + 1e-5)
    bn_shift = -bn_m * bn_scale + bn_b
    lin_w = sd_ec["linear.weight"].T  # (3072, 192)
    lin_b = sd_ec["linear.bias"]      # (192,)
    W_ec = bn_scale[:, None] * lin_w  # (3072, 192)
    b_ec = bn_shift @ lin_w + lin_b   # (192,)
    return W_ec, b_ec


def populate_wavlm_module(bb_p, sd_wlm, ds_sum, fe):
    """Populate a Flax WavLMBackbone from microsoft/wavlm-base-plus-sv and train Mel-to-SSL stem."""
    wg = sd_wlm["wavlm.encoder.pos_conv_embed.conv.weight_g"]  # (1, 1, 128)
    wv = sd_wlm["wavlm.encoder.pos_conv_embed.conv.weight_v"]  # (768, 48, 128)
    wn = wv * (wg / (np.linalg.norm(wv, axis=(0, 1), keepdims=True) + 1e-12))
    bb_p["pos_conv"]["kernel"] = jnp.asarray(np.transpose(wn, (2, 1, 0)))
    bb_p["pos_conv"]["bias"] = jnp.asarray(
        sd_wlm["wavlm.encoder.pos_conv_embed.conv.bias"]
    )

    bb_p["encoder_ln"]["scale"] = jnp.asarray(
        sd_wlm["wavlm.encoder.layer_norm.weight"]
    )
    bb_p["encoder_ln"]["bias"] = jnp.asarray(
        sd_wlm["wavlm.encoder.layer_norm.bias"]
    )

    for idx in range(12):
        lp = bb_p[f"layer_{idx}"]
        pt_l = f"wavlm.encoder.layers.{idx}"
        if idx == 0:
            lp["rel_attn_embed"] = jnp.asarray(
                sd_wlm[f"{pt_l}.attention.rel_attn_embed.weight"]
            )
        lp["gru_rel_pos_const"] = jnp.asarray(
            sd_wlm[f"{pt_l}.attention.gru_rel_pos_const"]
        )
        lp["gru_rel_pos_linear"]["kernel"] = jnp.asarray(
            sd_wlm[f"{pt_l}.attention.gru_rel_pos_linear.weight"].T
        )
        lp["gru_rel_pos_linear"]["bias"] = jnp.asarray(
            sd_wlm[f"{pt_l}.attention.gru_rel_pos_linear.bias"]
        )
        for proj_k in ("q_proj", "k_proj", "v_proj", "out_proj"):
            lp[proj_k]["kernel"] = jnp.asarray(
                sd_wlm[f"{pt_l}.attention.{proj_k}.weight"].T
            )
            lp[proj_k]["bias"] = jnp.asarray(
                sd_wlm[f"{pt_l}.attention.{proj_k}.bias"]
            )
        lp["layer_norm"]["scale"] = jnp.asarray(
            sd_wlm[f"{pt_l}.layer_norm.weight"]
        )
        lp["layer_norm"]["bias"] = jnp.asarray(sd_wlm[f"{pt_l}.layer_norm.bias"])
        lp["intermediate_dense"]["kernel"] = jnp.asarray(
            sd_wlm[f"{pt_l}.feed_forward.intermediate_dense.weight"].T
        )
        lp["intermediate_dense"]["bias"] = jnp.asarray(
            sd_wlm[f"{pt_l}.feed_forward.intermediate_dense.bias"]
        )
        lp["output_dense"]["kernel"] = jnp.asarray(
            sd_wlm[f"{pt_l}.feed_forward.output_dense.weight"].T
        )
        lp["output_dense"]["bias"] = jnp.asarray(
            sd_wlm[f"{pt_l}.feed_forward.output_dense.bias"]
        )
        lp["final_layer_norm"]["scale"] = jnp.asarray(
            sd_wlm[f"{pt_l}.final_layer_norm.weight"]
        )
        lp["final_layer_norm"]["bias"] = jnp.asarray(
            sd_wlm[f"{pt_l}.final_layer_norm.bias"]
        )

    bb_p["layer_weights"] = jnp.asarray(sd_wlm["layer_weights"])
    bb_p["projector"]["kernel"] = jnp.asarray(sd_wlm["projector.weight"].T)
    bb_p["projector"]["bias"] = jnp.asarray(sd_wlm["projector.bias"])

    tdnn_specs = [
        (512, 5, 512),
        (512, 3, 512),
        (512, 3, 512),
        (512, 1, 512),
        (1500, 1, 512),
    ]
    for t_idx, (out_c, k_sz, in_c) in enumerate(tdnn_specs):
        w_lin = sd_wlm[f"tdnn.{t_idx}.kernel.weight"]
        w_conv = np.transpose(w_lin.reshape(out_c, k_sz, in_c), (1, 2, 0))
        bb_p[f"tdnn_{t_idx}_conv"]["kernel"] = jnp.asarray(w_conv)
        bb_p[f"tdnn_{t_idx}_conv"]["bias"] = jnp.asarray(
            sd_wlm[f"tdnn.{t_idx}.kernel.bias"]
        )

    # Train Mel-to-SSL stem (stem_conv1, stem_conv2, feat_proj_ln, feat_proj_dense) on 384 3-domain utterances
    def wavlm_wav_fe_jax(wav_batch: jax.Array) -> jax.Array:
        h = wav_batch[:, :, None]
        k_sizes = (10, 3, 3, 3, 3, 2, 2)
        strides = (5, 2, 2, 2, 2, 2, 2)
        for i, (_, s) in enumerate(zip(k_sizes, strides)):
            w = jnp.asarray(
                np.transpose(
                    sd_wlm[f"wavlm.feature_extractor.conv_layers.{i}.conv.weight"],
                    (2, 1, 0),
                )
            )
            h = jax.lax.conv_general_dilated(
                h,
                w,
                window_strides=(s,),
                padding="VALID",
                dimension_numbers=("NHC", "HIO", "NHC"),
            )
            if i == 0:
                ln_w = jnp.asarray(
                    sd_wlm["wavlm.feature_extractor.conv_layers.0.layer_norm.weight"]
                )
                ln_b = jnp.asarray(
                    sd_wlm["wavlm.feature_extractor.conv_layers.0.layer_norm.bias"]
                )
                mu = jnp.mean(h, axis=-1, keepdims=True)
                var = jnp.var(h, axis=-1, keepdims=True)
                h = (h - mu) * jax.lax.rsqrt(var + 1e-5) * ln_w + ln_b
            h = jax.nn.gelu(h, approximate=False)
        fp_ln_w = jnp.asarray(sd_wlm["wavlm.feature_projection.layer_norm.weight"])
        fp_ln_b = jnp.asarray(sd_wlm["wavlm.feature_projection.layer_norm.bias"])
        mu = jnp.mean(h, axis=-1, keepdims=True)
        var = jnp.var(h, axis=-1, keepdims=True)
        h = (h - mu) * jax.lax.rsqrt(var + 1e-5) * fp_ln_w + fp_ln_b
        fp_w = jnp.asarray(sd_wlm["wavlm.feature_projection.projection.weight"].T)
        fp_b = jnp.asarray(sd_wlm["wavlm.feature_projection.projection.bias"])
        return h @ fp_w + fp_b

    jit_wav_fe = jax.jit(wavlm_wav_fe_jax)
    wav_list = []
    mel_list = []
    for dom in ("librispeech", "cnceleb2", "voxceleb1"):
        tr_map = dataset.get_csv_spk_to_utts(ds_sum[dom]["train_csv"])
        cnt = 0
        for spk in sorted(tr_map.keys()):
            for u in tr_map[spk][:5]:
                if cnt >= 128:
                    break
                wav = feature_extraction.load_waveform(u, target_sr=16000)
                if wav.shape[0] < 48000:
                    wav = np.tile(wav, (48000 // max(1, wav.shape[0])) + 1)[:48000]
                else:
                    off = (wav.shape[0] - 48000) // 2
                    wav = wav[off:off + 48000]
                mel = fe.extract_from_waveform(wav)[:298]
                wav_norm = (wav - np.mean(wav)) / np.sqrt(np.var(wav) + 1e-7)
                wav_list.append(wav_norm.astype(np.float32))
                mel_list.append(mel.astype(np.float32))
                cnt += 1
            if cnt >= 128:
                break

    W_arr = np.stack(wav_list, axis=0)
    M_arr = np.stack(mel_list, axis=0)
    Y_ssl = np.concatenate(
        [
            np.asarray(jit_wav_fe(jnp.asarray(W_arr[s:s + 64])))
            for s in range(0, len(W_arr), 64)
        ],
        axis=0,
    )

    class MelToSslStem(nn.Module):
        tdnn_dim: int = 512
        dim: int = 768

        @nn.compact
        def __call__(self, x):
            h = nn.Conv(
                self.tdnn_dim,
                kernel_size=(3,),
                strides=(2,),
                padding=((1, 1),),
                name="stem_conv1",
            )(x)
            h = jax.nn.gelu(h, approximate=False)
            h = nn.Conv(
                self.tdnn_dim,
                kernel_size=(3,),
                strides=(1,),
                padding=((1, 1),),
                name="stem_conv2",
            )(h)
            h = jax.nn.gelu(h, approximate=False)
            h = nn.LayerNorm(epsilon=1e-5, name="feat_proj_ln")(h)
            h = nn.Dense(self.dim, name="feat_proj_dense")(h)
            return h

    stem_mod = MelToSslStem()
    stem_params = stem_mod.init(jax.random.PRNGKey(42), jnp.ones((1, 298, 80)))["params"]
    tx = optax.adamw(
        learning_rate=optax.cosine_decay_schedule(3e-3, 400, 1e-4), weight_decay=1e-4
    )
    opt_state = tx.init(stem_params)

    @jax.jit
    def train_stem_step(p, o_st, x_b, y_b):
        def loss_fn(params):
            pred = stem_mod.apply({"params": params}, x_b)
            mse = jnp.mean(jnp.square(pred - y_b))
            p_n = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + 1e-6)
            y_n = y_b / (jnp.linalg.norm(y_b, axis=-1, keepdims=True) + 1e-6)
            cos = 1.0 - jnp.mean(jnp.sum(p_n * y_n, axis=-1))
            return mse + 0.5 * cos

        l_val, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_o = tx.update(grads, o_st, p)
        return optax.apply_updates(p, updates), new_o, l_val

    rng_s = np.random.default_rng(42)
    for _ in range(400):
        idx = rng_s.choice(len(M_arr), size=64, replace=False)
        stem_params, opt_state, _ = train_stem_step(
            stem_params, opt_state, jnp.asarray(M_arr[idx]), jnp.asarray(Y_ssl[idx])
        )

    for k in ("stem_conv1", "stem_conv2", "feat_proj_ln", "feat_proj_dense"):
        bb_p[k] = stem_params[k]

    W_xvec = sd_wlm["feature_extractor.weight"].T  # (3000, 512)
    b_xvec = sd_wlm["feature_extractor.bias"]      # (512,)
    return W_xvec, b_xvec


def make_flagship_configs():
    """Build the 6 canonical ExperimentConfig objects for v0.5.0."""
    def base_cfg(backbone: str) -> configs.ExperimentConfig:
        cfg = configs.ExperimentConfig()
        cfg.data.dataset_name = "universal_full"
        cfg.frontend.kaldi_fbank = True
        cfg.frontend.apply_cmvn = True
        cfg.frontend.cmvn_norm_vars = False
        cfg.vad.enabled = False
        cfg.vad.mode = "none"
        cfg.model.backbone = backbone
        cfg.model.pooling.pooling_type = "stats"
        cfg.model.pooling.unbiased_std = True
        cfg.model.pooling.use_bottleneck_ln = False
        cfg.model.pooling.use_nonlinear_bottleneck = False
        cfg.model.embedding_dim = 256
        cfg.model.normalize_embedding = True
        cfg.model.full_sequence_inference = True
        cfg.eval.full_sequence_inference = True
        cfg.eval.max_eval_frames_cap = 768
        cfg.train.loss.loss_type = "subcenter_arcface"
        cfg.train.loss.subcenter_k = 3
        cfg.train.loss.scale = 32.0
        cfg.train.loss.margin = 0.30
        cfg.loss = cfg.train.loss
        return cfg

    cfgs = {}

    # 1. univ_full_resnet34_arcface
    c_rn = base_cfg("resnet")
    c_rn.model.apply_size_variant("large")
    c_rn.model.resnet.base_channels = 64
    c_rn.model.resnet.num_blocks = (3, 4, 6, 3)
    c_rn.model.resnet.use_se = False
    c_rn.model.resnet.out_dim = 0
    c_rn.model.resnet.use_batch_norm = True
    c_rn.model.resnet.wespeaker_mode = True
    cfgs["univ_full_resnet34_arcface"] = c_rn

    # 2. univ_full_ecapa_c1024_arcface
    c_ec = base_cfg("ecapa_tdnn")
    c_ec.model.apply_size_variant("large")
    c_ec.model.ecapa_tdnn.channels = 1024
    c_ec.model.ecapa_tdnn.scale = 8
    c_ec.model.ecapa_tdnn.kernel_sizes = (5, 3, 3, 3)
    c_ec.model.ecapa_tdnn.dilations = (1, 2, 3, 4)
    c_ec.model.ecapa_tdnn.mfa_dim = 1536
    c_ec.model.ecapa_tdnn.se_bottleneck_dim = 128
    c_ec.model.ecapa_tdnn.use_batch_norm = True
    c_ec.model.ecapa_tdnn.norm_after_activation = True
    c_ec.model.ecapa_tdnn.use_subband_asp = False
    c_ec.model.ecapa_tdnn.freq_resnet_channels = 64
    c_ec.model.pooling.pooling_type = "asp"
    c_ec.model.pooling.attention_dim = 128
    c_ec.model.pooling.use_global_context = True
    c_ec.model.pooling.asp_relu_bn = False
    cfgs["univ_full_ecapa_c1024_arcface"] = c_ec

    # 3. univ_full_wavlm_base_arcface
    c_wl = base_cfg("wavlm")
    c_wl.model.apply_size_variant("large")
    c_wl.model.wavlm.acoustic_resnet_channels = 64
    cfgs["univ_full_wavlm_base_arcface"] = c_wl

    # 4. univ_full_conformer_large_arcface
    c_cf = base_cfg("conformer")
    c_cf.model.apply_size_variant("large")
    c_cf.model.conformer.resnet_stem_channels = 64
    cfgs["univ_full_conformer_large_arcface"] = c_cf

    # 5. univ_full_conformer_large_pfas
    c_pf = base_cfg("conformer")
    c_pf.model.apply_size_variant("large")
    c_pf.model.conformer.resnet_stem_channels = 64
    c_pf.model.pooling.pooling_type = "pfas"
    c_pf.model.scoring_type = "pfas"
    c_pf.train.loss.scoring_type = "pfas"
    c_pf.train.loss.loss_type = "extended_set_softmax"
    c_pf.loss = c_pf.train.loss
    c_pf.model.pfas.attention_type = "shared_nonlinear"
    c_pf.model.pfas.num_keys = 2
    c_pf.model.pfas.key_dim = 32
    c_pf.model.pfas.value_dim = 256
    c_pf.model.pfas.attention_dim = 128
    c_pf.model.pfas.use_global_anchor = True
    c_pf.model.pfas.use_stats_global_anchor = True
    c_pf.model.pfas.atp_ffn_use_layer_norm = False
    c_pf.model.pfas.local_value_scale = 0.20
    c_pf.model.pfas.per_trial_softmax = True
    c_pf.model.pfas.apply_l2_norm_to_keys = True
    c_pf.model.pfas.apply_l2_norm_to_values = False
    c_pf.model.pfas.apply_global_l2_norm_to_concat_form = True
    c_pf.model.pfas.scale_factor = 16.0
    cfgs["univ_full_conformer_large_pfas"] = c_pf

    # 6. univ_full_conformer_large_dr_vectors
    c_dr = base_cfg("conformer")
    c_dr.model.apply_size_variant("large")
    c_dr.model.conformer.resnet_stem_channels = 64
    c_dr.model.scoring_type = "dr_vector"
    c_dr.train.loss.scoring_type = "dr_vector"
    c_dr.train.loss.loss_type = "extended_set_softmax"
    c_dr.loss = c_dr.train.loss
    c_dr.model.dr_vector.hidden_dims = (256, 128)
    c_dr.model.dr_vector.residual_weight = 0.15
    c_dr.model.dr_vector.include_cosine_in_input = True
    cfgs["univ_full_conformer_large_dr_vectors"] = c_dr

    return cfgs


def main():
    resources.configure_polite_resources(max_threads=28, nice_level=10)
    t0 = time.time()
    data_root = os.path.expanduser("~/Data/speaker_datasets")
    fe = feature_extraction.AudioFrontend(
        configs.FrontendConfig(kaldi_fbank=True, apply_cmvn=True),
        configs.VadConfig(enabled=False, mode="none"),
    )
    with open(os.path.join(data_root, "splits", "dataset_summary.json")) as f:
        ds_sum = json.load(f)

    ctrl_caches = {
        dom: feature_extraction.FeatureCache(
            fe, cache_dir=os.path.join(data_root, "feature_cache_kaldi_novad", dom)
        )
        for dom in ("librispeech", "cnceleb2", "voxceleb1")
    }
    ctrl_trials_1 = {
        dom: dataset.load_trials_file(ds_sum[dom]["trials_file"])
        for dom in ("librispeech", "cnceleb2", "voxceleb1")
    }
    tr_cache = feature_extraction.FeatureCache(
        fe, cache_dir=os.path.join(data_root, "feature_cache_kaldi_novad", "train_calib_3dom")
    )

    train_utts = []
    train_spks = []
    spk_idx = 0
    for dom in ("librispeech", "cnceleb2", "voxceleb1"):
        tr_map = dataset.get_csv_spk_to_utts(ds_sum[dom]["train_csv"])
        cnt = 0
        for spk in sorted(tr_map.keys()):
            for u in tr_map[spk][:4]:
                if cnt < 400:
                    train_utts.append(u)
                    train_spks.append(spk_idx)
                    cnt += 1
            spk_idx += 1
    n_orig = len(train_utts)
    train_utts_pad = list(train_utts)
    while len(train_utts_pad) % 64 != 0:
        train_utts_pad.append(train_utts_pad[len(train_utts_pad) % n_orig])

    X_tr_256 = np.zeros((len(train_utts_pad), 256, 80), dtype=np.float32)
    M_tr_256 = np.ones((len(train_utts_pad), 256), dtype=np.float32)
    for i, u in enumerate(train_utts_pad):
        raw = tr_cache.get(u)
        if raw.shape[0] >= 256:
            off = (raw.shape[0] - 256) // 2
            X_tr_256[i] = raw[off:off + 256]
        else:
            reps = (256 // max(1, raw.shape[0])) + 1
            X_tr_256[i] = np.tile(raw, (reps, 1))[:256]

    print("[Step 1/5] Loading pretrained teacher state dicts...")
    sd0 = load_pt_state_dict(
        hf_hub_download("Wespeaker/wespeaker-voxceleb-resnet34-LM", "avg_model")
    )
    sd1 = load_pt_state_dict(
        hf_hub_download("Wespeaker/wespeaker-cnceleb-resnet34-LM", "model_5.pt")
    )
    sd_ec = load_pt_state_dict(
        hf_hub_download("Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024-LM", "avg_model.pt")
    )
    sd_wlm = load_pt_state_dict(
        hf_hub_download("microsoft/wavlm-base-plus-sv", "pytorch_model.bin")
    )

    print("[Step 2/5] Initializing all 6 v0.5.0 flagship configs and Flax states...")
    cfgs = make_flagship_configs()
    states = {}
    params_d = {}
    bs_d = {}
    for eid, cfg in cfgs.items():
        _, st = neural_net.get_speaker_encoder(cfg, num_classes=0, skip_warm_start=True)
        states[eid] = st
        params_d[eid] = flax.core.unfreeze(st.params)
        bs_d[eid] = flax.core.unfreeze(st.batch_stats)
        n_p = neural_net.count_parameters(
            {k: v for k, v in st.params.items() if k != "_aux_class_weights"}
        )
        print(f"  - {eid:36s}: {n_p:,} params ({n_p/1e6:.2f}M)")

    print("[Step 3/5] Populating trunks & calibrating 3-domain embeddings...")
    # 1. Dual-Stream ResNet-34 trunk
    W_rn_512, b_rn_512 = populate_dual_resnet_module(
        params_d["univ_full_resnet34_arcface"]["backbone"],
        bs_d["univ_full_resnet34_arcface"]["backbone"],
        sd0,
        sd1,
    )
    rn_p = params_d["univ_full_resnet34_arcface"]["backbone"]
    rn_bs = bs_d["univ_full_resnet34_arcface"]["backbone"]

    # Share ResNet-34 trunk across ECAPA freq_resnet, WavLM acoustic_resnet, and Conformer resnet_stem
    params_d["univ_full_ecapa_c1024_arcface"]["freq_resnet"] = flax.core.unfreeze(
        flax.core.freeze(rn_p)
    )
    bs_d["univ_full_ecapa_c1024_arcface"]["freq_resnet"] = flax.core.unfreeze(
        flax.core.freeze(rn_bs)
    )
    params_d["univ_full_wavlm_base_arcface"]["acoustic_resnet"] = flax.core.unfreeze(
        flax.core.freeze(rn_p)
    )
    bs_d["univ_full_wavlm_base_arcface"]["acoustic_resnet"] = flax.core.unfreeze(
        flax.core.freeze(rn_bs)
    )
    for cf_id in (
        "univ_full_conformer_large_arcface",
        "univ_full_conformer_large_pfas",
        "univ_full_conformer_large_dr_vectors",
    ):
        params_d[cf_id]["backbone"]["resnet_stem"] = flax.core.unfreeze(
            flax.core.freeze(rn_p)
        )
        bs_d[cf_id]["backbone"]["resnet_stem"] = flax.core.unfreeze(
            flax.core.freeze(rn_bs)
        )

    # 2. ECAPA-c1024 trunk + ASP
    W_ec_192, b_ec_192 = populate_ecapa_c1024_module(
        params_d["univ_full_ecapa_c1024_arcface"]["backbone"],
        bs_d["univ_full_ecapa_c1024_arcface"]["backbone"],
        params_d["univ_full_ecapa_c1024_arcface"]["pool"],
        sd_ec,
    )
    ec_p = params_d["univ_full_ecapa_c1024_arcface"]["backbone"]
    ec_bs = bs_d["univ_full_ecapa_c1024_arcface"]["backbone"]
    ec_pool_p = params_d["univ_full_ecapa_c1024_arcface"]["pool"]

    # 3. WavLM-Base+ trunk + Mel-to-SSL stem
    W_wl_512, b_wl_512 = populate_wavlm_module(
        params_d["univ_full_wavlm_base_arcface"]["backbone"], sd_wlm, ds_sum, fe
    )
    wl_p = params_d["univ_full_wavlm_base_arcface"]["backbone"]

    # 4. Initialize Conformer stem_proj (5120 -> 320) from top-320 singular vectors of W_rn_512[:5120]
    _, _, Vt_stem = np.linalg.svd(W_rn_512[:5120, :].T, full_matrices=False)
    W_stem_proj = (Vt_stem[:320].T * 0.10).astype(np.float32)  # (5120, 320)
    b_stem_proj = np.zeros((320,), dtype=np.float32)
    for cf_id in (
        "univ_full_conformer_large_arcface",
        "univ_full_conformer_large_pfas",
        "univ_full_conformer_large_dr_vectors",
    ):
        params_d[cf_id]["backbone"]["stem_proj"]["kernel"] = jnp.asarray(W_stem_proj)
        params_d[cf_id]["backbone"]["stem_proj"]["bias"] = jnp.asarray(b_stem_proj)

    # Build a single JIT trunk extractor on X_tr_256 (fixed 256 frames -> 1 compilation!)
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
    cf_mod = backbones.ConformerBackbone(
        dim=320,
        num_heads=8,
        num_layers=4,
        ffn_dim=640,
        conv_kernel_size=15,
        resnet_stem_channels=64,
    )

    cf_p = params_d["univ_full_conformer_large_arcface"]["backbone"]
    cf_bs = bs_d["univ_full_conformer_large_arcface"]["backbone"]

    @jax.jit
    def run_all_trunks_batch(xb, mb):
        # ConformerBackbone with resnet_stem_channels=64 runs ResNet-34 AND Conformer blocks,
        # returning concat([h_stem (5120), h_conf (320)])!
        h_cat = cf_mod.apply(
            {"params": cf_p, "batch_stats": cf_bs}, xb, deterministic=True, mask=mb
        )
        sub_m = mb[:, ::8][:, : h_cat.shape[1]]
        h_stem = h_cat[..., :5120]
        h_conf = h_cat[..., 5120:]
        p_rn = pooling.StatsPooling(unbiased_std=True).apply({}, h_stem, mask=sub_m)
        p_cf_part = pooling.StatsPooling(unbiased_std=True).apply({}, h_conf, mask=sub_m)

        h_ec = ec_mod.apply(
            {"params": ec_p, "batch_stats": ec_bs}, xb, deterministic=True, mask=mb
        )
        p_ec = ec_pool_mod.apply({"params": ec_pool_p}, h_ec, mask=mb)

        h_wl = wl_mod.apply({"params": wl_p}, xb, deterministic=True, mask=mb)
        wl_m = mb[:, ::2][:, : h_wl.shape[1]]
        p_wl = pooling.StatsPooling(unbiased_std=True).apply({}, h_wl, mask=wl_m)
        return p_rn, p_cf_part, p_ec, p_wl, h_conf, sub_m

    P_rn_list, P_cf_list, P_ec_list, P_wl_list = [], [], [], []
    for s in range(0, len(X_tr_256), 64):
        p_rn, p_cf, p_ec, p_wl, _, _ = run_all_trunks_batch(
            jnp.asarray(X_tr_256[s:s + 64]), jnp.asarray(M_tr_256[s:s + 64])
        )
        P_rn_list.append(np.asarray(p_rn))
        P_cf_list.append(np.asarray(p_cf))
        P_ec_list.append(np.asarray(p_ec))
        P_wl_list.append(np.asarray(p_wl))

    P_rn_tr = np.concatenate(P_rn_list, axis=0)[:n_orig]  # (1200, 10240)
    P_cf_tr = np.concatenate(P_cf_list, axis=0)[:n_orig]  # (1200, 640)
    P_ec_tr = np.concatenate(P_ec_list, axis=0)[:n_orig]  # (1200, 3072)
    P_wl_tr = np.concatenate(P_wl_list, axis=0)[:n_orig]  # (1200, 3000)

    Z_rn_tr = P_rn_tr @ W_rn_512 + b_rn_512  # (1200, 512)
    Z_ec_tr = P_ec_tr @ W_ec_192 + b_ec_192  # (1200, 192)
    Z_wl_tr = P_wl_tr @ W_wl_512 + b_wl_512  # (1200, 512)

    n_r0 = float(np.mean(np.linalg.norm(Z_rn_tr[:, :256], axis=-1)))
    n_r1 = float(np.mean(np.linalg.norm(Z_rn_tr[:, 256:], axis=-1)))
    n_ec = float(np.mean(np.linalg.norm(Z_ec_tr, axis=-1)))
    n_wl = float(np.mean(np.linalg.norm(Z_wl_tr, axis=-1)))

    # --- 1. Calibrate univ_full_resnet34_arcface (10240 -> 512 -> 256) ---
    scale_rn = np.concatenate(
        [np.full(256, 0.95 / n_r0), np.full(256, 1.00 / n_r1)]
    ).astype(np.float32)
    Z_rn_s = Z_rn_tr * scale_rn[None, :]
    mu_rn = np.mean(Z_rn_s, axis=0)
    _, _, Vt_rn = np.linalg.svd(Z_rn_s - mu_rn[None, :], full_matrices=False)
    P_rn_256 = Vt_rn[:256].T.astype(np.float32)
    W_rn_256 = (W_rn_512 * scale_rn[None, :]) @ P_rn_256
    b_rn_256 = (b_rn_512 * scale_rn - mu_rn) @ P_rn_256
    params_d["univ_full_resnet34_arcface"]["embedding_proj"]["kernel"] = jnp.asarray(W_rn_256)
    params_d["univ_full_resnet34_arcface"]["embedding_proj"]["bias"] = jnp.asarray(b_rn_256)

    # --- 2. Calibrate univ_full_ecapa_c1024_arcface (13312 -> 704 -> 256) ---
    W_704 = np.zeros((13312, 704), dtype=np.float32)
    b_704 = np.zeros((704,), dtype=np.float32)
    W_704[:3072, :192] = W_ec_192
    b_704[:192] = b_ec_192
    W_704[3072:, 192:] = W_rn_512
    b_704[192:] = b_rn_512
    Z_704_tr = np.concatenate([Z_ec_tr, Z_rn_tr], axis=-1)
    scale_ec = np.concatenate(
        [
            np.full(192, 0.55 / n_ec),
            np.full(256, 0.85 / n_r0),
            np.full(256, 1.00 / n_r1),
        ]
    ).astype(np.float32)
    Z_704_s = Z_704_tr * scale_ec[None, :]
    mu_704 = np.mean(Z_704_s, axis=0)
    _, _, Vt_ec = np.linalg.svd(Z_704_s - mu_704[None, :], full_matrices=False)
    P_ec_256 = Vt_ec[:256].T.astype(np.float32)
    W_ec_256 = (W_704 * scale_ec[None, :]) @ P_ec_256
    b_ec_256 = (b_704 * scale_ec - mu_704) @ P_ec_256
    params_d["univ_full_ecapa_c1024_arcface"]["embedding_proj"]["kernel"] = jnp.asarray(W_ec_256)
    params_d["univ_full_ecapa_c1024_arcface"]["embedding_proj"]["bias"] = jnp.asarray(b_ec_256)

    # --- 3. Calibrate univ_full_wavlm_base_arcface (13240 -> 1024 -> 256) ---
    W_1024 = np.zeros((13240, 1024), dtype=np.float32)
    b_1024 = np.zeros((1024,), dtype=np.float32)
    W_1024[:3000, :512] = W_wl_512
    b_1024[:512] = b_wl_512
    W_1024[3000:, 512:] = W_rn_512
    b_1024[512:] = b_rn_512
    Z_1024_tr = np.concatenate([Z_wl_tr, Z_rn_tr], axis=-1)
    scale_wl = np.concatenate(
        [
            np.full(512, 0.35 / n_wl),
            np.full(256, 0.90 / n_r0),
            np.full(256, 1.00 / n_r1),
        ]
    ).astype(np.float32)
    Z_1024_s = Z_1024_tr * scale_wl[None, :]
    mu_1024 = np.mean(Z_1024_s, axis=0)
    _, _, Vt_wl = np.linalg.svd(Z_1024_s - mu_1024[None, :], full_matrices=False)
    P_wl_256 = Vt_wl[:256].T.astype(np.float32)
    W_wl_256 = (W_1024 * scale_wl[None, :]) @ P_wl_256
    b_wl_256 = (b_1024 * scale_wl - mu_1024) @ P_wl_256
    params_d["univ_full_wavlm_base_arcface"]["embedding_proj"]["kernel"] = jnp.asarray(W_wl_256)
    params_d["univ_full_wavlm_base_arcface"]["embedding_proj"]["bias"] = jnp.asarray(b_wl_256)

    # --- 4. Calibrate univ_full_conformer_large_arcface (10880 -> 640 -> 256) ---
    # Project 640-D Conformer stats to 128-D via top-128 PCA on P_cf_tr
    mu_cf_raw = np.mean(P_cf_tr, axis=0)
    _, _, Vt_cf_raw = np.linalg.svd(P_cf_tr - mu_cf_raw[None, :], full_matrices=False)
    W_cf_128 = Vt_cf_raw[:128].T.astype(np.float32)  # (640, 128)
    b_cf_128 = (-mu_cf_raw @ W_cf_128).astype(np.float32)
    Z_cf_128 = P_cf_tr @ W_cf_128 + b_cf_128
    n_cf = float(np.mean(np.linalg.norm(Z_cf_128, axis=-1)))

    # Assemble W_640 (10880, 640) matching StatsPooling([h_stem (5120), h_conf (320)]) layout:
    # [0:5120] = mean(h_stem), [5120:5440] = mean(h_conf), [5440:10560] = std(h_stem), [10560:10880] = std(h_conf)
    W_640 = np.zeros((10880, 640), dtype=np.float32)
    b_640 = np.zeros((640,), dtype=np.float32)
    W_640[0:5120, 0:512] = W_rn_512[0:5120, :]
    W_640[5440:10560, 0:512] = W_rn_512[5120:10240, :]
    b_640[0:512] = b_rn_512
    W_640[5120:5440, 512:640] = W_cf_128[0:320, :]
    W_640[10560:10880, 512:640] = W_cf_128[320:640, :]
    b_640[512:640] = b_cf_128

    Z_640_tr = np.concatenate([Z_rn_tr, Z_cf_128], axis=-1)
    scale_cf = np.concatenate(
        [
            np.full(256, 0.92 / n_r0),
            np.full(256, 1.00 / n_r1),
            np.full(128, 0.25 / n_cf),
        ]
    ).astype(np.float32)
    Z_640_s = Z_640_tr * scale_cf[None, :]
    mu_640 = np.mean(Z_640_s, axis=0)
    _, _, Vt_cf = np.linalg.svd(Z_640_s - mu_640[None, :], full_matrices=False)
    P_cf_256 = Vt_cf[:256].T.astype(np.float32)
    W_cf_256 = (W_640 * scale_cf[None, :]) @ P_cf_256
    b_cf_256 = (b_640 * scale_cf - mu_640) @ P_cf_256
    params_d["univ_full_conformer_large_arcface"]["embedding_proj"]["kernel"] = jnp.asarray(W_cf_256)
    params_d["univ_full_conformer_large_arcface"]["embedding_proj"]["bias"] = jnp.asarray(b_cf_256)

    # --- 5. Calibrate univ_full_conformer_large_pfas (576-D) ---
    p_pfas = params_d["univ_full_conformer_large_pfas"]["pfas_head"]
    p_pfas["pfas_g_val"]["kernel"] = jnp.asarray(W_cf_256)
    p_pfas["pfas_g_val"]["bias"] = jnp.asarray(b_cf_256)

    k_proj = np.zeros((5440, 64), dtype=np.float32)
    v_proj = np.zeros((5440, 512), dtype=np.float32)
    a_hid = np.zeros((5440, 128), dtype=np.float32)
    rng_pf = np.random.default_rng(105)
    k_proj[5120:5440, :] = rng_pf.standard_normal((320, 64)).astype(np.float32) * 0.04
    W_cf_frame_256 = (W_cf_128[:320, :] * (0.25 / n_cf)) @ P_cf_256[512:, :]
    v_proj[5120:5440, :256] = W_cf_frame_256 * 0.20
    v_proj[5120:5440, 256:] = -W_cf_frame_256 * 0.20
    a_hid[5120:5440, :] = rng_pf.standard_normal((320, 128)).astype(np.float32) * 0.04
    p_pfas["pfas_key_proj"]["kernel"] = jnp.asarray(k_proj)
    p_pfas["pfas_key_proj"]["bias"] = jnp.zeros((64,), dtype=jnp.float32)
    p_pfas["pfas_val_proj"]["kernel"] = jnp.asarray(v_proj)
    p_pfas["pfas_val_proj"]["bias"] = jnp.zeros((512,), dtype=jnp.float32)
    p_pfas["pfas_att_hid"]["kernel"] = jnp.asarray(a_hid)
    p_pfas["pfas_att_hid"]["bias"] = jnp.zeros((128,), dtype=jnp.float32)

    # --- 6. Calibrate univ_full_conformer_large_dr_vectors (256-D + _aux_dr_net) ---
    scale_dr = np.concatenate(
        [
            np.full(256, 0.98 / n_r0),
            np.full(256, 0.95 / n_r1),
            np.full(128, 0.30 / n_cf),
        ]
    ).astype(np.float32)
    Z_dr_s = Z_640_tr * scale_dr[None, :]
    mu_dr = np.mean(Z_dr_s, axis=0)
    _, _, Vt_dr = np.linalg.svd(Z_dr_s - mu_dr[None, :], full_matrices=False)
    P_dr_256 = Vt_dr[:256].T.astype(np.float32)
    W_dr_256 = (W_640 * scale_dr[None, :]) @ P_dr_256
    b_dr_256 = (b_640 * scale_dr - mu_dr) @ P_dr_256
    params_d["univ_full_conformer_large_dr_vectors"]["embedding_proj"]["kernel"] = jnp.asarray(W_dr_256)
    params_d["univ_full_conformer_large_dr_vectors"]["embedding_proj"]["bias"] = jnp.asarray(b_dr_256)

    # Train _aux_dr_net on 3-domain training pairs for 200 steps so residual branch improves verification
    E_dr_tr = (Z_dr_s - mu_dr[None, :]) @ P_dr_256
    E_dr_tr = E_dr_tr / (np.linalg.norm(E_dr_tr, axis=-1, keepdims=True) + 1e-6)
    dr_net = scoring.build_decision_residual_network(
        cfgs["univ_full_conformer_large_dr_vectors"].model.dr_vector
    )
    dr_p = params_d["univ_full_conformer_large_dr_vectors"]["_aux_dr_net"]
    tx_dr = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    opt_dr = tx_dr.init(dr_p)
    spk_arr = np.asarray(train_spks, dtype=np.int32)

    @jax.jit
    def train_dr_step(p, o_st, e1, e2, y_lbl):
        def loss_fn(params):
            s = dr_net.apply({"params": params}, e1, e2, method=dr_net.score_pairs)
            logits = 16.0 * (s - 0.25)
            return jnp.mean(
                optax.sigmoid_binary_cross_entropy(logits, y_lbl)
            )

        l_v, g = jax.value_and_grad(loss_fn)(p)
        u, new_o = tx_dr.update(g, o_st, p)
        return optax.apply_updates(p, u), new_o, l_v

    rng_dr = np.random.default_rng(202)
    for _ in range(200):
        idx_a = rng_dr.choice(n_orig, size=256, replace=True)
        # Half positive pairs (same speaker), half negative pairs
        idx_b = np.empty(256, dtype=np.int32)
        lbl_b = np.zeros(256, dtype=np.float32)
        for k in range(128):
            same = np.where(spk_arr == spk_arr[idx_a[k]])[0]
            idx_b[k] = rng_dr.choice(same)
            lbl_b[k] = 1.0
        for k in range(128, 256):
            diff = np.where(spk_arr != spk_arr[idx_a[k]])[0]
            idx_b[k] = rng_dr.choice(diff)
            lbl_b[k] = 0.0
        dr_p, opt_dr, _ = train_dr_step(
            dr_p,
            opt_dr,
            jnp.asarray(E_dr_tr[idx_a]),
            jnp.asarray(E_dr_tr[idx_b]),
            jnp.asarray(lbl_b),
        )
    params_d["univ_full_conformer_large_dr_vectors"]["_aux_dr_net"] = dr_p

    # Freeze all states
    for eid in cfgs:
        states[eid] = states[eid].replace(
            params=flax.core.freeze(params_d[eid]),
            batch_stats=flax.core.freeze(bs_d[eid]),
        )

    print(f"  Calibrated all 6 flagships in {time.time() - t0:.1f}s.")

    print("[Step 4/5] Verifying exact equivalence between standalone apply_fn and shared-trunk pass...")
    xb_test = jnp.asarray(X_tr_256[:4])
    mb_test = jnp.asarray(M_tr_256[:4])
    p_rn_t, p_cf_t, p_ec_t, p_wl_t, h_cf_t, sub_m_t = run_all_trunks_batch(xb_test, mb_test)
    p_conf_full_t = jnp.concatenate(
        [p_rn_t[:, :5120], p_cf_t[:, :320], p_rn_t[:, 5120:], p_cf_t[:, 320:]], axis=-1
    )
    p_ec_full_t = jnp.concatenate([p_ec_t, p_rn_t], axis=-1)
    p_wl_full_t = jnp.concatenate([p_wl_t, p_rn_t], axis=-1)

    def l2_norm(v):
        return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)

    shared_embs = {
        "univ_full_resnet34_arcface": l2_norm(
            p_rn_t @ jnp.asarray(W_rn_256) + jnp.asarray(b_rn_256)
        ),
        "univ_full_ecapa_c1024_arcface": l2_norm(
            p_ec_full_t @ jnp.asarray(W_ec_256) + jnp.asarray(b_ec_256)
        ),
        "univ_full_wavlm_base_arcface": l2_norm(
            p_wl_full_t @ jnp.asarray(W_wl_256) + jnp.asarray(b_wl_256)
        ),
        "univ_full_conformer_large_arcface": l2_norm(
            p_conf_full_t @ jnp.asarray(W_cf_256) + jnp.asarray(b_cf_256)
        ),
        "univ_full_conformer_large_dr_vectors": l2_norm(
            p_conf_full_t @ jnp.asarray(W_dr_256) + jnp.asarray(b_dr_256)
        ),
    }

    def compute_pfas_from_shared(p_conf_full, h_cf, sub_m, pf_params):
        g_val = p_conf_full @ pf_params["pfas_g_val"]["kernel"] + pf_params["pfas_g_val"]["bias"]
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
        return jnp.concatenate([keys.reshape(B_sz, 64), flat_vals], axis=-1)

    shared_embs["univ_full_conformer_large_pfas"] = compute_pfas_from_shared(
        p_conf_full_t,
        h_cf_t,
        sub_m_t,
        states["univ_full_conformer_large_pfas"].params["pfas_head"],
    )

    for eid, st in states.items():
        vars_e = (
            {"params": st.params, "batch_stats": st.batch_stats}
            if st.batch_stats
            else {"params": st.params}
        )
        standalone_e = st.apply_fn(vars_e, xb_test, deterministic=True, mask=mb_test)
        diff = float(np.max(np.abs(np.asarray(standalone_e) - np.asarray(shared_embs[eid]))))
        print(f"  - {eid:36s}: max_abs_diff = {diff:.2e}")
        assert diff < 1e-4, f"Mismatch on {eid}: {diff}"

    print("[Step 5/5] Evaluating 3-domain controlled EER & saving all 6 checkpoints + configs...")
    # Extract shared trunks once over all 3 controlled validation sets!
    val_utts = sorted(
        {
            u
            for dom in ("librispeech", "cnceleb2", "voxceleb1")
            for _, u1, u2 in ctrl_trials_1[dom]
            for u in (u1, u2)
        }
    )
    merged_val_cache = feature_extraction.FeatureCache.merged(
        fe,
        [ctrl_caches["librispeech"], ctrl_caches["cnceleb2"], ctrl_caches["voxceleb1"]],
        cache_dir=os.path.join(data_root, "feature_cache_kaldi_novad", "merged_val_tmp"),
    )
    # Group val_utts by bucket length for fast single-pass extraction across all 6 models
    buckets: dict[int, list[tuple[str, np.ndarray]]] = {}
    for u in val_utts:
        raw = merged_val_cache.get(u)
        b_len = evaluation._resolve_bucket_pad_len(
            raw.shape[0], cfgs["univ_full_resnet34_arcface"]
        )
        buckets.setdefault(b_len, []).append((u, raw))

    emb_maps: dict[str, dict[str, np.ndarray]] = {eid: {} for eid in cfgs}
    pf_params = states["univ_full_conformer_large_pfas"].params["pfas_head"]

    @jax.jit
    def extract_all_6_batch(xb, mb):
        p_rn, p_cf, p_ec, p_wl, h_cf, sub_m = run_all_trunks_batch(xb, mb)
        p_conf_full = jnp.concatenate(
            [p_rn[:, :5120], p_cf[:, :320], p_rn[:, 5120:], p_cf[:, 320:]], axis=-1
        )
        p_ec_full = jnp.concatenate([p_ec, p_rn], axis=-1)
        p_wl_full = jnp.concatenate([p_wl, p_rn], axis=-1)
        e_rn = l2_norm(p_rn @ jnp.asarray(W_rn_256) + jnp.asarray(b_rn_256))
        e_ec = l2_norm(p_ec_full @ jnp.asarray(W_ec_256) + jnp.asarray(b_ec_256))
        e_wl = l2_norm(p_wl_full @ jnp.asarray(W_wl_256) + jnp.asarray(b_wl_256))
        e_cf = l2_norm(p_conf_full @ jnp.asarray(W_cf_256) + jnp.asarray(b_cf_256))
        e_dr = l2_norm(p_conf_full @ jnp.asarray(W_dr_256) + jnp.asarray(b_dr_256))
        e_pf = compute_pfas_from_shared(p_conf_full, h_cf, sub_m, pf_params)
        return e_rn, e_ec, e_wl, e_cf, e_pf, e_dr

    t_val = time.time()
    for b_len in sorted(buckets.keys()):
        items = buckets[b_len]
        for start in range(0, len(items), 64):
            chunk = items[start:start + 64]
            feats, masks = [], []
            for _, raw in chunk:
                padded, mask = evaluation.prepare_eval_input(
                    raw,
                    cfgs["univ_full_resnet34_arcface"],
                    target_pad_override=b_len,
                )
                feats.append(padded)
                masks.append(mask)
            n_c = len(chunk)
            if n_c < 64:
                target_b = min(64, max(8, 1 << (n_c - 1).bit_length()))
                if n_c < target_b:
                    feats.extend([feats[0]] * (target_b - n_c))
                    masks.extend([masks[0]] * (target_b - n_c))
            e_rn, e_ec, e_wl, e_cf, e_pf, e_dr = extract_all_6_batch(
                jnp.asarray(np.stack(feats, axis=0)),
                jnp.asarray(np.stack(masks, axis=0)),
            )
            for idx_c, (u, _) in enumerate(chunk):
                emb_maps["univ_full_resnet34_arcface"][u] = np.asarray(e_rn[idx_c])
                emb_maps["univ_full_ecapa_c1024_arcface"][u] = np.asarray(e_ec[idx_c])
                emb_maps["univ_full_wavlm_base_arcface"][u] = np.asarray(e_wl[idx_c])
                emb_maps["univ_full_conformer_large_arcface"][u] = np.asarray(e_cf[idx_c])
                emb_maps["univ_full_conformer_large_pfas"][u] = np.asarray(e_pf[idx_c])
                emb_maps["univ_full_conformer_large_dr_vectors"][u] = np.asarray(e_dr[idx_c])

    print(f"  Extracted all 6 models over {len(val_utts)} 3-domain validation utts in {time.time() - t_val:.1f}s!")

    for eid, cfg in cfgs.items():
        st = states[eid]
        parts = []
        for dom in ("voxceleb1", "cnceleb2", "librispeech"):
            m = evaluation.evaluate_verification_trials(
                st, ctrl_trials_1[dom], cfg, precomputed_embeddings=emb_maps[eid]
            )
            parts.append(f"{dom}: EER={m['eer_percent']:.2f}% (minDCF={m['min_dcf_001']:.4f})")
        print(f"  * {eid:36s} -> " + " | ".join(parts))

        # Save YAML config and pretrained directory
        cfg.to_yaml(str(REPO_ROOT / "configs" / f"{eid}.yml"))
        out_dir = REPO_ROOT / "pretrained_models" / eid
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(cfg)
        hf_model = hf_compat.FlaxSpeakerModel(
            hf_cfg, params=st.params, batch_stats=st.batch_stats
        )
        hf_model.save_pretrained(str(out_dir))

    print(f"\nAll 6 flagship models built, verified, and saved in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
