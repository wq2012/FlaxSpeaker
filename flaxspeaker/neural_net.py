"""Neural network models, training state, and end-to-end training loop for FlaxSpeaker v0.5.0.

Provides `ModernSpeakerEncoder` construction across all 6 neural backbones
(Conformer, ECAPA-TDNN with mask-aware BatchNorm, Mamba, ResNet, Transformer, LSTM),
`SpeakerTrainState` with `batch_stats` tracking, `AdamW` and `SGD + Nesterov`
optimizers with trunk warm-starting / differential learning rates, `Sub-Center ArcFace`
(`K=3`), `GE2E`, `Extended-Set Softmax`, `PFAS`, `Dr-Vectors`, `float16` memmap shard
training, and Stage-2 Large-Margin Fine-Tuning (`LMFT`).
"""

from functools import partial
import multiprocessing
import os
import random
import sys
import time
from typing import Any, Optional

import flax
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from flaxspeaker import backbones
from flaxspeaker import configs
from flaxspeaker import dataset
from flaxspeaker import feature_extraction
from flaxspeaker import losses
from flaxspeaker import scoring
from flaxspeaker import shard_cache


class SpeakerTrainState(train_state.TrainState):
    """Flax `TrainState` extended with `batch_stats` for BatchNorm backbones (`ECAPA-TDNN`)."""

    batch_stats: dict[str, Any] = flax.struct.field(default_factory=dict)


# Convenience re-exports
cosine_similarity = scoring.cosine_similarity
get_triplet_loss = losses.get_triplet_loss
get_triplet_loss_from_batch_output = losses.get_triplet_loss_from_batch_output


def count_parameters(params: Any) -> int:
    """Count total number of scalar parameters in a Flax parameter tree."""
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(int(np.prod(x.shape)) for x in leaves if hasattr(x, "shape")))


def save_model(
    saved_model_path: str,
    params: Any,
    batch_stats: Optional[dict[str, Any]] = None,
) -> None:
    """Save model parameters (and optional `batch_stats`) in SafeTensors or Flax msgpack format."""
    parent = os.path.dirname(os.path.abspath(saved_model_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    raw_params = (
        params["params"]
        if (isinstance(params, dict) and "params" in params and len(params) <= 2)
        else params
    )
    raw_bs = (
        batch_stats
        if batch_stats is not None
        else (
            params.get("batch_stats")
            if isinstance(params, dict) and "params" in params
            else None
        )
    )

    if saved_model_path.endswith(".safetensors"):
        from safetensors.flax import save_file

        flat = flax.traverse_util.flatten_dict(raw_params, sep=".")
        flat_arrays = {k: jnp.asarray(v) for k, v in flat.items()}
        if raw_bs:
            flat_bs = flax.traverse_util.flatten_dict(raw_bs, sep=".")
            for k, v in flat_bs.items():
                flat_arrays[f"_batch_stats.{k}"] = jnp.asarray(v)
        save_file(flat_arrays, saved_model_path)
        print("Model saved to: ", saved_model_path)
        return

    if not saved_model_path.endswith(".msgpack"):
        saved_model_path += ".msgpack"
    payload = (
        {"params": raw_params, "batch_stats": raw_bs}
        if raw_bs
        else (params if (isinstance(params, dict) and "params" in params) else {"params": raw_params})
    )
    bytes_output = flax.serialization.to_bytes(payload)
    with open(saved_model_path, "wb") as f:
        f.write(bytes_output)
    print("Model saved to: ", saved_model_path)


def load_model(
    saved_model_path: str,
    params: Any,
) -> Any:
    """Load model parameters (and optional `batch_stats`) from SafeTensors or Flax msgpack."""
    if saved_model_path.endswith(".safetensors"):
        from safetensors.flax import load_file

        flat_arrays = load_file(saved_model_path)
        flat_p = {
            k: v for k, v in flat_arrays.items() if not k.startswith("_batch_stats.")
        }
        flat_bs = {
            k.removeprefix("_batch_stats."): v
            for k, v in flat_arrays.items()
            if k.startswith("_batch_stats.")
        }
        unflat_p = flax.traverse_util.unflatten_dict(flat_p, sep=".")
        print("Model loaded from:", saved_model_path)
        if flat_bs:
            unflat_bs = flax.traverse_util.unflatten_dict(flat_bs, sep=".")
            return {"params": unflat_p, "batch_stats": unflat_bs}
        return unflat_p

    if not os.path.exists(saved_model_path) and os.path.exists(
        saved_model_path + ".msgpack"
    ):
        saved_model_path = saved_model_path + ".msgpack"
    with open(saved_model_path, "rb") as f:
        bytes_output = f.read()
    print("Model loaded from:", saved_model_path)
    if params is None:
        return flax.serialization.msgpack_restore(bytes_output)
    try:
        return flax.serialization.from_bytes(params, bytes_output)
    except ValueError:
        return flax.serialization.msgpack_restore(bytes_output)


def build_modern_encoder_from_config(
    myconfig: Any,
) -> backbones.ModernSpeakerEncoder:
    """Construct `ModernSpeakerEncoder` from an `ExperimentConfig`, dict, or YAML path."""
    exp_cfg = configs.ExperimentConfig.from_any(myconfig)

    m_cfg = exp_cfg.model
    b_type = str(getattr(m_cfg.backbone, "value", m_cfg.backbone))
    p_type = str(getattr(m_cfg.pooling.pooling_type, "value", m_cfg.pooling.pooling_type))
    scoring_type = str(
        getattr(exp_cfg.train.loss.scoring_type, "value", exp_cfg.train.loss.scoring_type)
    )
    if scoring_type == "pfas":
        p_type = "pfas"

    if b_type == "lstm":
        b_kwargs = {
            "hidden_size": m_cfg.lstm.hidden_size,
            "num_layers": m_cfg.lstm.num_layers,
            "bidirectional": m_cfg.lstm.bidirectional,
            "proj_size": m_cfg.lstm.proj_size,
        }
    elif b_type == "transformer":
        b_kwargs = {
            "dim": m_cfg.transformer.dim,
            "num_heads": m_cfg.transformer.num_heads,
            "num_layers": m_cfg.transformer.num_encoder_layers,
            "mlp_dim": m_cfg.transformer.mlp_dim,
            "use_positional_encoding": m_cfg.transformer.use_positional_encoding,
            "use_conv_stem": getattr(m_cfg.transformer, "use_conv_stem", True),
        }
    elif b_type == "conformer":
        b_kwargs = {
            "dim": m_cfg.conformer.dim,
            "num_heads": m_cfg.conformer.num_heads,
            "num_layers": m_cfg.conformer.num_layers,
            "ffn_dim": m_cfg.conformer.ffn_dim,
            "conv_kernel_size": m_cfg.conformer.conv_kernel_size,
            "use_positional_encoding": getattr(
                m_cfg.conformer, "use_positional_encoding", False
            ),
            "normalize_output": getattr(
                m_cfg.conformer, "normalize_output", False
            ),
            "use_conv_stem": getattr(m_cfg.conformer, "use_conv_stem", True),
            "use_group_norm": getattr(m_cfg.conformer, "use_group_norm", True),
            "layer_order": str(getattr(m_cfg.conformer, "layer_order", "mhsa_before_conv")),
            "subsample_factor": int(getattr(m_cfg.conformer, "subsample_factor", 1)),
            "subsample_after_layer": int(
                getattr(m_cfg.conformer, "subsample_after_layer", 3)
            ),
        }
    elif b_type == "mamba":
        b_kwargs = {
            "dim": m_cfg.mamba.dim,
            "num_layers": m_cfg.mamba.num_layers,
            "d_state": m_cfg.mamba.d_state,
            "d_conv": m_cfg.mamba.d_conv,
            "expand_factor": m_cfg.mamba.expand_factor,
            "dt_rank": m_cfg.mamba.dt_rank,
            "bidirectional": bool(getattr(m_cfg.mamba, "bidirectional", False)),
        }
    elif b_type == "ecapa_tdnn":
        b_kwargs = {
            "channels": m_cfg.ecapa_tdnn.channels,
            "scale": m_cfg.ecapa_tdnn.scale,
            "kernel_sizes": tuple(m_cfg.ecapa_tdnn.kernel_sizes),
            "dilations": tuple(m_cfg.ecapa_tdnn.dilations),
            "mfa_dim": m_cfg.ecapa_tdnn.mfa_dim,
            "se_bottleneck_dim": m_cfg.ecapa_tdnn.se_bottleneck_dim,
            "use_batch_norm": bool(
                getattr(m_cfg.ecapa_tdnn, "use_batch_norm", True)
            ),
            "norm_after_activation": bool(
                getattr(m_cfg.ecapa_tdnn, "norm_after_activation", True)
            ),
        }
    elif b_type == "resnet":
        b_kwargs = {
            "base_channels": m_cfg.resnet.base_channels,
            "num_blocks": tuple(m_cfg.resnet.num_blocks),
            "use_se": m_cfg.resnet.use_se,
            "out_dim": m_cfg.resnet.out_dim,
            "use_batch_norm": bool(
                getattr(m_cfg.resnet, "use_batch_norm", True)
            ),
        }
    else:
        raise ValueError(f"Unknown backbone: {b_type}")

    if p_type == "pfas":
        p_kwargs = {
            "num_keys": m_cfg.pfas.num_keys,
            "key_dim": m_cfg.pfas.key_dim,
            "value_dim": m_cfg.pfas.value_dim,
            "attention_dim": m_cfg.pfas.attention_dim,
            "bottleneck_dim": getattr(m_cfg.pooling, "bottleneck_dim", 256),
            "atp_num_heads": getattr(m_cfg.pooling, "num_heads", 4),
            "attention_type": m_cfg.pfas.attention_type,
            "atp_ffn_use_layer_norm": bool(
                getattr(m_cfg.pfas, "atp_ffn_use_layer_norm", True)
            ),
            "atp_attention_dim": int(getattr(m_cfg.pooling, "attention_dim", 128)),
            "atp_subspace_heads": bool(getattr(m_cfg.pooling, "subspace_heads", True)),
            "apply_l2_norm_to_keys": getattr(
                m_cfg.pfas, "apply_l2_norm_to_keys", True
            ),
            "apply_l2_norm_to_values": getattr(
                m_cfg.pfas, "apply_l2_norm_to_values", False
            ),
            "apply_global_l2_norm_to_concat_form": getattr(
                m_cfg.pfas, "apply_global_l2_norm_to_concat_form", True
            ),
            "use_global_anchor": getattr(
                m_cfg.pfas, "use_global_anchor", True
            ),
            "local_value_scale": float(
                getattr(m_cfg.pfas, "local_value_scale", 0.25)
            ),
            "normalize_keys_and_values": m_cfg.pfas.normalize_keys_and_values,
            "normalize_overall_output": m_cfg.pfas.normalize_overall_output,
        }
    else:
        stats_str = str(
            getattr(
                m_cfg.pooling,
                "pooling_stats",
                getattr(m_cfg.pooling, "stats_type", "MEAN_STD"),
            )
        ).lower()
        eps_val = float(
            getattr(
                m_cfg.pooling,
                "eps",
                getattr(m_cfg.pooling, "epsilon", 1e-4),
            )
        )
        p_kwargs = {
            "attention_dim": m_cfg.pooling.attention_dim,
            "num_heads": getattr(m_cfg.pooling, "num_heads", 4),
            "subspace_heads": getattr(m_cfg.pooling, "subspace_heads", True),
            "pooling_stats": stats_str,
            "eps": eps_val,
            "use_global_context": m_cfg.pooling.use_global_context,
            "bottleneck_dim": getattr(m_cfg.pooling, "bottleneck_dim", 256),
            "use_nonlinear_bottleneck": getattr(
                m_cfg.pooling, "use_nonlinear_bottleneck", True
            ),
        }

    return backbones.ModernSpeakerEncoder(
        backbone_type=b_type,
        pooling_type=p_type,
        embedding_dim=m_cfg.embedding_dim,
        normalize_embedding=m_cfg.normalize_embedding,
        backbone_kwargs=b_kwargs,
        pooling_kwargs=p_kwargs,
    )


def _get_feature_dim(myconfig: Any) -> int:
    """Resolve input feature dimension from config."""
    exp_cfg = configs.ExperimentConfig.from_any(myconfig)
    return exp_cfg.frontend.feature_dim


def build_optimizer(myconfig: Any) -> optax.GradientTransformation:
    """Create Optax optimizer (`adamw`, `sgd` with Nesterov, or `adam`) + optional trunk freezing/scaling."""
    train_cfg = getattr(myconfig, "train", myconfig)
    lr = float(getattr(train_cfg, "learning_rate", 1e-3))
    min_lr = float(getattr(train_cfg, "min_learning_rate", lr * 0.01))
    end_lr = min(lr, max(min_lr, 0.0))
    opt_name = str(getattr(train_cfg, "optimizer", "adamw")).lower()
    weight_decay = float(getattr(train_cfg, "weight_decay", 0.0))
    sgd_momentum = float(getattr(train_cfg, "sgd_momentum", 0.9))
    nesterov = bool(getattr(train_cfg, "nesterov", True))
    grad_clip = float(getattr(train_cfg, "grad_clip_norm", 0.0))
    use_cosine = bool(getattr(train_cfg, "lr_schedule_cosine", False))
    num_steps = int(getattr(train_cfg, "num_steps", 500))
    warmup_steps = min(
        int(getattr(train_cfg, "warmup_steps", 0)), max(0, num_steps - 1)
    )
    freeze_trunk = bool(getattr(train_cfg, "freeze_trunk", False))
    trunk_lr_mult = float(getattr(train_cfg, "trunk_lr_mult", 1.0))

    def _make_schedule(scale_mult: float = 1.0):
        s_lr = lr * scale_mult
        s_end = end_lr * scale_mult
        if use_cosine and num_steps > 2:
            if warmup_steps > 0:
                return optax.warmup_cosine_decay_schedule(
                    init_value=s_lr * 0.1,
                    peak_value=s_lr,
                    warmup_steps=warmup_steps,
                    decay_steps=num_steps,
                    end_value=s_end,
                )
            return optax.cosine_decay_schedule(
                init_value=s_lr,
                decay_steps=num_steps,
                alpha=s_end / max(s_lr, 1e-12),
            )
        return s_lr

    decay_mask = lambda params: jax.tree_util.tree_map(  # noqa: E731
        lambda p: getattr(p, "ndim", 0) >= 2, params
    )

    def _make_single_tx(scale_mult: float = 1.0) -> optax.GradientTransformation:
        sched = _make_schedule(scale_mult)
        transforms: list[optax.GradientTransformation] = []
        if grad_clip > 0:
            transforms.append(optax.clip_by_global_norm(grad_clip))
        if opt_name == "adamw":
            transforms.append(
                optax.adamw(
                    learning_rate=sched,
                    weight_decay=weight_decay,
                    mask=decay_mask,
                )
            )
        elif opt_name == "sgd":
            if weight_decay > 0.0:
                transforms.append(
                    optax.add_decayed_weights(weight_decay, mask=decay_mask)
                )
            transforms.append(
                optax.sgd(
                    learning_rate=sched,
                    momentum=sgd_momentum if sgd_momentum > 0.0 else None,
                    nesterov=nesterov and sgd_momentum > 0.0,
                )
            )
        else:
            transforms.append(optax.adam(learning_rate=sched))
        return optax.chain(*transforms) if len(transforms) > 1 else transforms[0]

    head_tx = _make_single_tx(1.0)
    if not freeze_trunk and abs(trunk_lr_mult - 1.0) < 1e-8:
        return head_tx

    trunk_tx = (
        optax.set_to_zero()
        if freeze_trunk
        else _make_single_tx(max(0.0, trunk_lr_mult))
    )

    def _partition_labels(params: dict[str, Any]) -> dict[str, Any]:
        labels: dict[str, Any] = {}
        for k, subtree in params.items():
            tag = "trunk" if k == "backbone" else "head"
            labels[k] = jax.tree_util.tree_map(lambda _: tag, subtree)
        return labels

    return optax.multi_transform(
        {"trunk": trunk_tx, "head": head_tx}, _partition_labels
    )


def warm_start_from_checkpoint(
    state: SpeakerTrainState, ckpt_path: str
) -> SpeakerTrainState:
    """Warm-starts matching-shape encoder/backbone parameters (and `batch_stats`) from `ckpt_path`.

    Incompatible head or auxiliary loss parameters (e.g., `_aux_class_weights` vs `pfas_head`
    or `_aux_dr_net`) keep their freshly initialized values in `state`.
    """
    resolved = os.path.expanduser(ckpt_path)
    if not os.path.exists(resolved) and not os.path.isabs(ckpt_path):
        repo_cand = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ckpt_path
        )
        if os.path.exists(repo_cand):
            resolved = repo_cand
    if os.path.isdir(resolved):
        st_cand = os.path.join(resolved, "model.safetensors")
        mp_cand = os.path.join(resolved, "flax_model.msgpack")
        if os.path.isfile(st_cand):
            resolved = st_cand
        elif os.path.isfile(mp_cand):
            resolved = mp_cand
        else:
            raise FileNotFoundError(
                f"No model.safetensors or flax_model.msgpack found in {resolved}"
            )

    loaded = load_model(resolved, None)
    if isinstance(loaded, dict) and "params" in loaded:
        src_params = loaded["params"]
        src_bs = loaded.get("batch_stats", {}) or {}
    else:
        src_params = loaded
        src_bs = {}

    flat_target = flax.traverse_util.flatten_dict(state.params, sep=".")
    flat_src = flax.traverse_util.flatten_dict(src_params, sep=".")
    copied = 0
    for k, src_v in flat_src.items():
        if k.startswith("_aux_class_weights"):
            continue
        if k in flat_target and tuple(src_v.shape) == tuple(flat_target[k].shape):
            flat_target[k] = jnp.asarray(src_v, dtype=flat_target[k].dtype)
            copied += 1
    new_params = flax.traverse_util.unflatten_dict(flat_target, sep=".")

    new_bs = state.batch_stats
    if state.batch_stats and src_bs:
        flat_bs_tgt = flax.traverse_util.flatten_dict(state.batch_stats, sep=".")
        flat_bs_src = flax.traverse_util.flatten_dict(src_bs, sep=".")
        for k, src_v in flat_bs_src.items():
            if k in flat_bs_tgt and tuple(src_v.shape) == tuple(flat_bs_tgt[k].shape):
                flat_bs_tgt[k] = jnp.asarray(src_v, dtype=flat_bs_tgt[k].dtype)
        new_bs = flax.traverse_util.unflatten_dict(flat_bs_tgt, sep=".")

    print(
        f"Warm-started {copied} parameter tensors from {resolved} "
        f"(kept {len(flat_target) - copied} freshly initialized tensors)."
    )
    return state.replace(params=new_params, batch_stats=new_bs)


def create_train_state(
    module: nn.Module,
    rng: jax.Array,
    myconfig: Any,
    num_classes: int = 0,
) -> SpeakerTrainState:
    """Creates an initial `SpeakerTrainState` (including `batch_stats` and auxiliary loss params)."""
    exp_cfg = configs.ExperimentConfig.from_any(myconfig)
    seq_len = int(exp_cfg.model.seq_len)
    feat_dim = exp_cfg.frontend.feature_dim
    init_vars = module.init(rng, jnp.ones([1, seq_len, feat_dim]))
    params = dict(init_vars["params"])
    batch_stats = dict(init_vars.get("batch_stats", {}))

    loss_cfg = exp_cfg.train.loss
    loss_type = str(getattr(loss_cfg.loss_type, "value", loss_cfg.loss_type))
    scoring_type = str(
        getattr(loss_cfg.scoring_type, "value", loss_cfg.scoring_type)
    )

    if loss_type in (
        "ge2e_softmax",
        "ge2e_contrast",
        "extended_set_softmax",
        "pairwise",
    ):
        init_w = float(loss_cfg.ge2e_init_w)
        init_b = float(loss_cfg.ge2e_init_b)
        params["_aux_w"] = jnp.array(init_w, dtype=jnp.float32)
        params["_aux_b"] = jnp.array(init_b, dtype=jnp.float32)

    if scoring_type == "pfas" and getattr(
        exp_cfg.model.pfas, "use_trainable_scale_factor", True
    ):
        init_scale = max(float(exp_cfg.model.pfas.scale_factor), 1.0)
        params["_aux_pfas_log_scale"] = jnp.array(
            float(np.log(init_scale)), dtype=jnp.float32
        )

    if scoring_type == "dr_vector":
        dr_net = scoring.build_decision_residual_network(exp_cfg.model.dr_vector)
        emb_dim = exp_cfg.model.output_embedding_dim
        dr_rng = jax.random.fold_in(rng, 101)
        dr_params = dr_net.init(
            dr_rng,
            jnp.ones((2, emb_dim), dtype=jnp.float32),
            jnp.ones((2, emb_dim), dtype=jnp.float32),
        )["params"]
        params["_aux_dr_net"] = dr_params

    if (
        loss_type
        in ("arcface", "subcenter_arcface", "cosface", "sphereface", "softmax")
        and num_classes > 0
    ):
        cls_rng = jax.random.fold_in(rng, 202)
        emb_dim = exp_cfg.model.output_embedding_dim
        n_sub = (
            max(1, int(getattr(loss_cfg, "subcenter_k", 3)))
            if loss_type == "subcenter_arcface"
            else 1
        )
        total_cols = num_classes * n_sub
        w_init = jax.random.normal(cls_rng, (emb_dim, total_cols)) * 0.05
        w_init = w_init / (
            jnp.linalg.norm(w_init, axis=0, keepdims=True) + 1e-6
        )
        params["_aux_class_weights"] = w_init

    tx = build_optimizer(exp_cfg)
    return SpeakerTrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )


def get_speaker_encoder(
    myconfig: Any,
    load_from: str = "",
    num_classes: int = 0,
    skip_warm_start: bool = False,
) -> tuple[nn.Module, SpeakerTrainState]:
    """Create `ModernSpeakerEncoder` and initialize, load, or warm-start `SpeakerTrainState`."""
    exp_cfg = configs.ExperimentConfig.from_any(myconfig)
    encoder = build_modern_encoder_from_config(exp_cfg)
    seed = int(exp_cfg.train.seed)
    init_rng = jax.random.PRNGKey(seed)
    state = create_train_state(
        encoder, init_rng, exp_cfg, num_classes=num_classes
    )
    if load_from:
        template = (
            {"params": state.params, "batch_stats": state.batch_stats}
            if state.batch_stats
            else {"params": state.params}
        )
        loaded = load_model(load_from, template)
        if isinstance(loaded, dict) and "params" in loaded:
            loaded_params = loaded["params"]
            loaded_bs = loaded.get("batch_stats", state.batch_stats) or state.batch_stats
        else:
            loaded_params = loaded
            loaded_bs = state.batch_stats
        state = state.replace(params=loaded_params, batch_stats=loaded_bs)
    elif exp_cfg.train.init_from_checkpoint and not skip_warm_start:
        state = warm_start_from_checkpoint(
            state, exp_cfg.train.init_from_checkpoint
        )

    return encoder, state


def make_modern_train_step(exp_cfg: configs.ExperimentConfig):
    """Build a JIT-compiled training step function for the configured loss & scoring."""
    exp_cfg = configs.ExperimentConfig.from_any(exp_cfg)
    loss_cfg = exp_cfg.train.loss
    loss_type = str(getattr(loss_cfg.loss_type, "value", loss_cfg.loss_type))
    scoring_type = str(
        getattr(loss_cfg.scoring_type, "value", loss_cfg.scoring_type)
    )
    n_spk = int(loss_cfg.num_speakers_per_batch)
    n_utt = int(loss_cfg.num_utts_per_speaker)
    split_batch = bool(loss_cfg.ge2e_split_batch)
    exact_loo = bool(getattr(loss_cfg, "ge2e_exact_leave_one_out", True))
    learn_wb = bool(getattr(loss_cfg, "ge2e_learn_wb", True))
    wb_grad_scale = float(getattr(exp_cfg.train, "ge2e_grad_scale", 0.01))
    contrast_surrogate = bool(getattr(loss_cfg, "ge2e_contrast_surrogate", False))
    contrast_b_scale = float(getattr(loss_cfg, "ge2e_contrast_b_grad_scale", 0.0))
    b_grad_scale = (
        contrast_b_scale
        if (loss_type == "ge2e_contrast" and contrast_b_scale > 0.0)
        else wb_grad_scale
    )
    if loss_type == "ge2e_contrast" and scoring_type == "pfas":
        raise ValueError(
            "loss_type='ge2e_contrast' does not support scoring_type='pfas' "
            "(the packed PFAS key/value representation cannot be mean-pooled into "
            "GE2E centroids). Use 'ge2e_softmax' or 'extended_set_softmax' with PFAS."
        )
    multi_row_window = int(loss_cfg.multi_row_window)
    pure_ext = bool(getattr(loss_cfg, "pure_extended_set", False))
    ext_norm_spks = bool(
        getattr(loss_cfg, "ext_softmax_normalize_by_num_speakers", True)
    )
    ext_row_w = float(
        getattr(loss_cfg, "ext_softmax_row_softmax_weight", 0.5)
    )
    scale = float(loss_cfg.scale)
    margin = float(loss_cfg.margin)
    subcenter_k = max(1, int(getattr(loss_cfg, "subcenter_k", 3)))
    sphere_m = float(loss_cfg.sphereface_m)
    sf_l_min = float(getattr(loss_cfg, "sphereface_lambda_min", 5.0))
    sf_l_base = float(getattr(loss_cfg, "sphereface_lambda_base", 1000.0))
    sf_l_gamma = float(getattr(loss_cfg, "sphereface_lambda_gamma", 0.12))
    sf_l_power = float(getattr(loss_cfg, "sphereface_lambda_power", 1.0))
    margin_warmup_steps = int(getattr(loss_cfg, "margin_warmup_steps", 0))
    triplet_alpha = float(exp_cfg.train.triplet_alpha)
    batch_size = int(exp_cfg.train.batch_size)
    freeze_trunk = bool(getattr(exp_cfg.train, "freeze_trunk", False))

    pfas_cfg = exp_cfg.model.pfas
    is_pfas = scoring_type == "pfas"
    dr_net = (
        scoring.build_decision_residual_network(exp_cfg.model.dr_vector)
        if scoring_type == "dr_vector"
        else None
    )

    @jax.jit
    def _step(
        state: SpeakerTrainState,
        batch_input: jax.Array,
        batch_labels: Optional[jax.Array] = None,
        batch_mask: Optional[jax.Array] = None,
    ) -> tuple[SpeakerTrainState, jax.Array]:
        step_f = state.step.astype(jnp.float32)
        sf_lambda = jnp.maximum(
            sf_l_min,
            sf_l_base * jnp.power(1.0 + sf_l_gamma * step_f, -sf_l_power),
        )
        if margin_warmup_steps > 0:
            warmup_frac = jnp.clip(
                step_f / float(margin_warmup_steps), 0.0, 1.0
            )
            eff_margin = margin * warmup_frac
            eff_sphere_m = 1.0 + (sphere_m - 1.0) * warmup_frac
        else:
            eff_margin = margin
            eff_sphere_m = sphere_m

        cur_bs = getattr(state, "batch_stats", None) or {}
        has_bs = bool(cur_bs) and not freeze_trunk

        def loss_fn(params: dict[str, Any]) -> tuple[jax.Array, dict[str, Any]]:
            if has_bs:
                embeddings, updates = state.apply_fn(
                    {"params": params, "batch_stats": cur_bs},
                    batch_input,
                    deterministic=False,
                    mask=batch_mask,
                    mutable=["batch_stats"],
                )
                next_bs = updates["batch_stats"]
            elif cur_bs:
                embeddings = state.apply_fn(
                    {"params": params, "batch_stats": cur_bs},
                    batch_input,
                    deterministic=True,
                    mask=batch_mask,
                )
                next_bs = cur_bs
            else:
                embeddings = state.apply_fn(
                    {"params": params},
                    batch_input,
                    deterministic=False,
                    mask=batch_mask,
                )
                next_bs = cur_bs

            score_fn = None
            if scoring_type == "dr_vector" and dr_net is not None:
                dr_params = params["_aux_dr_net"]

                def _dr_score(t_e: jax.Array, e_e: jax.Array) -> jax.Array:
                    return dr_net.apply({"params": dr_params}, t_e, e_e)

                score_fn = _dr_score
            elif scoring_type == "pfas":
                pfas_scale = (
                    jnp.exp(params["_aux_pfas_log_scale"])
                    if "_aux_pfas_log_scale" in params
                    else pfas_cfg.scale_factor
                )
                score_fn = partial(
                    scoring.pfas_score_matrix,
                    num_keys=pfas_cfg.num_keys,
                    key_dim=pfas_cfg.key_dim,
                    value_dim=pfas_cfg.value_dim,
                    scale_factor=pfas_scale,
                    per_trial_softmax=pfas_cfg.per_trial_softmax,
                    apply_softmax_per_test_key=bool(
                        getattr(pfas_cfg, "apply_softmax_per_test_key", False)
                    ),
                    apply_l2_norm_to_keys=getattr(
                        pfas_cfg, "apply_l2_norm_to_keys", True
                    ),
                    apply_l2_norm_to_values=getattr(
                        pfas_cfg, "apply_l2_norm_to_values", False
                    ),
                    apply_global_l2_norm_to_concat_form=getattr(
                        pfas_cfg, "apply_global_l2_norm_to_concat_form", True
                    ),
                    attention_weight_mode=getattr(
                        pfas_cfg, "attention_weight_mode", "exp"
                    ),
                )

            if loss_type == "triplet":
                loss_v = losses.get_triplet_loss_from_batch_output(
                    embeddings, batch_size, triplet_alpha
                )
            elif loss_type == "ge2e_softmax":
                loss_v = losses.ge2e_softmax_loss(
                    embeddings,
                    num_speakers=n_spk,
                    num_utts_per_speaker=n_utt,
                    w=params["_aux_w"],
                    b=params["_aux_b"],
                    split_batch=split_batch,
                    exact_leave_one_out=exact_loo,
                    score_matrix_fn=score_fn,
                    is_pfas=is_pfas,
                    pfas_num_keys=pfas_cfg.num_keys,
                    pfas_key_dim=pfas_cfg.key_dim,
                    pfas_value_dim=pfas_cfg.value_dim,
                )
            elif loss_type == "ge2e_contrast":
                loss_v = losses.ge2e_contrast_loss(
                    embeddings,
                    num_speakers=n_spk,
                    num_utts_per_speaker=n_utt,
                    w=params["_aux_w"],
                    b=params["_aux_b"],
                    split_batch=split_batch,
                    exact_leave_one_out=exact_loo,
                    score_matrix_fn=score_fn,
                    surrogate=contrast_surrogate,
                )
            elif loss_type == "extended_set_softmax":
                loss_v = losses.extended_set_softmax_loss(
                    embeddings,
                    num_speakers=n_spk,
                    num_utts_per_speaker=n_utt,
                    w=params["_aux_w"],
                    b=params["_aux_b"],
                    multi_row_window=multi_row_window,
                    score_matrix_fn=score_fn,
                    is_pfas=is_pfas,
                    pfas_num_keys=pfas_cfg.num_keys,
                    pfas_key_dim=pfas_cfg.key_dim,
                    pfas_value_dim=pfas_cfg.value_dim,
                    pure_extended_set=pure_ext,
                    normalize_by_num_speakers=ext_norm_spks,
                    row_softmax_weight=ext_row_w,
                )
            elif loss_type == "pairwise":
                loss_v = losses.pairwise_loss(
                    embeddings,
                    num_speakers=n_spk,
                    num_utts_per_speaker=n_utt,
                    w=params["_aux_w"],
                    b=params["_aux_b"],
                    score_matrix_fn=score_fn,
                    is_pfas=is_pfas,
                    pfas_num_keys=pfas_cfg.num_keys,
                    pfas_key_dim=pfas_cfg.key_dim,
                    pfas_value_dim=pfas_cfg.value_dim,
                )
            elif loss_type == "arcface":
                loss_v = losses.arcface_loss(
                    embeddings,
                    class_weights=params["_aux_class_weights"],
                    labels=batch_labels,
                    scale=scale,
                    margin=eff_margin,
                )
            elif loss_type == "subcenter_arcface":
                loss_v = losses.subcenter_arcface_loss(
                    embeddings,
                    class_weights=params["_aux_class_weights"],
                    labels=batch_labels,
                    scale=scale,
                    margin=eff_margin,
                    subcenter_k=subcenter_k,
                )
            elif loss_type == "cosface":
                loss_v = losses.cosface_loss(
                    embeddings,
                    class_weights=params["_aux_class_weights"],
                    labels=batch_labels,
                    scale=scale,
                    margin=eff_margin,
                )
            elif loss_type == "sphereface":
                loss_v = losses.sphereface_loss(
                    embeddings,
                    class_weights=params["_aux_class_weights"],
                    labels=batch_labels,
                    scale=scale,
                    margin=eff_sphere_m,
                    anneal_lambda=sf_lambda,
                )
            elif loss_type == "softmax":
                loss_v = losses.softmax_loss(
                    embeddings,
                    class_weights=params["_aux_class_weights"],
                    labels=batch_labels,
                    scale=scale,
                )
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")
            return loss_v, next_bs

        if freeze_trunk and "backbone" in state.params:
            frozen_bb = jax.lax.stop_gradient(state.params["backbone"])
            head_only_params = {
                k: v for k, v in state.params.items() if k != "backbone"
            }

            def head_loss_fn(
                h_params: dict[str, Any],
            ) -> tuple[jax.Array, dict[str, Any]]:
                return loss_fn({"backbone": frozen_bb, **h_params})

            (loss_val, new_bs), head_grads = jax.value_and_grad(
                head_loss_fn, has_aux=True
            )(head_only_params)
            grads = {
                "backbone": jax.tree_util.tree_map(jnp.zeros_like, frozen_bb),
                **head_grads,
            }
        else:
            (loss_val, new_bs), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params)
        if "_aux_w" in grads and "_aux_b" in grads:
            grads = dict(grads)
            if not learn_wb:
                grads["_aux_w"] = jnp.zeros_like(grads["_aux_w"])
                grads["_aux_b"] = jnp.zeros_like(grads["_aux_b"])
            else:
                grads["_aux_w"] = grads["_aux_w"] * wb_grad_scale
                grads["_aux_b"] = grads["_aux_b"] * b_grad_scale
        new_state = state.apply_gradients(grads=grads)
        if hasattr(new_state, "batch_stats"):
            new_state = new_state.replace(batch_stats=new_bs)
        return new_state, loss_val

    return _step


def train_network(
    spk_to_utts: Optional[dataset.SpkToUtts],
    myconfig: Any,
    pool: Optional[multiprocessing.pool.Pool] = None,
    feature_cache: Optional[feature_extraction.FeatureCache] = None,
    shard_store: Optional[shard_cache.ShardedFeatureStore] = None,
) -> list[float]:
    """Train speaker recognition network from `spk_to_utts` or `ShardedFeatureStore`."""
    del pool  # ThreadPool kept optional in signature for call-site convenience
    start_time = time.time()
    losses_history: list[float] = []

    exp_cfg = configs.ExperimentConfig.from_any(myconfig)
    loss_type = str(
        getattr(
            exp_cfg.train.loss.loss_type, "value", exp_cfg.train.loss.loss_type
        )
    )
    is_cls_loss = loss_type in (
        "arcface",
        "subcenter_arcface",
        "cosface",
        "sphereface",
        "softmax",
    )

    active_shards = shard_store
    if active_shards is None and exp_cfg.data.sharded_cache_dir:
        active_shards = shard_cache.ShardedFeatureStore(
            exp_cfg.data.sharded_cache_dir, base_speakers_only=not is_cls_loss
        )

    if active_shards is not None:
        num_classes = active_shards.num_speakers if is_cls_loss else 0
        speaker_to_id: dict[str, int] = active_shards.speaker_to_id
    else:
        if not spk_to_utts:
            raise ValueError(
                "Either spk_to_utts or data.sharded_cache_dir / shard_store must be provided."
            )
        speaker_to_id = {
            spk: idx for idx, spk in enumerate(sorted(spk_to_utts.keys()))
        }
        num_classes = len(speaker_to_id) if is_cls_loss else 0

    _, state = get_speaker_encoder(exp_cfg, num_classes=num_classes)
    step_fn = make_modern_train_step(exp_cfg)
    frontend = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)
    rng = np.random.default_rng(exp_cfg.train.seed)

    max_seq_len = int(
        getattr(exp_cfg.model, "max_seq_len", exp_cfg.model.seq_len)
    )
    min_seq_len = int(getattr(exp_cfg.model, "min_seq_len", max_seq_len // 2))
    var_len = bool(getattr(exp_cfg.model, "variable_length_training", True))
    triplet_rng = random.Random(int(exp_cfg.train.seed))

    for step in range(exp_cfg.train.num_steps):
        if active_shards is not None:
            if is_cls_loss:
                batch_input, batch_labels, batch_mask = (
                    active_shards.sample_classification_batch(
                        batch_size=exp_cfg.train.batch_size,
                        max_seq_len=max_seq_len,
                        min_seq_len=min_seq_len,
                        variable_length=var_len,
                        specaug_config=exp_cfg.train.specaug,
                        rng=rng,
                        domain_temp_alpha=float(
                            getattr(exp_cfg.data, "domain_temp_alpha", 0.35)
                        ),
                    )
                )
            elif loss_type == "triplet":
                batch_input, batch_labels, batch_mask = (
                    active_shards.sample_triplet_batch(
                        batch_size=exp_cfg.train.batch_size,
                        max_seq_len=max_seq_len,
                        min_seq_len=min_seq_len,
                        variable_length=var_len,
                        specaug_config=exp_cfg.train.specaug,
                        rng=rng,
                    )
                )
            else:
                batch_input, batch_labels, batch_mask = (
                    active_shards.sample_nm_batch(
                        num_speakers=exp_cfg.train.loss.num_speakers_per_batch,
                        num_utts_per_speaker=exp_cfg.train.loss.num_utts_per_speaker,
                        max_seq_len=max_seq_len,
                        min_seq_len=min_seq_len,
                        variable_length=var_len,
                        specaug_config=exp_cfg.train.specaug,
                        rng=rng,
                        within_domain_prob=float(
                            getattr(exp_cfg.data, "within_domain_prob", 0.75)
                        ),
                        domain_temp_alpha=float(
                            getattr(exp_cfg.data, "domain_temp_alpha", 0.35)
                        ),
                    )
                )
        elif loss_type == "triplet":
            assert spk_to_utts is not None
            batch_list = []
            mask_list = []
            for _ in range(exp_cfg.train.batch_size):
                anchor_u, pos_u, neg_u = dataset.get_triplet(
                    spk_to_utts, rng=triplet_rng
                )
                for u in (anchor_u, pos_u, neg_u):
                    feat = (
                        feature_cache.get(u)
                        if feature_cache is not None
                        else frontend.extract_from_file(u)
                    )
                    padded, mask, _ = (
                        feature_extraction.sample_variable_length_features(
                            feat,
                            max_seq_len=max_seq_len,
                            min_seq_len=min_seq_len,
                            variable_length=var_len,
                            specaug_config=exp_cfg.train.specaug,
                            waveform_aug_config=exp_cfg.train.waveform_aug,
                            random_crop=True,
                            rng=rng,
                        )
                    )
                    batch_list.append(padded)
                    mask_list.append(mask)
            batch_input = jnp.asarray(
                np.stack(batch_list, axis=0), dtype=jnp.float32
            )
            batch_mask = jnp.asarray(
                np.stack(mask_list, axis=0), dtype=jnp.float32
            )
            batch_labels = None
        elif is_cls_loss:
            assert spk_to_utts is not None
            batch_input, batch_labels, batch_mask = (
                feature_extraction.get_batched_classification_input(
                    spk_to_utts,
                    speaker_to_id,
                    batch_size=exp_cfg.train.batch_size,
                    seq_len=max_seq_len,
                    frontend=frontend,
                    specaug_config=exp_cfg.train.specaug,
                    waveform_aug_config=exp_cfg.train.waveform_aug,
                    cache=feature_cache,
                    rng=rng,
                    min_seq_len=min_seq_len,
                    variable_length=var_len,
                    return_mask=True,
                    domain_temp_alpha=float(
                        getattr(exp_cfg.data, "domain_temp_alpha", 0.35)
                    ),
                )
            )
        else:
            assert spk_to_utts is not None
            batch_input, batch_labels, batch_mask = (
                feature_extraction.get_batched_nm_input(
                    spk_to_utts,
                    num_speakers=exp_cfg.train.loss.num_speakers_per_batch,
                    num_utts_per_speaker=exp_cfg.train.loss.num_utts_per_speaker,
                    seq_len=max_seq_len,
                    frontend=frontend,
                    specaug_config=exp_cfg.train.specaug,
                    waveform_aug_config=exp_cfg.train.waveform_aug,
                    cache=feature_cache,
                    rng=rng,
                    min_seq_len=min_seq_len,
                    variable_length=var_len,
                    return_mask=True,
                    within_domain_prob=float(
                        getattr(exp_cfg.data, "within_domain_prob", 0.75)
                    ),
                    domain_temp_alpha=float(
                        getattr(exp_cfg.data, "domain_temp_alpha", 0.35)
                    ),
                )
            )

        state, loss_val = step_fn(state, batch_input, batch_labels, batch_mask)
        loss_scalar = float(loss_val.item())
        losses_history.append(loss_scalar)

        if step % 10 == 0 or step == exp_cfg.train.num_steps - 1:
            print(
                f"step: {step + 1}/{exp_cfg.train.num_steps} "
                f"loss ({loss_type}): {loss_scalar:.5f}"
            )

        if (
            exp_cfg.model.saved_model_path
            and exp_cfg.train.save_model_frequency > 0
            and (step + 1) % exp_cfg.train.save_model_frequency == 0
        ):
            checkpoint = exp_cfg.model.saved_model_path
            if checkpoint.endswith(".msgpack"):
                checkpoint = checkpoint[:-8]
            checkpoint += f".ckpt-{step + 1}.msgpack"
            save_model(checkpoint, state.params, batch_stats=state.batch_stats)

    training_time = time.time() - start_time
    print("Finished training in", training_time, "seconds")
    if exp_cfg.model.saved_model_path:
        save_model(
            exp_cfg.model.saved_model_path,
            state.params,
            batch_stats=state.batch_stats,
        )
    return losses_history


def run_lmft_stage(
    state: SpeakerTrainState,
    base_cfg: configs.ExperimentConfig,
    shard_store: shard_cache.ShardedFeatureStore,
    lmft_steps: int = 4000,
    lmft_seq_len: int = 600,
    lmft_batch_size: int = 64,
    lmft_lr: float = 1e-4,
    lmft_margin: float = 0.5,
) -> tuple[SpeakerTrainState, list[float]]:
    """Stage-2 Large-Margin Fine-Tuning (`LMFT`, Thienpondt et al., Interspeech 2021).

    Continues training `state` on long `lmft_seq_len`-frame (`6.0s`) utterances with
    tighter angular margin (`lmft_margin = 0.5`), reduced learning rate, and disabled
    SpecAugment so long-context statistics and decision boundaries sharpen cleanly.
    """
    import copy

    lmft_cfg = copy.deepcopy(base_cfg)
    lmft_cfg.model.seq_len = int(lmft_seq_len)
    lmft_cfg.model.max_seq_len = int(lmft_seq_len)
    lmft_cfg.model.min_seq_len = int(lmft_seq_len)
    lmft_cfg.model.variable_length_training = False
    lmft_cfg.train.batch_size = int(lmft_batch_size)
    lmft_cfg.train.num_steps = int(lmft_steps)
    lmft_cfg.train.learning_rate = float(lmft_lr)
    lmft_cfg.train.min_learning_rate = float(lmft_lr) * 0.05
    lmft_cfg.train.warmup_steps = min(200, max(1, lmft_steps // 10))
    lmft_cfg.train.specaug.use_specaug = False
    lmft_cfg.train.loss.margin = float(lmft_margin)
    lmft_cfg.train.loss.margin_warmup_steps = 0
    lmft_cfg.loss = lmft_cfg.train.loss

    tx = build_optimizer(lmft_cfg)
    lmft_state = SpeakerTrainState.create(
        apply_fn=state.apply_fn,
        params=state.params,
        tx=tx,
        batch_stats=state.batch_stats,
    )
    step_fn = make_modern_train_step(lmft_cfg)
    rng = np.random.default_rng(int(lmft_cfg.train.seed) + 777)
    loss_type = str(
        getattr(
            lmft_cfg.train.loss.loss_type, "value", lmft_cfg.train.loss.loss_type
        )
    )
    is_cls = loss_type in (
        "arcface",
        "subcenter_arcface",
        "cosface",
        "sphereface",
        "softmax",
    )

    lmft_losses: list[float] = []
    for step in range(lmft_steps):
        if is_cls:
            b_in, b_lbl, b_m = shard_store.sample_classification_batch(
                batch_size=lmft_batch_size,
                max_seq_len=lmft_seq_len,
                min_seq_len=lmft_seq_len,
                variable_length=False,
                specaug_config=lmft_cfg.train.specaug,
                rng=rng,
                domain_temp_alpha=float(
                    getattr(lmft_cfg.data, "domain_temp_alpha", 0.35)
                ),
            )
        else:
            b_in, b_lbl, b_m = shard_store.sample_nm_batch(
                num_speakers=lmft_cfg.train.loss.num_speakers_per_batch,
                num_utts_per_speaker=lmft_cfg.train.loss.num_utts_per_speaker,
                max_seq_len=lmft_seq_len,
                min_seq_len=lmft_seq_len,
                variable_length=False,
                specaug_config=lmft_cfg.train.specaug,
                rng=rng,
                within_domain_prob=float(
                    getattr(lmft_cfg.data, "within_domain_prob", 0.75)
                ),
                domain_temp_alpha=float(
                    getattr(lmft_cfg.data, "domain_temp_alpha", 0.35)
                ),
            )
        lmft_state, l_val = step_fn(lmft_state, b_in, b_lbl, b_m)
        lmft_losses.append(float(l_val.item()))
        if step % 50 == 0 or step == lmft_steps - 1:
            print(
                f"[LMFT] step: {step + 1}/{lmft_steps} "
                f"loss ({loss_type}, m={lmft_margin}): {lmft_losses[-1]:.5f}"
            )
    return lmft_state, lmft_losses


def visualize_losses(losses_list: list[float], output_path: Optional[str] = None) -> None:
    """Plot training loss curve."""
    plt.figure(figsize=(7, 4))
    plt.plot([float(x) for x in losses_list])
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def run_training(myconfig: Any) -> None:
    """Entry point to load dataset (or sharded cache) and run training from config."""
    exp_cfg = configs.ExperimentConfig.from_any(myconfig)
    if exp_cfg.data.sharded_cache_dir:
        print(f"Training from sharded cache: {exp_cfg.data.sharded_cache_dir}")
        train_network(None, exp_cfg)
        return
    spk_to_utts = dataset.load_dataset_from_config(exp_cfg, split="train")
    print(f"Loaded {len(spk_to_utts)} speakers for training.")
    train_network(spk_to_utts, exp_cfg)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        config_file = "myconfig.yml"
    elif len(args) == 1:
        config_file = args[0]
    else:
        raise ValueError("Expecting a single argument: config file path")
    run_training(configs.ExperimentConfig.from_yaml(config_file))
