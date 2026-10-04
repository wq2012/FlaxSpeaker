"""Neural network models, training state, and end-to-end training loop.

Supports both:
1. Legacy FlaxSpeaker API (`LstmSpeakerEncoder`, `TransformerSpeakerEncoder`,
   `get_triplet_loss`, `train_network`) for 100% backward compatibility.
2. Modernized FlaxSpeaker architecture (`ModernSpeakerEncoder`, Conformer,
   Mamba, ECAPA-TDNN, ResNet, LSTM, Transformer) with GE2E, Extended-Set Softmax,
   Dr-Vectors, Parameter-Free Attentive Scoring (PFAS), ArcFace, CosFace,
   SphereFace, Softmax, Pairwise, and Triplet losses.
"""

from functools import partial
import multiprocessing
import os
import sys
import time
from typing import Any, Optional

import flax
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import munch
import numpy as np
import optax

from flaxspeaker import backbones
from flaxspeaker import configs
from flaxspeaker import dataset
from flaxspeaker import feature_extraction
from flaxspeaker import losses
from flaxspeaker import scoring
from flaxspeaker import specaug


class BaseSpeakerEncoder(nn.Module):
    """Base class for speaker encoders."""
    pass


class LstmSpeakerEncoder(BaseSpeakerEncoder):
    """Legacy multi-layer LSTM speaker encoder."""

    lstm_config: Any

    def setup(self):
        self.lstm_layers = [
            nn.RNN(nn.OptimizedLSTMCell(features=self.lstm_config["hidden_size"]))
            for _ in range(self.lstm_config["num_layers"])
        ]

    def _aggregate_frames(self, batch_output: jax.Array) -> jax.Array:
        """Aggregate output frames."""
        if self.lstm_config["frame_aggregation_mean"]:
            return jnp.mean(batch_output, axis=1, keepdims=False)
        else:
            return batch_output[:, -1, :]

    def __call__(self, x: jax.Array) -> jax.Array:
        for lstm in self.lstm_layers:
            x = lstm(x)
        return self._aggregate_frames(x)


class TransformerSpeakerEncoder(BaseSpeakerEncoder):
    """Legacy Transformer speaker encoder."""

    transformer_config: Any

    def setup(self):
        self.linear_layer = nn.Dense(features=self.transformer_config["dim"])
        self.encoders = [
            nn.MultiHeadDotProductAttention(
                num_heads=self.transformer_config["num_heads"]
            )
            for _ in range(self.transformer_config["num_encoder_layers"])
        ]
        self.temporal_attention = nn.Dense(features=1)

    def __call__(self, x: jax.Array) -> jax.Array:
        encoder_input = nn.activation.sigmoid(self.linear_layer(x))
        encoder_output = encoder_input
        for encoder in self.encoders:
            encoder_output = encoder(encoder_input)
            encoder_input = encoder_output

        # Attentive temporal pooling.
        temporal_weights = self.temporal_attention(encoder_output)
        weighted_output = jnp.multiply(encoder_output, temporal_weights)
        weighted_output = jax.nn.softmax(weighted_output)
        return jnp.mean(weighted_output, axis=1, keepdims=False)


# Re-export legacy functions for backward compatibility
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
) -> None:
    """Save model parameters to disk in Flax msgpack (or safetensors) format."""
    parent = os.path.dirname(os.path.abspath(saved_model_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if saved_model_path.endswith(".safetensors"):
        from safetensors.flax import save_file

        flat = flax.traverse_util.flatten_dict(params, sep=".")
        flat_arrays = {k: jnp.asarray(v) for k, v in flat.items()}
        save_file(flat_arrays, saved_model_path)
        print("Model saved to: ", saved_model_path)
        return

    if not saved_model_path.endswith(".msgpack"):
        saved_model_path += ".msgpack"
    bytes_output = flax.serialization.to_bytes(params)
    with open(saved_model_path, "wb") as f:
        f.write(bytes_output)
    print("Model saved to: ", saved_model_path)


def load_model(
    saved_model_path: str,
    params: Any,
) -> Any:
    """Load model parameters from disk."""
    if saved_model_path.endswith(".safetensors"):
        from safetensors.flax import load_file

        flat_arrays = load_file(saved_model_path)
        unflat = flax.traverse_util.unflatten_dict(flat_arrays, sep=".")
        print("Model loaded from:", saved_model_path)
        return unflat

    if not os.path.exists(saved_model_path) and os.path.exists(
        saved_model_path + ".msgpack"
    ):
        saved_model_path = saved_model_path + ".msgpack"
    with open(saved_model_path, "rb") as f:
        bytes_output = f.read()
    print("Model loaded from:", saved_model_path)
    return flax.serialization.from_bytes(params, bytes_output)


def _is_modern_config(myconfig: Any) -> bool:
    """Determine whether `myconfig` uses the modern `ModernSpeakerEncoder` API."""
    if isinstance(myconfig, configs.ExperimentConfig):
        return True
    model_cfg = getattr(myconfig, "model", None)
    if model_cfg is None:
        return False
    if hasattr(model_cfg, "backbone") or (
        isinstance(model_cfg, dict) and "backbone" in model_cfg
    ):
        return True
    return False


def build_modern_encoder_from_config(
    myconfig: Any,
) -> backbones.ModernSpeakerEncoder:
    """Construct `ModernSpeakerEncoder` from an `ExperimentConfig` or dict/Munch."""
    if not isinstance(myconfig, configs.ExperimentConfig):
        exp_cfg = configs.ExperimentConfig.from_munch(myconfig)
    else:
        exp_cfg = myconfig

    m_cfg = exp_cfg.model
    b_type = m_cfg.backbone.value if hasattr(m_cfg.backbone, "value") else str(m_cfg.backbone)
    p_type = (
        m_cfg.pooling.pooling_type.value
        if hasattr(m_cfg.pooling.pooling_type, "value")
        else str(m_cfg.pooling.pooling_type)
    )
    scoring_type = (
        exp_cfg.train.loss.scoring_type.value
        if hasattr(exp_cfg.train.loss.scoring_type, "value")
        else str(exp_cfg.train.loss.scoring_type)
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
        }
    elif b_type == "conformer":
        b_kwargs = {
            "dim": m_cfg.conformer.dim,
            "num_heads": m_cfg.conformer.num_heads,
            "num_layers": m_cfg.conformer.num_layers,
            "ffn_dim": m_cfg.conformer.ffn_dim,
            "conv_kernel_size": m_cfg.conformer.conv_kernel_size,
        }
    elif b_type == "mamba":
        b_kwargs = {
            "dim": m_cfg.mamba.dim,
            "num_layers": m_cfg.mamba.num_layers,
            "d_state": m_cfg.mamba.d_state,
            "d_conv": m_cfg.mamba.d_conv,
            "expand_factor": m_cfg.mamba.expand_factor,
            "dt_rank": m_cfg.mamba.dt_rank,
        }
    elif b_type == "ecapa_tdnn":
        b_kwargs = {
            "channels": m_cfg.ecapa_tdnn.channels,
            "scale": m_cfg.ecapa_tdnn.scale,
            "kernel_sizes": tuple(m_cfg.ecapa_tdnn.kernel_sizes),
            "dilations": tuple(m_cfg.ecapa_tdnn.dilations),
            "mfa_dim": m_cfg.ecapa_tdnn.mfa_dim,
            "se_bottleneck_dim": m_cfg.ecapa_tdnn.se_bottleneck_dim,
        }
    elif b_type == "resnet":
        b_kwargs = {
            "base_channels": m_cfg.resnet.base_channels,
            "num_blocks": tuple(m_cfg.resnet.num_blocks),
            "use_se": m_cfg.resnet.use_se,
            "out_dim": m_cfg.resnet.out_dim,
        }
    else:
        raise ValueError(f"Unknown backbone: {b_type}")

    if p_type == "pfas":
        p_kwargs = {
            "num_keys": m_cfg.pfas.num_keys,
            "key_dim": m_cfg.pfas.key_dim,
            "value_dim": m_cfg.pfas.value_dim,
            "attention_dim": m_cfg.pfas.attention_dim,
            "attention_type": m_cfg.pfas.attention_type,
            "normalize_keys_and_values": m_cfg.pfas.normalize_keys_and_values,
            "normalize_overall_output": m_cfg.pfas.normalize_overall_output,
        }
    else:
        p_kwargs = {
            "attention_dim": m_cfg.pooling.attention_dim,
            "use_global_context": m_cfg.pooling.use_global_context,
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
    if isinstance(myconfig, configs.ExperimentConfig):
        return myconfig.frontend.feature_dim
    model_cfg = getattr(myconfig, "model", None)
    if model_cfg is not None and hasattr(model_cfg, "n_mfcc") and model_cfg.n_mfcc:
        frontend_cfg = getattr(myconfig, "frontend", None)
        if frontend_cfg is not None:
            ftype = str(getattr(frontend_cfg, "feature_type", "mfcc")).lower()
            if ftype != "mfcc":
                return int(getattr(frontend_cfg, "n_mels", 80))
        return int(model_cfg.n_mfcc)
    return 40


def build_optimizer(myconfig: Any) -> optax.GradientTransformation:
    """Create Optax optimizer with optional warmup cosine schedule and grad clip."""
    train_cfg = getattr(myconfig, "train", myconfig)
    lr = float(getattr(train_cfg, "learning_rate", 1e-3))
    opt_name = str(getattr(train_cfg, "optimizer", "adam")).lower()
    weight_decay = float(getattr(train_cfg, "weight_decay", 0.0))
    grad_clip = float(getattr(train_cfg, "grad_clip_norm", 0.0))
    use_cosine = bool(getattr(train_cfg, "lr_schedule_cosine", False))
    num_steps = int(getattr(train_cfg, "num_steps", 500))
    warmup_steps = min(int(getattr(train_cfg, "warmup_steps", 0)), max(0, num_steps - 1))

    if use_cosine and num_steps > 2:
        if warmup_steps > 0:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=lr * 0.1,
                peak_value=lr,
                warmup_steps=warmup_steps,
                decay_steps=num_steps,
                end_value=lr * 0.05,
            )
        else:
            schedule = optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=num_steps,
                alpha=0.05,
            )
    else:
        schedule = lr

    transforms = []
    if grad_clip > 0 and _is_modern_config(myconfig):
        transforms.append(optax.clip_by_global_norm(grad_clip))

    if opt_name == "adamw" and _is_modern_config(myconfig):
        transforms.append(optax.adamw(learning_rate=schedule, weight_decay=weight_decay))
    else:
        transforms.append(optax.adam(learning_rate=schedule))

    return optax.chain(*transforms) if len(transforms) > 1 else transforms[0]


def create_train_state(
    module: nn.Module,
    rng: jax.Array,
    myconfig: Any,
    num_classes: int = 0,
) -> train_state.TrainState:
    """Creates an initial `TrainState` (including optional auxiliary loss params)."""
    seq_len = int(getattr(myconfig.model, "seq_len", 100))
    feat_dim = _get_feature_dim(myconfig)
    params = dict(module.init(rng, jnp.ones([1, seq_len, feat_dim]))["params"])

    if _is_modern_config(myconfig):
        exp_cfg = (
            myconfig
            if isinstance(myconfig, configs.ExperimentConfig)
            else configs.ExperimentConfig.from_munch(myconfig)
        )
        loss_cfg = exp_cfg.train.loss
        loss_type = (
            loss_cfg.loss_type.value
            if hasattr(loss_cfg.loss_type, "value")
            else str(loss_cfg.loss_type)
        )
        scoring_type = (
            loss_cfg.scoring_type.value
            if hasattr(loss_cfg.scoring_type, "value")
            else str(loss_cfg.scoring_type)
        )

        # Initialize trainable scale w and bias b for GE2E / Extended-Set / Pairwise
        if loss_type in ("ge2e_softmax", "ge2e_contrast", "extended_set_softmax", "pairwise"):
            params["_aux_w"] = jnp.array(float(loss_cfg.ge2e_init_w), dtype=jnp.float32)
            params["_aux_b"] = jnp.array(float(loss_cfg.ge2e_init_b), dtype=jnp.float32)

        # Initialize Decision Residual Network (Dr-Vectors) parameters if enabled
        if scoring_type == "dr_vector":
            dr_cfg = exp_cfg.model.dr_vector
            dr_net = scoring.DecisionResidualNetwork(
                hidden_dims=tuple(dr_cfg.hidden_dims),
                activation=dr_cfg.activation,
                use_layer_norm=dr_cfg.use_layer_norm,
                include_cosine_in_input=dr_cfg.include_cosine_in_input,
                residual_weight=dr_cfg.residual_weight,
            )
            emb_dim = exp_cfg.model.output_embedding_dim
            dr_rng = jax.random.fold_in(rng, 101)
            dr_params = dr_net.init(
                dr_rng,
                jnp.ones((2, emb_dim), dtype=jnp.float32),
                jnp.ones((2, emb_dim), dtype=jnp.float32),
            )["params"]
            params["_aux_dr_net"] = dr_params

        # Initialize speaker classification weights for ArcFace / CosFace / SphereFace / Softmax
        if loss_type in ("arcface", "cosface", "sphereface", "softmax") and num_classes > 0:
            cls_rng = jax.random.fold_in(rng, 202)
            emb_dim = exp_cfg.model.output_embedding_dim
            w_init = jax.random.normal(cls_rng, (emb_dim, num_classes)) * 0.05
            w_init = w_init / (jnp.linalg.norm(w_init, axis=0, keepdims=True) + 1e-6)
            params["_aux_class_weights"] = w_init

    tx = build_optimizer(myconfig)
    return train_state.TrainState.create(
        apply_fn=module.apply,
        params=flax.core.freeze(params) if not _is_modern_config(myconfig) else params,
        tx=tx,
    )


def get_speaker_encoder(
    myconfig: Any,
    load_from: str = "",
    num_classes: int = 0,
) -> tuple[nn.Module, train_state.TrainState]:
    """Create speaker encoder model and initialize or load `TrainState`."""
    if _is_modern_config(myconfig):
        encoder = build_modern_encoder_from_config(myconfig)
    elif myconfig.model.use_transformer:
        encoder = TransformerSpeakerEncoder(
            transformer_config=myconfig.model.transformer
        )
    else:
        lstm_cfg = dict(myconfig.model.lstm)
        if "frame_aggregation_mean" in myconfig.model:
            lstm_cfg["frame_aggregation_mean"] = myconfig.model.frame_aggregation_mean
        encoder = LstmSpeakerEncoder(lstm_config=munch.Munch(lstm_cfg))

    seed = int(getattr(getattr(myconfig, "train", None), "seed", 0))
    init_rng = jax.random.PRNGKey(seed)
    state = create_train_state(encoder, init_rng, myconfig, num_classes=num_classes)
    if load_from:
        loaded = load_model(load_from, {"params": state.params})
        loaded_params = loaded["params"] if "params" in loaded else loaded
        state = state.replace(params=loaded_params)

    return encoder, state


@partial(jax.jit, static_argnames=["batch_size", "triplet_alpha"])
def train_step(
    state: train_state.TrainState,
    batch_input: jax.Array,
    batch_size: int,
    triplet_alpha: float,
) -> tuple[train_state.TrainState, jax.Array]:
    """Train for a single step with Triplet Loss."""

    def loss_fn(params: Any) -> jax.Array:
        batch_output = state.apply_fn({"params": params}, batch_input)
        return get_triplet_loss_from_batch_output(
            batch_output, batch_size, triplet_alpha
        )

    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = loss_grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_val


def make_modern_train_step(exp_cfg: configs.ExperimentConfig):
    """Build a JIT-compiled training step function for the configured loss & scoring."""
    loss_cfg = exp_cfg.train.loss
    loss_type = (
        loss_cfg.loss_type.value
        if hasattr(loss_cfg.loss_type, "value")
        else str(loss_cfg.loss_type)
    )
    scoring_type = (
        loss_cfg.scoring_type.value
        if hasattr(loss_cfg.scoring_type, "value")
        else str(loss_cfg.scoring_type)
    )
    n_spk = int(loss_cfg.num_speakers_per_batch)
    n_utt = int(loss_cfg.num_utts_per_speaker)
    split_batch = bool(loss_cfg.ge2e_split_batch)
    multi_row_window = int(loss_cfg.multi_row_window)
    scale = float(loss_cfg.scale)
    margin = float(loss_cfg.margin)
    sphere_m = float(loss_cfg.sphereface_m)
    triplet_alpha = float(exp_cfg.train.triplet_alpha)
    batch_size = int(exp_cfg.train.batch_size)

    pfas_cfg = exp_cfg.model.pfas
    is_pfas = scoring_type == "pfas"
    dr_cfg = exp_cfg.model.dr_vector
    dr_net = (
        scoring.DecisionResidualNetwork(
            hidden_dims=tuple(dr_cfg.hidden_dims),
            activation=dr_cfg.activation,
            use_layer_norm=dr_cfg.use_layer_norm,
            include_cosine_in_input=dr_cfg.include_cosine_in_input,
            residual_weight=dr_cfg.residual_weight,
        )
        if scoring_type == "dr_vector"
        else None
    )

    @jax.jit
    def _step(
        state: train_state.TrainState,
        batch_input: jax.Array,
        batch_labels: Optional[jax.Array] = None,
    ) -> tuple[train_state.TrainState, jax.Array]:
        def loss_fn(params: dict[str, Any]) -> jax.Array:
            embeddings = state.apply_fn({"params": params}, batch_input)

            score_fn = None
            if scoring_type == "dr_vector" and dr_net is not None:
                dr_params = params["_aux_dr_net"]

                def _dr_score(t_e: jax.Array, e_e: jax.Array) -> jax.Array:
                    return dr_net.apply({"params": dr_params}, t_e, e_e)

                score_fn = _dr_score
            elif scoring_type == "pfas":
                score_fn = partial(
                    scoring.pfas_score_matrix,
                    num_keys=pfas_cfg.num_keys,
                    key_dim=pfas_cfg.key_dim,
                    value_dim=pfas_cfg.value_dim,
                    scale_factor=pfas_cfg.scale_factor,
                    per_trial_softmax=pfas_cfg.per_trial_softmax,
                )

            if loss_type == "triplet":
                return losses.get_triplet_loss_from_batch_output(
                    embeddings, batch_size, triplet_alpha
                )
            elif loss_type == "ge2e_softmax":
                return losses.ge2e_softmax_loss(
                    embeddings,
                    num_speakers=n_spk,
                    num_utts_per_speaker=n_utt,
                    w=params["_aux_w"],
                    b=params["_aux_b"],
                    split_batch=split_batch,
                    score_matrix_fn=score_fn,
                    is_pfas=is_pfas,
                    pfas_num_keys=pfas_cfg.num_keys,
                    pfas_key_dim=pfas_cfg.key_dim,
                    pfas_value_dim=pfas_cfg.value_dim,
                )
            elif loss_type == "ge2e_contrast":
                return losses.ge2e_contrast_loss(
                    embeddings,
                    num_speakers=n_spk,
                    num_utts_per_speaker=n_utt,
                    w=params["_aux_w"],
                    b=params["_aux_b"],
                    split_batch=split_batch,
                    score_matrix_fn=score_fn,
                )
            elif loss_type == "extended_set_softmax":
                return losses.extended_set_softmax_loss(
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
                )
            elif loss_type == "pairwise":
                return losses.pairwise_loss(
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
                return losses.arcface_loss(
                    embeddings,
                    class_weights=params["_aux_class_weights"],
                    labels=batch_labels,
                    scale=scale,
                    margin=margin,
                )
            elif loss_type == "cosface":
                return losses.cosface_loss(
                    embeddings,
                    class_weights=params["_aux_class_weights"],
                    labels=batch_labels,
                    scale=scale,
                    margin=margin,
                )
            elif loss_type == "sphereface":
                return losses.sphereface_loss(
                    embeddings,
                    class_weights=params["_aux_class_weights"],
                    labels=batch_labels,
                    scale=scale,
                    margin=sphere_m,
                )
            elif loss_type == "softmax":
                return losses.softmax_loss(
                    embeddings,
                    class_weights=params["_aux_class_weights"],
                    labels=batch_labels,
                    scale=scale,
                )
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

        loss_val, grads = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_val

    return _step


def train_network(
    spk_to_utts: dataset.SpkToUtts,
    myconfig: Any,
    pool: Optional[multiprocessing.pool.Pool] = None,
    feature_cache: Optional[feature_extraction.FeatureCache] = None,
) -> list[float]:
    """Train speaker recognition network (supports both legacy and modern configs)."""
    start_time = time.time()
    losses_history: list[float] = []

    if not _is_modern_config(myconfig):
        # Legacy path for 100% compatibility with existing tests
        _, state = get_speaker_encoder(myconfig)
        for step in range(myconfig.train.num_steps):
            batch_input = feature_extraction.get_batched_triplet_input(
                spk_to_utts, myconfig, pool
            )
            state, loss = train_step(
                state,
                batch_input,
                myconfig.train.batch_size,
                myconfig.train.triplet_alpha,
            )
            losses_history.append(loss)
            print("step:", step, "/", myconfig.train.num_steps, "loss:", loss)

            if (
                myconfig.model.saved_model_path
                and (step + 1) % myconfig.train.save_model_frequency == 0
            ):
                checkpoint = myconfig.model.saved_model_path
                if checkpoint.endswith(".msgpack"):
                    checkpoint = checkpoint[:-8]
                checkpoint += ".ckpt-" + str(step + 1) + ".msgpack"
                save_model(checkpoint, {"params": state.params})

        training_time = time.time() - start_time
        print("Finished training in", training_time, "seconds")
        if myconfig.model.saved_model_path:
            save_model(myconfig.model.saved_model_path, {"params": state.params})
        return losses_history

    # Modern training path
    exp_cfg = (
        myconfig
        if isinstance(myconfig, configs.ExperimentConfig)
        else configs.ExperimentConfig.from_munch(myconfig)
    )
    loss_type = (
        exp_cfg.train.loss.loss_type.value
        if hasattr(exp_cfg.train.loss.loss_type, "value")
        else str(exp_cfg.train.loss.loss_type)
    )
    is_cls_loss = loss_type in ("arcface", "cosface", "sphereface", "softmax")
    speaker_to_id = {spk: idx for idx, spk in enumerate(sorted(spk_to_utts.keys()))}
    num_classes = len(speaker_to_id) if is_cls_loss else 0

    _, state = get_speaker_encoder(exp_cfg, num_classes=num_classes)
    step_fn = make_modern_train_step(exp_cfg)
    frontend = feature_extraction.AudioFrontend(exp_cfg.frontend, exp_cfg.vad)
    rng = np.random.default_rng(exp_cfg.train.seed)

    for step in range(exp_cfg.train.num_steps):
        if loss_type == "triplet":
            batch_list = []
            for _ in range(exp_cfg.train.batch_size):
                anchor_u, pos_u, neg_u = dataset.get_triplet(spk_to_utts)
                for u in (anchor_u, pos_u, neg_u):
                    feat = (
                        feature_cache.get(u)
                        if feature_cache is not None
                        else frontend.extract_from_file(u)
                    )
                    crop = feature_extraction.random_crop_or_pad(
                        feat, exp_cfg.model.seq_len, rng=rng
                    )
                    crop = specaug.apply_specaug(crop, exp_cfg.train.specaug, rng=rng)
                    batch_list.append(crop)
            batch_input = jnp.asarray(np.stack(batch_list, axis=0), dtype=jnp.float32)
            batch_labels = None
        elif is_cls_loss:
            batch_input, batch_labels = feature_extraction.get_batched_classification_input(
                spk_to_utts,
                speaker_to_id,
                batch_size=exp_cfg.train.batch_size,
                seq_len=exp_cfg.model.seq_len,
                frontend=frontend,
                specaug_config=exp_cfg.train.specaug,
                cache=feature_cache,
                rng=rng,
            )
        else:
            # Structured N x M batch for GE2E / Extended-Set Softmax / Pairwise
            batch_input, batch_labels = feature_extraction.get_batched_nm_input(
                spk_to_utts,
                num_speakers=exp_cfg.train.loss.num_speakers_per_batch,
                num_utts_per_speaker=exp_cfg.train.loss.num_utts_per_speaker,
                seq_len=exp_cfg.model.seq_len,
                frontend=frontend,
                specaug_config=exp_cfg.train.specaug,
                cache=feature_cache,
                rng=rng,
            )

        state, loss_val = step_fn(state, batch_input, batch_labels)
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
            save_model(checkpoint, {"params": state.params})

    training_time = time.time() - start_time
    print("Finished training in", training_time, "seconds")
    if exp_cfg.model.saved_model_path:
        save_model(exp_cfg.model.saved_model_path, {"params": state.params})
    return losses_history


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
    """Entry point to load dataset and run training from config."""
    spk_to_utts = dataset.load_dataset_from_config(myconfig, split="train")
    print(f"Loaded {len(spk_to_utts)} speakers for training.")
    with multiprocessing.Pool(myconfig.train.num_processes) as pool:
        losses_list = train_network(spk_to_utts, myconfig, pool)
    if not _is_modern_config(myconfig):
        visualize_losses(losses_list)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        config_file = "myconfig.yml"
    elif len(args) == 1:
        config_file = args[0]
    else:
        raise ValueError("Expecting a single argument: config file path")
    with open(config_file, "r", encoding="utf-8") as f:
        myconfig = munch.Munch.fromYAML(f.read())
    run_training(myconfig)
