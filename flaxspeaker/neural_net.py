import os
import time
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
import multiprocessing
from typing import Optional
import munch
from functools import partial
import sys

from flaxspeaker import dataset
from flaxspeaker import feature_extraction


class BaseSpeakerEncoder(nn.Module):
    pass


class LstmSpeakerEncoder(BaseSpeakerEncoder):
    lstm_config: munch.Munch

    def setup(self):
        self.lstm_layers = [
            nn.RNN(nn.OptimizedLSTMCell(
                features=self.lstm_config["hidden_size"]))
            for i in range(self.lstm_config["num_layers"])]

    def _aggregate_frames(self, batch_output: jax.Array) -> jax.Array:
        """Aggregate output frames."""
        if self.lstm_config["frame_aggregation_mean"]:
            return jnp.mean(
                batch_output, axis=1, keepdims=False)
        else:
            return batch_output[:, -1, :]

    def __call__(self, x: jax.Array) -> jax.Array:
        for lstm in self.lstm_layers:
            x = lstm(x)
        return self._aggregate_frames(x)


class TransformerSpeakerEncoder(BaseSpeakerEncoder):
    transformer_config: munch.Munch

    def setup(self):
        # Define the Transformer network.
        self.linear_layer = nn.Dense(features=self.transformer_config["dim"])
        self.encoders = [
            nn.SelfAttention(num_heads=self.transformer_config["num_heads"])
            for i in range(self.transformer_config["num_encoder_layers"])]
        self.temporal_attention = nn.Dense(features=1)

    def __call__(self, x: jax.Array) -> jax.Array:
        encoder_input = nn.activation.sigmoid(self.linear_layer(x))
        for encoder in self.encoders:
            encoder_output = encoder(encoder_input)
            encoder_input = encoder_output

        # Attentive temporal pooling.
        temporal_weights = self.temporal_attention(encoder_output)
        weighted_output = jnp.multiply(encoder_output, temporal_weights)
        weighted_output = jax.nn.softmax(weighted_output)
        return jnp.mean(weighted_output, axis=1, keepdims=False)


@jax.jit
def cosine_similarity(a: jax.Array, b: jax.Array) -> jax.Array:
    """Compute cosine similarity between two embeddings."""
    eps = 1e-6
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + eps)


@jax.jit
def get_triplet_loss(anchor: jax.Array, pos: jax.Array, neg: jax.Array,
                     alpha: float) -> jax.Array:
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""

    return jnp.maximum(
        jax.vmap(cosine_similarity, in_axes=[0, 0])(anchor, neg) -
        jax.vmap(cosine_similarity, in_axes=[0, 0])(anchor, pos) +
        alpha,
        0.0)


@partial(jax.jit, static_argnames=["batch_size", "triplet_alpha"])
def get_triplet_loss_from_batch_output(batch_output: jax.Array,
                                       batch_size: int,
                                       triplet_alpha: float
                                       ) -> jax.Array:
    """Triplet loss from N*(a|p|n) batch output."""
    batch_output_reshaped = jnp.reshape(
        batch_output, (batch_size, 3, batch_output.shape[1]))
    batch_loss = get_triplet_loss(
        batch_output_reshaped[:, 0, :],
        batch_output_reshaped[:, 1, :],
        batch_output_reshaped[:, 2, :],
        triplet_alpha)
    loss = jnp.mean(batch_loss)
    return loss


def save_model(saved_model_path: str,
               params: flax.core.frozen_dict.FrozenDict) -> None:
    """Save model to disk."""
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    if not saved_model_path.endswith(".msgpack"):
        saved_model_path += ".msgpack"
    bytes_output = flax.serialization.to_bytes(params)
    with open(saved_model_path, "wb") as f:
        f.write(bytes_output)
    print("Model saved to: ", saved_model_path)


def load_model(saved_model_path: str,
               params: flax.core.frozen_dict.FrozenDict
               ) -> flax.core.frozen_dict.FrozenDict:
    """Load model from disk."""
    with open(saved_model_path, "rb") as f:
        bytes_output = f.read()
    print("Model loaded from:", saved_model_path)
    return flax.serialization.from_bytes(params, bytes_output)


def create_train_state(module: nn.Module, rng: jax.Array, myconfig: munch.Munch
                       ) -> train_state.TrainState:
    """Creates an initial `TrainState`."""
    params = module.init(
        rng, jnp.ones([1, myconfig.model.seq_len, myconfig.model.n_mfcc])
    )["params"]
    tx = optax.adam(myconfig.train.learning_rate)
    return train_state.TrainState.create(
        apply_fn=module.apply, params=params, tx=tx)


def get_speaker_encoder(
    myconfig: munch.Munch,
    load_from: str = "",
) -> tuple[BaseSpeakerEncoder, train_state.TrainState]:
    """Create speaker encoder model."""
    if myconfig.model.use_transformer:
        encoder = TransformerSpeakerEncoder(
            transformer_config=myconfig.model.transformer)
    else:
        encoder = LstmSpeakerEncoder(lstm_config=myconfig.model.lstm)

    init_rng = jax.random.PRNGKey(0)
    state = create_train_state(encoder, init_rng, myconfig)
    if load_from:
        params = load_model(load_from, {"params": state.params})
        state = state.replace(params=params["params"])

    return encoder, state


@partial(jax.jit, static_argnames=["batch_size", "triplet_alpha"])
def train_step(state: train_state.TrainState,
               batch_input: jax.Array,
               batch_size: int,
               triplet_alpha: float
               ) -> tuple[train_state.TrainState, jax.Array]:
    """Train for a single step."""
    def loss_fn(params: flax.core.frozen_dict.FrozenDict) -> jax.Array:
        # Compute loss.
        batch_output = state.apply_fn({'params': params}, batch_input)
        loss = get_triplet_loss_from_batch_output(
            batch_output, batch_size, triplet_alpha)
        return loss
    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = loss_grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_val


def train_network(spk_to_utts: dataset.SpkToUtts,
                  myconfig: munch.Munch,
                  pool: Optional[multiprocessing.pool.Pool] = None
                  ) -> list[float]:
    start_time = time.time()
    losses = []
    _, state = get_speaker_encoder(myconfig)

    # Train
    for step in range(myconfig.train.num_steps):
        # Build batched input.
        batch_input = feature_extraction.get_batched_triplet_input(
            spk_to_utts, myconfig, pool)

        state, loss = train_step(
            state,
            batch_input,
            myconfig.train.batch_size,
            myconfig.train.triplet_alpha)
        losses.append(loss)

        print("step:", step, "/", myconfig.train.num_steps, "loss:", loss)

        if (myconfig.model.saved_model_path and
                (step + 1) % myconfig.train.save_model_frequency == 0):
            checkpoint = myconfig.model.saved_model_path
            if checkpoint.endswith(".msgpack"):
                checkpoint = checkpoint[:-8]
            checkpoint += ".ckpt-" + str(step + 1) + ".msgpack"
            save_model(checkpoint, {"params": state.params})

    training_time = time.time() - start_time
    print("Finished training in", training_time, "seconds")
    if myconfig.model.saved_model_path:
        save_model(myconfig.model.saved_model_path, {"params": state.params})
    return losses


def visualize_losses(losses: list[float]) -> None:
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


def run_training(myconfig: munch.Munch) -> None:
    if myconfig.data.train_csv:
        spk_to_utts = dataset.get_csv_spk_to_utts(
            myconfig.data.train_csv)
        print("Training data:", myconfig.data.train_csv)
    else:
        spk_to_utts = dataset.get_librispeech_spk_to_utts(
            myconfig.data.train_librispeech_dir)
        print("Training data:", myconfig.data.train_librispeech_dir)
    with multiprocessing.Pool(myconfig.train.num_processes) as pool:
        losses = train_network(spk_to_utts,
                               myconfig,
                               pool)
    visualize_losses(losses)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        config_file = "myconfig.yml"
    elif len(args) == 1:
        config_file = args[0]
    else:
        raise ValueError("Expecting a single argument: config file path")
    with open(config_file) as f:
        myconfig = munch.Munch.fromYAML(f.read())
    run_training(myconfig)
