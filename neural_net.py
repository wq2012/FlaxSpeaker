import os
import time
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
from flax import linen as nn
import tensorflow as tf
import optax
import matplotlib.pyplot as plt
import multiprocessing

import dataset
import feature_extraction
import myconfig


class BaseSpeakerEncoder(nn.Module):
    pass


class LstmSpeakerEncoder(BaseSpeakerEncoder):

    def setup(self):
        self.lstm_layers = [
            nn.RNN(nn.OptimizedLSTMCell(
                features=myconfig.LSTM_HIDDEN_SIZE))
            for i in range(myconfig.LSTM_NUM_LAYERS)]

    def _aggregate_frames(self, batch_output):
        """Aggregate output frames."""
        if myconfig.FRAME_AGGREGATION_MEAN:
            return jnp.mean(
                batch_output, axis=1, keepdims=False)
        else:
            return batch_output[:, -1, :]

    def __call__(self, x):
        for lstm in self.lstm_layers:
            x = lstm(x)
        return self._aggregate_frames(x)


class TransformerSpeakerEncoder(BaseSpeakerEncoder):

    def setup(self):
        # Define the Transformer network.
        self.linear_layer = nn.Dense(features=myconfig.TRANSFORMER_DIM)
        self.encoders = [nn.SelfAttention(num_heads=myconfig.TRANSFORMER_HEADS)
                         for i in range(myconfig.TRANSFORMER_ENCODER_LAYERS)]
        self.temporal_attention = nn.Dense(features=1)

    def __call__(self, x):
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
def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    eps = 1e-6
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + eps)


@jax.jit
def get_triplet_loss(anchor, pos, neg):
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""

    return jnp.maximum(
        jax.vmap(cosine_similarity, in_axes=[0, 0])(anchor, neg) -
        jax.vmap(cosine_similarity, in_axes=[0, 0])(anchor, pos) +
        myconfig.TRIPLET_ALPHA,
        0.0)


def get_triplet_loss_from_batch_output(batch_output, batch_size):
    """Triplet loss from N*(a|p|n) batch output."""
    batch_output_reshaped = jnp.reshape(
        batch_output, (batch_size, 3, batch_output.shape[1]))
    batch_loss = get_triplet_loss(
        batch_output_reshaped[:, 0, :],
        batch_output_reshaped[:, 1, :],
        batch_output_reshaped[:, 2, :])
    loss = jnp.mean(batch_loss)
    return loss


def save_model(saved_model_path, state):
    """Save model to disk."""
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    if not saved_model_path.endswith(".msgpack"):
        saved_model_path += ".msgpack"
    bytes_output = flax.serialization.to_bytes(state)
    with open(saved_model_path, "wb") as f:
        f.write(bytes_output)
    print("Model saved to: ", saved_model_path)


def load_model(saved_model_path, state):
    """Load model from disk."""
    with open(saved_model_path, "rb") as f:
        bytes_output = f.read()
    print("Model loaded from:", saved_model_path)
    return flax.serialization.from_bytes(state, bytes_output)


def create_train_state(module, rng, learning_rate):
    """Creates an initial `TrainState`."""
    params = module.init(
        rng, jnp.ones([1, myconfig.SEQ_LEN, myconfig.N_MFCC]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=module.apply, params=params, tx=tx)


def get_speaker_encoder(load_from=None):
    """Create speaker encoder model."""
    if myconfig.USE_TRANSFORMER:
        encoder = TransformerSpeakerEncoder()
    else:
        encoder = LstmSpeakerEncoder()

    tf.random.set_seed(0)
    init_rng = jax.random.PRNGKey(0)
    state = create_train_state(encoder, init_rng, myconfig.LEARNING_RATE)
    if load_from:
        state = load_model(load_from, state)

    return encoder, state


@jax.jit
def train_step(state, batch_input):
    """Train for a single step."""
    def loss_fn(params):
        # Compute loss.
        batch_output = state.apply_fn({'params': params}, batch_input)
        loss = get_triplet_loss_from_batch_output(
            batch_output, myconfig.BATCH_SIZE)
        return loss
    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = loss_grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_val


def train_network(spk_to_utts, num_steps, saved_model=None, pool=None):
    start_time = time.time()
    losses = []
    _, state = get_speaker_encoder(saved_model)

    # Train
    for step in range(num_steps):
        # Build batched input.
        batch_input = feature_extraction.get_batched_triplet_input(
            spk_to_utts, myconfig.BATCH_SIZE, pool)

        state, loss = train_step(state, batch_input)
        losses.append(loss)

        print("step:", step, "/", num_steps, "loss:", loss)

        if (saved_model is not None and
                (step + 1) % myconfig.SAVE_MODEL_FREQUENCY == 0):
            checkpoint = saved_model
            if checkpoint.endswith(".msgpack"):
                checkpoint = checkpoint[:-8]
            checkpoint += ".ckpt-" + str(step + 1) + ".msgpack"
            save_model(checkpoint, state)

    training_time = time.time() - start_time
    print("Finished training in", training_time, "seconds")
    if saved_model is not None:
        save_model(saved_model, state)
    return losses


def visualize_losses(losses):
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


def run_training():
    if myconfig.TRAIN_DATA_CSV:
        spk_to_utts = dataset.get_csv_spk_to_utts(
            myconfig.TRAIN_DATA_CSV)
        print("Training data:", myconfig.TRAIN_DATA_CSV)
    else:
        spk_to_utts = dataset.get_librispeech_spk_to_utts(
            myconfig.TRAIN_DATA_DIR)
        print("Training data:", myconfig.TRAIN_DATA_DIR)
    with multiprocessing.Pool(myconfig.NUM_PROCESSES) as pool:
        losses = train_network(spk_to_utts,
                               myconfig.TRAINING_STEPS,
                               myconfig.SAVED_MODEL_PATH,
                               pool)
    visualize_losses(losses)


if __name__ == "__main__":
    run_training()
