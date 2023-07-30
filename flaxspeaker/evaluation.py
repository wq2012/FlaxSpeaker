import jax
import jax.numpy as jnp
from flax.training import train_state
from multiprocessing.pool import ThreadPool
import time
from typing import Optional
import munch
import sys
import functools

from flaxspeaker import dataset
from flaxspeaker import feature_extraction
from flaxspeaker import neural_net


def run_inference(features: jax.Array,
                  state: train_state.TrainState,
                  myconfig: munch.Munch
                  ) -> jax.Array:
    """Get the embedding of an utterance using the encoder."""
    if myconfig.eval.full_sequence_inference:
        # Full sequence inference.
        batch_input = jnp.expand_dims(
            features, axis=0)
        batch_output = state.apply_fn({'params': state.params}, batch_input)
        return batch_output[0, :]
    else:
        # Sliding window inference.
        sliding_windows = feature_extraction.extract_sliding_windows(
            features, myconfig)
        if not sliding_windows:
            return None
        batch_input = jnp.stack(sliding_windows)
        batch_output = state.apply_fn({'params': state.params}, batch_input)

        # Aggregate the inference outputs from sliding windows.
        aggregated_output = jnp.mean(batch_output, axis=0, keepdims=False)
        return aggregated_output


def compute_triplet_scores(
        i: int,
        spk_to_utts: dataset.SpkToUtts,
        state: train_state.TrainState,
        config: munch.Munch) -> tuple[list[int], list[float]]:
    """Get the labels and scores from a triplet."""
    anchor, pos, neg = feature_extraction.get_triplet_features(
        spk_to_utts, config.model.n_mfcc)
    anchor_embedding = run_inference(
        anchor, state, config)
    pos_embedding = run_inference(
        pos, state, config)
    neg_embedding = run_inference(
        neg, state, config)
    if ((anchor_embedding is None) or
        (pos_embedding is None) or
            (neg_embedding is None)):
        # Some utterances might be smaller than a single sliding window.
        return ([], [])
    triplet_labels = [1, 0]
    triplet_scores = [
        neural_net.cosine_similarity(anchor_embedding, pos_embedding),
        neural_net.cosine_similarity(anchor_embedding, neg_embedding)]
    print("triplets evaluated:", i, "/", config.eval.num_triplets)
    return (triplet_labels, triplet_scores)


def compute_scores(
        state: train_state.TrainState,
        spk_to_utts: dataset.SpkToUtts,
        myconfig: munch.Munch
) -> tuple[list[int], list[float]]:
    """Compute cosine similarity scores from testing data."""
    labels = []
    scores = []
    score_fetcher = functools.partial(
        compute_triplet_scores,
        spk_to_utts=spk_to_utts,
        state=state,
        config=myconfig)
    # CUDA does not support multi-processing, so using a ThreadPool.
    with ThreadPool(myconfig.train.num_processes) as pool:
        while myconfig.eval.num_triplets > len(labels) // 2:
            label_score_pairs = pool.map(score_fetcher, range(
                len(labels) // 2, myconfig.eval.num_triplets))
            for triplet_labels, triplet_scores in label_score_pairs:
                labels += triplet_labels
                scores += triplet_scores
    print("Evaluated", len(labels) // 2, "triplets in total")
    return (labels, scores)


def compute_eer(labels: list[int], scores: list[float],
                eval_threshold_step: float
                ) -> tuple[Optional[float], Optional[float]]:
    """Compute the Equal Error Rate (EER)."""
    if len(labels) != len(scores):
        raise ValueError("Length of labels and scored must match")
    eer_threshold = None
    eer = None
    min_delta = 1
    threshold = 0.0
    while threshold < 1.0:
        accept = [score >= threshold for score in scores]
        fa = [a and (1-l) for a, l in zip(accept, labels)]
        fr = [(1-a) and l for a, l in zip(accept, labels)]
        far = sum(fa) / (len(labels) - sum(labels))
        frr = sum(fr) / sum(labels)
        delta = abs(far - frr)
        if delta < min_delta:
            min_delta = delta
            eer = (far + frr) / 2
            eer_threshold = threshold
        threshold += eval_threshold_step

    return eer, eer_threshold


def run_eval(myconfig: munch.Munch) -> None:
    """Run evaluation of the saved model on test data."""
    start_time = time.time()
    if myconfig.data.test_csv:
        spk_to_utts = dataset.get_csv_spk_to_utts(
            myconfig.data.test_csv)
        print("Evaluation data:", myconfig.data.test_csv)
    else:
        spk_to_utts = dataset.get_librispeech_spk_to_utts(
            myconfig.data.test_librispeech_dir)
        print("Evaluation data:", myconfig.data.test_librispeech_dir)
    _, state = neural_net.get_speaker_encoder(
        myconfig,
        myconfig.model.saved_model_path)
    labels, scores = compute_scores(
        state, spk_to_utts, myconfig)
    eer, eer_threshold = compute_eer(
        labels, scores, myconfig.eval.threshold_step)
    eval_time = time.time() - start_time
    print("Finished evaluation in", eval_time, "seconds")
    print("eer_threshold =", eer_threshold, "eer =", eer)


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
    run_eval(myconfig)
