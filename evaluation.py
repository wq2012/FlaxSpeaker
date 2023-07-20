import jax.numpy as jnp
import numpy as np
from multiprocessing.pool import ThreadPool
import time

import dataset
import feature_extraction
import neural_net
import myconfig


def run_inference(features, state,
                  full_sequence=myconfig.USE_FULL_SEQUENCE_INFERENCE):
    """Get the embedding of an utterance using the encoder."""
    if full_sequence:
        # Full sequence inference.
        batch_input = jnp.expand_dims(
            features, axis=0)
        batch_output = state.apply_fn({'params': state.params}, batch_input)
        return batch_output[0, :]
    else:
        # Sliding window inference.
        sliding_windows = feature_extraction.extract_sliding_windows(features)
        if not sliding_windows:
            return None
        batch_input = jnp.stack(sliding_windows)
        batch_output = state.apply_fn({'params': state.params}, batch_input)

        # Aggregate the inference outputs from sliding windows.
        aggregated_output = jnp.mean(batch_output, axis=0, keepdims=False)
        return aggregated_output


class TripletScoreFetcher:
    """Class for computing triplet scores with multi-processing."""

    def __init__(self, spk_to_utts, state, num_eval_triplets):
        self.spk_to_utts = spk_to_utts
        self.state = state
        self.num_eval_triplets = num_eval_triplets

    def __call__(self, i):
        """Get the labels and scores from a triplet."""
        anchor, pos, neg = feature_extraction.get_triplet_features(
            self.spk_to_utts)
        anchor_embedding = run_inference(anchor, self.state)
        pos_embedding = run_inference(pos, self.state)
        neg_embedding = run_inference(neg, self.state)
        if ((anchor_embedding is None) or
            (pos_embedding is None) or
                (neg_embedding is None)):
            # Some utterances might be smaller than a single sliding window.
            return ([], [])
        triplet_labels = [1, 0]
        triplet_scores = [
            neural_net.cosine_similarity(anchor_embedding, pos_embedding),
            neural_net.cosine_similarity(anchor_embedding, neg_embedding)]
        print("triplets evaluated:", i, "/", self.num_eval_triplets)
        return (triplet_labels, triplet_scores)


def compute_scores(state, spk_to_utts, num_eval_triplets=myconfig.NUM_EVAL_TRIPLETS):
    """Compute cosine similarity scores from testing data."""
    labels = []
    scores = []
    fetcher = TripletScoreFetcher(
        spk_to_utts, state, num_eval_triplets)
    # CUDA does not support multi-processing, so using a ThreadPool.
    with ThreadPool(myconfig.NUM_PROCESSES) as pool:
        while num_eval_triplets > len(labels) // 2:
            label_score_pairs = pool.map(fetcher, range(
                len(labels) // 2, num_eval_triplets))
            for triplet_labels, triplet_scores in label_score_pairs:
                labels += triplet_labels
                scores += triplet_scores
    print("Evaluated", len(labels) // 2, "triplets in total")
    return (labels, scores)


def compute_eer(labels, scores):
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
        threshold += myconfig.EVAL_THRESHOLD_STEP

    return eer, eer_threshold


def run_eval():
    """Run evaluation of the saved model on test data."""
    start_time = time.time()
    if myconfig.TEST_DATA_CSV:
        spk_to_utts = dataset.get_csv_spk_to_utts(
            myconfig.TEST_DATA_CSV)
        print("Evaluation data:", myconfig.TEST_DATA_CSV)
    else:
        spk_to_utts = dataset.get_librispeech_spk_to_utts(
            myconfig.TEST_DATA_DIR)
        print("Evaluation data:", myconfig.TEST_DATA_DIR)
    _, state = neural_net.get_speaker_encoder(
        myconfig.SAVED_MODEL_PATH)
    labels, scores = compute_scores(
        state, spk_to_utts, myconfig.NUM_EVAL_TRIPLETS)
    eer, eer_threshold = compute_eer(labels, scores)
    eval_time = time.time() - start_time
    print("Finished evaluation in", eval_time, "seconds")
    print("eer_threshold =", eer_threshold, "eer =", eer)


if __name__ == "__main__":
    run_eval()
