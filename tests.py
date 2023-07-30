import os
import jax.numpy as jnp
import unittest
import numpy as np
import multiprocessing
import tempfile
import munch
import functools

from flaxspeaker import dataset
from flaxspeaker import specaug
from flaxspeaker import feature_extraction
from flaxspeaker import neural_net
from flaxspeaker import evaluation


EPS = 1e-6


class TestBase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        with open("myconfig.yml") as f:
            self.config = munch.Munch.fromYAML(f.read())

        self.config.data.train_librispeech_dir = (
            "testdata/LibriSpeech/train-clean-100")
        self.config.data.test_librispeech_dir = (
            "testdata/LibriSpeech/test-clean")


class TestDataset(TestBase):
    def setUp(self):
        super().setUp()
        self.spk_to_utts = dataset.get_librispeech_spk_to_utts(
            self.config.data.test_librispeech_dir)

    def test_get_librispeech_spk_to_utts(self):
        self.assertEqual(len(self.spk_to_utts.keys()), 3)
        self.assertEqual(len(self.spk_to_utts["121"]), 6)

    def test_get_csv_spk_to_utts(self):
        csv_content = """
spk1,/path/to/utt1
spk1, /path/to/utt2
spk2 ,/path/to/utt3
        """
        _, csv_file = tempfile.mkstemp()
        with open(csv_file, "wt") as f:
            f.write(csv_content)
        spk_to_utts = dataset.get_csv_spk_to_utts(csv_file)
        self.assertEqual(len(spk_to_utts.keys()), 2)
        self.assertEqual(len(spk_to_utts["spk1"]), 2)
        self.assertEqual(len(spk_to_utts["spk2"]), 1)

    def test_get_triplet(self):
        anchor1, pos1, neg1 = dataset.get_triplet(self.spk_to_utts)
        anchor1_spk = os.path.basename(anchor1).split("-")[0]
        pos1_spk = os.path.basename(pos1).split("-")[0]
        neg1_spk = os.path.basename(neg1).split("-")[0]
        self.assertEqual(anchor1_spk, pos1_spk)
        self.assertNotEqual(anchor1_spk, neg1_spk)

        # The following lines are commented out because the test data
        # is too small to guarantee generating two different triplets.

        # anchor2, pos2, neg2 = dataset.get_triplet(self.spk_to_utts)
        # anchor2_spk = os.path.basename(anchor2).split("-")[0]
        # pos2_spk = os.path.basename(pos2).split("-")[0]
        # neg2_spk = os.path.basename(neg2).split("-")[0]
        # self.assertNotEqual(anchor1_spk, anchor2_spk)
        # self.assertNotEqual(pos1_spk, pos2_spk)
        # self.assertNotEqual(neg1_spk, neg2_spk)


class TestSpecAug(TestBase):
    def test_specaug(self):
        features = np.random.rand(
            self.config.model.seq_len, self.config.model.n_mfcc)
        outputs = specaug.apply_specaug(features, self.config.train.specaug)
        self.assertEqual(
            outputs.shape,
            (self.config.model.seq_len, self.config.model.n_mfcc))


class TestFeatureExtraction(TestBase):
    def setUp(self):
        super().setUp()
        self.spk_to_utts = dataset.get_librispeech_spk_to_utts(
            self.config.data.test_librispeech_dir)

    def test_extract_features(self):
        features = feature_extraction.extract_features(os.path.join(
            self.config.data.test_librispeech_dir,
            "61/70968/61-70968-0000.flac"), self.config.model.n_mfcc)
        self.assertEqual(features.shape, (154, self.config.model.n_mfcc))

    def test_extract_sliding_windows(self):
        features = feature_extraction.extract_features(os.path.join(
            self.config.data.test_librispeech_dir,
            "61/70968/61-70968-0000.flac"), self.config.model.n_mfcc)
        sliding_windows = feature_extraction.extract_sliding_windows(
            features, self.config)
        self.assertEqual(len(sliding_windows), 2)
        self.assertEqual(sliding_windows[0].shape,
                         (self.config.model.seq_len, self.config.model.n_mfcc))

    def test_get_triplet_features(self):
        anchor, pos, neg = feature_extraction.get_triplet_features(
            self.spk_to_utts, self.config.model.n_mfcc)
        self.assertEqual(self.config.model.n_mfcc, anchor.shape[1])
        self.assertEqual(self.config.model.n_mfcc, pos.shape[1])
        self.assertEqual(self.config.model.n_mfcc, neg.shape[1])

    def test_get_triplet_features_trimmed(self):
        feature_fetcher = functools.partial(
            feature_extraction.get_trimmed_triplet_features,
            spk_to_utts=self.spk_to_utts,
            config=self.config)
        fetched = feature_fetcher(None)
        anchor = fetched[0, :, :]
        pos = fetched[1, :, :]
        neg = fetched[2, :, :]
        self.assertEqual(
            anchor.shape,
            (self.config.model.seq_len, self.config.model.n_mfcc))
        self.assertEqual(
            pos.shape, (self.config.model.seq_len, self.config.model.n_mfcc))
        self.assertEqual(
            neg.shape, (self.config.model.seq_len, self.config.model.n_mfcc))

    def test_get_batched_triplet_input(self):
        self.config.train.batch_size = 4
        batch_input = feature_extraction.get_batched_triplet_input(
            self.spk_to_utts, self.config)
        self.assertTupleEqual(
            batch_input.shape,
            (3 * 4, self.config.model.seq_len, self.config.model.n_mfcc))


class TestNeuralNet(TestBase):
    def setUp(self):
        super().setUp()
        self.spk_to_utts = dataset.get_librispeech_spk_to_utts(
            self.config.data.train_librispeech_dir)
        self.config.model.saved_model_path = ""

    def test_cosine_similarity(self):
        a = jnp.array([0.6, 0.8, 0.0])
        b = jnp.array([0.6, 0.8, 0.0])
        self.assertAlmostEqual(
            1.0, neural_net.cosine_similarity(a, b).item(), delta=EPS)

        a = jnp.array([0.6, 0.8, 0.0])
        b = jnp.array([0.8, -0.6, 0.0])
        self.assertAlmostEqual(
            0.0, neural_net.cosine_similarity(a, b).item(), delta=EPS)

        a = jnp.array([0.6, 0.8, 0.0])
        b = jnp.array([0.8, 0.6, 0.0])
        self.assertAlmostEqual(
            0.96, neural_net.cosine_similarity(a, b).item(), delta=EPS)

        a = jnp.array([0.6, 0.8, 0.0])
        b = jnp.array([0.0, 0.8, -0.6])
        self.assertAlmostEqual(
            0.64, neural_net.cosine_similarity(a, b).item(), delta=EPS)

    def test_get_triplet_loss1(self):
        anchor = jnp.array([[0.0, 1.0]])
        pos = jnp.array([[0.0, 1.0]])
        neg = jnp.array([[0.0, 1.0]])
        loss = neural_net.get_triplet_loss(
            anchor, pos, neg, self.config.train.triplet_alpha)
        self.assertAlmostEqual(
            loss.item(), self.config.train.triplet_alpha, delta=EPS)

    def test_get_triplet_loss2(self):
        anchor = jnp.array([[0.6, 0.8]])
        pos = jnp.array([[0.6, 0.8]])
        neg = jnp.array([[-0.8, 0.6]])
        loss = neural_net.get_triplet_loss(
            anchor, pos, neg, self.config.train.triplet_alpha)
        self.assertAlmostEqual(loss.item(), 0, delta=EPS)

    def test_get_triplet_loss3(self):
        anchor = jnp.array([[0.6, 0.8]])
        pos = jnp.array([[-0.8, 0.6]])
        neg = jnp.array([[0.6, 0.8]])
        loss = neural_net.get_triplet_loss(
            anchor, pos, neg, self.config.train.triplet_alpha)
        self.assertAlmostEqual(
            loss.item(), 1 + self.config.train.triplet_alpha, delta=EPS)

    def test_get_triplet_loss_from_batch_output1(self):
        batch_output = jnp.array([[0.6, 0.8], [-0.8, 0.6], [0.6, 0.8]])
        loss = neural_net.get_triplet_loss_from_batch_output(
            batch_output, batch_size=1,
            triplet_alpha=self.config.train.triplet_alpha)
        self.assertAlmostEqual(
            loss.item(), 1 + self.config.train.triplet_alpha,
            delta=EPS)

    def test_get_triplet_loss_from_batch_output2(self):
        batch_output = jnp.array(
            [[0.6, 0.8], [-0.8, 0.6], [0.6, 0.8],
             [0.6, 0.8], [-0.8, 0.6], [0.6, 0.8]])
        loss = neural_net.get_triplet_loss_from_batch_output(
            batch_output, batch_size=2,
            triplet_alpha=self.config.train.triplet_alpha)
        self.assertAlmostEqual(
            loss.item(), 1 + self.config.train.triplet_alpha,
            delta=EPS)

    def test_train_lstm_last_network(self):
        self.config.model.use_transformer = False
        self.config.model.frame_aggregation_mean = False
        self.config.train.num_steps = 2
        losses = neural_net.train_network(self.spk_to_utts, self.config)
        self.assertEqual(len(losses), 2)

    def test_train_lstm_mean_network(self):
        self.config.model.use_transformer = False
        self.config.model.frame_aggregation_mean = True
        self.config.train.num_steps = 2
        with multiprocessing.Pool(self.config.train.num_processes) as pool:
            losses = neural_net.train_network(
                self.spk_to_utts, self.config, pool=pool)
        self.assertEqual(len(losses), 2)

    def test_train_transformer_network(self):
        self.config.model.use_transformer = True
        self.config.train.num_steps = 2
        with multiprocessing.Pool(self.config.train.num_processes) as pool:
            losses = neural_net.train_network(
                self.spk_to_utts, self.config, pool=pool)
        self.assertEqual(len(losses), 2)


class TestEvaluation(TestBase):
    def setUp(self):
        super().setUp()
        self.config.model.frame_aggregation_mean = False
        self.config.model.use_transformer = False
        _, self.state = neural_net.get_speaker_encoder(self.config)
        self.spk_to_utts = dataset.get_librispeech_spk_to_utts(
            self.config.data.test_librispeech_dir)
        self.config.model.saved_model_path = "testdata/flax_model.msgpack"

    def test_run_lstm_inference(self):
        self.config.model.frame_aggregation_mean = False
        self.config.model.use_transformer = False
        self.config.model.full_sequence_inference = False
        features = feature_extraction.extract_features(os.path.join(
            self.config.data.test_librispeech_dir,
            "61/70968/61-70968-0000.flac"), self.config.model.n_mfcc)
        embedding = evaluation.run_inference(features, self.state, self.config)
        self.assertTupleEqual(
            embedding.shape, (self.config.model.lstm.hidden_size,))

    def test_run_lstm_full_sequence_inference(self):
        self.config.model.frame_aggregation_mean = True
        self.config.model.use_transformer = False
        self.config.model.full_sequence_inference = True
        _, self.state = neural_net.get_speaker_encoder(self.config)
        features = feature_extraction.extract_features(os.path.join(
            self.config.data.test_librispeech_dir,
            "61/70968/61-70968-0000.flac"), self.config.model.n_mfcc)
        embedding = evaluation.run_inference(features, self.state, self.config)
        self.assertTupleEqual(
            embedding.shape, (self.config.model.lstm.hidden_size,))

    def test_compute_scores(self):
        self.config.eval.num_triplets = 3
        labels, scores = evaluation.compute_scores(
            self.state, self.spk_to_utts, self.config)
        self.assertListEqual(labels, [1, 0, 1, 0, 1, 0])
        self.assertEqual(len(scores), 6)

    def test_compute_eer(self):
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        scores = [0.2, 0.3, 0.4, 0.59, 0.6, 0.588, 0.602, 0.7, 0.8, 0.9]
        eer, eer_threshold = evaluation.compute_eer(
            labels, scores, self.config.eval.threshold_step)
        self.assertAlmostEqual(eer, 0.2)
        self.assertAlmostEqual(eer_threshold, 0.59)


if __name__ == "__main__":
    unittest.main()
