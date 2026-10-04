import functools
import multiprocessing
import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import munch
import numpy as np

from flaxspeaker import configs
from flaxspeaker import dataset
from flaxspeaker import evaluation
from flaxspeaker import export
from flaxspeaker import feature_extraction
from flaxspeaker import hf_compat
from flaxspeaker import losses
from flaxspeaker import neural_net
from flaxspeaker import pooling
from flaxspeaker import scoring
from flaxspeaker import specaug
from flaxspeaker import vad


EPS = 1e-5


class TestBase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        with open("myconfig.yml") as f:
            self.config = munch.Munch.fromYAML(f.read())

        self.config.data.train_librispeech_dir = (
            "testdata/LibriSpeech/train-clean-100"
        )
        self.config.data.test_librispeech_dir = (
            "testdata/LibriSpeech/test-clean"
        )


class TestDataset(TestBase):

    def setUp(self):
        super().setUp()
        self.spk_to_utts = dataset.get_librispeech_spk_to_utts(
            self.config.data.test_librispeech_dir
        )

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

    def test_splits_and_trials(self):
        train_spk, eval_spk = dataset.split_speakers_train_eval(
            self.spk_to_utts, train_ratio=0.5, min_utts_per_spk=2, seed=42
        )
        self.assertTrue(set(train_spk.keys()).isdisjoint(set(eval_spk.keys())))
        trials = dataset.generate_verification_trials(
            self.spk_to_utts, num_trials=20, seed=42
        )
        self.assertEqual(len(trials), 20)
        multi_trials = dataset.generate_multi_enroll_trials(
            self.spk_to_utts, num_trials=10, num_enroll_utts=2, seed=42
        )
        self.assertEqual(len(multi_trials), 10)
        self.assertEqual(len(multi_trials[0][1]), 2)


class TestSpecAug(TestBase):

    def test_specaug(self):
        features = np.random.rand(
            self.config.model.seq_len, self.config.model.n_mfcc
        )
        outputs = specaug.apply_specaug(features, self.config.train.specaug)
        self.assertEqual(
            outputs.shape,
            (self.config.model.seq_len, self.config.model.n_mfcc),
        )

    def test_specaug_jax_batch(self):
        batch = jnp.ones((4, 50, 40), dtype=jnp.float32)
        rng = jax.random.PRNGKey(0)
        masked = specaug.apply_specaug_batch_jax(batch, rng)
        self.assertEqual(masked.shape, (4, 50, 40))


class TestVadAndFrontend(TestBase):

    def test_vad_modes(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        silence = np.zeros(sr // 2, dtype=np.float32)
        wav = np.concatenate([silence, tone, silence]).astype(np.float32)

        for mode in (
            configs.VadMode.ENERGY,
            configs.VadMode.SPECTRAL,
            configs.VadMode.HYBRID,
        ):
            v_cfg = configs.VadConfig(mode=mode)
            proc = vad.VadProcessor(v_cfg, sample_rate=sr)
            filtered = proc.filter_waveform(wav)
            self.assertGreater(len(filtered), 0)
            self.assertLessEqual(len(filtered), len(wav))

    def test_frontend_feature_types(self):
        sample_flac = os.path.join(
            self.config.data.test_librispeech_dir,
            "61/70968/61-70968-0000.flac",
        )
        for ftype in (
            configs.FeatureType.LOG_MEL,
            configs.FeatureType.WHISPER_MEL,
            configs.FeatureType.HF_COMPAT,
            configs.FeatureType.MFCC,
        ):
            f_cfg = configs.FrontendConfig(
                feature_type=ftype, n_mels=80, n_mfcc=40
            )
            fe = feature_extraction.AudioFrontend(f_cfg)
            feats = fe.extract_from_file(sample_flac)
            self.assertEqual(feats.shape[1], f_cfg.feature_dim)
            self.assertGreater(feats.shape[0], 50)


class TestFeatureExtraction(TestBase):

    def setUp(self):
        super().setUp()
        self.spk_to_utts = dataset.get_librispeech_spk_to_utts(
            self.config.data.test_librispeech_dir
        )

    def test_extract_features(self):
        features = feature_extraction.extract_features(
            os.path.join(
                self.config.data.test_librispeech_dir,
                "61/70968/61-70968-0000.flac",
            ),
            self.config.model.n_mfcc,
        )
        self.assertEqual(features.shape, (154, self.config.model.n_mfcc))

    def test_extract_sliding_windows(self):
        features = feature_extraction.extract_features(
            os.path.join(
                self.config.data.test_librispeech_dir,
                "61/70968/61-70968-0000.flac",
            ),
            self.config.model.n_mfcc,
        )
        sliding_windows = feature_extraction.extract_sliding_windows(
            features, self.config
        )
        self.assertEqual(len(sliding_windows), 2)
        self.assertEqual(
            sliding_windows[0].shape,
            (self.config.model.seq_len, self.config.model.n_mfcc),
        )

    def test_get_triplet_features(self):
        anchor, pos, neg = feature_extraction.get_triplet_features(
            self.spk_to_utts, self.config.model.n_mfcc
        )
        self.assertEqual(self.config.model.n_mfcc, anchor.shape[1])
        self.assertEqual(self.config.model.n_mfcc, pos.shape[1])
        self.assertEqual(self.config.model.n_mfcc, neg.shape[1])

    def test_get_triplet_features_trimmed(self):
        feature_fetcher = functools.partial(
            feature_extraction.get_trimmed_triplet_features,
            spk_to_utts=self.spk_to_utts,
            config=self.config,
        )
        fetched = feature_fetcher(None)
        anchor = fetched[0, :, :]
        pos = fetched[1, :, :]
        neg = fetched[2, :, :]
        self.assertEqual(
            anchor.shape,
            (self.config.model.seq_len, self.config.model.n_mfcc),
        )
        self.assertEqual(
            pos.shape, (self.config.model.seq_len, self.config.model.n_mfcc)
        )
        self.assertEqual(
            neg.shape, (self.config.model.seq_len, self.config.model.n_mfcc)
        )

    def test_get_batched_triplet_input(self):
        self.config.train.batch_size = 4
        batch_input = feature_extraction.get_batched_triplet_input(
            self.spk_to_utts, self.config
        )
        self.assertTupleEqual(
            batch_input.shape,
            (3 * 4, self.config.model.seq_len, self.config.model.n_mfcc),
        )


class TestBackbonesAndPooling(unittest.TestCase):

    def test_all_backbones_and_pooling(self):
        x = jnp.ones((2, 40, 80), dtype=jnp.float32)
        rng = jax.random.PRNGKey(0)

        for b_type in configs.BackboneType:
            cfg = configs.ExperimentConfig()
            cfg.model.backbone = b_type
            cfg.model.apply_size_variant(configs.SizeVariant.TINY)
            cfg.model.embedding_dim = 64
            enc = neural_net.build_modern_encoder_from_config(cfg)
            params = enc.init(rng, x)["params"]
            out = enc.apply({"params": params}, x)
            self.assertEqual(out.shape, (2, 64))
            n_params = neural_net.count_parameters(params)
            self.assertLess(n_params, 100_000_000)

        for p_type in (
            configs.PoolingType.MEAN,
            configs.PoolingType.LAST,
            configs.PoolingType.STATS,
            configs.PoolingType.SAP,
            configs.PoolingType.ASP,
            configs.PoolingType.CUMULATIVE_STATS,
        ):
            cfg = configs.ExperimentConfig()
            cfg.model.backbone = configs.BackboneType.CONFORMER
            cfg.model.apply_size_variant(configs.SizeVariant.TINY)
            cfg.model.pooling.pooling_type = p_type
            cfg.model.embedding_dim = 64
            enc = neural_net.build_modern_encoder_from_config(cfg)
            params = enc.init(rng, x)["params"]
            out = enc.apply({"params": params}, x)
            self.assertEqual(out.shape, (2, 64))


class TestAdvancedLossesAndScoring(unittest.TestCase):

    def test_ge2e_softmax_and_contrast_losses(self):
        rng = jax.random.PRNGKey(42)
        # 4 speakers, 4 utterances per speaker, dim 32
        embs = jax.random.normal(rng, (16, 32))
        w = jnp.array(10.0)
        b = jnp.array(-5.0)

        loss_loo = losses.ge2e_softmax_loss(
            embs, num_speakers=4, num_utts_per_speaker=4, w=w, b=b, split_batch=False
        )
        loss_split = losses.ge2e_softmax_loss(
            embs, num_speakers=4, num_utts_per_speaker=4, w=w, b=b, split_batch=True
        )
        loss_contrast = losses.ge2e_contrast_loss(
            embs, num_speakers=4, num_utts_per_speaker=4, w=w, b=b
        )
        self.assertGreater(float(loss_loo), 0.0)
        self.assertGreater(float(loss_split), 0.0)
        self.assertGreater(float(loss_contrast), 0.0)

    def test_extended_set_softmax_and_dr_vectors(self):
        rng = jax.random.PRNGKey(7)
        embs = jax.random.normal(rng, (16, 32))
        w = jnp.array(10.0)
        b = jnp.array(-5.0)

        dr_net = scoring.DecisionResidualNetwork(hidden_dims=(32, 16))
        dr_params = dr_net.init(rng, jnp.ones((4, 32)), jnp.ones((4, 32)))["params"]

        def score_fn(t: jax.Array, e: jax.Array) -> jax.Array:
            return dr_net.apply({"params": dr_params}, t, e)

        loss_ext = losses.extended_set_softmax_loss(
            embs,
            num_speakers=4,
            num_utts_per_speaker=4,
            w=w,
            b=b,
            score_matrix_fn=score_fn,
        )
        self.assertGreater(float(loss_ext), 0.0)

    def test_pfas_representation_and_scoring(self):
        rng = jax.random.PRNGKey(11)
        head = pooling.PfasRepresentationHead(
            num_keys=4, key_dim=16, value_dim=16, attention_dim=32
        )
        frames = jax.random.normal(rng, (12, 30, 64))
        params = head.init(rng, frames)["params"]
        packed = head.apply({"params": params}, frames)
        self.assertEqual(packed.shape, (12, 4 * (16 + 16)))

        sim_mat = scoring.pfas_score_matrix(
            packed[:6], packed[6:], num_keys=4, key_dim=16, value_dim=16
        )
        self.assertEqual(sim_mat.shape, (6, 6))

        # Multi-utterance PFAS enrollment stacking
        stacked_enroll = scoring.stack_multi_enroll_pfas(
            [packed[0], packed[1], packed[2]], num_keys=4, key_dim=16, value_dim=16
        )
        self.assertEqual(stacked_enroll.shape, (3 * 4 * (16 + 16),))

    def test_margin_classification_losses(self):
        rng = jax.random.PRNGKey(99)
        embs = jax.random.normal(rng, (8, 32))
        weights = jax.random.normal(rng, (32, 10))
        labels = jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.int32)

        l_arc = losses.arcface_loss(embs, weights, labels, scale=30.0, margin=0.2)
        l_cos = losses.cosface_loss(embs, weights, labels, scale=30.0, margin=0.2)
        l_sph = losses.sphereface_loss(embs, weights, labels, scale=30.0, margin=1.35)
        l_soft = losses.softmax_loss(embs, weights, labels, scale=16.0)

        for val in (l_arc, l_cos, l_sph, l_soft):
            self.assertTrue(np.isfinite(float(val)))
            self.assertGreater(float(val), 0.0)


class TestNeuralNet(TestBase):

    def setUp(self):
        super().setUp()
        self.spk_to_utts = dataset.get_librispeech_spk_to_utts(
            self.config.data.train_librispeech_dir
        )
        self.config.model.saved_model_path = ""

    def test_cosine_similarity(self):
        a = jnp.array([0.6, 0.8, 0.0])
        b = jnp.array([0.6, 0.8, 0.0])
        self.assertAlmostEqual(
            1.0, neural_net.cosine_similarity(a, b).item(), delta=EPS
        )

        a = jnp.array([0.6, 0.8, 0.0])
        b = jnp.array([0.8, -0.6, 0.0])
        self.assertAlmostEqual(
            0.0, neural_net.cosine_similarity(a, b).item(), delta=EPS
        )

        a = jnp.array([0.6, 0.8, 0.0])
        b = jnp.array([0.8, 0.6, 0.0])
        self.assertAlmostEqual(
            0.96, neural_net.cosine_similarity(a, b).item(), delta=EPS
        )

        a = jnp.array([0.6, 0.8, 0.0])
        b = jnp.array([0.0, 0.8, -0.6])
        self.assertAlmostEqual(
            0.64, neural_net.cosine_similarity(a, b).item(), delta=EPS
        )

    def test_get_triplet_loss1(self):
        anchor = jnp.array([[0.0, 1.0]])
        pos = jnp.array([[0.0, 1.0]])
        neg = jnp.array([[0.0, 1.0]])
        loss = neural_net.get_triplet_loss(
            anchor, pos, neg, self.config.train.triplet_alpha
        )
        self.assertAlmostEqual(
            loss.item(), self.config.train.triplet_alpha, delta=EPS
        )

    def test_get_triplet_loss2(self):
        anchor = jnp.array([[0.6, 0.8]])
        pos = jnp.array([[0.6, 0.8]])
        neg = jnp.array([[-0.8, 0.6]])
        loss = neural_net.get_triplet_loss(
            anchor, pos, neg, self.config.train.triplet_alpha
        )
        self.assertAlmostEqual(loss.item(), 0, delta=EPS)

    def test_get_triplet_loss3(self):
        anchor = jnp.array([[0.6, 0.8]])
        pos = jnp.array([[-0.8, 0.6]])
        neg = jnp.array([[0.6, 0.8]])
        loss = neural_net.get_triplet_loss(
            anchor, pos, neg, self.config.train.triplet_alpha
        )
        self.assertAlmostEqual(
            loss.item(), 1 + self.config.train.triplet_alpha, delta=EPS
        )

    def test_get_triplet_loss_from_batch_output1(self):
        batch_output = jnp.array([[0.6, 0.8], [-0.8, 0.6], [0.6, 0.8]])
        loss = neural_net.get_triplet_loss_from_batch_output(
            batch_output,
            batch_size=1,
            triplet_alpha=self.config.train.triplet_alpha,
        )
        self.assertAlmostEqual(
            loss.item(), 1 + self.config.train.triplet_alpha, delta=EPS
        )

    def test_get_triplet_loss_from_batch_output2(self):
        batch_output = jnp.array(
            [
                [0.6, 0.8],
                [-0.8, 0.6],
                [0.6, 0.8],
                [0.6, 0.8],
                [-0.8, 0.6],
                [0.6, 0.8],
            ]
        )
        loss = neural_net.get_triplet_loss_from_batch_output(
            batch_output,
            batch_size=2,
            triplet_alpha=self.config.train.triplet_alpha,
        )
        self.assertAlmostEqual(
            loss.item(), 1 + self.config.train.triplet_alpha, delta=EPS
        )

    def test_train_lstm_last_network(self):
        self.config.model.use_transformer = False
        self.config.model.frame_aggregation_mean = False
        self.config.train.num_steps = 2
        losses_out = neural_net.train_network(self.spk_to_utts, self.config)
        self.assertEqual(len(losses_out), 2)

    def test_train_lstm_mean_network(self):
        self.config.model.use_transformer = False
        self.config.model.frame_aggregation_mean = True
        self.config.train.num_steps = 2
        with multiprocessing.Pool(self.config.train.num_processes) as pool:
            losses_out = neural_net.train_network(
                self.spk_to_utts, self.config, pool=pool
            )
        self.assertEqual(len(losses_out), 2)

    def test_train_transformer_network(self):
        self.config.model.use_transformer = True
        self.config.train.num_steps = 2
        with multiprocessing.Pool(self.config.train.num_processes) as pool:
            losses_out = neural_net.train_network(
                self.spk_to_utts, self.config, pool=pool
            )
        self.assertEqual(len(losses_out), 2)


class TestEvaluation(TestBase):

    def setUp(self):
        super().setUp()
        self.config.model.frame_aggregation_mean = False
        self.config.model.use_transformer = False
        self.spk_to_utts = dataset.get_librispeech_spk_to_utts(
            self.config.data.test_librispeech_dir
        )

    def test_run_lstm_inference(self):
        self.config.model.frame_aggregation_mean = False
        self.config.model.use_transformer = False
        self.config.model.full_sequence_inference = False
        features = feature_extraction.extract_features(
            os.path.join(
                self.config.data.test_librispeech_dir,
                "61/70968/61-70968-0000.flac",
            ),
            self.config.model.n_mfcc,
        )
        _, self.state = neural_net.get_speaker_encoder(
            self.config, load_from="testdata/lstm_model.msgpack"
        )
        embedding = evaluation.run_inference(features, self.state, self.config)
        self.assertTupleEqual(
            embedding.shape, (self.config.model.lstm.hidden_size,)
        )

    def test_run_lstm_full_sequence_inference(self):
        self.config.model.frame_aggregation_mean = True
        self.config.model.use_transformer = False
        self.config.model.full_sequence_inference = True
        features = feature_extraction.extract_features(
            os.path.join(
                self.config.data.test_librispeech_dir,
                "61/70968/61-70968-0000.flac",
            ),
            self.config.model.n_mfcc,
        )
        _, self.state = neural_net.get_speaker_encoder(
            self.config, load_from="testdata/lstm_model.msgpack"
        )
        embedding = evaluation.run_inference(features, self.state, self.config)
        self.assertTupleEqual(
            embedding.shape, (self.config.model.lstm.hidden_size,)
        )

    def test_run_transformer_inference(self):
        self.config.model.use_transformer = True
        features = feature_extraction.extract_features(
            os.path.join(
                self.config.data.test_librispeech_dir,
                "61/70968/61-70968-0000.flac",
            ),
            self.config.model.n_mfcc,
        )
        _, self.state = neural_net.get_speaker_encoder(
            self.config, load_from="testdata/transformer_model.msgpack"
        )
        embedding = evaluation.run_inference(features, self.state, self.config)
        self.assertTupleEqual(
            embedding.shape, (self.config.model.transformer.dim,)
        )

    def test_compute_scores(self):
        self.config.eval.num_triplets = 3
        _, self.state = neural_net.get_speaker_encoder(
            self.config, load_from="testdata/lstm_model.msgpack"
        )
        labels, scores = evaluation.compute_scores(
            self.state, self.spk_to_utts, self.config
        )
        self.assertListEqual(labels, [1, 0, 1, 0, 1, 0])
        self.assertEqual(len(scores), 6)

    def test_compute_eer(self):
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        scores = [0.2, 0.3, 0.4, 0.59, 0.6, 0.588, 0.602, 0.7, 0.8, 0.9]
        eer, eer_threshold = evaluation.compute_eer(
            labels, scores, self.config.eval.threshold_step
        )
        self.assertAlmostEqual(eer, 0.2)
        self.assertAlmostEqual(eer_threshold, 0.59)


class TestExportAndHuggingFaceCompat(TestBase):

    def test_tflite_export_and_runner(self):
        exp_cfg = configs.ExperimentConfig()
        exp_cfg.model.backbone = configs.BackboneType.ECAPA_TDNN
        exp_cfg.model.apply_size_variant(configs.SizeVariant.TINY)
        exp_cfg.model.seq_len = 50
        exp_cfg.model.embedding_dim = 64

        _, state = neural_net.get_speaker_encoder(exp_cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = os.path.join(tmpdir, "model.tflite")
            info = export.export_to_tflite(
                state, exp_cfg, tflite_path, quantize_int8=False
            )
            self.assertTrue(os.path.exists(info["tflite_path"]))
            runner = export.TFLiteSpeakerRunner(tflite_path)
            dummy_feat = np.ones((50, 80), dtype=np.float32)
            emb = runner(dummy_feat)
            self.assertEqual(emb.shape, (64,))

    def test_hf_save_and_load_pretrained(self):
        exp_cfg = configs.ExperimentConfig()
        exp_cfg.model.backbone = configs.BackboneType.CONFORMER
        exp_cfg.model.apply_size_variant(configs.SizeVariant.TINY)
        exp_cfg.model.seq_len = 50
        exp_cfg.model.embedding_dim = 64

        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(exp_cfg)
        model = hf_compat.FlaxSpeakerModel(config=hf_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "config.json")))
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "model.safetensors"))
            )
            loaded = hf_compat.FlaxSpeakerModel.from_pretrained(tmpdir)
            x = jnp.ones((1, 50, 80), dtype=jnp.float32)
            out1 = model(x)
            out2 = loaded(x)
            self.assertTrue(np.allclose(out1, out2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
