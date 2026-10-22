import functools
from multiprocessing import pool as mp_pool
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


def _released_weights_available(model_id: str = "univ_conformer_tiny_ge2e") -> bool:
    """True if the released checkpoint can be loaded without network access.

    `model.safetensors` is git-ignored, so a fresh clone has no local weights and
    `from_pretrained` would fall back to the Hub; tests that need real released weights
    skip when neither local weights nor Hub access (`HF_HUB_OFFLINE` unset and a cached
    snapshot or connectivity) are available.
    """
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models", model_id)
    if hf_compat.FlaxSpeakerModel._has_local_weights(local):
        return True
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
        return False
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=f"wq2012/{model_id}", allow_patterns=["config.json"])
        return True
    except Exception:
        return False


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

        for b_type in (
            configs.BackboneType.LSTM,
            configs.BackboneType.TRANSFORMER,
            configs.BackboneType.CONFORMER,
            configs.BackboneType.MAMBA,
            configs.BackboneType.ECAPA_TDNN,
            configs.BackboneType.RESNET,
        ):
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
            configs.PoolingType.ATP,
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

    def test_mask_padding_invariance_all_backbones(self):
        rng = jax.random.PRNGKey(0)
        x_short = jax.random.normal(rng, (2, 30, 80))
        m_short = jnp.ones((2, 30), dtype=jnp.float32)
        x_pad = jnp.pad(x_short, ((0, 0), (0, 30), (0, 0)))
        m_pad = jnp.pad(m_short, ((0, 0), (0, 30)))

        for b_type in (
            configs.BackboneType.CONFORMER,
            configs.BackboneType.TRANSFORMER,
            configs.BackboneType.LSTM,
            configs.BackboneType.MAMBA,
            configs.BackboneType.ECAPA_TDNN,
            configs.BackboneType.RESNET,
        ):
            cfg = configs.ExperimentConfig()
            cfg.model.backbone = b_type
            cfg.model.apply_size_variant(configs.SizeVariant.TINY)
            enc = neural_net.build_modern_encoder_from_config(cfg)
            params = enc.init(rng, x_short, mask=m_short)["params"]
            emb_short = enc.apply({"params": params}, x_short, mask=m_short)
            emb_pad = enc.apply({"params": params}, x_pad, mask=m_pad)
            cos = jnp.sum(emb_short * emb_pad, axis=-1)
            self.assertTrue(np.all(np.asarray(cos) > 0.9999))

    def test_mutually_exclusive_mtr_and_specaug_and_varlen(self):
        rng = np.random.default_rng(123)
        feats = np.ones((120, 80), dtype=np.float32)
        w_cfg = configs.WaveformAugConfig(use_waveform_aug=True, aug_prob=1.0)
        s_cfg = configs.SpecAugConfig(use_specaug=True, apply_prob=1.0)
        out, applied_mtr, applied_spec = specaug.apply_mutually_exclusive_augmentation(
            feats, waveform_aug_config=w_cfg, specaug_config=s_cfg, rng=rng
        )
        self.assertEqual(out.shape, (120, 80))
        self.assertFalse(applied_mtr and applied_spec)
        self.assertTrue(applied_mtr)
        self.assertFalse(applied_spec)


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
        with mp_pool.ThreadPool(self.config.train.num_processes) as pool:
            losses_out = neural_net.train_network(
                self.spk_to_utts, self.config, pool=pool
            )
        self.assertEqual(len(losses_out), 2)

    def test_train_transformer_network(self):
        self.config.model.use_transformer = True
        self.config.train.num_steps = 2
        with mp_pool.ThreadPool(self.config.train.num_processes) as pool:
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

    def test_hf_pretrained_model_ids_and_hub_resolution(self):
        from unittest import mock

        self.assertEqual(len(hf_compat.PRETRAINED_MODEL_IDS), 44)
        self.assertEqual(
            hf_compat.HF_COLLECTION_URL,
            "https://huggingface.co/collections/wq2012/flaxspeaker",
        )
        local_pm = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pretrained_models",
            "univ_conformer_tiny_ge2e",
        )
        if hf_compat.FlaxSpeakerModel._has_local_weights(local_pm) and _released_weights_available():
            m_local = hf_compat.load_pretrained(local_pm)
            m_hub = hf_compat.load_pretrained("wq2012/univ_conformer_tiny_ge2e")
            m_short = hf_compat.load_pretrained("univ_conformer_tiny_ge2e")
            x = jnp.ones((1, 50, 80), dtype=jnp.float32)
            out_local = m_local(x)
            out_hub = m_hub(x)
            out_short = m_short(x)
            self.assertTrue(np.allclose(out_local, out_hub, atol=1e-5))
            self.assertTrue(np.allclose(out_local, out_short, atol=1e-5))

        # Test Hub snapshot_download fallback when local directory lacks weights
        # (e.g. fresh git clone where .safetensors/.msgpack are gitignored) or force_hub=True
        exp_cfg = configs.ExperimentConfig()
        exp_cfg.model.backbone = configs.BackboneType.CONFORMER
        exp_cfg.model.apply_size_variant(configs.SizeVariant.TINY)
        exp_cfg.model.seq_len = 50
        exp_cfg.model.embedding_dim = 64
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(exp_cfg)
        ref_model = hf_compat.FlaxSpeakerModel(config=hf_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            hub_cache_dir = os.path.join(tmpdir, "hub_snapshot")
            ref_model.save_pretrained(hub_cache_dir)

            weightless_dir = os.path.join(
                tmpdir, "pretrained_models", "univ_conformer_tiny_ge2e"
            )
            os.makedirs(weightless_dir, exist_ok=True)
            hf_cfg.save_pretrained(weightless_dir)
            self.assertFalse(
                hf_compat.FlaxSpeakerModel._has_local_weights(weightless_dir)
            )

            with mock.patch(
                "huggingface_hub.snapshot_download",
                return_value=hub_cache_dir,
            ) as mock_dl:
                m_fallback = hf_compat.load_pretrained(weightless_dir, force_hub=True)
                mock_dl.assert_called_once_with(
                    repo_id="wq2012/univ_conformer_tiny_ge2e",
                    cache_dir=None,
                    revision=None,
                    local_files_only=False,
                )
                x = jnp.ones((1, 50, 80), dtype=jnp.float32)
                self.assertTrue(
                    np.allclose(ref_model(x), m_fallback(x), atol=1e-5)
                )

                # Test default force_hub=False on a fresh git clone (no local weights anywhere)
                mock_dl.reset_mock()
                with mock.patch.object(
                    hf_compat.FlaxSpeakerModel,
                    "_has_local_weights",
                    return_value=False,
                ):
                    hf_compat.load_pretrained(weightless_dir)
                    mock_dl.assert_called_once_with(
                        repo_id="wq2012/univ_conformer_tiny_ge2e",
                        cache_dir=None,
                        revision=None,
                        local_files_only=False,
                    )

                # Test trailing slashes and full Hugging Face URLs
                for ref_str, expected_repo in (
                    (
                        "wq2012/univ_conformer_tiny_ge2e/",
                        "wq2012/univ_conformer_tiny_ge2e",
                    ),
                    (
                        "https://huggingface.co/wq2012/univ_conformer_tiny_ge2e/",
                        "wq2012/univ_conformer_tiny_ge2e",
                    ),
                    (
                        "other_user/custom_model/",
                        "other_user/custom_model",
                    ),
                    (
                        "https://huggingface.co/other_user/custom_model",
                        "other_user/custom_model",
                    ),
                ):
                    mock_dl.reset_mock()
                    hf_compat.load_pretrained(ref_str, force_hub=True)
                    mock_dl.assert_called_once_with(
                        repo_id=expected_repo,
                        cache_dir=None,
                        revision=None,
                        local_files_only=False,
                    )

                # Invalid local paths must raise FileNotFoundError without calling snapshot_download
                mock_dl.reset_mock()
                with self.assertRaises(FileNotFoundError):
                    hf_compat.load_pretrained("/nonexistent/flaxspeaker/model_dir_xyz")
                with self.assertRaises(FileNotFoundError):
                    hf_compat.load_pretrained("./nonexistent_local_dir")
                mock_dl.assert_not_called()

    def test_hf_config_size_variant_and_multi_enroll_verify(self):
        # Verify FlaxSpeakerConfig.to_experiment_config() propagates size_variant
        for variant, expected_dim, expected_layers in (
            ("tiny", 128, 2),
            ("small", 192, 3),
            ("base", 256, 4),
        ):
            hf_cfg = hf_compat.FlaxSpeakerConfig(
                backbone="conformer", size_variant=variant
            )
            exp_cfg = hf_cfg.to_experiment_config()
            self.assertEqual(exp_cfg.model.conformer.dim, expected_dim)
            self.assertEqual(exp_cfg.model.conformer.num_layers, expected_layers)
            self.assertEqual(exp_cfg.model.embedding_dim, expected_dim)

        # Verify multi-utterance verify() and score_embeddings() across cosine, dr_vector, and pfas
        sr = 16000
        rng_np = np.random.default_rng(42)
        wav1 = rng_np.standard_normal(sr).astype(np.float32) * 0.2
        wav2 = rng_np.standard_normal(sr).astype(np.float32) * 0.2
        wav_test = rng_np.standard_normal(sr).astype(np.float32) * 0.2

        for scoring_mode, pool_mode in (
            ("cosine", "atp"),
            ("dr_vector", "atp"),
            ("pfas", "pfas"),
        ):
            exp_cfg = configs.ExperimentConfig()
            exp_cfg.model.backbone = configs.BackboneType.CONFORMER
            exp_cfg.model.apply_size_variant(configs.SizeVariant.TINY)
            exp_cfg.model.pooling.pooling_type = pool_mode
            exp_cfg.model.scoring_type = scoring_mode
            exp_cfg.train.loss.scoring_type = scoring_mode
            exp_cfg.train.loss.loss_type = "extended_set_softmax"
            hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(exp_cfg)
            model = hf_compat.FlaxSpeakerModel(config=hf_cfg)
            s_single = model.verify(wav1, wav_test)
            s_multi = model.verify([wav1, wav2], wav_test)
            self.assertTrue(np.isfinite(s_single))
            self.assertTrue(np.isfinite(s_multi))
            self.assertNotAlmostEqual(s_single, s_multi, places=4)


class TestAlgorithmicFidelityFixes(unittest.TestCase):

    def test_pfas_maxpool_and_attention_padding_invariance(self):
        rng = jax.random.PRNGKey(123)
        x_short = jax.random.normal(rng, (2, 32, 64))
        m_short = jnp.ones((2, 32), dtype=jnp.float32)
        x_pad = jnp.pad(x_short, ((0, 0), (0, 28), (0, 0)))
        m_pad = jnp.pad(m_short, ((0, 0), (0, 28)))

        for attn_type in ("mlp", "dot", "maxpool"):
            for use_anchor in (True, False):
                head = pooling.PfasRepresentationHead(
                    num_keys=4,
                    key_dim=32,
                    value_dim=128,
                    attention_dim=64,
                    attention_type=attn_type,
                    use_global_anchor=use_anchor,
                )
                params = head.init(rng, x_short, mask=m_short)["params"]
                out_short = head.apply(
                    {"params": params}, x_short, mask=m_short
                )
                out_pad = head.apply({"params": params}, x_pad, mask=m_pad)
                max_diff = float(jnp.max(jnp.abs(out_short - out_pad)))
                self.assertLess(
                    max_diff,
                    1e-5,
                    msg=f"PFAS padding invariance failed for {attn_type} (anchor={use_anchor}): {max_diff}",
                )

    def test_ge2e_contrast_non_saturating_grad_and_unclamped_w(self):
        rng = jax.random.PRNGKey(99)
        embs = jax.random.normal(rng, (16, 32))
        embs = embs / (jnp.linalg.norm(embs, axis=-1, keepdims=True) + 1e-6)
        w10 = jnp.array(10.0, dtype=jnp.float32)
        w20 = jnp.array(20.0, dtype=jnp.float32)
        b = jnp.array(-5.0, dtype=jnp.float32)

        l10 = losses.ge2e_contrast_loss(embs, 4, 4, w=w10, b=b)
        l20 = losses.ge2e_contrast_loss(embs, 4, 4, w=w20, b=b)
        # Verify w > 5.0 is NOT clamped to 5.0
        self.assertNotAlmostEqual(float(l10), float(l20), places=3)

        grad_fn = jax.grad(losses.ge2e_contrast_loss, argnums=(0, 3, 4))
        g_embs, g_w, g_b = grad_fn(embs, 4, 4, w10, b)
        self.assertGreater(float(jnp.linalg.norm(g_embs)), 1e-2)
        self.assertTrue(np.isfinite(float(g_w)))
        self.assertTrue(np.isfinite(float(g_b)))

    def test_extended_set_softmax_pure_vs_blended_and_dr_vector_terms(self):
        rng = jax.random.PRNGKey(17)
        embs = jax.random.normal(rng, (16, 32))
        w = jnp.array(10.0, dtype=jnp.float32)
        b = jnp.array(-5.0, dtype=jnp.float32)

        l_blended = losses.extended_set_softmax_loss(
            embs, 4, 4, w=w, b=b, pure_extended_set=False
        )
        l_pure = losses.extended_set_softmax_loss(
            embs, 4, 4, w=w, b=b, pure_extended_set=True
        )
        self.assertGreater(float(l_blended), 0.0)
        self.assertGreater(float(l_pure), 0.0)
        self.assertNotAlmostEqual(float(l_blended), float(l_pure), places=3)

        # Test DecisionResidualNetwork with 3 cosine terms (Pelecanos et al., 2021 Eq. 7) and symmetric input
        dr_net = scoring.DecisionResidualNetwork(
            hidden_dims=(32, 16),
            num_cosine_terms=3,
            symmetric_input=True,
        )
        t_v = jax.random.normal(rng, (4, 32))
        e_v = jax.random.normal(jax.random.PRNGKey(18), (4, 32))
        params = dr_net.init(rng, t_v, e_v)["params"]
        s_te = dr_net.apply({"params": params}, t_v, e_v, method=dr_net.score_pairs)
        s_et = dr_net.apply({"params": params}, e_v, t_v, method=dr_net.score_pairs)
        self.assertTrue(np.allclose(s_te, s_et, atol=1e-5))

    def test_bidirectional_mamba_padding_invariance(self):
        rng = jax.random.PRNGKey(55)
        x_short = jax.random.normal(rng, (2, 24, 80))
        m_short = jnp.ones((2, 24), dtype=jnp.float32)
        x_pad = jnp.pad(x_short, ((0, 0), (0, 16), (0, 0)))
        m_pad = jnp.pad(m_short, ((0, 0), (0, 16)))

        cfg = configs.ExperimentConfig()
        cfg.model.backbone = configs.BackboneType.MAMBA
        cfg.model.apply_size_variant(configs.SizeVariant.TINY)
        cfg.model.mamba.bidirectional = True
        enc = neural_net.build_modern_encoder_from_config(cfg)
        params = enc.init(rng, x_short, mask=m_short)["params"]
        out_short = enc.apply({"params": params}, x_short, mask=m_short)
        out_pad = enc.apply({"params": params}, x_pad, mask=m_pad)
        cos = jnp.sum(out_short * out_pad, axis=-1)
        self.assertTrue(np.all(np.asarray(cos) > 0.9999))

    def test_frontend_agc_ecapa_norm_order_and_eval_cap(self):
        wav = np.sin(np.linspace(0.0, 40.0 * np.pi, 16000)).astype(np.float32) * 0.1
        fe_cfg = configs.FrontendConfig(apply_cmvn=False)
        fe_no_agc = feature_extraction.AudioFrontend(
            fe_cfg, configs.VadConfig(enabled=False, apply_agc=False)
        )
        fe_with_agc = feature_extraction.AudioFrontend(
            fe_cfg, configs.VadConfig(enabled=False, apply_agc=True)
        )
        f_no = fe_no_agc.extract_from_waveform(wav)
        f_yes = fe_with_agc.extract_from_waveform(wav)
        self.assertGreater(float(np.max(f_yes - f_no)), 2.0)
        self.assertGreater(float(np.mean(f_yes - f_no)), 0.2)

        rng = jax.random.PRNGKey(77)
        cfg = configs.ExperimentConfig()
        cfg.model.backbone = configs.BackboneType.ECAPA_TDNN
        cfg.model.apply_size_variant(configs.SizeVariant.TINY)
        cfg.model.ecapa_tdnn.norm_after_activation = True
        cfg.model.max_eval_seq_len = 64
        _, state = neural_net.get_speaker_encoder(cfg)
        long_feats = jax.random.normal(rng, (120, 80))
        emb_long = evaluation.run_inference(long_feats, state, cfg)
        emb_crop = evaluation.run_inference(long_feats[28:92], state, cfg)
        self.assertTrue(np.allclose(np.asarray(emb_long), np.asarray(emb_crop), atol=1e-5))
        self.assertEqual(
            dataset.resolve_data_path("hf_space/examples/librispeech_spk1_utt1.flac"),
            "hf_space/examples/librispeech_spk1_utt1.flac",
        )

    def test_hf_space_gradio_app_verification(self):
        from hf_space import app as hf_app  # pytype: disable=import-error

        models = hf_app.list_available_models()
        self.assertGreaterEqual(len(models), 44)
        if not _released_weights_available("univ_conformer_small_ge2e"):
            self.skipTest("released weights unavailable offline (no local safetensors, no Hub)")
        p1 = "hf_space/examples/librispeech_spk1_utt1.flac"
        p2 = "hf_space/examples/librispeech_spk1_utt2.flac"
        # Default threshold = the model's own pooled EER operating point for the
        # enrollment count actually used (1-utt for one, 3-utt for two or three).
        ops = hf_app.get_default_threshold("univ_conformer_small_ge2e")
        self.assertIn("pooled", ops)
        self.assertIn("pooled_3utt", ops)
        self.assertGreater(ops["pooled"], 0.0)
        self.assertNotEqual(ops["pooled"], ops["pooled_3utt"])
        verdict, details, fig = hf_app.verify_pair(
            p1,
            p2,
            "univ_conformer_small_ge2e",
            None,  # -> per-model default threshold
            enroll_audio_2=p1,
            use_vad=True,
            vad_snr_margin_db=4.5,
        )
        self.assertIn("SPEAKER", verdict)
        self.assertEqual(details["num_enrollment_utterances"], 2)
        self.assertAlmostEqual(details["threshold"], round(ops["pooled_3utt"], 4), places=4)
        self.assertIn("pooled_3utt", details["threshold_source"])
        self.assertIn("model_inference_latency_ms", details)
        self.assertNotIn("as_norm_applied", details)
        self.assertIsNotNone(fig)
        # Single enrollment -> 1-utt operating point; manual slider honoured when
        # `auto_threshold=False`; both VAD knobs are reported.
        _, details_1, _ = hf_app.verify_pair(p1, p2, "univ_conformer_small_ge2e", 0.123)
        self.assertEqual(details_1["num_enrollment_utterances"], 1)
        self.assertAlmostEqual(details_1["threshold"], round(ops["pooled"], 4), places=4)
        _, details_m, _ = hf_app.verify_pair(
            p1, p2, "univ_conformer_small_ge2e", 0.123, auto_threshold=False, vad_top_db=30.0
        )
        self.assertAlmostEqual(details_m["threshold"], 0.123, places=4)
        self.assertEqual(details_m["threshold_source"], "manual slider")
        self.assertAlmostEqual(details_m["vad_top_db"], 30.0)
        self.assertAlmostEqual(details_m["vad_snr_margin_db"], 4.5)

    def test_encode_features_matches_run_inference_and_eer_threshold(self):
        """`encode_features` (masked fixed window; one compiled kernel) == `run_inference`."""
        rng = jax.random.PRNGKey(909)
        cfg = configs.ExperimentConfig()
        cfg.model.backbone = configs.BackboneType.CONFORMER
        cfg.model.apply_size_variant(configs.SizeVariant.TINY)
        cfg.model.max_eval_seq_len = 96
        _, state = neural_net.get_speaker_encoder(cfg)
        # Two branches where `run_inference` and the benchmark path agree: shorter than
        # the window (zero-padded + masked) and longer than the window (centre-cropped).
        for n_frames in (70, 150):
            feats = jax.random.normal(jax.random.fold_in(rng, n_frames), (n_frames, 80))
            ref = np.asarray(evaluation.run_inference(feats, state, cfg))
            got = evaluation.encode_features(np.asarray(feats), state, cfg)
            padded, mask = evaluation.prepare_eval_input(np.asarray(feats), cfg)
            self.assertEqual(padded.shape, (96, 80))
            self.assertEqual(mask.shape, (96,))
            self.assertEqual(int(mask.sum()), min(n_frames, 96))
            self.assertTrue(
                np.allclose(ref, got, atol=1e-4),
                msg=f"n_frames={n_frames}: max diff {np.max(np.abs(ref - got))}",
            )
        # Utterances shorter than 32 frames are tiled and truncated to 32 valid frames
        # in both `prepare_eval_input` / `encode_features` and `run_inference`.
        short = np.asarray(jax.random.normal(jax.random.fold_in(rng, 10), (10, 80)))
        padded, mask = evaluation.prepare_eval_input(short, cfg)
        self.assertEqual(int(mask.sum()), 32)
        self.assertTrue(np.allclose(padded[:10], short) and np.allclose(padded[10:20], short))
        got = evaluation.encode_features(short, state, cfg)
        ref_short = np.asarray(evaluation.run_inference(short, state, cfg))
        self.assertEqual(got.shape, (cfg.model.output_embedding_dim,))
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertTrue(np.allclose(ref_short, got, atol=1e-4))
        # Checkpoint EER operating point: absent -> None; present -> the archived value.
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(cfg)
        model = hf_compat.FlaxSpeakerModel(config=hf_cfg, params=state.params)
        self.assertIsNone(model.eer_threshold())
        model.eval_results = {
            "multi_domain_eval": {"domains_1utt": {"universal_pooled": {"eer_threshold": 0.625}}}
        }
        self.assertAlmostEqual(model.eer_threshold(), 0.625)
        self.assertIsNone(model.eer_threshold("librispeech"))
        if not _released_weights_available():
            self.skipTest("released weights unavailable offline (no local safetensors, no Hub)")
        released = hf_compat.FlaxSpeakerModel.from_pretrained(
            "pretrained_models/univ_conformer_tiny_ge2e"
        )
        self.assertIsNotNone(released.eval_results)
        thr = released.eer_threshold()
        self.assertIsNotNone(thr)
        self.assertGreater(thr, 0.0)
        self.assertLess(thr, 1.0)

    def test_ge2e_contrast_loss_wan2018_eq7(self):
        rng = jax.random.PRNGKey(303)
        n_spk, n_utt, dim = 4, 3, 16
        embs = jax.random.normal(rng, (n_spk * n_utt, dim))
        embs = embs / (jnp.linalg.norm(embs, axis=-1, keepdims=True) + 1e-6)
        w = jnp.array(8.0, dtype=jnp.float32)
        b = jnp.array(-2.0, dtype=jnp.float32)

        scaled, pos_idx = losses.compute_ge2e_similarity_matrix_leave_one_out(
            embs, n_spk, n_utt, w=w, b=b, exact_leave_one_out=True
        )
        pos_logits = scaled[jnp.arange(n_spk * n_utt), pos_idx]
        neg_mask = jax.nn.one_hot(pos_idx, n_spk) > 0.5
        neg_logits = jnp.where(neg_mask, -1e9, scaled)
        max_neg_logits = jnp.max(neg_logits, axis=-1)
        expected = jnp.mean(
            1.0 - jax.nn.sigmoid(pos_logits) + jax.nn.sigmoid(max_neg_logits)
        )
        actual = losses.ge2e_contrast_loss(
            embs, n_spk, n_utt, w=w, b=b, exact_leave_one_out=True
        )
        self.assertAlmostEqual(float(expected), float(actual), places=5)

    def test_atp_all_pooling_stats_modes_and_dr_skip_concat(self):
        rng = jax.random.PRNGKey(404)
        x = jax.random.normal(rng, (2, 20, 16))
        mask = jnp.ones((2, 20), dtype=jnp.float32)
        expected_dims = {
            "mean": 16,
            "std": 16,
            "var": 16,
            "mean_std": 32,
            "mean_var": 32,
        }
        for stype, exp_d in expected_dims.items():
            pool = pooling.AttentiveTemporalPooling(
                attention_dim=8,
                num_heads=4,
                subspace_heads=True,
                pooling_stats=stype,
            )
            params = pool.init(rng, x, mask=mask)["params"]
            out = pool.apply({"params": params}, x, mask=mask)
            self.assertEqual(out.shape, (2, exp_d))

        dr_skip = scoring.DecisionResidualNetwork(
            hidden_dims=(32, 16),
            num_cosine_terms=3,
            append_cosine_to_final_hidden=True,
        )
        t_v = jax.random.normal(rng, (3, 24))
        e_v = jax.random.normal(jax.random.PRNGKey(405), (3, 24))
        dr_params = dr_skip.init(rng, t_v, e_v)["params"]
        self.assertEqual(dr_params["dr_out"]["kernel"].shape, (16 + 3, 1))
        scores = dr_skip.apply(
            {"params": dr_params}, t_v, e_v, method=dr_skip.score_pairs
        )
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.all(np.isfinite(np.asarray(scores))))

    def test_paper_presets_and_foreign_path_resolution(self):
        c_ge2e = configs.icassp2018_ge2e_config(paper_faithful=True)
        self.assertIs(c_ge2e.loss, c_ge2e.train.loss)
        self.assertTrue(c_ge2e.train.loss.ge2e_exact_leave_one_out)
        self.assertEqual(c_ge2e.model.pooling.pooling_type, "last")
        self.assertEqual(c_ge2e.model.lstm.hidden_size, 768)
        self.assertEqual(c_ge2e.model.lstm.proj_size, 256)
        self.assertEqual(c_ge2e.model.embedding_dim, 256)
        self.assertFalse(c_ge2e.model.pooling.use_nonlinear_bottleneck)

        c_dr = configs.interspeech2021_dr_vectors_config(pure_paper_mode=True)
        self.assertIs(c_dr.loss, c_dr.train.loss)
        # Paper (arXiv:2104.01989): no 3-term cosine expansion, d=200 base dims,
        # 3 x 256 leaky-ReLU(0.2), weighted sum, trainable affine, pure Eq. 2.
        self.assertEqual(c_dr.model.dr_vector.num_cosine_terms, 1)
        self.assertEqual(c_dr.model.dr_vector.cosine_dims, 200)
        self.assertEqual(tuple(c_dr.model.dr_vector.hidden_dims), (256, 256, 256))
        self.assertEqual(c_dr.model.dr_vector.activation, "leaky_relu")
        self.assertFalse(c_dr.model.dr_vector.use_layer_norm)
        self.assertEqual(c_dr.model.dr_vector.readout, "weighted_sum")
        self.assertTrue(c_dr.model.dr_vector.trainable_affine)
        self.assertTrue(c_dr.train.loss.pure_extended_set)

        c_atp = configs.odyssey2022_conformer_atp_config(pure_paper_mode=True)
        self.assertIs(c_atp.loss, c_atp.train.loss)
        self.assertEqual(c_atp.model.conformer.layer_order, "conv_before_mhsa")
        self.assertFalse(c_atp.model.conformer.use_conv_stem)
        self.assertTrue(c_atp.model.conformer.use_positional_encoding)
        self.assertEqual(c_atp.model.conformer.subsample_factor, 2)
        self.assertEqual(c_atp.model.pooling.num_heads, 1)
        self.assertEqual(c_atp.model.pooling.attention_dim, 0)

        c_pfas = configs.odyssey2022_pfas_config(pure_paper_mode=True)
        self.assertIs(c_pfas.loss, c_pfas.train.loss)
        self.assertEqual(c_pfas.model.pfas.attention_type, "atp_ffn")
        self.assertEqual(c_pfas.model.pfas.num_keys, 32)
        self.assertEqual(c_pfas.model.pfas.representation_dim, 2048)
        self.assertFalse(c_pfas.model.pfas.use_global_anchor)
        self.assertTrue(c_pfas.model.pfas.apply_global_l2_norm_to_concat_form)
        self.assertEqual(c_pfas.train.loss.loss_type, "extended_set_softmax")
        self.assertTrue(c_pfas.train.loss.pure_extended_set)

        with tempfile.TemporaryDirectory() as tmpdir:
            splits_dir = os.path.join(tmpdir, "splits")
            os.makedirs(splits_dir, exist_ok=True)
            dummy_csv = os.path.join(splits_dir, "dummy.csv")
            with open(dummy_csv, "w", encoding="utf-8") as f:
                f.write("spk1,/other_host/data/speaker_datasets/splits/dummy.csv\n")
            old_env = os.environ.get("FLAXSPEAKER_DATA_ROOT")
            os.environ["FLAXSPEAKER_DATA_ROOT"] = tmpdir
            try:
                resolved = dataset.resolve_data_path(
                    "/other_host/data/speaker_datasets/splits/dummy.csv"
                )
                self.assertEqual(resolved, dummy_csv)
                spk_map = dataset.get_csv_spk_to_utts(
                    "/other_host/data/speaker_datasets/splits/dummy.csv"
                )
                self.assertEqual(spk_map["spk1"], [dummy_csv])
            finally:
                if old_env is None:
                    os.environ.pop("FLAXSPEAKER_DATA_ROOT", None)
                else:
                    os.environ["FLAXSPEAKER_DATA_ROOT"] = old_env


class TestRound2NumericReferences(unittest.TestCase):
    """Numeric-reference tests added in Round 2 (lifted from the reviewers' checks)."""

    def _padding_invariance(self, module, x, valid, atol):
        params = module.init(jax.random.PRNGKey(1), x)
        full = module.apply(params, x[:, :valid], mask=jnp.ones((x.shape[0], valid)))
        mask = jnp.concatenate(
            [jnp.ones((x.shape[0], valid)), jnp.zeros((x.shape[0], x.shape[1] - valid))],
            axis=1,
        )
        padded = module.apply(params, x, mask=mask)
        if padded.ndim == 3:  # frame outputs
            diff = float(jnp.max(jnp.abs(padded[:, :valid] - full)))
            tail = float(jnp.max(jnp.abs(padded[:, valid:])))
            self.assertLessEqual(tail, atol)
        else:  # pooled embeddings
            diff = float(jnp.max(jnp.abs(padded - full)))
        self.assertLessEqual(diff, atol, f"padding leaked: max diff {diff}")

    def test_padding_invariance_all_seven_backbone_variants(self):
        from flaxspeaker import backbones

        x = jax.random.normal(jax.random.PRNGKey(0), (2, 24, 16))
        variants = {
            "lstm_uni": backbones.LstmBackbone(hidden_size=16, num_layers=2),
            "lstm_bi": backbones.LstmBackbone(hidden_size=16, num_layers=2, bidirectional=True),
            "conformer": backbones.ConformerBackbone(dim=16, num_heads=2, num_layers=2, ffn_dim=32),
            "transformer": backbones.TransformerBackbone(dim=16, num_heads=2, num_layers=2, mlp_dim=32),
            "mamba_uni": backbones.MambaBackbone(dim=16, num_layers=2, d_state=4, dt_rank=4),
            "mamba_bi": backbones.MambaBackbone(dim=16, num_layers=2, d_state=4, dt_rank=4, bidirectional=True),
            "ecapa": backbones.EcapaTdnnBackbone(channels=16, mfa_dim=32, se_bottleneck_dim=8),
        }
        for name, module in variants.items():
            with self.subTest(backbone=name):
                self._padding_invariance(module, x, valid=15, atol=5e-5)

    def test_bidirectional_lstm_backward_direction_is_time_aligned(self):
        """`keep_order=True`: the backward half at frame t must depend on frames >= t only."""
        from flaxspeaker import backbones

        module = backbones.LstmBackbone(hidden_size=8, num_layers=1, bidirectional=True)
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 10, 4))
        params = module.init(jax.random.PRNGKey(1), x)
        # Perturbing the *last* frame must change the backward state at frame 0 ...
        x_pert = x.at[0, -1].add(1.0)
        out = module.apply(params, x, mask=jnp.ones((1, 10)))
        out_p = module.apply(params, x_pert, mask=jnp.ones((1, 10)))
        self.assertGreater(float(jnp.max(jnp.abs(out[0, 0] - out_p[0, 0]))), 1e-6)
        # ... but perturbing the *first* frame must not change the backward direction at
        # the last frame (the backward recurrence at t=T-1 has seen only frame T-1). The
        # per-layer LayerNorm mixes fwd/bwd halves, so test the raw reversed RNN with the
        # exact construction used in `LstmBackbone` (reverse=True, keep_order=True).
        from flax import linen as nn

        bwd = nn.RNN(nn.OptimizedLSTMCell(features=8), reverse=True, keep_order=True)
        bp = bwd.init(jax.random.PRNGKey(2), x, seq_lengths=jnp.array([10]))
        y = bwd.apply(bp, x, seq_lengths=jnp.array([10]))
        y_p = bwd.apply(bp, x.at[0, 0].add(1.0), seq_lengths=jnp.array([10]))
        self.assertLess(float(jnp.max(jnp.abs(y[0, -1] - y_p[0, -1]))), 1e-6)
        self.assertGreater(float(jnp.max(jnp.abs(y[0, 0] - y_p[0, 0]))), 1e-6)
        # With a shorter valid length the backward pass must start at the last *valid* frame.
        y_short = bwd.apply(bp, x, seq_lengths=jnp.array([6]))
        y_ref = bwd.apply(bp, x[:, :6], seq_lengths=jnp.array([6]))
        self.assertLess(float(jnp.max(jnp.abs(y_short[0, :6] - y_ref[0]))), 1e-6)

    def test_contrast_loss_pure_vs_surrogate_gradients(self):
        embs = jax.random.normal(jax.random.PRNGKey(3), (16, 12))
        embs = embs / jnp.linalg.norm(embs, axis=-1, keepdims=True)
        w, b = jnp.array(10.0), jnp.array(-5.0)

        def pure(e):
            return losses.ge2e_contrast_loss(e, 4, 4, w, b, surrogate=False)

        def surrogate(e):
            return losses.ge2e_contrast_loss(e, 4, 4, w, b, surrogate=True)

        # Same value, different gradient.
        self.assertAlmostEqual(float(pure(embs)), float(surrogate(embs)), places=6)
        g_pure = jax.grad(pure)(embs)
        g_sur = jax.grad(surrogate)(embs)

        # Pure gradient equals the analytic Eq. 7 gradient computed independently.
        def eq7_reference(e):
            sim, labels = losses.compute_ge2e_similarity_matrix_leave_one_out(
                e, 4, 4, jnp.maximum(jnp.abs(w), 1e-3), b, exact_leave_one_out=True
            )
            sig = jax.nn.sigmoid(sim)
            tm = jax.nn.one_hot(labels, 4)
            pos = jnp.sum(sig * tm, -1)
            neg = jnp.max(jnp.where(tm > 0, -1e9, sig), -1)
            return jnp.mean(1 - pos + neg)

        g_ref = jax.grad(eq7_reference)(embs)
        self.assertLess(float(jnp.max(jnp.abs(g_pure - g_ref))), 1e-5)
        cos = float(jnp.sum(g_pure * g_sur) / (jnp.linalg.norm(g_pure) * jnp.linalg.norm(g_sur)))
        self.assertLess(cos, 0.99, "surrogate gradient should differ from pure Eq. 7")

    def test_unknown_config_key_raises_and_deprecated_key_warns(self):
        with self.assertRaises(ValueError):
            configs.ExperimentConfig.from_dict({"model": {"conformer": {"nmu_layers": 3}}})
        with self.assertRaises(ValueError):
            configs.ExperimentConfig.from_dict({"train": {"lerning_rate": 1e-3}})
        import glob
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = configs.ExperimentConfig.from_dict({"loss": {"num_classes": 256}})
        self.assertTrue(any(issubclass(c.category, DeprecationWarning) for c in caught))
        self.assertFalse(hasattr(cfg.train.loss, "num_classes"))
        # Every shipped YAML loads cleanly (no deprecated keys left).
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            for p in glob.glob(os.path.join(os.path.dirname(__file__), "configs", "*.yml")):
                configs.ExperimentConfig.from_yaml(p)

    def test_pure_paper_mode_fields_change_the_built_module(self):
        """Every field a pure preset sets must be consumed by the built encoder."""
        x = jnp.ones((1, 40, 80))

        def n_params_and_shape(cfg):
            enc = neural_net.build_modern_encoder_from_config(cfg)
            p = enc.init(jax.random.PRNGKey(0), x)["params"]
            return neural_net.count_parameters(p), jax.tree_util.tree_structure(p)

        # Conformer-ATP: conv_before_mhsa / positional encoding / no stem / subsample /
        # single-head linear ATP all change parameters or outputs.
        pure = configs.odyssey2022_conformer_atp_config(pure_paper_mode=True)
        base = configs.odyssey2022_conformer_atp_config(pure_paper_mode=False)
        self.assertNotEqual(n_params_and_shape(pure)[1], n_params_and_shape(base)[1])
        enc_pure = neural_net.build_modern_encoder_from_config(pure)
        p_pure = enc_pure.init(jax.random.PRNGKey(0), x)["params"]
        self.assertIn("subsample_proj", p_pure["backbone"])
        self.assertNotIn("stem_conv", p_pure["backbone"])
        self.assertNotIn("atp_hid", p_pure["pool"])  # attention_dim = 0 -> linear sigmoid
        self.assertEqual(p_pure["pool"]["atp_ffn"]["kernel"].shape[-1], 1)  # single head
        # layer_order changes the forward output for identical parameters.
        swapped = configs.odyssey2022_conformer_atp_config(pure_paper_mode=True)
        swapped.model.conformer.layer_order = "mhsa_before_conv"
        enc_sw = neural_net.build_modern_encoder_from_config(swapped)
        xr = jax.random.normal(jax.random.PRNGKey(5), (1, 40, 80))
        self.assertGreater(
            float(jnp.max(jnp.abs(enc_pure.apply({"params": p_pure}, xr) - enc_sw.apply({"params": p_pure}, xr)))),
            1e-6,
        )
        # positional encoding changes outputs for identical parameters.
        nope = configs.odyssey2022_conformer_atp_config(pure_paper_mode=True)
        nope.model.conformer.use_positional_encoding = False
        enc_np = neural_net.build_modern_encoder_from_config(nope)
        self.assertGreater(
            float(jnp.max(jnp.abs(enc_pure.apply({"params": p_pure}, xr) - enc_np.apply({"params": p_pure}, xr)))),
            1e-6,
        )
        # PFAS pure: atp_ffn head, 32 keys, no LayerNorm.
        pf = configs.odyssey2022_pfas_config(pure_paper_mode=True)
        enc_pf = neural_net.build_modern_encoder_from_config(pf)
        p_pf = enc_pf.init(jax.random.PRNGKey(0), x)["params"]["pfas_head"]
        self.assertIn("pfas_kv_out", p_pf)
        self.assertNotIn("pfas_pool_ln", p_pf)
        self.assertEqual(p_pf["pfas_kv_out"]["kernel"].shape[-1], 32 * (16 + 48))
        full_pf = enc_pf.init(jax.random.PRNGKey(0), x)
        self.assertEqual(enc_pf.apply(full_pf, x).shape, (1, 2048))
        # Dr-Vectors pure: trainable affine params exist, cosine over 200 dims.
        dr = configs.interspeech2021_dr_vectors_config(pure_paper_mode=True)
        _, st = neural_net.get_speaker_encoder(dr, num_classes=0)
        self.assertIn("dr_affine_scale", st.params["_aux_dr_net"])
        self.assertEqual(st.params["_aux_dr_net"]["dr_fc_0"]["kernel"].shape[-1], 256)
        self.assertEqual(st.params["_aux_dr_net"]["dr_out"]["kernel"].shape[0], 3 * 256)
        # GE2E pure: 768-cell LSTM with 256 projection, plain linear output.
        ge = configs.icassp2018_ge2e_config(pure_paper_mode=True)
        _, st_ge = neural_net.get_speaker_encoder(ge, num_classes=0)
        self.assertEqual(st_ge.params["backbone"]["lstm_proj_0"]["kernel"].shape, (768, 256))
        self.assertNotIn("bottleneck_fc", st_ge.params)

    def test_conformer_subsampling_shapes_and_mask_invariance(self):
        from flaxspeaker import backbones

        module = backbones.ConformerBackbone(
            dim=16, num_heads=2, num_layers=3, ffn_dim=32, subsample_factor=2, subsample_after_layer=1
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 21, 8))
        params = module.init(jax.random.PRNGKey(1), x)
        out = module.apply(params, x, mask=jnp.ones((2, 21)))
        self.assertEqual(out.shape, (2, 11, 16))
        self.assertEqual(backbones.subsample_mask(jnp.ones((2, 21)), 2).shape, (2, 11))
        # Padding invariance on the subsampled frames (valid = 14 -> 7 groups).
        full = module.apply(params, x[:, :14], mask=jnp.ones((2, 14)))
        mask = jnp.concatenate([jnp.ones((2, 14)), jnp.zeros((2, 7))], axis=1)
        padded = module.apply(params, x, mask=mask)
        self.assertLess(float(jnp.max(jnp.abs(padded[:, :7] - full))), 1e-4)
        self.assertLess(float(jnp.max(jnp.abs(padded[:, 7:]))), 1e-6)
        # Unsupported layer order is rejected.
        bad = backbones.ConformerBackbone(dim=16, num_heads=2, num_layers=1, ffn_dim=32, layer_order="foo")
        with self.assertRaises(ValueError):
            bad.init(jax.random.PRNGKey(1), x)

    def test_contrast_loss_with_pfas_scoring_is_rejected(self):
        cfg = configs.odyssey2022_pfas_config(loss_type="ge2e_contrast", pure_paper_mode=False)
        with self.assertRaises(ValueError):
            neural_net.make_modern_train_step(cfg)

    def test_as_norm_requires_disjoint_cohort(self):
        rng = np.random.default_rng(0)
        utt_to_emb = {f"u{i}": rng.standard_normal(8).astype(np.float32) for i in range(12)}
        with self.assertRaises(ValueError):
            evaluation.require_disjoint_cohort(None, utt_to_emb)
        with self.assertRaises(ValueError):  # too small
            evaluation.require_disjoint_cohort(rng.standard_normal((3, 8)), utt_to_emb)
        with self.assertRaises(ValueError):  # cohort == eval set
            evaluation.require_disjoint_cohort(np.stack(list(utt_to_emb.values())), utt_to_emb)
        cohort = evaluation.require_disjoint_cohort(rng.standard_normal((20, 8)), utt_to_emb)
        self.assertEqual(cohort.shape, (20, 8))
        # End-to-end: evaluating with AS-Norm on and no cohort raises.
        cfg = configs.ExperimentConfig()
        cfg.eval.use_as_norm = True
        cfg.model.apply_size_variant("tiny")
        _, state = neural_net.get_speaker_encoder(cfg, num_classes=0)
        emb = {f"u{i}": rng.standard_normal(cfg.model.embedding_dim).astype(np.float32) for i in range(12)}
        trials = [(1, "u0", "u1"), (0, "u2", "u3"), (1, "u4", "u5"), (0, "u6", "u7")]
        with self.assertRaises(ValueError):
            evaluation.evaluate_verification_trials(
                state, trials, cfg, precomputed_embeddings=emb
            )

    def test_export_window_resolution_has_no_sentinel(self):
        cfg = configs.ExperimentConfig()
        cfg.model.seq_len = 180
        cfg.export.sequence_length = 0
        self.assertEqual(export._resolve_export_seq_len(cfg), 180)
        cfg.export.sequence_length = 100  # used to be a magic "unset" value
        self.assertEqual(export._resolve_export_seq_len(cfg), 100)
        cfg.export.sequence_length = 250
        self.assertEqual(export._resolve_export_seq_len(cfg), 250)
        self.assertEqual(export._resolve_export_seq_len(cfg, seq_len=64), 64)
        # Every shipped config declares the 180-frame window the TFLite files were traced with.
        import glob

        for p in glob.glob(os.path.join(os.path.dirname(__file__), "configs", "*.yml")):
            c = configs.ExperimentConfig.from_yaml(p)
            self.assertEqual(export._resolve_export_seq_len(c), 180, p)

    def test_min_learning_rate_is_honoured(self):
        import optax

        cfg = configs.ExperimentConfig()
        cfg.train.learning_rate = 1e-3
        cfg.train.min_learning_rate = 1e-6
        cfg.train.num_steps = 100
        cfg.train.warmup_steps = 10
        cfg.train.lr_schedule_cosine = True
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-4, peak_value=1e-3, warmup_steps=10, decay_steps=100, end_value=1e-6
        )
        tx = neural_net.build_optimizer(cfg)
        params = {"w": jnp.ones((2, 2))}
        state = tx.init(params)
        # Drive the optimizer to the last step and compare the applied LR with the schedule.
        g = {"w": jnp.ones((2, 2))}
        for _ in range(100):
            updates, state = tx.update(g, state, params)
        # AdamW with constant unit gradients gives update magnitude ~ lr at convergence.
        self.assertAlmostEqual(float(-updates["w"][0, 0]), float(schedule(99)), delta=2e-7)

    def test_decision_residual_network_paper_options(self):
        rng = jax.random.PRNGKey(0)
        net = scoring.DecisionResidualNetwork(
            hidden_dims=(256, 256, 256), activation="leaky_relu", use_layer_norm=False,
            cosine_dims=200, readout="weighted_sum", trainable_affine=True, residual_weight=1.0,
        )
        t = jax.random.normal(rng, (5, 256))
        e = jax.random.normal(jax.random.PRNGKey(1), (5, 256))
        params = net.init(rng, t, e)["params"]
        self.assertEqual(params["dr_out"]["kernel"].shape, (3 * 256, 1))
        self.assertIn("dr_affine_scale", params)
        self.assertNotIn("dr_ln_0", params)
        scores = net.apply({"params": params}, t, e, method=net.score_pairs)
        # With dr_out ~ 0 at init and affine = identity, score ~ cosine over first 200 dims.
        tb = t[:, :200] / jnp.linalg.norm(t[:, :200], axis=-1, keepdims=True)
        eb = e[:, :200] / jnp.linalg.norm(e[:, :200], axis=-1, keepdims=True)
        self.assertLess(float(jnp.max(jnp.abs(scores - jnp.sum(tb * eb, -1)))), 0.1)
        with self.assertRaises(ValueError):
            scoring.DecisionResidualNetwork(readout="bogus").init(rng, t, e)
        with self.assertRaises(ValueError):
            scoring.DecisionResidualNetwork(cosine_dims=300).init(rng, t, e)

    def test_hf_model_score_embeddings_pure_paper_dr_vectors_and_extract_overrides(self):
        """FlaxSpeakerModel.score_embeddings must forward all paper Dr-Vector fields."""
        cfg = configs.interspeech2021_dr_vectors_config(pure_paper_mode=True)
        cfg.model.apply_size_variant("tiny")
        cfg.model.embedding_dim = 256
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(cfg)
        model = hf_compat.FlaxSpeakerModel(config=hf_cfg)
        e1 = np.ones((256,), dtype=np.float32) / np.sqrt(256.0)
        e2 = np.ones((256,), dtype=np.float32) / np.sqrt(256.0)
        s = model.score_embeddings(e1, e2)
        self.assertTrue(np.isfinite(s))

        # Test extract_embedding and verify optional full_sequence / apply_vad / vad_top_db overrides
        wav = (
            np.sin(np.linspace(0.0, 80.0 * np.pi, 16000 * 2)).astype(np.float32)
            * 0.3
        )
        emb_full = model.extract_embedding(wav, full_sequence=True)
        emb_slide = model.extract_embedding(wav, full_sequence=False)
        emb_novad = model.extract_embedding(
            wav, apply_vad=False, vad_top_db=20.0
        )
        self.assertEqual(emb_full.shape, (256,))
        self.assertEqual(emb_slide.shape, (256,))
        self.assertEqual(emb_novad.shape, (256,))
        v_score = model.verify(
            wav, wav, full_sequence=True, apply_vad=True, vad_top_db=26.0
        )
        self.assertTrue(np.isfinite(v_score))

        # Test FLAXSPEAKER_PRETRAINED_ROOT resolution in from_pretrained
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = os.path.join(tmpdir, "univ_conformer_tiny_ge2e")
            model.save_pretrained(custom_dir)
            old_env = os.environ.get("FLAXSPEAKER_PRETRAINED_ROOT")
            os.environ["FLAXSPEAKER_PRETRAINED_ROOT"] = tmpdir
            try:
                loaded = hf_compat.FlaxSpeakerModel.from_pretrained(
                    "univ_conformer_tiny_ge2e"
                )
                self.assertEqual(loaded.exp_config.model.embedding_dim, 256)
            finally:
                if old_env is None:
                    os.environ.pop("FLAXSPEAKER_PRETRAINED_ROOT", None)
                else:
                    os.environ["FLAXSPEAKER_PRETRAINED_ROOT"] = old_env

        from scripts import inference as inference_cli  # pytype: disable=import-error

        cli_parser = inference_cli.build_parser()
        parsed_1 = cli_parser.parse_args(
            [
                "--model_dir",
                "wq2012/univ_conformer_tiny_ge2e",
                "--apply_vad",
                "--vad_top_db",
                "22.5",
                "--full_sequence",
            ]
        )
        self.assertTrue(parsed_1.apply_vad)
        self.assertAlmostEqual(parsed_1.vad_top_db, 22.5)
        self.assertTrue(parsed_1.full_sequence)
        parsed_2 = cli_parser.parse_args(
            [
                "--model_dir",
                "wq2012/univ_conformer_tiny_ge2e",
                "--no_vad",
                "--sliding_window",
            ]
        )
        self.assertFalse(parsed_2.apply_vad)
        self.assertFalse(parsed_2.full_sequence)

    def test_seeded_triplet_determinism_and_degenerate_eval_labels(self):
        import random

        spk_map = {
            "spk1": ["u1_a", "u1_b", "u1_c"],
            "spk2": ["u2_a", "u2_b", "u2_c"],
            "spk3": ["u3_a", "u3_b", "u3_c"],
        }
        draws_1 = [
            dataset.get_triplet(spk_map, rng=random.Random(42 + i))
            for i in range(5)
        ]
        draws_2 = [
            dataset.get_triplet(spk_map, rng=random.Random(42 + i))
            for i in range(5)
        ]
        self.assertEqual(draws_1, draws_2)

        # Degenerate single-class or empty trial lists must raise ValueError
        for bad_labels, bad_scores in (
            ([], []),
            ([1, 1, 1, 1], [0.1, 0.2, 0.8, 0.9]),
            ([0, 0, 0, 0], [0.1, 0.2, 0.8, 0.9]),
        ):
            with self.assertRaises(ValueError):
                evaluation.compute_eer(bad_labels, bad_scores)
            with self.assertRaises(ValueError):
                evaluation.compute_eer_exact(bad_labels, bad_scores)
            with self.assertRaises(ValueError):
                evaluation.compute_min_dcf(bad_labels, bad_scores)
            with self.assertRaises(ValueError):
                evaluation.compute_roc_auc(bad_labels, bad_scores)
            with self.assertRaises(ValueError):
                evaluation.compute_bootstrap_eer_ci(bad_labels, bad_scores)


class TestRound3ReleaseHygiene(unittest.TestCase):
    """Round-3 checks: packaging, documentation references and training provenance."""

    REPO = os.path.dirname(os.path.abspath(__file__))

    def test_version_consistency(self):
        import re

        import flaxspeaker

        with open(os.path.join(self.REPO, "pyproject.toml"), "r", encoding="utf-8") as f:
            m_py = re.search(r'^version = "([^"]+)"', f.read(), re.M)
        with open(os.path.join(self.REPO, "setup.py"), "r", encoding="utf-8") as f:
            m_setup = re.search(r'^VERSION = "([^"]+)"', f.read(), re.M)
        self.assertIsNotNone(m_py)
        self.assertIsNotNone(m_setup)
        assert m_py is not None and m_setup is not None
        py_ver = m_py.group(1)
        setup_ver = m_setup.group(1)
        self.assertEqual(flaxspeaker.__version__, py_ver)
        self.assertEqual(flaxspeaker.__version__, setup_ver)
        # The published 0.2.0 wheel predates the released checkpoints; anything that
        # ships `load_pretrained` must carry a strictly newer version number.
        self.assertGreaterEqual(tuple(int(x) for x in py_ver.split(".")[:3]), (0, 3, 0))
        with open(os.path.join(self.REPO, "CHANGELOG.md"), "r", encoding="utf-8") as f:
            self.assertIn(f"## {py_ver}", f.read())

    def test_readme_script_references_exist(self):
        import re

        for doc in ("README.md", "RESEARCH_REPORT.md", "CHANGELOG.md"):
            with open(os.path.join(self.REPO, doc), "r", encoding="utf-8") as f:
                text = f.read()
            for rel in sorted(set(re.findall(r"scripts/[a-z_0-9]+\.py", text))):
                self.assertTrue(
                    os.path.exists(os.path.join(self.REPO, rel)),
                    f"{doc} references non-existent {rel}",
                )

    def test_training_provenance_recorded_for_all_released_checkpoints(self):
        import json

        pm = os.path.join(self.REPO, "pretrained_models")
        fp_path = os.path.join(pm, "step0_fingerprints.json")
        with open(fp_path, "r", encoding="utf-8") as f:
            fps = json.load(f)["checkpoints"]
        self.assertEqual(len(fps), 44)
        for mid in hf_compat.PRETRAINED_MODEL_IDS:
            with open(os.path.join(pm, mid, "config.json"), "r", encoding="utf-8") as f:
                cfg_json = json.load(f)
            prov = cfg_json.get("training_provenance")
            self.assertIsInstance(prov, dict, f"{mid}: no training_provenance block")
            self.assertIn(prov["variable_length_training_actual"], (True, False, "unknown"))
            self.assertEqual(prov["step0_fingerprint_match"], fps[mid]["match"])
            self.assertEqual(prov["released_initial_loss"], fps[mid]["released_initial_loss"])
            flag = cfg_json["experiment_dict"]["model"]["variable_length_training"]
            if prov["variable_length_training_actual"] != "unknown":
                self.assertEqual(flag, prov["variable_length_training_actual"], mid)
            ycfg = configs.ExperimentConfig.from_yaml(
                os.path.join(self.REPO, "configs", f"{mid}.yml")
            )
            self.assertEqual(bool(ycfg.model.variable_length_training), bool(flag), mid)
            # The block round-trips through FlaxSpeakerConfig.
            hf_cfg = hf_compat.FlaxSpeakerConfig.from_pretrained(os.path.join(pm, mid))
            self.assertEqual(hf_cfg.training_provenance, prov)
        # Every checkpoint reproduces its archived step-0 loss under exactly one crop regime,
        # except the two unseeded triplet runs.
        unknown = sorted(m for m, v in fps.items() if v["match"] == "neither")
        self.assertEqual(unknown, ["ls_conformer_small_triplet", "univ_conformer_small_triplet"])

    def test_model_card_provenance_paragraph(self):
        from flaxspeaker import model_card

        pm = os.path.join(self.REPO, "pretrained_models")
        for mid, needle in (
            ("univ_conformer_small_ge2e", "fixed 180-frame"),
            ("ls_conformer_small_ge2e_contrast", "variable-length crops"),
            ("univ_conformer_small_triplet", "cannot be established"),
        ):
            exp_cfg = hf_compat.FlaxSpeakerModel.load_exp_config(os.path.join(pm, mid))
            text = model_card._provenance_paragraph(os.path.join(pm, mid), exp_cfg)
            self.assertIn(needle, text, mid)
            self.assertIn("step0_fingerprints.json", text)
            self.assertNotIn("2026-10", text)  # no date-based reasoning

    def test_feature_cache_signature_and_legacy_fallback(self):
        cfg = configs.ExperimentConfig()
        fe = feature_extraction.AudioFrontend(cfg.frontend, cfg.vad)
        with tempfile.TemporaryDirectory() as tmp:
            c = feature_extraction.FeatureCache(fe, cache_dir=os.path.join(tmp, "librispeech"))
            self.assertTrue(c._cfg_signature.startswith("v3_"))
            self.assertIsNotNone(c._legacy_signature)
            self.assertTrue(c._legacy_signature.startswith("v2_"))
            # Changing a field the v2 signature omitted changes the v3 signature and
            # disables the legacy fallback (no stale .npy can match).
            cfg2 = configs.ExperimentConfig()
            cfg2.frontend.fmax = 7000.0
            c2 = feature_extraction.FeatureCache(
                feature_extraction.AudioFrontend(cfg2.frontend, cfg2.vad),
                cache_dir=os.path.join(tmp, "librispeech"),
            )
            self.assertNotEqual(c._cfg_signature, c2._cfg_signature)
            self.assertIsNone(c2._legacy_signature)
            cfg3 = configs.ExperimentConfig()
            cfg3.vad.snr_margin_db = 6.0
            c3 = feature_extraction.FeatureCache(
                feature_extraction.AudioFrontend(cfg3.frontend, cfg3.vad),
                cache_dir=os.path.join(tmp, "librispeech"),
            )
            self.assertNotEqual(c._cfg_signature, c3._cfg_signature)
            self.assertIsNone(c3._legacy_signature)
            # A legacy (v2-named) file is found by the v3 cache with default config.
            arr = np.zeros((7, 80), dtype=np.float32)
            legacy_name = c._digest_name(c._legacy_signature, "/x/a.flac")
            np.save(os.path.join(tmp, "librispeech", legacy_name), arr)
            self.assertTrue(np.allclose(c.get("/x/a.flac"), arr))
            v3_name = c._digest_name(c2._cfg_signature, "/x/a.flac")
            self.assertFalse(os.path.exists(os.path.join(tmp, "librispeech", v3_name)))
            # merged() unions in-memory entries and refuses mismatched signatures.
            m = feature_extraction.FeatureCache.merged(fe, [c], cache_dir=os.path.join(tmp, "universal"))
            self.assertEqual(len(m), 1)
            with self.assertRaises(ValueError):
                feature_extraction.FeatureCache.merged(fe, [c2], cache_dir=os.path.join(tmp, "universal"))
            # Bounded memory.
            small = feature_extraction.FeatureCache(fe, cache_dir=os.path.join(tmp, "librispeech"), max_memory_items=1)
            small._cache["k1"] = arr
            small._cache["k2"] = arr
            small._evict_if_needed()
            self.assertEqual(len(small), 1)

    def test_vad_processor_does_not_mutate_config(self):
        vc = configs.VadConfig(enabled=True, sample_rate=16000)
        proc = vad.VadProcessor(vc, sample_rate=8000)
        self.assertEqual(vc.sample_rate, 16000)
        self.assertEqual(proc.config.sample_rate, 8000)

    def test_eer_threshold_by_enrollment_count_and_vad_knobs(self):
        cfg = configs.ExperimentConfig()
        cfg.model.backbone = configs.BackboneType.CONFORMER
        cfg.model.apply_size_variant(configs.SizeVariant.TINY)
        hf_cfg = hf_compat.FlaxSpeakerConfig.from_experiment_config(cfg)
        model = hf_compat.FlaxSpeakerModel(config=hf_cfg)
        model.eval_results = {
            "multi_domain_eval": {
                "domains_1utt": {"universal_pooled": {"eer_threshold": 0.6}},
                "domains_3utt": {"universal_pooled": {"eer_threshold": 0.7}},
            }
        }
        self.assertAlmostEqual(model.eer_threshold(), 0.6)
        self.assertAlmostEqual(model.eer_threshold(num_enroll=1), 0.6)
        self.assertAlmostEqual(model.eer_threshold(num_enroll=2), 0.7)
        self.assertAlmostEqual(model.eer_threshold(num_enroll=3), 0.7)
        fe = model.resolve_frontend(vad_top_db=30.0, vad_snr_margin_db=6.0)
        self.assertAlmostEqual(fe.vad_processor.config.energy_db_threshold, 30.0)
        self.assertAlmostEqual(fe.vad_processor.config.snr_margin_db, 6.0)
        self.assertIs(model.resolve_frontend(), model.frontend)
        # The private alias used by older scripts still works.
        self.assertIs(model._resolve_frontend(), model.frontend)


if __name__ == "__main__":
    unittest.main()
