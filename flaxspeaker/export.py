"""Production model export to TFLite, TensorFlow SavedModel, and SafeTensors.

Provides:
- `export_to_tflite`: Converts a trained Flax speaker encoder to a standalone
  `.tflite` FlatBuffer (with optional 8-bit dynamic-range quantization via
  `tf.lite.Optimize.DEFAULT`).
- `export_to_saved_model`: Exports a TensorFlow `SavedModel` directory.
- `TFLiteSpeakerRunner`: Lightweight production inference wrapper using
  `ai_edge_litert` / `tf.lite.Interpreter` for embedded/on-device deployment.
"""

from __future__ import annotations

import os
from typing import Any

from flax.training import train_state
import jax
import numpy as np

from flaxspeaker import neural_net


def _make_tf_module(
    state: train_state.TrainState,
    seq_len: int,
    feat_dim: int,
) -> tuple[Any, Any]:
    """Wrap a Flax `TrainState` forward pass into a traceable `tf.Module`."""
    from jax.experimental import jax2tf
    import tensorflow as tf

    encoder_params = {
        k: v for k, v in state.params.items() if not k.startswith("_aux_")
    }

    def _forward_jax(x: jax.Array) -> jax.Array:
        return state.apply_fn({"params": encoder_params}, x)

    tf_Module = tf.Module()
    input_spec = tf.TensorSpec(shape=[1, seq_len, feat_dim], dtype=tf.float32, name="features")

    tf_fn_no_xla = jax2tf.convert(_forward_jax, with_gradient=False, enable_xla=False)
    tf_Module.f = tf.function(tf_fn_no_xla, autograph=False, input_signature=[input_spec])
    return tf_Module, input_spec


def export_to_tflite(
    state: train_state.TrainState,
    myconfig: Any,
    output_tflite_path: str,
    quantize_int8: bool = False,
) -> dict[str, Any]:
    """Export a trained Flax speaker encoder to TFLite (`.tflite`).

    Args:
        state: Trained Flax `TrainState`.
        myconfig: `ExperimentConfig` or legacy `Munch` config.
        output_tflite_path: Destination `.tflite` file path.
        quantize_int8: Whether to apply post-training dynamic range quantization.

    Returns:
        Dictionary with `tflite_path`, `size_bytes`, `size_kb`, and `quantize_int8`.
    """
    from jax.experimental import jax2tf
    import tensorflow as tf

    parent = os.path.dirname(os.path.abspath(output_tflite_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    seq_len = int(getattr(myconfig.model, "seq_len", 100))
    feat_dim = neural_net._get_feature_dim(myconfig)

    encoder_params = {
        k: v for k, v in state.params.items() if not k.startswith("_aux_")
    }

    def _forward_jax(x: jax.Array) -> jax.Array:
        return state.apply_fn({"params": encoder_params}, x)

    input_spec = tf.TensorSpec(
        shape=[1, seq_len, feat_dim], dtype=tf.float32, name="features"
    )

    try:
        tf_fn = tf.function(
            jax2tf.convert(_forward_jax, with_gradient=False, enable_xla=False),
            autograph=False,
            input_signature=[input_spec],
        )
        concrete_fn = tf_fn.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
        if quantize_int8:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()
    except Exception:
        # Fallback for models using XLA-specific primitives (e.g. associative_scan):
        # Enable XLA ops with SELECT_TF_OPS
        tf_fn = tf.function(
            jax2tf.convert(_forward_jax, with_gradient=False, enable_xla=True),
            autograph=False,
            input_signature=[input_spec],
        )
        concrete_fn = tf_fn.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
        if quantize_int8:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()

    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)

    size_bytes = os.path.getsize(output_tflite_path)
    return {
        "tflite_path": output_tflite_path,
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024.0, 2),
        "quantize_int8": quantize_int8,
    }


def export_to_saved_model(
    state: train_state.TrainState,
    myconfig: Any,
    saved_model_dir: str,
) -> str:
    """Export a trained Flax speaker encoder to a TensorFlow `SavedModel` directory."""
    import tensorflow as tf

    os.makedirs(saved_model_dir, exist_ok=True)
    seq_len = int(getattr(myconfig.model, "seq_len", 100))
    feat_dim = neural_net._get_feature_dim(myconfig)
    tf_mod, input_spec = _make_tf_module(state, seq_len, feat_dim)
    tf.saved_model.save(
        tf_mod,
        saved_model_dir,
        signatures={"serving_default": tf_mod.f.get_concrete_function(input_spec)},
    )
    return saved_model_dir


class TFLiteSpeakerRunner:
    """Production TFLite inference runner for exported FlaxSpeaker `.tflite` models."""

    def __init__(self, tflite_path: str):
        self.tflite_path = tflite_path
        try:
            from ai_edge_litert import interpreter as litert_interpreter

            self.interpreter = litert_interpreter.Interpreter(model_path=tflite_path)
        except ImportError:
            import tensorflow as tf

            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = tuple(self.input_details[0]["shape"])
        self.seq_len = int(self.input_shape[1])
        self.feat_dim = int(self.input_shape[2])

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Compute speaker embedding from feature array `(T, F)` or `(1, T, F)`."""
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] >= self.seq_len:
                offset = (arr.shape[0] - self.seq_len) // 2
                arr = arr[offset:offset + self.seq_len]
            else:
                reps = (self.seq_len // max(1, arr.shape[0])) + 1
                arr = np.tile(arr, (reps, 1))[:self.seq_len]
            arr = arr[None, :, :]
        self.interpreter.set_tensor(self.input_details[0]["index"], arr)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])
        return np.asarray(out[0], dtype=np.float32)

    def verify(self, enroll_features: np.ndarray, test_features: np.ndarray) -> float:
        """Compute cosine similarity between two feature sequences using TFLite."""
        e1 = self(enroll_features)
        e2 = self(test_features)
        return float(
            np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-6)
        )
