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
from typing import Any, Optional
import warnings

from flax.training import train_state
import jax
import numpy as np

from flaxspeaker import neural_net


def _resolve_export_seq_len(
    myconfig: Any, seq_len: Optional[int] = None
) -> int:
    if seq_len is not None and seq_len > 0:
        return int(seq_len)
    # Precedence: explicit argument > `export.sequence_length` (if > 0) > `model.seq_len`.
    # No magic sentinel values: any positive `export.sequence_length` is honoured.
    model_cfg = getattr(myconfig, "model", None)
    export_cfg = getattr(myconfig, "export", None)
    if export_cfg is not None and hasattr(export_cfg, "sequence_length"):
        exp_seq_len = int(getattr(export_cfg, "sequence_length"))
        if exp_seq_len > 0:
            return exp_seq_len
    if model_cfg is not None and hasattr(model_cfg, "seq_len"):
        return int(getattr(model_cfg, "seq_len"))
    return 180


def _make_tf_module(
    state: train_state.TrainState,
    seq_len: int,
    feat_dim: int,
    enable_xla: bool = False,
) -> tuple[Any, Any]:
    """Wrap a Flax `TrainState` forward pass into a traceable `tf.Module`.

    Shared by the SavedModel and TFLite exporters. `enable_xla=True` lets
    `jax2tf` emit XLA ops (needed e.g. for Mamba's `associative_scan`), at the
    price of requiring `SELECT_TF_OPS` in the TFLite converter.
    """
    from jax.experimental import jax2tf
    import tensorflow as tf

    encoder_params = {
        k: v for k, v in state.params.items() if not k.startswith("_aux_")
    }
    batch_stats = getattr(state, "batch_stats", None) or {}
    variables: dict[str, Any] = {"params": encoder_params}
    if batch_stats:
        variables["batch_stats"] = batch_stats

    def _forward_jax(x: jax.Array) -> jax.Array:
        return state.apply_fn(variables, x)

    tf_Module = tf.Module()
    input_spec = tf.TensorSpec(
        shape=[1, seq_len, feat_dim], dtype=tf.float32, name="features"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        tf_fn = jax2tf.convert(
            _forward_jax, with_gradient=False, enable_xla=enable_xla
        )
    tf_Module.f = tf.function(
        tf_fn, autograph=False, input_signature=[input_spec]
    )
    return tf_Module, input_spec


def _convert_tflite(tf_mod: Any, input_spec: Any, quantize_int8: bool, lower_tensor_list_ops: bool = True) -> bytes:
    import tensorflow as tf

    concrete_fn = tf_mod.f.get_concrete_function(input_spec)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    if quantize_int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    if not lower_tensor_list_ops:
        converter._experimental_lower_tensor_list_ops = False
    return converter.convert()


def export_to_tflite(
    state: train_state.TrainState,
    myconfig: Any,
    output_tflite_path: str,
    quantize_int8: bool = False,
    seq_len: Optional[int] = None,
) -> dict[str, Any]:
    """Export a trained Flax speaker encoder to TFLite (`.tflite`).

    Args:
        state: Trained Flax `TrainState`.
        myconfig: `ExperimentConfig` or legacy `Munch` config.
        output_tflite_path: Destination `.tflite` file path.
        quantize_int8: Whether to apply post-training dynamic range quantization.
        seq_len: Optional sequence length override for exported input signature.

    Returns:
        Dictionary with `tflite_path`, `size_bytes`, `size_kb`, and `quantize_int8`.
    """
    parent = os.path.dirname(os.path.abspath(output_tflite_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    resolved_seq_len = _resolve_export_seq_len(myconfig, seq_len=seq_len)
    feat_dim = neural_net._get_feature_dim(myconfig)

    try:
        tf_mod, input_spec = _make_tf_module(state, resolved_seq_len, feat_dim, enable_xla=False)
        tflite_model = _convert_tflite(tf_mod, input_spec, quantize_int8)
    except Exception as first_exc:  # converter errors are not a single exception type
        # Fallback for models using XLA-specific primitives (e.g. Mamba's
        # associative_scan): emit XLA ops and rely on SELECT_TF_OPS. The first
        # error is surfaced instead of being swallowed so that a genuine export
        # bug is not hidden behind the fallback.
        warnings.warn(
            "jax2tf conversion without XLA ops failed "
            f"({type(first_exc).__name__}: {first_exc}); retrying with enable_xla=True.",
            RuntimeWarning,
            stacklevel=2,
        )
        tf_mod, input_spec = _make_tf_module(state, resolved_seq_len, feat_dim, enable_xla=True)
        tflite_model = _convert_tflite(tf_mod, input_spec, quantize_int8, lower_tensor_list_ops=False)

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
    seq_len: Optional[int] = None,
) -> str:
    """Export a trained Flax speaker encoder to a TensorFlow `SavedModel` directory."""
    import tensorflow as tf

    os.makedirs(saved_model_dir, exist_ok=True)
    resolved_seq_len = _resolve_export_seq_len(myconfig, seq_len=seq_len)
    feat_dim = neural_net._get_feature_dim(myconfig)
    tf_mod, input_spec = _make_tf_module(state, resolved_seq_len, feat_dim)
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
        """Compute speaker embedding from feature array `(T, F)` or `(1, T, F)`.

        2-D inputs are centre-cropped (if `T >= seq_len`) or tiled (if shorter) to
        the exported window, exactly as the benchmark's fixed-window path. 3-D inputs
        must already have the exported shape `(1, seq_len, F)`; a mismatch raises
        instead of letting the interpreter fail with an opaque shape error.
        """
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] == 0:
                raise ValueError("TFLiteSpeakerRunner: empty feature matrix (0 frames).")
            if arr.shape[1] != self.feat_dim:
                raise ValueError(
                    f"TFLiteSpeakerRunner: expected {self.feat_dim} feature dims, got {arr.shape[1]}."
                )
            if arr.shape[0] >= self.seq_len:
                offset = (arr.shape[0] - self.seq_len) // 2
                arr = arr[offset:offset + self.seq_len]
            else:
                reps = (self.seq_len // max(1, arr.shape[0])) + 1
                arr = np.tile(arr, (reps, 1))[:self.seq_len]
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            if tuple(arr.shape) != tuple(self.input_shape):
                raise ValueError(
                    f"TFLiteSpeakerRunner: 3-D input must have shape {tuple(self.input_shape)}, "
                    f"got {tuple(arr.shape)}; pass a 2-D (T, F) matrix to crop/tile automatically."
                )
        else:
            raise ValueError(f"TFLiteSpeakerRunner: expected a 2-D or 3-D array, got ndim={arr.ndim}.")
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
