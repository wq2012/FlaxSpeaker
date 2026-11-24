"""Streaming `np.float16` memory-mapped feature shard writer and reader.

Designed for million-utterance speaker recognition training on shared CPU/GPU hosts
with strict memory ceilings (`< 1 GiB` Python RSS overhead regardless of total corpus
size on disk):
- Each shard (`shard_XXXX.f16`) stores contiguous `(total_shard_frames, feat_dim)`
  `np.float16` acoustic features (~48 KB per 3-second 80-dim utterance, with `< 1.5e-3`
  quantization error on CMVN-normalized log-Mel filterbanks — `8x` more accurate than
  `bfloat16` at identical 2 bytes/element).
- `index.npz` + `metadata.json` store compact CSR arrays (`shard_ids`, `frame_offsets`,
  `frame_lengths`, `speaker_ids`, `domain_ids`, `aug_flags`) and speaker/domain maps.
- `ShardedFeatureStore` opens shards lazily with `np.memmap(..., mode="r")`, letting
  the OS page cache serve random segment crops at NVMe speed without heap allocation.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from flaxspeaker import dataset
from flaxspeaker import specaug


class ShardedFeatureWriter:
    """Incrementally writes variable-length `(T, D)` features into `float16` `.f16` shards."""

    def __init__(
        self,
        shard_dir: str,
        feat_dim: int = 80,
        max_shard_bytes: int = 2 * 1024 * 1024 * 1024,
    ):
        self.shard_dir = os.path.abspath(os.path.expanduser(shard_dir))
        os.makedirs(self.shard_dir, exist_ok=True)
        self.feat_dim = int(feat_dim)
        self.bytes_per_frame = self.feat_dim * 2  # float16
        self.max_frames_per_shard = max(
            1024, int(max_shard_bytes // max(1, self.bytes_per_frame))
        )

        self._cur_shard_id = 0
        self._cur_frame_offset = 0
        self._cur_file = open(self._shard_path(0), "wb")
        self._shard_frame_counts: list[int] = []

        self.speaker_to_id: dict[str, int] = {}
        self.domain_to_id: dict[str, int] = {}

        self._shard_ids: list[int] = []
        self._frame_offsets: list[int] = []
        self._frame_lengths: list[int] = []
        self._speaker_ids: list[int] = []
        self._domain_ids: list[int] = []
        self._aug_flags: list[int] = []
        self._closed = False

    def _shard_path(self, shard_id: int) -> str:
        return os.path.join(self.shard_dir, f"shard_{shard_id:04d}.f16")

    def _rotate_shard(self) -> None:
        self._cur_file.flush()
        self._cur_file.close()
        self._shard_frame_counts.append(self._cur_frame_offset)
        self._cur_shard_id += 1
        self._cur_frame_offset = 0
        self._cur_file = open(self._shard_path(self._cur_shard_id), "wb")

    def add(
        self,
        features: np.ndarray,
        speaker: str,
        domain: Optional[str] = None,
        is_augmented: bool = False,
    ) -> int:
        """Appends one `(T, feat_dim)` utterance feature matrix to the current shard."""
        if self._closed:
            raise RuntimeError("ShardedFeatureWriter is already closed.")
        arr = np.asarray(features, dtype=np.float16)
        if arr.ndim != 2 or arr.shape[1] != self.feat_dim:
            raise ValueError(
                f"Expected shape (T, {self.feat_dim}), got {arr.shape}."
            )
        num_frames = int(arr.shape[0])
        if num_frames <= 0:
            raise ValueError("Cannot write empty utterance (0 frames) to shard.")

        if (
            self._cur_frame_offset > 0
            and self._cur_frame_offset + num_frames > self.max_frames_per_shard
        ):
            self._rotate_shard()

        dom = domain if domain is not None else (
            speaker.split(":", 1)[0] if ":" in speaker else "default"
        )
        spk_id = self.speaker_to_id.setdefault(speaker, len(self.speaker_to_id))
        dom_id = self.domain_to_id.setdefault(dom, len(self.domain_to_id))

        arr_c = np.ascontiguousarray(arr, dtype=np.float16)
        self._cur_file.write(arr_c.tobytes())

        item_idx = len(self._frame_lengths)
        self._shard_ids.append(self._cur_shard_id)
        self._frame_offsets.append(self._cur_frame_offset)
        self._frame_lengths.append(num_frames)
        self._speaker_ids.append(spk_id)
        self._domain_ids.append(dom_id)
        self._aug_flags.append(1 if is_augmented else 0)
        self._cur_frame_offset += num_frames
        return item_idx

    def close(self, extra_metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Flushes the final shard and writes `index.npz` + `metadata.json`."""
        if self._closed:
            raise RuntimeError("ShardedFeatureWriter is already closed.")
        self._cur_file.flush()
        self._cur_file.close()
        if self._cur_frame_offset > 0:
            self._shard_frame_counts.append(self._cur_frame_offset)
        elif os.path.exists(self._shard_path(self._cur_shard_id)):
            os.remove(self._shard_path(self._cur_shard_id))
        self._closed = True

        np.savez(
            os.path.join(self.shard_dir, "index.npz"),
            shard_ids=np.asarray(self._shard_ids, dtype=np.int32),
            frame_offsets=np.asarray(self._frame_offsets, dtype=np.int64),
            frame_lengths=np.asarray(self._frame_lengths, dtype=np.int32),
            speaker_ids=np.asarray(self._speaker_ids, dtype=np.int32),
            domain_ids=np.asarray(self._domain_ids, dtype=np.int16),
            aug_flags=np.asarray(self._aug_flags, dtype=np.uint8),
        )
        with open(
            os.path.join(self.shard_dir, "speakers.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "speaker_to_id": self.speaker_to_id,
                    "domain_to_id": self.domain_to_id,
                },
                f,
                indent=2,
            )

        meta: dict[str, Any] = {
            "format_version": "flaxspeaker_f16_v1",
            "dtype": "float16",
            "feat_dim": self.feat_dim,
            "num_shards": len(self._shard_frame_counts),
            "shard_frame_counts": self._shard_frame_counts,
            "num_items": len(self._frame_lengths),
            "total_frames": int(sum(self._shard_frame_counts)),
            "num_speakers": len(self.speaker_to_id),
            "num_domains": len(self.domain_to_id),
        }
        if extra_metadata:
            meta.update(extra_metadata)
        with open(
            os.path.join(self.shard_dir, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2)
        return meta

    def __enter__(self) -> "ShardedFeatureWriter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._closed:
            self.close()


class ShardedFeatureStore:
    """Zero-copy `np.memmap` reader and fast batch sampler over `float16` feature shards."""

    def __init__(self, shard_dir: str, base_speakers_only: bool = False):
        resolved_dir = dataset.resolve_data_path(shard_dir)
        self.shard_dir = os.path.abspath(os.path.expanduser(resolved_dir))
        meta_path = os.path.join(self.shard_dir, "metadata.json")
        idx_path = os.path.join(self.shard_dir, "index.npz")
        spk_path = os.path.join(self.shard_dir, "speakers.json")
        if not (os.path.isfile(meta_path) and os.path.isfile(idx_path)):
            raise FileNotFoundError(
                f"Missing metadata.json or index.npz in shard_dir={self.shard_dir}"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)
        with open(spk_path, "r", encoding="utf-8") as f:
            spk_data = json.load(f)

        self.feat_dim = int(self.metadata["feat_dim"])
        self.shard_frame_counts: list[int] = [
            int(c) for c in self.metadata["shard_frame_counts"]
        ]
        self.raw_speaker_to_id: dict[str, int] = {
            str(k): int(v) for k, v in spk_data["speaker_to_id"].items()
        }
        self.domain_to_id: dict[str, int] = {
            str(k): int(v) for k, v in spk_data["domain_to_id"].items()
        }

        idx = np.load(idx_path)
        self.shard_ids: np.ndarray = idx["shard_ids"]
        self.frame_offsets: np.ndarray = idx["frame_offsets"]
        self.frame_lengths: np.ndarray = idx["frame_lengths"]
        raw_spk_ids: np.ndarray = idx["speaker_ids"]
        self.domain_ids: np.ndarray = idx["domain_ids"]
        self.aug_flags: np.ndarray = idx["aug_flags"]

        # Speed perturbation appends `@sp0.9` / `@sp1.1` to speaker names so
        # classification losses treat them as 3x distinct classes; tuple/metric
        # losses (`GE2E`, `PFAS`, `Dr-Vectors`) optionally collapse `@sp*` back
        # to the base speaker identity via `base_speakers_only=True`.
        if base_speakers_only:
            id_to_raw = {v: k for k, v in self.raw_speaker_to_id.items()}
            base_map: dict[str, int] = {}
            remap = np.zeros((len(self.raw_speaker_to_id),), dtype=np.int32)
            for rid in range(len(self.raw_speaker_to_id)):
                base_name = id_to_raw[rid].split("@sp", 1)[0]
                bid = base_map.setdefault(base_name, len(base_map))
                remap[rid] = bid
            self.speaker_to_id = base_map
            self.speaker_ids = remap[raw_spk_ids]
        else:
            self.speaker_to_id = self.raw_speaker_to_id
            self.speaker_ids = raw_spk_ids

        self.num_items = int(self.frame_lengths.shape[0])
        self.num_speakers = len(self.speaker_to_id)
        self.num_domains = max(1, len(self.domain_to_id))

        # Open read-only memory maps
        self._memmaps: list[np.memmap] = []
        for s_id, n_frames in enumerate(self.shard_frame_counts):
            s_path = os.path.join(self.shard_dir, f"shard_{s_id:04d}.f16")
            mm = np.memmap(
                s_path, dtype=np.float16, mode="r", shape=(n_frames, self.feat_dim)
            )
            self._memmaps.append(mm)

        # Build CSR index: speaker_id -> slice of sorted item indices
        order = np.argsort(self.speaker_ids, kind="mergesort")
        self._spk_sorted_items = order.astype(np.int32)
        counts = np.bincount(self.speaker_ids, minlength=self.num_speakers)
        self._spk_indptr = np.zeros((self.num_speakers + 1,), dtype=np.int64)
        np.cumsum(counts, out=self._spk_indptr[1:])
        self.spk_utt_counts = counts

        # Map each speaker to its primary domain_id
        self.spk_domain_id = np.zeros((self.num_speakers,), dtype=np.int16)
        for spk_id in range(self.num_speakers):
            start, end = int(self._spk_indptr[spk_id]), int(self._spk_indptr[spk_id + 1])
            if end > start:
                first_item = int(self._spk_sorted_items[start])
                self.spk_domain_id[spk_id] = self.domain_ids[first_item]

        self.domain_speakers: dict[int, np.ndarray] = {}
        self.domain_eligible_nm: dict[int, np.ndarray] = {}
        for d_id in range(self.num_domains):
            spks_d = np.where(self.spk_domain_id == d_id)[0].astype(np.int32)
            if spks_d.size > 0:
                self.domain_speakers[d_id] = spks_d
                elig = spks_d[self.spk_utt_counts[spks_d] >= 2]
                if elig.size > 0:
                    self.domain_eligible_nm[d_id] = elig
        self.all_eligible_nm = np.where(self.spk_utt_counts >= 2)[0].astype(np.int32)

    def get_item(self, item_idx: int) -> np.ndarray:
        """Reads full `(T, feat_dim)` utterance features as `float32`."""
        s_id = int(self.shard_ids[item_idx])
        offset = int(self.frame_offsets[item_idx])
        length = int(self.frame_lengths[item_idx])
        return np.asarray(
            self._memmaps[s_id][offset:offset + length, :], dtype=np.float32
        )

    def get_speaker_items(self, spk_id: int) -> np.ndarray:
        start = int(self._spk_indptr[spk_id])
        end = int(self._spk_indptr[spk_id + 1])
        return self._spk_sorted_items[start:end]

    def _read_cropped_padded(
        self,
        item_idx: int,
        target_len: int,
        max_seq_len: int,
        min_len: int,
        specaug_config: Optional[Any],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slices ONLY the needed segment directly from the `float16` memmap."""
        s_id = int(self.shard_ids[item_idx])
        offset = int(self.frame_offsets[item_idx])
        full_len = int(self.frame_lengths[item_idx])
        mm = self._memmaps[s_id]

        if full_len < min_len:
            raw = np.asarray(mm[offset:offset + full_len, :], dtype=np.float32)
            reps = (min_len // max(1, full_len)) + 1
            raw = np.tile(raw, (reps, 1))
            full_len = raw.shape[0]
            valid_len = min(full_len, target_len)
            start = (
                int(rng.integers(0, full_len - valid_len + 1))
                if full_len > valid_len
                else 0
            )
            seg = raw[start:start + valid_len, :].copy()
        else:
            valid_len = min(full_len, target_len)
            start = (
                int(rng.integers(0, full_len - valid_len + 1))
                if full_len > valid_len
                else 0
            )
            seg = np.asarray(
                mm[offset + start:offset + start + valid_len, :], dtype=np.float32
            ).copy()

        # Corpus MTR was already baked into augmented shard entries; apply SpecAugment cleanly
        if (
            specaug_config is not None
            and getattr(specaug_config, "use_specaug", True)
            and float(rng.random()) < float(getattr(specaug_config, "apply_prob", 0.5))
        ):
            seg = specaug.apply_specaug(seg, specaug_config, rng=rng)

        padded = np.zeros((max_seq_len, self.feat_dim), dtype=np.float32)
        padded[:valid_len, :] = seg
        mask = np.zeros((max_seq_len,), dtype=np.float32)
        mask[:valid_len] = 1.0
        return padded, mask

    def sample_classification_batch(
        self,
        batch_size: int,
        max_seq_len: int,
        min_seq_len: Optional[int] = None,
        variable_length: bool = True,
        specaug_config: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        domain_temp_alpha: float = 0.35,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Samples a domain-balanced classification batch `(B, T, D)` from `memmap` shards."""
        gen = rng if rng is not None else np.random.default_rng()
        min_len = (
            max(16, min(int(min_seq_len), max_seq_len))
            if (variable_length and min_seq_len is not None)
            else max_seq_len
        )
        dom_keys = sorted(self.domain_speakers.keys())
        raw_w = np.array(
            [len(self.domain_speakers[d]) ** domain_temp_alpha for d in dom_keys],
            dtype=np.float64,
        )
        dom_probs = raw_w / np.sum(raw_w)

        feats = np.empty((batch_size, max_seq_len, self.feat_dim), dtype=np.float32)
        masks = np.empty((batch_size, max_seq_len), dtype=np.float32)
        labels = np.empty((batch_size,), dtype=np.int32)

        for b in range(batch_size):
            if len(dom_keys) > 1:
                d_choice = dom_keys[int(gen.choice(len(dom_keys), p=dom_probs))]
                spks_d = self.domain_speakers[d_choice]
                spk_id = int(spks_d[int(gen.integers(0, spks_d.size))])
            else:
                spk_id = int(gen.integers(0, self.num_speakers))
            items = self.get_speaker_items(spk_id)
            item_idx = int(items[int(gen.integers(0, items.size))])
            t_len = (
                int(gen.integers(min_len, max_seq_len + 1))
                if (variable_length and min_len < max_seq_len)
                else max_seq_len
            )
            padded, mask = self._read_cropped_padded(
                item_idx, t_len, max_seq_len, min_len, specaug_config, gen
            )
            feats[b] = padded
            masks[b] = mask
            labels[b] = spk_id

        return (
            jnp.asarray(feats, dtype=jnp.float32),
            jnp.asarray(labels, dtype=jnp.int32),
            jnp.asarray(masks, dtype=jnp.float32),
        )

    def sample_nm_batch(
        self,
        num_speakers: int,
        num_utts_per_speaker: int,
        max_seq_len: int,
        min_seq_len: Optional[int] = None,
        variable_length: bool = True,
        specaug_config: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        within_domain_prob: float = 0.75,
        domain_temp_alpha: float = 0.35,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Samples a structured `(N * M, T, D)` speaker-utterance batch from `memmap` shards."""
        gen = rng if rng is not None else np.random.default_rng()
        min_len = (
            max(16, min(int(min_seq_len), max_seq_len))
            if (variable_length and min_seq_len is not None)
            else max_seq_len
        )
        dom_keys = sorted(self.domain_eligible_nm.keys())
        valid_within = [
            d for d in dom_keys if self.domain_eligible_nm[d].size >= num_speakers
        ]

        if len(dom_keys) > 1 and valid_within and float(gen.random()) < within_domain_prob:
            w_arr = np.array(
                [
                    self.domain_eligible_nm[d].size ** domain_temp_alpha
                    for d in valid_within
                ],
                dtype=np.float64,
            )
            w_arr /= np.sum(w_arr)
            d_choice = valid_within[int(gen.choice(len(valid_within), p=w_arr))]
            pool = self.domain_eligible_nm[d_choice]
            chosen_spks = gen.choice(pool, size=num_speakers, replace=False)
        elif self.all_eligible_nm.size >= num_speakers:
            if len(dom_keys) > 1:
                dom_w = {
                    d: (self.domain_eligible_nm[d].size ** domain_temp_alpha)
                    / self.domain_eligible_nm[d].size
                    for d in dom_keys
                }
                spk_w = np.array(
                    [
                        dom_w[int(self.spk_domain_id[int(s)])]
                        for s in self.all_eligible_nm
                    ],
                    dtype=np.float64,
                )
                spk_w /= np.sum(spk_w)
                chosen_spks = gen.choice(
                    self.all_eligible_nm, size=num_speakers, replace=False, p=spk_w
                )
            else:
                chosen_spks = gen.choice(
                    self.all_eligible_nm, size=num_speakers, replace=False
                )
        else:
            chosen_spks = gen.choice(
                self.all_eligible_nm, size=num_speakers, replace=True
            )

        total = num_speakers * num_utts_per_speaker
        feats = np.empty((total, max_seq_len, self.feat_dim), dtype=np.float32)
        masks = np.empty((total, max_seq_len), dtype=np.float32)
        labels = np.empty((total,), dtype=np.int32)

        row = 0
        for local_idx, spk_id in enumerate(chosen_spks):
            items = self.get_speaker_items(int(spk_id))
            replace = items.size < num_utts_per_speaker
            chosen_items = gen.choice(items, size=num_utts_per_speaker, replace=replace)
            for item_idx in chosen_items:
                t_len = (
                    int(gen.integers(min_len, max_seq_len + 1))
                    if (variable_length and min_len < max_seq_len)
                    else max_seq_len
                )
                padded, mask = self._read_cropped_padded(
                    int(item_idx), t_len, max_seq_len, min_len, specaug_config, gen
                )
                feats[row] = padded
                masks[row] = mask
                labels[row] = local_idx
                row += 1

        return (
            jnp.asarray(feats, dtype=jnp.float32),
            jnp.asarray(labels, dtype=jnp.int32),
            jnp.asarray(masks, dtype=jnp.float32),
        )

    def sample_triplet_batch(
        self,
        batch_size: int,
        max_seq_len: int,
        min_seq_len: Optional[int] = None,
        variable_length: bool = True,
        specaug_config: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[jax.Array, None, jax.Array]:
        """Samples `batch_size` `(anchor, positive, negative)` triplets `(3 * B, T, D)`."""
        gen = rng if rng is not None else np.random.default_rng()
        min_len = (
            max(16, min(int(min_seq_len), max_seq_len))
            if (variable_length and min_seq_len is not None)
            else max_seq_len
        )
        feats = np.empty((3 * batch_size, max_seq_len, self.feat_dim), dtype=np.float32)
        masks = np.empty((3 * batch_size, max_seq_len), dtype=np.float32)
        row = 0
        for _ in range(batch_size):
            s_pos, s_neg = gen.choice(self.all_eligible_nm, size=2, replace=False)
            pos_items = self.get_speaker_items(int(s_pos))
            neg_items = self.get_speaker_items(int(s_neg))
            a_idx, p_idx = gen.choice(pos_items, size=2, replace=False)
            n_idx = neg_items[int(gen.integers(0, neg_items.size))]
            for item_idx in (int(a_idx), int(p_idx), int(n_idx)):
                t_len = (
                    int(gen.integers(min_len, max_seq_len + 1))
                    if (variable_length and min_len < max_seq_len)
                    else max_seq_len
                )
                padded, mask = self._read_cropped_padded(
                    item_idx, t_len, max_seq_len, min_len, specaug_config, gen
                )
                feats[row] = padded
                masks[row] = mask
                row += 1
        return (
            jnp.asarray(feats, dtype=jnp.float32),
            None,
            jnp.asarray(masks, dtype=jnp.float32),
        )


class WindowedMultiDomainSampler:
    """Stratified rolling-window multi-domain `.f16` shard sampler for multi-GPU / TPU training.

    Maintains a stratified subset of `shard_budget` shards (`0` = all shards) across
    all active domains (`(0, 1, 2)` or `(0, 1, 2, 3, 4)` when conversational/meeting
    diarization shards are appended via `diar_shard_dirs`) over `ShardedFeatureStore`
    `memmap` arrays so multi-device `jax.pmap` runs keep bounded RSS on any host while
    visiting all shards via `rotate_window()`.
    """

    def __init__(
        self,
        store: ShardedFeatureStore,
        shard_budget: int = 18,
        seed: int = 42,
        shuffled_30_57_dir: Optional[str] = None,
        extra_shard_dirs: Optional[list[str]] = None,
        diar_shard_dirs: Optional[list[str]] = None,
        max_total_shards: int = 0,
    ):
        self.store = store
        self.base_n_shards = len(store.shard_frame_counts)
        self.feat_dim = store.feat_dim
        merged_spk_to_id: dict[str, int] = {
            k: int(v) for k, v in store.raw_speaker_to_id.items()
        }

        base_idx = np.load(os.path.join(store.shard_dir, "index.npz"))
        self.orig_speaker_ids = base_idx["speaker_ids"].astype(np.int32)
        self.orig_shard_ids = store.shard_ids.astype(np.int32).copy()
        self.orig_frame_offsets = store.frame_offsets.astype(np.int64).copy()
        self.orig_frame_lengths = store.frame_lengths.astype(np.int32).copy()
        self.orig_domain_ids = store.domain_ids.astype(np.int16).copy()

        self.shard_paths: list[str] = [
            os.path.join(store.shard_dir, f"shard_{s_i:04d}.f16")
            for s_i in range(self.base_n_shards)
        ]
        self.using_shuffled_30_57 = False
        if shuffled_30_57_dir:
            shuf_dir = os.path.abspath(
                os.path.expanduser(dataset.resolve_data_path(shuffled_30_57_dir))
            )
            shuf_idx_path = os.path.join(shuf_dir, "index_30_57.npz")
            if os.path.isfile(shuf_idx_path):
                sz = np.load(shuf_idx_path)
                s_iids = sz["item_indices"].astype(np.int32)
                self.orig_shard_ids[s_iids] = sz["new_shard_ids"].astype(np.int32)
                self.orig_frame_offsets[s_iids] = sz["new_frame_offsets"].astype(
                    np.int64
                )
                for s_i in range(30, min(58, self.base_n_shards)):
                    self.shard_paths[s_i] = os.path.join(
                        shuf_dir, f"shard_{s_i:04d}.f16"
                    )
                self.using_shuffled_30_57 = True

        for ext_raw in extra_shard_dirs or []:
            ext_dir = os.path.abspath(
                os.path.expanduser(dataset.resolve_data_path(ext_raw))
            )
            emeta_path = os.path.join(ext_dir, "metadata.json")
            eidx_path = os.path.join(ext_dir, "index.npz")
            if not (os.path.isfile(emeta_path) and os.path.isfile(eidx_path)):
                continue
            with open(emeta_path, "r", encoding="utf-8") as ef:
                emeta = json.load(ef)
            ez = np.load(eidx_path)
            sid_offset = len(self.shard_paths)
            ez_sids = ez["shard_ids"].astype(np.int32)
            ext_n_sh = max(
                int(emeta["num_shards"]),
                int(np.max(ez_sids)) + 1 if ez_sids.size > 0 else 0,
            )
            if max_total_shards > 0:
                ext_n_sh = max(0, min(ext_n_sh, max_total_shards - sid_offset))
                keep = ez_sids < ext_n_sh
            else:
                keep = np.ones((ez_sids.shape[0],), dtype=np.bool_)
            for s_i in range(ext_n_sh):
                self.shard_paths.append(
                    os.path.join(ext_dir, f"shard_{s_i:04d}.f16")
                )
            self.orig_shard_ids = np.concatenate(
                [self.orig_shard_ids, ez_sids[keep] + sid_offset]
            )
            self.orig_frame_offsets = np.concatenate(
                [self.orig_frame_offsets, ez["frame_offsets"][keep].astype(np.int64)]
            )
            self.orig_frame_lengths = np.concatenate(
                [self.orig_frame_lengths, ez["frame_lengths"][keep].astype(np.int32)]
            )
            self.orig_speaker_ids = np.concatenate(
                [self.orig_speaker_ids, ez["speaker_ids"][keep].astype(np.int32)]
            )
            self.orig_domain_ids = np.concatenate(
                [self.orig_domain_ids, ez["domain_ids"][keep].astype(np.int16)]
            )

        self.diar_start_sid = len(self.shard_paths)
        for diar_raw in diar_shard_dirs or []:
            diar_dir = os.path.abspath(
                os.path.expanduser(dataset.resolve_data_path(diar_raw))
            )
            dmeta_path = os.path.join(diar_dir, "metadata.json")
            dspk_path = os.path.join(diar_dir, "speakers.json")
            didx_path = os.path.join(diar_dir, "index.npz")
            if not (
                os.path.isfile(dmeta_path)
                and os.path.isfile(dspk_path)
                and os.path.isfile(didx_path)
            ):
                continue
            with open(dmeta_path, "r", encoding="utf-8") as df:
                dmeta = json.load(df)
            with open(dspk_path, "r", encoding="utf-8") as df:
                dspks = json.load(df)
            dz = np.load(didx_path)
            sid_offset = len(self.shard_paths)
            spk_offset = len(merged_spk_to_id)
            for s_name, local_sid in dspks["speaker_to_id"].items():
                if s_name in merged_spk_to_id:
                    raise ValueError(f"Duplicate speaker {s_name} in {diar_dir}")
                merged_spk_to_id[s_name] = int(local_sid) + spk_offset
            dz_sids = dz["shard_ids"].astype(np.int32)
            diar_n_sh = max(
                int(dmeta["num_shards"]),
                int(np.max(dz_sids)) + 1 if dz_sids.size > 0 else 0,
            )
            if max_total_shards > 0:
                diar_n_sh = max(0, min(diar_n_sh, max_total_shards - sid_offset))
                keep = dz_sids < diar_n_sh
            else:
                keep = np.ones((dz_sids.shape[0],), dtype=np.bool_)
            for s_i in range(diar_n_sh):
                self.shard_paths.append(
                    os.path.join(diar_dir, f"shard_{s_i:04d}.f16")
                )
            self.orig_shard_ids = np.concatenate(
                [self.orig_shard_ids, dz_sids[keep] + sid_offset]
            )
            self.orig_frame_offsets = np.concatenate(
                [self.orig_frame_offsets, dz["frame_offsets"][keep].astype(np.int64)]
            )
            self.orig_frame_lengths = np.concatenate(
                [self.orig_frame_lengths, dz["frame_lengths"][keep].astype(np.int32)]
            )
            self.orig_speaker_ids = np.concatenate(
                [
                    self.orig_speaker_ids,
                    dz["speaker_ids"][keep].astype(np.int32) + spk_offset,
                ]
            )
            self.orig_domain_ids = np.concatenate(
                [self.orig_domain_ids, dz["domain_ids"][keep].astype(np.int16)]
            )

        self.num_classes = len(merged_spk_to_id)
        id_to_spk = {int(v): k for k, v in merged_spk_to_id.items()}
        is_sp = np.array(
            ["@sp" in id_to_spk[i] for i in range(self.num_classes)], dtype=np.bool_
        )
        self.orig_item_is_sp = is_sp[self.orig_speaker_ids]
        self.clean_class_ids = np.nonzero(~is_sp)[0].astype(np.int32)
        self.orig_to_clean_label = np.zeros((self.num_classes,), dtype=np.int32)
        for clean_j, orig_cid in enumerate(self.clean_class_ids):
            self.orig_to_clean_label[int(orig_cid)] = clean_j

        self.total_shards = (
            min(len(self.shard_paths), max_total_shards)
            if max_total_shards > 0
            else len(self.shard_paths)
        )
        self._ext_memmaps: dict[int, np.memmap] = {}

        uniq_doms = sorted(int(d) for d in np.unique(self.orig_domain_ids))
        self.all_domain_ids: tuple[int, ...] = (
            tuple(uniq_doms) if len(uniq_doms) > 3 else (0, 1, 2)
        )
        self.dom_shards: dict[int, list[int]] = {
            d: [] for d in self.all_domain_ids
        }
        for sid in range(self.total_shards):
            m = self.orig_shard_ids == sid
            if np.any(m):
                counts = [
                    int(np.sum(self.orig_domain_ids[m] == d))
                    for d in self.all_domain_ids
                ]
                dom_major = self.all_domain_ids[int(np.argmax(counts))]
                self.dom_shards[dom_major].append(sid)

        self.rolling_enabled = 0 < shard_budget < self.total_shards
        self._rot_rng = np.random.default_rng(int(seed) + 1337)
        has_diar_shards = self.diar_start_sid < self.total_shards
        if self.rolling_enabled:
            if has_diar_shards and self.using_shuffled_30_57:
                pool_d = list(range(self.diar_start_sid, self.total_shards))
                pool_a = [
                    s
                    for s in range(self.diar_start_sid)
                    if s < 30 or s >= self.base_n_shards
                ]
                pool_b = list(
                    range(30, min(self.base_n_shards, self.diar_start_sid))
                )
                k_d = min(len(pool_d), max(4, round(shard_budget * 0.30)))
                rem_u = max(2, shard_budget - k_d)
                k_b = min(len(pool_b), max(1, round(rem_u * 0.45)))
                k_a = min(len(pool_a), max(1, rem_u - k_b))
                init_parts: list[int] = []
                for p_grp, k_pick in ((pool_a, k_a), (pool_b, k_b), (pool_d, k_d)):
                    if p_grp and k_pick > 0:
                        chosen_p = self._rot_rng.choice(
                            p_grp, size=min(len(p_grp), k_pick), replace=False
                        )
                        init_parts.extend(int(x) for x in chosen_p)
                self.active_sids = sorted(set(init_parts))
            elif self.using_shuffled_30_57:
                pool_a = [
                    s
                    for s in range(self.total_shards)
                    if s < 30 or s >= self.base_n_shards
                ]
                pool_b = [
                    s
                    for s in range(30, min(self.base_n_shards, self.total_shards))
                ]
                k_b = min(len(pool_b), max(1, round(shard_budget * 0.45)))
                k_a = min(len(pool_a), max(1, shard_budget - k_b))
                c_a = (
                    [
                        int(x)
                        for x in self._rot_rng.choice(
                            pool_a, size=k_a, replace=False
                        )
                    ]
                    if pool_a
                    else []
                )
                c_b = (
                    [
                        int(x)
                        for x in self._rot_rng.choice(
                            pool_b, size=k_b, replace=False
                        )
                    ]
                    if pool_b
                    else []
                )
                self.active_sids = sorted(set(c_a + c_b))
            else:
                n_d0 = min(
                    len(self.dom_shards.get(0, [])), max(1, round(shard_budget * 0.50))
                )
                n_d1 = min(
                    len(self.dom_shards.get(1, [])), max(1, round(shard_budget * 0.28))
                )
                n_d2 = min(
                    len(self.dom_shards.get(2, [])),
                    max(1, shard_budget - n_d0 - n_d1),
                )
                init_sids: list[int] = []
                for d_id, k_cnt in ((0, n_d0), (1, n_d1), (2, n_d2)):
                    if self.dom_shards.get(d_id):
                        chosen = self._rot_rng.choice(
                            self.dom_shards[d_id],
                            size=min(k_cnt, len(self.dom_shards[d_id])),
                            replace=False,
                        )
                        init_sids.extend(int(x) for x in chosen)
                self.active_sids = sorted(set(init_sids))
        else:
            self.active_sids = list(range(self.total_shards))

        self.dom_items_all: dict[int, np.ndarray] = {}
        self.dom_items_non_sp: dict[int, np.ndarray] = {}
        self.dom_spk_to_items: dict[int, list[np.ndarray]] = {}
        self._rebuild_indices()

    def _get_shard_memmap(self, sid: int) -> np.memmap:
        if (
            sid < self.base_n_shards
            and not (self.using_shuffled_30_57 and 30 <= sid < 58)
        ):
            return self.store._memmaps[sid]
        mm = self._ext_memmaps.get(sid)
        if mm is None:
            s_path = self.shard_paths[sid]
            n_frames = os.path.getsize(s_path) // (self.feat_dim * 2)
            mm = np.memmap(
                s_path, dtype=np.float16, mode="r", shape=(n_frames, self.feat_dim)
            )
            self._ext_memmaps[sid] = mm
        return mm

    def _rebuild_indices(self) -> None:
        in_mem = np.isin(self.orig_shard_ids, self.active_sids)
        for d_id in self.all_domain_ids:
            d_mask = in_mem & (self.orig_domain_ids == d_id)
            self.dom_items_all[d_id] = np.nonzero(d_mask)[0].astype(np.int32)
            self.dom_items_non_sp[d_id] = np.nonzero(
                d_mask & (~self.orig_item_is_sp)
            )[0].astype(np.int32)
            d_idx = self.dom_items_all[d_id]
            if d_idx.size > 0:
                order = np.argsort(self.orig_speaker_ids[d_idx], kind="mergesort")
                sorted_items = d_idx[order]
                sorted_spks = self.orig_speaker_ids[sorted_items]
                splits = np.nonzero(np.diff(sorted_spks) != 0)[0] + 1
                self.dom_spk_to_items[d_id] = [
                    g for g in np.split(sorted_items, splits) if g.size >= 2
                ]
            else:
                self.dom_spk_to_items[d_id] = []

    def rotate_window(self) -> None:
        if not self.rolling_enabled:
            return
        cur_keys = set(self.active_sids)
        if self.diar_start_sid < self.total_shards and self.using_shuffled_30_57:
            pool_d = list(range(self.diar_start_sid, self.total_shards))
            pool_a = [
                s
                for s in range(self.diar_start_sid)
                if s < 30 or s >= self.base_n_shards
            ]
            pool_b = list(
                range(30, min(self.base_n_shards, self.diar_start_sid))
            )
            for pool_grp, k_max in ((pool_a, 2), (pool_b, 2), (pool_d, 2)):
                in_p = [s for s in pool_grp if s in cur_keys]
                out_p = [s for s in pool_grp if s not in cur_keys]
                n_sw = min(k_max, len(in_p), len(out_p))
                if n_sw > 0:
                    ev_l = self._rot_rng.choice(in_p, size=n_sw, replace=False)
                    ld_l = self._rot_rng.choice(out_p, size=n_sw, replace=False)
                    for e, ld_s in zip(ev_l, ld_l):
                        cur_keys.discard(int(e))
                        cur_keys.add(int(ld_s))
        elif self.using_shuffled_30_57:
            pool_a = [
                s
                for s in range(self.total_shards)
                if s < 30 or s >= self.base_n_shards
            ]
            pool_b = [
                s for s in range(30, min(self.base_n_shards, self.total_shards))
            ]
            for pool_grp in (pool_a, pool_b):
                in_p = [s for s in pool_grp if s in cur_keys]
                out_p = [s for s in pool_grp if s not in cur_keys]
                n_sw = min(3, len(in_p), len(out_p))
                if n_sw > 0:
                    ev_l = self._rot_rng.choice(in_p, size=n_sw, replace=False)
                    ld_l = self._rot_rng.choice(out_p, size=n_sw, replace=False)
                    for e, ld_s in zip(ev_l, ld_l):
                        cur_keys.discard(int(e))
                        cur_keys.add(int(ld_s))
        else:
            for d_id in self.all_domain_ids:
                all_d = self.dom_shards.get(d_id, [])
                in_d = [s for s in all_d if s in cur_keys]
                out_d = [s for s in all_d if s not in cur_keys]
                if in_d and out_d:
                    evict_sid = int(self._rot_rng.choice(in_d))
                    load_sid = int(self._rot_rng.choice(out_d))
                    cur_keys.discard(evict_sid)
                    cur_keys.add(load_sid)
        self.active_sids = sorted(cur_keys)
        self._rebuild_indices()

    @staticmethod
    def _dom_weight(dom_probs: Sequence[float], d_id: int) -> float:
        if d_id < len(dom_probs):
            return max(1e-6, float(dom_probs[d_id]))
        return max(1e-6, float(dom_probs[-1]) if dom_probs else 0.1)

    def sample_classification_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        seq_len: int,
        min_seq_len: int,
        variable_length: bool = True,
        non_sp_only: bool = False,
        dom_probs: Sequence[float] = (0.45, 0.32, 0.23),
        clean_remap_labels: bool = False,
        short_crop_prob: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pool_map = self.dom_items_non_sp if non_sp_only else self.dom_items_all
        avail_doms = [
            d for d in self.all_domain_ids if pool_map.get(d, np.empty(0)).size > 0
        ]
        probs = np.array(
            [self._dom_weight(dom_probs, d) for d in avail_doms], dtype=np.float64
        )
        probs /= probs.sum()
        chosen_doms = rng.choice(avail_doms, size=batch_size, p=probs)

        item_indices = np.empty((batch_size,), dtype=np.int32)
        for d in avail_doms:
            pos = np.nonzero(chosen_doms == d)[0]
            if pos.size > 0:
                pool = pool_map[d]
                item_indices[pos] = pool[rng.integers(0, pool.size, size=pos.size)]

        feats = np.zeros((batch_size, seq_len, self.feat_dim), dtype=np.float32)
        masks = np.zeros((batch_size, seq_len), dtype=np.float32)
        labels = self.orig_speaker_ids[item_indices].astype(np.int32)
        if clean_remap_labels and non_sp_only:
            labels = self.orig_to_clean_label[labels]
        eff_lens = rng.integers(min_seq_len, seq_len + 1, size=batch_size)
        if variable_length and short_crop_prob > 0.0 and seq_len > 220:
            short_hi = min(220, seq_len)
            short_lens = rng.integers(100, short_hi + 1, size=batch_size)
            use_short = rng.random(size=batch_size) < short_crop_prob
            eff_lens = np.where(use_short, short_lens, eff_lens)
        if not variable_length:
            eff_lens = np.full((batch_size,), seq_len, dtype=np.int64)

        for i in range(batch_size):
            idx = int(item_indices[i])
            mm = self._get_shard_memmap(int(self.orig_shard_ids[idx]))
            off = int(self.orig_frame_offsets[idx])
            ln = int(self.orig_frame_lengths[idx])
            target_l = int(eff_lens[i])
            if ln >= target_l:
                st = int(rng.integers(0, ln - target_l + 1))
                feats[i, :target_l] = mm[off + st:off + st + target_l]
                masks[i, :target_l] = 1.0
            else:
                valid_l = min(ln, target_l)
                feats[i, :valid_l] = mm[off:off + valid_l]
                masks[i, :valid_l] = 1.0
        return feats, labels, masks

    def sample_episode_batch(
        self,
        rng: np.random.Generator,
        num_speakers: int,
        num_utts_per_spk: int,
        seq_len: int,
        min_seq_len: int,
        variable_length: bool = True,
        dom_probs: Sequence[float] = (0.45, 0.32, 0.23),
    ) -> tuple[np.ndarray, np.ndarray]:
        avail_doms = [
            d for d in self.all_domain_ids if self.dom_spk_to_items.get(d)
        ]
        probs = np.array(
            [self._dom_weight(dom_probs, d) for d in avail_doms], dtype=np.float64
        )
        probs /= probs.sum()
        dom_choice = int(rng.choice(avail_doms, p=probs))
        groups = self.dom_spk_to_items[dom_choice]
        replace_spks = len(groups) < num_speakers
        spk_choices = rng.choice(len(groups), size=num_speakers, replace=replace_spks)

        total = num_speakers * num_utts_per_spk
        feats = np.zeros((total, seq_len, self.feat_dim), dtype=np.float32)
        masks = np.zeros((total, seq_len), dtype=np.float32)
        eff_lens = rng.integers(min_seq_len, seq_len + 1, size=total)
        if not variable_length:
            eff_lens = np.full((total,), seq_len, dtype=np.int64)

        row = 0
        for s_idx in spk_choices:
            u_pool = groups[int(s_idx)]
            chosen_u = rng.choice(
                u_pool, size=num_utts_per_spk, replace=(u_pool.size < num_utts_per_spk)
            )
            for idx in chosen_u:
                idx_i = int(idx)
                mm = self._get_shard_memmap(int(self.orig_shard_ids[idx_i]))
                off = int(self.orig_frame_offsets[idx_i])
                ln = int(self.orig_frame_lengths[idx_i])
                target_l = int(eff_lens[row])
                valid_l = min(ln, target_l)
                st = int(rng.integers(0, ln - valid_l + 1)) if ln > valid_l else 0
                feats[row, :valid_l] = mm[off + st:off + st + valid_l]
                masks[row, :valid_l] = 1.0
                row += 1
        return feats, masks
