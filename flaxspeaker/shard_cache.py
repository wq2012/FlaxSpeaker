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
