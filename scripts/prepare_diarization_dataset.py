#!/usr/bin/env python3
"""Prepare speaker encoder training & verification datasets from diarization corpora.

Converts public multi-speaker diarization recordings + timestamped annotations
(`RTTM`, `CSV`/`TSV`, `JSON`, or LDC Fisher/CallHome transcript `.txt`) into:
  1. Overlap-free 16 kHz mono WAV speaker segments (`wavs/<speaker_id>/<seg>.wav`),
  2. Speaker-disjoint `train.csv` and `eval.csv` manifests compatible with
     `scripts/train.py` and `flaxspeaker.shard_cache`,
  3. Diarization-derived speaker verification trial lists (`trials_standard.tsv`,
     `trials_within_session.tsv`, `trials_short_turn.tsv`) compatible with
     `scripts/evaluate.py`.

Usage:
    # Real diarization dataset from a manifest CSV (`session_id,audio_path,annotation_path`):
    python scripts/prepare_diarization_dataset.py \
        --manifest_csv /path/to/ami_sessions.csv \
        --corpus ami \
        --ami_meetings_xml /path/to/corpusResources/meetings.xml \
        --output_dir ~/Data/speaker_datasets/ami_sv

    # Self-contained synthetic multi-speaker meeting smoke test:
    python scripts/prepare_diarization_dataset.py \
        --generate_synthetic \
        --output_dir /tmp/flaxspeaker_synthetic_diar
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import random
import sys
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flaxspeaker import dataset  # noqa: E402
from flaxspeaker import diarization_prep  # noqa: E402


def generate_synthetic_diarization_corpus(
    output_dir: str,
    num_sessions: int = 8,
    speakers_per_session: int = 3,
    sample_rate: int = 16000,
    session_duration_sec: float = 28.0,
    seed: int = 42,
) -> str:
    """Generates synthetic multi-speaker meeting WAVs + RTTMs with overlaps."""
    rng = np.random.default_rng(seed)
    raw_dir = os.path.join(output_dir, "raw_meetings")
    os.makedirs(raw_dir, exist_ok=True)
    manifest_csv = os.path.join(raw_dir, "sessions.csv")

    n_total_spks = max(speakers_per_session * 2, 10)
    spk_f0 = {
        f"spk{idx:02d}": float(rng.uniform(95.0, 260.0))
        for idx in range(n_total_spks)
    }

    rows: list[list[str]] = []
    n_samples = int(round(session_duration_sec * sample_rate))
    for s_idx in range(num_sessions):
        sess_id = f"meet_{s_idx:03d}"
        # First half of sessions use train speakers, second half use disjoint eval speakers
        half_spks = n_total_spks // 2
        pool = (
            sorted(spk_f0.keys())[:half_spks]
            if s_idx < num_sessions // 2
            else sorted(spk_f0.keys())[half_spks:]
        )
        chosen = list(rng.choice(pool, size=min(speakers_per_session, len(pool)), replace=False))
        mix = rng.normal(0.0, 0.005, size=n_samples).astype(np.float32)
        rttm_lines: list[str] = []

        cur_t = 0.25
        turn_idx = 0
        while cur_t + 2.2 < session_duration_sec:
            spk = chosen[turn_idx % len(chosen)]
            dur = float(rng.uniform(1.4, 3.6))
            end_t = min(session_duration_sec - 0.1, cur_t + dur)
            s_i = int(round(cur_t * sample_rate))
            e_i = int(round(end_t * sample_rate))
            t_vec = np.arange(e_i - s_i, dtype=np.float32) / float(sample_rate)
            f0 = spk_f0[spk] * float(rng.normal(1.0, 0.01))
            harmonic = (
                0.45 * np.sin(2.0 * np.pi * f0 * t_vec)
                + 0.25 * np.sin(2.0 * np.pi * 2.0 * f0 * t_vec)
                + 0.15 * np.sin(2.0 * np.pi * 3.1 * f0 * t_vec)
            )
            mix[s_i:e_i] += harmonic.astype(np.float32)
            rttm_lines.append(
                f"SPEAKER {sess_id} 1 {cur_t:.3f} {end_t - cur_t:.3f} <NA> <NA> {spk} <NA> <NA>"
            )
            # Inject an overlapping backchannel every 3rd turn
            if turn_idx % 3 == 1 and (end_t - cur_t) > 2.2:
                bc_spk = chosen[(turn_idx + 1) % len(chosen)]
                bc_s = cur_t + 0.8
                bc_d = 0.45
                bs_i = int(round(bc_s * sample_rate))
                be_i = min(n_samples, int(round((bc_s + bc_d) * sample_rate)))
                bt_vec = np.arange(be_i - bs_i, dtype=np.float32) / float(sample_rate)
                bf0 = spk_f0[bc_spk]
                mix[bs_i:be_i] += (0.35 * np.sin(2.0 * np.pi * bf0 * bt_vec)).astype(np.float32)
                rttm_lines.append(
                    f"SPEAKER {sess_id} 1 {bc_s:.3f} {bc_d:.3f} <NA> <NA> {bc_spk} <NA> <NA>"
                )
            cur_t = end_t + float(rng.uniform(0.15, 0.45))
            turn_idx += 1

        peak = float(np.max(np.abs(mix)))
        if peak > 0.95:
            mix = mix / peak * 0.90
        wav_p = os.path.join(raw_dir, f"{sess_id}.wav")
        rttm_p = os.path.join(raw_dir, f"{sess_id}.rttm")
        sf.write(wav_p, mix, sample_rate)
        with open(rttm_p, "w", encoding="utf-8") as rf:
            rf.write("\n".join(rttm_lines) + "\n")
        split_tag = "train" if s_idx < num_sessions // 2 else "eval"
        rows.append([sess_id, wav_p, rttm_p, split_tag])

    with open(manifest_csv, "w", encoding="utf-8", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["session_id", "audio_path", "annotation_path", "split"])
        writer.writerows(rows)
    return manifest_csv


def prepare_diarization_corpus(
    manifest_csv: str,
    output_dir: str,
    corpus: str = "diar",
    fisher_calldata_tbls: list[str] | None = None,
    ami_meetings_xml: str = "",
    global_speaker_ids: bool = False,
    crop_config: diarization_prep.DiarizationCropConfig | None = None,
    eval_speaker_ratio: float = 0.20,
    num_eval_trials: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Crops overlap-free speaker segments and builds disjoint SV train/eval splits."""
    cfg = crop_config or diarization_prep.DiarizationCropConfig()
    os.makedirs(output_dir, exist_ok=True)
    wav_out_root = os.path.join(output_dir, "wavs")
    os.makedirs(wav_out_root, exist_ok=True)

    fisher_map = diarization_prep.parse_fisher_calldata_tbl(fisher_calldata_tbls or [])
    ami_map = diarization_prep.parse_ami_meetings_xml(ami_meetings_xml)

    sessions: list[tuple[str, str, str, str]] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().lower() == "session_id":
                continue
            sess_id = row[0].strip()
            audio_p = row[1].strip()
            ann_p = row[2].strip()
            split_hint = row[3].strip().lower() if len(row) >= 4 else ""
            sessions.append((sess_id, audio_p, ann_p, split_hint))

    all_records: list[tuple[diarization_prep.CroppedSegmentSpec, str, str]] = []
    for sess_id, audio_p, ann_p, split_hint in sessions:
        raw_turns = diarization_prep.parse_diarization_annotations(
            ann_p, default_session_id=sess_id
        )
        resolved_turns: list[diarization_prep.SpeakerTurn] = []
        canon_sess = sess_id
        for t in raw_turns:
            canon_sess, g_spk = diarization_prep.resolve_global_speaker_id(
                corpus=corpus,
                session_or_utt_id=sess_id,
                raw_speaker_tag=t.speaker,
                fisher_pin_map=fisher_map,
                ami_speaker_map=ami_map,
                global_speaker_ids=global_speaker_ids,
            )
            resolved_turns.append(
                diarization_prep.SpeakerTurn(
                    start_sec=t.start_sec,
                    end_sec=t.end_sec,
                    speaker=g_spk,
                    session_id=canon_sess,
                    transcript=t.transcript,
                )
            )
        cropped = diarization_prep.load_and_crop_recording(
            audio_p, resolved_turns, config=cfg, session_id_override=canon_sess
        )
        for seg_spec, wav_arr in cropped:
            safe_spk = seg_spec.speaker_id.replace(":", "_").replace("/", "_")
            safe_seg = seg_spec.segment_id.replace(":", "_").replace("/", "_")
            spk_dir = os.path.join(wav_out_root, safe_spk)
            os.makedirs(spk_dir, exist_ok=True)
            out_wav = os.path.abspath(os.path.join(spk_dir, f"{safe_seg}.wav"))
            sf.write(out_wav, wav_arr, cfg.target_sample_rate)
            all_records.append((seg_spec, out_wav, split_hint))

    # Enforce strict speaker disjointness between train and eval.
    # When speakers are session-scoped (each speaker appears in only 1 session),
    # hold out whole sessions so every eval recording retains all of its speakers
    # for within-session negative trials.
    explicit_eval_spks = {
        rec[0].speaker_id for rec in all_records if rec[2] in ("eval", "test", "dev")
    }
    all_spks = sorted({rec[0].speaker_id for rec in all_records})
    if not explicit_eval_spks and len(all_spks) >= 2:
        rng = random.Random(seed)
        spk_to_sessions: dict[str, set[str]] = {}
        sess_to_spks: dict[str, set[str]] = {}
        for rec in all_records:
            spk_to_sessions.setdefault(rec[0].speaker_id, set()).add(rec[0].session_id)
            sess_to_spks.setdefault(rec[0].session_id, set()).add(rec[0].speaker_id)
        all_sessions = sorted(sess_to_spks.keys())
        if len(all_sessions) >= 2 and all(
            len(s_set) == 1 for s_set in spk_to_sessions.values()
        ):
            n_eval_sess = max(1, int(round(len(all_sessions) * eval_speaker_ratio)))
            n_eval_sess = min(n_eval_sess, len(all_sessions) - 1)
            chosen_sess = set(rng.sample(all_sessions, n_eval_sess))
            explicit_eval_spks = {
                spk for s in chosen_sess for spk in sess_to_spks[s]
            }
        else:
            n_eval = max(2, int(round(len(all_spks) * eval_speaker_ratio)))
            n_eval = min(n_eval, len(all_spks) - 1)
            explicit_eval_spks = set(rng.sample(all_spks, n_eval))

    train_spk_to_utts: dict[str, list[str]] = {}
    eval_spk_to_utts: dict[str, list[str]] = {}
    eval_specs: list[diarization_prep.CroppedSegmentSpec] = []
    seg_id_to_path: dict[str, str] = {}
    seg_id_to_idx: dict[str, int] = {}

    for seg_spec, out_wav, _ in all_records:
        if seg_spec.speaker_id in explicit_eval_spks:
            eval_spk_to_utts.setdefault(seg_spec.speaker_id, []).append(out_wav)
            seg_id_to_idx[seg_spec.segment_id] = len(eval_specs)
            seg_id_to_path[seg_spec.segment_id] = out_wav
            eval_specs.append(seg_spec)
        else:
            train_spk_to_utts.setdefault(seg_spec.speaker_id, []).append(out_wav)

    overlap = set(train_spk_to_utts.keys()) & set(eval_spk_to_utts.keys())
    if overlap:
        raise AssertionError(f"Speaker overlap detected: {overlap}")

    train_csv = os.path.join(output_dir, "train.csv")
    eval_csv = os.path.join(output_dir, "eval.csv")
    dataset.save_spk_to_utts_csv(train_spk_to_utts, train_csv)
    dataset.save_spk_to_utts_csv(eval_spk_to_utts, eval_csv)

    trial_files: dict[str, int] = {}
    for mode in ("standard", "within_session", "short_turn"):
        idx_trials = diarization_prep.generate_diarization_sv_trials(
            eval_specs,
            seg_id_to_idx,
            num_trials=num_eval_trials,
            mode=mode,
            min_same_session_gap_sec=3.0,
            seed=seed,
        )
        path_trials = [
            (
                int(lbl),
                seg_id_to_path[eval_specs[ia].segment_id],
                seg_id_to_path[eval_specs[ib].segment_id],
            )
            for lbl, ia, ib in idx_trials
        ]
        if path_trials:
            t_path = os.path.join(output_dir, f"trials_{mode}.tsv")
            dataset.save_trials_file(path_trials, t_path)
            trial_files[mode] = len(path_trials)

    summary = {
        "corpus": corpus,
        "output_dir": os.path.abspath(output_dir),
        "train_csv": os.path.abspath(train_csv),
        "eval_csv": os.path.abspath(eval_csv),
        "train_speakers": len(train_spk_to_utts),
        "eval_speakers": len(eval_spk_to_utts),
        "train_segments": sum(len(v) for v in train_spk_to_utts.values()),
        "eval_segments": sum(len(v) for v in eval_spk_to_utts.values()),
        "zero_speaker_overlap": True,
        "trial_counts": trial_files,
        "crop_config": dataclasses.asdict(cfg),
    }
    with open(os.path.join(output_dir, "diarization_summary.json"), "w", encoding="utf-8") as sf_out:
        json.dump(summary, sf_out, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop overlap-free speaker segments and SV trials from diarization datasets."
    )
    parser.add_argument("--manifest_csv", type=str, default="", help="Session manifest CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--corpus", type=str, default="diar", help="Corpus name (ami, fisher, icsi, callhome).")
    parser.add_argument(
        "--fisher_calldata_tbls",
        type=str,
        default="",
        help="Comma-separated Fisher calldata.tbl paths.",
    )
    parser.add_argument("--ami_meetings_xml", type=str, default="", help="Path to AMI meetings.xml.")
    parser.add_argument(
        "--global_speaker_ids",
        action="store_true",
        help="Treat raw RTTM/transcript speaker tags as corpus-global IDs instead of session-local labels.",
    )
    parser.add_argument("--overlap_collar_sec", type=float, default=0.10, help="Collar around multi-speaker overlaps.")
    parser.add_argument("--min_duration_sec", type=float, default=0.80, help="Minimum cropped segment duration (s).")
    parser.add_argument("--max_duration_sec", type=float, default=8.00, help="Maximum cropped segment duration (s).")
    parser.add_argument("--generate_synthetic", action="store_true", help="Generate synthetic meetings first.")
    args = parser.parse_args()

    manifest = args.manifest_csv
    if args.generate_synthetic:
        manifest = generate_synthetic_diarization_corpus(args.output_dir)
    if not manifest:
        parser.error("Provide --manifest_csv or --generate_synthetic.")

    crop_cfg = diarization_prep.DiarizationCropConfig(
        overlap_collar_sec=args.overlap_collar_sec,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )
    tbls = [p.strip() for p in args.fisher_calldata_tbls.split(",") if p.strip()]
    summary = prepare_diarization_corpus(
        manifest_csv=manifest,
        output_dir=args.output_dir,
        corpus=args.corpus,
        fisher_calldata_tbls=tbls,
        ami_meetings_xml=args.ami_meetings_xml,
        global_speaker_ids=bool(args.global_speaker_ids or args.generate_synthetic),
        crop_config=crop_cfg,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
