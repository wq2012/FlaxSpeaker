"""Diarization-to-Speaker-Encoder dataset preparation and trial generation.

Converts multi-speaker conversational and meeting diarization corpora (such as
AMI, Fisher, ICSI, CallHome, VoxConverse, and DIHARD) into overlap-free
single-speaker segments suitable for training and evaluating speaker
verification / speaker encoder models.

Key capabilities:
1. Multi-format annotation parsing (`RTTM`, `CTM`, LDC/Fisher/CallHome
   transcript `.txt`, CSV/TSV, and JSON) plus official global speaker metadata
   resolvers (`fe_03_p*_calldata.tbl` for Fisher, `meetings.xml` for AMI).
2. Sweep-line multi-speaker overlap subtraction with configurable boundary
   collars (`overlap_collar_sec`) so cropped segments never mix simultaneous
   voices (e.g., backchannels inside long turns).
3. Pause-aware same-speaker turn merging (`max_merge_gap_sec`), duration
   filtering (`min_duration_sec`), monologue windowing (`max_duration_sec`),
   rational polyphase resampling (`8 kHz -> 16 kHz`), and RMS silence gating.
4. Diarization-derived speaker verification trial builders covering:
   - `standard`: corpus-wide verification with both within-session and
     cross-session impostors,
   - `within_session`: same-recording speaker discrimination under shared room
     or telephone channel acoustics (directly reflecting diarization clustering),
   - `short_turn`: rapid conversational turns (`0.80s - 2.00s`).
"""

from __future__ import annotations

import csv
import dataclasses
from fractions import Fraction
import json
import math
import os
import random
import re
from typing import Mapping, Optional, Sequence
from xml.etree import ElementTree as ET

import numpy as np
from scipy import signal
import soundfile as sf

_FISHER_CALL_RE = re.compile(r"(\d{5})")
_AMI_MEET_RE = re.compile(r"([A-Z]{2}\d{4}[a-z])")
_ICSI_MEET_RE = re.compile(r"([A-Za-z]{3}\d{3})")
_TRANSCRIPT_LINE_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([^:\s]+)\s*:?(.*)$"
)


@dataclasses.dataclass(frozen=True)
class SpeakerTurn:
    """Raw annotated speaker turn in a multi-speaker recording."""

    start_sec: float
    end_sec: float
    speaker: str
    session_id: str = ""
    transcript: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclasses.dataclass(frozen=True)
class CroppedSegmentSpec:
    """Overlap-free single-speaker segment specification."""

    session_id: str
    speaker_id: str
    start_sec: float
    end_sec: float
    segment_id: str

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclasses.dataclass(frozen=True)
class DiarizationCropConfig:
    """Configuration for converting diarization turns to speaker segments."""

    overlap_collar_sec: float = 0.10
    intra_turn_jitter_sec: float = 0.05
    max_merge_gap_sec: float = 0.30
    min_duration_sec: float = 0.80
    max_duration_sec: float = 8.00
    window_hop_sec: float = 6.00
    target_sample_rate: int = 16000
    min_rms: float = 1e-4


def parse_fisher_calldata_tbl(
    tbl_paths: Sequence[str],
) -> dict[tuple[str, str], str]:
    """Parses LDC Fisher `fe_03_p{1,2}_calldata.tbl` global caller PIN tables.

    Returns a mapping `(call_id_5digit, channel_id) -> "fisher:pin_<PIN>"`
    supporting both `'A'`/`'B'` and `'1'`/`'2'` channel labels.
    """
    pin_map: dict[tuple[str, str], str] = {}
    for path in tbl_paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                call_id = (row.get("CALL_ID") or "").strip()
                apin = (row.get("APIN") or "").strip()
                bpin = (row.get("BPIN") or "").strip()
                if call_id and apin:
                    pin_map[(call_id, "A")] = f"fisher:pin_{apin}"
                    pin_map[(call_id, "1")] = f"fisher:pin_{apin}"
                if call_id and bpin:
                    pin_map[(call_id, "B")] = f"fisher:pin_{bpin}"
                    pin_map[(call_id, "2")] = f"fisher:pin_{bpin}"
    return pin_map


def parse_ami_meetings_xml(meetings_xml_path: str) -> dict[tuple[str, str], str]:
    """Parses AMI `corpusResources/meetings.xml` into `(meeting, agent) -> global_id`."""
    ami_map: dict[tuple[str, str], str] = {}
    if not meetings_xml_path or not os.path.isfile(meetings_xml_path):
        return ami_map
    tree = ET.parse(meetings_xml_path)
    for meeting in tree.getroot():
        obs = meeting.attrib.get("observation", "").strip()
        for spk in meeting:
            agent = spk.attrib.get("nxt_agent", "").strip().upper()
            gname = spk.attrib.get("global_name", "").strip()
            if obs and agent and gname:
                ami_map[(obs, agent)] = f"ami:{gname}"
    return ami_map


def resolve_global_speaker_id(
    corpus: str,
    session_or_utt_id: str,
    raw_speaker_tag: str,
    fisher_pin_map: Optional[Mapping[tuple[str, str], str]] = None,
    ami_speaker_map: Optional[Mapping[tuple[str, str], str]] = None,
) -> tuple[str, str]:
    """Resolves `(canonical_session_id, global_speaker_id)` for diarization corpora."""
    corpus_norm = corpus.strip().lower()
    tag = raw_speaker_tag.strip()
    if tag.lower().startswith("speaker:"):
        tag = tag.split(":", 1)[1].strip()

    if corpus_norm == "fisher":
        m = _FISHER_CALL_RE.search(session_or_utt_id)
        call_id = m.group(1) if m else session_or_utt_id.strip()
        chan = tag.upper()
        if chan.startswith("FE_03_") and "_" in chan:
            chan = chan.rsplit("_", 1)[-1]
        elif chan.endswith("_A") or chan.endswith(".A"):
            chan = "A"
        elif chan.endswith("_B") or chan.endswith(".B"):
            chan = "B"
        sess_id = f"fisher:{call_id}"
        if fisher_pin_map and (call_id, chan) in fisher_pin_map:
            return sess_id, fisher_pin_map[(call_id, chan)]
        return sess_id, f"fisher:{call_id}:{chan}"

    if corpus_norm == "ami":
        m = _AMI_MEET_RE.search(session_or_utt_id)
        meet_id = m.group(1) if m else session_or_utt_id.strip()
        agent = tag.upper()
        if "." in agent:
            agent = agent.split(".")[-1]
        sess_id = f"ami:{meet_id}"
        if ami_speaker_map and (meet_id, agent) in ami_speaker_map:
            return sess_id, ami_speaker_map[(meet_id, agent)]
        # Scenario prefix (e.g. ES2002 across ES2002a/b/c/d) shares participants A/B/C/D
        scenario = meet_id[:-1] if len(meet_id) == 7 and meet_id[-1].islower() else meet_id
        return sess_id, f"ami:{scenario}:{agent}"

    if corpus_norm == "icsi":
        m = _ICSI_MEET_RE.search(session_or_utt_id)
        meet_id = m.group(1) if m else session_or_utt_id.strip()
        return f"icsi:{meet_id}", f"icsi:{tag.lower()}"

    if corpus_norm in ("callhome", "callhome_english"):
        call_id = os.path.basename(session_or_utt_id.strip())
        for ext in (".wav", ".sph", ".flac", ".txt", ".rttm"):
            if call_id.lower().endswith(ext):
                call_id = call_id[: -len(ext)]
        return f"callhome:{call_id}", f"callhome:{call_id}:{tag}"

    sess = session_or_utt_id.strip() or "session"
    prefix = corpus_norm or "diar"
    if tag.startswith(f"{prefix}:"):
        return f"{prefix}:{sess}", tag
    return f"{prefix}:{sess}", f"{prefix}:{tag}"


def parse_diarization_annotations(
    annotation_path: str,
    fmt: str = "auto",
    default_session_id: str = "",
) -> list[SpeakerTurn]:
    """Parses `RTTM`, `CSV`/`TSV`, `JSON`, or LDC transcript `.txt` into `SpeakerTurn`s."""
    ext = os.path.splitext(annotation_path)[1].lower()
    mode = fmt.lower().strip()
    if mode == "auto":
        if ext == ".rttm":
            mode = "rttm"
        elif ext == ".ctm":
            mode = "ctm"
        elif ext == ".json":
            mode = "json"
        elif ext in (".csv", ".tsv"):
            mode = "csv"
        else:
            mode = "txt"

    fallback_sess = default_session_id or os.path.splitext(
        os.path.basename(annotation_path)
    )[0]
    turns: list[SpeakerTurn] = []

    if mode == "rttm":
        with open(annotation_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                toks = line.strip().split()
                if len(toks) < 8 or toks[0].upper() != "SPEAKER":
                    continue
                sess = toks[1] or fallback_sess
                start = float(toks[3])
                dur = float(toks[4])
                spk = toks[7]
                if dur > 0.0:
                    turns.append(
                        SpeakerTurn(
                            start_sec=start,
                            end_sec=start + dur,
                            speaker=spk,
                            session_id=sess,
                        )
                    )
        return turns

    if mode == "ctm":
        with open(annotation_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                toks = line.strip().split()
                if len(toks) < 5 or toks[0].startswith(";"):
                    continue
                sess = toks[0] or fallback_sess
                chan = toks[1]
                start = float(toks[2])
                dur = float(toks[3])
                word = toks[4]
                if dur > 0.0:
                    turns.append(
                        SpeakerTurn(
                            start_sec=start,
                            end_sec=start + dur,
                            speaker=chan,
                            session_id=sess,
                            transcript=word,
                        )
                    )
        return turns

    if mode == "json":
        with open(annotation_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        items = payload.get("segments", payload) if isinstance(payload, dict) else payload
        for row in items:
            start = float(row.get("start", row.get("start_time", 0.0)))
            if "end" in row or "end_time" in row:
                end = float(row.get("end", row.get("end_time", start)))
            else:
                end = start + float(row.get("duration", 0.0))
            spk = str(row.get("speaker", row.get("speaker_id", ""))).strip()
            sess = str(row.get("session_id", fallback_sess)).strip() or fallback_sess
            if spk and end > start:
                turns.append(
                    SpeakerTurn(
                        start_sec=start,
                        end_sec=end,
                        speaker=spk,
                        session_id=sess,
                        transcript=str(row.get("transcript", row.get("text", ""))),
                    )
                )
        return turns

    if mode == "csv":
        delim = "\t" if ext == ".tsv" else ","
        with open(annotation_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delim)
            header: Optional[list[str]] = None
            for row in reader:
                if not row or all(not c.strip() for c in row):
                    continue
                if header is None:
                    low = [c.strip().lower() for c in row]
                    if any(k in low for k in ("start", "start_time", "speaker", "speaker_id")):
                        header = low
                        continue
                    header = ["start_time", "end_time", "speaker_id", "transcript"]
                col_map = {header[i]: row[i].strip() for i in range(min(len(header), len(row)))}
                try:
                    start = float(col_map.get("start_time", col_map.get("start", "")))
                    end = float(col_map.get("end_time", col_map.get("end", "")))
                except ValueError:
                    continue
                spk = col_map.get("speaker_id", col_map.get("speaker", "")).strip()
                sess = col_map.get("session_id", fallback_sess).strip() or fallback_sess
                text = col_map.get("transcript", col_map.get("text", "")).strip()
                if spk and end > start:
                    turns.append(
                        SpeakerTurn(
                            start_sec=start,
                            end_sec=end,
                            speaker=spk,
                            session_id=sess,
                            transcript=text,
                        )
                    )
        return turns

    # LDC Fisher / CallHome transcript `.txt` (`<start> <end> <spk>: <words>`)
    with open(annotation_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = _TRANSCRIPT_LINE_RE.match(stripped)
            if not m:
                continue
            start = float(m.group(1))
            end = float(m.group(2))
            spk = m.group(3).strip()
            text = m.group(4).strip()
            if spk and end > start:
                turns.append(
                    SpeakerTurn(
                        start_sec=start,
                        end_sec=end,
                        speaker=spk,
                        session_id=fallback_sess,
                        transcript=text,
                    )
                )
    return turns


def _merge_intervals(
    intervals: Sequence[tuple[float, float]], max_gap: float = 0.0
) -> list[tuple[float, float]]:
    """Merges sorted or unsorted intervals separated by at most `max_gap`."""
    valid = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not valid:
        return []
    merged: list[tuple[float, float]] = [valid[0]]
    for s, e in valid[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e + max_gap + 1e-9:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def _subtract_intervals(
    source: Sequence[tuple[float, float]],
    exclusions: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Subtracts sorted non-overlapping `exclusions` from `source` intervals."""
    if not source:
        return []
    if not exclusions:
        return list(source)
    out: list[tuple[float, float]] = []
    ex_idx = 0
    n_ex = len(exclusions)
    for s, e in source:
        cur = s
        while ex_idx < n_ex and exclusions[ex_idx][1] <= cur:
            ex_idx += 1
        j = ex_idx
        while j < n_ex and exclusions[j][0] < e:
            ex_s, ex_e = exclusions[j]
            if ex_s > cur + 1e-6:
                out.append((cur, min(e, ex_s)))
            cur = max(cur, ex_e)
            if cur >= e:
                break
            j += 1
        if cur + 1e-6 < e:
            out.append((cur, e))
    return out


def _interval_intersects_any(
    start: float, end: float, sorted_intervals: Sequence[tuple[float, float]]
) -> bool:
    """Returns True if `(start, end)` overlaps any interval in `sorted_intervals`."""
    if end <= start or not sorted_intervals:
        return False
    lo, hi = 0, len(sorted_intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        ms, me = sorted_intervals[mid]
        if me <= start + 1e-9:
            lo = mid + 1
        elif ms >= end - 1e-9:
            hi = mid - 1
        else:
            return True
    return False


def extract_clean_speaker_segments(
    turns: Sequence[SpeakerTurn],
    config: Optional[DiarizationCropConfig] = None,
    session_id_override: Optional[str] = None,
    audio_duration_sec: Optional[float] = None,
) -> list[CroppedSegmentSpec]:
    """Extracts overlap-free single-speaker segments from multi-speaker turns.

    Steps:
    1. Coalesce micro-gaps (`<= intra_turn_jitter_sec`) within each speaker's
       own annotations.
    2. Sweep across all speakers to find every interval where `>= 2` distinct
       speakers are active simultaneously, expand each overlap by
       `overlap_collar_sec` on both sides, and subtract them from all speakers.
    3. Merge adjacent single-speaker intervals of the same speaker when the
       inter-turn silence is `<= max_merge_gap_sec` and no other speaker was
       active during the gap.
    4. Filter out fragments shorter than `min_duration_sec` and split long
       monologues (`> max_duration_sec`) using `window_hop_sec`.
    """
    cfg = config or DiarizationCropConfig()
    if not turns:
        return []

    by_spk: dict[str, list[tuple[float, float]]] = {}
    inferred_sess = session_id_override or turns[0].session_id or "session"
    for t in turns:
        s = max(0.0, float(t.start_sec))
        e = float(t.end_sec)
        if audio_duration_sec is not None:
            e = min(e, float(audio_duration_sec))
        if t.speaker and e > s + 1e-4:
            by_spk.setdefault(t.speaker, []).append((s, e))

    if not by_spk:
        return []

    coalesced_by_spk: dict[str, list[tuple[float, float]]] = {
        spk: _merge_intervals(ivs, max_gap=cfg.intra_turn_jitter_sec)
        for spk, ivs in sorted(by_spk.items())
    }

    # Sweep-line over speaker interval boundaries to find >=2-speaker overlaps
    events: list[tuple[float, int]] = []
    all_raw_intervals: list[tuple[float, float, str]] = []
    for spk, ivs in coalesced_by_spk.items():
        for s, e in ivs:
            events.append((s, +1))
            events.append((e, -1))
            all_raw_intervals.append((s, e, spk))
    events.sort(key=lambda x: (x[0], -x[1]))

    raw_overlaps: list[tuple[float, float]] = []
    active = 0
    ov_start: Optional[float] = None
    for t_pt, delta in events:
        prev_active = active
        active += delta
        if prev_active < 2 and active >= 2:
            ov_start = t_pt
        elif prev_active >= 2 and active < 2 and ov_start is not None:
            if t_pt > ov_start + 1e-6:
                c_s = max(0.0, ov_start - cfg.overlap_collar_sec)
                c_e = t_pt + cfg.overlap_collar_sec
                if audio_duration_sec is not None:
                    c_e = min(c_e, float(audio_duration_sec))
                raw_overlaps.append((c_s, c_e))
            ov_start = None
    collared_overlaps = _merge_intervals(raw_overlaps, max_gap=0.0)

    segments: list[CroppedSegmentSpec] = []
    for spk, ivs in coalesced_by_spk.items():
        pure = _subtract_intervals(ivs, collared_overlaps)
        if not pure:
            continue
        other_ivs = _merge_intervals(
            [
                (s, e)
                for s, e, o_spk in all_raw_intervals
                if o_spk != spk
            ],
            max_gap=0.0,
        )
        merged_spk: list[tuple[float, float]] = [pure[0]]
        for s, e in pure[1:]:
            ps, pe = merged_spk[-1]
            gap = s - pe
            if (
                gap <= cfg.max_merge_gap_sec + 1e-9
                and not _interval_intersects_any(pe, s, other_ivs)
            ):
                merged_spk[-1] = (ps, e)
            else:
                merged_spk.append((s, e))

        for s, e in merged_spk:
            dur = e - s
            if dur + 1e-6 < cfg.min_duration_sec:
                continue
            if dur <= cfg.max_duration_sec + 1e-6:
                seg_id = (
                    f"{inferred_sess}__{spk}"
                    f"__{int(round(s * 100)):06d}_{int(round(e * 100)):06d}"
                )
                segments.append(
                    CroppedSegmentSpec(
                        session_id=inferred_sess,
                        speaker_id=spk,
                        start_sec=round(s, 4),
                        end_sec=round(e, 4),
                        segment_id=seg_id,
                    )
                )
            else:
                hop = max(cfg.min_duration_sec, cfg.window_hop_sec)
                cur = s
                while cur + cfg.min_duration_sec <= e + 1e-6:
                    win_end = min(e, cur + cfg.max_duration_sec)
                    if e - win_end < cfg.min_duration_sec:
                        win_end = e
                    seg_id = (
                        f"{inferred_sess}__{spk}"
                        f"__{int(round(cur * 100)):06d}_{int(round(win_end * 100)):06d}"
                    )
                    segments.append(
                        CroppedSegmentSpec(
                            session_id=inferred_sess,
                            speaker_id=spk,
                            start_sec=round(cur, 4),
                            end_sec=round(win_end, 4),
                            segment_id=seg_id,
                        )
                    )
                    if win_end >= e - 1e-6:
                        break
                    cur += hop

    segments.sort(key=lambda seg: (seg.start_sec, seg.end_sec, seg.speaker_id))
    return segments


def resample_waveform_poly(
    waveform: np.ndarray, orig_sr: int, target_sr: int = 16000
) -> np.ndarray:
    """Resamples a 1D float32 waveform using rational polyphase filtering."""
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr or wav.size == 0:
        return wav
    frac = Fraction(int(target_sr), int(orig_sr)).limit_denominator(200)
    resampled = signal.resample_poly(wav, up=frac.numerator, down=frac.denominator)
    return np.asarray(resampled, dtype=np.float32)


def crop_waveform_segments(
    waveform: np.ndarray,
    sample_rate: int,
    segments: Sequence[CroppedSegmentSpec],
    config: Optional[DiarizationCropConfig] = None,
) -> list[tuple[CroppedSegmentSpec, np.ndarray]]:
    """Slices and resamples overlap-free speaker segments from a recording."""
    cfg = config or DiarizationCropConfig()
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    total_samples = int(wav.shape[0])
    results: list[tuple[CroppedSegmentSpec, np.ndarray]] = []
    min_target_samples = int(round(cfg.min_duration_sec * cfg.target_sample_rate))

    for seg in segments:
        s_idx = max(0, int(round(seg.start_sec * sample_rate)))
        e_idx = min(total_samples, int(round(seg.end_sec * sample_rate)))
        if e_idx <= s_idx:
            continue
        crop = wav[s_idx:e_idx]
        if sample_rate != cfg.target_sample_rate:
            crop = resample_waveform_poly(crop, sample_rate, cfg.target_sample_rate)
        if crop.shape[0] < min_target_samples:
            continue
        rms = float(math.sqrt(float(np.mean(np.square(crop, dtype=np.float64)))))
        if rms < cfg.min_rms:
            continue
        results.append((seg, crop))
    return results


def load_and_crop_recording(
    audio_path: str,
    turns: Sequence[SpeakerTurn],
    config: Optional[DiarizationCropConfig] = None,
    session_id_override: Optional[str] = None,
) -> list[tuple[CroppedSegmentSpec, np.ndarray]]:
    """Loads an audio file from disk and returns overlap-free speaker crops."""
    cfg = config or DiarizationCropConfig()
    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    dur_sec = float(wav.shape[0]) / max(1.0, float(sr))
    seg_specs = extract_clean_speaker_segments(
        turns,
        config=cfg,
        session_id_override=session_id_override,
        audio_duration_sec=dur_sec,
    )
    return crop_waveform_segments(wav, int(sr), seg_specs, config=cfg)


def generate_diarization_sv_trials(
    segments: Sequence[CroppedSegmentSpec],
    utt_id_to_index: Mapping[str, int],
    num_trials: int = 4000,
    mode: str = "standard",
    min_same_session_gap_sec: float = 15.0,
    short_turn_max_sec: float = 2.05,
    seed: int = 42,
) -> list[list[int]]:
    """Generates deterministic `[label, enroll_idx, test_idx]` diarization SV trials.

    Modes:
    - `within_session`: Both positive and negative pairs come strictly from the
      same recording/session (`seg_a.session_id == seg_b.session_id`), testing
      speaker discrimination under identical room/channel acoustics. Positives
      enforce temporal separation `>= min_same_session_gap_sec` so adjacent
      chunks of a single monologue are never paired.
    - `standard`: Prefers cross-session positive pairs when speakers appear in
      multiple sessions (falling back to temporally separated within-session
      positives for single-session speakers), paired with a 50/50 mix of hard
      within-session impostors and cross-session impostors.
    - `short_turn`: Enforces `test_seg.duration_sec <= short_turn_max_sec`
      (`0.80s - 2.00s` rapid conversational turns) against `>= 1.50s` enrollment
      segments, with balanced within-session and cross-session negatives.
    """
    rng = random.Random(seed)
    indexed = [s for s in segments if s.segment_id in utt_id_to_index]
    if len(indexed) < 4:
        return []

    by_session: dict[str, list[CroppedSegmentSpec]] = {}
    by_speaker: dict[str, list[CroppedSegmentSpec]] = {}
    for seg in indexed:
        by_session.setdefault(seg.session_id, []).append(seg)
        by_speaker.setdefault(seg.speaker_id, []).append(seg)

    pos_candidates: list[tuple[int, int]] = []
    neg_within_candidates: list[tuple[int, int]] = []
    neg_cross_candidates: list[tuple[int, int]] = []

    if mode == "within_session":
        for sess_segs in by_session.values():
            spk_groups: dict[str, list[CroppedSegmentSpec]] = {}
            for s in sess_segs:
                spk_groups.setdefault(s.speaker_id, []).append(s)
            for spk_s in spk_groups.values():
                if len(spk_s) < 2:
                    continue
                sampled_s = (
                    rng.sample(spk_s, 40) if len(spk_s) > 40 else list(spk_s)
                )
                for i in range(len(sampled_s)):
                    for j in range(i + 1, len(sampled_s)):
                        a, b = sampled_s[i], sampled_s[j]
                        gap = max(a.start_sec - b.end_sec, b.start_sec - a.end_sec)
                        if gap >= min_same_session_gap_sec:
                            ia = utt_id_to_index[a.segment_id]
                            ib = utt_id_to_index[b.segment_id]
                            pos_candidates.append((min(ia, ib), max(ia, ib)))
            spk_keys = sorted(spk_groups.keys())
            for i in range(len(spk_keys)):
                for j in range(i + 1, len(spk_keys)):
                    ga = (
                        rng.sample(spk_groups[spk_keys[i]], 25)
                        if len(spk_groups[spk_keys[i]]) > 25
                        else spk_groups[spk_keys[i]]
                    )
                    gb = (
                        rng.sample(spk_groups[spk_keys[j]], 25)
                        if len(spk_groups[spk_keys[j]]) > 25
                        else spk_groups[spk_keys[j]]
                    )
                    for a in ga:
                        for b in gb:
                            ia = utt_id_to_index[a.segment_id]
                            ib = utt_id_to_index[b.segment_id]
                            neg_within_candidates.append((ia, ib))

    elif mode == "short_turn":
        for spk_s in by_speaker.values():
            enrolls = [s for s in spk_s if s.duration_sec >= 1.50]
            shorts = [s for s in spk_s if s.duration_sec <= short_turn_max_sec]
            if not enrolls or not shorts:
                continue
            enrolls_sub = rng.sample(enrolls, 25) if len(enrolls) > 25 else enrolls
            shorts_sub = rng.sample(shorts, 25) if len(shorts) > 25 else shorts
            for a in enrolls_sub:
                for b in shorts_sub:
                    if a.segment_id == b.segment_id:
                        continue
                    if a.session_id == b.session_id:
                        gap = max(a.start_sec - b.end_sec, b.start_sec - a.end_sec)
                        if gap < min_same_session_gap_sec:
                            continue
                    pos_candidates.append(
                        (utt_id_to_index[a.segment_id], utt_id_to_index[b.segment_id])
                    )
        for sess_segs in by_session.values():
            enrolls = [s for s in sess_segs if s.duration_sec >= 1.50]
            shorts = [s for s in sess_segs if s.duration_sec <= short_turn_max_sec]
            enrolls_sub = rng.sample(enrolls, 30) if len(enrolls) > 30 else enrolls
            shorts_sub = rng.sample(shorts, 30) if len(shorts) > 30 else shorts
            for a in enrolls_sub:
                for b in shorts_sub:
                    if a.speaker_id != b.speaker_id:
                        neg_within_candidates.append(
                            (utt_id_to_index[a.segment_id], utt_id_to_index[b.segment_id])
                        )
        all_enrolls = [s for s in indexed if s.duration_sec >= 1.50]
        all_shorts = [s for s in indexed if s.duration_sec <= short_turn_max_sec]
        if all_enrolls and all_shorts:
            for _ in range(max(num_trials * 2, 2000)):
                a = rng.choice(all_enrolls)
                b = rng.choice(all_shorts)
                if a.speaker_id != b.speaker_id and a.session_id != b.session_id:
                    neg_cross_candidates.append(
                        (utt_id_to_index[a.segment_id], utt_id_to_index[b.segment_id])
                    )

    else:  # standard
        for spk_s in by_speaker.values():
            if len(spk_s) < 2:
                continue
            cross_pairs: list[tuple[int, int]] = []
            within_pairs: list[tuple[int, int]] = []
            sub = rng.sample(spk_s, 45) if len(spk_s) > 45 else list(spk_s)
            for i in range(len(sub)):
                for j in range(i + 1, len(sub)):
                    a, b = sub[i], sub[j]
                    ia = utt_id_to_index[a.segment_id]
                    ib = utt_id_to_index[b.segment_id]
                    if a.session_id != b.session_id:
                        cross_pairs.append((ia, ib))
                    else:
                        gap = max(a.start_sec - b.end_sec, b.start_sec - a.end_sec)
                        if gap >= min_same_session_gap_sec:
                            within_pairs.append((ia, ib))
            chosen_spk_pos = cross_pairs if cross_pairs else within_pairs
            pos_candidates.extend(chosen_spk_pos)

        for sess_segs in by_session.values():
            sub = rng.sample(sess_segs, 40) if len(sess_segs) > 40 else sess_segs
            for i in range(len(sub)):
                for j in range(i + 1, len(sub)):
                    if sub[i].speaker_id != sub[j].speaker_id:
                        neg_within_candidates.append(
                            (
                                utt_id_to_index[sub[i].segment_id],
                                utt_id_to_index[sub[j].segment_id],
                            )
                        )
        spk_list = sorted(by_speaker.keys())
        if len(spk_list) >= 2:
            for _ in range(max(num_trials * 2, 2000)):
                sa, sb = rng.sample(spk_list, 2)
                a = rng.choice(by_speaker[sa])
                b = rng.choice(by_speaker[sb])
                if a.session_id != b.session_id:
                    neg_cross_candidates.append(
                        (utt_id_to_index[a.segment_id], utt_id_to_index[b.segment_id])
                    )

    pos_unique = sorted(set(pos_candidates))
    neg_w_unique = sorted(set(neg_within_candidates))
    neg_c_unique = sorted(set(neg_cross_candidates))

    target_pos = num_trials // 2
    target_neg = num_trials - target_pos
    if not pos_unique or (not neg_w_unique and not neg_c_unique):
        return []

    chosen_pos = (
        rng.sample(pos_unique, target_pos)
        if len(pos_unique) >= target_pos
        else pos_unique
    )
    target_neg = min(target_neg, len(chosen_pos) * 2)

    if mode == "within_session" or not neg_c_unique:
        chosen_neg = (
            rng.sample(neg_w_unique, min(target_neg, len(neg_w_unique)))
            if neg_w_unique
            else []
        )
    elif not neg_w_unique:
        chosen_neg = rng.sample(neg_c_unique, min(target_neg, len(neg_c_unique)))
    else:
        half_w = min(len(neg_w_unique), target_neg // 2)
        rem_c = min(len(neg_c_unique), target_neg - half_w)
        chosen_neg = rng.sample(neg_w_unique, half_w) + rng.sample(neg_c_unique, rem_c)
        if len(chosen_neg) < target_neg:
            leftover = [p for p in neg_w_unique if p not in set(chosen_neg)]
            extra = min(len(leftover), target_neg - len(chosen_neg))
            if extra > 0:
                chosen_neg.extend(rng.sample(leftover, extra))

    trials: list[list[int]] = [[1, a, b] for a, b in chosen_pos] + [
        [0, a, b] for a, b in chosen_neg
    ]
    rng.shuffle(trials)
    return trials
