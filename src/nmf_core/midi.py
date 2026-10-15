"""Post-processing: activations H -> note events -> MIDI file, plus .npz dumps.

1. ``pitch_activations`` sums each pitch's atom rows of H into one row per
   pitch (the piano roll).
2. ``roll_to_notes`` normalises the roll by its global maximum, so the
   threshold is relative to the loudest activation in the piece, and applies
   peak thresholding with hysteresis per pitch:
     * every activation peak at least ``threshold`` high, and standing at
       least ``threshold`` above the valleys around it, triggers a Note-On
       placed at the start of its rise (so repeated notes re-attack);
     * the Note-Off comes when the activation falls below
       ``threshold * offset_ratio``, or at the next Note-On of that pitch.
   Frames become seconds with t = frame * hop_length / sample_rate.
3. ``refine_offsets`` optionally re-decides Note-Offs from a second roll
   (used for group-sparse NMF, see ``pipeline``).
4. ``write_midi`` writes the notes with pretty_midi.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pretty_midi
from scipy.signal import find_peaks

from .errors import ParameterError


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    start: float
    end: float
    velocity: int


def pitch_activations(H: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Collapse atom activations (K x T) to one row per pitch (P x T)."""
    return np.stack([H[idx].sum(axis=0) for idx in groups])


def frames_to_seconds(frames: np.ndarray | int, sr: int, hop_length: int):
    return np.asarray(frames) * hop_length / sr


def roll_to_notes(
    roll: np.ndarray,
    pitches: list[int],
    sr: int,
    hop_length: int,
    threshold: float = 0.1,
    offset_ratio: float = 0.5,
    min_duration: float = 0.05,
) -> list[NoteEvent]:
    """Turn a pitch activation roll (P x T) into note events."""
    roll = np.asarray(roll, dtype=np.float64)
    if roll.ndim != 2 or roll.shape[0] != len(pitches):
        raise ParameterError(
            f"roll shape {roll.shape} does not match {len(pitches)} pitches."
        )
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ParameterError(f"threshold must be in (0, 1], got {threshold}.")
    if not 0 < offset_ratio <= 1:
        raise ParameterError("offset_ratio must be in (0, 1].")
    if min_duration < 0:
        raise ParameterError("min_duration must be >= 0.")
    if sr <= 0 or hop_length <= 0:
        raise ParameterError("sr and hop_length must be positive.")
    peak = roll.max() if roll.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return []
    roll = roll / peak
    off_level = threshold * offset_ratio
    T = roll.shape[1]
    frame_s = hop_length / sr

    notes: list[NoteEvent] = []
    for pitch, row in zip(pitches, roll, strict=True):
        if row.max() < threshold:
            continue
        # Pad with zeros so a note sounding at the very first/last frame
        # still forms a peak.
        padded = np.concatenate(([0.0], row, [0.0]))
        peaks, _ = find_peaks(padded, height=threshold, prominence=threshold)
        peaks = peaks - 1
        onsets = []
        for p in peaks:
            # Walk back to the start of the rise, then forward to the frame
            # where the rise is half complete. The half-rise point is robust
            # to the time smearing of long low-frequency CQT filters.
            base = p
            while base > 0 and row[base] > 0 and row[base - 1] <= row[base]:
                base -= 1
            half = row[base] + 0.5 * (row[p] - row[base])
            onsets.append(base + int(np.argmax(row[base : p + 1] >= half)))
        for i, (on, p) in enumerate(zip(onsets, peaks, strict=True)):
            limit = onsets[i + 1] if i + 1 < len(onsets) else T
            below = np.nonzero(row[p:limit] < off_level)[0]
            off = p + below[0] if below.size else limit
            if off > on:
                notes.append(_note(pitch, on, off, row[p], frame_s))
    notes = [n for n in notes if n.end - n.start >= min_duration]
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def refine_offsets(
    notes: list[NoteEvent],
    roll: np.ndarray,
    pitches: list[int],
    sr: int,
    hop_length: int,
    off_level: float = 0.05,
    peak_window: int = 10,
) -> list[NoteEvent]:
    """Re-decide each note's Note-Off from a second activation roll.

    Pitch and onset of every note are kept; only the end changes. For each
    note the peak of ``roll`` within ``peak_window`` frames after the onset is
    found, and the note ends at the first later frame where the roll (scaled
    by its global maximum) drops below ``off_level``, or at the next onset of
    the same pitch. Used to undo the tail shrinkage of group-sparse NMF.
    """
    roll = np.asarray(roll, dtype=np.float64)
    if roll.ndim != 2 or roll.shape[0] != len(pitches):
        raise ParameterError(
            f"roll shape {roll.shape} does not match {len(pitches)} pitches."
        )
    peak = roll.max() if roll.size else 0.0
    if not notes or not np.isfinite(peak) or peak <= 0:
        return list(notes)
    roll = roll / peak
    T = roll.shape[1]
    frame_s = hop_length / sr
    row_of = {p: i for i, p in enumerate(pitches)}
    starts: dict[int, list[int]] = {}
    for n in notes:
        starts.setdefault(n.pitch, []).append(int(round(n.start / frame_s)))

    refined = []
    for n in notes:
        if n.pitch not in row_of:
            refined.append(n)
            continue
        row = roll[row_of[n.pitch]]
        on = int(round(n.start / frame_s))
        later = [s for s in starts[n.pitch] if s > on]
        limit = min(later) if later else T
        if on >= limit:
            refined.append(n)
            continue
        p = on + int(np.argmax(row[on : min(on + peak_window, limit)]))
        below = np.nonzero(row[p:limit] < off_level)[0]
        off = p + below[0] if below.size else limit
        if off > on:
            refined.append(NoteEvent(n.pitch, n.start, off * frame_s, n.velocity))
        else:
            refined.append(n)
    return sorted(refined, key=lambda n: (n.start, n.pitch))


def _note(pitch: int, on: int, off: int, peak: float, frame_s: float) -> NoteEvent:
    velocity = int(np.clip(round(30 + 97 * np.sqrt(peak)), 1, 127))
    return NoteEvent(int(pitch), on * frame_s, off * frame_s, velocity)


def write_midi(notes: list[NoteEvent], path: str | Path) -> Path:
    """Write notes to a single-track piano MIDI file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    piano.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in notes]
    pm.instruments.append(piano)
    pm.write(str(path))
    return path


def save_activations(
    path: str | Path,
    H: np.ndarray,
    roll: np.ndarray,
    pitches: list[int],
    groups: list[list[int]],
    sr: int,
    hop_length: int,
    notes: list[NoteEvent],
) -> Path:
    """Save H, the pitch roll and everything needed to interpret them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        H=H,
        pitch_roll=roll,
        pitches=np.asarray(pitches, dtype=np.int64),
        atom_pitch=np.asarray(
            [p for p, idx in zip(pitches, groups, strict=True) for _ in idx],
            dtype=np.int64,
        ),
        times=frames_to_seconds(np.arange(H.shape[1]), sr, hop_length),
        sr=sr,
        hop_length=hop_length,
        notes=json.dumps([asdict(n) for n in notes]),
    )
    return path
