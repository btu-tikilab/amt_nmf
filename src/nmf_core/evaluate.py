"""Scoring transcribed MIDI against a reference MIDI with mir_eval.

Note-level matching follows the MIREX rules:
  * onset:          same pitch (within 50 cents) and onset within
                    ``onset_tolerance`` seconds;
  * onset + offset: additionally the offset within
                    max(``offset_min``, ``offset_ratio`` x reference duration).
Frame-level scores compare binary 10 ms piano rolls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from .errors import TranscriberError

FRAME_RATE = 100.0  # frames per second for the frame-level scores


def load_notes(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(intervals in s, pitches in Hz) of all non-drum notes in a MIDI file."""
    path = Path(path)
    if not path.is_file():
        raise TranscriberError(f"MIDI file not found: {path}")
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        raise TranscriberError(f"Cannot read MIDI file {path}: {exc}") from exc
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    intervals = np.array([[n.start, n.end] for n in notes]).reshape(-1, 2)
    pitches = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes])
    return intervals, pitches


def _roll(intervals: np.ndarray, pitches_hz: np.ndarray, n_frames: int) -> np.ndarray:
    roll = np.zeros((128, n_frames), dtype=bool)
    midi = np.round(pretty_midi.hz_to_note_number(pitches_hz)).astype(int)
    for (on, off), m in zip(intervals, midi, strict=True):
        a, b = int(round(on * FRAME_RATE)), int(round(off * FRAME_RATE))
        roll[m, a : max(b, a + 1)] = True
    return roll


def frame_scores(
    ref: tuple[np.ndarray, np.ndarray], est: tuple[np.ndarray, np.ndarray]
) -> dict:
    """Frame-level precision / recall / F1 / accuracy of two note lists."""
    end = max(
        float(ref[0][:, 1].max(initial=0.0)), float(est[0][:, 1].max(initial=0.0))
    )
    n = int(np.ceil(end * FRAME_RATE)) + 1
    r, e = _roll(*ref, n), _roll(*est, n)
    tp = float(np.sum(r & e))
    fp = float(np.sum(~r & e))
    fn = float(np.sum(r & ~e))
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * rec / (p + rec) if p + rec else 0.0
    acc = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": p, "recall": rec, "f1": f, "accuracy": acc}


def score(
    reference: str | Path,
    predicted: str | Path,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    offset_min: float = 0.05,
) -> dict:
    """Note- and frame-level scores of one predicted MIDI file."""
    try:
        from mir_eval.transcription import precision_recall_f1_overlap
    except ImportError as exc:
        raise TranscriberError(
            "mir_eval is not installed. Run: pip install -e ."
        ) from exc

    ref = load_notes(reference)
    est = load_notes(predicted)
    result = {
        "file": str(predicted),
        "n_reference": len(ref[1]),
        "n_predicted": len(est[1]),
    }
    for key, ratio in (("onset", None), ("onset_offset", offset_ratio)):
        if len(ref[1]) == 0 or len(est[1]) == 0:
            p = r = f = 0.0
        else:
            p, r, f, _ = precision_recall_f1_overlap(
                ref[0],
                ref[1],
                est[0],
                est[1],
                onset_tolerance=onset_tolerance,
                offset_ratio=ratio,
                offset_min_tolerance=offset_min,
            )
        result[key] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    result["frame"] = frame_scores(ref, est)
    return result
