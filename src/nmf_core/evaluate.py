"""
src/nmf_core/evaluate.py
─────────────────────────────────────────────────────────────────
Evaluation metrics for polyphonic piano transcription.

Compares a predicted MIDI file against a reference (ground-truth)
MIDI using standard AMT metrics from the mir_eval library:

  Frame-level  — precision, recall, F-measure per time frame
  Note-level   — onset precision, recall, F-measure
  Note+offset  — stricter version (onset + offset both must match)

These are the same metrics used in the Music Information Retrieval
Evaluation eXchange (MIREX) and in the O'Hanlon & Plumbley (2014)
paper to compare AMT systems.

Usage
─────
  from nmf_core.evaluate import evaluate_midi, print_report
  scores = evaluate_midi("predicted.mid", "reference.mid")
  print_report(scores)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

# ── Score containers ──────────────────────────────────────────────────────────

@dataclass
class FrameScores:
    """Frame-level binary piano-roll metrics (10ms frames)."""
    precision:  float
    recall:     float
    f_measure:  float
    accuracy:   float    # TP / (TP + FP + FN)


@dataclass
class NoteScores:
    """Note-level onset metrics (±50 ms tolerance)."""
    precision: float
    recall:    float
    f_measure: float


@dataclass
class EvalResult:
    """Complete evaluation result for one predicted MIDI."""
    predicted_path: str
    reference_path: str
    frame:          FrameScores
    note_onset:     NoteScores
    note_offset:    NoteScores     # onset + offset must both match
    n_ref_notes:    int
    n_pred_notes:   int

    def to_dict(self) -> dict:
        return {
            "predicted":    self.predicted_path,
            "reference":    self.reference_path,
            "frame":        asdict(self.frame),
            "note_onset":   asdict(self.note_onset),
            "note_offset":  asdict(self.note_offset),
            "n_ref_notes":  self.n_ref_notes,
            "n_pred_notes": self.n_pred_notes,
        }


# ── MIDI helpers ──────────────────────────────────────────────────────────────

def _load_intervals_and_pitches(
    midi_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a MIDI file and return (intervals, pitches) arrays for mir_eval.

    intervals : (N, 2) float array — [[onset_s, offset_s], ...]
    pitches   : (N,)   float array — MIDI note numbers as floats
    """
    path = Path(midi_path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    pm    = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments for n in inst.notes]

    if not notes:
        return np.zeros((0, 2)), np.zeros(0)

    intervals = np.array([[n.start, n.end] for n in notes], dtype=float)
    pitches   = np.array([n.pitch for n in notes], dtype=float)
    return intervals, pitches


def _piano_roll_from_midi(
    midi_path: str | Path,
    fs: float = 100.0,    # frames per second (10 ms resolution)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert MIDI to a binary piano roll for frame-level evaluation.

    Returns (roll, times) where roll is (128, T) boolean.
    """
    pm   = pretty_midi.PrettyMIDI(str(midi_path))
    roll = pm.get_piano_roll(fs=fs) > 0   # (128, T) bool
    T    = roll.shape[1]
    times = np.arange(T) / fs
    return roll, times


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_midi(
    predicted_path: str | Path,
    reference_path: str | Path,
    onset_tol:   float = 0.05,    # ±50 ms onset tolerance (MIREX standard)
    offset_ratio: float = 0.2,    # offset within 20% of note duration
    frame_fs:    float = 100.0,   # frames per second for frame metrics
) -> EvalResult:
    """
    Evaluate a predicted MIDI against a reference MIDI.

    Parameters
    ----------
    predicted_path : path to predicted .mid file
    reference_path : path to reference (ground-truth) .mid file
    onset_tol      : onset tolerance in seconds (default 50 ms)
    offset_ratio   : offset tolerance as fraction of note duration (default 20%)
    frame_fs       : frame rate for piano-roll metrics (default 100 fps = 10 ms)

    Returns
    -------
    EvalResult with frame-level and note-level metrics
    """
    pred_path = Path(predicted_path)
    ref_path  = Path(reference_path)

    # ── Note-level metrics ────────────────────────────────────────────────────
    ref_int,  ref_pit  = _load_intervals_and_pitches(ref_path)
    pred_int, pred_pit = _load_intervals_and_pitches(pred_path)

    n_ref  = len(ref_pit)
    n_pred = len(pred_pit)

    if n_ref == 0 or n_pred == 0:
        zero_note = NoteScores(0.0, 0.0, 0.0)
        zero_frame = FrameScores(0.0, 0.0, 0.0, 0.0)
        return EvalResult(
            predicted_path = str(pred_path),
            reference_path = str(ref_path),
            frame          = zero_frame,
            note_onset     = zero_note,
            note_offset    = zero_note,
            n_ref_notes    = n_ref,
            n_pred_notes   = n_pred,
        )

    # Onset-only (pitch + onset must match)
    p_on, r_on, f_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pit,
        pred_int, pred_pit,
        onset_tolerance  = onset_tol,
        pitch_tolerance  = 0.25,       # quarter-tone tolerance
        offset_ratio     = None,       # ignore offsets
        offset_min_tolerance = None,
    )

    # Onset + offset (stricter)
    p_off, r_off, f_off, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pit,
        pred_int, pred_pit,
        onset_tolerance      = onset_tol,
        pitch_tolerance      = 0.25,
        offset_ratio         = offset_ratio,
        offset_min_tolerance = 0.05,
    )

    # ── Frame-level metrics ───────────────────────────────────────────────────
    ref_roll,  _ = _piano_roll_from_midi(ref_path,  fs=frame_fs)
    pred_roll, _ = _piano_roll_from_midi(pred_path, fs=frame_fs)

    # Align lengths
    T = min(ref_roll.shape[1], pred_roll.shape[1])
    if T == 0:
        frame_scores = FrameScores(0.0, 0.0, 0.0, 0.0)
    else:
        ref_roll  = ref_roll[:, :T]
        pred_roll = pred_roll[:, :T]

        # mir_eval frame metrics want (T, 88) arrays (pitches 21-108)
        ref_f  = ref_roll[21:109,  :].T.astype(bool)
        pred_f = pred_roll[21:109, :].T.astype(bool)

        p_fr, r_fr, f_fr, a_fr = mir_eval.transcription.frame_accuracy(
            ref_f, pred_f
        ) if hasattr(mir_eval.transcription, "frame_accuracy") else \
            _frame_metrics(ref_f, pred_f)

        frame_scores = FrameScores(
            precision = round(float(p_fr), 4),
            recall    = round(float(r_fr), 4),
            f_measure = round(float(f_fr), 4),
            accuracy  = round(float(a_fr), 4),
        )

    return EvalResult(
        predicted_path = str(pred_path),
        reference_path = str(ref_path),
        frame          = frame_scores,
        note_onset     = NoteScores(round(p_on,4), round(r_on,4), round(f_on,4)),
        note_offset    = NoteScores(round(p_off,4), round(r_off,4), round(f_off,4)),
        n_ref_notes    = n_ref,
        n_pred_notes   = n_pred,
    )


def _frame_metrics(
    ref: np.ndarray, pred: np.ndarray
) -> tuple[float, float, float, float]:
    """Fallback frame metrics if mir_eval version lacks frame_accuracy."""
    tp = float(np.logical_and(ref, pred).sum())
    fp = float(np.logical_and(~ref, pred).sum())
    fn = float(np.logical_and(ref, ~pred).sum())
    eps = 1e-10
    p  = tp / (tp + fp + eps)
    r  = tp / (tp + fn + eps)
    f  = 2*p*r / (p + r + eps)
    a  = tp / (tp + fp + fn + eps)
    return p, r, f, a


# ── Comparison helper ─────────────────────────────────────────────────────────

def compare_methods(
    standard_mid:    str | Path,
    group_sparse_mid: str | Path,
    reference_mid:   str | Path,
) -> dict:
    """
    Evaluate both Standard and Group-Sparse outputs against the same reference.
    Returns a dict suitable for summary.json.
    """
    std_result = evaluate_midi(standard_mid,    reference_mid)
    gs_result  = evaluate_midi(group_sparse_mid, reference_mid)

    return {
        "standard":    std_result.to_dict(),
        "group_sparse": gs_result.to_dict(),
        "winner": {
            "frame_f": (
                "standard"
                if std_result.frame.f_measure >= gs_result.frame.f_measure
                else "group_sparse"
            ),
            "note_onset_f": (
                "standard"
                if std_result.note_onset.f_measure >= gs_result.note_onset.f_measure
                else "group_sparse"
            ),
        },
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_report(result: EvalResult, label: str = "") -> None:
    """Print a formatted evaluation report to stdout."""
    tag = f"  [{label}]" if label else ""
    print(f"\n{'─'*55}")
    print(f"  Evaluation Report{tag}")
    print(f"{'─'*55}")
    print(f"  Predicted : {Path(result.predicted_path).name}")
    print(f"  Reference : {Path(result.reference_path).name}")
    print(
        f"  Notes     : {result.n_pred_notes} predicted"
        f"  /  {result.n_ref_notes} reference"
    )
    print()
    print(f"  {'Metric':<28}  {'P':>6}  {'R':>6}  {'F':>6}")
    print(f"  {'─'*50}")
    print(f"  {'Frame-level':<28}  "
          f"{result.frame.precision:>6.3f}  "
          f"{result.frame.recall:>6.3f}  "
          f"{result.frame.f_measure:>6.3f}")
    print(f"  {'Note (onset only)':<28}  "
          f"{result.note_onset.precision:>6.3f}  "
          f"{result.note_onset.recall:>6.3f}  "
          f"{result.note_onset.f_measure:>6.3f}")
    print(f"  {'Note (onset + offset)':<28}  "
          f"{result.note_offset.precision:>6.3f}  "
          f"{result.note_offset.recall:>6.3f}  "
          f"{result.note_offset.f_measure:>6.3f}")
    print(f"  Frame accuracy: {result.frame.accuracy:.3f}")
    print(f"{'─'*55}\n")


def save_report_json(result: EvalResult, path: str | Path) -> None:
    """Save evaluation result to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"  Evaluation saved → {path}")
