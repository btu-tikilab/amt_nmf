"""Note-level precision / recall / F1 of transcribed MIDI against a reference.

Uses mir_eval's standard note-matching rules (MIREX):
  * onset-only:      same pitch (within 50 cents) and onset within
                     --onset-tolerance seconds;
  * onset + offset:  additionally the offset within
                     max(--offset-min, --offset-ratio x reference duration).

Needs mir_eval, installed by ``pip install -e ".[dev]"``.

Usage:
  python scripts/evaluate_midi.py REFERENCE.mid PREDICTED.mid [PREDICTED.mid ...]
      [--onset-tolerance 0.05] [--offset-ratio 0.2] [--offset-min 0.05]
      [--json scores.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi


def load_notes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(intervals in s, pitches in Hz) of all non-drum notes in a MIDI file."""
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    intervals = np.array([[n.start, n.end] for n in notes]).reshape(-1, 2)
    pitches = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes])
    return intervals, pitches


def score(
    reference: Path,
    predicted: Path,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    offset_min: float = 0.05,
) -> dict:
    from mir_eval.transcription import precision_recall_f1_overlap

    ref_i, ref_p = load_notes(reference)
    est_i, est_p = load_notes(predicted)
    result = {
        "file": str(predicted),
        "n_reference": len(ref_p),
        "n_predicted": len(est_p),
    }
    for key, ratio in (("onset", None), ("onset_offset", offset_ratio)):
        if len(ref_p) == 0 or len(est_p) == 0:
            p = r = f = 0.0
        else:
            p, r, f, _ = precision_recall_f1_overlap(
                ref_i,
                ref_p,
                est_i,
                est_p,
                onset_tolerance=onset_tolerance,
                offset_ratio=ratio,
                offset_min_tolerance=offset_min,
            )
        result[key] = {"precision": p, "recall": r, "f1": f}
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reference", type=Path, help="reference (ground-truth) .mid")
    ap.add_argument("predicted", type=Path, nargs="+", help="transcribed .mid file(s)")
    ap.add_argument("--onset-tolerance", type=float, default=0.05, help="seconds")
    ap.add_argument(
        "--offset-ratio",
        type=float,
        default=0.2,
        help="fraction of the reference note duration",
    )
    ap.add_argument(
        "--offset-min",
        type=float,
        default=0.05,
        help="minimum offset tolerance in seconds",
    )
    ap.add_argument("--json", type=Path, help="also write the scores to this file")
    a = ap.parse_args(argv)

    try:
        import mir_eval  # noqa: F401
    except ImportError:
        print(
            'Error: mir_eval is not installed. Run: pip install -e ".[dev]"',
            file=sys.stderr,
        )
        return 1
    for path in (a.reference, *a.predicted):
        if not path.is_file():
            print(f"Error: MIDI file not found: {path}", file=sys.stderr)
            return 1
    try:
        results = [
            score(a.reference, p, a.onset_tolerance, a.offset_ratio, a.offset_min)
            for p in a.predicted
        ]
    except Exception as exc:  # corrupt MIDI etc.
        print(
            f"Error: cannot evaluate: {exc.__class__.__name__}: {exc}", file=sys.stderr
        )
        return 1

    print(f"Reference: {a.reference} ({results[0]['n_reference']} notes)")
    print(
        f"Onset tolerance {a.onset_tolerance * 1000:.0f} ms; offset tolerance "
        f"max({a.offset_min * 1000:.0f} ms, {a.offset_ratio:.0%} of duration)\n"
    )

    def fmt(m: dict) -> str:
        return f"{m['precision']:.2f} / {m['recall']:.2f} / {m['f1']:.2f}"

    head = (
        f"{'file':<32} {'notes':>5}   {'onset P / R / F1':<18}   "
        "onset+offset P / R / F1"
    )
    print(head)
    print("-" * len(head))
    for r in results:
        print(
            f"{Path(r['file']).name:<32} {r['n_predicted']:>5}   "
            f"{fmt(r['onset']):<18}   {fmt(r['onset_offset'])}"
        )
    if a.json:
        settings = {
            "onset_tolerance": a.onset_tolerance,
            "offset_ratio": a.offset_ratio,
            "offset_min": a.offset_min,
        }
        a.json.write_text(
            json.dumps(
                {
                    "reference": str(a.reference),
                    "settings": settings,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nScores written to {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
