"""Generate demo audio so the pipeline can be tried without a dataset.

Writes into OUT_DIR (default: demo/):
  single_notes/MAPS_ISOL_NO_F_S0_M<midi>_synth.wav  one file per piano key
  piece.wav                                          a short polyphonic piece
  piece_reference.mid                                the notes in piece.wav

Usage:  python scripts/make_demo_audio.py [OUT_DIR]
"""

import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

SR = 22050
PIECE = [  # (midi pitch, onset s, duration s)
    (60, 0.0, 0.9), (64, 0.0, 0.9), (67, 0.0, 0.9),
    (62, 1.0, 0.5), (65, 1.0, 0.5), (69, 1.0, 0.5),
    (48, 1.5, 1.0), (55, 1.5, 1.0), (72, 1.5, 0.5),
    (76, 2.5, 1.0), (60, 2.5, 0.45), (60, 3.0, 0.45),
]  # fmt: skip


def piano_note(midi: int, dur: float) -> np.ndarray:
    """Additive piano-like tone: stretched partials, faster decay up high."""
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for h in range(1, 16):
        fh = h * f0 * np.sqrt(1 + 1e-4 * h * h)
        if fh > SR / 2 - 500:
            break
        y += (
            np.exp(-0.3 * (h - 1))
            / h
            * np.exp(-(1.5 + 0.6 * h) * t)
            * np.sin(2 * np.pi * fh * t + h)
        )
    return y * np.minimum(1, t / 0.005) * np.clip((dur - t) / 0.05, 0, 1)


def main(out: Path) -> None:
    notes = out / "single_notes"
    notes.mkdir(parents=True, exist_ok=True)
    for m in range(21, 109):
        y = piano_note(m, 1.0)
        sf.write(
            notes / f"MAPS_ISOL_NO_F_S0_M{m}_synth.wav", 0.5 * y / abs(y).max(), SR
        )

    y = np.zeros(4 * SR)
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="Piano")
    for midi, start, dur in PIECE:
        n = piano_note(midi, dur)
        i = int(start * SR)
        y[i : i + len(n)] += n[: len(y) - i]
        piano.notes.append(pretty_midi.Note(100, midi, start, start + dur))
    sf.write(out / "piece.wav", 0.8 * y / abs(y).max(), SR)
    pm.instruments.append(piano)
    pm.write(str(out / "piece_reference.mid"))
    print(f"Wrote 88 single notes to {notes}")
    print(f"Wrote {out / 'piece.wav'} and {out / 'piece_reference.mid'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "demo"))
