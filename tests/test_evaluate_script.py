"""scripts/evaluate_midi.py: note-level P/R/F1 via mir_eval."""

import json
import subprocess
import sys
from pathlib import Path

import pretty_midi
import pytest

pytest.importorskip("mir_eval")
SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_midi.py"
REF = [(60, 0.0, 1.0), (64, 0.0, 1.0), (67, 1.0, 0.5), (72, 2.0, 0.5)]


def _midi(path, notes):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(100, p, s, s + d) for p, s, d in notes]
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


def test_scores(tmp_path):
    ref = _midi(tmp_path / "ref.mid", REF)
    exact = _midi(tmp_path / "exact.mid", REF)
    # Same onsets, but two notes end far too early, plus one wrong pitch.
    short = _midi(
        tmp_path / "short.mid",
        [(60, 0.01, 0.5), (64, 0.0, 1.0), (67, 1.02, 0.2), (72, 2.0, 0.5),
         (50, 3.0, 0.5)],
    )  # fmt: skip
    out = tmp_path / "scores.json"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(ref), str(exact), str(short),
         "--json", str(out)],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "onset+offset" in r.stdout
    exact_s, short_s = json.loads(out.read_text())["results"]
    assert exact_s["onset"]["f1"] == exact_s["onset_offset"]["f1"] == 1.0
    assert short_s["onset"]["recall"] == 1.0
    assert short_s["onset"]["precision"] == pytest.approx(4 / 5)
    assert short_s["onset_offset"]["recall"] == pytest.approx(2 / 4)
    assert short_s["onset_offset"]["precision"] == pytest.approx(2 / 5)


def test_missing_file(tmp_path):
    ref = _midi(tmp_path / "ref.mid", REF)
    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(ref), str(tmp_path / "nope.mid")],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 1 and "not found" in r.stderr
