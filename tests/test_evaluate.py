"""tests/test_evaluate.py — evaluation metrics tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pretty_midi
import pytest

from nmf_core.evaluate import (
    EvalResult,
    FrameScores,
    NoteScores,
    compare_methods,
    evaluate_midi,
    print_report,
    save_report_json,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_midi(path: Path, notes: list[tuple[int, float, float]]) -> Path:
    """Create a MIDI file with given (pitch, start, end) notes."""
    pm    = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        piano.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(piano)
    pm.write(str(path))
    return path


@pytest.fixture
def perfect_ref(tmp_path) -> Path:
    """Reference: C4 E4 G4 each 1 second."""
    return _make_midi(tmp_path / "ref.mid", [
        (60, 0.0, 1.0),
        (64, 1.0, 2.0),
        (67, 2.0, 3.0),
    ])


@pytest.fixture
def perfect_pred(tmp_path, perfect_ref) -> Path:
    """Predicted identical to reference -> scores should be ~1.0."""
    return _make_midi(tmp_path / "pred_perfect.mid", [
        (60, 0.0, 1.0),
        (64, 1.0, 2.0),
        (67, 2.0, 3.0),
    ])


@pytest.fixture
def shifted_pred(tmp_path) -> Path:
    """Predicted with 30 ms onset shift — within 50 ms tolerance."""
    return _make_midi(tmp_path / "pred_shifted.mid", [
        (60, 0.03, 1.03),
        (64, 1.03, 2.03),
        (67, 2.03, 3.03),
    ])


@pytest.fixture
def wrong_pred(tmp_path) -> Path:
    """Completely wrong pitches — scores should be ~0."""
    return _make_midi(tmp_path / "pred_wrong.mid", [
        (40, 0.0, 1.0),
        (41, 1.0, 2.0),
        (42, 2.0, 3.0),
    ])


@pytest.fixture
def empty_pred(tmp_path) -> Path:
    """Empty prediction file."""
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(program=0))
    pm.write(str(tmp_path / "empty.mid"))
    return tmp_path / "empty.mid"


# ── EvalResult structure tests ─────────────────────────────────────────────────

class TestEvalResult:
    def test_returns_eval_result(self, perfect_pred, perfect_ref):
        r = evaluate_midi(perfect_pred, perfect_ref)
        assert isinstance(r, EvalResult)

    def test_has_frame_scores(self, perfect_pred, perfect_ref):
        r = evaluate_midi(perfect_pred, perfect_ref)
        assert isinstance(r.frame, FrameScores)

    def test_has_note_onset_scores(self, perfect_pred, perfect_ref):
        r = evaluate_midi(perfect_pred, perfect_ref)
        assert isinstance(r.note_onset, NoteScores)

    def test_has_note_offset_scores(self, perfect_pred, perfect_ref):
        r = evaluate_midi(perfect_pred, perfect_ref)
        assert isinstance(r.note_offset, NoteScores)

    def test_note_counts(self, perfect_pred, perfect_ref):
        r = evaluate_midi(perfect_pred, perfect_ref)
        assert r.n_ref_notes  == 3
        assert r.n_pred_notes == 3

    def test_scores_in_0_1(self, perfect_pred, perfect_ref):
        r = evaluate_midi(perfect_pred, perfect_ref)
        for val in [r.frame.precision, r.frame.recall, r.frame.f_measure,
                    r.note_onset.precision, r.note_onset.recall,
                    r.note_onset.f_measure]:
            assert 0.0 <= val <= 1.0


# ── Perfect match ──────────────────────────────────────────────────────────────

class TestPerfectMatch:
    def test_frame_f_high(self, perfect_pred, perfect_ref):
        r = evaluate_midi(perfect_pred, perfect_ref)
        assert r.frame.f_measure > 0.95

    def test_note_onset_f_perfect(self, perfect_pred, perfect_ref):
        r = evaluate_midi(perfect_pred, perfect_ref)
        assert r.note_onset.f_measure > 0.95

    def test_note_offset_f_high(self, perfect_pred, perfect_ref):
        r = evaluate_midi(perfect_pred, perfect_ref)
        assert r.note_offset.f_measure > 0.90


# ── Tolerance tests ────────────────────────────────────────────────────────────

class TestTolerance:
    def test_30ms_shift_still_detected(self, shifted_pred, perfect_ref):
        """30 ms shift is within the 50 ms MIREX tolerance."""
        r = evaluate_midi(shifted_pred, perfect_ref)
        assert r.note_onset.f_measure > 0.80

    def test_wrong_pitches_low_f(self, wrong_pred, perfect_ref):
        """Completely wrong pitches should give near-zero F-measure."""
        r = evaluate_midi(wrong_pred, perfect_ref)
        assert r.note_onset.f_measure < 0.1

    def test_empty_prediction_zero_f(self, empty_pred, perfect_ref):
        r = evaluate_midi(empty_pred, perfect_ref)
        assert r.note_onset.f_measure == 0.0
        assert r.frame.f_measure      == 0.0


# ── compare_methods ────────────────────────────────────────────────────────────

class TestCompareMethods:
    def test_returns_dict(self, perfect_pred, wrong_pred, perfect_ref):
        result = compare_methods(perfect_pred, wrong_pred, perfect_ref)
        assert isinstance(result, dict)

    def test_has_standard_and_gs_keys(self, perfect_pred, wrong_pred,
                                       perfect_ref):
        result = compare_methods(perfect_pred, wrong_pred, perfect_ref)
        assert "standard"     in result
        assert "group_sparse" in result

    def test_has_winner_key(self, perfect_pred, wrong_pred, perfect_ref):
        result = compare_methods(perfect_pred, wrong_pred, perfect_ref)
        assert "winner" in result
        assert "frame_f"      in result["winner"]
        assert "note_onset_f" in result["winner"]

    def test_winner_is_better_method(self, perfect_pred, wrong_pred,
                                      perfect_ref):
        # perfect_pred is 'standard', wrong_pred is 'group_sparse'
        result  = compare_methods(perfect_pred, wrong_pred, perfect_ref)
        winner  = result["winner"]["note_onset_f"]
        assert winner == "standard"


# ── Serialization ──────────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_keys(self, perfect_pred, perfect_ref):
        r = evaluate_midi(perfect_pred, perfect_ref)
        d = r.to_dict()
        assert "frame"       in d
        assert "note_onset"  in d
        assert "note_offset" in d
        assert "n_ref_notes" in d

    def test_save_report_json(self, tmp_path, perfect_pred, perfect_ref):
        r    = evaluate_midi(perfect_pred, perfect_ref)
        path = tmp_path / "eval.json"
        save_report_json(r, path)
        assert path.exists()

    def test_saved_json_valid(self, tmp_path, perfect_pred, perfect_ref):
        r    = evaluate_midi(perfect_pred, perfect_ref)
        path = tmp_path / "eval.json"
        save_report_json(r, path)
        with open(path) as f:
            data = json.load(f)
        assert "frame"      in data
        assert "note_onset" in data

    def test_print_report_runs(self, perfect_pred, perfect_ref, capsys):
        r = evaluate_midi(perfect_pred, perfect_ref)
        print_report(r, label="test")
        out = capsys.readouterr().out
        assert "Precision" in out or "P/R/F" in out or "Evaluation" in out


# ── CLI integration ────────────────────────────────────────────────────────────

class TestEvaluateCLI:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "nmf_cli", "evaluate", *args],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )

    def test_exits_zero(self, perfect_pred, perfect_ref):
        r = self._run("--predicted", str(perfect_pred),
                      "--reference", str(perfect_ref))
        assert r.returncode == 0, r.stderr

    def test_missing_predicted_exits_nonzero(self, tmp_path, perfect_ref):
        r = self._run("--predicted", str(tmp_path / "missing.mid"),
                      "--reference", str(perfect_ref))
        assert r.returncode != 0

    def test_saves_json(self, tmp_path, perfect_pred, perfect_ref):
        out = tmp_path / "scores.json"
        self._run("--predicted", str(perfect_pred),
                  "--reference", str(perfect_ref),
                  "--out-json",  str(out))
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert "frame" in data

    def test_compare_mode(self, tmp_path, perfect_pred, wrong_pred,
                          perfect_ref):
        out = tmp_path / "compare.json"
        r   = self._run("--predicted",   str(wrong_pred),
                        "--compare-std", str(perfect_pred),
                        "--reference",   str(perfect_ref),
                        "--out-json",    str(out))
        assert r.returncode == 0, r.stderr
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert "standard"     in data
        assert "group_sparse" in data
        assert "winner"       in data
