"""tests/test_report_sheet.py — HTML report and sheet music export tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pretty_midi
import pytest

# ── shared fixtures ───────────────────────────────────────────────────────────

def _make_midi(path: Path, notes: list) -> Path:
    pm    = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        piano.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch,
                             start=start, end=end)
        )
    pm.instruments.append(piano)
    pm.write(str(path))
    return path


def _make_summary(path: Path) -> Path:
    summary = {
        "method": "compare",
        "audio_input": "test.wav",
        "output_dir": str(path.parent),
        "threshold": 0.15,
        "sparsity_weight": 0.5,
        "results": {
            "standard": {
                "converged": True,
                "n_iter": 100,
                "final_error": 5.2,
                "error_reduction_pct": 90.1,
                "elapsed_s": 2.3,
                "K": 8,
            },
            "group_sparse": {
                "converged": True,
                "n_iter": 60,
                "final_cost": 3.1,
                "elapsed_s": 4.1,
                "beta": 0.5,
                "lam": 0.5,
                "q": 0.5,
                "mean_active_per_frame": 0.75,
                "sparsity_pct": 97.8,
            },
        },
    }
    path.write_text(json.dumps(summary, indent=2))
    return path


@pytest.fixture(scope="module")
def std_mid(tmp_path_factory):
    p = tmp_path_factory.mktemp("midi") / "std.mid"
    return _make_midi(p, [(60, 0.0, 1.0), (64, 1.0, 2.0), (67, 2.0, 3.0)])


@pytest.fixture(scope="module")
def gs_mid(tmp_path_factory):
    p = tmp_path_factory.mktemp("midi") / "gs.mid"
    return _make_midi(p, [(60, 0.0, 1.0), (67, 2.0, 3.0)])


@pytest.fixture(scope="module")
def summary_json(tmp_path_factory, std_mid):
    p = tmp_path_factory.mktemp("json") / "summary.json"
    return _make_summary(p)


# ── HTML report tests ─────────────────────────────────────────────────────────

class TestReport:
    def test_creates_html(self, tmp_path, std_mid, gs_mid, summary_json):
        from nmf_core.report import generate_report
        out = tmp_path / "report.html"
        generate_report(out, std_mid, gs_mid, summary_json)
        assert out.exists()

    def test_html_nonempty(self, tmp_path, std_mid, gs_mid, summary_json):
        from nmf_core.report import generate_report
        out = tmp_path / "report.html"
        generate_report(out, std_mid, gs_mid, summary_json)
        assert out.stat().st_size > 10_000  # at least 10 KB

    def test_html_contains_piano_roll_image(
        self, tmp_path, std_mid, gs_mid, summary_json
    ):
        from nmf_core.report import generate_report
        out = tmp_path / "report.html"
        generate_report(out, std_mid, gs_mid, summary_json)
        content = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content

    def test_html_contains_sparsity_section(
        self, tmp_path, std_mid, gs_mid, summary_json
    ):
        from nmf_core.report import generate_report
        out = tmp_path / "report.html"
        generate_report(out, std_mid, gs_mid, summary_json)
        content = out.read_text(encoding="utf-8")
        assert "Sparsity" in content

    def test_html_is_valid_markup(
        self, tmp_path, std_mid, gs_mid, summary_json
    ):
        from nmf_core.report import generate_report
        out = tmp_path / "report.html"
        generate_report(out, std_mid, gs_mid, summary_json)
        content = out.read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")
        assert "</html>" in content

    def test_report_with_eval_json(
        self, tmp_path, std_mid, gs_mid, summary_json
    ):
        from nmf_core.evaluate import compare_methods
        from nmf_core.report import generate_report

        # Build a quick eval json
        eval_scores = compare_methods(std_mid, gs_mid, std_mid)
        eval_path   = tmp_path / "eval.json"
        eval_path.write_text(json.dumps(eval_scores))

        out = tmp_path / "report_eval.html"
        generate_report(out, std_mid, gs_mid, summary_json,
                        eval_json=eval_path)
        content = out.read_text(encoding="utf-8")
        assert "Evaluation Scores" in content


# ── sheet music tests ─────────────────────────────────────────────────────────

class TestSheetMusic:
    def test_creates_xml(self, tmp_path, std_mid):
        from nmf_core.sheet import midi_to_musicxml
        out = tmp_path / "score.xml"
        midi_to_musicxml(std_mid, out, title="Test Score")
        assert out.exists()

    def test_xml_nonempty(self, tmp_path, std_mid):
        from nmf_core.sheet import midi_to_musicxml
        out = tmp_path / "score.xml"
        midi_to_musicxml(std_mid, out)
        assert out.stat().st_size > 500

    def test_xml_is_valid_musicxml(self, tmp_path, std_mid):
        from nmf_core.sheet import midi_to_musicxml
        out = tmp_path / "score.xml"
        midi_to_musicxml(std_mid, out)
        content = out.read_text(encoding="utf-8")
        assert "score-partwise" in content or "MusicXML" in content or \
               "measure" in content

    def test_title_embedded(self, tmp_path, std_mid):
        from nmf_core.sheet import midi_to_musicxml
        out = tmp_path / "score.xml"
        midi_to_musicxml(std_mid, out, title="My NMF Score")
        content = out.read_text(encoding="utf-8")
        assert "My NMF Score" in content

    def test_missing_midi_raises(self, tmp_path):
        from nmf_core.sheet import midi_to_musicxml
        with pytest.raises(FileNotFoundError):
            midi_to_musicxml(tmp_path / "missing.mid",
                             tmp_path / "out.xml")


# ── CLI integration tests ─────────────────────────────────────────────────────

class TestReportCLI:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "nmf_cli", *args],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )

    def test_report_exits_zero(
        self, tmp_path, std_mid, gs_mid, summary_json
    ):
        out = tmp_path / "report.html"
        r   = self._run(
            "report",
            "--std-mid", str(std_mid),
            "--gs-mid",  str(gs_mid),
            "--summary", str(summary_json),
            "--out",     str(out),
        )
        assert r.returncode == 0, r.stderr

    def test_report_creates_html(
        self, tmp_path, std_mid, gs_mid, summary_json
    ):
        out = tmp_path / "report.html"
        self._run(
            "report",
            "--std-mid", str(std_mid),
            "--gs-mid",  str(gs_mid),
            "--summary", str(summary_json),
            "--out",     str(out),
        )
        assert out.exists()

    def test_sheet_exits_zero(self, tmp_path, std_mid):
        out = tmp_path / "score.xml"
        r   = self._run(
            "sheet",
            "--input", str(std_mid),
            "--out",   str(out),
        )
        assert r.returncode == 0, r.stderr

    def test_sheet_creates_xml(self, tmp_path, std_mid):
        out = tmp_path / "score.xml"
        self._run("sheet", "--input", str(std_mid), "--out", str(out))
        assert out.exists()

    def test_sheet_missing_input(self, tmp_path):
        r = self._run(
            "sheet",
            "--input", str(tmp_path / "missing.mid"),
            "--out",   str(tmp_path / "score.xml"),
        )
        assert r.returncode != 0
