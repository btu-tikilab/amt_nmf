"""evaluate, report and sheet commands, run through the terminal."""

import json

import pretty_midi
import pytest

from conftest import run_cli

REF = [(60, 0.0, 1.0), (64, 0.0, 1.0), (67, 1.0, 0.5), (72, 2.0, 0.5)]


def _midi(path, notes):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(100, p, s, s + d) for p, s, d in notes]
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


@pytest.fixture(scope="module")
def run(built_dict, piece_wav, tmp_path_factory):
    out = tmp_path_factory.mktemp("extras")
    r = run_cli(
        "transcribe", "--input", piece_wav, "--dict", built_dict, "--out-dir", out
    )
    assert r.returncode == 0, r.stderr
    return out


def test_evaluate_scores_and_json(tmp_path):
    ref = _midi(tmp_path / "ref.mid", REF)
    exact = _midi(tmp_path / "exact.mid", REF)
    # One note ends far too early, plus one wrong pitch.
    short = _midi(
        tmp_path / "short.mid",
        [(60, 0.01, 0.5), (64, 0.0, 1.0), (67, 1.0, 0.5), (72, 2.0, 0.5),
         (50, 3.0, 0.5)],
    )  # fmt: skip
    out = tmp_path / "scores.json"
    r = run_cli(
        "evaluate", "--reference", ref, "--predicted", short,
        "--compare-std", exact, "--out-json", out,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "onset+offset" in r.stdout and "frame F1" in r.stdout
    exact_s, short_s = json.loads(out.read_text())["results"]
    assert exact_s["onset"]["f1"] == exact_s["onset_offset"]["f1"] == 1.0
    assert exact_s["frame"]["f1"] == pytest.approx(1.0)
    assert short_s["onset"]["precision"] == pytest.approx(4 / 5)
    assert short_s["onset_offset"]["recall"] == pytest.approx(3 / 4)
    assert short_s["frame"]["precision"] < 1.0


def test_evaluate_missing_file(tmp_path):
    ref = _midi(tmp_path / "ref.mid", REF)
    r = run_cli("evaluate", "--reference", ref, "--predicted", tmp_path / "no.mid")
    assert r.returncode == 1
    assert r.stderr.startswith("Error:") and "not found" in r.stderr


def test_report_is_self_contained_html(run, tmp_path):
    ref = _midi(tmp_path / "ref.mid", REF)
    scores = tmp_path / "scores.json"
    std, gs = run / "piece_standard.mid", run / "piece_group_sparse.mid"
    r = run_cli(
        "evaluate", "--reference", ref, "--predicted", gs, "--compare-std", std,
        "--out-json", scores,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    out = tmp_path / "report.html"
    r = run_cli(
        "report", "--std-mid", std, "--gs-mid", gs, "--summary",
        run / "summary.json", "--eval-json", scores, "--reference", ref,
        "--out", out,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    page = out.read_text(encoding="utf-8")
    assert page.count("data:image/png;base64,") == 2  # piano rolls, convergence
    assert "Scores against the reference" in page
    assert "piece_group_sparse.mid" in page
    assert "http://" not in page and "https://" not in page  # opens offline


def test_report_missing_summary(run, tmp_path):
    r = run_cli(
        "report", "--std-mid", run / "piece_standard.mid", "--gs-mid",
        run / "piece_group_sparse.mid", "--summary", tmp_path / "none.json",
    )  # fmt: skip
    assert r.returncode == 1 and "summary.json not found" in r.stderr


def test_sheet_writes_musicxml(tmp_path):
    pytest.importorskip("music21")
    src = _midi(tmp_path / "in.mid", REF)
    out = tmp_path / "score.xml"
    r = run_cli("sheet", "--input", src, "--out", out, "--title", "Test")
    assert r.returncode == 0, r.stderr
    xml = out.read_text(encoding="utf-8")
    assert "<score-partwise" in xml and "Test" in xml
    assert xml.count("<note") >= len(REF)


def test_non_ascii_paths_do_not_crash_the_console(built_dict, piece_wav, tmp_path):
    """Printing a character cp1252 lacks must not crash (the Windows bug)."""
    out = tmp_path / "out_→_♫"
    r = run_cli(
        "transcribe", "--input", piece_wav, "--dict", built_dict,
        "--method", "group-sparse", "--out-dir", out,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert (out / "piece_group_sparse.mid").is_file()
