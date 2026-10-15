"""End-to-end: real terminal commands through subprocess."""

import json

import numpy as np
import pretty_midi
import pytest

from conftest import PITCHES, run_cli


@pytest.fixture(scope="module")
def cli_dict(notes_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("cli") / "piano_dict.npy"
    r = run_cli(
        "build-dict",
        "--audio-dir",
        notes_dir,
        "--output",
        out,
        "--pitch-range",
        min(PITCHES),
        max(PITCHES),
    )
    assert r.returncode == 0, r.stderr
    return out


def test_build_dict_creates_valid_dictionary(cli_dict):
    W = np.load(cli_dict)
    meta = json.loads(cli_dict.with_suffix(".json").read_text())
    assert W.shape == (288, 2 * len(PITCHES))
    assert meta["pitch_range"] == [min(PITCHES), max(PITCHES)]
    assert meta["source"] == "supervised"


def test_transcribe_compare_end_to_end(cli_dict, piece_wav, tmp_path):
    r = run_cli(
        "transcribe",
        "--input",
        piece_wav,
        "--dict",
        cli_dict,
        "--method",
        "compare",
        "--sparsity-weight",
        "0.2",
        "--threshold",
        "0.1",
        "--out-dir",
        tmp_path,
    )
    assert r.returncode == 0, r.stderr
    for name in ("piece_standard.mid", "piece_group_sparse.mid"):
        pm = pretty_midi.PrettyMIDI(str(tmp_path / name))
        notes = pm.instruments[0].notes
        assert len(notes) > 0
        assert {60, 64, 67} <= {n.pitch for n in notes}
        assert 3.0 < pm.get_end_time() < 4.5
    for name in ("piece_standard_H.npz", "piece_group_sparse_H.npz"):
        assert "H" in np.load(tmp_path / name)
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["config"]["method"] == "compare"
    assert s["processing_time_s"] > 0
    for key in ("standard", "group_sparse"):
        assert {"converged", "iterations", "solve_time_s"} <= set(s["results"][key])


def test_harmonic_build_dict(tmp_path):
    r = run_cli("build-dict", "--harmonic", "--output", tmp_path / "h.npy")
    assert r.returncode == 0, r.stderr
    assert np.load(tmp_path / "h.npy").shape == (288, 176)


def _assert_clean_error(r, text):
    assert r.returncode == 1
    assert text in r.stderr
    assert "Traceback" not in r.stderr + r.stdout


def test_missing_input_audio(cli_dict, tmp_path):
    r = run_cli(
        "transcribe",
        "--input",
        tmp_path / "nope.wav",
        "--dict",
        cli_dict,
        "--out-dir",
        tmp_path,
    )
    _assert_clean_error(r, "Audio file not found")


def test_unreadable_input_audio(cli_dict, tmp_path):
    bad = tmp_path / "bad.flac"
    bad.write_bytes(b"\x00" * 1000)
    r = run_cli("transcribe", "--input", bad, "--dict", cli_dict, "--out-dir", tmp_path)
    _assert_clean_error(r, "Cannot read audio file")


def test_missing_dictionary(piece_wav, tmp_path):
    r = run_cli(
        "transcribe",
        "--input",
        piece_wav,
        "--dict",
        tmp_path / "x.npy",
        "--out-dir",
        tmp_path,
    )
    _assert_clean_error(r, "Dictionary file not found")


def test_corrupted_dictionary(cli_dict, piece_wav, tmp_path):
    bad = tmp_path / "bad.npy"
    bad.write_bytes(b"garbage")
    bad.with_suffix(".json").write_text(cli_dict.with_suffix(".json").read_text())
    r = run_cli(
        "transcribe", "--input", piece_wav, "--dict", bad, "--out-dir", tmp_path
    )
    _assert_clean_error(r, "corrupted")


def test_dimension_mismatch(cli_dict, piece_wav, tmp_path):
    bad = tmp_path / "short.npy"
    np.save(bad, np.load(cli_dict)[:100])
    bad.with_suffix(".json").write_text(cli_dict.with_suffix(".json").read_text())
    r = run_cli(
        "transcribe", "--input", piece_wav, "--dict", bad, "--out-dir", tmp_path
    )
    _assert_clean_error(r, "Dimension mismatch")


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--sparsity-weight", "-1"), ("--threshold", "-0.5"), ("--threshold", "1.5")],
)
def test_invalid_hyperparameters(cli_dict, piece_wav, tmp_path, flag, value):
    r = run_cli(
        "transcribe",
        "--input",
        piece_wav,
        "--dict",
        cli_dict,
        "--out-dir",
        tmp_path,
        flag,
        value,
    )
    _assert_clean_error(r, flag)


def test_invalid_method_and_pitch_range(cli_dict, piece_wav, tmp_path):
    r = run_cli(
        "transcribe", "--input", piece_wav, "--dict", cli_dict, "--method", "magic"
    )
    assert r.returncode == 2 and "invalid choice" in r.stderr
    r = run_cli(
        "build-dict",
        "--harmonic",
        "--pitch-range",
        "90",
        "20",
        "--output",
        tmp_path / "d.npy",
    )
    _assert_clean_error(r, "Invalid pitch range")


def test_build_dict_errors(tmp_path):
    r = run_cli(
        "build-dict", "--audio-dir", tmp_path / "none", "--output", tmp_path / "d.npy"
    )
    _assert_clean_error(r, "Audio directory not found")
    (tmp_path / "empty").mkdir()
    r = run_cli(
        "build-dict", "--audio-dir", tmp_path / "empty", "--output", tmp_path / "d.npy"
    )
    _assert_clean_error(r, "No audio files")
    r = run_cli("build-dict", "--output", tmp_path / "d.npy")
    _assert_clean_error(r, "--audio-dir or --harmonic")
