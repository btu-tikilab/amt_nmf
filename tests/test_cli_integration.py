"""tests/test_cli_integration.py — end-to-end CLI integration tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# ── helpers ───────────────────────────────────────────────────────────────────

def _run(*args) -> subprocess.CompletedProcess:
    """Run a CLI command via subprocess with the current Python."""
    return subprocess.run(
        [sys.executable, "-m", "nmf_cli", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )


def _make_wav(path: Path, duration: float = 2.0, sr: int = 22050) -> Path:
    """Create a simple synthetic polyphonic WAV for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # C major chord: C4 (261.63 Hz), E4 (329.63 Hz), G4 (392.00 Hz)
    signal = (
        np.sin(2 * np.pi * 261.63 * t) * np.exp(-t * 1.5)
        + np.sin(2 * np.pi * 329.63 * t) * np.exp(-t * 1.5)
        + np.sin(2 * np.pi * 392.00 * t) * np.exp(-t * 1.5)
    )
    signal = signal / np.max(np.abs(signal))
    sf.write(str(path), signal.astype(np.float32), sr, subtype="PCM_16")
    return path


@pytest.fixture(scope="module")
def audio_wav(tmp_path_factory):
    p = tmp_path_factory.mktemp("audio") / "test_chord.wav"
    return _make_wav(p)


# ── build-dict tests ──────────────────────────────────────────────────────────

class TestBuildDictCLI:
    def test_harmonic_dict_exits_zero(self, tmp_path):
        out = tmp_path / "dict.npy"
        r   = _run("build-dict", "--harmonic", "--output", str(out),
                   "--pitch-min", "60", "--pitch-max", "72")
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_harmonic_dict_creates_npy(self, tmp_path):
        out = tmp_path / "dict.npy"
        _run("build-dict", "--harmonic", "--output", str(out),
             "--pitch-min", "60", "--pitch-max", "72")
        assert out.exists()

    def test_harmonic_dict_creates_json(self, tmp_path):
        out = tmp_path / "dict.npy"
        _run("build-dict", "--harmonic", "--output", str(out),
             "--pitch-min", "60", "--pitch-max", "72")
        assert out.with_suffix(".json").exists()

    def test_invalid_pitch_range_exits_nonzero(self, tmp_path):
        out = tmp_path / "dict.npy"
        r   = _run("build-dict", "--harmonic", "--output", str(out),
                   "--pitch-min", "90", "--pitch-max", "20")   # invalid
        assert r.returncode != 0

    def test_missing_audio_dir_exits_nonzero(self, tmp_path):
        out = tmp_path / "dict.npy"
        r   = _run("build-dict",
                   "--audio-dir", str(tmp_path / "nonexistent"),
                   "--output", str(out))
        assert r.returncode != 0


# ── transcribe tests ──────────────────────────────────────────────────────────

class TestTranscribeCLI:
    def test_compare_exits_zero(self, audio_wav, tmp_path):
        r = _run("transcribe",
                 "--input",  str(audio_wav),
                 "--method", "compare",
                 "--out-dir", str(tmp_path))
        assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"

    def test_compare_creates_standard_mid(self, audio_wav, tmp_path):
        _run("transcribe", "--input", str(audio_wav),
             "--method", "compare", "--out-dir", str(tmp_path))
        assert (tmp_path / "test_chord_standard.mid").exists()

    def test_compare_creates_group_sparse_mid(self, audio_wav, tmp_path):
        _run("transcribe", "--input", str(audio_wav),
             "--method", "compare", "--out-dir", str(tmp_path))
        assert (tmp_path / "test_chord_group_sparse.mid").exists()

    def test_compare_creates_summary_json(self, audio_wav, tmp_path):
        _run("transcribe", "--input", str(audio_wav),
             "--method", "compare", "--out-dir", str(tmp_path))
        assert (tmp_path / "summary.json").exists()

    def test_summary_json_valid(self, audio_wav, tmp_path):
        _run("transcribe", "--input", str(audio_wav),
             "--method", "compare", "--out-dir", str(tmp_path))
        with open(tmp_path / "summary.json") as f:
            s = json.load(f)
        assert s["method"]  == "compare"
        assert "results"    in s
        assert "standard"   in s["results"]
        assert "group_sparse" in s["results"]

    def test_standard_mid_is_valid(self, audio_wav, tmp_path):
        import pretty_midi
        _run("transcribe", "--input", str(audio_wav),
             "--method", "compare", "--out-dir", str(tmp_path))
        pm = pretty_midi.PrettyMIDI(
            str(tmp_path / "test_chord_standard.mid"))
        assert len(pm.instruments) >= 1

    def test_group_sparse_mid_is_valid(self, audio_wav, tmp_path):
        import pretty_midi
        _run("transcribe", "--input", str(audio_wav),
             "--method", "compare", "--out-dir", str(tmp_path))
        pm = pretty_midi.PrettyMIDI(
            str(tmp_path / "test_chord_group_sparse.mid"))
        assert len(pm.instruments) >= 1

    def test_missing_input_exits_nonzero(self, tmp_path):
        r = _run("transcribe",
                 "--input", str(tmp_path / "missing.wav"),
                 "--method", "compare",
                 "--out-dir", str(tmp_path))
        assert r.returncode != 0

    def test_invalid_method_exits_nonzero(self, audio_wav, tmp_path):
        r = _run("transcribe",
                 "--input", str(audio_wav),
                 "--method", "invalid_method",
                 "--out-dir", str(tmp_path))
        assert r.returncode != 0

    def test_negative_sparsity_exits_nonzero(self, audio_wav, tmp_path):
        r = _run("transcribe",
                 "--input", str(audio_wav),
                 "--sparsity-weight", "-1.0",
                 "--out-dir", str(tmp_path))
        assert r.returncode != 0

    def test_invalid_threshold_exits_nonzero(self, audio_wav, tmp_path):
        r = _run("transcribe",
                 "--input", str(audio_wav),
                 "--threshold", "1.5",
                 "--out-dir", str(tmp_path))
        assert r.returncode != 0

    def test_standard_method_only(self, audio_wav, tmp_path):
        r = _run("transcribe",
                 "--input", str(audio_wav),
                 "--method", "standard",
                 "--out-dir", str(tmp_path))
        assert r.returncode == 0
        assert (tmp_path / "test_chord_standard.mid").exists()
        assert not (tmp_path / "test_chord_group_sparse.mid").exists()

    def test_group_sparse_method_only(self, audio_wav, tmp_path):
        r = _run("transcribe",
                 "--input", str(audio_wav),
                 "--method", "group-sparse",
                 "--out-dir", str(tmp_path))
        assert r.returncode == 0
        assert (tmp_path / "test_chord_group_sparse.mid").exists()
        assert not (tmp_path / "test_chord_standard.mid").exists()
