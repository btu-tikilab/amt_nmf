"""tests/test_dictionary.py — dictionary serialization and validation."""
from __future__ import annotations

import numpy as np
import pytest

from nmf_core.audio import validate_dimensions
from nmf_core.dictionary import (
    build_harmonic_dictionary,
    load_dictionary,
    save_dictionary,
)


@pytest.fixture
def sample_W():
    rng = np.random.default_rng(1)
    return rng.uniform(0, 1, (1025, 12)).astype(np.float64)


@pytest.fixture
def pitch_map():
    return [
        {"component": i, "midi_note": 60 + i, "note_name": f"note_{i}"}
        for i in range(12)
    ]


class TestSerialization:
    def test_save_creates_npy(self, tmp_path, sample_W, pitch_map):
        p = tmp_path / "test_dict.npy"
        save_dictionary(sample_W, p, n_fft=2048, sr=22050,
                        hop_length=512, pitch_map=pitch_map)
        assert p.exists()

    def test_save_creates_json_sidecar(self, tmp_path, sample_W, pitch_map):
        p = tmp_path / "test_dict.npy"
        save_dictionary(sample_W, p, n_fft=2048, sr=22050,
                        hop_length=512, pitch_map=pitch_map)
        assert p.with_suffix(".json").exists()

    def test_roundtrip_exact(self, tmp_path, sample_W, pitch_map):
        """Saving W and reloading must reproduce the exact numpy array."""
        p = tmp_path / "dict.npy"
        save_dictionary(sample_W, p, n_fft=2048, sr=22050,
                        hop_length=512, pitch_map=pitch_map)
        W_loaded, meta = load_dictionary(p)
        np.testing.assert_array_equal(sample_W, W_loaded)

    def test_metadata_fields(self, tmp_path, sample_W, pitch_map):
        p = tmp_path / "dict.npy"
        save_dictionary(sample_W, p, n_fft=2048, sr=22050,
                        hop_length=512, pitch_map=pitch_map)
        _, meta = load_dictionary(p)
        assert meta["n_fft"]      == 2048
        assert meta["sr"]         == 22050
        assert meta["hop_length"] == 512
        assert meta["F"]          == sample_W.shape[0]
        assert meta["K"]          == sample_W.shape[1]

    def test_pitch_map_preserved(self, tmp_path, sample_W, pitch_map):
        p = tmp_path / "dict.npy"
        save_dictionary(sample_W, p, n_fft=2048, sr=22050,
                        hop_length=512, pitch_map=pitch_map)
        _, meta = load_dictionary(p)
        assert meta["pitch_map"] == pitch_map

    def test_missing_npy_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_dictionary(tmp_path / "nonexistent.npy")

    def test_missing_json_raises(self, tmp_path, sample_W, pitch_map):
        p = tmp_path / "dict.npy"
        np.save(str(p), sample_W)
        # No JSON sidecar
        with pytest.raises(FileNotFoundError, match="sidecar"):
            load_dictionary(p)


class TestDimensionValidation:
    def test_correct_dims_passes(self):
        V = np.ones((1025, 100))
        W = np.ones((1025, 5))
        validate_dimensions(V, W)   # should not raise

    def test_mismatch_raises_clear_message(self):
        V = np.ones((1025, 100))
        W = np.ones((512, 5))      # wrong F
        with pytest.raises(ValueError, match="Dimension mismatch"):
            validate_dimensions(V, W)

    def test_3d_W_raises(self):
        V = np.ones((1025, 100))
        W = np.ones((1025, 5, 2))
        with pytest.raises(ValueError, match="2-D"):
            validate_dimensions(V, W)


class TestHarmonicDictionary:
    def test_shape(self):
        import librosa
        freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
        D, groups, pm = build_harmonic_dictionary(
            freqs, midi_low=60, midi_high=67, n_harmonics=4
        )
        assert D.ndim == 2
        assert D.shape[0] == len(freqs)

    def test_nonneg(self):
        import librosa
        freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
        D, _, _ = build_harmonic_dictionary(freqs, midi_low=60, midi_high=67)
        assert np.all(D >= 0)

    def test_groups_cover_all_columns(self):
        import librosa
        freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
        D, groups, _ = build_harmonic_dictionary(
            freqs, midi_low=60, midi_high=67
        )
        all_cols = [c for g in groups for c in g]
        assert sorted(all_cols) == list(range(D.shape[1]))
