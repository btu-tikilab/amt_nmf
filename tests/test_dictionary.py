"""Dictionary building, serialization and corruption handling."""

import json

import numpy as np
import pytest

from conftest import PITCHES
from nmf_core import (
    Dictionary,
    SpectrogramConfig,
    build_harmonic_dictionary,
    load_dictionary,
    save_dictionary,
)
from nmf_core.dictionary import pitch_from_filename, sidecar_path
from nmf_core.errors import DictionaryError, ParameterError


@pytest.fixture
def small_dict():
    rng = np.random.default_rng(3)
    return Dictionary(
        W=rng.random((288, 6)),
        config=SpectrogramConfig(),
        pitch_map=[
            {"midi": 60 + i, "note": f"n{i}", "columns": [2 * i, 2 * i + 1]}
            for i in range(3)
        ],
    )


def test_roundtrip_is_exact(tmp_path, small_dict):
    npy, meta = save_dictionary(small_dict, tmp_path / "d.npy")
    loaded = load_dictionary(npy)
    np.testing.assert_array_equal(loaded.W, small_dict.W)
    assert loaded.W.dtype == np.float64
    assert loaded.config == small_dict.config
    assert loaded.pitch_map == small_dict.pitch_map
    assert loaded.groups == [[0, 1], [2, 3], [4, 5]]


def test_sidecar_contains_spectrogram_metadata(tmp_path, small_dict):
    _, meta = save_dictionary(small_dict, tmp_path / "d.npy")
    m = json.loads(meta.read_text())
    spec = m["spectrogram"]
    for key in (
        "sr",
        "hop_length",
        "fmin",
        "bins_per_octave",
        "n_bins",
        "bin_resolution_semitones",
    ):
        assert key in spec
    assert spec["bin_resolution_semitones"] == pytest.approx(1 / 3)
    assert m["shape"] == [288, 6]
    assert m["pitch_map"][0] == {"midi": 60, "note": "n0", "columns": [0, 1]}


def test_missing_files(tmp_path, small_dict):
    with pytest.raises(DictionaryError, match="not found"):
        load_dictionary(tmp_path / "nope.npy")
    npy, meta = save_dictionary(small_dict, tmp_path / "d.npy")
    meta.unlink()
    with pytest.raises(DictionaryError, match="sidecar not found"):
        load_dictionary(npy)


def test_corrupted_npy(tmp_path, small_dict):
    npy, _ = save_dictionary(small_dict, tmp_path / "d.npy")
    npy.write_bytes(b"this is not a numpy file")
    with pytest.raises(DictionaryError, match="corrupted"):
        load_dictionary(npy)


def test_truncated_npy(tmp_path, small_dict):
    npy, _ = save_dictionary(small_dict, tmp_path / "d.npy")
    npy.write_bytes(npy.read_bytes()[:500])
    with pytest.raises(DictionaryError):
        load_dictionary(npy)


def test_corrupted_sidecar(tmp_path, small_dict):
    npy, meta = save_dictionary(small_dict, tmp_path / "d.npy")
    meta.write_text("{ not json")
    with pytest.raises(DictionaryError, match="corrupted"):
        load_dictionary(npy)
    meta.write_text(json.dumps({"pitch_map": []}))
    with pytest.raises(DictionaryError, match="incomplete"):
        load_dictionary(npy)


def test_npy_and_sidecar_from_different_builds(tmp_path, small_dict):
    npy, _ = save_dictionary(small_dict, tmp_path / "d.npy")
    np.save(npy, small_dict.W[:, :4])
    with pytest.raises(DictionaryError, match="do not belong together"):
        load_dictionary(npy)


def test_negative_values_rejected(tmp_path, small_dict):
    npy, _ = save_dictionary(small_dict, tmp_path / "d.npy")
    np.save(npy, -small_dict.W)
    with pytest.raises(DictionaryError, match="negative"):
        load_dictionary(npy)


@pytest.mark.parametrize(
    ("stem", "midi"),
    [
        ("MAPS_ISOL_NO_F_S0_M60_AkPnBcht", 60),
        ("MAPS_ISOL_NO_M_S1_M21_ENSTDkAm", 21),
        ("note_64", 64),
        ("MIDI_108", 108),
        ("piano_A4_69", 69),
        ("72", 72),
        ("piano_C4", 60),
        ("Db5", 73),
        ("ambience", None),
    ],
)
def test_pitch_from_filename(stem, midi):
    assert pitch_from_filename(stem) == midi


def test_supervised_build(built_dict):
    d = load_dictionary(built_dict)
    assert d.pitches == PITCHES
    assert d.W.shape == (288, 2 * len(PITCHES))
    assert np.all(d.W >= 0)
    np.testing.assert_allclose(np.linalg.norm(d.W, axis=0), 1.0)
    # Each pitch's main atom peaks near its fundamental's CQT bin.
    cfg = d.config
    for p in d.pitch_map:
        f0_bin = cfg.bins_per_octave * (p["midi"] - 21) / 12
        peak = np.argmax(d.W[:, p["columns"][0]])
        assert abs(peak - f0_bin) <= 1 or abs(peak - f0_bin - 36) <= 1


def test_harmonic_build():
    d = build_harmonic_dictionary((21, 108), atoms_per_pitch=3)
    assert d.pitches == list(range(21, 109))
    assert d.W.shape == (288, 88 * 3)
    assert np.all(np.isfinite(d.W)) and np.all(d.W >= 0)


def test_invalid_pitch_range():
    with pytest.raises(ParameterError):
        build_harmonic_dictionary((90, 20))
    with pytest.raises(ParameterError):
        build_harmonic_dictionary((0, 128))


def test_sidecar_path():
    assert sidecar_path("a/b/dict.npy").name == "dict.json"
