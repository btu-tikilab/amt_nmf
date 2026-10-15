"""Full pipeline in-process: dimension validation, outputs, accuracy."""

import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from conftest import PIECE
from nmf_core import Dictionary, load_dictionary, save_dictionary, transcribe
from nmf_core.errors import (
    AudioLoadError,
    DimensionMismatchError,
    ParameterError,
    TranscriberError,
)


def _onset_f1(midi_path, events, tol=0.05):
    est = pretty_midi.PrettyMIDI(str(midi_path)).instruments[0].notes
    used, tp = set(), 0
    for pitch, start, _ in events:
        for i, n in enumerate(est):
            if i not in used and n.pitch == pitch and abs(n.start - start) <= tol:
                used.add(i)
                tp += 1
                break
    p, r = tp / max(len(est), 1), tp / len(events)
    return 2 * p * r / max(p + r, 1e-9)


def test_mismatched_dictionary_raises_validation_error(tmp_path, piece_wav, built_dict):
    d = load_dictionary(built_dict)
    bad = Dictionary(W=d.W[:200], config=d.config, pitch_map=d.pitch_map)
    with pytest.raises(DimensionMismatchError, match="Dimension mismatch"):
        transcribe(piece_wav, bad, tmp_path)
    # Same when the mismatched dictionary comes from disk.
    path = tmp_path / "bad.npy"
    save_dictionary(bad, path)
    with pytest.raises(DimensionMismatchError) as exc:
        transcribe(piece_wav, path, tmp_path)
    assert isinstance(exc.value, TranscriberError)  # handled by the CLI
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sparsity_weight": -0.1},
        {"threshold": -0.2},
        {"threshold": 0},
        {"threshold": 2},
        {"method": "fast"},
        {"beta": 3},
        {"max_iter": 0},
        {"sparsity_weight": float("nan")},
    ],
)
def test_invalid_hyperparameters(tmp_path, piece_wav, built_dict, kwargs):
    with pytest.raises(ParameterError):
        transcribe(piece_wav, built_dict, tmp_path, **kwargs)


def test_bad_audio(tmp_path, built_dict):
    with pytest.raises(AudioLoadError, match="not found"):
        transcribe(tmp_path / "missing.wav", built_dict, tmp_path)
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"RIFF....not really audio")
    with pytest.raises(AudioLoadError, match="Cannot read"):
        transcribe(junk, built_dict, tmp_path)
    empty = tmp_path / "empty.wav"
    sf.write(empty, np.zeros(0), 22050)
    with pytest.raises(AudioLoadError):
        transcribe(empty, built_dict, tmp_path)
    with pytest.raises(AudioLoadError, match="Unsupported"):
        transcribe(built_dict, built_dict, tmp_path)


def test_silent_input_gives_empty_but_valid_midi(tmp_path, built_dict):
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(22050), 22050)
    s = transcribe(silent, built_dict, tmp_path)
    for key in ("standard", "group_sparse"):
        assert s["results"][key]["n_notes"] == 0
        pretty_midi.PrettyMIDI(s["results"][key]["midi"])


def test_compare_outputs_and_accuracy(tmp_path, piece_wav, built_dict):
    s = transcribe(piece_wav, built_dict, tmp_path, method="compare")
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [
        "piece_group_sparse.mid",
        "piece_group_sparse_H.npz",
        "piece_standard.mid",
        "piece_standard_H.npz",
        "summary.json",
    ]
    assert json.loads((tmp_path / "summary.json").read_text()) == s
    assert set(s["comparison"]) >= {"n_notes", "pitch_roll_hoyer_sparsity"}
    for key in ("standard", "group_sparse"):
        r = s["results"][key]
        assert r["iterations"] >= 1 and isinstance(r["converged"], bool)
        assert _onset_f1(r["midi"], PIECE) >= 0.85, key
    gs = s["results"]["group_sparse"]
    std = s["results"]["standard"]
    assert gs["pitch_roll_hoyer_sparsity"] > std["pitch_roll_hoyer_sparsity"]


def test_single_method_writes_only_its_files(tmp_path, piece_wav, built_dict):
    transcribe(piece_wav, built_dict, tmp_path, method="group-sparse")
    assert (tmp_path / "piece_group_sparse.mid").exists()
    assert not (tmp_path / "piece_standard.mid").exists()
