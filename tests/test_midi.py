"""tests/test_midi.py — MIDI generation tests."""
from __future__ import annotations

import numpy as np
import pretty_midi
import pytest

from nmf_core.midi import activations_to_midi, save_activations_npz

SR, HOP = 22050, 512


@pytest.fixture
def simple_H():
    """H matrix with clear on/off pattern for 3 notes."""
    H = np.zeros((3, 200))
    H[0, 10:80]  = 0.9   # note 60 active frames 10-80
    H[1, 50:120] = 0.8   # note 64 active frames 50-120
    H[2, 130:180]= 0.7   # note 67 active frames 130-180
    return H


@pytest.fixture
def pitch_map():
    return [
        {"component": 0, "midi_note": 60, "note_name": "C4"},
        {"component": 1, "midi_note": 64, "note_name": "E4"},
        {"component": 2, "midi_note": 67, "note_name": "G4"},
    ]


class TestMIDIGeneration:
    def test_creates_file(self, tmp_path, simple_H, pitch_map):
        out = tmp_path / "test.mid"
        activations_to_midi(simple_H, pitch_map, SR, HOP, out)
        assert out.exists()

    def test_file_is_valid_midi(self, tmp_path, simple_H, pitch_map):
        out = tmp_path / "test.mid"
        activations_to_midi(simple_H, pitch_map, SR, HOP, out)
        pm = pretty_midi.PrettyMIDI(str(out))
        assert len(pm.instruments) >= 1

    def test_file_is_nonempty(self, tmp_path, simple_H, pitch_map):
        out = tmp_path / "test.mid"
        activations_to_midi(simple_H, pitch_map, SR, HOP, out)
        pm = pretty_midi.PrettyMIDI(str(out))
        all_notes = [n for inst in pm.instruments for n in inst.notes]
        assert len(all_notes) > 0

    def test_expected_pitches_present(self, tmp_path, simple_H, pitch_map):
        out = tmp_path / "test.mid"
        activations_to_midi(simple_H, pitch_map, SR, HOP, out)
        pm     = pretty_midi.PrettyMIDI(str(out))
        pitches = {n.pitch for inst in pm.instruments for n in inst.notes}
        assert 60 in pitches   # C4
        assert 64 in pitches   # E4
        assert 67 in pitches   # G4

    def test_note_timing_approximate(self, tmp_path, simple_H, pitch_map):
        """C4 should start around frame 10 * hop/sr = ~0.23s."""
        out = tmp_path / "test.mid"
        activations_to_midi(simple_H, pitch_map, SR, HOP, out,
                            threshold=0.5)
        pm   = pretty_midi.PrettyMIDI(str(out))
        c4   = [n for inst in pm.instruments for n in inst.notes if n.pitch == 60]
        assert len(c4) >= 1
        expected_start = 10 * HOP / SR
        assert abs(c4[0].start - expected_start) < 0.1

    def test_zero_H_produces_no_notes(self, tmp_path, pitch_map):
        H_zero = np.zeros((3, 100))
        out    = tmp_path / "silent.mid"
        activations_to_midi(H_zero, pitch_map, SR, HOP, out, threshold=0.1)
        pm     = pretty_midi.PrettyMIDI(str(out))
        all_notes = [n for inst in pm.instruments for n in inst.notes]
        assert len(all_notes) == 0

    def test_velocity_in_valid_range(self, tmp_path, simple_H, pitch_map):
        out = tmp_path / "vel.mid"
        activations_to_midi(simple_H, pitch_map, SR, HOP, out)
        pm  = pretty_midi.PrettyMIDI(str(out))
        for inst in pm.instruments:
            for n in inst.notes:
                assert 0 <= n.velocity <= 127


class TestNPZExport:
    def test_creates_npz(self, tmp_path, simple_H, pitch_map):
        out = tmp_path / "activations.npz"
        save_activations_npz(simple_H, pitch_map, out)
        assert out.exists()

    def test_H_reloadable(self, tmp_path, simple_H, pitch_map):
        out = tmp_path / "activations.npz"
        save_activations_npz(simple_H, pitch_map, out)
        data = np.load(str(out), allow_pickle=True)
        np.testing.assert_array_equal(data["H"], simple_H)
