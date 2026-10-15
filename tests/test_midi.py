"""Activation -> MIDI conversion."""

import json

import numpy as np
import pretty_midi
import pytest

from nmf_core.errors import ParameterError
from nmf_core.midi import (
    frames_to_seconds,
    pitch_activations,
    roll_to_notes,
    save_activations,
    write_midi,
)

SR, HOP = 22050, 512
FRAME = HOP / SR  # 23.2 ms


def _bump(T, on, off, height=1.0, decay=0.02):
    """Instant attack at frame ``on``, exponential decay, zero from ``off``."""
    row = np.zeros(T)
    n = off - on
    row[on:off] = height * np.exp(-decay * np.arange(n))
    return row


@pytest.fixture
def roll():
    T = 200
    return np.stack(
        [
            _bump(T, 10, 60),  # pitch 60: one note
            _bump(T, 50, 120, 0.6),  # pitch 64: overlaps the first
            _bump(T, 0, 40, 0.02) + 0.0,  # pitch 67: below threshold
            _bump(T, 100, 130) + _bump(T, 140, 190, 0.8),  # 72: two attacks
        ]
    )


def test_notes_pitches_and_timing(roll):
    notes = roll_to_notes(roll, [60, 64, 67, 72], SR, HOP, threshold=0.1)
    got = [(n.pitch, round(n.start / FRAME), round(n.end / FRAME)) for n in notes]
    assert [g[0] for g in got] == [60, 64, 72, 72]
    starts = {(p, s) for p, s, _ in got}
    assert starts == {(60, 10), (64, 50), (72, 100), (72, 140)}
    for n in notes:
        assert 1 <= n.velocity <= 127 and n.end > n.start


def test_repeated_note_without_silence_retriggers():
    T = 120
    row = _bump(T, 10, 120, 1.0, decay=0.03)
    row[60:] += _bump(T, 60, 120, 0.9, decay=0.03)[60:]  # re-attack, no gap
    notes = roll_to_notes(row[None], [60], SR, HOP, threshold=0.1)
    assert [round(n.start / FRAME) for n in notes] == [10, 60]
    assert notes[0].end == pytest.approx(notes[1].start)


def test_note_off_uses_hysteresis():
    row = np.zeros(100)
    row[10:20] = 1.0
    row[20:40] = 0.07  # below threshold 0.1 but above off level 0.05
    row[40:50] = 0.03  # below off level
    (note,) = roll_to_notes(row[None], [60], SR, HOP, threshold=0.1)
    assert round(note.start / FRAME) == 10 and round(note.end / FRAME) == 40


def test_threshold_and_min_duration():
    row = np.zeros(100)
    row[10:12] = 1.0  # 46 ms
    row[50:80] = 0.5
    assert len(roll_to_notes(row[None], [60], SR, HOP, 0.1, min_duration=0.05)) == 1
    assert len(roll_to_notes(row[None], [60], SR, HOP, 0.6)) == 0
    assert len(roll_to_notes(row[None], [60], SR, HOP, 0.6, min_duration=0)) == 1
    assert roll_to_notes(np.zeros((1, 50)), [60], SR, HOP) == []
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ParameterError):
            roll_to_notes(row[None], [60], SR, HOP, threshold=bad)


def test_frame_to_time_conversion():
    assert frames_to_seconds(43, 22050, 512) == pytest.approx(43 * 512 / 22050)
    row = np.zeros(400)
    row[100:200] = 1.0
    (note,) = roll_to_notes(row[None], [60], 44100, 256)
    assert note.start == pytest.approx(100 * 256 / 44100)
    assert note.end == pytest.approx(200 * 256 / 44100)


def test_pitch_activations_sums_groups():
    H = np.arange(12.0).reshape(4, 3)
    np.testing.assert_array_equal(
        pitch_activations(H, [[0, 1], [2, 3]]), [[3, 5, 7], [15, 17, 19]]
    )


def test_written_midi_is_valid(tmp_path, roll):
    notes = roll_to_notes(roll, [60, 64, 67, 72], SR, HOP)
    path = write_midi(notes, tmp_path / "x.mid")
    pm = pretty_midi.PrettyMIDI(str(path))
    written = pm.instruments[0].notes
    assert len(written) == len(notes) > 0
    for w, n in zip(
        sorted(written, key=lambda x: (x.start, x.pitch)), notes, strict=True
    ):
        assert w.pitch == n.pitch
        assert w.start == pytest.approx(n.start, abs=2e-3)
        assert w.end == pytest.approx(n.end, abs=2e-3)


def test_npz_contents(tmp_path, roll):
    H = np.repeat(roll, 2, axis=0) / 2
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    notes = roll_to_notes(roll, [60, 64, 67, 72], SR, HOP)
    p = save_activations(
        tmp_path / "h.npz", H, roll, [60, 64, 67, 72], groups, SR, HOP, notes
    )
    z = np.load(p)
    np.testing.assert_array_equal(z["H"], H)
    np.testing.assert_array_equal(z["pitch_roll"], roll)
    assert list(z["atom_pitch"]) == [60, 60, 64, 64, 67, 67, 72, 72]
    assert z["times"][1] == pytest.approx(FRAME)
    assert len(json.loads(str(z["notes"]))) == len(notes)


def test_refine_offsets_keeps_onsets_and_extends_tails():
    from nmf_core.midi import NoteEvent, refine_offsets

    T = 100
    # The detection roll ended the note early (frame 30); the refit roll
    # still shows the decaying tail until frame 60.
    notes = [NoteEvent(60, 10 * FRAME, 30 * FRAME, 90)]
    refit = _bump(T, 10, 60, decay=0.03)[None]
    (n,) = refine_offsets(notes, refit, [60], SR, HOP, off_level=0.05)
    assert n.start == notes[0].start and n.pitch == 60 and n.velocity == 90
    assert round(n.end / FRAME) == 60
    # A following onset of the same pitch still caps the note.
    two = [*notes, NoteEvent(60, 40 * FRAME, 50 * FRAME, 90)]
    first = refine_offsets(two, refit, [60], SR, HOP)[0]
    assert round(first.end / FRAME) == 40
