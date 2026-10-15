"""Shared fixtures: synthetic piano-like audio, so the tests need no datasets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 22050
PITCHES = list(range(48, 85))  # C3..C6, enough for the test pieces
# (midi pitch, onset s, duration s) of the polyphonic test piece
PIECE = [
    (60, 0.0, 0.9), (64, 0.0, 0.9), (67, 0.0, 0.9),  # C major
    (62, 1.0, 0.5), (65, 1.0, 0.5), (69, 1.0, 0.5),  # D minor
    (48, 1.5, 1.0), (55, 1.5, 1.0), (72, 1.5, 0.5),
    (76, 2.5, 1.0), (60, 2.5, 0.45), (60, 3.0, 0.45),  # repeated C4
]  # fmt: skip


def piano_note(midi: int, dur: float, sr: int = SR) -> np.ndarray:
    """Additive piano-like tone: stretched partials, faster decay up high."""
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    for h in range(1, 16):
        fh = h * f0 * np.sqrt(1 + 1e-4 * h * h)
        if fh > sr / 2 - 500:
            break
        y += (
            np.exp(-0.3 * (h - 1))
            / h
            * np.exp(-(1.5 + 0.6 * h) * t)
            * np.sin(2 * np.pi * fh * t + h)
        )
    return y * np.minimum(1, t / 0.005) * np.clip((dur - t) / 0.05, 0, 1)


def render(events, total: float, sr: int = SR) -> np.ndarray:
    y = np.zeros(int(total * sr))
    for midi, start, dur in events:
        n = piano_note(midi, dur, sr)
        i = int(start * sr)
        y[i : i + len(n)] += n[: len(y) - i]
    return 0.8 * y / np.abs(y).max()


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run ``python -m nmf_cli ...`` exactly as a user would in a terminal."""
    return subprocess.run(
        [sys.executable, "-m", "nmf_cli", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture(scope="session")
def notes_dir(tmp_path_factory) -> Path:
    """Single-note recordings named like the MAPS dataset."""
    d = tmp_path_factory.mktemp("single_notes")
    for m in PITCHES:
        y = piano_note(m, 1.0)
        sf.write(d / f"MAPS_ISOL_NO_F_S0_M{m}_AkPnBcht.wav", 0.5 * y / abs(y).max(), SR)
    return d


@pytest.fixture(scope="session")
def piece_wav(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("audio") / "piece.wav"
    sf.write(p, render(PIECE, 4.0), SR)
    return p


@pytest.fixture(scope="session")
def built_dict(notes_dir, tmp_path_factory) -> Path:
    from nmf_core import build_from_audio_dir, save_dictionary

    out = tmp_path_factory.mktemp("dict") / "piano.npy"
    save_dictionary(build_from_audio_dir(notes_dir, (min(PITCHES), max(PITCHES))), out)
    return out
