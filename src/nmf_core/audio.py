"""
src/nmf_core/audio.py
─────────────────────────────────────────────────────────────────
Audio ingestion pipeline — load audio and compute magnitude spectrogram.

All NMF algorithms in this package consume a magnitude spectrogram V,
not the raw waveform. This module is the single entry point for turning
any WAV/FLAC/MP3 file into V.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class AudioSpec:
    """All spectrogram data needed downstream by NMF solvers."""

    V: np.ndarray        # (F, T)  magnitude spectrogram — NMF input
    sr: int              # sample rate
    n_fft: int
    hop_length: int
    freqs: np.ndarray    # (F,)  Hz per bin
    times: np.ndarray    # (T,)  seconds per frame
    duration_s: float
    source_path: str


# Default STFT parameters (matches librosa defaults)
DEFAULT_N_FFT      = 2048
DEFAULT_HOP_LENGTH = 512


def load_spectrogram(
    audio_path: str | Path,
    sr: int = 22050,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    normalize: bool = False,
) -> AudioSpec:
    """
    Load audio and compute its magnitude spectrogram V.

    Parameters
    ----------
    audio_path : path to WAV / FLAC / MP3
    sr         : target sample rate (None = keep original)
    n_fft      : FFT window size
    hop_length : STFT hop in samples
    normalize  : if True, divide V by its max (required for β<1 divergence)

    Returns
    -------
    AudioSpec with V, freqs, times, sr, etc.

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValueError       if the file cannot be decoded.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        y, sr_actual = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        raise ValueError(f"Cannot decode audio file '{path}': {exc}") from exc

    D     = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    V     = np.abs(D).astype(np.float64)
    freqs = librosa.fft_frequencies(sr=sr_actual, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(V.shape[1]), sr=sr_actual, hop_length=hop_length
    )

    if normalize:
        v_max = V.max()
        if v_max > 0:
            V = V / v_max

    return AudioSpec(
        V=V,
        sr=sr_actual,
        n_fft=n_fft,
        hop_length=hop_length,
        freqs=freqs,
        times=times,
        duration_s=float(len(y) / sr_actual),
        source_path=str(path),
    )


def validate_dimensions(V: np.ndarray, W: np.ndarray) -> None:
    """
    Assert that dictionary W is compatible with spectrogram V.

    Raises ValueError with a clear human-readable message if not.
    """
    if W.ndim != 2:
        raise ValueError(
            f"Dictionary W must be 2-D (F x K), got shape {W.shape}."
        )
    if V.ndim != 2:
        raise ValueError(
            f"Spectrogram V must be 2-D (F x T), got shape {V.shape}."
        )
    if V.shape[0] != W.shape[0]:
        raise ValueError(
            f"Dimension mismatch: spectrogram V has {V.shape[0]} frequency bins "
            f"but dictionary W has {W.shape[0]} rows. "
            f"Rebuild the dictionary with the same n_fft and sample rate "
            f"as the input audio (current n_fft produces {V.shape[0]} bins)."
        )
