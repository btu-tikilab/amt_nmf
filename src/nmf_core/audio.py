"""Audio ingestion: load a file with librosa and compute the magnitude CQT.

A constant-Q spectrogram is used because its bins are spaced in fractions of a
semitone, which is what the dictionary metadata (``fmin``,
``bins_per_octave``) describes and what makes one pitch's harmonics line up the
same way for every note.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from .errors import AudioLoadError, DimensionMismatchError, ParameterError

SUPPORTED_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff")


@dataclass(frozen=True)
class SpectrogramConfig:
    """Parameters that must match between a dictionary and the audio it is
    applied to. Stored in the dictionary's JSON sidecar."""

    sr: int = 22050
    hop_length: int = 512
    fmin: float = float(librosa.note_to_hz("A0"))  # 27.5 Hz, lowest piano key
    bins_per_octave: int = 36  # 3 bins per semitone
    n_bins: int = 288  # 8 octaves: A0 .. ~G#8

    def __post_init__(self) -> None:
        if self.sr <= 0 or self.hop_length <= 0:
            raise ParameterError("sr and hop_length must be positive.")
        if self.fmin <= 0 or self.bins_per_octave < 12 or self.n_bins <= 0:
            raise ParameterError(
                "fmin must be > 0, bins_per_octave >= 12 and n_bins > 0."
            )
        if self.bins_per_octave % 12:
            raise ParameterError("bins_per_octave must be a multiple of 12.")
        f_max = self.fmin * 2 ** (self.n_bins / self.bins_per_octave)
        if f_max >= self.sr / 2:
            raise ParameterError(
                f"The CQT's top bin ({f_max:.0f} Hz) exceeds the Nyquist "
                f"frequency ({self.sr / 2:.0f} Hz). Lower n_bins or raise sr."
            )

    @property
    def bin_resolution_semitones(self) -> float:
        return 12.0 / self.bins_per_octave

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = "cqt"
        d["bin_resolution_semitones"] = self.bin_resolution_semitones
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SpectrogramConfig:
        return cls(
            sr=int(d["sr"]),
            hop_length=int(d["hop_length"]),
            fmin=float(d["fmin"]),
            bins_per_octave=int(d["bins_per_octave"]),
            n_bins=int(d["n_bins"]),
        )


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    """Load a mono waveform resampled to ``sr``.

    Raises AudioLoadError for missing, unsupported, undecodable or empty files.
    """
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioLoadError(f"Audio path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioLoadError(
            f"Unsupported audio format '{path.suffix}' for {path.name}. "
            f"Use one of: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    try:
        with warnings.catch_warnings():
            # librosa warns before falling back to audioread; the failure
            # itself is reported below.
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as exc:  # librosa/soundfile raise many types
        raise AudioLoadError(
            f"Cannot read audio file {path}: {exc.__class__.__name__}: {exc}"
        ) from exc
    if y.size == 0:
        raise AudioLoadError(f"Audio file contains no samples: {path}")
    if not np.all(np.isfinite(y)):
        raise AudioLoadError(f"Audio file contains NaN or infinite samples: {path}")
    return y


def spectrogram(y: np.ndarray, cfg: SpectrogramConfig) -> np.ndarray:
    """Magnitude CQT, shape (n_bins, n_frames), dtype float64."""
    with warnings.catch_warnings():
        # librosa warns when a very short clip is shorter than an internal
        # FFT after downsampling; the result is still valid.
        warnings.filterwarnings("ignore", message="n_fft=.*too large")
        C = librosa.cqt(
            y,
            sr=cfg.sr,
            hop_length=cfg.hop_length,
            fmin=cfg.fmin,
            n_bins=cfg.n_bins,
            bins_per_octave=cfg.bins_per_octave,
        )
    return np.abs(C).astype(np.float64)


def load_spectrogram(path: str | Path, cfg: SpectrogramConfig) -> np.ndarray:
    """Load ``path`` and return its magnitude CQT V (F x T)."""
    return spectrogram(load_audio(path, cfg.sr), cfg)


def frame_times(n_frames: int, cfg: SpectrogramConfig) -> np.ndarray:
    """Centre time in seconds of each spectrogram frame (frame * hop / sr)."""
    return np.arange(n_frames) * cfg.hop_length / cfg.sr


def validate_dimensions(V: np.ndarray, W: np.ndarray) -> None:
    """Check that dictionary W (F x K) can explain spectrogram V (F x T)."""
    if W.ndim != 2 or V.ndim != 2:
        raise DimensionMismatchError(
            f"W and V must be 2-D, got W{W.shape} and V{V.shape}."
        )
    if W.shape[0] != V.shape[0]:
        raise DimensionMismatchError(
            f"Dimension mismatch: the dictionary has {W.shape[0]} frequency "
            f"bins per atom but the input spectrogram has {V.shape[0]}. "
            "The dictionary was built with different spectrogram settings "
            "(or its .npy and .json files come from different builds). "
            "Rebuild it with build-dict."
        )
