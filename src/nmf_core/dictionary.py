"""
src/nmf_core/dictionary.py
─────────────────────────────────────────────────────────────────
Dictionary Management Subsystem.

Two ways to build a dictionary W:

A) Supervised (preferred) — extract from isolated single-note recordings.
   Each note clip
   NMF with K=1 -> the one W column is that note's
   spectral fingerprint.  Much cleaner than unsupervised because the
   algorithm has exactly one note to represent.

B) Harmonic (fast, no audio needed) — build Gaussian harmonic atoms
   from musical physics (same approach as group_snmf/).

The dictionary is stored as two files:
  <name>.npy   — the W matrix (float64, F × K)
  <name>.json  — metadata sidecar (n_fft, sr, hop_length, pitch_map)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np

from .audio import load_spectrogram
from .standard_nmf import run_nmf

# ── metadata sidecar ──────────────────────────────────────────────────────────


def _meta_path(npy_path: str | Path) -> Path:
    p = Path(npy_path)
    return p.with_suffix(".json")


def save_dictionary(
    W: np.ndarray,
    path: str | Path,
    n_fft: int,
    sr: int,
    hop_length: int,
    pitch_map: List[Dict],
) -> None:
    """
    Save dictionary matrix W + JSON sidecar.

    pitch_map: list of {"component": k, "midi_note": m, "note_name": n}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(path), W.astype(np.float64))

    meta = {
        "n_fft": n_fft,
        "sr": sr,
        "hop_length": hop_length,
        "F": int(W.shape[0]),
        "K": int(W.shape[1]),
        "pitch_map": pitch_map,
    }
    with open(_meta_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Dictionary saved -> {path}  ({W.shape[0]}x{W.shape[1]})")
    print(f"  Metadata   saved -> {_meta_path(path)}")


def load_dictionary(path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Load dictionary W and its metadata sidecar.

    Returns (W, meta_dict).
    Raises FileNotFoundError if either file is missing.
    Raises ValueError if the .npy is corrupt.
    """
    path = Path(path)
    meta_p = _meta_path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dictionary file not found: {path}\n"
            f"Run:  python -m nmf_cli build-dict  to create one."
        )
    if not meta_p.exists():
        raise FileNotFoundError(
            f"Dictionary metadata sidecar missing: {meta_p}\n"
            f"Rebuild the dictionary — the .json sidecar is required."
        )

    try:
        W = np.load(str(path)).astype(np.float64)
    except Exception as exc:
        raise ValueError(f"Cannot read dictionary file '{path}': {exc}") from exc

    with open(meta_p, encoding="utf-8") as f:
        meta = json.load(f)

    # Verify stored shape matches the file
    if W.shape != (meta["F"], meta["K"]):
        raise ValueError(
            f"Shape mismatch: file says ({meta['F']}, {meta['K']}) "
            f"but loaded array is {W.shape}. The .npy may be corrupt."
        )

    return W, meta


# ── supervised builder ────────────────────────────────────────────────────────


def build_from_audio_dir(
    audio_dir: str | Path,
    output_path: str | Path,
    pitch_range: tuple[int, int] = (21, 108),
    n_fft: int = 2048,
    hop_length: int = 512,
    sr: int = 22050,
    n_iter: int = 100,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build a supervised dictionary from a folder of isolated single-note WAVs.

    File-naming convention (MAPS-compatible):
        MIDI_<note>.wav  or  <any_prefix>_M<note>.wav
    Any file whose name contains the MIDI note number is accepted.
    If the MIDI note cannot be parsed from the filename, the file is
    assigned the next available MIDI number sequentially.

    For each file:
        1. Load audio -> magnitude spectrogram V  (F × T)
        2. Run NMF with K=1 to get the dominant spectral template
        3. Store the single W column as the note's fingerprint

    Parameters
    ----------
    audio_dir   : folder containing single-note WAV/FLAC files
    output_path : where to save the .npy + .json
    pitch_range : (midi_low, midi_high) — only notes in this range are kept
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    # Collect audio files
    exts = {".wav", ".flac", ".mp3", ".ogg"}
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise ValueError(
            f"No audio files found in {audio_dir}. Expected .wav / .flac / .mp3 files."
        )

    midi_low, midi_high = pitch_range
    columns: List[np.ndarray] = []
    pitch_map: List[Dict] = []
    F: Optional[int] = None

    for file_path in files:
        # Try to parse MIDI note from filename
        midi_note = _midi_from_filename(file_path.stem)
        if midi_note is None or not (midi_low <= midi_note <= midi_high):
            if verbose:
                print(
                    f"  Skipping {file_path.name} "
                    f"(MIDI {midi_note} outside range {pitch_range})"
                )
            continue

        if verbose:
            note_name = librosa.midi_to_note(midi_note)
            print(f"  Processing {file_path.name}  ->  MIDI {midi_note} ({note_name})")

        try:
            spec = load_spectrogram(
                file_path, sr=sr, n_fft=n_fft, hop_length=hop_length
            )
        except (FileNotFoundError, ValueError) as exc:
            if verbose:
                print(f"    WARNING: skipping ({exc})")
            continue

        if F is None:
            F = spec.V.shape[0]

        # NMF with K=1 -> extract single spectral basis vector
        result = run_nmf(
            spec.V,
            K=1,
            n_iter=n_iter,
            verbose=False,
            progress_callback=None,
        )
        w_col = result.W[:, 0]  # (F,)
        w_col /= w_col.max() + 1e-10  # normalise to [0,1]
        columns.append(w_col)

        note_name = librosa.midi_to_note(midi_note)
        pitch_map.append(
            {
                "component": len(columns) - 1,
                "midi_note": int(midi_note),
                "note_name": note_name,
                "source": file_path.name,
            }
        )

    if not columns:
        raise ValueError(
            "No valid audio files produced dictionary columns. "
            "Check that filenames contain MIDI note numbers and "
            "that files are readable."
        )

    W = np.stack(columns, axis=1)  # (F, K)

    save_dictionary(
        W,
        output_path,
        n_fft=n_fft,
        sr=sr,
        hop_length=hop_length,
        pitch_map=pitch_map,
    )
    if verbose:
        print(f"\n  Built dictionary: {W.shape[0]} freq bins × {W.shape[1]} notes")

    return W


# ── harmonic dictionary builder ───────────────────────────────────────────────



def build_harmonic_dictionary(
    freqs: np.ndarray,
    midi_low: int = 21,
    midi_high: int = 108,
    n_harmonics: int = 6,
    sigma_hz: float = 8.0,
) -> tuple[np.ndarray, List[List[int]], List[Dict]]:
    """
    Build one combined harmonic template per MIDI note.

    Returns:
      D:         (F, K) dictionary, one column per note
      groups:    singleton column-index lists, one per note
      pitch_map: one metadata entry per note
    """
    templates: List[np.ndarray] = []
    groups: List[List[int]] = []
    pitch_map: List[Dict] = []

    freqs = np.asarray(freqs, dtype=np.float64)

    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("freqs must be a non-empty 1-D array.")

    if not np.all(np.isfinite(freqs)):
        raise ValueError("freqs must contain only finite values.")

    if np.any(freqs < 0) or np.any(np.diff(freqs) < 0):
        raise ValueError("freqs must be nonnegative and sorted in ascending order.")

    if not 0 <= midi_low <= midi_high <= 127:
        raise ValueError("MIDI range must satisfy 0 <= midi_low <= midi_high <= 127.")

    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1.")

    if not np.isfinite(sigma_hz) or sigma_hz <= 0:
        raise ValueError("sigma_hz must be a positive finite number.")

    for m in range(midi_low, midi_high + 1):
        f0 = float(librosa.midi_to_hz(m))
        template = np.zeros_like(freqs, dtype=np.float64)

        for h in range(1, n_harmonics + 1):
            f_h = h * f0
            if f_h > freqs[-1]:
                break

            harmonic = (1.0 / h) * np.exp(
                -0.5 * ((freqs - f_h) / sigma_hz) ** 2
            )
            template += harmonic

        norm = np.linalg.norm(template)
        if norm <= 1e-12:
            continue

        template /= norm
        component = len(templates)
        templates.append(template)
        groups.append([component])

        pitch_map.append(
            {
                "component": component,
                "midi_note": int(m),
                "note_name": librosa.midi_to_note(m),
            }
        )

    if not templates:
        raise ValueError("No harmonic templates could be generated.")

    D = np.stack(templates, axis=1).astype(np.float64)
    return D, groups, pitch_map


# ── helpers ────────────────────────────────────────────────────────────────────


def _midi_from_filename(stem: str) -> Optional[int]:
    """
    Parse MIDI note number from common filename patterns:
      piano_A4_69.wav -> 69
      MIDI_060.wav    -> 60
      note60.wav      -> 60
    """
    import re

    # Pattern: ends with _<number> or contains MIDI_<number>
    patterns = [
        r"_M?(\d{1,3})$",
        r"MIDI[_-]?(\d{1,3})",
        r"note[_-]?(\d{1,3})",
        r"(\d{1,3})$",
    ]
    for pat in patterns:
        m = re.search(pat, stem, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 21 <= val <= 108:
                return val
    return None
