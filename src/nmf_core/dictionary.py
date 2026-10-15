"""Dictionary management: build, save and load the pitch dictionary W.

A dictionary is two files side by side:

* ``<name>.npy``  - W, float64, shape (n_bins, n_pitches * atoms_per_pitch)
* ``<name>.json`` - sidecar with the spectrogram configuration (sr, hop_length,
  fmin, bins_per_octave, n_bins, bin resolution) and the pitch mapping, i.e.
  which columns of W belong to which MIDI pitch. Transcription computes the
  input spectrogram from this configuration, so W and V always agree.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np

from .audio import (
    SUPPORTED_EXTENSIONS,
    SpectrogramConfig,
    load_audio,
    spectrogram,
)
from .errors import AudioLoadError, DictionaryError, ParameterError
from .nmf import EPS, standard_nmf

FORMAT_VERSION = 1


@dataclass
class Dictionary:
    W: np.ndarray
    config: SpectrogramConfig
    pitch_map: list[dict]  # [{"midi": 60, "note": "C4", "columns": [0, 1]}, ...]
    extra: dict = field(default_factory=dict)

    @property
    def pitches(self) -> list[int]:
        return [int(p["midi"]) for p in self.pitch_map]

    @property
    def groups(self) -> list[list[int]]:
        return [list(p["columns"]) for p in self.pitch_map]

    def metadata(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "shape": list(self.W.shape),
            "spectrogram": self.config.to_dict(),
            "pitch_range": [min(self.pitches), max(self.pitches)],
            "n_pitches": len(self.pitch_map),
            "pitch_map": self.pitch_map,
            **self.extra,
        }


def sidecar_path(npy_path: str | Path) -> Path:
    return Path(npy_path).with_suffix(".json")


def save_dictionary(d: Dictionary, path: str | Path) -> tuple[Path, Path]:
    """Write W to ``path`` (.npy) and the metadata to the .json sidecar."""
    path = Path(path)
    if path.suffix.lower() != ".npy":
        raise ParameterError(f"Dictionary output must end in .npy, got {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(d.W, dtype=np.float64), allow_pickle=False)
    meta = sidecar_path(path)
    meta.write_text(json.dumps(d.metadata(), indent=2), encoding="utf-8")
    return path, meta


def load_dictionary(path: str | Path) -> Dictionary:
    """Load and validate a dictionary saved by ``save_dictionary``.

    Raises DictionaryError with a readable message if either file is missing,
    unreadable, corrupted or inconsistent with the other.
    """
    path = Path(path)
    rebuild = "Rebuild it with: python -m nmf_cli build-dict"
    if not path.is_file():
        raise DictionaryError(f"Dictionary file not found: {path}. {rebuild}")
    meta_path = sidecar_path(path)
    if not meta_path.is_file():
        raise DictionaryError(
            f"Dictionary metadata sidecar not found: {meta_path}. "
            f"The .json file must sit next to the .npy file. {rebuild}"
        )
    try:
        W = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise DictionaryError(
            f"Dictionary file {path} is corrupted or not a NumPy .npy array "
            f"({exc.__class__.__name__}: {exc}). {rebuild}"
        ) from exc
    if W.ndim != 2 or W.size == 0 or not np.issubdtype(W.dtype, np.number):
        raise DictionaryError(
            f"Dictionary {path} must be a non-empty 2-D numeric array, "
            f"got shape {W.shape} dtype {W.dtype}."
        )
    W = W.astype(np.float64)
    if not np.all(np.isfinite(W)) or np.any(W < 0):
        raise DictionaryError(
            f"Dictionary {path} contains negative, NaN or infinite values."
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        config = SpectrogramConfig.from_dict(meta["spectrogram"])
        pitch_map = meta["pitch_map"]
        columns = [int(c) for p in pitch_map for c in p["columns"]]
        [int(p["midi"]) for p in pitch_map]
    except (ValueError, KeyError, TypeError, ParameterError) as exc:
        raise DictionaryError(
            f"Dictionary sidecar {meta_path} is corrupted or incomplete "
            f"({exc.__class__.__name__}: {exc}). {rebuild}"
        ) from exc
    if sorted(columns) != list(range(W.shape[1])):
        raise DictionaryError(
            f"Dictionary {path} has {W.shape[1]} atoms but its sidecar maps "
            f"{len(columns)} columns to pitches. The .npy and .json files do "
            f"not belong together. {rebuild}"
        )
    extra = {
        k: v
        for k, v in meta.items()
        if k
        not in {
            "format_version",
            "shape",
            "spectrogram",
            "pitch_range",
            "n_pitches",
            "pitch_map",
        }
    }
    return Dictionary(W=W, config=config, pitch_map=pitch_map, extra=extra)


# ── filename -> MIDI pitch ────────────────────────────────────────────────────

_PITCH_PATTERNS = (
    re.compile(r"(?:^|_)M(\d{1,3})(?:_|$)"),  # MAPS: MAPS_ISOL_NO_F_S0_M60_...
    re.compile(r"midi[_-]?(\d{1,3})", re.IGNORECASE),  # midi60, MIDI_060
    re.compile(r"note[_-]?(\d{1,3})", re.IGNORECASE),  # note60, note_60
    re.compile(r"(?:^|[_\-. ])(\d{1,3})$"),  # piano_60, 060
    re.compile(r"^(\d{1,3})(?:[_\-. ]|$)"),  # 60_piano
)
_NOTE_NAME = re.compile(r"(?:^|[_\-. ])([A-Ga-g][#b]?-?\d)(?:[_\-. ]|$)")


def pitch_from_filename(stem: str) -> int | None:
    """Parse a MIDI pitch from a file name stem, or return None.

    Understands MAPS names (``..._M60_...``), ``midi60``, ``note_60``,
    a trailing or leading number (``piano_60``) and note names (``piano_C4``,
    ``Db5``).
    """
    for pat in _PITCH_PATTERNS:
        m = pat.search(stem)
        if m and 0 <= int(m.group(1)) <= 127:
            return int(m.group(1))
    m = _NOTE_NAME.search(stem)
    if m:
        try:
            midi = int(librosa.note_to_midi(m.group(1)))
        except Exception:  # librosa raises ParameterError on bad names
            return None
        if 0 <= midi <= 127:
            return midi
    return None


# ── builders ──────────────────────────────────────────────────────────────────


def _validate_pitch_range(pitch_range: tuple[int, int]) -> tuple[int, int]:
    lo, hi = (int(p) for p in pitch_range)
    if not 0 <= lo <= hi <= 127:
        raise ParameterError(
            f"Invalid pitch range {lo}..{hi}: need 0 <= min <= max <= 127."
        )
    return lo, hi


def _normalise_columns(W: np.ndarray) -> np.ndarray:
    return W / (np.linalg.norm(W, axis=0, keepdims=True) + EPS)


def build_from_audio_dir(
    audio_dir: str | Path,
    pitch_range: tuple[int, int] = (21, 108),
    atoms_per_pitch: int = 2,
    config: SpectrogramConfig | None = None,
    max_iter: int = 200,
    log=None,
) -> Dictionary:
    """Supervised dictionary from isolated single-note recordings.

    Files are matched to pitches by name (see ``pitch_from_filename``); several
    files for one pitch (e.g. MAPS dynamics F/M/P) are pooled. For each pitch
    the clips' spectrograms are concatenated and factorised with Euclidean NMF
    into ``atoms_per_pitch`` unit-norm spectral atoms, which form that pitch's
    group of columns in W.

    ``log`` is an optional callable receiving progress strings.
    """
    config = config or SpectrogramConfig()
    lo, hi = _validate_pitch_range(pitch_range)
    if atoms_per_pitch < 1:
        raise ParameterError("atoms_per_pitch must be >= 1.")
    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        raise AudioLoadError(f"Audio directory not found: {audio_dir}")
    files = sorted(
        p
        for p in audio_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise AudioLoadError(
            f"No audio files ({', '.join(SUPPORTED_EXTENSIONS)}) in {audio_dir}."
        )

    by_pitch: dict[int, list[Path]] = defaultdict(list)
    unparsed = []
    for f in files:
        midi = pitch_from_filename(f.stem)
        if midi is None:
            unparsed.append(f.name)
        elif lo <= midi <= hi:
            by_pitch[midi].append(f)
    if unparsed and log:
        log(
            f"Skipped {len(unparsed)} file(s) with no pitch in the name, "
            f"e.g. {unparsed[0]}"
        )
    if not by_pitch:
        raise AudioLoadError(
            f"None of the {len(files)} audio files in {audio_dir} is named with "
            f"a MIDI pitch inside {lo}..{hi} (expected e.g. "
            "'MAPS_ISOL_NO_F_S0_M60_AkPnBcht.wav', 'note_60.wav' or 'C4.wav')."
        )

    columns, pitch_map, skipped = [], [], []
    for midi in sorted(by_pitch):
        specs = []
        for f in by_pitch[midi]:
            try:
                y = load_audio(f, config.sr)
            except AudioLoadError as exc:
                skipped.append(f"{f.name} ({exc})")
                continue
            y, _ = librosa.effects.trim(y, top_db=60)
            V = spectrogram(y, config)
            if V.max() > 0:
                specs.append(V / V.max())
        if not specs:
            continue
        V = np.concatenate(specs, axis=1)
        k = min(atoms_per_pitch, V.shape[1])
        res = standard_nmf(V, n_components=k, max_iter=max_iter, seed=midi)
        # Order atoms by total activation so atom 0 is the dominant spectrum.
        order = np.argsort(-res.H.sum(axis=1))
        atoms = _normalise_columns(res.W[:, order])
        if k < atoms_per_pitch:  # very short clip: repeat the dominant atom
            atoms = np.repeat(atoms[:, :1], atoms_per_pitch, axis=1)
        start = sum(a.shape[1] for a in columns)
        columns.append(atoms)
        pitch_map.append(
            {
                "midi": midi,
                "note": librosa.midi_to_note(midi, unicode=False),
                "columns": list(range(start, start + atoms_per_pitch)),
                "sources": [f.name for f in by_pitch[midi]],
            }
        )
        if log:
            log(
                f"  MIDI {midi:3d} ({pitch_map[-1]['note']:>4}): "
                f"{len(specs)} clip(s), {atoms_per_pitch} atom(s)"
            )
    for s in skipped:
        if log:
            log(f"  Skipped unreadable file: {s}")
    if not columns:
        raise AudioLoadError(f"No readable audio files in {audio_dir}.")

    return Dictionary(
        W=np.concatenate(columns, axis=1),
        config=config,
        pitch_map=pitch_map,
        extra={
            "source": "supervised",
            "audio_dir": str(audio_dir),
            "atoms_per_pitch": atoms_per_pitch,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )


def build_harmonic_dictionary(
    pitch_range: tuple[int, int] = (21, 108),
    atoms_per_pitch: int = 2,
    config: SpectrogramConfig | None = None,
    n_harmonics: int = 20,
    inharmonicity: float = 1e-4,
) -> Dictionary:
    """Synthetic dictionary when no recordings are available.

    Each atom is a harmonic comb in the CQT domain: Gaussian peaks (one bin
    wide) at the stretched partials f_h = h f0 sqrt(1 + B h^2) of a piano
    string. Atom ``a`` of a pitch uses amplitudes h^-(0.5 + a), so the group
    spans brighter (attack) and duller (decay) timbres.
    """
    config = config or SpectrogramConfig()
    lo, hi = _validate_pitch_range(pitch_range)
    if atoms_per_pitch < 1 or n_harmonics < 1:
        raise ParameterError("atoms_per_pitch and n_harmonics must be >= 1.")
    bins = np.arange(config.n_bins)
    f_top = config.fmin * 2 ** ((config.n_bins - 1) / config.bins_per_octave)
    columns, pitch_map = [], []
    for midi in range(lo, hi + 1):
        f0 = float(librosa.midi_to_hz(midi))
        h = np.arange(1, n_harmonics + 1)
        fh = h * f0 * np.sqrt(1 + inharmonicity * h**2)
        keep = (fh >= config.fmin) & (fh <= f_top)
        if not np.any(keep):
            continue
        pos = config.bins_per_octave * np.log2(fh[keep] / config.fmin)
        peaks = np.exp(-0.5 * (bins[:, None] - pos[None, :]) ** 2)
        atoms = np.stack(
            [peaks @ h[keep] ** -(0.5 + a) for a in range(atoms_per_pitch)], axis=1
        )
        start = len(pitch_map) * atoms_per_pitch
        columns.append(_normalise_columns(atoms))
        pitch_map.append(
            {
                "midi": midi,
                "note": librosa.midi_to_note(midi, unicode=False),
                "columns": list(range(start, start + atoms_per_pitch)),
            }
        )
    if not columns:
        raise ParameterError("No pitch in the range falls inside the CQT range.")
    return Dictionary(
        W=np.concatenate(columns, axis=1),
        config=config,
        pitch_map=pitch_map,
        extra={
            "source": "harmonic",
            "atoms_per_pitch": atoms_per_pitch,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
