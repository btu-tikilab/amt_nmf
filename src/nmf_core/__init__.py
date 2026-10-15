"""nmf_core: NMF-based polyphonic piano transcription (no UI code)."""

__version__ = "1.0.0"

from .audio import SpectrogramConfig, load_spectrogram, validate_dimensions  # noqa: E402
from .dictionary import (  # noqa: E402
    Dictionary,
    build_from_audio_dir,
    build_harmonic_dictionary,
    load_dictionary,
    save_dictionary,
)
from .errors import (  # noqa: E402
    AudioLoadError,
    DictionaryError,
    DimensionMismatchError,
    ParameterError,
    TranscriberError,
)
from .midi import roll_to_notes, write_midi  # noqa: E402
from .nmf import NMFResult, group_sparse_nmf, standard_nmf  # noqa: E402
from .pipeline import transcribe  # noqa: E402

__all__ = [
    "AudioLoadError",
    "Dictionary",
    "DictionaryError",
    "DimensionMismatchError",
    "NMFResult",
    "ParameterError",
    "SpectrogramConfig",
    "TranscriberError",
    "build_from_audio_dir",
    "build_harmonic_dictionary",
    "group_sparse_nmf",
    "load_dictionary",
    "load_spectrogram",
    "roll_to_notes",
    "save_dictionary",
    "standard_nmf",
    "transcribe",
    "validate_dimensions",
    "write_midi",
]
