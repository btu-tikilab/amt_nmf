"""nmf_core — NMF Transcription Backend (no UI code)."""

from .audio import AudioSpec, load_spectrogram, validate_dimensions
from .dictionary import (
    build_from_audio_dir,
    build_harmonic_dictionary,
    load_dictionary,
    save_dictionary,
)
from .evaluate import (
    EvalResult,
    FrameScores,
    NoteScores,
    compare_methods,
    evaluate_midi,
    print_report,
    save_report_json,
)
from .group_sparse_nmf import GroupSparseResult, compute_psi, group_sparse_nmd
from .midi import (
    activations_to_midi,
    build_summary,
    piano_roll_to_midi,
    save_activations_npz,
)
from .report import generate_report
from .sheet import midi_to_musicxml, midi_to_pdf
from .standard_nmf import NMFResult, run_nmf

__all__ = [
    "AudioSpec", "load_spectrogram", "validate_dimensions",
    "NMFResult", "run_nmf",
    "GroupSparseResult", "group_sparse_nmd", "compute_psi",
    "build_from_audio_dir", "build_harmonic_dictionary",
    "load_dictionary", "save_dictionary",
    "activations_to_midi", "piano_roll_to_midi",
    "save_activations_npz", "build_summary",
    "EvalResult", "FrameScores", "NoteScores",
    "evaluate_midi", "compare_methods",
    "print_report", "save_report_json",
    "generate_report",
    "midi_to_musicxml", "midi_to_pdf",
]
