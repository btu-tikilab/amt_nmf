"""
src/nmf_core/midi.py
─────────────────────────────────────────────────────────────────
Post-processing: convert activation matrix H -> playable MIDI file.

Pipeline
────────
H (K × T)  ->  threshold  ->  detect note-on/off events
             ->  frame_to_time  ->  pretty_midi.Note objects
             ->  .mid file

Threshold strategy
──────────────────
Each H row is normalised to [0,1] independently.
A note is ON when the normalised activation exceeds `threshold`.
Contiguous ON frames form one MIDI note.
Minimum note duration of 30 ms is enforced to avoid spurious blips.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pretty_midi


def activations_to_midi(
    H: np.ndarray,
    pitch_map: List[Dict],
    sr: int,
    hop_length: int,
    out_path: str | Path,
    threshold: float = 0.15,
    min_note_dur_s: float = 0.03,
    tempo: float = 120.0,
    min_row_peak_ratio: float = 0.0,
    release_threshold: float | None = None,
) -> pretty_midi.PrettyMIDI:
    """
    Convert activation matrix H to a MIDI file.

    Parameters
    ----------
    H           : (K, T) activation matrix (rows = components)
    pitch_map   : list of {"component": k, "midi_note": m, "note_name": n}
    sr          : sample rate used during analysis
    hop_length  : STFT hop size used during analysis
    out_path    : where to write the .mid file
    threshold   : normalised activation threshold for note detection [0..1]
    min_note_dur_s : minimum note duration in seconds (shorter notes dropped)
    tempo       : BPM for the output MIDI file

    Returns
    -------
    pretty_midi.PrettyMIDI object (also saved to out_path)
    """

    H = np.asarray(H, dtype=np.float64)

    if H.ndim != 2:
        raise ValueError("H must be a 2-D activation matrix.")
    if not np.all(np.isfinite(H)):
        raise ValueError("H must contain only finite values.")
    if np.any(H < 0):
        raise ValueError("H must be nonnegative.")

    if (
        not isinstance(sr, (int, np.integer))
        or isinstance(sr, (bool, np.bool_))
        or sr <= 0
    ):
        raise ValueError("sr must be a positive integer.")
    if (
        not isinstance(hop_length, (int, np.integer))
        or isinstance(hop_length, (bool, np.bool_))
        or hop_length <= 0
    ):
        raise ValueError("hop_length must be a positive integer.")

    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in the range (0, 1].")
    if not np.isfinite(min_note_dur_s) or min_note_dur_s < 0:
        raise ValueError("min_note_dur_s must be finite and nonnegative.")
    if not np.isfinite(tempo) or tempo <= 0:
        raise ValueError("tempo must be positive and finite.")
    if not np.isfinite(min_row_peak_ratio) or min_row_peak_ratio < 0:
        raise ValueError("min_row_peak_ratio must be finite and nonnegative.")

    if release_threshold is not None and (
        not np.isfinite(release_threshold) or not 0 <= release_threshold <= threshold
    ):
        raise ValueError("release_threshold must be between 0 and threshold.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    frame_dur = hop_length / sr  # seconds per frame
    global_max = float(np.max(H)) if H.size else 0.0

    release = threshold if release_threshold is None else release_threshold
    if not 0.0 <= release <= threshold:
        raise ValueError("release_threshold must be between 0 and threshold.")

    for entry in pitch_map:
        k = entry["component"]
        midi_note = entry["midi_note"]

        if k >= H.shape[0]:
            continue

        row = H[k, :].astype(np.float64)

        # Normalize each pitch independently.
        row_max = float(row.max())
        if row_max < 1e-10:
            continue

        row_norm = row / row_max
        global_cutoff = global_max * min_row_peak_ratio
        if global_max <= 0 or row_max < global_cutoff:
            continue

        # A frame must pass both the per-pitch threshold and the
        # global activation threshold.
        active = np.zeros(row_norm.shape, dtype=bool)
        is_active = False

        for frame, value in enumerate(row_norm):
            if not is_active and value >= threshold:
                is_active = True
            elif is_active and value < release:
                is_active = False

            active[frame] = is_active and row[frame] >= global_cutoff

        # Detect contiguous active regions
        changes = np.diff(active.astype(int), prepend=0, append=0)
        onsets = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]

        for on_frame, off_frame in zip(onsets, offsets):
            start_s = float(on_frame * frame_dur)
            end_s = float(off_frame * frame_dur)
            dur = end_s - start_s

            if dur < min_note_dur_s:
                continue

            # MIDI velocity proportional to mean activation in the window
            mean_act = float(row_norm[on_frame:off_frame].mean())
            velocity = int(np.clip(mean_act * 127, 20, 127))

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=int(midi_note),
                start=start_s,
                end=end_s,
            )
            piano.notes.append(note)

    pm.instruments.append(piano)
    pm.write(str(out_path))

    n_notes = len(piano.notes)
    print(f"  MIDI saved -> {out_path}  ({n_notes} notes)")
    return pm


def piano_roll_to_midi(
    roll: np.ndarray,
    note_names: List[str],
    midi_notes: List[int],
    sr: int,
    hop_length: int,
    out_path: str | Path,
    threshold: float = 0.15,
    min_note_dur_s: float = 0.03,
    tempo: float = 120.0,
    min_row_peak_ratio: float = 0.0,
) -> pretty_midi.PrettyMIDI:
    """
    Convert a note-level piano roll (n_notes × T) to MIDI.
    Used by the Group-Sparse NMF path where the roll is already
    collapsed to one row per note.
    """
    # Build a synthetic pitch_map from the note list
    pitch_map = [
        {"component": i, "midi_note": m, "note_name": n}
        for i, (m, n) in enumerate(zip(midi_notes, note_names))
    ]
    return activations_to_midi(
        H=roll,
        pitch_map=[
            {
                "component": i,
                "midi_note": m,
                "note_name": n,
            }
            for i, (m, n) in enumerate(zip(midi_notes, note_names))
        ],
        sr=sr,
        hop_length=hop_length,
        out_path=out_path,
        threshold=threshold,
        min_note_dur_s=min_note_dur_s,
        tempo=tempo,
        min_row_peak_ratio=min_row_peak_ratio,
    )


def save_activations_npz(
    H: np.ndarray,
    pitch_map: List[Dict],
    out_path: str | Path,
) -> None:
    """Save H matrix + pitch metadata as .npz for inspection."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(out_path),
        H=H,
        pitch_map=np.array([json.dumps(p) for p in pitch_map], dtype=object),
    )
    print(f"  Activations saved -> {out_path}")


def build_summary(
    method: str,
    audio_path: str,
    out_dir: str,
    result_std=None,
    result_gs=None,
    threshold: float = 0.15,
    sparsity_weight: float = 0.5,
) -> dict:
    """Build the execution log summary.json."""
    summary: dict = {
        "method": method,
        "audio_input": str(audio_path),
        "output_dir": str(out_dir),
        "threshold": threshold,
        "sparsity_weight": sparsity_weight,
        "results": {},
    }

    if result_std is not None:
        summary["results"]["standard"] = {
            "converged": result_std.converged,
            "n_iter": result_std.n_iter,
            "final_error": round(result_std.final_error, 6),
            "error_reduction_pct": round(result_std.error_reduction_pct, 2),
            "elapsed_s": round(result_std.elapsed_s, 3),
            "K": result_std.K,
        }

    if result_gs is not None:
        stats = result_gs.sparsity_stats()
        summary["results"]["group_sparse"] = {
            "converged": result_gs.converged,
            "n_iter": result_gs.n_iter,
            "final_cost": round(result_gs.cost_history[-1], 6)
            if result_gs.cost_history
            else None,
            "elapsed_s": round(result_gs.elapsed_s, 3),
            "beta": result_gs.beta,
            "lam": result_gs.lam,
            "q": result_gs.q,
            "mean_active_per_frame": round(stats["mean_active_per_frame"], 4),
            "sparsity_pct": round(stats["sparsity_pct"], 2),
        }

    return summary
