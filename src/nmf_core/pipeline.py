"""End-to-end transcription: audio + dictionary -> MIDI, .npz and summary.json."""

from __future__ import annotations

import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .audio import load_spectrogram, validate_dimensions
from .dictionary import Dictionary, load_dictionary
from .errors import ParameterError
from .midi import (
    pitch_activations,
    refine_offsets,
    roll_to_notes,
    save_activations,
    write_midi,
)
from .nmf import NMFResult, group_sparse_nmf, hoyer_sparsity, standard_nmf

METHODS = ("standard", "group-sparse", "compare")
# Output-file suffix for each solver (the CLI's "group-sparse" -> "_group_sparse").
_SUFFIX = {"standard": "standard", "group-sparse": "group_sparse"}


def validate_params(
    method: str,
    sparsity_weight: float,
    threshold: float,
    beta: float = 2.0,
    max_iter: int = 500,
    tol: float = 1e-4,
    min_duration: float = 0.05,
) -> None:
    """Raise ParameterError naming the CLI flag of the first invalid value."""
    if method not in METHODS:
        raise ParameterError(
            f"--method must be one of {', '.join(METHODS)}; got '{method}'."
        )
    checks = [
        ("--sparsity-weight", sparsity_weight, sparsity_weight >= 0, ">= 0"),
        ("--threshold", threshold, 0 < threshold <= 1, "in (0, 1]"),
        ("--beta", beta, 0 <= beta <= 2, "in [0, 2]"),
        ("--max-iter", max_iter, max_iter >= 1, ">= 1"),
        ("--tol", tol, tol > 0, "> 0"),
        ("--min-duration", min_duration, min_duration >= 0, ">= 0"),
    ]
    for flag, value, ok, rule in checks:
        if not (isinstance(value, (int, float)) and math.isfinite(value) and ok):
            raise ParameterError(f"{flag} must be {rule}; got {value}.")


def transcribe(
    input_path: str | Path,
    dict_path: str | Path | Dictionary,
    out_dir: str | Path,
    method: str = "compare",
    sparsity_weight: float = 0.2,
    threshold: float = 0.1,
    beta: float = 2.0,
    max_iter: int = 500,
    tol: float = 1e-4,
    min_duration: float = 0.05,
    log=None,
) -> dict:
    """Transcribe ``input_path`` and write all artifacts into ``out_dir``.

    ``dict_path`` may also be an already loaded ``Dictionary``. Returns the
    summary that is written to ``out_dir/summary.json``. Raises a subclass of
    TranscriberError for any invalid input.
    """
    t_start = time.perf_counter()
    log = log or (lambda msg: None)
    validate_params(
        method, sparsity_weight, threshold, beta, max_iter, tol, min_duration
    )

    d = dict_path if isinstance(dict_path, Dictionary) else load_dictionary(dict_path)
    cfg = d.config
    log(
        f"Dictionary: {d.W.shape[0]} bins x {d.W.shape[1]} atoms, "
        f"{len(d.pitch_map)} pitches ({min(d.pitches)}-{max(d.pitches)})"
    )

    input_path = Path(input_path)
    V = load_spectrogram(input_path, cfg)
    validate_dimensions(V, d.W)
    duration = V.shape[1] * cfg.hop_length / cfg.sr
    log(f"Spectrogram V: {V.shape[0]} bins x {V.shape[1]} frames ({duration:.2f} s)")
    scale = float(V.max())
    if scale > 0:
        V = V / scale  # makes --sparsity-weight independent of input loudness
    else:
        log("Warning: the input is silent; the MIDI files will contain no notes.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    solvers = ["standard", "group-sparse"] if method == "compare" else [method]
    results: dict[str, dict] = {}
    for name in solvers:
        log(f"Running {name} NMF ...")
        if name == "standard":
            res = standard_nmf(V, d.W, max_iter=max_iter, tol=tol)
        else:
            res = group_sparse_nmf(
                V,
                d.W,
                d.groups,
                sparsity_weight=sparsity_weight,
                beta=beta,
                max_iter=max_iter,
                tol=tol,
            )
        results[name] = _write_outputs(
            res,
            V,
            d,
            out_dir,
            f"{stem}_{_SUFFIX[name]}",
            threshold,
            min_duration,
            refit_offsets=name == "group-sparse",
            max_iter=max_iter,
            tol=tol,
        )
        r = results[name]
        log(
            f"  {r['iterations']} iterations, converged={r['converged']}, "
            f"{r['n_notes']} notes -> {r['midi']}"
        )

    summary = {
        "tool": "nmf-transcriber",
        "version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "config": {
            "input": str(input_path),
            "dictionary": str(dict_path)
            if not isinstance(dict_path, Dictionary)
            else "<in-memory>",
            "out_dir": str(out_dir),
            "method": method,
            "sparsity_weight": sparsity_weight,
            "threshold": threshold,
            "beta": beta,
            "max_iter": max_iter,
            "tol": tol,
            "min_duration_s": min_duration,
            "spectrogram": cfg.to_dict(),
            "dictionary_shape": list(d.W.shape),
            "pitch_range": [min(d.pitches), max(d.pitches)],
        },
        "input": {
            "duration_s": round(duration, 3),
            "n_frames": int(V.shape[1]),
            "peak_magnitude": scale,
        },
        "results": {_SUFFIX[k]: v for k, v in results.items()},
    }
    if method == "compare":
        summary["comparison"] = _compare(results["standard"], results["group-sparse"])
    summary["processing_time_s"] = round(time.perf_counter() - t_start, 3)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _write_outputs(
    res: NMFResult,
    V: np.ndarray,
    d: Dictionary,
    out_dir: Path,
    name: str,
    threshold: float,
    min_duration: float,
    refit_offsets: bool = False,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> dict:
    cfg = d.config
    roll = pitch_activations(res.H, d.groups)
    notes = roll_to_notes(
        roll,
        d.pitches,
        cfg.sr,
        cfg.hop_length,
        threshold=threshold,
        min_duration=min_duration,
    )
    refit_info = None
    if refit_offsets and notes:
        # The l1,2 penalty also shrinks each note's quiet decaying tail to
        # zero, so notes would end too early. Keep the group-sparse decision
        # of which notes exist and when they start, but take each Note-Off
        # from an unpenalised re-fit restricted to the selected pitches
        # (a "relaxed" / debiased estimate).
        keep = sorted({n.pitch for n in notes})
        entries = [p for p in d.pitch_map if int(p["midi"]) in keep]
        cols = [c for p in entries for c in p["columns"]]
        groups = [[cols.index(c) for c in p["columns"]] for p in entries]
        refit = standard_nmf(V, d.W[:, cols], max_iter=max_iter, tol=tol)
        notes = refine_offsets(
            notes,
            pitch_activations(refit.H, groups),
            [int(p["midi"]) for p in entries],
            cfg.sr,
            cfg.hop_length,
            off_level=threshold * 0.5,
        )
        refit_info = {
            "description": "note offsets from an unpenalised refit on the "
            "selected pitches",
            "n_pitches": len(keep),
            "iterations": refit.n_iter,
            "converged": refit.converged,
            "solve_time_s": round(refit.elapsed_s, 3),
        }
    midi_path = write_midi(notes, out_dir / f"{name}.mid")
    npz_path = save_activations(
        out_dir / f"{name}_H.npz",
        res.H,
        roll,
        d.pitches,
        d.groups,
        cfg.sr,
        cfg.hop_length,
        notes,
    )
    return {
        "midi": str(midi_path),
        "activations": str(npz_path),
        "converged": res.converged,
        "iterations": res.n_iter,
        "max_iterations": res.max_iter,
        "solve_time_s": round(res.elapsed_s, 3),
        "final_cost": res.final_cost,
        "relative_reconstruction_error": round(res.reconstruction_error(V), 6),
        "n_notes": len(notes),
        "pitches_detected": sorted({n.pitch for n in notes}),
        "pitch_roll_hoyer_sparsity": round(hoyer_sparsity(roll), 4),
        "mean_active_pitches_per_frame": round(
            float((roll / max(roll.max(), 1e-12) >= threshold).sum(axis=0).mean()), 4
        ),
        "solver": res.params,
        **({"offset_refit": refit_info} if refit_info else {}),
    }


def _compare(std: dict, gs: dict) -> dict:
    a, b = set(std["pitches_detected"]), set(gs["pitches_detected"])
    return {
        "n_notes": {"standard": std["n_notes"], "group_sparse": gs["n_notes"]},
        "pitch_roll_hoyer_sparsity": {
            "standard": std["pitch_roll_hoyer_sparsity"],
            "group_sparse": gs["pitch_roll_hoyer_sparsity"],
        },
        "mean_active_pitches_per_frame": {
            "standard": std["mean_active_pitches_per_frame"],
            "group_sparse": gs["mean_active_pitches_per_frame"],
        },
        "relative_reconstruction_error": {
            "standard": std["relative_reconstruction_error"],
            "group_sparse": gs["relative_reconstruction_error"],
        },
        "pitches_only_in_standard": sorted(a - b),
        "pitches_only_in_group_sparse": sorted(b - a),
    }
