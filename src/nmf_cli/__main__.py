"""
src/nmf_cli/__main__.py
─────────────────────────────────────────────────────────────────
NMF Transcriber CLI

Entry points
────────────
  python -m nmf_cli build-dict   --audio-dir DIR --output dict.npy
  python -m nmf_cli transcribe   --input audio.wav --dict dict.npy
                                 --method [standard|group-sparse|compare]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

# ── app ───────────────────────────────────────────────────────────────────────
app = typer.Typer(
    name="nmf-transcriber",
    help="NMF Polyphonic Piano Transcription — Standard and Group-Sparse",
    add_completion=False,
)

VALID_METHODS = ("standard", "group-sparse", "compare")


# ── helpers ───────────────────────────────────────────────────────────────────


def _err(msg: str) -> None:
    """Print a red error and exit with code 1."""
    typer.echo(typer.style(f"\n  ERROR: {msg}", fg=typer.colors.RED, bold=True))
    raise typer.Exit(code=1)


def _ok(msg: str) -> None:
    typer.echo(typer.style(f"  ok  {msg}", fg=typer.colors.GREEN))


def _info(msg: str) -> None:
    typer.echo(f"  {msg}")


def _validate_audio(path: Path) -> None:
    if not path.exists():
        _err(f"Audio file not found: {path}")
    if path.suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg"}:
        _err(f"Unsupported audio format: '{path.suffix}'. Use .wav / .flac / .mp3")


def _validate_dict_path(path: Path) -> None:
    if not path.exists():
        _err(
            f"Dictionary file not found: {path}\n"
            f"  Run:  python -m nmf_cli build-dict --output {path}"
        )
    meta = path.with_suffix(".json")
    if not meta.exists():
        _err(
            f"Dictionary metadata sidecar missing: {meta}\n"
            f"  The .json sidecar is required alongside the .npy file.\n"
            f"  Rebuild the dictionary to regenerate it."
        )


def _validate_hyperparams(sparsity_weight: float, threshold: float) -> None:
    if sparsity_weight < 0:
        _err(
            f"--sparsity-weight must be >= 0, got {sparsity_weight}.\n"
            f"  Typical value: 0.5 (O'Hanlon & Plumbley 2014)."
        )
    if not (0.0 < threshold < 1.0):
        _err(
            f"--threshold must be in (0, 1), got {threshold}.\n"
            f"  Typical value: 0.1 to 0.2."
        )


# ── build-dict command ────────────────────────────────────────────────────────


@app.command("build-dict")
def build_dict(
    audio_dir: Optional[Path] = typer.Option(
        None,
        "--audio-dir",
        "-d",
        help="Folder of isolated single-note WAV/FLAC recordings. "
        "Not required when --harmonic is used.",
    ),
    output: Path = typer.Option(
        Path("dict.npy"),
        "--output",
        "-o",
        help="Output path for the dictionary .npy file.",
    ),
    pitch_min: int = typer.Option(
        21,
        "--pitch-min",
        help="Minimum MIDI pitch to include (default 21 = A0).",
    ),
    pitch_max: int = typer.Option(
        108,
        "--pitch-max",
        help="Maximum MIDI pitch to include (default 108 = C8).",
    ),
    n_fft: int = typer.Option(
        2048,
        "--n-fft",
        help="FFT window size.",
    ),
    hop_length: int = typer.Option(
        512,
        "--hop-length",
        help="STFT hop size.",
    ),
    sr: int = typer.Option(
        22050,
        "--sr",
        help="Target sample rate.",
    ),
    harmonic: bool = typer.Option(
        False,
        "--harmonic",
        help="Build a harmonic (physics-based) dictionary instead of "
        "extracting from audio. Does not require --audio-dir.",
    ),
    n_harmonics: int = typer.Option(
        6,
        "--n-harmonics",
        help="Harmonics per note (harmonic mode only).",
    ),
) -> None:
    """
    Build a spectral dictionary from single-note recordings or harmonic physics.

    \b
    Examples
    --------
    # From single-note recordings (supervised):
      python -m nmf_cli build-dict --audio-dir ./notes --output dict.npy

    # From harmonic physics (fast, no audio needed):
      python -m nmf_cli build-dict --harmonic --output dict_harmonic.npy
    """
    typer.echo(
        typer.style("\nNMF Transcriber — build-dict", fg=typer.colors.CYAN, bold=True)
    )

    # Validate pitch range
    if pitch_min < 0 or pitch_max > 127 or pitch_min >= pitch_max:
        _err(
            f"Invalid pitch range: {pitch_min}–{pitch_max}. "
            f"Use values between 0 and 127 with min < max."
        )

    if harmonic:
        # ── Harmonic (physics-based) dictionary ──────────────────────────────
        _info(
            f"Mode: harmonic  (MIDI {pitch_min}–{pitch_max}, "
            f"{n_harmonics} harmonics/note)"
        )
        _info(f"Output: {output}")

        import librosa
        import numpy as np

        from nmf_core.dictionary import (
            build_harmonic_dictionary,
            save_dictionary,
        )

        # Build a dummy freq axis matching the requested n_fft/sr
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        D, groups, pitch_map = build_harmonic_dictionary(
            freqs,
            midi_low=pitch_min,
            midi_high=pitch_max,
            n_harmonics=n_harmonics,
        )

        # Flatten groups into a single W-style matrix for the transcriber:
        # each NOTE gets one column (the sum of its harmonic atoms).
        # The full atom matrix D is stored separately in the sidecar groups.
        # For the CLI we expose one column per note (collapsed).
        n_notes = len(groups)
        W = np.zeros((D.shape[0], n_notes), dtype=np.float64)
        for ni, idx in enumerate(groups):
            W[:, ni] = D[:, idx].sum(axis=1)
        # Normalise each column
        col_max = W.max(axis=0, keepdims=True) + 1e-10
        W /= col_max

        save_dictionary(
            W,
            output,
            n_fft=n_fft,
            sr=sr,
            hop_length=hop_length,
            pitch_map=pitch_map,
        )

    else:
        # ── Supervised (from audio files) ────────────────────────────────────
        if not audio_dir.exists():
            _err(f"Audio directory not found: {audio_dir}")

        _info(f"Mode: supervised  (audio dir: {audio_dir})")
        _info(f"Pitch range: MIDI {pitch_min}–{pitch_max}")
        _info(f"Output: {output}")

        from nmf_core.dictionary import build_from_audio_dir

        try:
            build_from_audio_dir(
                audio_dir=audio_dir,
                output_path=output,
                pitch_range=(pitch_min, pitch_max),
                n_fft=n_fft,
                hop_length=hop_length,
                sr=sr,
                verbose=True,
            )
        except (FileNotFoundError, ValueError) as exc:
            _err(str(exc))

    _ok(f"Dictionary ready -> {output}")
    _ok(f"Metadata sidecar -> {output.with_suffix('.json')}")


# ── transcribe command ─────────────────────────────────────────────────────────


@app.command("transcribe")
def transcribe(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to the polyphonic audio file (.wav / .flac).",
    ),
    dict_path: Optional[Path] = typer.Option(
        None,
        "--dict",
        help="Pre-built dictionary .npy file. "
        "If omitted, a harmonic dictionary is generated automatically.",
    ),
    method: str = typer.Option(
        "compare",
        "--method",
        "-m",
        help="Transcription method: standard | group-sparse | compare.",
    ),
    sparsity_weight: float = typer.Option(
        0.5,
        "--sparsity-weight",
        help="Group sparsity penalty λ (group-sparse / compare only).",
    ),
    threshold: float = typer.Option(
        0.15,
        "--threshold",
        help="Activation threshold for note onset detection [0..1].",
    ),
    out_dir: Path = typer.Option(
        Path("outputs"),
        "--out-dir",
        help="Destination directory for .mid files and logs.",
    ),
    K: int = typer.Option(
        12,
        "--K",
        help="Number of components (standard method, unsupervised only).",
    ),
    n_iter: int = typer.Option(
        200,
        "--n-iter",
        help="Maximum NMF iterations.",
    ),
    beta: float = typer.Option(
        0.5,
        "--beta",
        help="beta-divergence parameter for group-sparse (default 0.5).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print iteration-level progress.",
    ),
) -> None:
    """
    Transcribe a polyphonic audio file to MIDI.

    \b
    Examples
    --------
    # Compare both methods:
      python -m nmf_cli transcribe --input piano.wav --method compare

    # Standard NMF only, with pre-built dictionary:
      python -m nmf_cli transcribe --input piano.wav --dict dict.npy --method standard

    # Group-sparse only:
      python -m nmf_cli transcribe --input piano.wav --method group-sparse
    """
    typer.echo(
        typer.style("\nNMF Transcriber — transcribe", fg=typer.colors.CYAN, bold=True)
    )

    # ── Input validation ──────────────────────────────────────────────────────
    _validate_audio(input_path)
    if method not in VALID_METHODS:
        _err(f"Unknown method '{method}'. Choose from: {', '.join(VALID_METHODS)}")
    _validate_hyperparams(sparsity_weight, threshold)
    if dict_path is not None:
        _validate_dict_path(dict_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    _info(f"Input  : {input_path}")
    _info(f"Method : {method}")
    _info(f"Output : {out_dir}")

    # ── Load audio ────────────────────────────────────────────────────────────
    _info("\nLoading audio...")
    from nmf_core.audio import load_spectrogram, validate_dimensions

    try:
        spec = load_spectrogram(input_path, normalize=True)
    except (FileNotFoundError, ValueError) as exc:
        _err(str(exc))

    _info(f"  V shape: {spec.V.shape[0]} freq bins × {spec.V.shape[1]} frames")
    _info(f"  Duration: {spec.duration_s:.2f}s  SR: {spec.sr} Hz")

    # ── Load or build dictionary ──────────────────────────────────────────────
    import numpy as np

    from nmf_core.dictionary import (
        build_harmonic_dictionary,
        load_dictionary,
        save_dictionary,
    )

    if dict_path is not None:
        _info("\nLoading dictionary...")
        try:
            W_dict, meta = load_dictionary(dict_path)
        except (FileNotFoundError, ValueError) as exc:
            _err(str(exc))

        # Verify dimensions match
        try:
            validate_dimensions(spec.V, W_dict)
        except ValueError as exc:
            _err(str(exc))

        pitch_map = meta["pitch_map"]
        K_dict = W_dict.shape[1]
        _info(f"  Dictionary: {W_dict.shape[0]}×{W_dict.shape[1]}")

    else:
        _info("\nNo --dict provided — building harmonic dictionary...")
        freqs = spec.freqs
        D_full, groups_full, pitch_map_full = build_harmonic_dictionary(
            freqs, midi_low=48, midi_high=84, n_harmonics=6
        )
        # Collapse to one column per note for standard method
        n_notes = len(groups_full)
        W_dict = np.zeros((D_full.shape[0], n_notes), dtype=np.float64)
        for ni, idx in enumerate(groups_full):
            W_dict[:, ni] = D_full[:, idx].sum(axis=1)
        col_max = W_dict.max(axis=0, keepdims=True) + 1e-10
        W_dict /= col_max
        K_dict = n_notes
        pitch_map = pitch_map_full

        # Save auto-generated dict for reference
        auto_dict_path = out_dir / f"{stem}_auto_dict.npy"
        save_dictionary(
            W_dict,
            auto_dict_path,
            n_fft=spec.n_fft,
            sr=spec.sr,
            hop_length=spec.hop_length,
            pitch_map=pitch_map,
        )
        _info(f"  Auto dictionary saved -> {auto_dict_path}")

    # ── Run Standard NMF ──────────────────────────────────────────────────────
    result_std = None
    result_gs = None
    pitch_map_std = pitch_map

    if method in ("standard", "compare"):
        _info("\nRunning Standard Euclidean NMF...")
        from nmf_core.standard_nmf import run_nmf

        result_std = run_nmf(
            spec.V,
            K=K_dict,
            W_init=W_dict,  # supervised: fix W to dictionary
            n_iter=n_iter,
            verbose=verbose,
        )
        _info(
            f"  Converged: {result_std.converged}  "
            f"({result_std.n_iter} iters, "
            f"error reduction{result_std.error_reduction_pct:.1f}%)"
        )

        # MIDI
        mid_std = out_dir / f"{stem}_standard.mid"
        from nmf_core.midi import activations_to_midi, save_activations_npz

        activations_to_midi(
            H=result_std.H,
            pitch_map=pitch_map_std,
            sr=spec.sr,
            hop_length=spec.hop_length,
            out_path=mid_std,
            threshold=threshold,
        )
        save_activations_npz(
            result_std.H,
            pitch_map_std,
            out_dir / f"{stem}_standard_H.npz",
        )
        _ok(f"Standard MIDI -> {mid_std}")

    # ── Run Group-Sparse NMF ─────────────────────────────────────────────────
    if method in ("group-sparse", "compare"):
        _info("\nRunning Group-Sparse beta-NMF...")
        from nmf_core.dictionary import build_harmonic_dictionary
        from nmf_core.group_sparse_nmf import group_sparse_nmd

        # Always use full harmonic dictionary for group-sparse
        freqs_gs = spec.freqs
        D_gs, groups_gs, pm_gs = build_harmonic_dictionary(
            freqs_gs, midi_low=48, midi_high=84, n_harmonics=6
        )
        midi_notes_gs = [p["midi_note"] for p in pm_gs]
        note_names_gs = [p["note_name"] for p in pm_gs]

        result_gs = group_sparse_nmd(
            S=spec.V,
            D=D_gs,
            groups=groups_gs,
            beta=beta,
            lam=sparsity_weight,
            verbose=verbose,
        )
        stats = result_gs.sparsity_stats()
        _info(f"  Converged: {result_gs.converged}  ({result_gs.n_iter} iters)")
        _info(
            f"  Sparsity: {stats['sparsity_pct']:.1f}%  "
            f"Mean active notes/frame: {stats['mean_active_per_frame']:.2f}"
        )

        # Collapse to piano roll and write MIDI
        roll = result_gs.piano_roll()
        mid_gs = out_dir / f"{stem}_group_sparse.mid"
        from nmf_core.midi import piano_roll_to_midi, save_activations_npz

        piano_roll_to_midi(
            roll=roll,
            note_names=note_names_gs,
            midi_notes=midi_notes_gs,
            sr=spec.sr,
            hop_length=spec.hop_length,
            out_path=mid_gs,
            threshold=threshold,
        )
        save_activations_npz(
            roll,
            pm_gs,
            out_dir / f"{stem}_group_sparse_H.npz",
        )
        _ok(f"Group-Sparse MIDI -> {mid_gs}")

    # ── Write summary.json ────────────────────────────────────────────────────
    from nmf_core.midi import build_summary

    summary = build_summary(
        method=method,
        audio_path=input_path,
        out_dir=out_dir,
        result_std=result_std,
        result_gs=result_gs,
        threshold=threshold,
        sparsity_weight=sparsity_weight,
    )
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _ok(f"Summary log -> {summary_path}")

    typer.echo(
        typer.style("\n  Transcription complete!", fg=typer.colors.CYAN, bold=True)
    )


# ── evaluate command ──────────────────────────────────────────────────────────


@app.command("evaluate")
def evaluate(
    predicted: Path = typer.Option(
        ...,
        "--predicted",
        "-p",
        help="Predicted .mid file output from transcribe.",
    ),
    reference: Path = typer.Option(
        ...,
        "--reference",
        "-r",
        help="Reference (ground-truth) .mid file.",
    ),
    compare_std: Optional[Path] = typer.Option(
        None,
        "--compare-std",
        help="Also evaluate a standard .mid alongside the predicted file.",
    ),
    out_json: Optional[Path] = typer.Option(
        None,
        "--out-json",
        help="Save evaluation scores to this JSON file.",
    ),
) -> None:
    """
    Evaluate a transcribed MIDI against a ground-truth reference.

    \b
    Examples
    --------
    # Evaluate one file:
      python -m nmf_cli evaluate --predicted out/piano_gs.mid --reference gt.mid

    # Compare both methods against the same reference:
      python -m nmf_cli evaluate \\
          --predicted out/piano_group_sparse.mid \\
          --compare-std out/piano_standard.mid \\
          --reference gt.mid --out-json eval.json
    """
    typer.echo(
        typer.style("\nNMF Transcriber — evaluate", fg=typer.colors.CYAN, bold=True)
    )

    for p, label in [(predicted, "predicted"), (reference, "reference")]:
        if not p.exists():
            _err(f"{label} file not found: {p}")

    import json

    from nmf_core.evaluate import (
        compare_methods,
        evaluate_midi,
        print_report,
        save_report_json,
    )

    if compare_std is not None:
        if not compare_std.exists():
            _err(f"--compare-std file not found: {compare_std}")

        _info("Evaluating both Standard and Group-Sparse against reference...")
        scores = compare_methods(compare_std, predicted, reference)

        # Pretty print both
        for key, label in [
            ("standard", "Standard NMF"),
            ("group_sparse", "Group-Sparse NMF"),
        ]:
            s = scores[key]
            fr = s["frame"]
            no = s["note_onset"]
            typer.echo(f"\n  -- {label} --")
            typer.echo(
                f"  Notes   : {s['n_pred_notes']} predicted"
                f" / {s['n_ref_notes']} reference"
            )
            typer.echo(
                f"  Frame P/R/F: "
                f"{fr['precision']:.3f} / {fr['recall']:.3f}"
                f" / {fr['f_measure']:.3f}"
            )
            typer.echo(
                f"  Note  P/R/F: "
                f"{no['precision']:.3f} / {no['recall']:.3f}"
                f" / {no['f_measure']:.3f}"
            )

        winner = scores["winner"]
        typer.echo(
            typer.style(
                f"\n  Frame F winner  : {winner['frame_f']}\n"
                f"  Note F winner   : {winner['note_onset_f']}",
                fg=typer.colors.GREEN,
            )
        )

        if out_json:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=2)
            _ok(f"Scores saved -> {out_json}")

    else:
        _info(f"Evaluating {predicted.name} against {reference.name}...")
        result = evaluate_midi(predicted, reference)
        print_report(result, label=predicted.stem)

        if out_json:
            save_report_json(result, out_json)
            _ok(f"Scores saved -> {out_json}")

    typer.echo(typer.style("  Evaluation complete!", fg=typer.colors.CYAN, bold=True))


# ── report command ────────────────────────────────────────────────────────────


@app.command("report")
def report(
    std_mid: Path = typer.Option(
        ...,
        "--std-mid",
        help="Standard NMF .mid file from transcribe.",
    ),
    gs_mid: Path = typer.Option(
        ...,
        "--gs-mid",
        help="Group-Sparse NMF .mid file from transcribe.",
    ),
    summary: Path = typer.Option(
        ...,
        "--summary",
        help="summary.json produced by transcribe.",
    ),
    out_html: Path = typer.Option(
        Path("outputs/report.html"),
        "--out",
        help="Output .html file path.",
    ),
    eval_json: Optional[Path] = typer.Option(
        None,
        "--eval-json",
        help="Optional evaluation scores JSON from evaluate command.",
    ),
) -> None:
    """
    Generate a self-contained HTML comparison report.

    \b
    Example
    -------
      python -m nmf_cli report \\
          --std-mid outputs/piano_standard.mid \\
          --gs-mid  outputs/piano_group_sparse.mid \\
          --summary outputs/summary.json \\
          --out     outputs/report.html
    """
    typer.echo(
        typer.style("\nNMF Transcriber — report", fg=typer.colors.CYAN, bold=True)
    )

    for p, label in [
        (std_mid, "--std-mid"),
        (gs_mid, "--gs-mid"),
        (summary, "--summary"),
    ]:
        if not p.exists():
            _err(f"{label} file not found: {p}")

    from nmf_core.report import generate_report

    try:
        generate_report(
            out_path=out_html,
            std_mid=std_mid,
            gs_mid=gs_mid,
            summary_json=summary,
            eval_json=eval_json if eval_json and eval_json.exists() else None,
        )
    except Exception as exc:
        _err(f"Report generation failed: {exc}")

    _ok(f"Report -> {out_html}")
    _info("Open the .html file in any browser — no internet required.")


# ── sheet command ─────────────────────────────────────────────────────────────


@app.command("sheet")
def sheet(
    midi_in: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input .mid file to convert.",
    ),
    out_xml: Path = typer.Option(
        Path("outputs/score.xml"),
        "--out",
        help="Output MusicXML .xml file path.",
    ),
    title: str = typer.Option(
        "NMF Transcription",
        "--title",
        help="Score title embedded in the MusicXML.",
    ),
    pdf: bool = typer.Option(
        False,
        "--pdf",
        help="Also attempt PDF export via MuseScore (must be on PATH).",
    ),
) -> None:
    """
    Export a MIDI file to MusicXML sheet music.

    Opens in MuseScore, Sibelius, Finale, or Flat.io.

    \b
    Example
    -------
      python -m nmf_cli sheet \\
          --input  outputs/piano_group_sparse.mid \\
          --out    outputs/score.xml
    """
    typer.echo(
        typer.style("\nNMF Transcriber — sheet", fg=typer.colors.CYAN, bold=True)
    )

    if not midi_in.exists():
        _err(f"MIDI file not found: {midi_in}")

    try:
        from nmf_core.sheet import midi_to_musicxml, midi_to_pdf

        if pdf:
            midi_to_pdf(midi_in, out_xml.with_suffix(".pdf"), title=title)
        else:
            midi_to_musicxml(midi_in, out_xml, title=title)

    except ImportError as exc:
        _err(str(exc))
    except Exception as exc:
        _err(f"Sheet export failed: {exc}")

    _ok(f"Sheet music -> {out_xml}")
    _info("Open in MuseScore (free) to view, edit, or print.")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
