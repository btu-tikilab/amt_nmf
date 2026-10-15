"""
NMF Transcriber CLI

Commands:
  build-dict   Build a spectral dictionary.
  transcribe   Transcribe audio to MIDI.
  evaluate     Evaluate predicted MIDI against reference MIDI.
  report       Generate an HTML comparison report.
  sheet        Export MIDI to MusicXML.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="nmf-transcriber",
    help="NMF Polyphonic Piano Transcription — Standard and Group-Sparse",
    add_completion=False,
)

VALID_METHODS = ("standard", "group-sparse", "compare")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _err(msg: str) -> None:
    """Print an error and exit with code 1."""
    typer.echo(typer.style(f"\n  ERROR: {msg}", fg=typer.colors.RED, bold=True))
    raise typer.Exit(code=1)


def _ok(msg: str) -> None:
    typer.echo(typer.style(f"  ok  {msg}", fg=typer.colors.GREEN))


def _info(msg: str) -> None:
    typer.echo(msg)


def _validate_audio(path: Path) -> None:
    if not path.exists():
        _err(f"Audio file not found: {path}")

    if path.suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg"}:
        _err(
            f"Unsupported audio format: '{path.suffix}'. "
            "Use .wav / .flac / .mp3 / .ogg."
        )


def _validate_dict_path(path: Path) -> None:
    if not path.exists():
        _err(
            f"Dictionary file not found: {path}\n"
            f"  Run: nmf-transcriber build-dict --output {path}"
        )

    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        _err(
            f"Dictionary metadata sidecar missing: {metadata_path}\n"
            "  Rebuild the dictionary to regenerate it."
        )


def _validate_hyperparams(
    sparsity_weight: float,
    threshold: float,
) -> None:
    if not math.isfinite(sparsity_weight) or sparsity_weight < 0:
        _err(f"--sparsity-weight must be finite and >= 0, got {sparsity_weight}.")

    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        _err(f"--threshold must be finite and in (0, 1), got {threshold}.")


# ── Build dictionary command ─────────────────────────────────────────────────


@app.command("build-dict")
def build_dict(
    audio_dir: Optional[Path] = typer.Option(
        None,
        "--audio-dir",
        "-d",
        help="Folder containing isolated-note recordings.",
    ),
    output: Path = typer.Option(
        Path("dict.npy"),
        "--output",
        "-o",
        help="Output dictionary .npy path.",
    ),
    pitch_min: int = typer.Option(21, "--pitch-min"),
    pitch_max: int = typer.Option(108, "--pitch-max"),
    n_fft: int = typer.Option(2048, "--n-fft"),
    hop_length: int = typer.Option(512, "--hop-length"),
    sr: int = typer.Option(22050, "--sr"),
    harmonic: bool = typer.Option(
        False,
        "--harmonic",
        help="Build a physics-based harmonic dictionary.",
    ),
    n_harmonics: int = typer.Option(6, "--n-harmonics"),
) -> None:
    """Build a dictionary from note recordings or harmonic templates."""
    typer.echo(
        typer.style(
            "\nNMF Transcriber — build-dict",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )

    if pitch_min < 0 or pitch_max > 127 or pitch_min >= pitch_max:
        _err(
            f"Invalid pitch range: {pitch_min}–{pitch_max}. "
            "Use MIDI values 0–127 with min < max."
        )

    if n_fft < 1 or hop_length < 1 or sr < 1:
        _err("n_fft, hop_length and sr must be positive.")

    if n_harmonics < 1:
        _err("--n-harmonics must be at least 1.")

    from nmf_core.dictionary import (
        build_from_audio_dir,
        build_harmonic_dictionary,
        save_dictionary,
    )

    if harmonic:
        import librosa
        import numpy as np

        _info(
            f"Mode: harmonic (MIDI {pitch_min}–{pitch_max}, "
            f"{n_harmonics} harmonics/note)"
        )
        _info(f"Output: {output}")

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        dictionary, groups, pitch_map = build_harmonic_dictionary(
            freqs,
            midi_low=pitch_min,
            midi_high=pitch_max,
            n_harmonics=n_harmonics,
        )

        n_notes = len(groups)
        W = np.zeros(
            (dictionary.shape[0], n_notes),
            dtype=np.float64,
        )

        for note_index, indices in enumerate(groups):
            W[:, note_index] = dictionary[:, indices].sum(axis=1)

        W /= W.max(axis=0, keepdims=True) + 1e-10

        save_dictionary(
            W,
            output,
            n_fft=n_fft,
            sr=sr,
            hop_length=hop_length,
            pitch_map=pitch_map,
        )

    else:
        if audio_dir is None:
            _err("Provide --audio-dir or use --harmonic.")

        if not audio_dir.exists():
            _err(f"Audio directory not found: {audio_dir}")

        _info(f"Mode: supervised (audio dir: {audio_dir})")
        _info(f"Pitch range: MIDI {pitch_min}–{pitch_max}")
        _info(f"Output: {output}")

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


# ── Transcribe command ────────────────────────────────────────────────────────


@app.command("transcribe")
def transcribe(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input audio file.",
    ),
    dict_path: Optional[Path] = typer.Option(
        None,
        "--dict",
        help="Optional pre-built dictionary .npy file.",
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
        help="Group sparsity penalty.",
    ),
    threshold: float = typer.Option(
        0.15,
        "--threshold",
        help="Activation threshold in (0, 1).",
    ),
    out_dir: Path = typer.Option(
        Path("outputs"),
        "--out-dir",
        help="Output directory.",
    ),
    K: int = typer.Option(12, "--K"),
    n_iter: int = typer.Option(200, "--n-iter"),
    beta: float = typer.Option(0.5, "--beta"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Transcribe an audio file into MIDI."""
    typer.echo(
        typer.style(
            "\nNMF Transcriber — transcribe",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )

    _validate_audio(input_path)

    if method not in VALID_METHODS:
        _err(f"Unknown method '{method}'. Choose from: {', '.join(VALID_METHODS)}")

    _validate_hyperparams(sparsity_weight, threshold)

    if not math.isfinite(beta) or beta < 0:
        _err("--beta must be finite and >= 0.")

    if n_iter < 1 or K < 1:
        _err("--n-iter and --K must be positive.")

    if dict_path is not None:
        _validate_dict_path(dict_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    _info(f"Input  : {input_path}")
    _info(f"Method : {method}")
    _info(f"Output : {out_dir}")

    _info("\nLoading audio...")
    from nmf_core.audio import load_spectrogram, validate_dimensions

    try:
        spec = load_spectrogram(input_path, normalize=True)
    except (FileNotFoundError, ValueError) as exc:
        _err(str(exc))

    _info(f"  V shape: {spec.V.shape[0]} frequency bins × {spec.V.shape[1]} frames")
    _info(f"  Duration: {spec.duration_s:.2f}s  SR: {spec.sr} Hz")

    import numpy as np

    from nmf_core.dictionary import (
        build_harmonic_dictionary,
        load_dictionary,
        save_dictionary,
    )

    if dict_path is not None:
        _info("\nLoading dictionary...")

        try:
            W_dict, metadata = load_dictionary(dict_path)
        except (FileNotFoundError, ValueError) as exc:
            _err(str(exc))

        try:
            validate_dimensions(spec.V, W_dict)
        except ValueError as exc:
            _err(str(exc))

        pitch_map = metadata["pitch_map"]
        K_dict = W_dict.shape[1]
        _info(f"  Dictionary: {W_dict.shape[0]}×{W_dict.shape[1]}")

    else:
        _info("\nNo --dict provided — building harmonic dictionary...")

        D_full, groups_full, pitch_map = build_harmonic_dictionary(
            spec.freqs,
            midi_low=48,
            midi_high=84,
            n_harmonics=6,
        )

        n_notes = len(groups_full)
        W_dict = np.zeros(
            (D_full.shape[0], n_notes),
            dtype=np.float64,
        )

        for note_index, indices in enumerate(groups_full):
            W_dict[:, note_index] = D_full[:, indices].sum(axis=1)

        W_dict /= W_dict.max(axis=0, keepdims=True) + 1e-10
        K_dict = n_notes

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

    # ── Standard NMF ─────────────────────────────────────────────────────────
    result_std = None
    result_gs = None
    pitch_map_std = pitch_map

    if method in ("standard", "compare"):
        _info("\nRunning Standard Euclidean NMF...")

        from nmf_core.standard_nmf import run_nmf
        from nmf_core.midi import (
            activations_to_midi,
            save_activations_npz,
        )

        result_std = run_nmf(
            spec.V,
            K=K_dict,
            W_init=W_dict,
            n_iter=n_iter,
            verbose=verbose,
        )

        _info(
            f"  Converged: {result_std.converged} "
            f"({result_std.n_iter} iters, "
            f"error reduction {result_std.error_reduction_pct:.1f}%)"
        )

        mid_std = out_dir / f"{stem}_standard.mid"

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

    # ── Group-Sparse NMF ─────────────────────────────────────────────────────
    if method in ("group-sparse", "compare"):
        _info("\nRunning Group-Sparse beta-NMF...")

        from nmf_core.group_sparse_nmf import group_sparse_nmd
        from nmf_core.midi import (
            piano_roll_to_midi,
            save_activations_npz,
        )

        D_gs, groups_gs, pm_gs = build_harmonic_dictionary(
            spec.freqs,
            midi_low=48,
            midi_high=84,
            n_harmonics=6,
        )

        midi_notes_gs = [int(item["midi_note"]) for item in pm_gs]
        note_names_gs = [item["note_name"] for item in pm_gs]

        if len(groups_gs) != len(pm_gs):
            _err(
                f"Dictionary mismatch: {len(groups_gs)} groups but "
                f"{len(pm_gs)} pitch entries."
            )

        result_gs = group_sparse_nmd(
            S=spec.V,
            D=D_gs,
            groups=groups_gs,
            beta=beta,
            lam=sparsity_weight,
            verbose=verbose,
        )

        stats = result_gs.sparsity_stats()
        _info(f"  Converged: {result_gs.converged} ({result_gs.n_iter} iters)")
        _info(
            f"  Sparsity: {stats['sparsity_pct']:.1f}%  "
            f"Mean active notes/frame: "
            f"{stats['mean_active_per_frame']:.2f}"
        )

        roll = result_gs.piano_roll()

        if roll.shape[0] != len(midi_notes_gs):
            _err(
                f"Piano-roll mismatch: {roll.shape[0]} rows but "
                f"{len(midi_notes_gs)} pitches."
            )

        # Diagnostic summary to check which pitches dominate.

        mid_gs = out_dir / f"{stem}_group_sparse.mid"

        piano_roll_to_midi(
            roll=roll,
            note_names=note_names_gs,
            midi_notes=midi_notes_gs,
            sr=spec.sr,
            hop_length=spec.hop_length,
            out_path=mid_gs,
            threshold=threshold,
            min_row_peak_ratio=0.0,
            min_note_dur_s=0.10,
        )

        # Save metadata that corresponds to the collapsed piano-roll rows.
        pitch_map_roll = [
            {
                "component": index,
                "midi_note": midi_notes_gs[index],
                "note_name": note_names_gs[index],
            }
            for index in range(len(midi_notes_gs))
        ]

        save_activations_npz(
            roll,
            pitch_map_roll,
            out_dir / f"{stem}_group_sparse_H.npz",
        )

        _ok(f"Group-Sparse MIDI -> {mid_gs}")

    # ── Summary ───────────────────────────────────────────────────────────────
    from nmf_core.midi import build_summary

    summary = build_summary(
        method=method,
        audio_path=str(input_path),
        out_dir=str(out_dir),
        result_std=result_std,
        result_gs=result_gs,
        threshold=threshold,
        sparsity_weight=sparsity_weight,
    )

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    _ok(f"Summary log -> {summary_path}")
    typer.echo(
        typer.style(
            "\n  Transcription complete!",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )


# ── Evaluate command ─────────────────────────────────────────────────────────


@app.command("evaluate")
def evaluate(
    predicted: Path = typer.Option(..., "--predicted", "-p"),
    reference: Path = typer.Option(..., "--reference", "-r"),
    compare_std: Optional[Path] = typer.Option(None, "--compare-std"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
) -> None:
    """Evaluate predicted MIDI against a reference MIDI."""
    typer.echo(
        typer.style(
            "\nNMF Transcriber — evaluate",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )

    for path, label in (
        (predicted, "predicted"),
        (reference, "reference"),
    ):
        if not path.exists():
            _err(f"{label} file not found: {path}")

    from nmf_core.evaluate import (
        compare_methods,
        evaluate_midi,
        print_report,
        save_report_json,
    )

    if compare_std is not None:
        if not compare_std.exists():
            _err(f"--compare-std file not found: {compare_std}")

        _info("Evaluating Standard and Group-Sparse against reference...")
        scores = compare_methods(compare_std, predicted, reference)

        for key, label in (
            ("standard", "Standard NMF"),
            ("group_sparse", "Group-Sparse NMF"),
        ):
            score = scores[key]
            frame = score["frame"]
            onset = score["note_onset"]

            typer.echo(f"\n  -- {label} --")
            typer.echo(
                f"  Notes: {score['n_pred_notes']} predicted / "
                f"{score['n_ref_notes']} reference"
            )
            typer.echo(
                f"  Frame P/R/F: {frame['precision']:.3f} / "
                f"{frame['recall']:.3f} / {frame['f_measure']:.3f}"
            )
            typer.echo(
                f"  Note P/R/F: {onset['precision']:.3f} / "
                f"{onset['recall']:.3f} / {onset['f_measure']:.3f}"
            )

        winner = scores["winner"]
        typer.echo(
            typer.style(
                f"\n  Frame F winner: {winner['frame_f']}\n"
                f"  Note F winner: {winner['note_onset_f']}",
                fg=typer.colors.GREEN,
            )
        )

        if out_json is not None:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with out_json.open("w", encoding="utf-8") as file:
                json.dump(scores, file, indent=2)
            _ok(f"Scores saved -> {out_json}")

    else:
        _info(f"Evaluating {predicted.name} against {reference.name}...")
        result = evaluate_midi(predicted, reference)
        print_report(result, label=predicted.stem)

        if out_json is not None:
            save_report_json(result, out_json)
            _ok(f"Scores saved -> {out_json}")

    typer.echo(
        typer.style(
            "  Evaluation complete!",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )


# ── Report command ────────────────────────────────────────────────────────────


@app.command("report")
def report(
    std_mid: Path = typer.Option(..., "--std-mid"),
    gs_mid: Path = typer.Option(..., "--gs-mid"),
    summary: Path = typer.Option(..., "--summary"),
    out_html: Path = typer.Option(
        Path("outputs/report.html"),
        "--out",
    ),
    eval_json: Optional[Path] = typer.Option(None, "--eval-json"),
) -> None:
    """Generate a self-contained HTML comparison report."""
    typer.echo(
        typer.style(
            "\nNMF Transcriber — report",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )

    for path, label in (
        (std_mid, "--std-mid"),
        (gs_mid, "--gs-mid"),
        (summary, "--summary"),
    ):
        if not path.exists():
            _err(f"{label} file not found: {path}")

    from nmf_core.report import generate_report

    try:
        generate_report(
            out_path=out_html,
            std_mid=std_mid,
            gs_mid=gs_mid,
            summary_json=summary,
            eval_json=(
                eval_json if eval_json is not None and eval_json.exists() else None
            ),
        )
    except Exception as exc:
        _err(f"Report generation failed: {exc}")

    _ok(f"Report -> {out_html}")
    _info("Open the HTML file in a browser.")


# ── Sheet command ────────────────────────────────────────────────────────────


@app.command("sheet")
def sheet(
    midi_in: Path = typer.Option(..., "--input", "-i"),
    out_xml: Path = typer.Option(
        Path("outputs/score.xml"),
        "--out",
    ),
    title: str = typer.Option("NMF Transcription", "--title"),
    pdf: bool = typer.Option(False, "--pdf"),
) -> None:
    """Export MIDI to MusicXML or PDF."""
    typer.echo(
        typer.style(
            "\nNMF Transcriber — sheet",
            fg=typer.colors.CYAN,
            bold=True,
        )
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

    _ok(f"Sheet export -> {out_xml}")


if __name__ == "__main__":
    app()
