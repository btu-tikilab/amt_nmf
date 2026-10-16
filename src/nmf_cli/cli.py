"""argparse front end. All user-facing printing lives here, not in nmf_core."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path

from nmf_core import __version__
from nmf_core.audio import SpectrogramConfig
from nmf_core.errors import TranscriberError
from nmf_core.pipeline import METHODS


def _log(msg: str) -> None:
    print(msg, flush=True)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nmf_cli",
        description="Polyphonic piano transcription with standard and "
        "group-sparse NMF.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--debug",
        action="store_true",
        help="show full tracebacks for unexpected errors",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    b = sub.add_parser(
        "build-dict",
        help="build a pitch dictionary W",
        description="Build a pitch dictionary W (.npy) and its "
        "metadata sidecar (.json).",
    )
    b.add_argument(
        "--audio-dir",
        type=Path,
        help="folder of isolated single-note recordings (supervised). "
        "Pitches are read from file names, e.g. MAPS '..._M60_...', "
        "'note_60.wav' or 'C4.wav'",
    )
    b.add_argument(
        "--harmonic",
        action="store_true",
        help="build a synthetic harmonic dictionary instead (no recordings needed)",
    )
    b.add_argument(
        "--pitch-range",
        type=int,
        nargs=2,
        default=(21, 108),
        metavar=("MIN", "MAX"),
        help="MIDI pitch range to include (default: 21 108)",
    )
    b.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output .npy path; the .json sidecar is written next to it",
    )
    b.add_argument(
        "--atoms-per-pitch",
        type=int,
        default=2,
        help="spectral atoms extracted per pitch, i.e. the group size "
        "for group sparsity (default: 2)",
    )
    d = SpectrogramConfig()
    b.add_argument(
        "--sr", type=int, default=d.sr, help=f"analysis sample rate (default: {d.sr})"
    )
    b.add_argument(
        "--hop-length",
        type=int,
        default=d.hop_length,
        help=f"hop in samples (default: {d.hop_length})",
    )
    b.add_argument(
        "--fmin",
        type=float,
        default=d.fmin,
        help=f"lowest CQT frequency in Hz (default: {d.fmin})",
    )
    b.add_argument(
        "--bins-per-octave",
        type=int,
        default=d.bins_per_octave,
        help=f"CQT resolution (default: {d.bins_per_octave})",
    )
    b.add_argument(
        "--n-bins",
        type=int,
        default=d.n_bins,
        help=f"number of CQT bins (default: {d.n_bins})",
    )

    t = sub.add_parser(
        "transcribe",
        help="transcribe audio to MIDI",
        description="Transcribe a polyphonic recording to MIDI.",
    )
    t.add_argument(
        "--input", type=Path, required=True, help="polyphonic .wav or .flac file"
    )
    t.add_argument(
        "--dict",
        type=Path,
        required=True,
        dest="dict_path",
        help="dictionary .npy built by build-dict",
    )
    t.add_argument(
        "--method",
        default="compare",
        choices=METHODS,
        help="solver to run (default: compare = both)",
    )
    t.add_argument(
        "--sparsity-weight",
        type=float,
        default=0.2,
        help="lambda, weight of the l1,2 group penalty (default: 0.2)",
    )
    t.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="note-on threshold, relative to the loudest activation, "
        "in (0, 1] (default: 0.1)",
    )
    t.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="output directory (default: ./outputs)",
    )
    t.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="beta-divergence of the group-sparse cost: 2 Euclidean (same "
        "cost as standard NMF), 1 KL, 0 Itakura-Saito (default: 2)",
    )
    t.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="maximum NMF iterations (default: 1000)",
    )
    t.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="relative cost decrease that counts as converged (default: 1e-4)",
    )
    t.add_argument(
        "--min-duration",
        type=float,
        default=0.05,
        help="drop notes shorter than this, in seconds (default: 0.05)",
    )

    e = sub.add_parser(
        "evaluate",
        help="score transcribed MIDI against a reference MIDI",
        description="Note-level precision / recall / F1 (onset, and onset + "
        "offset) and frame-level F1 of transcribed MIDI against a reference.",
    )
    e.add_argument(
        "--reference", type=Path, required=True, help="reference (ground-truth) .mid"
    )
    e.add_argument(
        "--predicted",
        type=Path,
        nargs="+",
        required=True,
        help="transcribed .mid file(s)",
    )
    e.add_argument(
        "--compare-std",
        type=Path,
        help="a standard-NMF .mid to score alongside --predicted",
    )
    e.add_argument(
        "--out-json", type=Path, help="also write the scores to this .json file"
    )
    e.add_argument(
        "--onset-tolerance",
        type=float,
        default=0.05,
        help="onset tolerance in seconds (default: 0.05)",
    )
    e.add_argument(
        "--offset-ratio",
        type=float,
        default=0.2,
        help="offset tolerance as a fraction of the reference note length "
        "(default: 0.2)",
    )
    e.add_argument(
        "--offset-min",
        type=float,
        default=0.05,
        help="minimum offset tolerance in seconds (default: 0.05)",
    )

    r = sub.add_parser(
        "report",
        help="write an HTML report comparing both methods",
        description="Self-contained HTML report: piano rolls, solver runs, "
        "convergence and (optionally) scores against a reference.",
    )
    r.add_argument("--std-mid", type=Path, required=True, help="standard NMF .mid")
    r.add_argument("--gs-mid", type=Path, required=True, help="group-sparse NMF .mid")
    r.add_argument(
        "--summary", type=Path, required=True, help="summary.json from transcribe"
    )
    r.add_argument("--eval-json", type=Path, help="scores written by evaluate")
    r.add_argument(
        "--reference", type=Path, help="reference .mid, drawn as a third piano roll"
    )
    r.add_argument(
        "--out", type=Path, default=Path("outputs/report.html"), help="output .html"
    )

    s = sub.add_parser(
        "sheet",
        help="convert MIDI to sheet music (MusicXML, optional PDF)",
        description="Quantise a MIDI file to 16th notes and write MusicXML. "
        "--pdf also renders a PDF if MuseScore is installed.",
    )
    s.add_argument("--input", type=Path, required=True, help="input .mid")
    s.add_argument("--out", type=Path, required=True, help="output .xml (MusicXML)")
    s.add_argument("--title", default="NMF transcription", help="score title")
    s.add_argument(
        "--pdf", action="store_true", help="also write a PDF next to the .xml"
    )
    return p


def _build_dict(a: argparse.Namespace) -> None:
    from nmf_core.dictionary import (
        build_from_audio_dir,
        build_harmonic_dictionary,
        save_dictionary,
    )
    from nmf_core.errors import ParameterError

    if a.harmonic == (a.audio_dir is not None):
        raise ParameterError("Give exactly one of --audio-dir or --harmonic.")
    cfg = SpectrogramConfig(
        sr=a.sr,
        hop_length=a.hop_length,
        fmin=a.fmin,
        bins_per_octave=a.bins_per_octave,
        n_bins=a.n_bins,
    )
    rng = tuple(a.pitch_range)
    if a.harmonic:
        _log(f"Building harmonic dictionary for MIDI {rng[0]}-{rng[1]} ...")
        d = build_harmonic_dictionary(rng, a.atoms_per_pitch, cfg)
    else:
        _log(
            f"Building supervised dictionary from {a.audio_dir} "
            f"(MIDI {rng[0]}-{rng[1]}) ..."
        )
        d = build_from_audio_dir(a.audio_dir, rng, a.atoms_per_pitch, cfg, log=_log)
    npy, meta = save_dictionary(d, a.output)
    missing = sorted(set(range(rng[0], rng[1] + 1)) - set(d.pitches))
    if missing and not a.harmonic:
        _log(
            f"Note: no recordings for {len(missing)} pitch(es) in the range, "
            f"e.g. MIDI {missing[:5]}; they cannot be transcribed."
        )
    _log(
        f"Dictionary: {d.W.shape[0]} bins x {d.W.shape[1]} atoms "
        f"({len(d.pitch_map)} pitches)"
    )
    _log(f"Saved {npy}\nSaved {meta}")


def _transcribe(a: argparse.Namespace) -> None:
    from nmf_core.pipeline import transcribe

    s = transcribe(
        a.input,
        a.dict_path,
        a.out_dir,
        method=a.method,
        sparsity_weight=a.sparsity_weight,
        threshold=a.threshold,
        beta=a.beta,
        max_iter=a.max_iter,
        tol=a.tol,
        min_duration=a.min_duration,
        log=_log,
    )
    _log(
        f"Summary: {Path(a.out_dir) / 'summary.json'} "
        f"({s['processing_time_s']:.2f} s total)"
    )


def _evaluate(a: argparse.Namespace) -> None:
    from nmf_core.evaluate import score

    predicted = ([a.compare_std] if a.compare_std else []) + list(a.predicted)
    results = [
        score(a.reference, p, a.onset_tolerance, a.offset_ratio, a.offset_min)
        for p in predicted
    ]
    _log(f"Reference: {a.reference} ({results[0]['n_reference']} notes)")
    _log(
        f"Onset tolerance {a.onset_tolerance * 1000:.0f} ms; offset tolerance "
        f"max({a.offset_min * 1000:.0f} ms, {a.offset_ratio:.0%} of duration)\n"
    )

    def fmt(m: dict) -> str:
        return f"{m['precision']:.2f} / {m['recall']:.2f} / {m['f1']:.2f}"

    head = (
        f"{'file':<32} {'notes':>5}   {'onset P / R / F1':<18}   "
        f"{'onset+offset P / R / F1':<23}   frame F1"
    )
    _log(head)
    _log("-" * len(head))
    for r in results:
        _log(
            f"{Path(r['file']).name:<32} {r['n_predicted']:>5}   "
            f"{fmt(r['onset']):<18}   {fmt(r['onset_offset']):<23}   "
            f"{r['frame']['f1']:.2f}"
        )
    if a.out_json:
        settings = {
            "onset_tolerance": a.onset_tolerance,
            "offset_ratio": a.offset_ratio,
            "offset_min": a.offset_min,
        }
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(
            json.dumps(
                {"reference": str(a.reference), "settings": settings,
                 "results": results},
                indent=2,
            ),
            encoding="utf-8",
        )  # fmt: skip
        _log(f"\nScores written to {a.out_json}")


def _report(a: argparse.Namespace) -> None:
    from nmf_cli.report import generate_report

    out = generate_report(
        a.out, a.std_mid, a.gs_mid, a.summary, a.eval_json, a.reference
    )
    _log(f"Report written to {out}")


def _sheet(a: argparse.Namespace) -> None:
    from nmf_cli.sheet import midi_to_musicxml, musicxml_to_pdf

    xml = midi_to_musicxml(a.input, a.out, a.title)
    _log(f"MusicXML written to {xml}")
    if a.pdf:
        pdf = musicxml_to_pdf(xml, xml.with_suffix(".pdf"))
        if pdf is None:
            _log(
                "MuseScore was not found, so no PDF was made. Open the .xml in "
                "MuseScore and use File > Export > PDF."
            )
        else:
            _log(f"PDF written to {pdf}")


_COMMANDS = {
    "build-dict": _build_dict,
    "transcribe": _transcribe,
    "evaluate": _evaluate,
    "report": _report,
    "sheet": _sheet,
}


def _safe_console() -> None:
    """Never crash on a character the console cannot encode (Windows cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> int:
    _safe_console()
    args = _parser().parse_args(argv)
    try:
        _COMMANDS[args.command](args)
    except TranscriberError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # last-resort guard: never dump a raw traceback
        if args.debug:
            traceback.print_exc()
        print(
            f"Unexpected error: {exc.__class__.__name__}: {exc}\n"
            "Re-run with --debug for the full traceback.",
            file=sys.stderr,
        )
        return 2
    return 0
