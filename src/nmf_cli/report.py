"""Self-contained HTML report comparing the standard and group-sparse output.

Presentation only: it reads the files ``transcribe`` and ``evaluate`` wrote
and draws them with matplotlib. Images are embedded as base64 PNGs, so the
.html opens offline.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import pretty_midi

from nmf_core.errors import TranscriberError

_COLORS = {"reference": "#6b7280", "standard": "#2563eb", "group_sparse": "#ea580c"}
_LABELS = {
    "reference": "Reference",
    "standard": "Standard NMF",
    "group_sparse": "Group-sparse NMF",
}


def _notes(path: Path) -> list[pretty_midi.Note]:
    if not path.is_file():
        raise TranscriberError(f"MIDI file not found: {path}")
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        raise TranscriberError(f"Cannot read MIDI file {path}: {exc}") from exc
    return [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]


def _png(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _piano_rolls(rolls: dict[str, list[pretty_midi.Note]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    every = [n for notes in rolls.values() for n in notes]
    lo = min((n.pitch for n in every), default=48) - 2
    hi = max((n.pitch for n in every), default=84) + 2
    end = max((n.end for n in every), default=1.0)
    fig, axes = plt.subplots(
        len(rolls), 1, figsize=(10, 2.2 * len(rolls)), sharex=True, squeeze=False
    )
    for ax, (key, notes) in zip(axes[:, 0], rolls.items(), strict=True):
        for n in notes:
            ax.barh(
                n.pitch, n.end - n.start, left=n.start, height=0.8,
                color=_COLORS[key], edgecolor="none",
            )  # fmt: skip
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, end * 1.02)
        ax.set_ylabel("MIDI pitch")
        ax.set_title(f"{_LABELS[key]} ({len(notes)} notes)", loc="left", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    return _png(fig)


def _convergence(histories: dict[str, np.ndarray]) -> str | None:
    histories = {k: h for k, h in histories.items() if h.size > 1}
    if not histories:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(histories), figsize=(10, 3), squeeze=False)
    for ax, (key, h) in zip(axes[0], histories.items(), strict=True):
        # cost is recorded every few iterations; plot it against its index
        ax.plot(h / h[0], color=_COLORS[key])
        ax.set_yscale("log")
        ax.set_title(_LABELS[key], loc="left", fontsize=10)
        ax.set_xlabel("convergence check")
        ax.set_ylabel("cost / initial cost")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _png(fig)


def _history(npz: str | None) -> np.ndarray:
    if not npz or not Path(npz).is_file():
        return np.zeros(0)
    with np.load(npz) as z:
        return z["cost_history"] if "cost_history" in z.files else np.zeros(0)


def _row(cells: list[str], tag: str = "td") -> str:
    return "<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>"


def _fmt(v, digits: int = 3) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return "-" if v is None else str(v)


def generate_report(
    out: str | Path,
    std_mid: str | Path,
    gs_mid: str | Path,
    summary_json: str | Path,
    eval_json: str | Path | None = None,
    reference_mid: str | Path | None = None,
) -> Path:
    """Write the HTML report and return its path."""
    std_mid, gs_mid, out = Path(std_mid), Path(gs_mid), Path(out)
    try:
        summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TranscriberError(f"summary.json not found: {summary_json}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriberError(f"Cannot parse {summary_json}: {exc}") from exc
    results = summary.get("results", {})

    rolls: dict[str, list[pretty_midi.Note]] = {}
    if reference_mid:
        rolls["reference"] = _notes(Path(reference_mid))
    rolls["standard"] = _notes(std_mid)
    rolls["group_sparse"] = _notes(gs_mid)

    rows = [
        ("Notes", "n_notes", 0),
        ("Iterations", "iterations", 0),
        ("Converged", "converged", 0),
        ("Solve time (s)", "solve_time_s", 3),
        ("Relative reconstruction error", "relative_reconstruction_error", 3),
        ("Hoyer sparsity of the pitch roll", "pitch_roll_hoyer_sparsity", 3),
        ("Mean active pitches per frame", "mean_active_pitches_per_frame", 2),
    ]
    run_table = _row(["", _LABELS["standard"], _LABELS["group_sparse"]], "th")
    for label, key, digits in rows:
        run_table += _row(
            [label]
            + [
                _fmt(results.get(m, {}).get(key), digits)
                for m in ("standard", "group_sparse")
            ]
        )

    score_html = ""
    if eval_json:
        try:
            ev = json.loads(Path(eval_json).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise TranscriberError(f"Evaluation file not found: {eval_json}") from exc
        except json.JSONDecodeError as exc:
            raise TranscriberError(f"Cannot parse {eval_json}: {exc}") from exc
        table = _row(
            ["File", "Notes", "Onset P / R / F1", "Onset+offset P / R / F1",
             "Frame F1"],
            "th",
        )  # fmt: skip
        for r in ev.get("results", []):

            def prf(m: dict) -> str:
                return f"{m['precision']:.2f} / {m['recall']:.2f} / {m['f1']:.2f}"

            table += _row(
                [
                    Path(r["file"]).name,
                    str(r["n_predicted"]),
                    prf(r["onset"]),
                    prf(r["onset_offset"]),
                    f"{r['frame']['f1']:.2f}" if "frame" in r else "-",
                ]
            )
        s = ev.get("settings", {})
        score_html = (
            "<h2>Scores against the reference</h2>"
            f"<p>Reference: {html.escape(Path(ev.get('reference', '')).name)} "
            f"({ev.get('results', [{}])[0].get('n_reference', '?')} notes). "
            f"Onset tolerance {s.get('onset_tolerance', 0.05) * 1000:.0f} ms; "
            f"offset tolerance max({s.get('offset_min', 0.05) * 1000:.0f} ms, "
            f"{s.get('offset_ratio', 0.2):.0%} of the note length).</p>"
            f"<table>{table}</table>"
        )

    conv = _convergence(
        {
            m: _history(results.get(m, {}).get("activations"))
            for m in ("standard", "group_sparse")
        }
    )
    conv_html = (
        f'<h2>Convergence</h2><img alt="convergence" src="data:image/png;base64,'
        f'{conv}">'
        if conv
        else ""
    )
    cfg = summary.get("config", {})
    settings = ", ".join(
        f"{k} = {cfg[k]}"
        for k in ("sparsity_weight", "threshold", "beta", "max_iter")
        if k in cfg
    )
    title = Path(cfg.get("input", std_mid.stem)).name
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>NMF transcription report: {html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto;
       padding: 0 1rem; color: #111827; background: #ffffff; }}
table {{ border-collapse: collapse; margin: 0.5rem 0 1.5rem; }}
th, td {{ border: 1px solid #d1d5db; padding: 0.35rem 0.7rem; text-align: left; }}
th {{ background: #f3f4f6; }}
img {{ max-width: 100%; }}
.muted {{ color: #6b7280; }}
</style></head><body>
<h1>NMF transcription report</h1>
<p class="muted">Input: {html.escape(title)}. Settings: {html.escape(settings)}.</p>
<h2>Piano rolls</h2>
<img alt="piano rolls" src="data:image/png;base64,{_piano_rolls(rolls)}">
{score_html}
<h2>Solver runs</h2>
<table>{run_table}</table>
{conv_html}
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
