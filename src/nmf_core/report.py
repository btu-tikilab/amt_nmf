"""
src/nmf_core/report.py
─────────────────────────────────────────────────────────────────
HTML comparison report generator.

Produces a single self-contained .html file (all assets embedded
as base64) showing:
  - Side-by-side piano rolls (Standard NMF vs Group-Sparse NMF)
  - Convergence curves for both methods
  - Sparsity statistics table
  - Evaluation scores (if available)

No external CSS/JS dependencies — the file opens offline.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi

# ── Piano roll builder ────────────────────────────────────────────────────────

def _midi_to_roll(midi_path: str | Path,
                  fs: float = 50.0) -> np.ndarray:
    """Load MIDI -> (88, T) piano roll (pitches 21-108)."""
    pm   = pretty_midi.PrettyMIDI(str(midi_path))
    roll = pm.get_piano_roll(fs=fs)        # (128, T)
    return roll[21:109, :]                  # crop to 88 piano keys


def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Figure builders ───────────────────────────────────────────────────────────

def _piano_roll_figure(
    std_mid:  str | Path,
    gs_mid:   str | Path,
    title:    str = "",
) -> plt.Figure:
    """Side-by-side piano roll comparison figure."""
    BG   = "#0f172a"
    CARD = "#1e293b"
    TC   = "#e2e8f0"

    roll_std = _midi_to_roll(std_mid)
    roll_gs  = _midi_to_roll(gs_mid)
    T        = max(roll_std.shape[1], roll_gs.shape[1])

    # Pad shorter roll
    def _pad(r):
        if r.shape[1] < T:
            return np.pad(r, ((0, 0), (0, T - r.shape[1])))
        return r

    roll_std = _pad(roll_std)
    roll_gs  = _pad(roll_gs)

    duration = T / 50.0    # fs=50

    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                              facecolor=BG, sharey=True)
    fig.suptitle(title or "Piano Roll Comparison",
                 fontsize=12, color=TC, y=1.01)

    labels = {0: ("Standard NMF",    "#38bdf8"),
              1: ("Group-Sparse NMF", "#4ade80")}

    for ax, (roll, (method, colour)) in zip(
        axes, [(roll_std, labels[0]), (roll_gs, labels[1])]
    ):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TC, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#334155")

        ax.imshow(roll > 0, aspect="auto", origin="lower",
                  cmap="Blues", interpolation="nearest",
                  extent=[0, duration, 21, 108])

        ax.set_title(method, color=colour, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", color=TC, fontsize=9)
        ax.set_ylabel("MIDI pitch", color=TC, fontsize=9)

        # Note density badge
        active = int((roll > 0).sum())
        ax.text(0.98, 0.97, f"Active cells: {active:,}",
                transform=ax.transAxes, fontsize=8,
                color=colour, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=CARD, edgecolor=colour, alpha=0.9))

    fig.tight_layout()
    return fig


def _convergence_figure(
    std_errors:   list[float],
    std_iters:    list[int],
    gs_costs:     list[float],
    std_label:    str = "Standard NMF (Frobenius error)",
    gs_label:     str = "Group-Sparse NMF (GS cost)",
) -> plt.Figure:
    """Convergence curves for both methods on separate axes."""
    BG   = "#0f172a"
    CARD = "#1e293b"
    TC   = "#e2e8f0"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), facecolor=BG)

    for ax, xs, ys, colour, label in [
        (ax1, std_iters, std_errors, "#38bdf8", std_label),
        (ax2, list(range(len(gs_costs))), gs_costs, "#4ade80", gs_label),
    ]:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TC, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#334155")
        ax.semilogy(xs, ys, color=colour, linewidth=2)
        ax.fill_between(xs, ys, alpha=0.12, color=colour)
        ax.set_title(label, color=colour, fontsize=10, fontweight="bold")
        ax.set_xlabel("Iteration / Checkpoint", color=TC, fontsize=9)
        ax.set_ylabel("Cost (log scale)", color=TC, fontsize=9)
        ax.grid(True, color="#1e3a5f", linewidth=0.5, which="both")

    fig.tight_layout()
    return fig


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NMF Transcription Report — {audio_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background:#0f172a;color:#e2e8f0;padding:2rem}}
h1{{font-size:1.6rem;color:#38bdf8;margin-bottom:.25rem}}
.sub{{color:#64748b;font-size:.9rem;margin-bottom:2rem}}
section{{margin-bottom:2.5rem}}
h2{{font-size:1.1rem;color:#94a3b8;margin-bottom:1rem;
    text-transform:uppercase;letter-spacing:.06em;font-weight:600}}
img{{width:100%;border-radius:10px;border:1px solid #1e293b}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}
th{{background:#1e3a5f;color:#fff;padding:.6rem 1rem;text-align:left;
    font-weight:600;letter-spacing:.04em}}
td{{padding:.55rem 1rem;border-bottom:1px solid #1e293b;color:#e2e8f0}}
tr:nth-child(even) td{{background:#1e293b}}
.badge{{display:inline-block;padding:.2rem .6rem;border-radius:.4rem;
        font-size:.78rem;font-weight:600}}
.badge-std{{background:#0c4a6e;color:#38bdf8}}
.badge-gs{{background:#14532d;color:#4ade80}}
.badge-win{{background:#422006;color:#fb923c}}
.metrics-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:10px;
       padding:1.25rem}}
.card h3{{font-size:.95rem;margin-bottom:.75rem;color:#94a3b8}}
.score-row{{display:flex;justify-content:space-between;padding:.3rem 0;
            border-bottom:1px solid #334155;font-size:.88rem}}
.score-row:last-child{{border-bottom:none}}
.score-label{{color:#94a3b8}}
.score-val{{font-family:monospace;font-weight:600}}
.good{{color:#4ade80}}.ok{{color:#fbbf24}}.bad{{color:#f87171}}
footer{{margin-top:3rem;color:#475569;font-size:.8rem;text-align:center}}
</style>
</head>
<body>
<h1>NMF Transcription Report</h1>
<p class="sub">Audio: <strong>{audio_name}</strong> &nbsp;|&nbsp;
   Generated by NMF Transcriber V1</p>

<section>
  <h2>Piano Roll Comparison</h2>
  <img src="data:image/png;base64,{piano_roll_b64}" alt="Piano roll comparison">
</section>

<section>
  <h2>Convergence Curves</h2>
  <img src="data:image/png;base64,{convergence_b64}" alt="Convergence">
</section>

<section>
  <h2>Sparsity Analysis</h2>
  <table>
    <tr><th>Metric</th>
        <th><span class="badge badge-std">Standard NMF</span></th>
        <th><span class="badge badge-gs">Group-Sparse NMF</span></th>
        <th>Better</th></tr>
    {sparsity_rows}
  </table>
</section>

{eval_section}

<section>
  <h2>Run Configuration</h2>
  <table>
    {config_rows}
  </table>
</section>

<footer>NMF Transcriber V1 &nbsp;|&nbsp;
  Smaragdis &amp; Brown (2003) &nbsp;&amp;&nbsp;
  O&apos;Hanlon &amp; Plumbley (2014)</footer>
</body>
</html>"""


# ── Public entry point ────────────────────────────────────────────────────────

def generate_report(
    out_path:        str | Path,
    std_mid:         str | Path,
    gs_mid:          str | Path,
    summary_json:    str | Path,
    eval_json:       Optional[str | Path] = None,
    audio_name:      str = "",
) -> Path:
    """
    Generate a self-contained HTML comparison report.

    Parameters
    ----------
    out_path     : where to write the .html file
    std_mid      : Standard NMF output .mid file
    gs_mid       : Group-Sparse NMF output .mid file
    summary_json : summary.json from transcribe command
    eval_json    : optional evaluation scores JSON
    audio_name   : display name of the source audio

    Returns
    -------
    Path to the written HTML file
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_json, encoding="utf-8") as f:
        summary = json.load(f)

    std_res = summary.get("results", {}).get("standard",    {})
    gs_res  = summary.get("results", {}).get("group_sparse",{})

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_roll = _piano_roll_figure(
        std_mid, gs_mid,
        title=f"Estimated Piano Roll — {audio_name or summary.get('audio_input','')}"
    )
    roll_b64 = _fig_to_b64(fig_roll)
    plt.close(fig_roll)

    std_errors = std_res.get("errors", [std_res.get("final_error", 1.0)])
    std_iters  = list(range(len(std_errors)))
    gs_costs   = gs_res.get("cost_history", gs_res.get("cost_history", []))
    if not gs_costs:
        gs_costs = [gs_res.get("final_cost", 1.0)]

    fig_conv = _convergence_figure(
        std_errors, std_iters, gs_costs
    )
    conv_b64 = _fig_to_b64(fig_conv)
    plt.close(fig_conv)

    # ── Sparsity table ────────────────────────────────────────────────────────
    def _colour(val, good_high=True):
        if good_high:
            return "good" if val > 0.7 else "ok" if val > 0.4 else "bad"
        return "good" if val < 3 else "ok" if val < 6 else "bad"

    std_active = "N/A"
    gs_active  = f"{gs_res.get('mean_active_per_frame', 'N/A'):.2f}" \
                 if isinstance(gs_res.get("mean_active_per_frame"), float) \
                 else "N/A"
    gs_sparse  = f"{gs_res.get('sparsity_pct', 'N/A'):.1f}%" \
                 if isinstance(gs_res.get("sparsity_pct"), float) \
                 else "N/A"

    sparsity_rows = f"""
    <tr><td>Mean active notes/frame</td>
        <td>{std_active}</td>
        <td>{gs_active}</td>
        <td><span class="badge badge-gs">Group-Sparse</span></td></tr>
    <tr><td>Frame sparsity %</td>
        <td>—</td>
        <td>{gs_sparse}</td>
        <td><span class="badge badge-gs">Group-Sparse</span></td></tr>
    <tr><td>Error/cost reduction</td>
        <td>{std_res.get('error_reduction_pct','N/A')}%</td>
        <td>via GS penalty</td>
        <td>—</td></tr>
    <tr><td>Iterations</td>
        <td>{std_res.get('n_iter','N/A')}</td>
        <td>{gs_res.get('n_iter','N/A')}</td>
        <td>—</td></tr>
    <tr><td>Wall time</td>
        <td>{std_res.get('elapsed_s','N/A')}s</td>
        <td>{gs_res.get('elapsed_s','N/A')}s</td>
        <td>—</td></tr>
    """

    # ── Evaluation section ────────────────────────────────────────────────────
    eval_section = ""
    if eval_json and Path(eval_json).exists():
        with open(eval_json, encoding="utf-8") as f:
            ev = json.load(f)

        def _metrics_card(data: dict, label: str, badge_cls: str) -> str:
            fr  = data.get("frame",       {})
            no  = data.get("note_onset",  {})
            nof = data.get("note_offset", {})
            return f"""
            <div class="card">
              <h3><span class="badge {badge_cls}">{label}</span></h3>
              <div class="score-row">
                <span class="score-label">Frame F-measure</span>
                <span class="score-val {_colour(fr.get('f_measure',0))}">
                  {fr.get('f_measure',0):.3f}</span></div>
              <div class="score-row">
                <span class="score-label">Note onset F</span>
                <span class="score-val {_colour(no.get('f_measure',0))}">
                  {no.get('f_measure',0):.3f}</span></div>
              <div class="score-row">
                <span class="score-label">Note+offset F</span>
                <span class="score-val {_colour(nof.get('f_measure',0))}">
                  {nof.get('f_measure',0):.3f}</span></div>
              <div class="score-row">
                <span class="score-label">Precision / Recall</span>
                <span class="score-val">
                  {no.get('precision',0):.3f} / {no.get('recall',0):.3f}
                </span></div>
            </div>"""

        if "standard" in ev and "group_sparse" in ev:
            winner = ev.get("winner", {})
            cards  = (
                _metrics_card(ev["standard"],    "Standard NMF",    "badge-std") +
                _metrics_card(ev["group_sparse"], "Group-Sparse NMF","badge-gs")
            )
            winner_note = (
                f"<p style='margin-top:1rem;color:#fb923c;font-size:.9rem'>"
                f"&#9733; Frame F winner: "
                f"<strong>{winner.get('frame_f','—')}</strong> &nbsp;|&nbsp;"
                f" Note F winner: "
                f"<strong>{winner.get('note_onset_f','—')}</strong></p>"
            )
        else:
            cards = _metrics_card(ev, "Predicted MIDI", "badge-std")
            winner_note = ""

        eval_section = f"""
<section>
  <h2>Evaluation Scores (vs Reference MIDI)</h2>
  <div class="metrics-grid">{cards}</div>
  {winner_note}
</section>"""

    # ── Config rows ───────────────────────────────────────────────────────────
    config_rows = "".join(
        f"<tr><td style='color:#64748b;width:200px'>{k}</td><td>{v}</td></tr>"
        for k, v in {
            "Method":          summary.get("method", "compare"),
            "Audio input":     summary.get("audio_input", ""),
            "Threshold":       summary.get("threshold", ""),
            "Sparsity weight": summary.get("sparsity_weight", ""),
            "beta (beta)":        gs_res.get("beta", ""),
            "λ (lambda)":      gs_res.get("lam", ""),
        }.items()
    )

    # ── Write HTML ────────────────────────────────────────────────────────────
    html = _HTML.format(
        audio_name     = audio_name or Path(
            summary.get("audio_input", "audio")
        ).name,
        piano_roll_b64 = roll_b64,
        convergence_b64= conv_b64,
        sparsity_rows  = sparsity_rows,
        eval_section   = eval_section,
        config_rows    = config_rows,
    )

    out_path.write_text(html, encoding="utf-8")
    print(f"  Report saved -> {out_path}")
    return out_path
