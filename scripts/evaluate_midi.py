"""Note-level precision / recall / F1 of transcribed MIDI against a reference.

Kept for convenience; it is the same as ``python -m nmf_cli evaluate``.

Usage:
  python scripts/evaluate_midi.py REFERENCE.mid PREDICTED.mid [PREDICTED.mid ...]
      [--onset-tolerance 0.05] [--offset-ratio 0.2] [--offset-min 0.05]
      [--json scores.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nmf_cli.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reference", type=Path, help="reference (ground-truth) .mid")
    ap.add_argument("predicted", type=Path, nargs="+", help="transcribed .mid file(s)")
    ap.add_argument("--onset-tolerance", default="0.05", help="seconds")
    ap.add_argument("--offset-ratio", default="0.2", help="fraction of duration")
    ap.add_argument("--offset-min", default="0.05", help="seconds")
    ap.add_argument("--json", type=Path, help="also write the scores to this file")
    a = ap.parse_args(argv)
    args = ["evaluate", "--reference", str(a.reference), "--predicted"]
    args += [str(p) for p in a.predicted]
    args += ["--onset-tolerance", a.onset_tolerance, "--offset-ratio", a.offset_ratio]
    args += ["--offset-min", a.offset_min]
    if a.json:
        args += ["--out-json", str(a.json)]
    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
