"""MIDI to sheet music: MusicXML via music21, and PDF through MuseScore."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from nmf_core.errors import TranscriberError

_MUSESCORE = ("mscore", "musescore", "musescore4", "musescore3", "MuseScore4")


def midi_to_musicxml(
    midi: str | Path, out: str | Path, title: str = "NMF transcription"
) -> Path:
    """Quantise a MIDI file to a 16th-note grid and write it as MusicXML."""
    from music21 import converter, metadata

    midi, out = Path(midi), Path(out)
    if not midi.is_file():
        raise TranscriberError(f"MIDI file not found: {midi}")
    try:
        score = converter.parse(str(midi))
    except Exception as exc:
        raise TranscriberError(f"Cannot read MIDI file {midi}: {exc}") from exc
    score = score.quantize(
        quarterLengthDivisors=(4,),
        processOffsets=True,
        processDurations=True,
        inPlace=False,
    )
    score.metadata = metadata.Metadata()
    score.metadata.title = title
    out.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out))
    return out


def find_musescore() -> str | None:
    return next((p for p in map(shutil.which, _MUSESCORE) if p), None)


def musicxml_to_pdf(xml: Path, pdf: Path) -> Path | None:
    """Render MusicXML to PDF with MuseScore; None if MuseScore is not found."""
    exe = find_musescore()
    if exe is None:
        return None
    r = subprocess.run(
        [exe, "-o", str(pdf), str(xml)], capture_output=True, text=True, timeout=120
    )
    if r.returncode != 0 or not pdf.is_file():
        raise TranscriberError(f"MuseScore could not render the PDF: {r.stderr[-300:]}")
    return pdf
