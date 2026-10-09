"""
src/nmf_core/sheet.py
─────────────────────────────────────────────────────────────────
Sheet music export — convert MIDI to MusicXML.

The output .xml file can be opened in:
  - MuseScore (free, Windows/Mac/Linux)
  - Sibelius / Finale
  - Flat.io (web-based)
  - Any other MusicXML-compatible notation editor

Uses music21 which handles quantisation, key/time signature
inference, and multi-voice separation automatically.
"""

from __future__ import annotations

from pathlib import Path


def midi_to_musicxml(
    midi_path:   str | Path,
    out_path:    str | Path,
    title:       str = "NMF Transcription",
    composer:    str = "NMF Transcriber V1",
    quantise:    bool = True,
) -> Path:
    """
    Convert a MIDI file to MusicXML sheet music.

    Parameters
    ----------
    midi_path : input .mid file
    out_path  : output .xml file path
    title     : score title embedded in the MusicXML
    composer  : composer string embedded in the MusicXML
    quantise  : if True, snap notes to nearest 16th note grid

    Returns
    -------
    Path to the written .xml file

    Raises
    ------
    FileNotFoundError if midi_path does not exist
    ImportError      if music21 is not installed
    """
    try:
        from music21 import converter, metadata
    except ImportError as exc:
        raise ImportError(
            "music21 is required for sheet music export.\n"
            "Run:  pip install music21"
        ) from exc

    midi_path = Path(midi_path)
    out_path  = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    # Parse MIDI
    score = converter.parse(str(midi_path))

    # Add metadata
    score.metadata         = metadata.Metadata()
    score.metadata.title   = title
    score.metadata.composer = composer

    # Quantise to 16th note grid if requested
    if quantise:
        score = score.quantize(
            quarterLengthDivisors=(4,),   # 16th note = quarter / 4
            processOffsets       = True,
            processDurations     = True,
            inPlace              = False,
        )

    # Write MusicXML
    score.write("musicxml", fp=str(out_path))

    print(f"  MusicXML saved → {out_path}")
    return out_path


def midi_to_pdf(
    midi_path: str | Path,
    out_path:  str | Path,
    title:     str = "NMF Transcription",
) -> Path:
    """
    Convert MIDI → MusicXML → PDF (requires MuseScore installed).

    MuseScore must be on the system PATH for PDF rendering.
    Falls back to MusicXML if MuseScore is not found.
    """
    import subprocess

    xml_path = Path(out_path).with_suffix(".xml")
    midi_to_musicxml(midi_path, xml_path, title=title)

    # Try MuseScore for PDF rendering
    for cmd in ("mscore", "musescore", "musescore3", "musescore4"):
        try:
            result = subprocess.run(
                [cmd, "-o", str(out_path), str(xml_path)],
                capture_output=True, timeout=60,
            )
            if result.returncode == 0 and Path(out_path).exists():
                print(f"  PDF saved → {out_path}")
                return Path(out_path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    print(
        f"  MuseScore not found — MusicXML saved instead: {xml_path}\n"
        f"  Open {xml_path} in MuseScore to export PDF manually."
    )
    return xml_path
