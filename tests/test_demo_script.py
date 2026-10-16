"""scripts/make_demo_audio.py produces the demo inputs the README relies on."""

import subprocess
import sys
from pathlib import Path

import pretty_midi
import soundfile as sf

SCRIPT = Path(__file__).parents[1] / "scripts" / "make_demo_audio.py"


def test_make_demo_audio(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "single_notes").glob("MAPS_*_M*_synth.wav"))) == 88
    for name, seconds, n_notes in (("piece", 4.0, 12), ("varied", 30.0, None)):
        info = sf.info(tmp_path / f"{name}.wav")
        assert abs(info.duration - seconds) < 0.01
        notes = (
            pretty_midi.PrettyMIDI(str(tmp_path / f"{name}_reference.mid"))
            .instruments[0]
            .notes
        )
        assert len(notes) == n_notes if n_notes else len(notes) > 50
        assert max(n.end for n in notes) <= seconds + 1e-6
