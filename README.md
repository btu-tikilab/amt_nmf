# NMF Transcriber V1

Polyphonic piano transcription with non-negative matrix factorisation (NMF).
NMF Transcriber turns a piano recording (`.wav`, `.flac`, `.ogg`, `.mp3`,
`.aiff`) into a playable MIDI file. It does this with two methods and compares
them:

* **Standard NMF**: Euclidean multiplicative updates (Lee & Seung 2001), as in
  Smaragdis & Brown (2003).
* **Group-sparse NMF**: the l1,2 group penalty of O'Hanlon & Plumbley (2014).
  It keeps the number of active pitches per frame small, which removes
  overtone "ghost" notes.

On the 30-second demo piece (148 notes), group-sparse NMF raises onset
precision from 0.84 to 0.98 at the same recall (0.95). That cuts false notes
from about 27 to about 3, and onset F1 goes from 0.90 to 0.97.

Tracking issue: [#1 Implementation: NMF Transcriber V1](../../issues/1)

---

## Contents

1. [What you get](#1-what-you-get)
2. [Repository layout](#2-repository-layout)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Quick start: the demo in 7 commands](#5-quick-start-the-demo-in-7-commands)
6. [Transcribing your own audio](#6-transcribing-your-own-audio)
7. [Command reference](#7-command-reference)
8. [Output files](#8-output-files)
9. [How it works (the maths)](#9-how-it-works-the-maths)
10. [Evaluation and results](#10-evaluation-and-results)
11. [Known limitations](#11-known-limitations)
12. [Troubleshooting](#12-troubleshooting)
13. [Development](#13-development)
14. [Team](#14-team)
15. [References](#15-references)

---

## 1. What you get

| Command | What it does |
|---|---|
| `build-dict` | Learns a pitch dictionary **W** from single-note piano recordings (or builds a synthetic one) |
| `transcribe` | Converts a recording to MIDI with standard NMF, group-sparse NMF, or both (`--method compare`) |
| `evaluate` | Scores MIDI files against a reference MIDI (onset, onset+offset and frame P/R/F1, via mir_eval) |
| `report` | Writes a self-contained HTML report: piano rolls, scores, solver figures, convergence curves |
| `sheet` | Converts MIDI to MusicXML sheet music (and PDF if MuseScore is installed) |

All five commands run as `python -m nmf_cli <command>`. After installation they
also run as `nmf-transcriber <command>`.

---

## 2. Repository layout

```
amt_nmf/
├── pyproject.toml          dependencies, ruff and pytest settings
├── README.md               this file
├── .gitignore
├── src/
│   ├── nmf_core/           algorithm package (no printing or UI code)
│   │   ├── audio.py        librosa loading + constant-Q spectrogram V
│   │   ├── nmf.py          standard_nmf, group_sparse_nmf
│   │   ├── dictionary.py   build / save / load dictionary W (+ JSON sidecar)
│   │   ├── midi.py         activations H -> notes -> .mid, .npz dumps
│   │   ├── pipeline.py     transcribe(): validation, both solvers, summary.json
│   │   ├── evaluate.py     note- and frame-level scores (mir_eval)
│   │   └── errors.py       user-facing exception types
│   └── nmf_cli/            command line (python -m nmf_cli)
│       ├── cli.py          build-dict, transcribe, evaluate, report, sheet
│       ├── report.py       HTML comparison report (matplotlib)
│       └── sheet.py        MIDI -> MusicXML (music21), PDF via MuseScore
├── scripts/
│   ├── make_demo_audio.py  synthetic single notes + two test pieces
│   └── evaluate_midi.py    shortcut for `python -m nmf_cli evaluate`
├── tests/                  84 pytest tests (synthetic audio, no dataset needed)
└── examples/               saved outputs and scores of both demo pieces
```

These folders are created when you run the commands. They are **not** in the
repository (see `.gitignore`):

| Folder | Created by | Contents |
|---|---|---|
| `.venv/` | `python -m venv .venv` | your Python environment |
| `demo/` | `make_demo_audio.py` | synthetic audio and reference MIDI |
| `dictionaries/` | `build-dict` | `piano.npy` + `piano.json` |
| `outputs/` | `transcribe`, `evaluate`, `report`, `sheet` | results |
| `recordings/` | you | your own audio files to transcribe |
| `data/` | you | datasets such as MAPS (never commit these) |

---

## 3. Requirements

* **Python 3.10 to 3.14** (tested on 3.10, 3.13 and 3.14)
* **Git** (to clone the repository)
* About **1 GB** of disk space for the environment
* Optional: **MuseScore 4** (free, https://musescore.org) to view the sheet music
  and export PDF
* Optional: **ffmpeg** to read `.mp3` files or convert audio

Python packages (installed automatically): numpy, scipy, librosa, soundfile,
pretty_midi, mir_eval, matplotlib, music21, plus pytest and ruff for
development.

---

## 4. Installation

### Windows (PowerShell)

```powershell
git clone https://github.com/btu-tikilab/amt_nmf.git
cd amt_nmf
py -3.13 -m venv --upgrade-deps .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

* If PowerShell says *running scripts is disabled*, run this once, then
  activate again:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
* If `py` is not recognised, use `python` instead of `py -3.13`.
* Your prompt now starts with `(.venv)`. Run every later command in this
  activated terminal.

### macOS / Linux

```bash
git clone https://github.com/btu-tikilab/amt_nmf.git
cd amt_nmf
python3 -m venv --upgrade-deps .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Linux, if soundfile cannot find `libsndfile`:
`sudo apt-get install libsndfile1`

### Check the installation

```bash
python -m pytest
python -m nmf_cli --help
```

* pytest should end with **`84 passed`** (about one minute).
* `--help` should list `build-dict, transcribe, evaluate, report, sheet`.

The install takes 1 to 3 minutes and may look frozen while it downloads.
Do not press Ctrl+C until the prompt comes back.

---

## 5. Quick start: the demo in 7 commands

No dataset needed. The demo script synthesises 88 piano-like single notes and
two test pieces with their reference MIDI. Run the commands one at a time from
the project folder, with the environment activated.

**Step 1. Make the demo audio**

```bash
python scripts/make_demo_audio.py demo
```

Writes `demo/single_notes/` (88 notes), `demo/piece.wav` with
`demo/piece_reference.mid`, and `demo/varied.wav` with
`demo/varied_reference.mid`.

**Step 2. Build the dictionary**

```bash
python -m nmf_cli build-dict --audio-dir demo/single_notes --pitch-range 21 108 --output dictionaries/piano.npy
```

Ends with `Dictionary: 288 bins x 176 atoms (88 pitches)`.

**Step 3. Transcribe with both methods**

```bash
python -m nmf_cli transcribe --input demo/piece.wav --dict dictionaries/piano.npy --method compare --out-dir outputs/demo
```

Standard finds 13 notes, group-sparse 12. The reference has 12; standard adds
a B2 ghost note.

**Step 4. Score against the reference**

```bash
python -m nmf_cli evaluate --reference demo/piece_reference.mid --predicted outputs/demo/piece_group_sparse.mid --compare-std outputs/demo/piece_standard.mid --out-json outputs/demo/eval.json
```

```
file                             notes   onset P / R / F1     onset+offset P / R / F1   frame F1
------------------------------------------------------------------------------------------------
piece_standard.mid                  13   0.92 / 1.00 / 0.96   0.92 / 1.00 / 0.96        0.98
piece_group_sparse.mid              12   1.00 / 1.00 / 1.00   1.00 / 1.00 / 1.00        0.98
```

**Step 5. HTML report**

```bash
python -m nmf_cli report --std-mid outputs/demo/piece_standard.mid --gs-mid outputs/demo/piece_group_sparse.mid --summary outputs/demo/summary.json --eval-json outputs/demo/eval.json --reference demo/piece_reference.mid --out outputs/demo/report.html
```

Open it with `start outputs\demo\report.html` (Windows), `open` (macOS) or
`xdg-open` (Linux).

**Step 6. Sheet music**

```bash
python -m nmf_cli sheet --input outputs/demo/piece_group_sparse.mid --out outputs/demo/score.xml
```

Open `score.xml` in MuseScore.

**Step 7. The harder 30-second piece (optional)**

```bash
python -m nmf_cli transcribe --input demo/varied.wav --dict dictionaries/piano.npy --method compare --out-dir outputs/varied
python -m nmf_cli evaluate --reference demo/varied_reference.mid --predicted outputs/varied/varied_group_sparse.mid --compare-std outputs/varied/varied_standard.mid --out-json outputs/varied/eval.json
```

Onset F1: 0.90 for standard, 0.97 for group-sparse.

> **Copying commands on Windows:** paste each command as one line. If the
> prompt changes to `>>`, PowerShell thinks the command is unfinished: press
> Ctrl+C and paste it again. If a command shows `nmf\_cli` or `piece\_reference`,
> delete the backslashes: some apps add `\` before every `_` when you copy.

---

## 6. Transcribing your own audio

### 6.1 Prepare the audio file

| | Recommendation |
|---|---|
| **Format** | `.wav` or `.flac` work everywhere. `.ogg` and `.aiff` also work. `.mp3` needs ffmpeg installed. |
| **Channels / sample rate** | Anything. Stereo is mixed to mono and the audio is resampled to 22 050 Hz automatically. |
| **Content** | Solo piano. Other instruments, drums, singing or heavy reverb produce wrong notes. |
| **Length** | Start with 10 to 60 seconds. A 5-minute file can take several minutes with `--method compare`. |
| **File name** | Any name. Avoid spaces, or put the path in quotes. The name is reused for the outputs. |

Convert or trim with ffmpeg if needed:

```bash
# mp3 (or any format) -> mono wav
ffmpeg -i song.mp3 -ac 1 -ar 22050 recordings/song.wav

# keep only the first 30 seconds
ffmpeg -i recordings/song.wav -t 30 recordings/song_30s.wav
```

### 6.2 Put it in the `recordings/` folder

```powershell
mkdir recordings
copy "C:\Users\you\Music\my_piano.wav" recordings\
```

(On macOS/Linux: `mkdir -p recordings && cp ~/Music/my_piano.wav recordings/`.)

The file can live anywhere; `recordings/` just keeps things tidy and is
ignored by git, so your audio is never uploaded.

### 6.3 Choose a dictionary

The dictionary must match the *kind* of piano in your recording. This matters
more than any other setting.

| Option | When to use | Command |
|---|---|---|
| **A. Demo dictionary** | Trying the pipeline. Built from synthetic tones, so real recordings score lower. | Steps 1–2 of the quick start |
| **B. MAPS single notes** (recommended for real piano) | Real piano recordings. MAPS is a research dataset; download it separately and put it in `data/`. | see below |
| **C. Your own single notes** | You can record each key of the same piano you will transcribe. Best match. | see below |
| **D. Harmonic (synthetic)** | No recordings at all. Least accurate. | `python -m nmf_cli build-dict --harmonic --output dictionaries/harmonic.npy` |

**B. From MAPS** (`ISOL/NO` = isolated notes, normal playing):

```bash
python -m nmf_cli build-dict --audio-dir data/MAPS/ENSTDkCl/ISOL/NO --pitch-range 21 108 --output dictionaries/maps.npy
```

**C. From your own recordings.** Record one file per key (about 2 seconds,
let it ring) and name each file so the pitch can be read from it. Any of these
names work, and sub-folders are searched:

```
my_notes/
├── note_60.wav       MIDI number
├── midi61.wav        MIDI number
├── piano_62.wav      MIDI number
├── C4.wav            note name (C4 = MIDI 60)
├── Db5.wav           note name with flat
└── MAPS_ISOL_NO_F_S0_M64_AkPnBcht.wav    MAPS style
```

Several files for the same pitch (for example soft, medium and loud) are
pooled. Then build:

```bash
python -m nmf_cli build-dict --audio-dir my_notes --pitch-range 21 108 --output dictionaries/my_piano.npy
```

Keys with no recording cannot be transcribed; `build-dict` lists any missing
ones. To cover only part of the keyboard, narrow the range, for example
`--pitch-range 48 84` (C3 to C6).

Every dictionary is two files: `name.npy` (the matrix W) and `name.json` (the
settings it was built with). Always keep them together.

### 6.4 Transcribe

```bash
python -m nmf_cli transcribe --input recordings/my_piano.wav --dict dictionaries/maps.npy --method compare --out-dir outputs/my_piano
```

For **real recordings**, start with lower values than the demo defaults:

```bash
python -m nmf_cli transcribe --input recordings/my_piano.wav --dict dictionaries/maps.npy --method compare --threshold 0.05 --sparsity-weight 0.05 --out-dir outputs/my_piano
```

Then adjust:

| You see | Change |
|---|---|
| Many short wrong notes | Raise `--threshold` (0.1, 0.15) or `--sparsity-weight` (0.1 to 0.5) |
| Quiet notes missing | Lower `--threshold` (0.03 to 0.05) |
| Notes of a chord disappear in group-sparse | Lower `--sparsity-weight` |
| `converged: false` in summary.json | Raise `--max-iter` (e.g. 2000) |

Outputs in `outputs/my_piano/`: `my_piano_standard.mid`,
`my_piano_group_sparse.mid`, two `_H.npz` files and `summary.json`. Open the
`.mid` files in any MIDI player, DAW or MuseScore to listen.

### 6.5 Report and sheet music for your file

Without a reference MIDI, leave out `--eval-json` and `--reference`:

```bash
python -m nmf_cli report --std-mid outputs/my_piano/my_piano_standard.mid --gs-mid outputs/my_piano/my_piano_group_sparse.mid --summary outputs/my_piano/summary.json --out outputs/my_piano/report.html
python -m nmf_cli sheet --input outputs/my_piano/my_piano_group_sparse.mid --out outputs/my_piano/score.xml --title "My piano piece"
```

### 6.6 Scoring your file (only if you have a reference MIDI)

Scores need the true notes as a MIDI file, for example the `.mid` that comes
with every MAPS recording:

```bash
python -m nmf_cli evaluate --reference data/MAPS/ENSTDkCl/MUS/MAPS_MUS-bach_847_ENSTDkCl.mid --predicted outputs/bach/MAPS_MUS-bach_847_ENSTDkCl_group_sparse.mid --compare-std outputs/bach/MAPS_MUS-bach_847_ENSTDkCl_standard.mid --out-json outputs/bach/eval.json
```

### 6.7 Many files at once

Each file gets its own output folder.

Windows PowerShell:

```powershell
Get-ChildItem recordings\*.wav | ForEach-Object {
    python -m nmf_cli transcribe --input $_.FullName --dict dictionaries/maps.npy --method compare --out-dir "outputs/$($_.BaseName)"
}
```

macOS / Linux:

```bash
for f in recordings/*.wav; do
    name=$(basename "$f" .wav)
    python -m nmf_cli transcribe --input "$f" --dict dictionaries/maps.npy --method compare --out-dir "outputs/$name"
done
```

---

## 7. Command reference

Run any command with `-h` for its full help, e.g. `python -m nmf_cli transcribe -h`.
Add `--debug` right after `nmf_cli` to see a full traceback on unexpected errors.

### build-dict

```bash
python -m nmf_cli build-dict --audio-dir DIR --pitch-range MIN MAX --output FILE.npy
python -m nmf_cli build-dict --harmonic --output FILE.npy
```

| Flag | Meaning | Default |
|---|---|---|
| `--audio-dir` | folder of single-note recordings (supervised) | – |
| `--harmonic` | synthetic harmonic dictionary instead (no recordings) | off |
| `--pitch-range MIN MAX` | MIDI pitch range | `21 108` (full piano) |
| `--output` | output `.npy`; the `.json` sidecar is written next to it | required |
| `--atoms-per-pitch` | spectral atoms per pitch (group size) | `2` |
| `--sr`, `--hop-length` | analysis sample rate and hop | `22050`, `512` |
| `--fmin`, `--bins-per-octave`, `--n-bins` | CQT settings | `27.5`, `36`, `288` |

Give exactly one of `--audio-dir` or `--harmonic`.

### transcribe

```bash
python -m nmf_cli transcribe --input AUDIO --dict DICT.npy --method compare --out-dir DIR
```

| Flag | Meaning | Default |
|---|---|---|
| `--input` | audio file: `.wav`, `.flac`, `.ogg`, `.mp3`, `.aif`, `.aiff` | required |
| `--dict` | dictionary `.npy` from `build-dict` (its `.json` must sit next to it) | required |
| `--method` | `standard`, `group-sparse` or `compare` (both, one after the other) | `compare` |
| `--sparsity-weight` | λ, weight of the l1,2 group penalty, ≥ 0 | `0.2` |
| `--threshold` | Note-On threshold relative to the loudest activation, in (0, 1] | `0.1` |
| `--out-dir` | output folder for MIDI, `.npz` and `summary.json` | `outputs` |
| `--beta` | β-divergence of the group-sparse cost: 2 Euclidean, 1 KL, 0 Itakura-Saito | `2` |
| `--max-iter` | maximum NMF iterations | `1000` |
| `--tol` | relative cost decrease that counts as converged | `1e-4` |
| `--min-duration` | shortest note kept, in seconds | `0.05` |

The spectrogram settings always come from the dictionary's `.json` sidecar,
so the audio is analysed exactly as the dictionary was built.

### evaluate

```bash
python -m nmf_cli evaluate --reference REF.mid --predicted A.mid [B.mid ...] [--compare-std STD.mid] [--out-json FILE.json]
```

| Flag | Meaning | Default |
|---|---|---|
| `--reference` | ground-truth `.mid` | required |
| `--predicted` | one or more transcribed `.mid` files | required |
| `--compare-std` | a standard-NMF `.mid` scored alongside | – |
| `--out-json` | also save the scores as JSON (needed for the report) | – |
| `--onset-tolerance` | onset tolerance, seconds | `0.05` |
| `--offset-ratio` | offset tolerance as a fraction of note length | `0.2` |
| `--offset-min` | minimum offset tolerance, seconds | `0.05` |

### report

```bash
python -m nmf_cli report --std-mid STD.mid --gs-mid GS.mid --summary summary.json [--eval-json eval.json] [--reference REF.mid] --out report.html
```

| Flag | Meaning | Default |
|---|---|---|
| `--std-mid`, `--gs-mid` | the two transcriptions | required |
| `--summary` | `summary.json` from `transcribe` | required |
| `--eval-json` | scores from `evaluate` (adds the scores table) | – |
| `--reference` | reference `.mid` (adds a third piano roll) | – |
| `--out` | output `.html` | `outputs/report.html` |

### sheet

```bash
python -m nmf_cli sheet --input FILE.mid --out score.xml [--title "Title"] [--pdf]
```

| Flag | Meaning | Default |
|---|---|---|
| `--input` | `.mid` file | required |
| `--out` | output MusicXML `.xml` | required |
| `--title` | title printed on the score | `NMF transcription` |
| `--pdf` | also write a PDF (needs MuseScore on the PATH) | off |

Notes are quantised to a 16th-note grid.

---

## 8. Output files

For `--input piece.wav --method compare --out-dir outputs/demo`:

| File | Content |
|---|---|
| `piece_standard.mid` | standard NMF transcription, one piano track |
| `piece_group_sparse.mid` | group-sparse NMF transcription |
| `piece_standard_H.npz`, `piece_group_sparse_H.npz` | `H` (atoms × frames), `pitch_roll` (pitches × frames), `pitches`, `atom_pitch`, `times` (s), `sr`, `hop_length`, `notes` (JSON list), `cost_history` |
| `summary.json` | execution log (below) |
| `eval.json` | scores, from `evaluate --out-json` |
| `report.html` | HTML report, from `report` |
| `score.xml` | MusicXML sheet music, from `sheet` |

**summary.json** records the full run configuration (all flags, spectrogram
settings, dictionary shape and pitch range), the input duration, and for each
method: `converged`, `iterations`, `max_iterations`, `solve_time_s`,
`final_cost`, `relative_reconstruction_error`, `n_notes`, `pitches_detected`,
`pitch_roll_hoyer_sparsity` and `mean_active_pitches_per_frame`. In compare
mode a `comparison` block puts these side by side and lists pitches found by
only one method. `processing_time_s` is the total time.

How to read it:

* `relative_reconstruction_error` = ‖V − WH‖ / ‖V‖ of each solver's own fit.
  For group-sparse this is the **penalised** fit, so it is expected to be
  higher: the penalty gives up some reconstruction to switch off unneeded
  pitches. The unpenalised re-fit used for note endings reports its own error
  under `offset_refit`.
* Sparsity figures describe the activations, not accuracy. For accuracy,
  use `evaluate` against a reference.

Load activations in Python:

```python
import numpy as np

z = np.load("outputs/demo/piece_group_sparse_H.npz")
roll, times, pitches = z["pitch_roll"], z["times"], z["pitches"]
```

---

## 9. How it works (the maths)

```
audio ─► mono, 22 050 Hz ─► magnitude CQT  V (288 × T)
                                   │
              dictionary W (288 × 176) from build-dict
                                   │
          solve V ≈ W·H for H ≥ 0 (standard  and/or  group-sparse)
                                   │
         H ─► piano roll ─► peaks/thresholds ─► notes ─► .mid
```

### Spectrogram V
Audio is resampled to 22 050 Hz and converted to a magnitude constant-Q
transform: 288 bins from A0 (27.5 Hz), 36 bins per octave (one bin = 1/3
semitone), hop 512 samples (23.2 ms). On a log-frequency axis every note's
harmonics have the same spacing, so a pitch looks like a shifted copy of its
neighbours. V (bins × frames) is scaled to a maximum of 1, so
`--sparsity-weight` does not depend on loudness.

Why CQT and not an STFT: an STFT with n_fft = 2048 at 22 050 Hz has 10.8 Hz
bins, wider than the gap between neighbouring notes below about C3. CQT bins
follow the piano's semitone spacing.

### Factorisation V ≈ W·H
* **W** (F × K) is the dictionary. Each column is a spectral template (an
  *atom*). Every pitch owns `atoms_per_pitch` atoms (default 2, for example a
  bright attack spectrum and a duller decay spectrum): 88 × 2 = 176 atoms.
* **H** (K × T) holds the activations: how strongly each atom sounds in each
  frame. Summing a pitch's atom rows gives its piano-roll row.

W is fixed during transcription; only H is estimated, so H ≥ 0 *is* the
transcription.

### Building W
For each pitch, the single-note clips are trimmed of silence, converted to
CQT, normalised and joined. A small NMF extracts `atoms_per_pitch` atoms,
which are normalised to unit length and become that pitch's group of columns.

### Standard (Euclidean) NMF
Minimises ½‖V − WH‖²_F with the multiplicative update

```
H ← H ⊙ (WᵀV) / (WᵀW H)
```

which keeps H non-negative and never increases the cost.

### Group-sparse NMF and the l1,2 penalty
The atoms are split into groups G_p, one per pitch. The cost becomes

```
D_β(V | WH) + λ Σ_t Σ_p ‖H[G_p, t]‖₂
```

The inner l2 norm couples the atoms of one pitch (they switch on and off
together). The outer sum is an l1 norm over pitches, which drives whole
pitches to exactly zero, so each frame is explained by few notes. This removes
the typical NMF error of explaining a note's overtones with notes an octave or
a fifth above. The multiplicative update is

```
H ← H ⊙ [ Wᵀ(V ⊙ Λ^(β−2)) / (Wᵀ Λ^(β−1) + λ H / ‖H[G(k), t]‖₂) ]^φ(β),   Λ = WH
```

with φ(β) = 1/(2−β) for β < 1 and 1 for 1 ≤ β ≤ 2. As in the paper, the first
10 iterations run without the penalty (warm start). The default `--beta 2`
uses the same Euclidean cost as standard NMF, so `--method compare` isolates
the effect of the penalty. λ is `--sparsity-weight`; λ = 0 gives back
standard NMF exactly.

Iterations stop when the cost improves by less than `--tol` over 10
iterations, or at `--max-iter`.

### From H to MIDI
1. Sum each pitch's atom rows of H and divide by the global maximum.
2. Every peak at least `--threshold` high, and at least `--threshold` above
   the surrounding valleys, is a **Note-On**, timed at the half-rise point of
   the peak, so repeated notes of the same pitch re-attack.
3. **Note-Off** when the activation falls below `threshold / 2` (hysteresis)
   or at the next Note-On of that pitch. Notes shorter than `--min-duration`
   are dropped.
4. Frame f becomes time `f × hop_length / sample_rate` seconds. Velocity
   follows the peak height.
5. **Group-sparse only:** the penalty also shrinks each note's quiet tail,
   which would end notes too early. So group-sparse decides which notes exist
   and when they start, and each Note-Off comes from an unpenalised re-fit
   restricted to the pitches it selected. `summary.json` reports it under
   `offset_refit`.

---

## 10. Evaluation and results

### Metrics
`evaluate` uses mir_eval with the standard MIREX rules:

* **Onset P/R/F1:** correct pitch, onset within ±50 ms.
* **Onset + offset P/R/F1:** additionally, the offset within 50 ms or 20% of
  the reference note's length, whichever is larger.
* **Frame F1:** piano rolls compared every 10 ms.

### Test data
`scripts/make_demo_audio.py` synthesises piano-like tones by additive
synthesis (stretched, inharmonic partials with decay):

* `piece.wav`: 4 s, 12 notes with chords, same timbre as the dictionary
  (checks the pipeline).
* `varied.wav`: 30 s, 148 notes, random 1–4 note chords over MIDI 36–88,
  played with a *different* timbre (brighter, more inharmonic, random
  loudness), so it tests how the methods generalise.

### Results (default settings)

| Input | Method | Notes (ref) | Onset P / R / F1 | Onset+offset P / R / F1 | Frame F1 |
|---|---|---|---|---|---|
| `piece` | Standard | 13 (12) | 0.92 / 1.00 / 0.96 | 0.92 / 1.00 / 0.96 | 0.98 |
| | Group-sparse | 12 (12) | **1.00 / 1.00 / 1.00** | **1.00 / 1.00 / 1.00** | 0.98 |
| `varied` | Standard | 167 (148) | 0.84 / 0.95 / 0.90 | 0.72 / 0.81 / 0.76 | 0.91 |
| | Group-sparse | 144 (148) | **0.98 / 0.95 / 0.97** | **0.83 / 0.81 / 0.82** | 0.92 |

Group sparsity removes about 24 false notes on `varied` without losing a
single correct one (recall 0.95 for both). All outputs, reports and scores
are saved in [`examples/`](examples/README.md).

---

## 11. Known limitations

* **Synthetic tests score higher than real recordings.** On three 30-second
  real concert-grand excerpts from MAESTRO, onset F1 was about 0.4 to 0.6,
  and group-sparse was not always better: at the default λ it also removes
  real notes. For real recordings start with
  `--threshold 0.05 --sparsity-weight 0.05`.
* **Piano only.** Notes that swell rather than start with a hit (flute,
  strings, voice) are detected late. Mixes with other instruments or heavy
  reverb give many wrong notes.
* **Fixed dictionary.** W is learned once and not adapted to the recording,
  so a different piano or room lowers accuracy.
* **Sustain pedal and very soft notes.** Pedalled notes may end early, and
  notes far quieter than the loudest one may be missed.
* **Velocity** comes from activation height and is only approximate.
* **Speed.** Group-sparse on a 5-minute file can take several minutes; use
  `--method standard` or a shorter clip for quick checks.

---

## 12. Troubleshooting

Errors print as one `Error: ...` line and exit with status 1. Nothing is
written to `--out-dir` for a failed run.

### Program messages

| Message | Cause and fix |
|---|---|
| `Dimension mismatch: the dictionary has N frequency bins ... the input spectrogram has M` | The `.npy` and `.json` come from different builds. Rebuild with `build-dict` and keep both files together. |
| `Dictionary metadata sidecar not found` | The `.json` next to the `.npy` is missing. Copy it or rebuild. |
| `Dictionary file ... is corrupted` | Truncated or not a NumPy array. Rebuild. |
| `Audio file not found` / `Cannot read audio file` | Check the path (use quotes if it has spaces). If decoding fails, convert to 16-bit WAV. |
| `None of the ... audio files ... is named with a MIDI pitch` | Rename single-note files like `note_60.wav` or `C4.wav` (section 6.3). |
| `--threshold must be in (0, 1]`, `--sparsity-weight must be >= 0` | Value out of range. |
| `.mp3` cannot be read | Install ffmpeg, or convert to WAV (section 6.1). |
| Empty or almost empty MIDI | Lower `--threshold`; check the dictionary covers the notes played. |
| Many short wrong notes | Raise `--threshold` or `--sparsity-weight`; use a dictionary from the same kind of piano. |
| `Unexpected error` | Re-run with `python -m nmf_cli --debug <command> ...` and report the traceback. |

### Setup problems (mostly Windows)

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'nmf_core'` or `'librosa'` | The environment is not active or the package is not installed: `.venv\Scripts\Activate.ps1`, then `pip install -e ".[dev]"`. |
| `No module named pytest` | Install with the extras: `pip install -e ".[dev]"`. |
| `invalid choice: 'evaluate'` | You are running an old copy. Check which folder you are in, and reinstall there. |
| Works, then fails after **moving or renaming** the folder | A `.venv` cannot be moved. Close VS Code, delete `.venv`, `__pycache__` folders and `*.egg-info`, then recreate the environment (section 4). |
| `Access is denied: ...\.venv\Scripts\ruff.exe` | VS Code's Ruff extension is using that file. Close VS Code completely (or restart the PC), then delete or reinstall from a separate PowerShell window. |
| `ERROR: Operation cancelled by user` | Ctrl+C was pressed during the install. Run `pip install -e ".[dev]"` again and wait. |
| `[notice] A new release of pip is available` | Not an error. Turn it off for all projects: `pip config --user set global.disable-pip-version-check true`. |
| Prompt shows `>>` | The pasted command was incomplete. Press Ctrl+C and paste it as one line. |
| `No module named nmf\_cli` | Backslashes were added when copying. Remove every `\` before `_`. |
| Folder nested after unzipping (`name\name\`) | Windows "Extract All" adds a level. `cd` into the folder that contains `pyproject.toml`. |
| `sndfile library not found` (Linux) | `sudo apt-get install libsndfile1` |

---

## 13. Development

### Tests and linting

```bash
python -m pytest              # 84 tests, about one minute
ruff check .                  # lint
ruff format --check .         # formatting
ruff format .                 # apply formatting
```

The tests cover: W·H dimensions and non-negativity, that group sparsity gives
a sparser H than standard updates, exact dictionary save/load round-trip,
rejection of a mismatched dictionary, valid non-empty MIDI with the expected
pitches and timing, and end-to-end CLI runs through subprocess.

### Code style
* Ruff with line length 88 and the rules in `pyproject.toml`
  (`E, F, W, I, B, UP, SIM, NPY`).
* Algorithms live in `nmf_core` with no printing; all user-facing output is in
  `nmf_cli`.
* Errors shown to users are `TranscriberError` subclasses with a clear message.

### Contributing
1. Create a branch: `git checkout -b feature/short-name`
2. Make the change, add or update tests.
3. Run `python -m pytest` and `ruff check .` until both pass.
4. Push and open a pull request against `main`, referencing the issue
   (e.g. `Closes #2`).

Never commit `.venv/`, `outputs/`, `demo/`, `dictionaries/`, `recordings/`,
`data/` or dataset audio; `.gitignore` already excludes them.

---

## 14. Team

BTU TikiLab, Automatic Music Transcription with NMF.

* Supervisor: Prasanth Varadarajan
* Purnakash Gadde
* Varun Kankanala
* Venkata Sumathi yadala


---

## 15. References

* K. O'Hanlon and M. D. Plumbley, "Polyphonic piano transcription using
  non-negative matrix factorisation with group sparsity", ICASSP 2014.
* P. Smaragdis and J. C. Brown, "Non-negative matrix factorization for
  polyphonic music transcription", WASPAA 2003.
* D. D. Lee and H. S. Seung, "Algorithms for non-negative matrix
  factorization", NIPS 2001.
* C. Févotte and J. Idier, "Algorithms for nonnegative matrix factorization
  with the β-divergence", Neural Computation 2011.
* V. Emiya, R. Badeau and B. David, "Multipitch estimation of piano sounds
  using a new probabilistic spectral smoothness principle" (MAPS dataset),
  IEEE TASLP 2010.
* C. Raffel et al., "mir_eval: A transparent implementation of common MIR
  metrics", ISMIR 2014.
