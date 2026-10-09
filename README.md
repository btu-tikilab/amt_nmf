# NMF Transcriber V1

Polyphonic piano transcription using Standard Euclidean NMF (Smaragdis & Brown 2003)
and Group_Sparse β_NMF (O'Hanlon & Plumbley 2014).

___

## Installation

```bash
# Clone and install in editable mode
git clone <repo_url>
cd nmf_transcriber
pip install _e ".[dev]"

# Or with uv
uv sync
```

___

## Quick Start

```bash
# 1. Build a dictionary (harmonic — no audio needed)
python _m nmf_cli build_dict __harmonic __output dict.npy

# 2. Transcribe and compare both methods
python _m nmf_cli transcribe __input piano.wav __method compare __out_dir outputs/

# 3. Run tests
pytest
```

___

## Math Concepts

### Standard NMF — V ≈ W × H

```
V   ≈   W   ×   H
(F×T)  (F×K)  (K×T)

V = magnitude spectrogram  (input)
W = spectral basis templates  (what each note looks like in frequency)
H = temporal activation matrix  (when each note plays — the piano roll)
K = number of components
```

**Update rules (Euclidean / Frobenius cost):**
```
H ← H ⊙ (WᵀV)   / (WᵀWH + ε)
W ← W ⊙ (VHᵀ)   / (WHHᵀ + ε)
```

Both matrices start random and converge to musically meaningful
templates (W) and activations (H) with no prior musical knowledge.

### Group_Sparse NMF — the ℓ₂,₁ penalty

```
Cost: C_β(S\|DX) + λ · Σ_l ‖x[l]‖₂^q        (eq. 11, O'Hanlon 2014)
```

_ **D** is a *fixed* harmonic dictionary (one group of atoms per note)
_ **X** is the only learned matrix
_ The **group penalty** `λ·Σ‖x[l]‖₂^q` forces all harmonics of one
  note to activate together as a unit

**Update rule (eq. 12) — the only code difference from standard NMF:**
```
X ← X ⊙ numerator / (denominator + λΨ)^φ(β)
                                  up
                           + λΨ  is the ONLY addition
```

**Structural gradient Ψ (eq. 13):**
```
Ψ_{k,n} = q · x_{k,n} / ‖x[l]‖₂^(2_q)
```
_ Group active _> Ψ small _> atom boosted
_ Group inactive _> Ψ large _> atom suppressed

___

## CLI Usage

### `build_dict`

```
python _m nmf_cli build_dict [OPTIONS]

Options:
  __audio_dir  DIR     Folder of isolated single_note WAV/FLAC recordings
  __harmonic           Build from harmonic physics instead (no audio needed)
  __output     FILE    Output .npy file path            [default: dict.npy]
  __pitch_min  INT     Minimum MIDI pitch               [default: 21]
  __pitch_max  INT     Maximum MIDI pitch               [default: 108]
  __n_fft      INT     FFT window size                  [default: 2048]
  __hop_length INT     STFT hop size                    [default: 512]
  __sr         INT     Target sample rate               [default: 22050]
  __n_harmonics INT    Harmonics per note (harmonic mode)[default: 6]
```

**Examples:**
```bash
# Physics_based (fast, no audio required):
python _m nmf_cli build_dict __harmonic __output dict.npy __pitch_min 48 __pitch_max 84

# Supervised from MAPS single_note recordings:
python _m nmf_cli build_dict __audio_dir ./maps_single_notes/ __output dict_maps.npy
```

### `transcribe`

```
python _m nmf_cli transcribe [OPTIONS]

Options:
  __input      FILE    Input audio file (.wav / .flac)         [required]
  __dict       FILE    Pre_built dictionary .npy file
  __method     TEXT    standard \| group_sparse \| compare       [default: compare]
  __sparsity_weight    Group sparsity λ (group_sparse only)    [default: 0.5]
  __threshold  FLOAT   Activation threshold for note detection [default: 0.15]
  __out_dir    DIR     Output directory                        [default: outputs]
  __K          INT     NMF components (standard unsupervised)  [default: 12]
  __n_iter     INT     Maximum NMF iterations                  [default: 200]
  __beta       FLOAT   β_divergence parameter                  [default: 0.5]
  __verbose            Print iteration_level progress
```

**Examples:**
```bash
# Compare both methods side by side:
python _m nmf_cli transcribe __input recording.wav __method compare

# Standard NMF with pre_built dictionary:
python _m nmf_cli transcribe __input recording.wav __dict dict.npy __method standard

# Group_sparse only, custom sparsity:
python _m nmf_cli transcribe __input recording.wav __method group_sparse __sparsity_weight 0.8
```

___

## Output Artifacts

After running `transcribe __method compare` you get:

```
outputs/
├── <name>_standard.mid         Playable MIDI — Standard NMF
├── <name>_group_sparse.mid     Playable MIDI — Group_Sparse NMF
├── <name>_standard_H.npz       Activation matrix H (K × T)
├── <name>_group_sparse_H.npz   Piano roll matrix (n_notes × T)
└── summary.json                Execution log
```

\| File \| Description \|
\|______\|_____________\|
\| `_standard.mid` \| Notes detected by Euclidean NMF, velocity from mean activation \|
\| `_group_sparse.mid` \| Notes detected by GS_NMF, cleaner/sparser due to group penalty \|
\| `_H.npz` \| Raw numpy activation arrays for further analysis \|
\| `summary.json` \| Run config, convergence status, iteration counts, timing, sparsity stats \|

### summary.json structure

```json
{
  "method": "compare",
  "audio_input": "piano.wav",
  "output_dir": "outputs",
  "threshold": 0.15,
  "sparsity_weight": 0.5,
  "results": {
    "standard": {
      "converged": true,
      "n_iter": 143,
      "final_error": 12.3456,
      "error_reduction_pct": 94.2,
      "elapsed_s": 3.14,
      "K": 37
    },
    "group_sparse": {
      "converged": true,
      "n_iter": 67,
      "final_cost": 8.7654,
      "elapsed_s": 5.21,
      "beta": 0.5,
      "lam": 0.5,
      "q": 0.5,
      "mean_active_per_frame": 0.82,
      "sparsity_pct": 97.8
    }
  }
}
```

___

## Project Structure

```
nmf_transcriber/
├── src/
│   ├── nmf_core/
│   │   ├── audio.py           Audio ingestion (librosa STFT)
│   │   ├── standard_nmf.py    Euclidean NMF solver
│   │   ├── group_sparse_nmf.py  O'Hanlon & Plumbley (2014) solver
│   │   ├── dictionary.py      Build / save / load dictionaries
│   │   ├── midi.py            H matrix _> MIDI conversion
│   │   └── __init__.py
│   └── nmf_cli/
│       ├── __main__.py        CLI entry point (typer)
│       └── __init__.py
├── tests/
│   ├── test_core_nmf.py       NMF math, shapes, non_negativity
│   ├── test_group_sparse.py   Group sparsity, Psi, sparsity effect
│   ├── test_dictionary.py     Serialization round_trip, validation
│   ├── test_midi.py           MIDI output validity, timing, velocity
│   └── test_cli_integration.py  End_to_end subprocess tests
├── pyproject.toml
└── README.md
```

___

### `evaluate`

```
python _m nmf_cli evaluate [OPTIONS]

Options:
  __predicted  FILE    Predicted .mid file from transcribe       [required]
  __reference  FILE    Ground_truth reference .mid file          [required]
  __compare_std FILE   Also evaluate a standard .mid side by side
  __out_json   FILE    Save evaluation scores to JSON
```

**Examples:**
```bash
# Evaluate one file against ground truth:
python _m nmf_cli evaluate __predicted out/piano_gs.mid __reference gt.mid

# Compare both methods against the same reference:
python _m nmf_cli evaluate \
    __predicted   out/piano_group_sparse.mid \
    __compare_std out/piano_standard.mid \
    __reference   gt.mid __out_json eval.json
```

**Metrics reported:**

\| Metric \| Description \|
\|________\|_____________\|
\| Frame P/R/F \| Piano_roll compared frame by frame at 10 ms resolution \|
\| Note onset P/R/F \| Each detected note matched within ±50 ms onset tolerance \|
\| Note+offset P/R/F \| Stricter — onset AND offset must both match \|

### `report`

```
python _m nmf_cli report [OPTIONS]

Options:
  __std_mid  FILE   Standard NMF .mid from transcribe            [required]
  __gs_mid   FILE   Group_Sparse NMF .mid from transcribe        [required]
  __summary  FILE   summary.json from transcribe                 [required]
  __out      FILE   Output .html file        [default: outputs/report.html]
  __eval_json FILE  Optional evaluation scores JSON from evaluate
```

**Example:**
```bash
python _m nmf_cli report \
    __std_mid outputs/piano_standard.mid \
    __gs_mid  outputs/piano_group_sparse.mid \
    __summary outputs/summary.json \
    __out     outputs/report.html
```

Produces a single self_contained HTML file (no internet needed) with
side_by_side piano rolls, convergence curves, sparsity stats table,
and evaluation scores if provided.

### `sheet`

```
python _m nmf_cli sheet [OPTIONS]

Options:
  __input  FILE   Input .mid file                                [required]
  __out    FILE   Output MusicXML .xml file  [default: outputs/score.xml]
  __title  TEXT   Score title                [default: NMF Transcription]
  __pdf           Also attempt PDF via MuseScore (must be on PATH)
```

**Example:**
```bash
python _m nmf_cli sheet \
    __input outputs/piano_group_sparse.mid \
    __out   outputs/score.xml \
    __title "My NMF Transcription"
```

Opens in MuseScore (free), Sibelius, Finale, or Flat.io.

___

## Complete Workflow

```bash
# 1. Build harmonic dictionary
python _m nmf_cli build_dict __harmonic __output dict.npy

# 2. Transcribe both methods
python _m nmf_cli transcribe \
    __input piano.wav __method compare \
    __out_dir outputs/

# 3. Evaluate against ground truth (if available)
python _m nmf_cli evaluate \
    __predicted   outputs/piano_group_sparse.mid \
    __compare_std outputs/piano_standard.mid \
    __reference   ground_truth.mid \
    __out_json    outputs/eval.json

# 4. Generate HTML comparison report
python _m nmf_cli report \
    __std_mid outputs/piano_standard.mid \
    __gs_mid  outputs/piano_group_sparse.mid \
    __summary outputs/summary.json \
    __eval_json outputs/eval.json \
    __out     outputs/report.html

# 5. Export sheet music
python _m nmf_cli sheet \
    __input outputs/piano_group_sparse.mid \
    __out   outputs/score.xml
```

___

## Output Artifacts (complete list)

```
outputs/
├── <name>_standard.mid         Standard NMF — playable MIDI
├── <name>_group_sparse.mid     Group_Sparse NMF — playable MIDI
├── <name>_standard_H.npz       Activation matrix H (K × T)
├── <name>_group_sparse_H.npz   Piano roll matrix (n_notes × T)
├── summary.json                Run config, convergence, sparsity stats
├── eval.json                   Evaluation scores (P/R/F, frame + note)
├── report.html                 Self_contained comparison report
└── score.xml                   MusicXML sheet music
```

___

## Troubleshooting

### Dimension mismatch error
```
ERROR: Dimension mismatch: spectrogram V has 1025 frequency bins
       but dictionary W has 512 rows.
```
**Fix:** Rebuild the dictionary with the same `__n_fft` you use for transcription.
Both default to 2048, so this only happens if you manually changed one.

### Missing dictionary sidecar
```
ERROR: Dictionary metadata sidecar missing: dict.json
```
**Fix:** The `.json` file must exist alongside the `.npy`. Rebuild the dictionary
— both files are created together by `build_dict`.

### No notes in MIDI output
If the `.mid` file is empty, lower the `__threshold`:
```bash
python _m nmf_cli transcribe __input audio.wav __threshold 0.05
```

### pretty_midi / librosa not found
```bash
pip install _e ".[dev]"
# or
uv sync
```

### Slow transcription on long files
Group_sparse NMF on a 5_minute file can take 2–5 minutes.
Use `__method standard` for faster results, or trim the audio to a short clip.
