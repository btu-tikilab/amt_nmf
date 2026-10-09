# NMF Transcriber V1

Polyphonic piano transcription using Standard Euclidean NMF (Smaragdis & Brown 2003)
and Group-Sparse β-NMF (O'Hanlon & Plumbley 2014).

---

## Installation

```bash
# Clone and install in editable mode
git clone <repo-url>
cd nmf-transcriber
pip install -e ".[dev]"

# Or with uv
uv sync
```

---

## Quick Start

```bash
# 1. Build a dictionary (harmonic — no audio needed)
python -m nmf_cli build-dict --harmonic --output dict.npy

# 2. Transcribe and compare both methods
python -m nmf_cli transcribe --input piano.wav --method compare --out-dir outputs/

# 3. Run tests
pytest
```

---

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

### Group-Sparse NMF — the ℓ₂,₁ penalty

```
Cost: C_β(S|DX) + λ · Σ_l ‖x[l]‖₂^q        (eq. 11, O'Hanlon 2014)
```

- **D** is a *fixed* harmonic dictionary (one group of atoms per note)
- **X** is the only learned matrix
- The **group penalty** `λ·Σ‖x[l]‖₂^q` forces all harmonics of one
  note to activate together as a unit

**Update rule (eq. 12) — the only code difference from standard NMF:**
```
X ← X ⊙ numerator / (denominator + λΨ)^φ(β)
                                  ↑
                           + λΨ  is the ONLY addition
```

**Structural gradient Ψ (eq. 13):**
```
Ψ_{k,n} = q · x_{k,n} / ‖x[l]‖₂^(2-q)
```
- Group active → Ψ small → atom boosted
- Group inactive → Ψ large → atom suppressed

---

## CLI Usage

### `build-dict`

```
python -m nmf_cli build-dict [OPTIONS]

Options:
  --audio-dir  DIR     Folder of isolated single-note WAV/FLAC recordings
  --harmonic           Build from harmonic physics instead (no audio needed)
  --output     FILE    Output .npy file path            [default: dict.npy]
  --pitch-min  INT     Minimum MIDI pitch               [default: 21]
  --pitch-max  INT     Maximum MIDI pitch               [default: 108]
  --n-fft      INT     FFT window size                  [default: 2048]
  --hop-length INT     STFT hop size                    [default: 512]
  --sr         INT     Target sample rate               [default: 22050]
  --n-harmonics INT    Harmonics per note (harmonic mode)[default: 6]
```

**Examples:**
```bash
# Physics-based (fast, no audio required):
python -m nmf_cli build-dict --harmonic --output dict.npy --pitch-min 48 --pitch-max 84

# Supervised from MAPS single-note recordings:
python -m nmf_cli build-dict --audio-dir ./maps_single_notes/ --output dict_maps.npy
```

### `transcribe`

```
python -m nmf_cli transcribe [OPTIONS]

Options:
  --input      FILE    Input audio file (.wav / .flac)         [required]
  --dict       FILE    Pre-built dictionary .npy file
  --method     TEXT    standard | group-sparse | compare       [default: compare]
  --sparsity-weight    Group sparsity λ (group-sparse only)    [default: 0.5]
  --threshold  FLOAT   Activation threshold for note detection [default: 0.15]
  --out-dir    DIR     Output directory                        [default: outputs]
  --K          INT     NMF components (standard unsupervised)  [default: 12]
  --n-iter     INT     Maximum NMF iterations                  [default: 200]
  --beta       FLOAT   β-divergence parameter                  [default: 0.5]
  --verbose            Print iteration-level progress
```

**Examples:**
```bash
# Compare both methods side by side:
python -m nmf_cli transcribe --input recording.wav --method compare

# Standard NMF with pre-built dictionary:
python -m nmf_cli transcribe --input recording.wav --dict dict.npy --method standard

# Group-sparse only, custom sparsity:
python -m nmf_cli transcribe --input recording.wav --method group-sparse --sparsity-weight 0.8
```

---

## Output Artifacts

After running `transcribe --method compare` you get:

```
outputs/
├── <name>_standard.mid         Playable MIDI — Standard NMF
├── <name>_group_sparse.mid     Playable MIDI — Group-Sparse NMF
├── <name>_standard_H.npz       Activation matrix H (K × T)
├── <name>_group_sparse_H.npz   Piano roll matrix (n_notes × T)
└── summary.json                Execution log
```

| File | Description |
|------|-------------|
| `_standard.mid` | Notes detected by Euclidean NMF, velocity from mean activation |
| `_group_sparse.mid` | Notes detected by GS-NMF, cleaner/sparser due to group penalty |
| `_H.npz` | Raw numpy activation arrays for further analysis |
| `summary.json` | Run config, convergence status, iteration counts, timing, sparsity stats |

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

---

## Project Structure

```
nmf-transcriber/
├── src/
│   ├── nmf_core/
│   │   ├── audio.py           Audio ingestion (librosa STFT)
│   │   ├── standard_nmf.py    Euclidean NMF solver
│   │   ├── group_sparse_nmf.py  O'Hanlon & Plumbley (2014) solver
│   │   ├── dictionary.py      Build / save / load dictionaries
│   │   ├── midi.py            H matrix → MIDI conversion
│   │   └── __init__.py
│   └── nmf_cli/
│       ├── __main__.py        CLI entry point (typer)
│       └── __init__.py
├── tests/
│   ├── test_core_nmf.py       NMF math, shapes, non-negativity
│   ├── test_group_sparse.py   Group sparsity, Psi, sparsity effect
│   ├── test_dictionary.py     Serialization round-trip, validation
│   ├── test_midi.py           MIDI output validity, timing, velocity
│   └── test_cli_integration.py  End-to-end subprocess tests
├── pyproject.toml
└── README.md
```

---

### `evaluate`

```
python -m nmf_cli evaluate [OPTIONS]

Options:
  --predicted  FILE    Predicted .mid file from transcribe       [required]
  --reference  FILE    Ground-truth reference .mid file          [required]
  --compare-std FILE   Also evaluate a standard .mid side by side
  --out-json   FILE    Save evaluation scores to JSON
```

**Examples:**
```bash
# Evaluate one file against ground truth:
python -m nmf_cli evaluate --predicted out/piano_gs.mid --reference gt.mid

# Compare both methods against the same reference:
python -m nmf_cli evaluate \
    --predicted   out/piano_group_sparse.mid \
    --compare-std out/piano_standard.mid \
    --reference   gt.mid --out-json eval.json
```

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| Frame P/R/F | Piano-roll compared frame by frame at 10 ms resolution |
| Note onset P/R/F | Each detected note matched within ±50 ms onset tolerance |
| Note+offset P/R/F | Stricter — onset AND offset must both match |

### `report`

```
python -m nmf_cli report [OPTIONS]

Options:
  --std-mid  FILE   Standard NMF .mid from transcribe            [required]
  --gs-mid   FILE   Group-Sparse NMF .mid from transcribe        [required]
  --summary  FILE   summary.json from transcribe                 [required]
  --out      FILE   Output .html file        [default: outputs/report.html]
  --eval-json FILE  Optional evaluation scores JSON from evaluate
```

**Example:**
```bash
python -m nmf_cli report \
    --std-mid outputs/piano_standard.mid \
    --gs-mid  outputs/piano_group_sparse.mid \
    --summary outputs/summary.json \
    --out     outputs/report.html
```

Produces a single self-contained HTML file (no internet needed) with
side-by-side piano rolls, convergence curves, sparsity stats table,
and evaluation scores if provided.

### `sheet`

```
python -m nmf_cli sheet [OPTIONS]

Options:
  --input  FILE   Input .mid file                                [required]
  --out    FILE   Output MusicXML .xml file  [default: outputs/score.xml]
  --title  TEXT   Score title                [default: NMF Transcription]
  --pdf           Also attempt PDF via MuseScore (must be on PATH)
```

**Example:**
```bash
python -m nmf_cli sheet \
    --input outputs/piano_group_sparse.mid \
    --out   outputs/score.xml \
    --title "My NMF Transcription"
```

Opens in MuseScore (free), Sibelius, Finale, or Flat.io.

---

## Complete Workflow

```bash
# 1. Build harmonic dictionary
python -m nmf_cli build-dict --harmonic --output dict.npy

# 2. Transcribe both methods
python -m nmf_cli transcribe \
    --input piano.wav --method compare \
    --out-dir outputs/

# 3. Evaluate against ground truth (if available)
python -m nmf_cli evaluate \
    --predicted   outputs/piano_group_sparse.mid \
    --compare-std outputs/piano_standard.mid \
    --reference   ground_truth.mid \
    --out-json    outputs/eval.json

# 4. Generate HTML comparison report
python -m nmf_cli report \
    --std-mid outputs/piano_standard.mid \
    --gs-mid  outputs/piano_group_sparse.mid \
    --summary outputs/summary.json \
    --eval-json outputs/eval.json \
    --out     outputs/report.html

# 5. Export sheet music
python -m nmf_cli sheet \
    --input outputs/piano_group_sparse.mid \
    --out   outputs/score.xml
```

---

## Output Artifacts (complete list)

```
outputs/
├── <name>_standard.mid         Standard NMF — playable MIDI
├── <name>_group_sparse.mid     Group-Sparse NMF — playable MIDI
├── <name>_standard_H.npz       Activation matrix H (K × T)
├── <name>_group_sparse_H.npz   Piano roll matrix (n_notes × T)
├── summary.json                Run config, convergence, sparsity stats
├── eval.json                   Evaluation scores (P/R/F, frame + note)
├── report.html                 Self-contained comparison report
└── score.xml                   MusicXML sheet music
```

---

## Troubleshooting

### Dimension mismatch error
```
ERROR: Dimension mismatch: spectrogram V has 1025 frequency bins
       but dictionary W has 512 rows.
```
**Fix:** Rebuild the dictionary with the same `--n-fft` you use for transcription.
Both default to 2048, so this only happens if you manually changed one.

### Missing dictionary sidecar
```
ERROR: Dictionary metadata sidecar missing: dict.json
```
**Fix:** The `.json` file must exist alongside the `.npy`. Rebuild the dictionary
— both files are created together by `build-dict`.

### No notes in MIDI output
If the `.mid` file is empty, lower the `--threshold`:
```bash
python -m nmf_cli transcribe --input audio.wav --threshold 0.05
```

### pretty_midi / librosa not found
```bash
pip install -e ".[dev]"
# or
uv sync
```

### Slow transcription on long files
Group-sparse NMF on a 5-minute file can take 2–5 minutes.
Use `--method standard` for faster results, or trim the audio to a short clip.
