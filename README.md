# NMF Transcriber

Polyphonic piano transcription with non-negative matrix factorisation (NMF).
It turns a `.wav`/`.flac` recording into MIDI using a pitch dictionary learned
from single-note recordings, and compares two solvers:

* **Standard NMF**: Euclidean multiplicative updates (Lee & Seung).
* **Group-sparse NMF**: the l1,2 group penalty of O'Hanlon & Plumbley (2014),
  which keeps the number of active pitches per frame small and suppresses
  overtone "ghost" notes.

```
src/
  nmf_core/             algorithm package, no printing or UI code
    audio.py            librosa loading + constant-Q spectrogram V
    nmf.py              standard_nmf, group_sparse_nmf
    dictionary.py       build / save / load the dictionary W (+ JSON sidecar)
    midi.py             activations H -> note events -> .mid, .npz dumps
    pipeline.py         transcribe(): validation, both solvers, summary.json
    errors.py           user-facing exception types
  nmf_cli/              argparse command line (python -m nmf_cli)
tests/                  pytest suite (synthetic audio, no dataset needed)
```

## 1. Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # package + pytest + ruff
# runtime only, without the test tools:  pip install -e .
```

Check the install:

```bash
pytest                             # all unit and integration tests
ruff check . && ruff format --check .
```

## 2. The maths

### Spectrogram V
Audio is resampled to 22 050 Hz and converted to a magnitude constant-Q
transform: 288 bins from A0 (27.5 Hz), 36 bins per octave (one bin = 1/3
semitone), hop 512 samples (23.2 ms). On a log-frequency axis every note's
harmonics have the same spacing, so a pitch looks like a shifted copy of its
neighbours. V has shape F x T (bins x frames) and is scaled to max 1.

### Factorisation V ≈ W H
* **W** (F x K) is the dictionary: each column is a spectral template
  (an *atom*). Every pitch owns `atoms_per_pitch` atoms (default 2, e.g. a
  bright attack spectrum and a duller decay spectrum).
* **H** (K x T) holds the activations: how strongly each atom sounds in each
  frame. Summing a pitch's atom rows gives its piano-roll row.

W is fixed during transcription; only H is estimated, so H >= 0 is the
transcription.

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
together); the outer sum is an l1 norm over pitches, which drives whole
pitches to exactly zero, so each frame is explained by few notes. This removes
the typical NMF error of explaining a note's overtones with the notes an
octave or a fifth above. The multiplicative update is

```
H ← H ⊙ [ Wᵀ(V ⊙ Λ^(β−2)) / (Wᵀ Λ^(β−1) + λ H / ‖H[G(k), t]‖₂) ]^φ(β),   Λ = WH
```

with φ(β) = 1/(2−β) for β < 1, 1 for 1 ≤ β ≤ 2. As in the paper, the first
10 iterations run without the penalty (warm start). `--beta 2` (default) uses
the same Euclidean cost as the standard solver, so `--method compare` isolates
the effect of the penalty; `--beta 1` (KL) and `--beta 0` (Itakura-Saito)
are also available. `λ` is `--sparsity-weight`; λ = 0 gives back standard NMF.

### From H to MIDI
1. Sum each pitch's atom rows of H and divide by the global maximum.
2. Every peak at least `--threshold` high, and at least `--threshold` above the
   surrounding valleys, is a **Note-On**. Its time is the half-rise point of
   the peak, so repeated notes of the same pitch re-attack.
3. **Note-Off** when the activation falls below `threshold / 2` (hysteresis)
   or at the next Note-On of that pitch. Notes shorter than `--min-duration`
   (50 ms) are dropped.
4. Frame f becomes time `f × hop_length / sample_rate` seconds; velocity
   follows the peak height.
5. **Group-sparse only:** the penalty also shrinks each note's quiet decaying
   tail to zero, which would end notes too early. So group-sparse decides
   which notes exist and when they start, and each Note-Off is then taken from
   an unpenalised re-fit of H restricted to the pitches it selected (a
   "relaxed" or debiased estimate). `summary.json` reports this re-fit under
   `offset_refit`.

## 3. Usage

### Quick demo (no dataset needed)

`scripts/make_demo_audio.py` synthesizes 88 piano-like single notes with
MAPS-style names and a 4-second polyphonic piece, so the whole pipeline can be
tried right after installing:

```bash
python scripts/make_demo_audio.py demo
python -m nmf_cli build-dict --audio-dir demo/single_notes --pitch-range 21 108 --output dictionaries/piano.npy
python -m nmf_cli transcribe --input demo/piece.wav --dict dictionaries/piano.npy --method compare --out-dir outputs/demo
```

`demo/piece_reference.mid` holds the notes that were actually played, to
compare against `outputs/demo/piece_standard.mid` and
`outputs/demo/piece_group_sparse.mid`.

For real results use real single-note piano recordings, e.g. the `ISOL/NO`
folders of the MAPS piano dataset, which is distributed for research use.

### Build a dictionary

From isolated single-note recordings (supervised, recommended). The pitch is
read from each file name; MAPS names such as
`MAPS_ISOL_NO_F_S0_M60_AkPnBcht.wav` work as-is, as do `note_60.wav`,
`midi60.wav`, `piano_60.wav` and `C4.wav`. Sub-folders are searched, and
several files for the same pitch (e.g. MAPS dynamics F/M/P) are pooled.

```bash
python -m nmf_cli build-dict \
    --audio-dir data/MAPS/ISOL/NO \
    --pitch-range 21 108 \
    --output dictionaries/piano.npy
```

This writes `dictionaries/piano.npy` (W, 288 x 176 for 88 pitches) and
`dictionaries/piano.json`, the sidecar with `sr`, `hop_length`, `fmin`,
`bins_per_octave`, `n_bins`, `bin_resolution_semitones`, and the pitch map
(which W columns belong to which MIDI note).

Without recordings, a synthetic harmonic dictionary can be used (less
accurate):

```bash
python -m nmf_cli build-dict --harmonic --output dictionaries/harmonic.npy
```

Other options: `--atoms-per-pitch` (default 2), `--sr`, `--hop-length`,
`--fmin`, `--bins-per-octave`, `--n-bins`. Run `python -m nmf_cli build-dict -h`.

### Transcribe

```bash
python -m nmf_cli transcribe \
    --input recordings/piece.wav \
    --dict dictionaries/piano.npy \
    --method compare \
    --sparsity-weight 0.2 \
    --threshold 0.1 \
    --out-dir outputs/piece
```

| Flag | Meaning | Default |
|---|---|---|
| `--input` | polyphonic `.wav` / `.flac` (also `.ogg`, `.mp3`, `.aiff`) | required |
| `--dict` | dictionary `.npy` from `build-dict` (sidecar must sit next to it) | required |
| `--method` | `standard`, `group-sparse` or `compare` (both, sequentially) | `compare` |
| `--sparsity-weight` | λ of the l1,2 penalty, ≥ 0 | `0.2` |
| `--threshold` | Note-On threshold relative to the loudest activation, in (0, 1] | `0.1` |
| `--out-dir` | where MIDI, `.npz` and `summary.json` go | `outputs` |
| `--beta` | β-divergence of the group-sparse cost, in [0, 2] | `2` |
| `--max-iter`, `--tol` | iteration cap and convergence tolerance | `500`, `1e-4` |
| `--min-duration` | shortest note kept, seconds | `0.05` |

The spectrogram settings always come from the dictionary sidecar, so the input
is analysed exactly like the dictionary was built. After installation the same
commands are also available as `nmf-transcriber build-dict ...` and
`nmf-transcriber transcribe ...`.

Tuning: raise `--threshold` if there are too many spurious notes, lower it if
quiet notes are missed; raise `--sparsity-weight` (e.g. 0.3 to 0.5) to suppress
more overtone ghosts, lower it if real notes of a chord disappear.

## 4. Output artifacts

For `--input piece.wav --method compare` the output directory contains:

| File | Content |
|---|---|
| `piece_standard.mid` | standard NMF transcription, one piano track |
| `piece_group_sparse.mid` | group-sparse NMF transcription |
| `piece_standard_H.npz`, `piece_group_sparse_H.npz` | `H` (atoms x frames), `pitch_roll` (pitches x frames), `pitches`, `atom_pitch`, `times` (s), `sr`, `hop_length`, `notes` (JSON list) |
| `summary.json` | execution log, below |

`summary.json` records the full run configuration (all flags, spectrogram
settings, dictionary shape and pitch range), input duration, and per method:
`converged`, `iterations`, `max_iterations`, `solve_time_s`, `final_cost`,
`relative_reconstruction_error`, `n_notes`, `pitches_detected`,
`pitch_roll_hoyer_sparsity` and `mean_active_pitches_per_frame`. In compare
mode a `comparison` block puts these side by side and lists pitches found by
only one method. `processing_time_s` is the total wall-clock time.

Load activations in Python:

```python
import numpy as np

z = np.load("outputs/piece/piece_group_sparse_H.npz")
roll, times, pitches = z["pitch_roll"], z["times"], z["pitches"]
```

## 5. Troubleshooting

Errors are printed as one `Error: ...` line and the command exits with
status 1; nothing is written to `--out-dir` for a failed run.

| Message | Cause and fix |
|---|---|
| `Dimension mismatch: the dictionary has N frequency bins ... the input spectrogram has M` | The `.npy` and its `.json` sidecar come from different builds, or the `.npy` was edited. Rebuild with `build-dict` and keep both files together. |
| `Dictionary metadata sidecar not found` | Copy the `.json` file that was created next to the `.npy`, or rebuild. |
| `Dictionary file ... is corrupted` / `sidecar ... is corrupted or incomplete` | The file is truncated or not a NumPy array (e.g. a failed download). Rebuild. |
| `Audio file not found` / `Cannot read audio file` | Check the path. A decoding failure means the file is damaged or in an unsupported codec; convert it to 16-bit WAV. |
| `None of the ... audio files ... is named with a MIDI pitch` | `build-dict` could not find pitches in the file names; rename them like `note_60.wav` or use MAPS names. |
| `--threshold must be in (0, 1]`, `--sparsity-weight must be >= 0` | Hyperparameter out of range. |
| `ModuleNotFoundError: No module named 'nmf_core'` or `'librosa'` | The package is not installed in the active environment: activate the venv and run `pip install -e ".[dev]"`. |
| `sndfile library not found` / `OSError: cannot load library 'libsndfile'` | On Linux install it with `sudo apt-get install libsndfile1`; the pip wheels bundle it on Windows and macOS. |
| `.mp3` cannot be read | MP3 decoding needs ffmpeg on the system; convert to WAV/FLAC instead. |
| Many short wrong notes | Raise `--threshold` or `--sparsity-weight`; make sure the dictionary was built from the same kind of piano. |
| Unexpected error | Re-run with `python -m nmf_cli --debug transcribe ...` to see the traceback. |

## References

* K. O'Hanlon and M. D. Plumbley, "Polyphonic piano transcription using
  non-negative matrix factorisation with group sparsity", ICASSP 2014.
* P. Smaragdis and J. C. Brown, "Non-negative matrix factorization for
  polyphonic music transcription", WASPAA 2003.
* D. D. Lee and H. S. Seung, "Algorithms for non-negative matrix
  factorization", NIPS 2001.
* C. Févotte and J. Idier, "Algorithms for nonnegative matrix factorization
  with the β-divergence", Neural Computation 2011.
