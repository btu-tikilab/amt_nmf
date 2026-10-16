# Example outputs

Produced by the commands in the main README's "Quick demo" with the default
settings (`--sparsity-weight 0.2 --threshold 0.1 --max-iter 1000`). The
dictionary came from the 88 synthetic single notes of
`scripts/make_demo_audio.py`. Regenerate everything with:

```bash
python scripts/make_demo_audio.py demo
python -m nmf_cli build-dict --audio-dir demo/single_notes --pitch-range 21 108 --output dictionaries/piano.npy
python -m nmf_cli transcribe --input demo/piece.wav --dict dictionaries/piano.npy --out-dir examples/piece
python -m nmf_cli transcribe --input demo/varied.wav --dict dictionaries/piano.npy --out-dir examples/varied
python scripts/evaluate_midi.py demo/piece_reference.mid examples/piece/piece_standard.mid examples/piece/piece_group_sparse.mid --json examples/piece/scores.json
python scripts/evaluate_midi.py demo/varied_reference.mid examples/varied/varied_standard.mid examples/varied/varied_group_sparse.mid --json examples/varied/scores.json
```

Each folder holds both MIDI transcriptions, their `_H.npz` activations,
`summary.json`, the reference MIDI and `scores.json`.

## Scores

mir_eval note matching: onset within 50 ms; for "onset + offset" the end must
also be within 50 ms or 20% of the note's length, whichever is larger.

| Input | Method | Notes (ref) | Onset P / R / F1 | Onset + offset P / R / F1 |
|---|---|---|---|---|
| `piece` (4 s, same timbre as dictionary) | Standard | 13 (12) | 0.92 / 1.00 / 0.96 | 0.92 / 1.00 / 0.96 |
| | Group-sparse | 12 (12) | **1.00 / 1.00 / 1.00** | **1.00 / 1.00 / 1.00** |
| `varied` (30 s, different timbre) | Standard | 167 (148) | 0.84 / 0.95 / 0.90 | 0.72 / 0.81 / 0.76 |
| | Group-sparse | 144 (148) | **0.98 / 0.95 / 0.97** | **0.83 / 0.81 / 0.82** |

## Solver details (from summary.json)

| Input | Method | Iterations | Converged | Relative reconstruction error |
|---|---|---|---|---|
| `piece` | Standard | 550 | yes | 0.073 |
| | Group-sparse | 60 | yes | 0.301 (penalised fit); 0.083 (offset re-fit) |
| `varied` | Standard | 520 | yes | 0.087 |
| | Group-sparse | 70 | yes | 0.330 (penalised fit); 0.092 (offset re-fit) |

Standard NMF's only extra note on `piece` is a B2 ghost at 1.39 to 1.51 s,
just before the C3 + G3 bass notes. It is kept on purpose: it is the kind of
error the group-sparsity penalty removes, and the comparison is meant to show
that. These inputs are synthetic, so expect lower scores on real recordings.
