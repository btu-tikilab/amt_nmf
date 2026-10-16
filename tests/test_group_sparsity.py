"""Group sparsity must give sparser activations and suppress overtone ghosts.

Dummy setup: pitch A has partials at bins 10, 20, 30, 40; pitch B (an octave
up) has partials at 20 and 40, i.e. a subset of A's overtones; pitch C is
unrelated. Only pitch A plays. Standard NMF can explain part of A's even
partials with B (an overtone ghost); the l1,2 group penalty should push B
towards zero.
"""

import numpy as np
import pytest

from nmf_core.nmf import group_sparse_nmf, hoyer_sparsity, standard_nmf


def _atom(bins, weights, F=50):
    x = np.arange(F)
    a = sum(w * np.exp(-0.5 * (x - b) ** 2) for b, w in zip(bins, weights, strict=True))
    return a / np.linalg.norm(a)


@pytest.fixture(scope="module")
def problem():
    A = [
        _atom([10, 20, 30, 40], [1, 0.8, 0.6, 0.4]),
        _atom([10, 20, 30, 40], [1, 0.4, 0.2, 0.1]),
    ]
    B = [_atom([20, 40], [1, 0.5]), _atom([20, 40], [1, 0.2])]
    C = [_atom([13, 26, 39], [1, 0.5, 0.3]), _atom([13, 26, 39], [1, 0.2, 0.1])]
    W = np.stack(A + B + C, axis=1)
    groups = [[0, 1], [2, 3], [4, 5]]
    t = np.arange(40)
    # Pitch A plays with a timbre between its two atoms; a little noise.
    played = _atom([10, 20, 30, 40], [1, 0.9, 0.5, 0.35])
    rng = np.random.default_rng(1)
    V = np.outer(played, np.exp(-t / 15)) + 0.01 * rng.random((50, 40))
    return V / V.max(), W, groups


def _pitch_roll(H, groups):
    return np.stack([H[g].sum(axis=0) for g in groups])


def test_penalty_increases_sparsity_and_suppresses_overtone_ghost(problem):
    V, W, groups = problem
    std = _pitch_roll(standard_nmf(V, W, max_iter=1000).H, groups)
    gs = _pitch_roll(
        group_sparse_nmf(V, W, groups, sparsity_weight=0.2, beta=2, max_iter=1000).H,
        groups,
    )
    assert hoyer_sparsity(gs) > hoyer_sparsity(std)
    # Energy wrongly given to the octave/unrelated pitches, relative to A.
    ghost_std = std[1:].sum() / std[0].sum()
    ghost_gs = gs[1:].sum() / gs[0].sum()
    assert ghost_gs < 0.8 * ghost_std
    # The unrelated pitch C, which only fits noise, is switched off entirely.
    assert gs[2].sum() < 1e-3 * std[2].sum()
    assert gs[0].sum() > 0.5 * std[0].sum()  # the true pitch survives


def test_sparsity_grows_with_weight(problem):
    V, W, groups = problem
    s = [
        hoyer_sparsity(
            _pitch_roll(
                group_sparse_nmf(V, W, groups, sparsity_weight=lam, beta=2).H, groups
            )
        )
        for lam in (0.0, 0.05, 0.2)
    ]
    assert s[0] < s[1] < s[2]


def test_zero_weight_matches_standard_euclidean(problem):
    V, W, groups = problem
    gs = group_sparse_nmf(
        V, W, groups, sparsity_weight=0.0, beta=2, max_iter=3000, tol=1e-10
    )
    std = standard_nmf(V, W, max_iter=3000, tol=1e-10)
    assert gs.reconstruction_error(V) == pytest.approx(
        std.reconstruction_error(V), abs=1e-3
    )


def test_penalty_is_grouped_per_pitch():
    """l1,2 shrinks an atom by its pitch group's norm, not by its own size.

    Atoms 1 and 2 play at the same small level, but atom 1 shares a group
    with a loud atom and atom 2 is alone in its group. A per-pitch group
    penalty keeps atom 1 and suppresses atom 2; a plain (ungrouped) l1
    penalty would shrink both equally.
    """
    W = np.eye(8)  # orthogonal atoms, so the penalty alone decides
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    h = np.array([1.0, 0.1, 0.1, 0, 0, 0, 0, 0])
    V = np.outer(W @ h, np.ones(5))
    H = group_sparse_nmf(V, W, groups, sparsity_weight=0.08, beta=2, tol=1e-8).H
    weak_in_loud_group, weak_alone = H[1].mean(), H[2].mean()
    assert weak_in_loud_group > 0.08
    assert weak_alone < 0.5 * weak_in_loud_group
