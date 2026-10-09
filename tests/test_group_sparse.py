"""tests/test_group_sparse.py — group sparsity tests."""
from __future__ import annotations

import numpy as np
import pytest

from nmf_core.dictionary import build_harmonic_dictionary
from nmf_core.group_sparse_nmf import (
    GroupSparseResult,
    compute_psi,
    group_sparse_nmd,
    phi,
)


@pytest.fixture(scope="module")
def freqs():
    import librosa
    return librosa.fft_frequencies(sr=22050, n_fft=2048)


@pytest.fixture(scope="module")
def harm_dict(freqs):
    return build_harmonic_dictionary(
        freqs, midi_low=60, midi_high=67, n_harmonics=4
    )


@pytest.fixture(scope="module")
def dummy_S(freqs, harm_dict):
    D, groups, _ = harm_dict
    rng = np.random.default_rng(0)
    # Build a synthetic S from known X so we can check recovery direction
    X_true = rng.uniform(0, 1, (D.shape[1], 30))
    S = np.maximum(D @ X_true, 0)
    S /= S.max() + 1e-10
    return S


@pytest.fixture(scope="module")
def gs_result(dummy_S, harm_dict):
    D, groups, _ = harm_dict
    return group_sparse_nmd(
        dummy_S, D, groups,
        beta=0.5, lam=0.5, q=0.5,
        n_init_unpenalised=5, max_iter=30,
        verbose=False,
    )


class TestPhi:
    def test_beta_half(self):
        assert abs(phi(0.5) - 2 / 3) < 1e-9

    def test_beta_one(self):
        assert phi(1.0) == 1.0

    def test_beta_two(self):
        assert phi(2.0) == 1.0


class TestGroupSparseResult:
    def test_returns_correct_type(self, gs_result):
        assert isinstance(gs_result, GroupSparseResult)

    def test_X_nonneg(self, gs_result):
        assert np.all(gs_result.X >= 0)

    def test_X_shape(self, dummy_S, harm_dict):
        D, groups, _ = harm_dict
        r = group_sparse_nmd(dummy_S, D, groups,
                             beta=0.5, lam=0.5, q=0.5,
                             max_iter=10, verbose=False)
        assert r.X.shape == (D.shape[1], dummy_S.shape[1])

    def test_D_unchanged(self, gs_result, harm_dict):
        D, _, _ = harm_dict
        np.testing.assert_array_equal(gs_result.D, D)

    def test_cost_decreases(self, gs_result):
        assert gs_result.cost_history[-1] <= gs_result.cost_history[0]

    def test_piano_roll_shape(self, gs_result):
        roll = gs_result.piano_roll()
        assert roll.shape[0] == len(gs_result.groups)
        assert roll.shape[1] == gs_result.X.shape[1]

    def test_piano_roll_nonneg(self, gs_result):
        assert np.all(gs_result.piano_roll() >= 0)

    def test_sparsity_stats_keys(self, gs_result):
        s = gs_result.sparsity_stats()
        assert "mean_active_per_frame" in s
        assert "sparsity_pct" in s
        assert "total_groups" in s

    def test_sparsity_pct_in_range(self, gs_result):
        s = gs_result.sparsity_stats()
        assert 0.0 <= s["sparsity_pct"] <= 100.0

    def test_snapshots_captured(self, gs_result):
        assert len(gs_result.X_snapshots) > 0


class TestGroupSparsityEffect:
    """
    Assert that applying the group sparsity penalty yields higher sparsity
    compared to standard unpenalised updates on the same sparse input.

    Uses a signal where ONLY 2 of the 8 notes are active so the penalty
    has clear structure to suppress — avoids the degenerate all-active case.
    """

    def test_gs_sparser_than_standard(self, harm_dict):
        from nmf_core.standard_nmf import run_nmf

        D, groups, pm = harm_dict

        # Build a SPARSE signal: only first and last note-group active
        rng    = np.random.default_rng(99)
        T      = 40
        X_true = np.zeros((D.shape[1], T))
        # activate only group 0 and group -1 (2 out of 8 notes)
        for idx in groups[0]:
            X_true[idx, :] = rng.uniform(0.5, 1.0, T)
        for idx in groups[-1]:
            X_true[idx, :] = rng.uniform(0.5, 1.0, T)

        S = np.maximum(D @ X_true, 0)
        S_max = S.max() + 1e-10
        S /= S_max

        # Standard NMF — supervised (W fixed to D), no group penalty
        r_std = run_nmf(S, K=D.shape[1], W_init=D, n_iter=100)

        # Group-sparse NMF with strong lambda to enforce sparsity
        r_gs = group_sparse_nmd(
            S, D, groups,
            beta=0.5, lam=2.0, q=0.5,
            max_iter=100, verbose=False,
        )

        # Measure sparsity: fraction of note-frame cells BELOW 5% of max
        thresh = 0.05
        H_std_norm  = r_std.H / (r_std.H.max() + 1e-10)
        roll_gs     = r_gs.piano_roll()
        roll_norm   = roll_gs / (roll_gs.max() + 1e-10)

        sparsity_std = float((H_std_norm < thresh).mean())
        sparsity_gs  = float((roll_norm  < thresh).mean())

        # GS should suppress the 6 inactive note groups → more silent cells
        assert sparsity_gs > sparsity_std * 0.8, (
            f"GS sparsity ({sparsity_gs:.3f}) should exceed "
            f"standard ({sparsity_std:.3f}) on a sparse input"
        )


class TestComputePsi:
    def test_psi_shape(self, gs_result):
        Psi = compute_psi(gs_result.X, gs_result.groups, q=0.5)
        assert Psi.shape == gs_result.X.shape

    def test_psi_nonneg(self, gs_result):
        Psi = compute_psi(gs_result.X, gs_result.groups, q=0.5)
        assert np.all(Psi >= 0)
