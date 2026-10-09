"""tests/test_core_nmf.py — core NMF math tests."""
from __future__ import annotations

import numpy as np
import pytest

from nmf_core.standard_nmf import run_nmf


@pytest.fixture
def V_small():
    return np.abs(np.random.default_rng(0).normal(size=(50, 40)))


@pytest.fixture
def nmf_result(V_small):
    return run_nmf(V_small, K=4, n_iter=50, verbose=False)


class TestNMFDimensions:
    def test_W_shape(self, V_small, nmf_result):
        assert nmf_result.W.shape == (V_small.shape[0], 4)

    def test_H_shape(self, V_small, nmf_result):
        assert nmf_result.H.shape == (4, V_small.shape[1])

    def test_V_approx_shape(self, V_small, nmf_result):
        assert nmf_result.V_approx.shape == V_small.shape

    def test_WH_equals_V_approx(self, nmf_result):
        np.testing.assert_array_almost_equal(
            nmf_result.V_approx, nmf_result.W @ nmf_result.H
        )


class TestNMFNonNegativity:
    def test_W_nonneg(self, nmf_result):
        assert np.all(nmf_result.W >= 0), "W must be non-negative"

    def test_H_nonneg(self, nmf_result):
        assert np.all(nmf_result.H >= 0), "H must be non-negative"

    def test_V_approx_nonneg(self, nmf_result):
        assert np.all(nmf_result.V_approx >= 0)


class TestNMFConvergence:
    def test_error_decreases(self, nmf_result):
        assert nmf_result.errors[-1] < nmf_result.errors[0]

    def test_error_reduction_positive(self, nmf_result):
        assert nmf_result.error_reduction_pct > 0

    def test_reproducible(self, V_small):
        r1 = run_nmf(V_small, K=3, n_iter=20, random_state=0)
        r2 = run_nmf(V_small, K=3, n_iter=20, random_state=0)
        np.testing.assert_array_equal(r1.W, r2.W)
        np.testing.assert_array_equal(r1.H, r2.H)

    def test_different_seeds_differ(self, V_small):
        r1 = run_nmf(V_small, K=3, n_iter=5, random_state=0)
        r2 = run_nmf(V_small, K=3, n_iter=5, random_state=99)
        assert not np.allclose(r1.W, r2.W)


class TestNMFSupervised:
    def test_W_fixed_when_W_init_given(self, V_small):
        """When W_init is supplied, W must stay exactly equal to W_init."""
        rng    = np.random.default_rng(7)
        W_init = rng.uniform(0, 1, (V_small.shape[0], 5))
        result = run_nmf(V_small, K=5, W_init=W_init, n_iter=30)
        np.testing.assert_array_equal(result.W, W_init)

    def test_H_still_learned(self, V_small):
        rng    = np.random.default_rng(7)
        W_init = rng.uniform(0, 1, (V_small.shape[0], 5))
        result = run_nmf(V_small, K=5, W_init=W_init, n_iter=30)
        assert result.H.shape == (5, V_small.shape[1])
        assert np.all(result.H >= 0)


class TestNMFValidation:
    def test_negative_V_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            run_nmf(np.full((10, 10), -1.0), K=2)

    def test_wrong_W_init_shape_raises(self, V_small):
        bad_W = np.ones((10, 3))   # wrong F dimension
        with pytest.raises(ValueError, match="shape"):
            run_nmf(V_small, K=3, W_init=bad_W)

    def test_rank1_recovery(self):
        rng = np.random.default_rng(42)
        w   = rng.uniform(0.5, 1.0, (30, 1))
        h   = rng.uniform(0.5, 1.0, (1, 20))
        V   = w @ h
        r   = run_nmf(V, K=1, n_iter=1000, tol=1e-10)
        rel = np.linalg.norm(V - r.V_approx, "fro") / np.linalg.norm(V, "fro")
        assert rel < 0.01
