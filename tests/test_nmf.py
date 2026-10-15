"""Core NMF math: shapes, non-negativity, convergence, input checks."""

import numpy as np
import pytest

from nmf_core.errors import DimensionMismatchError, ParameterError
from nmf_core.nmf import (
    group_norms,
    group_sparse_nmf,
    hoyer_sparsity,
    standard_nmf,
)

rng = np.random.default_rng(0)
F, K, T = 40, 6, 30
W_TRUE = rng.uniform(0, 1, (F, K))
H_TRUE = rng.uniform(0, 1, (K, T))
V = W_TRUE @ H_TRUE
GROUPS = [[0, 1], [2, 3], [4, 5]]


@pytest.mark.parametrize(
    "result",
    [
        standard_nmf(V, W_TRUE),
        standard_nmf(V, n_components=K),
        group_sparse_nmf(V, W_TRUE, GROUPS, sparsity_weight=0.1),
    ],
    ids=["standard-fixed-W", "standard-learned-W", "group-sparse"],
)
def test_factor_shapes_and_nonnegativity(result):
    assert result.W.shape == (F, K)
    assert result.H.shape == (K, T)
    assert (result.W @ result.H).shape == V.shape
    assert np.all(result.W >= 0)
    assert np.all(result.H >= 0)
    assert np.all(np.isfinite(result.H))


def test_fixed_dictionary_is_not_modified():
    W = W_TRUE.copy()
    res = standard_nmf(V, W)
    np.testing.assert_array_equal(res.W, W_TRUE)
    np.testing.assert_array_equal(W, W_TRUE)


def test_standard_nmf_recovers_exact_factorisation():
    res = standard_nmf(V, W_TRUE, max_iter=2000, tol=1e-9)
    assert res.reconstruction_error(V) < 0.02


def test_costs_never_increase():
    for res in (
        standard_nmf(V, W_TRUE),
        group_sparse_nmf(V, W_TRUE, GROUPS, sparsity_weight=0.05, beta=2),
        group_sparse_nmf(V, W_TRUE, GROUPS, sparsity_weight=0.0, beta=1),
    ):
        c = np.array(res.cost_history[1:])
        assert np.all(np.diff(c) <= 1e-9 * c[:-1]), res.method


def test_converges_and_reports_iterations():
    noisy = V + 0.05 * rng.random(V.shape)  # no exact solution: cost plateaus
    res = standard_nmf(noisy, W_TRUE, max_iter=5000, tol=1e-3)
    assert res.converged and res.n_iter < 5000
    res = standard_nmf(V, W_TRUE, max_iter=3, tol=1e-12)
    assert not res.converged and res.n_iter == 3


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0])
def test_group_sparse_supports_beta_range(beta):
    res = group_sparse_nmf(V, W_TRUE, GROUPS, beta=beta, max_iter=50)
    assert np.all(np.isfinite(res.H)) and np.all(res.H >= 0)


def test_input_validation():
    with pytest.raises(ParameterError):
        standard_nmf(-V, W_TRUE)
    with pytest.raises(ParameterError):
        standard_nmf(np.full_like(V, np.nan), W_TRUE)
    with pytest.raises(DimensionMismatchError):
        standard_nmf(V, W_TRUE[:-1])
    with pytest.raises(ParameterError):
        group_sparse_nmf(V, W_TRUE, GROUPS, sparsity_weight=-1)
    with pytest.raises(ParameterError):
        group_sparse_nmf(V, W_TRUE, [[0, 1], [2, 3]])  # atoms 4, 5 ungrouped


def test_group_norms_and_hoyer():
    H = np.array([[3.0, 0.0], [4.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(
        group_norms(H, np.array([[1, 1, 0], [0, 0, 1]])), [[5.0, 0.0], [0.0, 1.0]]
    )
    assert hoyer_sparsity(np.array([[1.0], [0.0], [0.0]])) == pytest.approx(1.0)
    assert hoyer_sparsity(np.ones((4, 3))) == pytest.approx(0.0)
