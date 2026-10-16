"""NMF solvers: standard Euclidean NMF and group-sparse NMF.

Both factorise a magnitude spectrogram V (F x T) as V ~= W @ H with W (F x K)
spectral atoms and H (K x T) activations, all non-negative.

* ``standard_nmf``: Lee & Seung multiplicative updates for the squared
  Euclidean cost 1/2 ||V - WH||_F^2. With a fixed dictionary only H is
  updated; without one, W is learned too.
* ``group_sparse_nmf``: O'Hanlon & Plumbley (ICASSP 2014). The atoms are
  partitioned into groups (one group per pitch) and a mixed l1,2 norm

      Omega(H) = sum_t sum_g ||H[g, t]||_2

  is added to a beta-divergence cost. The penalty is l1 across groups (few
  pitches active per frame) and l2 inside a group (atoms of the same pitch
  activate together). Updates are multiplicative with an unpenalised warm
  start, as in the paper.

Pure numpy; no printing or other presentation code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .errors import DimensionMismatchError, ParameterError

EPS = 1e-12


@dataclass
class NMFResult:
    W: np.ndarray
    H: np.ndarray
    method: str
    n_iter: int
    max_iter: int
    converged: bool
    cost_history: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def final_cost(self) -> float:
        return self.cost_history[-1] if self.cost_history else float("nan")

    def reconstruction_error(self, V: np.ndarray) -> float:
        """Relative Frobenius error ||V - WH|| / ||V||."""
        return float(np.linalg.norm(V - self.W @ self.H) / (np.linalg.norm(V) + EPS))


def _check_inputs(V: np.ndarray, W: np.ndarray | None, max_iter: int, tol: float):
    if V.ndim != 2:
        raise DimensionMismatchError(f"V must be 2-D, got shape {V.shape}.")
    if not np.all(np.isfinite(V)) or np.any(V < 0):
        raise ParameterError("V must be finite and non-negative.")
    if W is not None:
        if W.ndim != 2 or W.shape[0] != V.shape[0]:
            raise DimensionMismatchError(
                f"W has shape {W.shape} but V has {V.shape[0]} rows."
            )
        if not np.all(np.isfinite(W)) or np.any(W < 0):
            raise ParameterError("W must be finite and non-negative.")
    if max_iter < 1:
        raise ParameterError("max_iter must be >= 1.")
    if not tol > 0:
        raise ParameterError("tol must be > 0.")


def _init_H(K: int, T: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.5, 1.5, (K, T)) * max(scale, EPS)


def standard_nmf(
    V: np.ndarray,
    W: np.ndarray | None = None,
    n_components: int | None = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    check_every: int = 10,
    seed: int = 0,
) -> NMFResult:
    """Euclidean NMF via multiplicative updates.

    If ``W`` is given it is held fixed (supervised transcription) and only H
    is estimated. Otherwise ``n_components`` atoms are learned as well.
    Iteration stops when the relative cost decrease over ``check_every``
    iterations drops below ``tol``.
    """
    V = np.asarray(V, dtype=np.float64)
    _check_inputs(V, W, max_iter, tol)
    rng = np.random.default_rng(seed)
    F, T = V.shape
    learn_W = W is None
    if learn_W:
        if not n_components or n_components < 1:
            raise ParameterError("n_components must be >= 1 when W is learned.")
        W = rng.uniform(0.5, 1.5, (F, n_components)) * np.sqrt(V.mean() + EPS)
    else:
        W = np.array(W, dtype=np.float64)
    K = W.shape[1]
    H = _init_H(K, T, np.sqrt(V.mean() + EPS), rng)

    t0 = time.perf_counter()
    costs = [0.5 * float(np.sum((V - W @ H) ** 2))]
    converged = False
    WtV, WtW = W.T @ V, W.T @ W  # constant while W is fixed
    it = 0
    for it in range(1, max_iter + 1):
        H *= WtV / (WtW @ H + EPS)
        if learn_W:
            W *= (V @ H.T) / (W @ (H @ H.T) + EPS)
            norms = np.linalg.norm(W, axis=0) + EPS
            W /= norms
            H *= norms[:, None]
            WtV, WtW = W.T @ V, W.T @ W
        if it % check_every == 0 or it == max_iter:
            costs.append(0.5 * float(np.sum((V - W @ H) ** 2)))
            if (costs[-2] - costs[-1]) <= tol * max(costs[-2], EPS):
                converged = True
                break

    return NMFResult(
        W=W,
        H=H,
        method="standard",
        n_iter=it,
        max_iter=max_iter,
        converged=converged,
        cost_history=costs,
        elapsed_s=time.perf_counter() - t0,
        params={"cost": "euclidean", "tol": tol, "learn_W": learn_W},
    )


def beta_divergence(V: np.ndarray, Vhat: np.ndarray, beta: float) -> float:
    """Sum of the element-wise beta-divergence d_beta(V | Vhat)."""
    Vhat = np.maximum(Vhat, EPS)
    if beta == 2:
        return 0.5 * float(np.sum((V - Vhat) ** 2))
    if beta == 1:
        Vc = np.maximum(V, EPS)
        return float(np.sum(V * np.log(Vc / Vhat) - V + Vhat))
    if beta == 0:
        Vc = np.maximum(V, EPS)
        return float(np.sum(Vc / Vhat - np.log(Vc / Vhat) - 1))
    return float(
        np.sum(V**beta + (beta - 1) * Vhat**beta - beta * V * Vhat ** (beta - 1))
        / (beta * (beta - 1))
    )


def _mm_exponent(beta: float) -> float:
    """Exponent that makes the multiplicative update monotone (Fevotte & Idier
    2011)."""
    if beta < 1:
        return 1.0 / (2.0 - beta)
    if beta > 2:
        return 1.0 / (beta - 1.0)
    return 1.0


def group_matrix(groups: list[list[int]], K: int) -> np.ndarray:
    """0/1 matrix G (n_groups x K) with G[g, k] = 1 if atom k is in group g."""
    G = np.zeros((len(groups), K))
    for g, idx in enumerate(groups):
        G[g, idx] = 1.0
    if np.any(G.sum(axis=0) != 1):
        raise ParameterError("Groups must partition the atoms (each atom once).")
    return G


def group_norms(H: np.ndarray, G: np.ndarray) -> np.ndarray:
    """||H[g, t]||_2 for every group g and frame t, shape (n_groups, T)."""
    return np.sqrt(G @ (H * H))


def group_sparse_nmf(
    V: np.ndarray,
    W: np.ndarray,
    groups: list[list[int]],
    sparsity_weight: float = 0.2,
    beta: float = 2.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    warm_start_iter: int = 10,
    check_every: int = 10,
    seed: int = 0,
) -> NMFResult:
    """Group-sparse NMF with a fixed dictionary (O'Hanlon & Plumbley 2014).

    Minimises  D_beta(V | WH) + sparsity_weight * sum_{g,t} ||H[g, t]||_2
    with the multiplicative update

        H <- H * ( W^T (V * (WH)^(beta-2)) /
                   (W^T (WH)^(beta-1) + lambda * H / ||H[g(k), t]||_2) )^phi

    where g(k) is the group of atom k. The first ``warm_start_iter`` updates
    use lambda = 0 so the penalty acts on a sensible estimate rather than on
    the random initialisation.
    """
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    _check_inputs(V, W, max_iter, tol)
    if not np.isfinite(sparsity_weight) or sparsity_weight < 0:
        raise ParameterError("sparsity_weight must be finite and >= 0.")
    if not np.isfinite(beta) or not 0 <= beta <= 2:
        raise ParameterError("beta must be in [0, 2].")
    if warm_start_iter < 0:
        raise ParameterError("warm_start_iter must be >= 0.")

    rng = np.random.default_rng(seed)
    F, T = V.shape
    K = W.shape[1]
    G = group_matrix(groups, K)
    H = _init_H(K, T, np.sqrt(V.mean() + EPS), rng)
    phi = _mm_exponent(beta)
    WtV, WtW = W.T @ V, W.T @ W

    def cost(H: np.ndarray) -> float:
        return beta_divergence(V, W @ H, beta) + sparsity_weight * float(
            group_norms(H, G).sum()
        )

    t0 = time.perf_counter()
    costs = [cost(H)]
    converged = False
    total = warm_start_iter + max_iter
    it = 0
    for it in range(1, total + 1):
        if beta == 2:
            num, den = WtV, WtW @ H
        else:
            Vhat = np.maximum(W @ H, EPS)
            num = W.T @ (V * Vhat ** (beta - 2))
            den = W.T @ Vhat ** (beta - 1)
        if it > warm_start_iter and sparsity_weight > 0:
            den = den + sparsity_weight * H / (G.T @ group_norms(H, G) + EPS)
        ratio = num / (den + EPS)
        H *= ratio if phi == 1.0 else ratio**phi
        if it > warm_start_iter and (
            (it - warm_start_iter) % check_every == 0 or it == total
        ):
            costs.append(cost(H))
            prev = costs[-2]
            if 0 <= prev - costs[-1] <= tol * max(abs(prev), EPS):
                converged = True
                break
        elif it == warm_start_iter:
            costs.append(cost(H))  # penalised cost after the warm start

    return NMFResult(
        W=W,
        H=H,
        method="group_sparse",
        n_iter=it,
        max_iter=total,
        converged=converged,
        cost_history=costs,
        elapsed_s=time.perf_counter() - t0,
        params={
            "cost": f"beta-divergence (beta={beta}) + l1,2 group penalty",
            "beta": beta,
            "sparsity_weight": sparsity_weight,
            "warm_start_iter": warm_start_iter,
            "tol": tol,
            "n_groups": len(groups),
        },
    )


def hoyer_sparsity(X: np.ndarray) -> float:
    """Hoyer (2004) sparsity of a matrix, averaged over columns, in [0, 1].

    1 means each column has a single non-zero entry, 0 means all entries are
    equal. Columns that are entirely zero are skipped."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return 1.0
    l1 = np.abs(X).sum(axis=0)
    l2 = np.sqrt((X * X).sum(axis=0))
    keep = l2 > EPS
    if not np.any(keep):
        return 1.0
    s = (np.sqrt(n) - l1[keep] / l2[keep]) / (np.sqrt(n) - 1)
    return float(np.mean(s))
