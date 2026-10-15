"""
src/nmf_core/group_sparse_nmf.py

Group-Sparse beta-NMF — O'Hanlon & Plumbley (2014).

Cost:
    C_beta(S | DX) + lam * sum_l ||x[l]||_2^q

D is fixed; only X is learned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


def phi(beta: float) -> float:
    """Step-size exponent for beta-divergence NMF."""
    if beta < 1:
        return 1.0 / (2.0 - beta)
    if beta > 2:
        return 1.0 / (beta - 1.0)
    return 1.0


def _spow(A: np.ndarray, p: float, floor: float = 1e-8) -> np.ndarray:
    """Safe power for nonnegative arrays."""
    return np.maximum(A, floor) ** p


def compute_psi(
    X: np.ndarray,
    groups: List[List[int]],
    q: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute the group-sparsity penalty gradient."""
    Psi = np.zeros_like(X)

    for idx in groups:
        sub = X[idx, :]
        group_norm = np.sqrt(np.sum(sub**2, axis=0))
        group_norm = np.maximum(group_norm, eps)

        Psi[idx, :] = q * sub / (group_norm[None, :] ** (2.0 - q))

    return Psi


def beta_divergence(
    S: np.ndarray,
    Z: np.ndarray,
    beta: float,
    eps: float = 1e-10,
) -> float:
    """Compute beta-divergence C_beta(S | Z)."""
    S = np.maximum(S, 0.0)
    Z = np.maximum(Z, eps)

    if abs(beta - 1.0) < 1e-8:
        Sc = np.maximum(S, eps)
        return float(np.sum(Sc * np.log(Sc / Z) - Sc + Z))

    if abs(beta) < 1e-8:
        Sc = np.maximum(S, eps)
        return float(np.sum(Sc / Z - np.log(Sc / Z) - 1.0))

    b = beta
    value = (S**b + (b - 1.0) * Z**b - b * S * Z ** (b - 1.0)) / (b * (b - 1.0))

    return float(np.sum(value))


@dataclass
class GroupSparseResult:
    X: np.ndarray
    D: np.ndarray
    groups: List[List[int]]
    cost_history: List[float]
    converged: bool
    n_iter: int
    elapsed_s: float
    beta: float
    lam: float
    q: float
    X_snapshots: dict = field(default_factory=dict)

    def piano_roll(self) -> np.ndarray:
        """Sum activations within each group."""
        return np.stack(
            [self.X[idx, :].sum(axis=0) for idx in self.groups],
            axis=0,
        )

    def sparsity_stats(self, thresh_db: float = -30.0) -> dict:
        """Return activation sparsity statistics."""
        roll = self.piano_roll()

        if roll.size == 0 or roll.max() <= 0:
            return {
                "mean_active_per_frame": 0.0,
                "total_groups": len(self.groups),
                "sparsity_pct": 100.0,
            }

        threshold = roll.max() * (10 ** (thresh_db / 20.0))
        active = roll > threshold

        return {
            "mean_active_per_frame": float(active.sum(axis=0).mean()),
            "total_groups": len(self.groups),
            "sparsity_pct": float(100.0 * (1.0 - active.mean())),
        }


def group_sparse_nmd(
    S: np.ndarray,
    D: np.ndarray,
    groups: List[List[int]],
    beta: float = 0.5,
    lam: float = 0.5,
    q: float = 0.5,
    n_init_unpenalised: int = 10,
    max_iter: int = 500,
    tol_pct: float = 0.5,
    tol_window: int = 5,
    profile_every: int = 5,
    snapshot_iters: Optional[List[int]] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> GroupSparseResult:
    """
    Group-sparse beta-NMF with fixed dictionary D.

    Phase 1: unpenalised beta-NMF warm start.
    Phase 2: group-sparse multiplicative updates.

    Diagnostic output reports the objective components and activation
    scale so that increasing cost can be detected.
    """
    S = np.asarray(S, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    if S.ndim != 2 or D.ndim != 2:
        raise ValueError("S and D must both be 2D arrays.")

    if D.shape[0] != S.shape[0]:
        raise ValueError(
            f"Frequency dimensions differ: D has {D.shape[0]}, S has {S.shape[0]}."
        )

    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(D)):
        raise ValueError("S and D must contain only finite values.")

    if np.any(S < 0) or np.any(D < 0):
        raise ValueError("S and D must be nonnegative.")

    if not 0 < q <= 1:
        raise ValueError("q must be in the interval (0, 1].")

    if lam < 0:
        raise ValueError("lam must be nonnegative.")

    if beta < 0:
        raise ValueError("beta must be nonnegative.")

    if n_init_unpenalised < 0 or max_iter < 1:
        raise ValueError("Invalid iteration count.")

    if profile_every < 1 or tol_window < 1:
        raise ValueError("profile_every and tol_window must be positive.")

    if snapshot_iters is None:
        snapshot_iters = [1, 5, 10, 25, 50, 100, 200, max_iter]

    eps = 1e-10
    phi_val = phi(beta)
    _, K = D.shape

    for idx in groups:
        if not idx:
            raise ValueError("Groups cannot be empty.")
        if min(idx) < 0 or max(idx) >= K:
            raise ValueError("Group contains an invalid dictionary index.")

    # Initial projection.
    X = np.maximum(D.T @ S, eps)

    if verbose:
        print(
            f"Initial scales: S_max={S.max():.6g}, "
            f"D_max={D.max():.6g}, "
            f"X_max={X.max():.6g}"
        )

    X_snaps = {}
    cost_hist = []
    converged = False
    total_iter = 0
    t0 = time.perf_counter()

    # Phase 1: unpenalised warm start.
    for step in range(n_init_unpenalised):
        DX = np.maximum(D @ X, eps)

        num = D.T @ (S * _spow(DX, beta - 2.0))
        den = np.maximum(D.T @ _spow(DX, beta - 1.0), eps)

        ratio = np.clip(num / den, 1e-8, 1e8)
        X = np.maximum(X * ratio**phi_val, eps)

        if verbose:
            print(f"Phase 1 step {step + 1}: max_activation={X.max():.6g}")

        if step + 1 in snapshot_iters:
            X_snaps[step + 1] = X.copy()

        if progress_callback:
            progress_callback(
                5 + int(20 * (step + 1) / max(n_init_unpenalised, 1)),
                f"Init {step + 1}/{n_init_unpenalised}",
            )

    # Phase 2: group-sparse updates.
    for i in range(1, max_iter + 1):
        total_iter = n_init_unpenalised + i

        DX = np.maximum(D @ X, eps)
        num = D.T @ (S * _spow(DX, beta - 2.0))

        Psi = compute_psi(X, groups, q, eps)
        den = np.maximum(
            D.T @ _spow(DX, beta - 1.0) + lam * Psi,
            eps,
        )

        ratio = np.clip(num / den, 1e-8, 1e8)
        X = np.maximum(X * ratio**phi_val, eps)

        if total_iter in snapshot_iters:
            X_snaps[total_iter] = X.copy()

        if i % profile_every == 0 or i == 1 or i == max_iter:
            DX_current = np.maximum(D @ X, eps)

            bdiv = beta_divergence(S, DX_current, beta)

            penalty = sum(
                float(np.sum(np.sqrt(np.sum(X[idx, :] ** 2, axis=0)) ** q))
                for idx in groups
            )

            cost = bdiv + lam * penalty
            cost_hist.append(cost)

            if verbose:
                print(
                    f"iter={i}, "
                    f"beta_divergence={bdiv:.6g}, "
                    f"penalty={penalty:.6g}, "
                    f"cost={cost:.6g}, "
                    f"max_activation={X.max():.6g}"
                )

            if verbose:
                print(f"  iter {i:4d}/{max_iter} GS-cost={cost:.4e}")

            if progress_callback:
                progress_callback(
                    25 + int(65 * i / max_iter),
                    f"GS-NMF iter {i}/{max_iter}",
                )

            if len(cost_hist) > tol_window:
                old = cost_hist[-tol_window - 1]
                change_pct = 100.0 * abs(old - cost) / max(abs(old), eps)

                # Require both a small change and no increase in cost.
                if cost <= old and change_pct < tol_pct:
                    converged = True
                    if verbose:
                        print(f"  ok Converged at iter {i}")
                    break

    elapsed = time.perf_counter() - t0
    X_snaps[total_iter] = X.copy()

    return GroupSparseResult(
        X=X,
        D=D,
        groups=groups,
        cost_history=cost_hist,
        converged=converged,
        n_iter=total_iter,
        elapsed_s=elapsed,
        beta=beta,
        lam=lam,
        q=q,
        X_snapshots=X_snaps,
    )
