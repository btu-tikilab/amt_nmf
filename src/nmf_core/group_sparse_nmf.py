"""
src/nmf_core/group_sparse_nmf.py
─────────────────────────────────────────────────────────────────
Group-Sparse beta-NMF — O'Hanlon & Plumbley (2014).

Cost:  C_beta(S|DX) + λ · Σ_l ‖x[l]‖_2^q        (eq. 11)
Update (eq. 12):
    X <- X ⊙ ([Dᵀ(S⊙(DX)^[beta-2])] / ([Dᵀ(DX)^[beta-1]] + λΨ))^φ(beta)

Structural gradient (eq. 13):
    Ψ_{k,n} = q · x_{k,n} / ‖x[l]‖_2^(2-q)

No UI code lives here — pure numpy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# ── helpers ────────────────────────────────────────────────────────────────

def phi(beta: float) -> float:
    """Step-size exponent φ(beta) guaranteeing cost descent."""
    if beta < 1:
        return 1.0 / (2.0 - beta)
    if beta > 2:
        return 1.0 / (beta - 1.0)
    return 1.0


def _spow(A: np.ndarray, p: float, floor: float = 1e-8) -> np.ndarray:
    return np.maximum(A, floor) ** p


def compute_psi(
    X: np.ndarray,
    groups: List[List[int]],
    q: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Structural gradient Ψ — links each atom to its whole group.

    Groups whose norm is large  -> Ψ small  -> atom boosted
    Groups whose norm is small  -> Ψ large  -> atom suppressed
    """
    Psi = np.zeros_like(X)
    exp = 2.0 - q
    for idx in groups:
        sub = X[idx, :]
        gl  = np.sqrt(np.sum(sub ** 2, axis=0)) + eps
        Psi[idx, :] = q * sub / (gl[None, :] ** exp)
    return Psi


def beta_divergence(
    S: np.ndarray,
    Z: np.ndarray,
    beta: float,
    eps: float = 1e-10,
) -> float:
    """beta-divergence C_beta(S|Z). beta=2 -> Euclidean."""
    Z = np.maximum(Z, eps)
    if abs(beta - 1.0) < 1e-8:
        Sc = np.maximum(S, eps)
        return float(np.sum(Sc * np.log(Sc / Z) - Sc + Z))
    if abs(beta) < 1e-8:
        Sc = np.maximum(S, eps)
        return float(np.sum(Sc / Z - np.log(Sc / Z) - 1.0))
    b = beta
    return float(
        np.sum(
            (np.maximum(S, 0) ** b + (b - 1) * Z ** b
             - b * np.maximum(S, 0) * Z ** (b - 1))
            / (b * (b - 1))
        )
    )


# ── result container ───────────────────────────────────────────────────────

@dataclass
class GroupSparseResult:
    X:            np.ndarray
    D:            np.ndarray        # fixed dictionary (unchanged)
    groups:       List[List[int]]
    cost_history: List[float]
    converged:    bool
    n_iter:       int
    elapsed_s:    float
    beta:         float
    lam:          float
    q:            float
    X_snapshots:  dict = field(default_factory=dict)

    def piano_roll(self) -> np.ndarray:
        """Sum atoms within each group -> (n_notes, T)."""
        return np.stack(
            [self.X[idx, :].sum(axis=0) for idx in self.groups], axis=0
        )

    def sparsity_stats(self, thresh_db: float = -30.0) -> dict:
        roll   = self.piano_roll()
        thresh = roll.max() * (10 ** (thresh_db / 20))
        active = roll > thresh
        return {
            "mean_active_per_frame": float(active.sum(axis=0).mean()),
            "total_groups":          len(self.groups),
            "sparsity_pct":          float(100.0 * (1.0 - active.mean())),
        }


# ── main algorithm ─────────────────────────────────────────────────────────

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
    Group-Sparse beta-NMD (O'Hanlon & Plumbley 2014).

    D is FIXED. Only X (activations) is learned.

    Phase 1: n_init_unpenalised standard beta-NMF updates (warm start)
    Phase 2: group-sparse update (eq. 12) until convergence

    Parameters
    ----------
    S       : (F, T) normalised magnitude spectrogram
    D       : (F, K) fixed harmonic dictionary
    groups  : column-index lists, one per note
    beta    : beta-divergence parameter (paper uses 0.5)
    lam     : group sparsity weight λ (paper uses 0.5)
    q       : group norm exponent (paper uses 0.5)
    """
    if snapshot_iters is None:
        snapshot_iters = [1, 5, 10, 25, 50, 100, 200, max_iter]

    phi_val = phi(beta)
    eps     = 1e-10
    X       = np.maximum(D.T @ S, eps)   # init via projection
    F, K    = D.shape

    X_snaps:   dict        = {}
    cost_hist: List[float] = []
    converged  = False
    total_iter = 0
    t0         = time.perf_counter()

    # Phase 1 — unpenalised warm start
    for step in range(n_init_unpenalised):
        DX  = np.maximum(D @ X, eps)
        num = D.T @ (S * _spow(DX, beta - 2))
        den = np.maximum(D.T @ _spow(DX, beta - 1), eps)
        X   = np.maximum(X * np.clip(num / den, 1e-8, 1e8) ** phi_val, eps)
        if step + 1 in snapshot_iters:
            X_snaps[step + 1] = X.copy()
        if progress_callback:
            progress_callback(
                5 + int(20 * (step + 1) / n_init_unpenalised),
                f"Init {step+1}/{n_init_unpenalised}",
            )

    # Phase 2 — group-sparse updates
    for i in range(1, max_iter + 1):
        total_iter = n_init_unpenalised + i

        DX  = np.maximum(D @ X, eps)
        num = D.T @ (S * _spow(DX, beta - 2))
        Psi = compute_psi(X, groups, q, eps)
        den = np.maximum(D.T @ _spow(DX, beta - 1) + lam * Psi, eps)
        X   = np.maximum(X * np.clip(num / den, 1e-8, 1e8) ** phi_val, eps)

        if total_iter in snapshot_iters:
            X_snaps[total_iter] = X.copy()

        if i % profile_every == 0 or i == 1 or i == max_iter:
            DX_c  = np.maximum(D @ X, eps)
            bdiv  = beta_divergence(S, DX_c, beta)
            pen   = sum(
                float(np.sum(np.sqrt(np.sum(X[idx, :] ** 2, axis=0)) ** q))
                for idx in groups
            )
            cost = bdiv + lam * pen
            cost_hist.append(cost)

            if verbose:
                print(f"  iter {i:4d}/{max_iter}  GS-cost={cost:.4e}")
            if progress_callback:
                progress_callback(
                    25 + int(65 * i / max_iter),
                    f"GS-NMF iter {i}/{max_iter}",
                )
            if len(cost_hist) > tol_window:
                old = cost_hist[-tol_window - 1]
                chg = 100.0 * abs(old - cost) / max(abs(old), eps)
                if chg < tol_pct:
                    converged = True
                    if verbose:
                        print(f"  ok Converged at iter {i}")
                    break

    elapsed           = time.perf_counter() - t0
    X_snaps[total_iter] = X.copy()

    return GroupSparseResult(
        X=X, D=D, groups=groups,
        cost_history=cost_hist,
        converged=converged, n_iter=total_iter,
        elapsed_s=elapsed, beta=beta, lam=lam, q=q,
        X_snapshots=X_snaps,
    )
