"""
src/nmf_core/standard_nmf.py
─────────────────────────────────────────────────────────────────
Standard Euclidean NMF — Smaragdis & Brown (2003).

V  ≈  W  @  H
(F×T)  (F×K)  (K×T)

Update rules (Lee & Seung 2001, Euclidean variant):
    H  <-  H  ⊙  (WᵀV)       /  (WᵀWH + ε)
    W  <-  W  ⊙  (VHᵀ)       /  (WHHᵀ + ε)

No UI code lives here — pure numpy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class NMFResult:
    """All outputs from one NMF run."""

    W: np.ndarray                # (F, K) spectral basis templates
    H: np.ndarray                # (K, T) temporal activations
    V_approx: np.ndarray         # (F, T) reconstruction W @ H
    K: int
    n_iter: int
    converged: bool
    final_error: float
    error_reduction_pct: float
    elapsed_s: float
    errors: List[float]          # one per checkpoint
    error_iters: List[int]
    W_snapshots: dict = field(default_factory=dict)
    H_snapshots: dict = field(default_factory=dict)


def run_nmf(
    V: np.ndarray,
    K: int,
    W_init: Optional[np.ndarray] = None,
    n_iter: int = 200,
    tol: float = 1e-6,
    random_state: int = 42,
    verbose: bool = False,
    profile_every: int = 10,
    snapshot_iters: Optional[List[int]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> NMFResult:
    """
    Euclidean NMF via multiplicative updates.

    Parameters
    ----------
    V           : (F, T) non-negative magnitude spectrogram
    K           : number of components
    W_init      : optional pre-built dictionary (F, K) — if supplied,
                  W is held FIXED and only H is updated (supervised mode)
    n_iter      : maximum iterations
    tol         : relative convergence tolerance on Frobenius error
    random_state: seed for reproducible random init
    profile_every: record error every N iterations
    snapshot_iters: iterations to save W/H copies for profiling

    Returns
    -------
    NMFResult dataclass
    """
    if snapshot_iters is None:
        snapshot_iters = [1, 5, 10, 25, 50, 100, 200, n_iter]

    V = np.asarray(V, dtype=np.float64)

    if V.ndim != 2:
        raise ValueError("V must be a 2-D non-negative matrix.")

    if not np.all(np.isfinite(V)):
        raise ValueError("V must contain only finite values (no NaN or infinity).")

    if np.any(V < 0):
        raise ValueError("V must be a 2-D non-negative matrix.")

    F, T = V.shape
    eps  = np.finfo(np.float64).eps
    rng  = np.random.default_rng(random_state)
    supervised = W_init is not None

    if supervised:
        W = np.asarray(W_init, dtype=np.float64)
        if W.shape != (F, K):
            raise ValueError(
                f"W_init shape {W.shape} does not match (F={F}, K={K})."
            )
    else:
        W = rng.uniform(0.0, 1.0, (F, K))

    H = rng.uniform(0.0, 1.0, (K, T))

    W_snaps: dict = {}
    H_snaps: dict = {}
    errors: List[float] = []
    e_iters: List[int]  = []
    converged = False
    t0 = time.perf_counter()

    init_err = float(np.linalg.norm(V - W @ H, "fro"))
    errors.append(init_err)
    e_iters.append(0)

    last_i = 0
    for i in range(1, n_iter + 1):
        last_i = i

        # H update (always)
        WtV  = W.T @ V
        WtWH = W.T @ (W @ H)
        H   *= WtV / (WtWH + eps)
        H    = np.maximum(H, eps)

        if not supervised:
            # W update (only in unsupervised mode)
            VHt  = V @ H.T
            WHHt = (W @ H) @ H.T
            W   *= VHt / (WHHt + eps)
            W    = np.maximum(W, eps)
            # Column-normalise to prevent scale drift
            norms = np.linalg.norm(W, axis=0) + eps
            W    /= norms
            H    *= norms[:, np.newaxis]

        if i in snapshot_iters:
            W_snaps[i] = W.copy()
            H_snaps[i] = H.copy()

        if i % profile_every == 0 or i == n_iter:
            err = float(np.linalg.norm(V - W @ H, "fro"))
            errors.append(err)
            e_iters.append(i)
            if verbose and i % 50 == 0:
                print(f"  Iter {i:4d}/{n_iter}  Error: {err:.4e}")
            if progress_callback:
                progress_callback(
                    25 + int(65 * i / n_iter),
                    f"NMF iter {i}/{n_iter}",
                )
            if len(errors) >= 2:
                rel = abs(errors[-2] - err) / (errors[-2] + eps)
                if rel < tol:
                    converged = True
                    if verbose:
                        print(f"  ok Converged at iter {i}")
                    break

    elapsed   = time.perf_counter() - t0
    W_snaps[last_i] = W.copy()
    H_snaps[last_i] = H.copy()
    final_err = float(np.linalg.norm(V - W @ H, "fro"))
    err_pct   = float((init_err - final_err) / (init_err + eps) * 100)

    return NMFResult(
        W=W, H=H, V_approx=W @ H,
        K=K, n_iter=last_i, converged=converged,
        final_error=final_err, error_reduction_pct=err_pct,
        elapsed_s=elapsed, errors=errors, error_iters=e_iters,
        W_snapshots=W_snaps, H_snapshots=H_snaps,
    )
