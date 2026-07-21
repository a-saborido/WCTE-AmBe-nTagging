"""Shared numeric helpers for WCTE AmBe neutron-tagging observables."""

from __future__ import annotations

import numpy as np


def safe_rms(values: np.ndarray) -> float:
    """Return the RMS around the mean, ignoring non-finite values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    return float(np.sqrt(np.mean((v - np.mean(v)) ** 2)))


def unit_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize row vectors while keeping invalid or zero rows finite-safe."""
    v = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(v, axis=1)
    out = np.full_like(v, np.nan, dtype=float)
    good = np.isfinite(norm) & (norm > 0)
    out[good] = v[good] / norm[good, None]
    return out


def angular_distance_deg(unit_dirs: np.ndarray) -> np.ndarray:
    """Pairwise angular distances between already-normalized directions."""
    u = np.asarray(unit_dirs, dtype=float)
    dot = np.clip(u @ u.T, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

