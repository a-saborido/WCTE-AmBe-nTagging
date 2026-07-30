"""Readout-level helper functions used before candidate observables are built."""

from __future__ import annotations

from typing import Dict

import numpy as np


def max_count_in_window(times: np.ndarray, width_ns: float) -> int:
    """
    Maximum number of hits in any half-open [t, t + width_ns) window.

    This is used for the Nmax200 delayed-search cleaning diagnostic. Very large
    values are usually not delayed capture candidates but bursts, pathological
    readout periods, or prompt-like activity leaking into the delayed search.
    """
    t = np.sort(np.asarray(times, dtype=float))
    t = t[np.isfinite(t)]
    n = len(t)
    if n == 0:
        return 0

    j = 0
    best = 0
    for i in range(n):
        while j < n and t[j] < t[i] + width_ns:
            j += 1
        best = max(best, j - i)
    return int(best)


def dense_time_window_keep_mask(times: np.ndarray, width_ns: float, max_hits: int) -> np.ndarray:
    """
    Keep hits outside locally dense time bursts.

    Nmax200-style cleaning should not discard a full WCTE delayed search: the
    search spans hundreds of microseconds, while the pathological activity is
    localized to an O(200 ns) burst. This mask removes hits that belong to any
    half-open [t, t + width_ns) window containing at least max_hits hits. If
    several dense windows overlap, their union is vetoed.
    """
    times = np.asarray(times, dtype=float)
    keep = np.ones(len(times), dtype=bool)
    finite_idx = np.flatnonzero(np.isfinite(times))
    if len(finite_idx) == 0:
        return keep

    order = finite_idx[np.argsort(times[finite_idx])]
    t = times[order]
    bad_sorted = np.zeros(len(order), dtype=bool)

    j = 0
    for i in range(len(t)):
        while j < len(t) and t[j] < t[i] + width_ns:
            j += 1
        if j - i >= max_hits:
            bad_sorted[i:j] = True

    keep[order[bad_sorted]] = False
    return keep


def continuous_noise_keep_mask(
    times: np.ndarray,
    pmt_key: np.ndarray,
    coincidence_ns: float = 6000.0,
) -> np.ndarray:
    """
    Remove same-PMT pairs separated by less than coincidence_ns.

    This readout-cleaning step is applied before candidate finding. When one
    tube fires repeatedly inside the coincidence window, both hits in each close
    pair are discarded. That avoids artificial Nn candidates from afterpulses
    or unstable channels, but the window is kept configurable because real
    photons can occasionally hit the same PMT more than once.
    """
    times = np.asarray(times, dtype=float)
    pmt_key = np.asarray(pmt_key)
    keep = np.ones(len(times), dtype=bool)
    if len(times) < 2:
        return keep

    order = np.lexsort((times, pmt_key))
    key_s = pmt_key[order]
    t_s = times[order]
    bad_sorted = np.zeros(len(times), dtype=bool)

    start = 0
    while start < len(times):
        end = start + 1
        while end < len(times) and key_s[end] == key_s[start]:
            end += 1

        if end - start >= 2:
            dt = np.diff(t_s[start:end])
            bad_local = np.zeros(end - start, dtype=bool)
            pair = np.flatnonzero(dt < coincidence_ns)
            bad_local[pair] = True
            bad_local[pair + 1] = True
            bad_sorted[start:end] = bad_local

        start = end

    keep[order[bad_sorted]] = False
    return keep


def make_pmt_key(w_evt: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Per-hit PMT key used for same-PMT noise removal.

    Slot/position is preferred because it directly matches the WCTE readout
    identity and the geometry lookup. Card/channel is only a fallback for files
    that do not carry slot/position, although extraction now requires those
    branches for geometry-based observables.
    """
    n = len(w_evt["time"])
    if "slot" in w_evt and "pos" in w_evt:
        return w_evt["slot"].astype(np.int64) * 1000 + w_evt["pos"].astype(np.int64)
    if "card" in w_evt and "channel" in w_evt:
        return w_evt["card"].astype(np.int64) * 1000 + w_evt["channel"].astype(np.int64)
    return np.arange(n, dtype=np.int64)
