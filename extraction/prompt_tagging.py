"""Prompt tagging for WCTE AmBe delayed neutron-capture searches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .observables.common import safe_rms


@dataclass(frozen=True)
class PromptCandidate:
    """Prompt scintillation-light candidate found inside one readout window."""

    prompt_id: int
    time_ns: float
    window_start_ns: float
    window_end_ns: float
    nhits: int
    trms_ns: float
    tmean_ns: float
    hit_indices: np.ndarray


def find_prompt_candidates(
    times_ns: np.ndarray,
    corrected_times_ns: np.ndarray,
    window_ns: float,
    min_hits: int,
    max_hits: int,
    min_trms_ns: float,
    max_trms_ns: float,
    min_tmean_ns: float,
    max_tmean_ns: float,
    isolation_ns: float,
) -> List[PromptCandidate]:
    """
    Find prompt scintillation candidates in one DAQ/readout window.

    The prompt definition is intentionally independent of truth information:
    a candidate is a high-multiplicity, broad scintillation peak. Raw hit time
    defines the sliding window and the prompt timestamp used by the delayed
    search. The tRMS and tmean shape cuts use times corrected for photon travel
    from the fixed source position. Here tmean is the mean corrected time
    relative to the earliest corrected hit in the prompt window.

    Sliding windows around the same physical peak produce many equivalent
    windows, so after applying the hit-count, tRMS, and tmean cuts we keep only
    the best local representative within the isolation scale.
    """
    t = np.asarray(times_ns, dtype=float)
    tcorr = np.asarray(corrected_times_ns, dtype=float)
    if t.shape != tcorr.shape:
        raise ValueError("Raw and TOF-corrected prompt-time arrays must have matching shapes")

    finite_idx = np.flatnonzero(np.isfinite(t) & np.isfinite(tcorr))
    if len(finite_idx) == 0:
        return []

    order = finite_idx[np.argsort(t[finite_idx])]
    ts = t[order]
    candidates = []
    j = 0

    for i in range(len(ts)):
        while j < len(ts) and ts[j] < ts[i] + window_ns:
            j += 1

        count = j - i
        if count < min_hits or count > max_hits:
            continue

        raw_hit_times = ts[i:j]
        corrected_hit_times = tcorr[order[i:j]]
        trms = safe_rms(corrected_hit_times)
        if not (min_trms_ns <= trms <= max_trms_ns):
            continue

        tmean = float(np.mean(corrected_hit_times - np.min(corrected_hit_times)))
        if not (min_tmean_ns <= tmean <= max_tmean_ns):
            continue

        prompt_time = float(np.mean(raw_hit_times))
        candidates.append(
            {
                "time_ns": prompt_time,
                "window_start_ns": float(ts[i]),
                "window_end_ns": float(ts[i] + window_ns),
                "nhits": int(count),
                "trms_ns": float(trms),
                "tmean_ns": tmean,
                "hit_indices": order[i:j].astype(int),
            }
        )

    if not candidates:
        return []

    target_trms = 0.5 * (min_trms_ns + max_trms_ns)
    target_tmean = 0.5 * (min_tmean_ns + max_tmean_ns)
    candidates.sort(
        key=lambda c: (
            -c["nhits"],
            abs(c["trms_ns"] - target_trms),
            abs(c["tmean_ns"] - target_tmean),
            c["time_ns"],
        )
    )

    selected = []
    for cand in candidates:
        # Sliding 1.5 us prompt windows around one broad scintillation pulse can
        # produce several mean times separated by a few hundred ns. We therefore
        # suppress candidates using the prompt-window overlap, not only the mean
        # time, so one physical prompt peak keeps only one representative.
        if any(
            cand["window_start_ns"] <= kept["window_end_ns"] + isolation_ns
            and kept["window_start_ns"] <= cand["window_end_ns"] + isolation_ns
            for kept in selected
        ):
            continue
        selected.append(cand)

    selected.sort(key=lambda c: c["time_ns"])
    return [
        PromptCandidate(
            prompt_id=i,
            time_ns=c["time_ns"],
            window_start_ns=c["window_start_ns"],
            window_end_ns=c["window_end_ns"],
            nhits=c["nhits"],
            trms_ns=c["trms_ns"],
            tmean_ns=c["tmean_ns"],
            hit_indices=c["hit_indices"],
        )
        for i, c in enumerate(selected)
    ]
