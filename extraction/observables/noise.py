"""Noise characterization observables for WCTE AmBe neutron tagging."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .common import angular_distance_deg, safe_rms, unit_vectors


def count_clustered_hits(unit_dirs: np.ndarray, angle_deg: float, min_size: int = 3) -> int:
    """
    Count hits in local angular connected components.

    Nclus is placed in the noise category because tight same-module or
    neighboring-module clusters are a useful proxy for local detector effects
    and PMT-related bursts. It is still geometry-derived: the threshold should
    track the mPMT angular span for the chosen detector and source location.
    """
    n = len(unit_dirs)
    if n < min_size:
        return 0

    adj = angular_distance_deg(unit_dirs) <= angle_deg
    seen = np.zeros(n, dtype=bool)
    nclus_hits = 0

    for i in range(n):
        if seen[i]:
            continue

        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            a = stack.pop()
            comp.append(a)
            neigh = np.flatnonzero(adj[a] & (~seen))
            seen[neigh] = True
            stack.extend(neigh.tolist())

        if len(comp) >= min_size:
            nclus_hits += len(comp)

    return int(nclus_hits)


def calculate_noise_observables(
    hit_pos_cm: np.ndarray,
    xfit_cm: np.ndarray,
    charges: np.ndarray,
    n300: int,
    cluster_angle_deg: float,
    high_charge_pe: float,
) -> Dict[str, float]:
    """
    Calculate charge and local-clustering observables.

    Definitions:
    - Qmean/Qrms summarize the charge distribution in the final candidate hits.
      Capture gamma hits should be low-PE dominated; large tails can indicate
      non-capture activity or merged hits.
    - NhighQ counts hits above high_charge_pe and is sensitive to the charge
      calibration and same-PMT hit merging.
    - Nclus counts hits in compact angular clusters around the fitted vertex.
      This helps flag localized detector noise; its threshold is WCTE-specific.
    - N300 counts hits in a wider 300 ns corrected-time window around the
      original candidate peak. It measures local time activity surrounding the
      candidate, so large values can indicate accidental pileup.
    """
    q = np.asarray(charges, dtype=float)
    qmean = float(np.nanmean(q)) if len(q) else np.nan
    qrms = safe_rms(q)
    nhighq = int(np.sum(q >= high_charge_pe))

    rel = hit_pos_cm - xfit_cm[None, :]
    unit_dirs = unit_vectors(rel)
    nclus = count_clustered_hits(unit_dirs, angle_deg=cluster_angle_deg, min_size=3)

    return {
        "Qmean": qmean,
        "Qrms": qrms,
        "NhighQ": nhighq,
        "Nclus": nclus,
        "N300": int(n300),
    }
