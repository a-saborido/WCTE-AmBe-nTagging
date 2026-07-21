"""Cherenkov event topology observables for WCTE AmBe neutron tagging."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .common import safe_rms, unit_vectors


def phi_rms_around_axis(unit_dirs: np.ndarray, axis: np.ndarray) -> float:
    """
    RMS of azimuthal hit gaps around the mean hit axis, in degrees.

    A capture gamma cascade should look fairly isotropic once viewed from the
    fitted vertex. A through-going or directional topology tends to produce
    uneven azimuthal gaps. The observable is based on gaps rather than raw phi
    values so it remains invariant under a global rotation around the axis.
    """
    n = len(unit_dirs)
    if n < 2 or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0:
        return np.nan

    a = axis / np.linalg.norm(axis)

    # Build a stable orthonormal basis perpendicular to the axis. The fallback
    # avoids a nearly zero cross-product when the axis is close to +x/-x.
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, a)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - np.dot(tmp, a) * a
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a, e1)

    phi = np.arctan2(unit_dirs @ e2, unit_dirs @ e1)
    phi = np.sort((phi + 2.0 * np.pi) % (2.0 * np.pi))

    # Equal gaps are expected for a perfectly uniform ring around the axis.
    gaps = np.diff(np.r_[phi, phi[0] + 2.0 * np.pi])
    target = 2.0 * np.pi / n
    return float(np.degrees(np.sqrt(np.mean((gaps - target) ** 2))))


def calculate_cherenkov_topology_observables(
    hit_pos_cm: np.ndarray,
    pmt_dir: Optional[np.ndarray],
    xfit_cm: np.ndarray,
    latt_cm: float,
    low_weight_cut: float,
    low_theta_deg: float,
    back_theta_deg: float,
) -> Dict[str, float]:
    """
    Calculate topology observables from PMT hit directions relative to xfit.

    Definitions:
    - theta_mean/theta_rms describe the angular spread around the mean hit axis.
    - phi_rms measures azimuthal non-uniformity around that same axis.
    - Nlowtheta counts hits very close to the mean axis; high values can signal
      a directional topology rather than an isotropic capture cascade.
    - Nback counts hits more than back_theta_deg from the mean axis; this is a
      simple handle on backward/side activity.
    - Nlow counts geometrically unlikely hits using a light-yield proxy. It is
      sensitive to the fitted vertex, PMT normals, attenuation length, and the
      WCTE geometry scale, so its threshold is intentionally exposed in the CLI.
    """
    rel = hit_pos_cm - xfit_cm[None, :]
    r = np.linalg.norm(rel, axis=1)
    u = unit_vectors(rel)
    n = len(u)

    if n == 0:
        return {
            "theta_mean": np.nan,
            "theta_rms": np.nan,
            "phi_rms": np.nan,
            "Nlowtheta": np.nan,
            "Nback": np.nan,
            "Nlow": np.nan,
        }

    vmean = np.sum(u, axis=0)
    if np.linalg.norm(vmean) == 0:
        vmean = np.array([np.nan, np.nan, np.nan])
        theta_i = np.full(n, np.nan)
    else:
        vmean = vmean / np.linalg.norm(vmean)
        theta_i = np.degrees(np.arccos(np.clip(u @ vmean, -1.0, 1.0)))

    theta_mean = float(np.nanmean(theta_i))
    theta_rms = safe_rms(theta_i)
    phi_rms = phi_rms_around_axis(u, vmean)
    nlowtheta = int(np.sum(theta_i < low_theta_deg))
    nback = int(np.sum(theta_i > back_theta_deg))

    # PMT normals matter for the geometric light proxy: a PMT facing the fitted
    # vertex should be more likely to collect light than one viewed from behind.
    # If normals are unavailable, use incidence=1 so Nlow remains defined but
    # less physically discriminating.
    if pmt_dir is None or len(pmt_dir) != n:
        incidence = np.ones(n, dtype=float)
    else:
        nd = unit_vectors(pmt_dir)
        incidence = np.abs(np.sum(u * nd, axis=1))
        incidence = np.clip(incidence, 0.0, 1.0)

    # The weight is only a proxy, not a calibrated PE expectation. The 1/r^2
    # term is protected at 1 cm so scan points landing on a PMT do not dominate.
    r_safe = np.maximum(r, 1.0)
    weight = incidence * np.exp(-r_safe / latt_cm) / (r_safe**2)
    nlow = int(np.sum(weight < low_weight_cut))

    return {
        "theta_mean": theta_mean,
        "theta_rms": theta_rms,
        "phi_rms": phi_rms,
        "Nlowtheta": nlowtheta,
        "Nback": nback,
        "Nlow": nlow,
    }

