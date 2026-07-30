"""Vertex determination observables for WCTE AmBe neutron tagging.

These observables combine a time-of-flight scan around the prompt vertex with
an independent BONSAI low-energy vertex fit.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import safe_rms


def calculate_bonsai_vertex_observables(
    xbonsai_cm: np.ndarray,
    xfit_cm: np.ndarray,
    wall: object,
) -> Dict[str, float]:
    """
    Calculate the BONSAI-related observables.

    Bpdist is the Euclidean separation between the independent BONSAI vertex
    and the local Nn-scan vertex. Bwall is the shortest distance from the
    BONSAI vertex to the current detector-wall model.
    """
    xbonsai = np.asarray(xbonsai_cm, dtype=float)
    xfit = np.asarray(xfit_cm, dtype=float)
    if xbonsai.shape != (3,) or xfit.shape != (3,):
        raise ValueError("BONSAI and local-fit vertices must each contain x, y, z")
    if not np.all(np.isfinite(xbonsai)) or not np.all(np.isfinite(xfit)):
        return {"Bpdist": np.nan, "Bwall": np.nan}

    return {
        "Bpdist": float(np.linalg.norm(xbonsai - xfit)),
        "Bwall": float(wall.absolute_distance_to_wall(xbonsai)),
    }


def best_window_indices(times: np.ndarray, width_ns: float) -> Tuple[int, np.ndarray, float, float, float]:
    """
    Find the densest half-open [t0, t0 + width_ns) timing window.

    This is the core primitive behind Nn. The half-open convention avoids
    double-counting hits exactly on a boundary when adjacent trial windows have
    nearly identical start times. Ties are resolved elsewhere with tRMS.
    """
    t = np.asarray(times, dtype=float)
    finite = np.isfinite(t)
    valid_idx = np.flatnonzero(finite)
    if len(valid_idx) == 0:
        return 0, np.array([], dtype=int), np.nan, np.nan, np.nan

    order = valid_idx[np.argsort(t[valid_idx])]
    ts = t[order]
    n = len(ts)
    j = 0
    best_count = 0
    best_i = 0
    best_j = 0
    best_rms = np.inf

    for i in range(n):
        while j < n and ts[j] < ts[i] + width_ns:
            j += 1
        count = j - i
        if count <= 0:
            continue
        rms = safe_rms(ts[i:j])
        # Prefer a higher multiplicity, then the tighter time cluster.
        if (count > best_count) or (count == best_count and rms < best_rms):
            best_count = count
            best_i = i
            best_j = j
            best_rms = rms

    loc = order[best_i:best_j]
    t0 = float(ts[best_i])
    t1 = float(t0 + width_ns)
    return int(best_count), loc.astype(int), t0, t1, float(best_rms)


def greedy_nn_candidates(
    t_corr: np.ndarray,
    width_ns: float = 10.0,
    nn_cut: int = 5,
    max_candidates: int = 50,
) -> List[Dict]:
    """
    Greedily find non-overlapping local Nn maxima in corrected time.

    Nn is the number of hits in the best configured-width window after
    correcting hit times for photon travel from a trial vertex. A capture
    candidate is kept only when Nn is greater than the configured cut, so the
    default cut of 5 means at least 6 hits. After a candidate is found, its hits
    are removed so the same peak is not returned repeatedly with slightly
    shifted windows.
    """
    remaining = np.flatnonzero(np.isfinite(t_corr)).astype(int)
    out: List[Dict] = []

    for _ in range(max_candidates):
        if len(remaining) <= nn_cut:
            break

        sub_t = t_corr[remaining]
        count, sub_loc, t0, t1, rms = best_window_indices(sub_t, width_ns)
        if count <= nn_cut:
            break

        loc = remaining[sub_loc]
        out.append(
            {
                "Nn": int(count),
                "idx_local": loc.astype(int),
                "t0_corr": float(t0),
                "t1_corr": float(t1),
                "tcorr_center": float(0.5 * (t0 + t1)),
                "trms": float(rms),
            }
        )

        chosen = set(loc.tolist())
        keep = np.array([idx not in chosen for idx in remaining], dtype=bool)
        remaining = remaining[keep]

    return out


def min_subset_rms(times: np.ndarray, k: int) -> float:
    """
    Minimum RMS over any subset of k corrected hit times.

    For one-dimensional times, the subset with the smallest RMS is contiguous
    after sorting, so this exact scan avoids a combinatorial search. trms3 and
    trms6 are sensitive to very compact subclusters, even when the full Nn
    group contains extra accidental hits.
    """
    t = np.sort(np.asarray(times, dtype=float))
    t = t[np.isfinite(t)]
    if len(t) < k:
        return np.nan

    best = np.inf
    for i in range(0, len(t) - k + 1):
        best = min(best, safe_rms(t[i : i + k]))
    return float(best)


def _grid_axis_values(center_cm: float, halfwidth_cm: float, step_cm: float) -> np.ndarray:
    if halfwidth_cm < 0.0:
        raise ValueError("Grid half-width must be non-negative")
    if step_cm <= 0.0:
        raise ValueError("Grid step must be positive")
    return np.arange(
        center_cm - halfwidth_cm,
        center_cm + halfwidth_cm + 0.5 * step_cm,
        step_cm,
    )


def double_scan_grid_counts(
    xyz_bounds_cm: float,
    coarse_step_cm: float,
    refine_halfwidth_cm: float,
    fine_step_cm: float,
) -> Dict[str, int]:
    """Return nominal coarse/fine point counts for the double-grid scan."""
    coarse_n = len(_grid_axis_values(0.0, xyz_bounds_cm, coarse_step_cm)) ** 3
    fine_n = len(_grid_axis_values(0.0, refine_halfwidth_cm, fine_step_cm)) ** 3
    return {
        "coarse": int(coarse_n),
        "fine": int(fine_n),
        "total": int(coarse_n + fine_n),
    }


def _keep_earliest_per_channel(
    times_ns: np.ndarray,
    pos_cm: np.ndarray,
    channel_keys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror the multilateration helper's earliest hit per PMT/channel rule."""
    order = np.argsort(times_ns)
    times_s = times_ns[order]
    pos_s = pos_cm[order]
    keys_s = np.asarray(channel_keys)[order]

    seen = set()
    keep = []
    for i, key_values in enumerate(keys_s):
        key_arr = np.asarray(key_values)
        key = tuple(key_arr.tolist()) if key_arr.ndim else (key_arr.item(),)
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)

    keep = np.asarray(keep, dtype=int)
    return times_s[keep], pos_s[keep]


def _best_window_at_vertex(
    times_ns: np.ndarray,
    pos_cm: np.ndarray,
    vertex_cm: np.ndarray,
    c_water_cm_per_ns: float,
    width_ns: float,
) -> Tuple[int, np.ndarray, float]:
    tc = times_ns - np.linalg.norm(pos_cm - vertex_cm[None, :], axis=1) / c_water_cm_per_ns
    count, loc, _, _, window_rms = best_window_indices(tc, width_ns)
    return int(count), loc.astype(int), float(window_rms)


def _points_inside_wall(points_cm: np.ndarray, wall: object, margin_cm: float) -> np.ndarray:
    """Vectorized containment for WallEstimator, with a generic object fallback."""
    axis = getattr(wall, "axis", None)
    if axis in {"x", "y", "z"} and all(
        hasattr(wall, attr) for attr in ("radius_cm", "axis_min_cm", "axis_max_cm")
    ):
        ax = {"x": 0, "y": 1, "z": 2}[axis]
        other = [i for i in range(3) if i != ax]
        rho = np.sqrt(points_cm[:, other[0]] ** 2 + points_cm[:, other[1]] ** 2)
        d_side = float(wall.radius_cm) - rho
        d_low = points_cm[:, ax] - float(wall.axis_min_cm)
        d_high = float(wall.axis_max_cm) - points_cm[:, ax]
        return np.minimum(np.minimum(d_side, d_low), d_high) >= margin_cm

    return np.array(
        [wall.contains(point, margin_cm=margin_cm) for point in points_cm],
        dtype=bool,
    )


def refit_vertex_by_multilateration_grid(
    times_ns: np.ndarray,
    pos_cm: np.ndarray,
    prompt_vertex_cm: np.ndarray,
    c_water_cm_per_ns: float,
    width_ns: float,
    fit_hit_indices: Optional[np.ndarray] = None,
    fit_channel_keys: Optional[np.ndarray] = None,
    xyz_bounds_cm: float = 120.0,
    coarse_step_cm: float = 10.0,
    fine_step_cm: float = 1.0,
    refine_halfwidth_cm: float = 20.0,
    dt_cut_ns: Optional[float] = 10.0,
    grid_chunk_size: int = 4096,
    min_fit_hits: int = 6,
    earliest_per_channel: bool = True,
    wall: Optional[object] = None,
    wall_margin_cm: float = 0.0,
) -> Tuple[np.ndarray, int, np.ndarray, float]:
    """
    Fit xfit with the coarse+fine grid, then recompute Nn.

    The grid objective is : 
    use the original Nn burst as timing constraints, subtract TOF for every trial
    vertex, center the corrected times with their median, and minimize the RMS
    of those median residuals. After xfit is chosen, the best Nn window is
    recomputed from the full local context at that fitted vertex.
    """
    times_ns = np.asarray(times_ns, dtype=float)
    pos_cm = np.asarray(pos_cm, dtype=float)
    prompt_vertex = np.asarray(prompt_vertex_cm, dtype=float)
    if prompt_vertex.shape != (3,):
        raise ValueError("prompt_vertex_cm must be a 3-element (x, y, z) sequence")
    if pos_cm.ndim != 2 or pos_cm.shape[1] != 3:
        raise ValueError("pos_cm must be an array with shape (n_hits, 3)")
    if len(times_ns) != len(pos_cm):
        raise ValueError("times_ns and pos_cm must have the same length")
    if c_water_cm_per_ns <= 0.0:
        raise ValueError("c_water_cm_per_ns must be positive")

    if fit_hit_indices is None:
        fit_idx = np.arange(len(times_ns), dtype=int)
    else:
        fit_idx = np.asarray(fit_hit_indices, dtype=int)
    if len(fit_idx) == 0:
        raise ValueError("Multilateration grid fit needs at least one seed hit")
    if np.any(fit_idx < 0) or np.any(fit_idx >= len(times_ns)):
        raise IndexError("Multilateration fit-hit index is outside the context hit array")

    fit_times = times_ns[fit_idx]
    fit_pos = pos_cm[fit_idx]
    finite_fit = np.isfinite(fit_times) & np.all(np.isfinite(fit_pos), axis=1)
    fit_times = fit_times[finite_fit]
    fit_pos = fit_pos[finite_fit]

    fit_keys = None
    if fit_channel_keys is not None:
        fit_channel_keys = np.asarray(fit_channel_keys)
        if len(fit_channel_keys) != len(times_ns):
            raise ValueError("fit_channel_keys must match the context hit array length")
        fit_keys = fit_channel_keys[fit_idx][finite_fit]

    if earliest_per_channel and fit_keys is not None and len(fit_times):
        fit_times, fit_pos = _keep_earliest_per_channel(fit_times, fit_pos, fit_keys)

    def fallback_to_prompt() -> Tuple[np.ndarray, int, np.ndarray, float]:
        count, loc, window_rms = _best_window_at_vertex(
            times_ns,
            pos_cm,
            prompt_vertex,
            c_water_cm_per_ns,
            width_ns,
        )
        return prompt_vertex.copy(), count, loc, window_rms

    min_fit_hits = max(1, int(min_fit_hits))
    if len(fit_times) < min_fit_hits:
        return fallback_to_prompt()

    tmin = float(np.min(fit_times))
    fit_times0 = fit_times - tmin
    chunk_size = max(1, int(grid_chunk_size))

    def score_vertex(vertex_cm: np.ndarray, use_cut: bool) -> Tuple[float, float, float, int, int]:
        dists = np.linalg.norm(fit_pos - vertex_cm[None, :], axis=1)
        t_corr = fit_times0 - dists / c_water_cm_per_ns
        t0 = float(np.median(t_corr))
        dt = t_corr - t0

        if use_cut and dt_cut_ns is not None:
            mask = np.abs(dt) < float(dt_cut_ns)
            n_used = int(np.count_nonzero(mask))
            if n_used < min_fit_hits:
                return np.inf, t0, np.inf, -1, n_used
            dt = dt[mask]
        else:
            n_used = int(len(dt))

        trms = float(np.sqrt(np.mean(dt * dt)))
        chi2 = float(np.sum(dt * dt))
        ndof = int(n_used - 4)
        return trms, t0, chi2, ndof, n_used

    def grid_points(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        points = np.column_stack(
            [gx.ravel(order="C"), gy.ravel(order="C"), gz.ravel(order="C")]
        )
        if wall is None:
            return points
        keep = _points_inside_wall(points, wall, wall_margin_cm)
        return points[keep]

    def best_on_grid(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        use_cut: bool,
    ) -> Optional[Dict[str, object]]:
        points = grid_points(xs, ys, zs)
        if len(points) == 0:
            return None

        best_metric = np.inf
        best_index = None
        for start in range(0, len(points), chunk_size):
            chunk = points[start : start + chunk_size]
            dists = np.linalg.norm(fit_pos[None, :, :] - chunk[:, None, :], axis=2)
            t_corr = fit_times0[None, :] - dists / c_water_cm_per_ns
            t0s = np.median(t_corr, axis=1)
            dt = t_corr - t0s[:, None]

            if use_cut and dt_cut_ns is not None:
                mask = np.abs(dt) < float(dt_cut_ns)
                counts = np.count_nonzero(mask, axis=1)
                sumsq = np.sum(np.where(mask, dt * dt, 0.0), axis=1)
                metrics = np.full(len(chunk), np.inf, dtype=float)
                valid = counts >= min_fit_hits
                metrics[valid] = np.sqrt(sumsq[valid] / counts[valid])
            else:
                metrics = np.sqrt(np.mean(dt * dt, axis=1))

            local = int(np.argmin(metrics))
            metric = float(metrics[local])
            if metric < best_metric:
                best_metric = metric
                best_index = start + local

        if best_index is None or not np.isfinite(best_metric):
            return None

        vertex = points[best_index].astype(float)
        trms, t0, chi2, ndof, n_used = score_vertex(vertex, use_cut=use_cut)
        return {
            "vertex": vertex,
            "trms": float(trms),
            "t0": float(t0 + tmin),
            "chi2": float(chi2),
            "ndof": int(ndof),
            "n_hits_used": int(n_used),
        }

    cx, cy, cz = prompt_vertex
    coarse = best_on_grid(
        _grid_axis_values(cx, xyz_bounds_cm, coarse_step_cm),
        _grid_axis_values(cy, xyz_bounds_cm, coarse_step_cm),
        _grid_axis_values(cz, xyz_bounds_cm, coarse_step_cm),
        use_cut=False,
    )
    if coarse is None:
        return fallback_to_prompt()

    bx, by, bz = np.asarray(coarse["vertex"], dtype=float)
    fine = best_on_grid(
        _grid_axis_values(bx, refine_halfwidth_cm, fine_step_cm),
        _grid_axis_values(by, refine_halfwidth_cm, fine_step_cm),
        _grid_axis_values(bz, refine_halfwidth_cm, fine_step_cm),
        use_cut=True,
    )
    if fine is None:
        fine = best_on_grid(
            _grid_axis_values(bx, refine_halfwidth_cm, fine_step_cm),
            _grid_axis_values(by, refine_halfwidth_cm, fine_step_cm),
            _grid_axis_values(bz, refine_halfwidth_cm, fine_step_cm),
            use_cut=False,
        )

    best_vertex = np.asarray((fine or coarse)["vertex"], dtype=float)
    count, loc, window_rms = _best_window_at_vertex(
        times_ns,
        pos_cm,
        best_vertex,
        c_water_cm_per_ns,
        width_ns,
    )
    return best_vertex, count, loc, window_rms
