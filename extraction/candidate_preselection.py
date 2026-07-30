"""Prompt tagging, delayed-candidate preselection, and observable extraction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    C_LIGHT_CM_PER_NS,
    DEFAULT_ATTENUATION_LENGTH_CM,
    DEFAULT_BACK_THETA_DEG,
    DEFAULT_BONSAI_DIR,
    DEFAULT_BONSAI_GEOMETRY_ROOT,
    DEFAULT_BONSAI_PARAM,
    DEFAULT_BONSAI_WINDOW_NS,
    DEFAULT_CAPTURE_SEARCH_WINDOW_NS,
    DEFAULT_CAPTURE_START_AFTER_PROMPT_NS,
    DEFAULT_CLUSTER_ANGLE_DEG,
    DEFAULT_CONTINUOUS_NOISE_NS,
    DEFAULT_FIT_CONTEXT_NS,
    DEFAULT_FIT_WALL_MARGIN_CM,
    DEFAULT_GEOMETRY_FILE,
    DEFAULT_GEOMETRY_SCALE_TO_CM,
    DEFAULT_HIGH_CHARGE_PE,
    DEFAULT_LOW_THETA_DEG,
    DEFAULT_LOW_WEIGHT_CUT,
    DEFAULT_MAX_CANDIDATES_PER_PROMPT,
    DEFAULT_MIN_CAPTURE_FRACTION,
    DEFAULT_MIN_CAPTURE_HITS,
    DEFAULT_MIN_CAPTURE_HITS_ABSOLUTE,
    DEFAULT_MULTILATERATION_COARSE_STEP_CM,
    DEFAULT_MULTILATERATION_DT_CUT_NS,
    DEFAULT_MULTILATERATION_EARLIEST_PER_CHANNEL,
    DEFAULT_MULTILATERATION_FINE_STEP_CM,
    DEFAULT_MULTILATERATION_GRID_CHUNK_SIZE,
    DEFAULT_MULTILATERATION_MIN_HITS,
    DEFAULT_MULTILATERATION_REFINE_HALFWIDTH_CM,
    DEFAULT_MULTILATERATION_XYZ_BOUNDS_CM,
    DEFAULT_NN_CUT,
    DEFAULT_NN_WINDOW_NS,
    DEFAULT_N300_WINDOW_NS,
    DEFAULT_NMAX200_CUT,
    DEFAULT_NMAX200_WINDOW_NS,
    DEFAULT_OBSERVABLE_PLOT_BINS,
    DEFAULT_PMT_POSITION_BASE,
    DEFAULT_PROMPT_ISOLATION_NS,
    DEFAULT_PROMPT_MAX_HITS,
    DEFAULT_PROMPT_MAX_TMEAN_NS,
    DEFAULT_PROMPT_MAX_TRMS_NS,
    DEFAULT_PROMPT_MIN_HITS,
    DEFAULT_PROMPT_MIN_TMEAN_NS,
    DEFAULT_PROMPT_MIN_TRMS_NS,
    DEFAULT_PROMPT_VERTEX_CM,
    DEFAULT_PROMPT_WINDOW_NS,
    DEFAULT_WALL_AXIS,
    DEFAULT_WATER_REFRACTIVE_INDEX,
    DEFAULT_WINDOW_DURATION_NS,
    DEFAULT_WINDOW_GAP_NS,
    DEFAULT_WCSIM_DIR,
    OBSERVABLE_COLUMNS,
    OBSERVABLE_GROUPS,
)
from .geometry import PmtGeometry, WallEstimator
from .bonsai import BonsaiVertexFitter

from .root_io import as_np, build_window_payload, combine_window_payloads, following_window_count

from .prompt_tagging import find_prompt_candidates

from .preselection_utils import (
    continuous_noise_keep_mask,
    dense_time_window_keep_mask,
    make_pmt_key,
    max_count_in_window,
)

from .observables.common import safe_rms
from .observables.noise import calculate_noise_observables
from .observables.topology import calculate_cherenkov_topology_observables
from .observables.vertex import (
    calculate_bonsai_vertex_observables,
    double_scan_grid_counts,
    greedy_nn_candidates,
    min_subset_rms,
    refit_vertex_by_multilateration_grid,
)

from .plotting import plot_observable_pdfs


def candidate_hitset_key(w_evt: Dict[str, np.ndarray], global_hit_indices: np.ndarray) -> Tuple:
    """
    Build a stable identity for a candidate hit set.

    The same physical delayed cluster can be rediscovered from different prompt
    candidates or from neighboring anchor windows. Array indices inside the
    combined search are not stable across those views, so the preferred key is
    the original hit identity: source ROOT entry plus hit index inside that
    entry. If older payloads do not provide this bookkeeping, fall back to the
    local combined-array indices, which still de-duplicates within one anchor
    search.
    """
    source_entry = w_evt.get("source_entry")
    source_hit_index = w_evt.get("source_hit_index")
    if source_entry is not None and source_hit_index is not None:
        entries = np.asarray(source_entry)[global_hit_indices]
        hit_indices = np.asarray(source_hit_index)[global_hit_indices]
        return tuple(sorted((int(entry), int(hit_i)) for entry, hit_i in zip(entries, hit_indices)))
    return tuple(sorted(int(hit_i) for hit_i in global_hit_indices))


def write_candidate_hit_info_root(root_rows: List[Dict], out_path: Path) -> Path:
    """
    Write final candidate hit information to a ROOT file.

    Each tree entry is one final de-duplicated candidate. Hit-level branches are
    jagged arrays with one value per selected hit. Scalar branches are candidate
    metadata. ``corrected_hitt`` stores hit-time minus TOF to xfit, centered by
    ``vertex_t`` so the absolute corrected time is recoverable as
    ``corrected_hitt + vertex_t``. The ``THits`` layout keeps the hit-level
    information useful for training point-cloud methods.
    """
    try:
        import awkward as ak
        import uproot
    except ImportError as exc:
        raise SystemExit("Missing ROOT output dependency. Install with: pip install uproot awkward") from exc

    jagged_float_keys = [
        "hitx",
        "hity",
        "hitz",
        "hitDirx",
        "hitDiry",
        "hitDirz",
        "hitc",
        "hitt",
        "corrected_hitt",
    ]
    scalar_float_keys = [
        "vertex_x",
        "vertex_y",
        "vertex_z",
        "vertex_t",
        "fpdist",
        "prompt_time_ns",
        "candidate_tcorr_center_ns",
        "dominant_capture_fraction",
        "capture_fraction",
    ]
    scalar_int_keys = [
        "label",
        "n_hits",
        "event_number",
        "candidate_id",
        "prompt_id",
        "n_capture_hits",
        "n_background_hits",
    ]

    arrays = {
        key: ak.Array([np.asarray(row[key], dtype=np.float32) for row in root_rows])
        for key in jagged_float_keys
    }
    for key in scalar_float_keys:
        arrays[key] = np.asarray([row[key] for row in root_rows], dtype=np.float32)
    for key in scalar_int_keys:
        arrays[key] = np.asarray([row[key] for row in root_rows], dtype=np.int32)

    with uproot.recreate(out_path) as fout:
        fout["THits"] = arrays
    return out_path


def count_truth_matched_prompts(
    prompts,
    prompt_search_idx: np.ndarray,
    prompt_flag: np.ndarray,
) -> Tuple[int, int]:
    """
    Count selected prompt candidates with and without truth-tagged prompt hits.

    PromptCandidate.hit_indices index the readout-only array passed to the
    prompt tagger. Map them back to the combined hit array before consulting
    hit_from_prompt. This function is diagnostic only and runs after prompt
    selection has finished.
    """
    matched = 0
    for prompt in prompts:
        global_hit_indices = prompt_search_idx[np.asarray(prompt.hit_indices, dtype=int)]
        matched += int(np.any(prompt_flag[global_hit_indices]))
    return matched, len(prompts) - matched


def event_to_candidates(
    event_number: int,
    w_evt: Dict[str, np.ndarray],
    t_evt: Dict[str, np.ndarray],
    prompt_vertex_cm: np.ndarray,
    wall: WallEstimator,
    bonsai_fitter: BonsaiVertexFitter,
    args: argparse.Namespace,
    seen_candidate_hitsets: Optional[set] = None,
    seen_raw_candidate_hitsets: Optional[set] = None,
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    Extract BDT candidate rows from one anchor readout window.

    In this analysis, a ROOT "event" is one 270 us DAQ/readout window with
    overlaid signal and background. It is not a physics event in the usual
    trigger sense. We first tag prompt-like scintillation peaks using readout
    hits in the anchor window only. For each prompt, delayed candidates are then
    searched in the next configured capture window, which may include hits from
    following ROOT entries shifted onto a continuous time axis.

    Truth arrays in t_evt are used only after all observables have been computed,
    to assign the training label and store diagnostics.
    """
    c_water = C_LIGHT_CM_PER_NS / args.water_refractive_index
    time = w_evt["time"].astype(float)
    q = w_evt["charge"].astype(float)
    pos = w_evt["pmt_pos_cm"].astype(float)
    pmt_dir = w_evt.get("pmt_dir")

    n = len(time)
    counters = {
        "windows_total": 1,
        "windows_with_prompt_candidates": 0,
        "windows_failed_length_match": 0,
        "windows_failed_truth_length_match": 0,
        "windows_missing_hit_from_capture_branch": 0,
        "windows_missing_hit_from_prompt_branch": 0,
        "windows_with_missing_geometry_hits": 0,
        "hits_missing_geometry": 0,
        "prompt_candidates_total": 0,
        "prompt_candidates_truth_checked": 0,
        "prompt_candidates_truth_matched": 0,
        "prompt_candidates_truth_unmatched": 0,
        "delayed_searches_with_nmax200_veto": 0,
        "delayed_searches_emptied_by_nmax200_veto": 0,
        "hits_removed_by_nmax200_veto": 0,
        "prompt_candidates_with_capture_candidates": 0,
        "windows_with_capture_candidates": 0,
        "windows_with_signal_capture_candidate": 0,
        "capture_candidates_total": 0,
        "raw_capture_candidates_skipped_by_hitset": 0,
        "capture_candidates_deduplicated_by_hitset": 0,
        "capture_candidates_signal": 0,
        "capture_candidates_background": 0,
        "bonsai_fits_attempted": 0,
        "bonsai_fits_succeeded": 0,
        "bonsai_fits_failed": 0,
    }

    if len(q) != n or len(pos) != n:
        counters["windows_failed_length_match"] += 1
        return [], [], counters

    geom_ok = np.all(np.isfinite(pos), axis=1)
    n_missing_geometry = int(np.sum(~geom_ok))
    counters["hits_missing_geometry"] = n_missing_geometry
    counters["windows_with_missing_geometry_hits"] = int(n_missing_geometry > 0)

    def hit_truth(name: str, default, dtype, missing_counter: Optional[str] = None) -> Optional[np.ndarray]:
        arr = t_evt.get(name)
        if arr is None:
            if missing_counter is not None:
                counters[missing_counter] += 1
            # Missing truth branches are allowed; use a neutral default instead.
            return np.full(n, default, dtype=dtype)
        arr = np.asarray(arr)
        if len(arr) != n:
            counters["windows_failed_truth_length_match"] += 1
            # A present but mismatched truth branch means this window is unsafe.
            return None
        return arr.astype(dtype, copy=False)

    capture_arr = hit_truth("hit_from_capture", 0, np.int8, "windows_missing_hit_from_capture_branch")
    prompt_truth_available = "hit_from_prompt" in t_evt
    prompt_arr = hit_truth("hit_from_prompt", 0, np.int8, "windows_missing_hit_from_prompt_branch")
    background_arr = hit_truth("is_background", 0, np.int8)
    source_event_idx = hit_truth("source_event_idx", -1, np.int64)
    if capture_arr is None or prompt_arr is None or background_arr is None or source_event_idx is None:
        return [], [], counters

    capture_flag = capture_arr.astype(bool)
    prompt_flag = prompt_arr.astype(bool)
    is_background = background_arr.astype(bool)

    finite = np.isfinite(time) & np.isfinite(q) & geom_ok
    anchor_mask = w_evt.get("is_anchor_window", np.ones(n, dtype=bool)).astype(bool)
    prompt_search_idx = np.flatnonzero(finite & anchor_mask)
    prompt_raw_times = time[prompt_search_idx]
    prompt_corrected_times = prompt_raw_times - (
        np.linalg.norm(pos[prompt_search_idx] - prompt_vertex_cm[None, :], axis=1) / c_water
    )
    prompts = find_prompt_candidates(
        prompt_raw_times,
        corrected_times_ns=prompt_corrected_times,
        window_ns=args.prompt_window_ns,
        min_hits=args.prompt_min_hits,
        max_hits=args.prompt_max_hits,
        min_trms_ns=args.prompt_min_trms_ns,
        max_trms_ns=args.prompt_max_trms_ns,
        min_tmean_ns=args.prompt_min_tmean_ns,
        max_tmean_ns=args.prompt_max_tmean_ns,
        isolation_ns=args.prompt_isolation_ns,
    )
    counters["prompt_candidates_total"] = int(len(prompts))
    counters["windows_with_prompt_candidates"] = int(len(prompts) > 0)
    if prompt_truth_available:
        counters["prompt_candidates_truth_checked"] = int(len(prompts))
        matched, unmatched = count_truth_matched_prompts(
            prompts,
            prompt_search_idx,
            prompt_flag,
        )
        counters["prompt_candidates_truth_matched"] = matched
        counters["prompt_candidates_truth_unmatched"] = unmatched
    if not prompts:
        return [], [], counters

    # Continuous dark-noise removal is applied inside each prompt-specific
    # delayed search. Applying it globally would allow unrelated hits in another
    # prompt search window to veto a valid candidate.
    pmt_key = make_pmt_key(w_evt)
    rows: List[Dict] = []
    root_rows: List[Dict] = []
    has_signal_candidate = False
    candidate_id = 0
    if seen_candidate_hitsets is None:
        seen_candidate_hitsets = set()
    if seen_raw_candidate_hitsets is None:
        seen_raw_candidate_hitsets = set()

    for prompt in prompts:
        search = finite.copy()
        search &= time >= prompt.time_ns + args.capture_start_after_prompt_ns
        search &= time <= prompt.time_ns + args.capture_search_window_ns

        search_idx0 = np.flatnonzero(search)
        if len(search_idx0) == 0:
            continue

        # Nmax200 is a local burst veto inside the delayed search, not a veto
        # of the full 300 us prompt-associated region. We first record the raw
        # delayed-search maximum for diagnostics, then remove only the hits that
        # lie in over-threshold 200 ns burst windows.
        nmax200 = max_count_in_window(time[search_idx0], args.nmax200_window_ns)
        keep_nmax200 = dense_time_window_keep_mask(
            time[search_idx0],
            width_ns=args.nmax200_window_ns,
            max_hits=args.nmax200_cut,
        )
        n_removed_nmax200 = int(np.sum(~keep_nmax200))
        if n_removed_nmax200:
            counters["delayed_searches_with_nmax200_veto"] += 1
            counters["hits_removed_by_nmax200_veto"] += n_removed_nmax200
            keep_burst = np.ones(n, dtype=bool)
            keep_burst[search_idx0] = keep_nmax200
            search &= keep_burst

        search_idx0 = np.flatnonzero(search)
        if len(search_idx0) == 0:
            counters["delayed_searches_emptied_by_nmax200_veto"] += 1
            continue

        # Same-PMT continuous-noise cleaning is applied after the local burst
        # veto so a dense burst cannot be hidden by removing repeated PMT hits
        # first. The cleaning remains prompt-specific, because unrelated hits in
        # another delayed search should not veto this one.
        keep_noise = np.ones(n, dtype=bool)
        keep_search = continuous_noise_keep_mask(
            time[search_idx0],
            pmt_key[search_idx0],
            args.continuous_noise_ns,
        )
        keep_noise[search_idx0] = keep_search
        search &= keep_noise

        search_idx = np.flatnonzero(search)
        if len(search_idx) == 0:
            continue

        pos_s = pos[search_idx]
        time_s = time[search_idx]
        tcorr_s = time_s - np.linalg.norm(pos_s - prompt_vertex_cm[None, :], axis=1) / c_water

        cand_defs = greedy_nn_candidates(
            tcorr_s,
            width_ns=args.nn_window_ns,
            nn_cut=args.nn_cut,
            max_candidates=args.max_candidates_per_prompt,
        )
        if not cand_defs:
            continue

        counters["prompt_candidates_with_capture_candidates"] += 1

        for cand in cand_defs:
            init_trms = float(cand["trms"])
            cand_center = float(cand["tcorr_center"])
            raw_global = search_idx[np.asarray(cand["idx_local"], dtype=int)]
            if len(raw_global) == 0:
                continue

            # Most repeated candidates are already identical at the raw Nn
            # stage. Skipping them here avoids the expensive vertex scan.
            raw_hitset_key = candidate_hitset_key(w_evt, raw_global)
            if raw_hitset_key in seen_raw_candidate_hitsets:
                counters["raw_capture_candidates_skipped_by_hitset"] += 1
                continue
            seen_raw_candidate_hitsets.add(raw_hitset_key)

            # Limit expensive vertex scans to hits near the original candidate
            # in corrected time. If the peak is too sparse, fall back to all
            # delayed-search hits so the refit remains defined.
            context = np.abs(tcorr_s - cand_center) <= args.fit_context_ns
            if np.sum(context) < max(args.nn_cut + 1, 6):
                context = np.ones_like(tcorr_s, dtype=bool)
            context_idx = np.flatnonzero(context)
            time_c = time_s[context_idx]
            pos_c = pos_s[context_idx]
            channel_keys_c = np.column_stack(
                [
                    w_evt["slot"][search_idx][context_idx],
                    w_evt["pos"][search_idx][context_idx],
                ]
            )
            seed_context_loc = np.flatnonzero(
                np.isin(context_idx, np.asarray(cand["idx_local"], dtype=int))
            )

            xfit, nn_refit, best_context_loc, trmsp = refit_vertex_by_multilateration_grid(
                time_c,
                pos_c,
                prompt_vertex_cm,
                c_water,
                args.nn_window_ns,
                fit_hit_indices=seed_context_loc,
                fit_channel_keys=channel_keys_c,
                xyz_bounds_cm=args.multilateration_xyz_bounds_cm,
                coarse_step_cm=args.multilateration_coarse_step_cm,
                fine_step_cm=args.multilateration_fine_step_cm,
                refine_halfwidth_cm=args.multilateration_refine_halfwidth_cm,
                dt_cut_ns=args.multilateration_dt_cut_ns,
                grid_chunk_size=args.multilateration_grid_chunk_size,
                min_fit_hits=args.multilateration_min_hits,
                earliest_per_channel=args.multilateration_earliest_per_channel,
                wall=wall,
                wall_margin_cm=args.fit_wall_margin_cm,
            )
            final_local_s = context_idx[best_context_loc]
            final_global = search_idx[final_local_s]
            if len(final_global) == 0:
                continue

            # The same physical capture can be selected by several prompt
            # candidates, and several greedy seeds can refit to the same final
            # hit set. Keep the first occurrence and count later exact repeats.
            hitset_key = candidate_hitset_key(w_evt, final_global)
            if hitset_key in seen_candidate_hitsets:
                counters["capture_candidates_deduplicated_by_hitset"] += 1
                continue
            seen_candidate_hitsets.add(hitset_key)

            final_tcorr = time[final_global] - (
                np.linalg.norm(pos[final_global] - xfit[None, :], axis=1) / c_water
            )
            vertex_t = float(np.nanmean(final_tcorr)) if len(final_tcorr) else np.nan

            # BONSAI receives a broader neighborhood around the corrected-time
            # peak than the final Nn hit set used for the local vertex.
            bonsai_local_mask = np.abs(tcorr_s - cand_center) <= 0.5 * args.bonsai_window_ns
            bonsai_global = search_idx[bonsai_local_mask]
            counters["bonsai_fits_attempted"] += 1
            bonsai_fit = bonsai_fitter.fit(
                cable_ids=w_evt["cable_id"][bonsai_global],
                times_ns=time[bonsai_global],
                charges_pe=q[bonsai_global],
            )
            counters["bonsai_fits_succeeded"] += int(bonsai_fit.success)
            counters["bonsai_fits_failed"] += int(not bonsai_fit.success)
            bonsai_observables = calculate_bonsai_vertex_observables(
                xbonsai_cm=bonsai_fit.vertex_cm,
                xfit_cm=xfit,
                wall=wall,
            )

            # Vertex determination observables are built above: initial Nn/trms,
            # refit improvement, fitted wall distance, BONSAI consistency, and
            # tight subcluster timing.
            # N300 is computed here but categorized as noise characterization below:
            # it counts surrounding corrected-time activity in a wider window.
            n300 = int(np.sum(np.abs(tcorr_s - cand_center) <= 0.5 * args.n300_window_ns))

            topology = calculate_cherenkov_topology_observables(
                hit_pos_cm=pos[final_global],
                pmt_dir=None if pmt_dir is None else pmt_dir[final_global],
                xfit_cm=xfit,
                latt_cm=args.attenuation_length_cm,
                low_weight_cut=args.low_weight_cut,
                low_theta_deg=args.low_theta_deg,
                back_theta_deg=args.back_theta_deg,
            )

            noise = calculate_noise_observables(
                hit_pos_cm=pos[final_global],
                xfit_cm=xfit,
                charges=q[final_global],
                n300=n300,
                cluster_angle_deg=args.cluster_angle_deg,
                high_charge_pe=args.high_charge_pe,
            )

            # Label production starts here. No truth flags affect the fitted
            # vertex, topology, charge, or timing observables above.
            n_capture = int(np.sum(capture_flag[final_global]))
            n_prompt = int(np.sum(prompt_flag[final_global]))
            n_bkg = int(np.sum(is_background[final_global]))
            n_final = int(len(final_global))
            capture_fraction = float(n_capture / n_final) if n_final else 0.0
            capture_sources = source_event_idx[final_global][capture_flag[final_global]]
            valid_capture_sources = capture_sources[capture_sources >= 0]
            if len(valid_capture_sources):
                source_ids, source_counts = np.unique(valid_capture_sources, return_counts=True)
                best_source_i = int(np.argmax(source_counts))
                dominant_capture_source_idx = int(source_ids[best_source_i])
                dominant_capture_hits = int(source_counts[best_source_i])
                n_capture_sources = int(len(source_ids))
            else:
                dominant_capture_source_idx = -1
                dominant_capture_hits = n_capture
                n_capture_sources = int(n_capture > 0)
            dominant_capture_fraction = float(dominant_capture_hits / n_final) if n_final else 0.0
            label_signal = int(
                (
                    dominant_capture_hits >= args.min_capture_hits
                    and dominant_capture_fraction >= args.min_capture_fraction
                )
                or (dominant_capture_hits >= args.min_capture_hits_absolute)
            )

            source_window_offset = w_evt.get("source_window_offset")
            if source_window_offset is not None and n_final:
                final_window_offsets = np.asarray(source_window_offset)[final_global]
                first_window_offset = int(np.min(final_window_offsets))
                last_window_offset = int(np.max(final_window_offsets))
            else:
                first_window_offset = 0
                last_window_offset = 0

            source_entry = w_evt.get("source_entry")
            if source_entry is not None and n_final:
                final_source_entries = np.asarray(source_entry)[final_global]
                first_source_entry = int(np.min(final_source_entries))
                last_source_entry = int(np.max(final_source_entries))
            else:
                first_source_entry = -1
                last_source_entry = -1

            candidate_raw_mean_time = float(np.nanmean(time[final_global])) if n_final else np.nan
            fpdist = float(np.linalg.norm(xfit - prompt_vertex_cm))
            row = {
                "event_number": int(event_number),
                "candidate_id": int(candidate_id),
                "prompt_id": int(prompt.prompt_id),
                "prompt_time_ns": float(prompt.time_ns),
                "prompt_window_start_ns": float(prompt.window_start_ns),
                "prompt_window_end_ns": float(prompt.window_end_ns),
                "prompt_nhits": int(prompt.nhits),
                "prompt_trms_ns": float(prompt.trms_ns),
                "prompt_tmean_ns": float(prompt.tmean_ns),
                "candidate_tcorr_center_ns": cand_center,
                "candidate_raw_mean_time_ns": candidate_raw_mean_time,
                "candidate_time_from_prompt_ns": candidate_raw_mean_time - float(prompt.time_ns),
                "candidate_first_window_offset": first_window_offset,
                "candidate_last_window_offset": last_window_offset,
                "candidate_first_source_entry": first_source_entry,
                "candidate_last_source_entry": last_source_entry,
                "xfit_cm": float(xfit[0]),
                "yfit_cm": float(xfit[1]),
                "zfit_cm": float(xfit[2]),
                "xbonsai_cm": float(bonsai_fit.vertex_cm[0]),
                "ybonsai_cm": float(bonsai_fit.vertex_cm[1]),
                "zbonsai_cm": float(bonsai_fit.vertex_cm[2]),
                "bonsai_fit_success": int(bonsai_fit.success),
                "bonsai_n_fit": int(bonsai_fit.n_fit),
                "bonsai_n_input_hits": int(len(bonsai_global)),
                "bonsai_n_selected": int(bonsai_fit.n_selected),
                "bonsai_n_window": int(bonsai_fit.n_window),
                "bonsai_fit_goodness": float(bonsai_fit.fit_goodness),
                "bonsai_time_goodness": float(bonsai_fit.time_goodness),
                "Nn": int(cand["Nn"]),
                "trms": init_trms,
                "fpdist": fpdist,
                "delta_trms": float(init_trms - trmsp),
                "delta_Nn": int(nn_refit - cand["Nn"]),
                "fwall": wall.distance_to_wall(xfit),
                "trms3": min_subset_rms(final_tcorr, 3),
                "trms6": min_subset_rms(final_tcorr, 6),
                "n_hits_final": n_final,
                "n_capture_hits": n_capture,
                "dominant_capture_hits": dominant_capture_hits,
                "dominant_capture_fraction": dominant_capture_fraction,
                "dominant_capture_source_idx": dominant_capture_source_idx,
                "n_capture_sources": n_capture_sources,
                "n_prompt_hits": n_prompt,
                "n_background_hits": n_bkg,
                "capture_fraction": capture_fraction,
                "label": label_signal,
                "nmax200": int(nmax200),
            }
            row.update(bonsai_observables)
            row.update(topology)
            row.update(noise)
            rows.append(row)

            if pmt_dir is None:
                final_pmt_dir = np.full((n_final, 3), np.nan, dtype=float)
            else:
                final_pmt_dir = np.asarray(pmt_dir[final_global], dtype=float)
            root_rows.append(
                {
                    "hitx": pos[final_global, 0],
                    "hity": pos[final_global, 1],
                    "hitz": pos[final_global, 2],
                    "hitDirx": final_pmt_dir[:, 0],
                    "hitDiry": final_pmt_dir[:, 1],
                    "hitDirz": final_pmt_dir[:, 2],
                    "hitc": q[final_global],
                    "hitt": time[final_global],
                    "corrected_hitt": final_tcorr - vertex_t,
                    "label": label_signal,
                    "n_hits": n_final,
                    "vertex_x": float(xfit[0]),
                    "vertex_y": float(xfit[1]),
                    "vertex_z": float(xfit[2]),
                    "vertex_t": vertex_t,
                    "fpdist": fpdist,
                    "event_number": int(event_number),
                    "candidate_id": int(candidate_id),
                    "prompt_id": int(prompt.prompt_id),
                    "prompt_time_ns": float(prompt.time_ns),
                    "candidate_tcorr_center_ns": cand_center,
                    "n_capture_hits": n_capture,
                    "n_background_hits": n_bkg,
                    "dominant_capture_fraction": dominant_capture_fraction,
                    "capture_fraction": capture_fraction,
                }
            )

            counters["capture_candidates_total"] += 1
            counters["capture_candidates_signal"] += int(label_signal)
            counters["capture_candidates_background"] += int(not label_signal)
            has_signal_candidate |= bool(label_signal)
            candidate_id += 1

    counters["windows_with_capture_candidates"] = int(len(rows) > 0)
    counters["windows_with_signal_capture_candidate"] = int(has_signal_candidate)
    return rows, root_rows, counters


def truth_prompt_group_stats(
    times_ns: np.ndarray,
    hit_from_prompt: np.ndarray,
    source_event_idx: np.ndarray,
    trms_cut_ns: float,
) -> Tuple[int, int]:
    """
    Count truth prompt groups and how many are broad scintillation-like.

    Prompt groups are formed from truth-tagged prompt hits, grouped by
    source_event_idx when available. The tRMS is computed from the readout hit
    times of those truth-tagged prompt hits.
    """
    times_ns = np.asarray(times_ns, dtype=float)
    prompt_mask = np.asarray(hit_from_prompt).astype(bool)
    source_event_idx = np.asarray(source_event_idx, dtype=np.int64)

    if not np.any(prompt_mask):
        return 0, 0

    prompt_sources = source_event_idx[prompt_mask]
    valid_sources = np.unique(prompt_sources[prompt_sources >= 0])
    total_groups = 0
    broad_groups = 0

    for src in valid_sources:
        group_times = times_ns[prompt_mask & (source_event_idx == src)]
        trms_ns = safe_rms(group_times)
        total_groups += 1
        broad_groups += int(np.isfinite(trms_ns) and trms_ns > trms_cut_ns)

    orphan_mask = prompt_mask & (source_event_idx < 0)
    if np.any(orphan_mask):
        trms_ns = safe_rms(times_ns[orphan_mask])
        total_groups += 1
        broad_groups += int(np.isfinite(trms_ns) and trms_ns > trms_cut_ns)

    return total_groups, broad_groups


def truth_counts_for_chunk(
    warr,
    tarr,
    args: argparse.Namespace,
    n_anchor_entries: int,
    geometry: PmtGeometry,
    prompt_vertex_cm: np.ndarray,
) -> Dict[str, int]:
    """Compute direct TTrueInfo counts for anchor windows in one chunk."""
    truth_counts = {
        "true_prompts_total": 0,
        "true_prompts_trms_gt_100_ns": 0,
        "true_neutron_captures_total": 0,
        "windows_with_true_capture_hits": 0,
    }

    for local_index in range(n_anchor_entries):
        if "prompt_time" in tarr.fields:
            prompt_times = as_np(tarr["prompt_time"][local_index], float)
            truth_counts["true_prompts_total"] += int(len(prompt_times))
        elif "hit_from_prompt" in tarr.fields:
            prompt_arr = as_np(tarr["hit_from_prompt"][local_index], np.int8)
            source_arr = (
                as_np(tarr["source_event_idx"][local_index], np.int64)
                if "source_event_idx" in tarr.fields
                else np.full(len(prompt_arr), -1, dtype=np.int64)
            )
            n_groups, _ = truth_prompt_group_stats(
                times_ns=np.zeros(len(prompt_arr), dtype=float),
                hit_from_prompt=prompt_arr,
                source_event_idx=source_arr,
                trms_cut_ns=100.0,
            )
            truth_counts["true_prompts_total"] += int(n_groups)

        if "capture_t" in tarr.fields:
            capture_times = as_np(tarr["capture_t"][local_index], float)
            n_captures = int(len(capture_times))
            truth_counts["true_neutron_captures_total"] += n_captures
            truth_counts["windows_with_true_capture_hits"] += int(n_captures > 0)
        elif "hit_from_capture" in tarr.fields:
            capture_arr = as_np(tarr["hit_from_capture"][local_index], np.int8)
            source_arr = (
                as_np(tarr["source_event_idx"][local_index], np.int64)
                if "source_event_idx" in tarr.fields
                else np.full(len(capture_arr), -1, dtype=np.int64)
            )
            capture_mask = capture_arr.astype(bool)
            capture_sources = source_arr[capture_mask]
            valid_sources = np.unique(capture_sources[capture_sources >= 0])
            n_captures = int(len(valid_sources))
            if n_captures == 0 and np.any(capture_mask):
                n_captures = 1
            truth_counts["true_neutron_captures_total"] += n_captures
            truth_counts["windows_with_true_capture_hits"] += int(np.any(capture_mask))

        if "hit_from_prompt" in tarr.fields:
            prompt_arr = as_np(tarr["hit_from_prompt"][local_index], np.int8)
            source_arr = (
                as_np(tarr["source_event_idx"][local_index], np.int64)
                if "source_event_idx" in tarr.fields
                else np.full(len(prompt_arr), -1, dtype=np.int64)
            )
            time_arr = as_np(warr[args.time_branch][local_index], float)
            slot_arr = as_np(warr["hit_mpmt_slot_ids"][local_index], int)
            pos_arr = as_np(warr["hit_pmt_position_ids"][local_index], int)
            pmt_pos_cm, _, _ = geometry.lookup_slotpos(
                slot_arr,
                pos_arr,
                position_base=args.pmt_position_base,
            )
            c_water = C_LIGHT_CM_PER_NS / args.water_refractive_index
            corrected_time_arr = time_arr - (
                np.linalg.norm(pmt_pos_cm - prompt_vertex_cm[None, :], axis=1) / c_water
            )
            _, broad_groups = truth_prompt_group_stats(
                times_ns=corrected_time_arr,
                hit_from_prompt=prompt_arr,
                source_event_idx=source_arr,
                trms_cut_ns=100.0,
            )
            truth_counts["true_prompts_trms_gt_100_ns"] += int(broad_groups)

    return truth_counts


def grouped_counter_summary(
    counters: Dict[str, int],
    truth_counts: Dict[str, int],
) -> Dict[str, Dict[str, int]]:
    """Return the extraction counters grouped by meaning for printing and JSON."""
    truth_keys = (
        "true_prompts_total",
        "true_prompts_trms_gt_100_ns",
        "true_neutron_captures_total",
        "windows_with_true_capture_hits",
    )
    preselection_keys = (
        "windows_total",
        "windows_with_prompt_candidates",
        "prompt_candidates_total",
        "prompt_candidates_truth_checked",
        "prompt_candidates_truth_matched",
        "prompt_candidates_truth_unmatched",
        "delayed_searches_with_nmax200_veto",
        "hits_removed_by_nmax200_veto",
        "delayed_searches_emptied_by_nmax200_veto",
        "prompt_candidates_with_capture_candidates",
        "windows_with_capture_candidates",
        "capture_candidates_total",
        "raw_capture_candidates_skipped_by_hitset",
        "capture_candidates_deduplicated_by_hitset",
        "capture_candidates_signal",
        "capture_candidates_background",
        "windows_with_signal_capture_candidate",
        "bonsai_fits_attempted",
        "bonsai_fits_succeeded",
        "bonsai_fits_failed",
    )
    sanity_keys = (
        "windows_failed_length_match",
        "windows_failed_truth_length_match",
        "windows_missing_hit_from_capture_branch",
        "windows_missing_hit_from_prompt_branch",
        "windows_with_missing_geometry_hits",
        "hits_missing_geometry",
    )
    return {
        "truth_info": {key: truth_counts[key] for key in truth_keys},
        "preselection": {key: counters[key] for key in preselection_keys},
        "sanity_checks": {key: counters[key] for key in sanity_keys},
    }


def format_grouped_counter_summary(grouped: Dict[str, Dict[str, int]]) -> str:
    """Human-readable extraction summary with short headers and explanations."""
    section_titles = {
        "truth_info": (
            "Truth Info (from TTrueInfo; prompt tRMS uses TOF-corrected "
            "truth-tagged prompt hits)"
        ),
        "preselection": "Pre-selection",
        "sanity_checks": "Sanity Checks",
    }
    descriptions = {
        "true_prompts_total": "Total true prompts.",
        "true_prompts_trms_gt_100_ns": (
            "True prompts whose fixed-source TOF-corrected truth-tagged prompt-hit "
            "tRMS is > 100 ns."
        ),
        "true_neutron_captures_total": "Total true neutron captures.",
        "windows_with_true_capture_hits": (
            "Readout windows containing at least one truth-tagged capture hit."
        ),
        "windows_total": "Anchor 270 us readout windows processed.",
        "windows_with_prompt_candidates": (
            "Windows where prompt tagging found at least one prompt candidate."
        ),
        "prompt_candidates_total": (
            "Total prompt candidates after prompt-window de-duplication."
        ),
        "prompt_candidates_truth_checked": (
            "Prompt candidates checked against the hit_from_prompt truth branch."
        ),
        "prompt_candidates_truth_matched": (
            "Prompt candidates containing at least one hit_from_prompt truth-tagged "
            "hit (diagnostic only)."
        ),
        "prompt_candidates_truth_unmatched": (
            "Prompt candidates containing no hit_from_prompt truth-tagged hits "
            "(diagnostic only)."
        ),
        "delayed_searches_with_nmax200_veto": (
            "Prompt-associated delayed searches where a local 200 ns burst was vetoed."
        ),
        "hits_removed_by_nmax200_veto": (
            "Total delayed-search hits removed by the local Nmax200 veto."
        ),
        "delayed_searches_emptied_by_nmax200_veto": (
            "Delayed searches left with no hits after the local Nmax200 veto."
        ),
        "prompt_candidates_with_capture_candidates": (
            "Prompt candidates whose delayed search produced at least one raw capture "
            "candidate before hit-set de-duplication."
        ),
        "windows_with_capture_candidates": (
            "Windows that produced at least one capture candidate."
        ),
        "capture_candidates_total": "Final capture candidate rows written out.",
        "raw_capture_candidates_skipped_by_hitset": (
            "Repeated raw Nn hit clusters skipped before vertex fitting."
        ),
        "capture_candidates_deduplicated_by_hitset": (
            "Exact repeated final hit clusters removed before writing."
        ),
        "capture_candidates_signal": (
            "Final capture candidates labeled as signal from truth."
        ),
        "capture_candidates_background": (
            "Final capture candidates labeled as background from truth."
        ),
        "windows_with_signal_capture_candidate": (
            "Windows containing at least one signal-labeled capture candidate."
        ),
        "bonsai_fits_attempted": "Final de-duplicated candidates sent to BONSAI.",
        "bonsai_fits_succeeded": "BONSAI fits that converged.",
        "bonsai_fits_failed": "BONSAI fits that did not return a usable vertex.",
        "windows_failed_length_match": (
            "Windows where readout hit arrays had inconsistent lengths."
        ),
        "windows_failed_truth_length_match": (
            "Windows where truth hit arrays did not match readout hit counts."
        ),
        "windows_missing_hit_from_capture_branch": (
            "Windows where the hit_from_capture truth branch was missing."
        ),
        "windows_missing_hit_from_prompt_branch": (
            "Windows where the hit_from_prompt truth branch was missing."
        ),
        "windows_with_missing_geometry_hits": (
            "Windows containing at least one hit not matched to the geofile."
        ),
        "hits_missing_geometry": "Total hits not matched to the geofile.",
    }

    lines = [""]
    for section, values in grouped.items():
        lines.append(section_titles[section])
        for name, value in values.items():
            lines.append(f"  {name}: {value}  {descriptions[name]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def extract_candidates(args: argparse.Namespace) -> Path:
    """Run prompt tagging, delayed candidate finding, and observable extraction."""
    try:
        import uproot
    except ImportError as exc:
        raise SystemExit("Missing ROOT I/O dependency. Install with: pip install uproot") from exc

    root_path = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.bonsai_window_ns <= 0:
        raise ValueError("--bonsai-window-ns must be positive")

    prompt_vertex_cm = np.asarray(args.prompt_vertex_cm, dtype=float)
    grid_counts = double_scan_grid_counts(
        xyz_bounds_cm=args.multilateration_xyz_bounds_cm,
        coarse_step_cm=args.multilateration_coarse_step_cm,
        refine_halfwidth_cm=args.multilateration_refine_halfwidth_cm,
        fine_step_cm=args.multilateration_fine_step_cm,
    )
    geometry = PmtGeometry.from_geofile(Path(args.geometry_file), scale_to_cm=args.pos_scale_to_cm)

    if args.wall_json is not None:
        wall_data = json.loads(Path(args.wall_json).read_text())
        wall = WallEstimator(**wall_data)
    else:
        wall = WallEstimator.from_geometry(geometry, axis=args.wall_axis)

    bonsai_fitter = BonsaiVertexFitter(
        bonsai_dir=Path(args.bonsai_dir),
        wcsim_dir=Path(args.wcsim_dir),
        fit_param=Path(args.bonsai_param),
        geometry_root=Path(args.bonsai_geometry_root),
    )
    bonsai_geometry_check = bonsai_fitter.validate_geometry(geometry)

    f = uproot.open(root_path)
    wtree = f["WCTEReadoutWindows"]
    ttree = f["TTrueInfo"]
    n_entries = min(wtree.num_entries, ttree.num_entries)
    if args.max_events and args.max_events > 0:
        n_entries = min(n_entries, args.max_events)

    required_w = [
        args.time_branch,
        args.charge_branch,
        "hit_mpmt_slot_ids",
        "hit_pmt_position_ids",
    ]
    wtree_keys = set(wtree.keys())
    ttree_keys = set(ttree.keys())
    missing_w = [b for b in required_w if b not in wtree_keys]
    if missing_w:
        raise RuntimeError(
            "Missing WCTE branches needed for observable geometry: " + ", ".join(missing_w)
        )

    w_branches = required_w + (["event_number"] if "event_number" in wtree_keys else [])
    optional_w = [
        "hit_mpmt_card_ids",
        "hit_pmt_channel_ids",
    ]
    w_branches += [b for b in optional_w if b in wtree_keys]

    optional_t = [
        "hit_from_capture",
        "hit_from_prompt",
        "is_background",
        "capture_t",
        "prompt_time",
        "source_event_idx",
    ]
    t_branches = [b for b in optional_t if b in ttree_keys]

    rows: List[Dict] = []
    root_rows: List[Dict] = []
    counters: Dict[str, int] = {}
    seen_candidate_hitsets = set()
    seen_raw_candidate_hitsets = set()
    truth_counts: Dict[str, int] = {
        "true_prompts_total": 0,
        "true_prompts_trms_gt_100_ns": 0,
        "true_neutron_captures_total": 0,
        "windows_with_true_capture_hits": 0,
    }
    window_period_ns = args.window_duration_ns + args.window_gap_ns
    max_following_windows = following_window_count(
        args.capture_search_window_ns,
        args.window_duration_ns,
        args.window_gap_ns,
    )

    for start in range(0, n_entries, args.chunk_size):
        stop = min(n_entries, start + args.chunk_size)
        read_stop = min(n_entries, stop + max_following_windows)
        warr = wtree.arrays(w_branches, entry_start=start, entry_stop=read_stop, library="ak")
        tarr = ttree.arrays(t_branches, entry_start=start, entry_stop=read_stop, library="ak")

        chunk_truth_counts = truth_counts_for_chunk(
            warr=warr,
            tarr=tarr,
            args=args,
            n_anchor_entries=stop - start,
            geometry=geometry,
            prompt_vertex_cm=prompt_vertex_cm,
        )
        for key, value in chunk_truth_counts.items():
            truth_counts[key] += int(value)

        payloads = [
            build_window_payload(
                absolute_entry=start + i,
                local_index=i,
                warr=warr,
                tarr=tarr,
                args=args,
                geometry=geometry,
            )
            for i in range(read_stop - start)
        ]

        for i in range(stop - start):
            event_number = payloads[i]["event_number"]
            w_evt, t_evt = combine_window_payloads(
                payloads,
                first_local_index=i,
                max_following_windows=max_following_windows,
                window_period_ns=window_period_ns,
            )

            evt_rows, evt_root_rows, evt_counts = event_to_candidates(
                event_number=event_number,
                w_evt=w_evt,
                t_evt=t_evt,
                prompt_vertex_cm=prompt_vertex_cm,
                wall=wall,
                bonsai_fitter=bonsai_fitter,
                args=args,
                seen_candidate_hitsets=seen_candidate_hitsets,
                seen_raw_candidate_hitsets=seen_raw_candidate_hitsets,
            )
            rows.extend(evt_rows)
            root_rows.extend(evt_root_rows)
            for k, v in evt_counts.items():
                counters[k] = counters.get(k, 0) + int(v)

        print(
            f"Processed entries {start:,}-{stop:,} / {n_entries:,}; "
            f"capture_candidates so far = {len(rows):,}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    for col in OBSERVABLE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    out_path = outdir / "candidates.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        out_path = outdir / "candidates.csv"
        df.to_csv(out_path, index=False)

    root_hit_info_path = write_candidate_hit_info_root(
        root_rows=root_rows,
        out_path=outdir / "candidates.root",
    )

    observable_plot_dir = outdir / "obs_plots"
    observable_plot_summary = plot_observable_pdfs(
        candidates=df,
        outdir=observable_plot_dir,
        observable_columns=OBSERVABLE_COLUMNS,
        n_bins=args.observable_plot_bins,
    )
    grouped_counts = grouped_counter_summary(counters, truth_counts)
    summary = {
        "root": str(root_path),
        "n_entries_processed": int(n_entries),
        "prompt_vertex_cm": prompt_vertex_cm.tolist(),
        "scan_type": "multilateration_double_grid",
        "scan_n_vertices": int(grid_counts["total"]),
        "scan_n_coarse_vertices": int(grid_counts["coarse"]),
        "scan_n_fine_vertices": int(grid_counts["fine"]),
        "capture_search_following_windows": int(max_following_windows),
        "window_period_ns": float(window_period_ns),
        "observable_columns": OBSERVABLE_COLUMNS,
        "observable_groups": OBSERVABLE_GROUPS,
        "observable_plots": {
            "capture_vs_accidental_directory": str(observable_plot_dir),
            "n_plots": int(observable_plot_summary["n_plots"]),
            "density_normalized_per_component": True,
        },
        "root_hit_info": {
            "path": str(root_hit_info_path),
            "tree": "THits",
            "downstream_use": "point-cloud method training",
            "hit_branches": [
                "hitx",
                "hity",
                "hitz",
                "hitDirx",
                "hitDiry",
                "hitDirz",
                "hitc",
                "hitt",
                "corrected_hitt",
            ],
            "label_branch": "label",
            "vertex_branches": ["vertex_x", "vertex_y", "vertex_z", "vertex_t"],
            "distance_branches": ["fpdist"],
            "excluded_standard_branches": ["eventType", "energy", "ncap_target"],
        },
        "geometry": geometry.summary(),
        "wall_estimator": asdict(wall),
        "bonsai": {
            **bonsai_fitter.summary(),
            **bonsai_geometry_check,
            "fit_window_ns": float(args.bonsai_window_ns),
        },
        "truth_info": grouped_counts["truth_info"],
        "preselection": grouped_counts["preselection"],
        "sanity_checks": grouped_counts["sanity_checks"],
        "args": vars(args),
    }
    (outdir / "extraction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (outdir / "wall_estimator.json").write_text(json.dumps(asdict(wall), indent=2, sort_keys=True))

    print(f"\nSaved candidates: {out_path}")
    print(f"Saved ROOT hit info: {root_hit_info_path}")
    print(f"Saved summary:    {outdir / 'extraction_summary.json'}")
    print(f"Saved obs plots:  {observable_plot_dir}")
    print(f"Geometry:         {geometry.summary()}")
    print(f"Wall estimator:   {asdict(wall)}")
    print(f"BONSAI:           {bonsai_fitter.summary()}")
    print(format_grouped_counter_summary(grouped_counts))
    return out_path


def optional_float_arg(value: str) -> Optional[float]:
    """Parse a float CLI value, accepting none/null/nan to disable a cut."""
    text = str(value).strip().lower()
    if text in {"none", "null", "nan"}:
        return None
    return float(value)


def add_extract_args(p: argparse.ArgumentParser) -> None:
    """Attach candidate-preselection and observable-extraction CLI options."""
    p.add_argument("--root", required=True, help="Input ROOT file")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--max-events", type=int, default=0, help="0 means all entries")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument(
        "--observable-plot-bins",
        type=int,
        default=DEFAULT_OBSERVABLE_PLOT_BINS,
        help="Requested number of bins for post-extraction observable PDFs",
    )

    p.add_argument("--time-branch", default="hit_pmt_calibrated_times")
    p.add_argument("--charge-branch", default="hit_pmt_charges")
    p.add_argument(
        "--geometry-file",
        default=str(DEFAULT_GEOMETRY_FILE),
        help="WCSim geofile used to map WCTE slot/position ids to PMT geometry",
    )
    p.add_argument(
        "--pos-scale-to-cm",
        type=float,
        default=DEFAULT_GEOMETRY_SCALE_TO_CM,
        help="Scale geofile PMT positions to cm",
    )
    p.add_argument(
        "--pmt-position-base",
        choices=["zero", "one", "auto"],
        default=DEFAULT_PMT_POSITION_BASE,
        help="Indexing convention for hit_pmt_position_ids",
    )
    p.add_argument(
        "--prompt-vertex-cm",
        type=float,
        nargs=3,
        default=list(DEFAULT_PROMPT_VERTEX_CM),
    )
    p.add_argument(
        "--bonsai-dir",
        default=str(DEFAULT_BONSAI_DIR),
        help="hk-BONSAI installation directory",
    )
    p.add_argument(
        "--bonsai-param",
        default=str(DEFAULT_BONSAI_PARAM),
        help="BONSAI fit parameter file",
    )
    p.add_argument(
        "--bonsai-geometry-root",
        default=str(DEFAULT_BONSAI_GEOMETRY_ROOT),
        help="WCSim ROOT geometry carrier used to initialize BONSAI",
    )
    p.add_argument(
        "--wcsim-dir",
        default=str(DEFAULT_WCSIM_DIR),
        help="WCSim installation directory",
    )
    p.add_argument(
        "--bonsai-window-ns",
        type=float,
        default=DEFAULT_BONSAI_WINDOW_NS,
        help="Width of the corrected-time hit neighborhood supplied to BONSAI",
    )

    p.add_argument("--window-duration-ns", type=float, default=DEFAULT_WINDOW_DURATION_NS)
    p.add_argument("--window-gap-ns", type=float, default=DEFAULT_WINDOW_GAP_NS)
    p.add_argument(
        "--capture-search-window-ns",
        type=float,
        default=DEFAULT_CAPTURE_SEARCH_WINDOW_NS,
    )
    p.add_argument(
        "--capture-start-after-prompt-ns",
        type=float,
        default=DEFAULT_CAPTURE_START_AFTER_PROMPT_NS,
    )

    p.add_argument("--prompt-window-ns", type=float, default=DEFAULT_PROMPT_WINDOW_NS)
    p.add_argument("--prompt-min-hits", type=int, default=DEFAULT_PROMPT_MIN_HITS)
    p.add_argument("--prompt-max-hits", type=int, default=DEFAULT_PROMPT_MAX_HITS)
    p.add_argument("--prompt-min-trms-ns", type=float, default=DEFAULT_PROMPT_MIN_TRMS_NS)
    p.add_argument("--prompt-max-trms-ns", type=float, default=DEFAULT_PROMPT_MAX_TRMS_NS)
    p.add_argument("--prompt-min-tmean-ns", type=float, default=DEFAULT_PROMPT_MIN_TMEAN_NS)
    p.add_argument("--prompt-max-tmean-ns", type=float, default=DEFAULT_PROMPT_MAX_TMEAN_NS)
    p.add_argument("--prompt-isolation-ns", type=float, default=DEFAULT_PROMPT_ISOLATION_NS)

    p.add_argument(
        "--water-refractive-index",
        type=float,
        default=DEFAULT_WATER_REFRACTIVE_INDEX,
    )
    p.add_argument("--nmax200-window-ns", type=float, default=DEFAULT_NMAX200_WINDOW_NS)
    p.add_argument(
        "--nmax200-cut",
        type=int,
        default=DEFAULT_NMAX200_CUT,
        help="Locally veto delayed-search 200 ns windows with at least this many hits",
    )
    p.add_argument("--continuous-noise-ns", type=float, default=DEFAULT_CONTINUOUS_NOISE_NS)
    p.add_argument("--nn-window-ns", type=float, default=DEFAULT_NN_WINDOW_NS)
    p.add_argument(
        "--nn-cut",
        type=int,
        default=DEFAULT_NN_CUT,
        help="Keep candidates with Nn > this value",
    )
    p.add_argument(
        "--max-candidates-per-prompt",
        type=int,
        default=DEFAULT_MAX_CANDIDATES_PER_PROMPT,
        help="Maximum number of delayed capture candidates returned for one prompt candidate",
    )

    p.add_argument("--fit-context-ns", type=float, default=DEFAULT_FIT_CONTEXT_NS)
    p.add_argument("--fit-wall-margin-cm", type=float, default=DEFAULT_FIT_WALL_MARGIN_CM)
    p.add_argument(
        "--multilateration-xyz-bounds-cm",
        type=float,
        default=DEFAULT_MULTILATERATION_XYZ_BOUNDS_CM,
    )
    p.add_argument(
        "--multilateration-coarse-step-cm",
        type=float,
        default=DEFAULT_MULTILATERATION_COARSE_STEP_CM,
    )
    p.add_argument(
        "--multilateration-fine-step-cm",
        type=float,
        default=DEFAULT_MULTILATERATION_FINE_STEP_CM,
    )
    p.add_argument(
        "--multilateration-refine-halfwidth-cm",
        type=float,
        default=DEFAULT_MULTILATERATION_REFINE_HALFWIDTH_CM,
    )
    p.add_argument(
        "--multilateration-dt-cut-ns",
        type=optional_float_arg,
        default=DEFAULT_MULTILATERATION_DT_CUT_NS,
        help="Fine-grid |dt| cut in ns; use 'none' to disable",
    )
    p.add_argument(
        "--multilateration-grid-chunk-size",
        type=int,
        default=DEFAULT_MULTILATERATION_GRID_CHUNK_SIZE,
    )
    p.add_argument("--multilateration-min-hits", type=int, default=DEFAULT_MULTILATERATION_MIN_HITS)
    p.add_argument(
        "--multilateration-earliest-per-channel",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MULTILATERATION_EARLIEST_PER_CHANNEL,
        help="Keep only the earliest Nn-burst hit per slot/PMT during the grid fit",
    )
    p.add_argument("--n300-window-ns", type=float, default=DEFAULT_N300_WINDOW_NS)

    p.add_argument("--attenuation-length-cm", type=float, default=DEFAULT_ATTENUATION_LENGTH_CM)
    p.add_argument("--low-weight-cut", type=float, default=DEFAULT_LOW_WEIGHT_CUT)
    p.add_argument("--low-theta-deg", type=float, default=DEFAULT_LOW_THETA_DEG)
    p.add_argument("--back-theta-deg", type=float, default=DEFAULT_BACK_THETA_DEG)
    p.add_argument("--cluster-angle-deg", type=float, default=DEFAULT_CLUSTER_ANGLE_DEG)
    p.add_argument("--high-charge-pe", type=float, default=DEFAULT_HIGH_CHARGE_PE)

    p.add_argument("--wall-axis", choices=["x", "y", "z"], default=DEFAULT_WALL_AXIS)
    p.add_argument("--wall-json", default=None, help="Optional JSON with WallEstimator fields")

    p.add_argument("--min-capture-hits", type=int, default=DEFAULT_MIN_CAPTURE_HITS)
    p.add_argument("--min-capture-fraction", type=float, default=DEFAULT_MIN_CAPTURE_FRACTION)
    p.add_argument("--min-capture-hits-absolute", type=int, default=DEFAULT_MIN_CAPTURE_HITS_ABSOLUTE)
