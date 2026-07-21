"""ROOT and readout-window helpers for candidate preselection."""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple

import numpy as np

from .geometry import PmtGeometry


HIT_TRUTH_BRANCHES = [
    "hit_from_capture",
    "hit_from_prompt",
    "is_background",
    "source_event_idx",
]


def as_np(x, dtype=None) -> np.ndarray:
    """Convert uproot/awkward scalars or jagged entries to a NumPy array."""
    try:
        import awkward as ak
    except ImportError as exc:
        raise SystemExit("Missing ROOT I/O dependency. Install with: pip install awkward") from exc

    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = ak.to_numpy(x) if hasattr(ak, "to_numpy") else np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.asarray(arr)


def following_window_count(
    capture_search_window_ns: float,
    window_duration_ns: float,
    window_gap_ns: float,
) -> int:
    """
    Number of following windows needed for a delayed search anchored in one window.

    The latest possible prompt is near the end of the current 270 us window. A
    300 us delayed search from that prompt can therefore reach into the next
    window and, for very late prompts, the beginning of the second following
    window. The inter-window gap is represented by the virtual time offset; no
    hits are inserted in the gap.
    """
    window_period_ns = window_duration_ns + window_gap_ns
    max_relative_time_ns = window_duration_ns + capture_search_window_ns
    return int(math.floor(max_relative_time_ns / window_period_ns))


def build_window_payload(
    absolute_entry: int,
    local_index: int,
    warr,
    tarr,
    args: argparse.Namespace,
    geometry: PmtGeometry,
) -> Dict[str, object]:
    """
    Convert one ROOT entry into arrays used by the preselection.

    Only the readout branches are used for hit geometry. Hit-level TTrueInfo
    arrays retain matching hit order. Capture times are carried separately for
    truth-count diagnostics, never for observable values.
    """
    event_number = (
        int(as_np(warr["event_number"][local_index]).item())
        if "event_number" in warr.fields
        else absolute_entry
    )
    w_evt = {
        "time": as_np(warr[args.time_branch][local_index], float),
        "charge": as_np(warr[args.charge_branch][local_index], float),
        "slot": as_np(warr["hit_mpmt_slot_ids"][local_index], int),
        "pos": as_np(warr["hit_pmt_position_ids"][local_index], int),
    }
    w_evt["hit_index"] = np.arange(len(w_evt["time"]), dtype=np.int64)
    if "hit_mpmt_card_ids" in warr.fields:
        w_evt["card"] = as_np(warr["hit_mpmt_card_ids"][local_index], int)
    if "hit_pmt_channel_ids" in warr.fields:
        w_evt["channel"] = as_np(warr["hit_pmt_channel_ids"][local_index], int)

    pmt_pos_cm, pmt_dir, _ = geometry.lookup_slotpos(
        w_evt["slot"],
        w_evt["pos"],
        position_base=args.pmt_position_base,
    )
    cable_id, _ = geometry.lookup_cable_ids(
        w_evt["slot"],
        w_evt["pos"],
        position_base=args.pmt_position_base,
    )
    w_evt["pmt_pos_cm"] = pmt_pos_cm
    w_evt["pmt_dir"] = pmt_dir
    w_evt["cable_id"] = cable_id

    t_evt = {}
    for b in HIT_TRUTH_BRANCHES:
        if b in tarr.fields:
            t_evt[b] = as_np(tarr[b][local_index])
    if "capture_t" in tarr.fields:
        t_evt["capture_t"] = as_np(tarr["capture_t"][local_index], float)

    return {
        "absolute_entry": int(absolute_entry),
        "event_number": event_number,
        "w_evt": w_evt,
        "t_evt": t_evt,
    }


def combine_window_payloads(
    payloads: List[Dict[str, object]],
    first_local_index: int,
    max_following_windows: int,
    window_period_ns: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Build the continuous-time hit array used for one anchor-window search.

    Hits from following readout windows are shifted by one or more periods
    (window duration plus gap). No artificial hits are inserted in the gap; the
    gap simply has no data while the delayed-search clock keeps running.
    """
    selected = payloads[first_local_index : first_local_index + max_following_windows + 1]

    w_out: Dict[str, List[np.ndarray]] = {
        "time": [],
        "charge": [],
        "slot": [],
        "pos": [],
        "cable_id": [],
        "pmt_pos_cm": [],
        "pmt_dir": [],
        "is_anchor_window": [],
        "source_window_offset": [],
        "source_entry": [],
        "source_hit_index": [],
        "source_event_number": [],
    }
    for optional_key in ["card", "channel"]:
        if optional_key in selected[0]["w_evt"]:
            w_out[optional_key] = []

    hit_truth_keys = [
        key for key in HIT_TRUTH_BRANCHES if key in selected[0]["t_evt"]
    ]
    t_out: Dict[str, List[np.ndarray]] = {key: [] for key in hit_truth_keys}
    if "capture_t" in selected[0]["t_evt"]:
        t_out["capture_t"] = []

    for window_offset, payload in enumerate(selected):
        w_evt = payload["w_evt"]
        n_hits = len(w_evt["time"])
        virtual_offset_ns = window_offset * window_period_ns

        w_out["time"].append(w_evt["time"] + virtual_offset_ns)
        w_out["charge"].append(w_evt["charge"])
        w_out["slot"].append(w_evt["slot"])
        w_out["pos"].append(w_evt["pos"])
        w_out["cable_id"].append(w_evt["cable_id"])
        w_out["pmt_pos_cm"].append(w_evt["pmt_pos_cm"])
        w_out["pmt_dir"].append(w_evt["pmt_dir"])
        w_out["is_anchor_window"].append(np.full(n_hits, window_offset == 0, dtype=bool))
        w_out["source_window_offset"].append(np.full(n_hits, window_offset, dtype=np.int16))
        w_out["source_entry"].append(np.full(n_hits, payload["absolute_entry"], dtype=np.int64))
        w_out["source_hit_index"].append(w_evt["hit_index"])
        w_out["source_event_number"].append(np.full(n_hits, payload["event_number"], dtype=np.int64))

        for optional_key in ["card", "channel"]:
            if optional_key in w_out:
                w_out[optional_key].append(w_evt[optional_key])

        t_evt = payload["t_evt"]
        for key in hit_truth_keys:
            t_out[key].append(t_evt[key])
        if "capture_t" in t_out:
            t_out["capture_t"].append(t_evt["capture_t"] + virtual_offset_ns)

    combined_w = {key: np.concatenate(parts, axis=0) for key, parts in w_out.items()}
    combined_t = {
        key: np.concatenate(parts, axis=0)
        if parts
        else np.array([], dtype=float)
        for key, parts in t_out.items()
    }
    return combined_w, combined_t
