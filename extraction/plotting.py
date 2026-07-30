"""Candidate-observable probability-density plots produced after extraction."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def observable_bin_edges(
    signal_values: np.ndarray,
    background_values: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Build common bin edges so the two truth classes are directly comparable."""
    values = np.concatenate(
        [
            np.asarray(signal_values, dtype=float),
            np.asarray(background_values, dtype=float),
        ]
    )
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)

    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        padding = max(0.5, 0.05 * abs(low))
        return np.linspace(low - padding, high + padding, n_bins + 1)

    # Give integer-valued observables such as Nn and NhighQ one centered bin
    # per integer when their range is modest.
    integer_valued = np.allclose(values, np.rint(values), rtol=0.0, atol=1e-9)
    if integer_valued and high - low <= n_bins:
        return np.arange(np.floor(low) - 0.5, np.ceil(high) + 1.5, 1.0)

    return np.linspace(low, high, n_bins + 1)


def plot_observable_pdfs(
    candidates: pd.DataFrame,
    outdir: Path,
    observable_columns: Sequence[str],
    n_bins: int = 60,
) -> Dict[str, object]:
    """
    Plot one normalized truth-split histogram for every candidate observable.

    ``label == 1`` denotes a truth-matched neutron-capture candidate and
    ``label == 0`` an accidental coincidence. Each displayed component is
    normalized independently with ``density=True``. Non-finite values, including
    failed BONSAI fits, are omitted only from the affected observable.
    """
    if n_bins < 1:
        raise ValueError("Observable histogram bin count must be positive")
    if "label" not in candidates.columns:
        raise ValueError("Candidate table must contain the truth-derived 'label' column")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Avoid attempts to create a Matplotlib cache below an unavailable home
    # directory on batch nodes.
    matplotlib_cache = outdir / ".matplotlib"
    matplotlib_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    with warnings.catch_warnings():
        # The cluster's Matplotlib/pyparsing combination emits import-time
        # deprecation warnings unrelated to these plots.
        warnings.simplefilter("ignore", DeprecationWarning)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

    labels = pd.to_numeric(candidates["label"], errors="coerce").to_numpy(dtype=float)
    signal_mask = labels == 1
    background_mask = labels == 0
    plot_metadata: Dict[str, Dict[str, object]] = {}

    for observable in observable_columns:
        if observable not in candidates.columns:
            values = np.full(len(candidates), np.nan, dtype=float)
        else:
            values = pd.to_numeric(candidates[observable], errors="coerce").to_numpy(dtype=float)

        signal = values[signal_mask & np.isfinite(values)]
        background = values[background_mask & np.isfinite(values)]
        edges = observable_bin_edges(signal, background, n_bins=n_bins)

        fig, ax = plt.subplots(figsize=(7, 5))
        if len(signal):
            ax.hist(
                signal,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.8,
                color="#1f77b4",
                label=f"Neutron captures (n={len(signal):,})",
            )
        if len(background):
            ax.hist(
                background,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.8,
                color="#d62728",
                label=f"Accidental coincidences (n={len(background):,})",
            )
        if not len(signal) and not len(background):
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlabel(observable)
        ax.set_ylabel("Probability density")
        ax.set_title(f"{observable}: truth-class distributions")
        ax.grid(True, alpha=0.25)
        if len(signal) or len(background):
            ax.legend()
        fig.tight_layout()

        plot_path = outdir / f"{observable}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

        plot_metadata[observable] = {
            "path": str(plot_path),
            "n_neutron_captures_finite": int(len(signal)),
            "n_accidental_coincidences_finite": int(len(background)),
            "n_bins": int(len(edges) - 1),
            "x_min": float(edges[0]),
            "x_max": float(edges[-1]),
        }

    summary = {
        "truth_definition": {
            "neutron_captures": "label == 1",
            "accidental_coincidences": "label == 0",
        },
        "density_normalized_per_component": True,
        "requested_bins": int(n_bins),
        "n_plots": int(len(observable_columns)),
        "plots": plot_metadata,
    }
    (outdir / "observable_plot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary
