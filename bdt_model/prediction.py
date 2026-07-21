"""Apply a trained neutron-tagging BDT to a new candidate table."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .evaluation import evaluate_predictions


def load_candidates(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet candidate table without training dependencies."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Candidate input must be a .csv or .parquet file")


def score_candidates(
    model_path: Path,
    candidates_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Apply a saved model and write all candidates with finite model inputs."""
    # The bundle supplies both the fitted model and its original feature order.
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    features = model_bundle["feature_columns"]

    candidates = load_candidates(candidates_path)
    missing = [feature for feature in features if feature not in candidates.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    finite = (
        candidates[features]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    scored = candidates.loc[finite].copy()
    if scored.empty:
        scored["bdt_score"] = np.array([], dtype=float)
    else:
        scored["bdt_score"] = model.predict_proba(
            scored[features].to_numpy(dtype=float)
        )[:, 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        scored.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        scored.to_csv(output_path, index=False)
    else:
        raise ValueError("Prediction output must be a .csv or .parquet file")

    print(f"Saved {len(scored):,} scored candidates to: {output_path}")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a trained neutron-tagging BDT")
    parser.add_argument("--model", type=Path, required=True, help="trained .joblib model bundle")
    parser.add_argument("--candidates", type=Path, required=True, help="new candidate table")
    parser.add_argument("--output", type=Path, required=True, help="scored .csv or .parquet table")
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        help="write evaluation artifacts here when the candidates contain truth labels",
    )
    parser.add_argument(
        "--bdt-cuts",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9, 0.99],
        help="score thresholds used only for labeled prediction evaluation",
    )
    args = parser.parse_args()

    scored = score_candidates(args.model, args.candidates, args.output)
    if args.evaluation_dir:
        evaluate_predictions(scored, args.evaluation_dir, args.bdt_cuts)


if __name__ == "__main__":
    main()
