"""Performance metrics and plots for trained or applied BDT models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def candidate_cut_table(
    scores: np.ndarray,
    labels: np.ndarray,
    cuts: Sequence[float],
    sample_name: str,
) -> pd.DataFrame:
    """Evaluate candidate-level efficiency and purity at each BDT cut."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rows = []

    for cut in cuts:
        passing = scores >= cut
        signal = labels == 1
        background = labels == 0
        rows.append(
            {
                "bdt_cut": float(cut),
                "signal_efficiency_candidate_level": (
                    float(np.mean(passing[signal])) if np.any(signal) else np.nan
                ),
                "background_acceptance_candidate_level": (
                    float(np.mean(passing[background]))
                    if np.any(background)
                    else np.nan
                ),
                f"purity_after_cut_in_{sample_name}_candidates": (
                    float(np.mean(labels[passing] == 1)) if np.any(passing) else np.nan
                ),
                f"n_{sample_name}_passing": int(np.sum(passing)),
            }
        )

    return pd.DataFrame(rows)


def save_roc_plot(
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    curve_name: str,
) -> float:
    """Save signal efficiency versus background acceptance and return its AUC."""
    from sklearn.metrics import auc, roc_curve

    import matplotlib.pyplot as plt

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    roc_auc = float(auc(false_positive_rate, true_positive_rate))
    plt.figure(figsize=(6, 5))
    plt.plot(
        true_positive_rate,
        false_positive_rate,
        label=f"{curve_name} AUC={roc_auc:.4f}",
    )
    plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background acceptance")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return roc_auc


def evaluate_training(
    args: argparse.Namespace,
    outdir: Path,
    model,
    use: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    split_strategy: str,
    n_signal: int,
    n_background: int,
    feature_columns: Sequence[str],
) -> None:
    """
    Evaluate the model and write all user-facing training artifacts.

    Keeping this separate from model fitting makes it easier to compare future
    classifiers while preserving the same candidate-level metrics, BDT cut
    tables, and diagnostic plots.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    import matplotlib.pyplot as plt

    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "n_candidates_finite": int(len(use)),
        "n_signal": int(n_signal),
        "n_background": int(n_background),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "split_strategy": split_strategy,
        "n_event_groups": (
            int(use["event_number"].nunique())
            if "event_number" in use.columns
            else int(len(use))
        ),
        "auc_train": float(roc_auc_score(y_train, p_train)),
        "auc_val": float(roc_auc_score(y_val, p_val)),
        "auc_test": float(roc_auc_score(y_test, p_test)),
        "average_precision_test": float(average_precision_score(y_test, p_test)),
        "feature_columns": list(feature_columns),
        "xgboost_params": model.get_params(),
    }

    cut_df = candidate_cut_table(
        scores=p_test,
        labels=y_test,
        cuts=args.bdt_cuts,
        sample_name="test",
    )

    split = np.full(len(use), "unused", dtype=object)
    split[idx_train] = "train"
    split[idx_val] = "validation"
    split[idx_test] = "test"
    training_table = use.assign(split=split)
    try:
        training_table.to_parquet(outdir / "training_table_finite.parquet", index=False)
    except Exception:
        training_table.to_csv(outdir / "training_table_finite.csv", index=False)

    cut_df.to_csv(outdir / "bdt_cut_table.csv", index=False)
    (outdir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str))

    # BDT score train/test distribution. The log scale makes background leakage
    # in the high-score tail visible even when the classes are well separated.
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, 51)
    plt.hist(p_train[y_train == 1], bins=bins, density=True, histtype="step", label="signal train")
    plt.hist(p_train[y_train == 0], bins=bins, density=True, histtype="step", label="background train")
    plt.hist(p_test[y_test == 1], bins=bins, density=True, histtype="step", label="signal test")
    plt.hist(p_test[y_test == 0], bins=bins, density=True, histtype="step", label="background test")
    plt.yscale("log")
    plt.xlabel("BDT discriminant")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bdt_score_train_test.png", dpi=160)
    plt.close()

    save_roc_plot(
        labels=y_test,
        scores=p_test,
        output_path=outdir / "roc_signal_eff_vs_bkg_acceptance.png",
        curve_name="test",
    )

    # Feature importance is model-specific; failures here should not hide a
    # successful training run.
    try:
        imp = model.feature_importances_
        order = np.argsort(imp)
        plt.figure(figsize=(7, max(4, 0.28 * len(feature_columns))))
        plt.barh(np.array(feature_columns)[order], imp[order])
        plt.xlabel("XGBoost feature importance")
        plt.tight_layout()
        plt.savefig(outdir / "feature_importance.png", dpi=160)
        plt.close()
    except Exception:
        pass

    print("\nTraining and evaluation complete")
    headline_metrics = {
        key: metrics[key]
        for key in ("auc_train", "auc_val", "auc_test", "average_precision_test")
    }
    print(json.dumps(headline_metrics, indent=2))
    print("\nCut table:")
    print(cut_df.to_string(index=False))
    print(f"\nSaved training evaluation artifacts in: {outdir}")


def evaluate_predictions(
    scored_candidates: pd.DataFrame,
    outdir: Path,
    bdt_cuts: Sequence[float],
) -> dict:
    """Evaluate newly scored candidates when binary truth labels are available."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    import matplotlib.pyplot as plt

    required = ["label", "bdt_score"]
    missing = [column for column in required if column not in scored_candidates.columns]
    if missing:
        raise ValueError(
            "Prediction evaluation requires these columns: " + ", ".join(missing)
        )

    labels = pd.to_numeric(scored_candidates["label"], errors="coerce")
    scores = pd.to_numeric(scored_candidates["bdt_score"], errors="coerce")
    finite = np.isfinite(labels) & np.isfinite(scores)
    labels = labels.loc[finite].to_numpy(dtype=float)
    scores = scores.loc[finite].to_numpy(dtype=float)

    if len(labels) == 0:
        raise ValueError("No candidates have finite labels and BDT scores")
    if not np.all(np.isin(labels, [0.0, 1.0])):
        raise ValueError("Prediction evaluation requires binary labels 0 and 1")

    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("Prediction evaluation requires both signal and background labels")

    outdir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_candidates_scored": int(len(scored_candidates)),
        "n_candidates_evaluated": int(len(labels)),
        "n_signal": int(np.sum(labels == 1)),
        "n_background": int(np.sum(labels == 0)),
        "auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }
    cut_df = candidate_cut_table(
        scores=scores,
        labels=labels,
        cuts=bdt_cuts,
        sample_name="prediction",
    )

    (outdir / "prediction_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )
    cut_df.to_csv(outdir / "prediction_bdt_cut_table.csv", index=False)

    bins = np.linspace(0, 1, 51)
    plt.figure(figsize=(7, 5))
    plt.hist(
        scores[labels == 1],
        bins=bins,
        density=True,
        histtype="step",
        label="signal",
    )
    plt.hist(
        scores[labels == 0],
        bins=bins,
        density=True,
        histtype="step",
        label="background",
    )
    plt.yscale("log")
    plt.xlabel("BDT discriminant")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "prediction_bdt_score.png", dpi=160)
    plt.close()

    save_roc_plot(
        labels=labels,
        scores=scores,
        output_path=outdir / "prediction_roc_signal_eff_vs_bkg_acceptance.png",
        curve_name="prediction",
    )

    print("\nPrediction evaluation complete")
    print(json.dumps(metrics, indent=2))
    print(f"Saved prediction evaluation in: {outdir}")
    return metrics
