"""BDT training orchestration for WCTE AmBe neutron tagging."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS
from .evaluation import evaluate_training


def load_candidates(path: Path) -> pd.DataFrame:
    """Load candidate rows produced by the extract command."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def train_and_evaluate(args: argparse.Namespace) -> None:
    """Train the XGBoost BDT, save it, and evaluate its performance."""
    from sklearn.model_selection import train_test_split

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit("Missing xgboost. Install with: pip install xgboost") from exc

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_candidates(Path(args.features))
    if not FEATURE_COLUMNS:
        raise ValueError("bdt_model/config.py must contain at least one feature")
    if len(FEATURE_COLUMNS) != len(set(FEATURE_COLUMNS)):
        raise ValueError("bdt_model/config.py contains duplicate feature names")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "label" not in df.columns:
        raise ValueError("Input candidate table must contain a 'label' column")

    train_cols = FEATURE_COLUMNS + ["label"]
    finite_mask = df[train_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    use = df.loc[finite_mask].copy()
    y = use["label"].astype(int).to_numpy()
    X = use[FEATURE_COLUMNS].to_numpy(dtype=float)

    n_sig = int(np.sum(y == 1))
    n_bkg = int(np.sum(y == 0))
    if n_sig < 10 or n_bkg < 10:
        raise ValueError(
            "Not enough candidates after finite-feature selection: "
            f"signal={n_sig}, background={n_bkg}"
        )

    idx = np.arange(len(y))
    idx_train, idx_tmp, _, y_tmp = train_test_split(
        idx,
        y,
        test_size=0.50,
        random_state=args.seed,
        stratify=y,
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_tmp,
        y_tmp,
        test_size=0.50,
        random_state=args.seed,
        stratify=y_tmp,
    )
    split_strategy = "candidate_row_stratified"

    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    for split_name, yy in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        if len(np.unique(yy)) < 2:
            raise ValueError(f"{split_name} split contains only one class; cannot train/evaluate robustly")

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method=args.tree_method,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=args.verbose_eval)

    # Prediction needs only the fitted model and its exact input-column order.
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(FEATURE_COLUMNS),
        },
        outdir / "ntag_xgb_model.joblib",
    )

    evaluate_training(
        args=args,
        outdir=outdir,
        model=model,
        use=use,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        split_strategy=split_strategy,
        n_signal=n_sig,
        n_background=n_bkg,
        feature_columns=FEATURE_COLUMNS,
    )


def add_train_args(
    p: argparse.ArgumentParser,
    require_features: bool = True,
    include_features: bool = True,
    include_outdir: bool = True,
) -> None:
    """Attach model-training CLI options."""
    if include_features:
        p.add_argument(
            "--features",
            required=require_features,
            help="candidates.parquet or candidates.csv from extract",
        )
    if include_outdir:
        p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-jobs", type=int, default=8)

    p.add_argument("--learning-rate", type=float, default=0.025219)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--subsample", type=float, default=0.97)
    p.add_argument("--colsample-bytree", type=float, default=1.0)
    p.add_argument("--tree-method", default="auto")
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--verbose-eval", type=int, default=50)
    p.add_argument("--bdt-cuts", type=float, nargs="+", default=[0.1, 0.5, 0.9, 0.99])
