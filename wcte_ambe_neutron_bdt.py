#!/usr/bin/env python3
"""Command-line orchestrator for the WCTE AmBe neutron-tagging BDT pipeline.

The implementation is split by analysis stage:
  - extraction.candidate_preselection: prompt tagging, delayed-candidate finding,
    geometry lookup, observable calculation, and truth-label production
  - bdt_model.training: train/validation/test split and XGBoost fitting
  - bdt_model.evaluation: training and labeled-prediction metrics and plots

Important convention:
  - times are ns
  - one ROOT entry is one 270 us DAQ/readout window, not a trigger
  - BDT rows are delayed candidates found after tagged prompt peaks
  - observables use WCTE readout/geometry only; TTrueInfo is used only for labels

Example:
  python wcte_ambe_neutron_bdt.py extract \
      --root data/wcte_ambe_mc_plus_clean_bkg_pe.root \
      --outdir ntag_out \
      --prompt-vertex-cm 0 30.5 0 \
      --geometry-file data/geofile_NuPRISMBeamTest_16cShort_mPMT.txt

  python wcte_ambe_neutron_bdt.py train \
      --features ntag_out/candidates.parquet \
      --outdir ntag_out/model

  python wcte_ambe_neutron_bdt.py all \
      --root data/wcte_ambe_mc_plus_clean_bkg_pe.root \
      --outdir ntag_out \
      --prompt-vertex-cm 0 30.5 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bdt_model.training import add_train_args, train_and_evaluate
from extraction.candidate_preselection import add_extract_args, extract_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="WCTE AmBe neutron-tagging BDT pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="prompt tagging, delayed candidates, and observables")
    add_extract_args(p_ext)

    p_train = sub.add_parser("train", help="BDT training and evaluation")
    add_train_args(p_train)

    p_all = sub.add_parser("all", help="run extract then train")
    add_extract_args(p_all)
    add_train_args(p_all, include_features=False, include_outdir=False)

    args = parser.parse_args()

    if args.cmd == "extract":
        extract_candidates(args)
    elif args.cmd == "train":
        train_and_evaluate(args)
    elif args.cmd == "all":
        features = extract_candidates(args)
        args.features = str(features)
        args.outdir = str(Path(args.outdir) / "model")
        train_and_evaluate(args)
    else:
        raise RuntimeError(args.cmd)


if __name__ == "__main__":
    main()
