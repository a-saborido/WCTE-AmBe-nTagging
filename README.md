# WCTE AmBe Neutron-Tagging BDT

This repository builds candidate-level neutron-tagging inputs for WCTE AmBe
samples and trains, evaluates, and applies an XGBoost BDT.

## Data Model

One entry in `WCTEReadoutWindows` is a 270 microsecond DAQ/readout window.

The pipeline works in three nested units:

1. Readout window: one ROOT entry containing signal plus background hits.
2. Prompt candidate: a scintillation-like peak tagged inside the anchor window.
3. Delayed candidate: an N10-like cluster found after a tagged prompt.

For each prompt, delayed candidates are searched in the following 300
microseconds. If the search crosses into the next ROOT entry, the next window is
shifted onto a continuous time axis by `270 us + 7.75 us`. No hits are inserted
in the 7.75 us gap; the gap is simply empty data.

Because neighboring anchor windows can see the same later physical hits, exact
repeated delayed candidates are removed using the final fitted hit identity:
`(source ROOT entry, hit index inside that entry)` for each selected hit.

Truth information from `TTrueInfo` is used only after candidate observables are
computed, to produce labels and diagnostics. PMT positions and directions come
from the readout slot/position ids plus the WCSim geofile.

## Repository Layout

```text
wcte_ambe_neutron_bdt.py      CLI orchestrator for extract/train/all
data/
  wcte_ambe_mc_plus_clean_bkg_pe.root
  geofile_NuPRISMBeamTest_16cShort_mPMT.txt
extraction/
  candidate_preselection.py   Prompt tagging, delayed candidates, labels
  bonsai.py                   Safe hk-BONSAI/WCSim bridge
  config.py                   Extraction settings and observable groups
  geometry.py                 PMT geometry lookup and fwall proxy
  plotting.py                 Observable diagnostic plots
  preselection_utils.py       Readout-level cleaning helpers before observables
  prompt_tagging.py           Prompt scintillation peak finder
  root_io.py                  ROOT/window assembly helpers
  observables/
    vertex.py                 Vertex-determination observables
    topology.py               Cherenkov event-topology observables
    noise.py                  Noise-characterization observables
    common.py                 Shared numeric helpers
bdt_model/
  config.py                   BDT feature subset
  training.py                 Candidate loading, splits, XGBoost fit
  evaluation.py               Training and labeled-prediction evaluation
  prediction.py               Isolated application to new candidate tables
tests/                        Fast regression tests for both stages
```

The main tuning placeholders are centralized in `extraction/config.py`: prompt cuts,
delayed-search timing, N10/Nmax200 settings, double-grid multilateration
settings, opening-angle thresholds, and label thresholds.

## Geometry Lookup

The nTagging code uses only the `(slot, pos)` pairs stored in the ROOT readout
branches and in the geofile. It does not convert through any other detector
identifier.

The geofile rows contain:

- column 1: WCTE mPMT slot id
- column 2: PMT position inside the mPMT, one-indexed in the file
- columns 3-5: PMT `x, y, z`
- columns 6-8: PMT direction

The readout branches provide `hit_mpmt_slot_ids` and
`hit_pmt_position_ids`. The only convention shift is the PMT position index:
the geofile stores positions as `1..19`, while the readout stores them as
`0..18`. During geometry loading, the geofile position is converted once with
`pos0 = pos - 1`; after that, lookup is direct:

```text
(hit_mpmt_slot_id, hit_pmt_position_id)
  -> geofile row with matching (slot, pos0)
  -> PMT x/y/z and direction
```

## Pipeline

The model workflow separates fitting, prediction, and evaluation:

```text
Training
  -> fit and save the model with its feature order
  -> evaluate train/validation/test performance
  -> save metrics, a BDT cut table, and diagnostic plots

Prediction
  -> load the trained model and its feature order
  -> read a new candidate table
  -> calculate and save a BDT score for each valid candidate

Prediction evaluation, only when truth labels are available
  -> evaluate the scores from the new data
  -> save prediction metrics, a BDT cut table, and diagnostic plots
```

Prediction never reads the training table or the current feature configuration.
The saved feature order determines the inputs expected by that particular
model. Without truth labels, prediction can save scores but cannot measure AUC,
signal efficiency, background acceptance, or purity.

### 1. Extract Candidates And Observables

Load the ROOT/WCSim/BONSAI environment first:

```bash
source ~/setup.sh
```

```bash
python wcte_ambe_neutron_bdt.py extract \
  --root data/wcte_ambe_mc_plus_clean_bkg_pe.root \
  --outdir outputs/ntag_bdt_out \
  --geometry-file data/geofile_NuPRISMBeamTest_16cShort_mPMT.txt \
  --prompt-vertex-cm 0 30.5 0
```

Outputs:

- `candidates.parquet` or `candidates.csv`
- `candidates.root`: final candidate hit information in a `THits` layout that can
  be used to train point-cloud methods.
- `extraction_summary.json`
- `wall_estimator.json`
- `obs_plots/*.png`: original aggregate neutron-capture versus accidental
  probability-density histograms.
- `obs_plots/observable_plot_summary.json`: finite counts and histogram ranges.

Each displayed signal/background component is normalized independently. Non-finite
values are omitted only from the affected observable, so failed BONSAI fits do
not remove a candidate from unrelated plots. The default is 60 bins and can be
changed with `--observable-plot-bins`.

### 2. Train And Evaluate The BDT

```bash
python wcte_ambe_neutron_bdt.py train \
  --features outputs/ntag_bdt_out/candidates.parquet \
  --outdir outputs/ntag_bdt_out/model
```

Outputs:

- `ntag_xgb_model.joblib`
- `training_metrics.json`
- `bdt_cut_table.csv`
- `bdt_score_train_test.png`
- `roc_signal_eff_vs_bkg_acceptance.png`
- `feature_importance.png`
- `training_table_finite.parquet` or `.csv`

`ntag_xgb_model.joblib` contains only the fitted XGBoost model and its ordered
feature list. Prediction loads these together so it can reproduce the model
input exactly without training again. Metrics and BDT cut results are evaluation
artifacts stored in the separate JSON, CSV, and PNG files listed above. Because
joblib files can contain executable Python objects, only load model files from
trusted sources.

### 3. Run Both Stages

```bash
python wcte_ambe_neutron_bdt.py all \
  --root data/wcte_ambe_mc_plus_clean_bkg_pe.root \
  --outdir outputs/ntag_bdt_out \
  --geometry-file data/geofile_NuPRISMBeamTest_16cShort_mPMT.txt \
  --prompt-vertex-cm 0 30.5 0
```

The `all` command writes extraction outputs in `outputs/ntag_bdt_out/` and model
outputs in `outputs/ntag_bdt_out/model/`.

### 4. Apply A Trained Model

The prediction module takes all input and output paths explicitly:

```bash
python -m bdt_model.prediction \
  --model outputs/ntag_bdt_out/model/ntag_xgb_model.joblib \
  --candidates outputs/ntag_bdt_out_RealSample/candidates.parquet \
  --output outputs/ntag_bdt_out_RealSample/candidates_scored.parquet
```

This command is independent of the training candidate table. When the new
candidate table contains binary truth labels, prediction can also evaluate the
new scores:

```bash
python -m bdt_model.prediction \
  --model outputs/ntag_bdt_out/model/ntag_xgb_model.joblib \
  --candidates outputs/labeled_sample/candidates.parquet \
  --output outputs/labeled_sample/candidates_scored.parquet \
  --evaluation-dir outputs/labeled_sample/evaluation
```

The labeled prediction evaluation writes:

- `prediction_metrics.json`
- `prediction_bdt_cut_table.csv`
- `prediction_bdt_score.png`
- `prediction_roc_signal_eff_vs_bkg_acceptance.png`

## Observable Groups

Vertex determination:

- `N10`, `trms`: compactness of the initial time-of-flight-corrected peak.
- `fpdist`: distance between the prompt vertex and fitted delayed vertex.
- `delta_trms`: timing improvement after the local vertex scan.
- `delta_N10`: signed change `N10′ - N10`; negative values are retained.
- `fwall`: distance to the current cylindrical wall proxy.
- `trms3`, `trms6`: tight subcluster timing inside the final candidate.
- `Bpdist`: distance between the independent BONSAI and local N10 vertices.
- `Bwall`: distance from the BONSAI vertex to the wall proxy.

Cherenkov event topology:

- `theta_mean`, `theta_rms`: angular spread around the mean hit direction.
- `phi_rms`: azimuthal non-uniformity around that direction.
- `Nlowtheta`, `Nback`: forward/backward angular population counts.
- `Nlow`: count of hits with low geometric light-collection proxy.

Noise characterization:

- `Qmean`, `Qrms`, `NhighQ`: charge distribution and high-charge tails.
- `Nclus`: compact angular hit clusters, tuned to WCTE mPMT scale.
- `N300`: nearby corrected-time activity around the candidate peak.

## Training Labels

Candidate labels are produced from `TTrueInfo` after all observables have been
computed. A candidate is labeled signal if the dominant capture source among
the final selected hits satisfies either:

- at least `min_capture_hits` hits and at least `min_capture_fraction` of the
  final candidate hits, or
- at least `min_capture_hits_absolute` hits.

These defaults live in `extraction/config.py` and are exposed as CLI options. Truth
positions are not used for geometry, candidate finding, vertex fitting, or any
observable.

## How To Interpret Outputs

Start with `extraction_summary.json`. The most useful fields are:

- `truth_info.true_prompts_total`: direct truth count of prompt events in the
  anchor readout windows.
- `truth_info.true_neutron_captures_total`: direct truth count of neutron
  captures in the anchor readout windows.
- `preselection.windows_with_prompt_candidates`: windows where the prompt tagger
  found at least one scintillation-like prompt candidate.
- `preselection.prompt_candidates_truth_matched`: selected prompt candidates
  whose own hit set contains at least one `hit_from_prompt` truth-tagged hit.
  This is a truth-only diagnostic and does not affect prompt selection.
- `preselection.prompt_candidates_with_capture_candidates`: prompt candidates
  whose delayed search produced at least one raw capture candidate before
  final hit-set de-duplication.
- `preselection.capture_candidates_total`: final de-duplicated capture candidate
  rows written out.
- `preselection.raw_capture_candidates_skipped_by_hitset`: repeated raw N10 hit
  clusters skipped before the expensive vertex refit.
- `preselection.capture_candidates_deduplicated_by_hitset`: repeated final hit
  clusters removed before writing.
- `preselection.capture_candidates_signal` and
  `preselection.capture_candidates_background`: truth labels for the final
  capture candidate rows.
- `sanity_checks.hits_missing_geometry`: hits whose `(slot, pos)` did not match
  the geofile.
- `capture_search_following_windows`: how many later ROOT entries were loaded so
  a 300 us delayed search can continue across the 7.75 us gap.
- `geometry` and `wall_estimator`: sanity checks for PMT coordinate scale and the
  current fwall proxy.
- `bonsai`: the parameter file, geometry consistency check, and fit-window
  configuration.

Then inspect `candidates.parquet` or `candidates.csv`. Each row is one delayed
candidate, not one readout window. Important columns include:

- `event_number`, `prompt_id`, `candidate_id`: where the candidate came from.
- `prompt_time_ns`, `prompt_nhits`, `prompt_trms_ns`, `prompt_tmean_ns`:
  prompt-tagging metadata. The timestamp is raw readout time; `tRMS` and
  `tmean` use fixed-source TOF-corrected hit times.
- `candidate_time_from_prompt_ns`: raw mean time of the final delayed-candidate
  hits minus `prompt_time_ns`.
- `candidate_first_window_offset`, `candidate_last_window_offset`: whether the
  candidate used hits from the anchor window only (`0`) or later windows (`1`,
  `2`, ...).
- `candidate_first_source_entry`, `candidate_last_source_entry`: physical ROOT
  entries touched by the final candidate hits.
- `label`: truth label used for training.
- `xbonsai_cm`, `ybonsai_cm`, `zbonsai_cm`, `bonsai_fit_success`, and BONSAI
  goodness/hit counters: reconstruction diagnostics.
- `n_capture_hits`, `n_background_hits`, `dominant_capture_fraction`: label
  diagnostics, not BDT observables.
- the observable columns listed in `extraction/config.py`: all values available
  for model training.

The actual BDT inputs are the subset listed in `bdt_model/config.py`. Training
stores this list in the model file, and prediction reads it from there.

For training, read `training_metrics.json` and `bdt_cut_table.csv` first. The
default split strategy is `candidate_row_stratified`: finite candidates are split
row by row into train, validation, and test samples while preserving the
signal/background balance. The plots are diagnostics:

- `bdt_score_train_test.png`: compares train/test score distributions. Large
  train-test differences can indicate overtraining or split leakage.
- `roc_signal_eff_vs_bkg_acceptance.png`: signal efficiency vs background
  acceptance on the test split.
- `feature_importance.png`: XGBoost feature usage. Treat this as model-specific,
  not as a physics ranking by itself.

## Notes For Development

- Add future observable code under `extraction/observables/` and expose it
  through the observable groups in `extraction/config.py`.
- Select the subset used for training in `bdt_model/config.py`; changing that
  list does not require another extraction.
- The small `__init__.py` files make imports reliable for the CLI and tests;
  they contain no library setup or hidden behavior.
- The current `fwall` is a simple cylinder inferred from PMT positions. It is
  useful as a WCTE-sized placeholder but can be replaced by a true detector
  distance-to-wall calculation later.
- `wcsim_dummy.root` is the WCSim ROOT geometry object needed by BONSAI.
  Startup verifies its 1843 PMT positions against the production text geofile.
- If your shell environment has multiple C++ runtimes, use the same environment
  that successfully imports `pandas`, `uproot`, `awkward`, `sklearn`, `xgboost`,
  and `matplotlib`.
