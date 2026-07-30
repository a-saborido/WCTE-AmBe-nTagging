# nTag Extraction Implementation Guide

This document explains what happens when running:

```bash
python wcte_ambe_neutron_bdt.py extract \
  --root data/wcte_ambe_mc_plus_clean_bkg_pe.root \
  --outdir ntag_bdt_out \
  --geometry-file data/geofile_NuPRISMBeamTest_16cShort_mPMT.txt \
  --prompt-vertex-cm 0 30.5 0
```

The extraction stage converts readout windows into delayed neutron-capture
candidate rows. Each output row is one capture candidate, not one ROOT entry.
The row contains BDT observables, prompt/candidate bookkeeping, truth-label
diagnostics, and the final `label`.

Important conventions:

- Times are in ns.
- Positions are in cm.
- One input ROOT entry is one 270 us readout window.
- The input readout window is not treated as a trigger.
- Trigger branches are ignored by the extraction code.
- `TTrueInfo` is never used to choose hits, find prompts, fit vertices, or compute observables.
- `TTrueInfo` is used only after observables are computed, to produce labels and diagnostic counters.

## File Map

The extraction command is orchestrated by:

- `../wcte_ambe_neutron_bdt.py`: command-line entry point.
- `bonsai.py`: hk-BONSAI initialization, geometry validation, and candidate fits.
- `candidate_preselection.py`: extraction driver, delayed-candidate selection, labels, counters, output files.
- `root_io.py`: ROOT/awkward conversion and continuous-window assembly.
- `prompt_tagging.py`: prompt scintillation peak finder.
- `preselection_utils.py`: readout-level cleaning masks used before observables.
- `geometry.py`: geofile loading, PMT `(slot, pos)` lookup, simple wall-distance proxy.
- `observables/vertex.py`: Nn candidate finding and local vertex refit.
- `observables/topology.py`: Cherenkov/topology observables.
- `observables/noise.py`: charge, clustering, and local time-activity observables.
- `plotting.py`: post-extraction truth-split observable probability densities.
- `config.py`: default tunable values and extracted observable groups.

## 1. CLI Dispatch

The `extract` subcommand is defined in `../wcte_ambe_neutron_bdt.py`.

The relevant functions are:

- `main()` in `../wcte_ambe_neutron_bdt.py`
- `add_extract_args()` in `candidate_preselection.py`
- `extract_candidates()` in `candidate_preselection.py`

`main()` parses the command line. For `extract`, it calls:

```python
extract_candidates(args)
```

All extraction options are attached in `add_extract_args()`. Defaults come from
`config.py`, so timing windows, double-grid multilateration settings, angular
cuts, Nn/Nmax200 cuts, and truth-label thresholds are all in one place.

The command above explicitly sets:

- `--root`: input ROOT file.
- `--outdir`: output directory.
- `--geometry-file`: geofile used to map readout `(slot, pos)` to PMT coordinates.
- `--prompt-vertex-cm 0 30.5 0`: fixed AmBe prompt/source position used as the center of the delayed vertex scan.

## 2. Extraction Startup

The main extraction driver is `extract_candidates()` in `candidate_preselection.py`.

It first prepares analysis-wide objects:

- `prompt_vertex_cm`: NumPy array from `--prompt-vertex-cm`.
- double-grid multilateration settings from `config.py`, used by `refit_vertex_by_multilateration_grid()`.
- `geometry`: `PmtGeometry.from_geofile()` from `geometry.py`.
- `wall`: `WallEstimator` built from the PMT geometry, unless `--wall-json` is supplied.
- `bonsai_fitter`: one process-wide `BonsaiVertexFitter`, initialized from the
  WCSim ROOT geometry carrier and checked against the text geofile.

The local Nn multilateration and BONSAI are independent vertex reconstructions.
The double grid is centered on the fixed prompt vertex, fits only the original
Nn burst, and then the pipeline recomputes the final Nn window at that fitted
vertex. BONSAI uses measured PMT times, charges, and one-based WCSim tube IDs in
a configurable 1.3 us corrected-time neighborhood around the candidate.

## 3. Geometry Loading

Geometry is loaded in `PmtGeometry.from_geofile()` in `geometry.py`.

The geofile columns used are:

- Column 1: WCTE mPMT slot.
- Column 2: PMT position inside the mPMT.
- Columns 3 to 5: PMT `x, y, z`.
- Columns 6 to 8: PMT direction.

The key convention is the PMT-position shift:

```python
pos0 = data[:, 2].astype(np.int64) - 1
```

The geofile stores PMT position as `1..19`, while the readout branch
`hit_pmt_position_ids` stores it as `0..18`. The geometry object stores the
geofile position in readout convention, so later lookup is direct:

```text
(hit_mpmt_slot_ids, hit_pmt_position_ids) -> PMT position and direction
```

No conversion through `tube_id`, `mPMTid`, or any DAQ notebook mapping is used
inside this project.

## 4. ROOT Trees And Branches

`extract_candidates()` opens the input file with `uproot` and reads:

- `WCTEReadoutWindows`
- `TTrueInfo`

The required readout branches are:

- `hit_pmt_calibrated_times`, unless `--time-branch` changes this.
- `hit_pmt_charges`, unless `--charge-branch` changes this.
- `hit_mpmt_slot_ids`
- `hit_pmt_position_ids`

Optional readout branches are carried if present:

- `event_number`
- `hit_mpmt_card_ids`
- `hit_pmt_channel_ids`

Optional truth branches are carried if present:

- `hit_from_capture`
- `hit_from_prompt`
- `is_background`
- `relative_capture_t`
- `capture_t`
- `prompt_time`
- `source_event_idx`

The optional truth arrays are carried through the same hit order as the readout
hits. They are not used for hit selection, geometry, timing correction, vertex
fitting, topology, or noise observables.

## 5. Chunking And Following Windows

The extraction loops over ROOT entries in chunks:

```python
for start in range(0, n_entries, args.chunk_size):
```

The default chunk size is defined by `--chunk-size` and defaults to `1000`.

For each chunk, the code also reads a small number of following windows. This is
needed because a prompt near the end of one 270 us window can have a capture
candidate in the next readout window.

The number of following windows is computed by `following_window_count()` in
`root_io.py`.

With the default values:

```text
window_duration_ns = 270000
window_gap_ns      = 7750
window_period_ns   = 277750
capture_search_ns  = 300000
```

The code reads up to two following windows:

```text
floor((270000 + 300000) / 277750) = 2
```

The 7.75 us gap is represented only as empty time. No fake hits are inserted in
the gap.

## 6. One-Window Payload Construction

Each ROOT entry is converted by `build_window_payload()` in `root_io.py`.

For one input window, this function builds:

- `time`: calibrated PMT hit times.
- `charge`: PMT hit charges.
- `slot`: readout mPMT slot ids.
- `pos`: readout PMT positions inside each mPMT.
- `hit_index`: original hit index inside this ROOT entry.
- `pmt_pos_cm`: PMT positions from the geofile.
- `pmt_dir`: PMT directions from the geofile.
- optional `card` and `channel` fields.
- matching truth arrays in `t_evt`.

`hit_index` is important for de-duplication. A physical hit can appear in
several anchor-window searches because following windows are reused. The pair:

```text
(source ROOT entry, hit index inside that entry)
```

is a stable identity for a hit across those repeated views.

## 7. Continuous-Time Window Assembly

For each anchor readout window, the code calls `combine_window_payloads()` in
`root_io.py`.

This function concatenates:

- the anchor window,
- the first following window,
- the second following window when needed.

Hits from following windows are shifted by:

```python
virtual_offset_ns = window_offset * window_period_ns
```

The combined payload also stores:

- `is_anchor_window`: true only for hits from the anchor entry.
- `source_window_offset`: `0`, `1`, `2`, etc.
- `source_entry`: original ROOT entry number.
- `source_hit_index`: original hit index inside that ROOT entry.
- `source_event_number`: original `event_number` if available.

Prompt tagging only uses anchor-window hits. Delayed capture searches can use
the combined continuous-time array.

## 8. Truth Counters For The Chunk

Before candidate extraction, `truth_counts_for_chunk()` in
`candidate_preselection.py` computes direct truth diagnostics for the anchor
entries in the chunk.

It fills:

- `true_prompts_total`
- `true_prompts_trms_gt_100_ns`
- `true_neutron_captures_total`
- `windows_with_true_capture_hits`

These values are diagnostics printed in `extraction_summary.json` and the log.
They do not affect prompt tagging or capture-candidate selection.

## 9. Candidate Extraction For One Anchor Window

For each anchor window, `extract_candidates()` calls:

```python
event_to_candidates(...)
```

This is the core extraction function in `candidate_preselection.py`.

Despite the name `event_to_candidates`, the input is one 270 us readout window
plus any following windows needed for the delayed search. The word `event` here
means ROOT/readout entry, not a trigger.

Inside `event_to_candidates()`, the code first creates basic arrays:

- `time`
- `q`
- `pos`
- `pmt_dir`
- truth flags copied from `TTrueInfo`

Then it defines:

```python
finite = np.isfinite(time) & np.isfinite(q) & geom_ok
anchor_mask = is_anchor_window
```

Only finite hits with valid geometry can be used for prompt tagging and
candidate selection. Hits with missing geometry are counted in sanity counters.

## 10. Prompt Tagging

Prompt candidates are found by `find_prompt_candidates()` in `prompt_tagging.py`.

Only anchor-window hits are passed to this function:

```python
prompt_search_idx = np.flatnonzero(finite & anchor_mask)
```

The prompt definition is readout-only:

- A sliding `prompt_window_ns` window is built in raw readout time.
- The hit count must be between `prompt_min_hits` and `prompt_max_hits`.
- Each hit time is corrected for photon travel from the fixed source position.
- The corrected-time RMS must be between `prompt_min_trms_ns` and `prompt_max_trms_ns`.
- The corrected `tmean` must be between `prompt_min_tmean_ns` and `prompt_max_tmean_ns`.
- Overlapping prompt windows are de-duplicated using `prompt_isolation_ns`.

For prompt-window hits with corrected times `tcorr`, `tmean` is defined as:

```text
tmean = mean(tcorr - min(tcorr))
```

Thus, the first hit means the earliest hit in TOF-corrected time. This
calculation uses PMT geometry and the fixed position supplied through
`--prompt-vertex-cm`; it does not use truth information.

With current defaults from `config.py`, the prompt cuts are:

- `DEFAULT_PROMPT_WINDOW_NS = 1000`
- `DEFAULT_PROMPT_MIN_HITS = 80`
- `DEFAULT_PROMPT_MAX_HITS = 300`
- `DEFAULT_PROMPT_MIN_TRMS_NS = 200`
- `DEFAULT_PROMPT_MAX_TRMS_NS = 500`
- `DEFAULT_PROMPT_MIN_TMEAN_NS = 200`
- `DEFAULT_PROMPT_MAX_TMEAN_NS = 400`
- `DEFAULT_PROMPT_ISOLATION_NS = 200`

The de-duplication uses overlap between prompt windows, not just the mean prompt
time. This avoids one wide scintillation peak being split into several prompt
candidates.

`prompt_time_ns` remains the mean raw readout time of the selected prompt hits.
It is deliberately not TOF-corrected because it anchors the subsequent delayed
search on the continuous raw readout-time axis.

If no prompt candidate is found, the anchor window produces no delayed
candidate rows.

### Prompt Truth-Match Diagnostic

After prompt selection is complete, the code maps each prompt candidate's
selected hit indices back to the full readout hit array. A prompt candidate is
reported as truth-matched when at least one of those selected hits has:

```text
hit_from_prompt != 0
```

This truth check is diagnostic only. It occurs after prompt tagging and cannot
change whether a prompt candidate is selected or whether its delayed search is
performed.

The summary reports:

- `prompt_candidates_truth_checked`: selected prompt candidates for which the
  `hit_from_prompt` branch was available.
- `prompt_candidates_truth_matched`: checked candidates containing at least one
  truth-tagged prompt hit.
- `prompt_candidates_truth_unmatched`: checked candidates containing no
  truth-tagged prompt hits.

## 11. Delayed Search Region For Each Prompt

For each prompt candidate, the code builds a delayed-search mask:

```python
search = finite.copy()
search &= time >= prompt.time_ns + args.capture_start_after_prompt_ns
search &= time <= prompt.time_ns + args.capture_search_window_ns
```

With current defaults:

- The capture search starts 1.5 us after the prompt mean time.
- The capture search extends to 300 us after the prompt mean time.

Because following windows were shifted onto a continuous axis, the search can
cross the 270 us window boundary and the 7.75 us gap naturally.

## 12. Local Nmax200 Burst Cleaning

Before Nn candidate finding, the code applies a local burst veto using helpers
from `preselection_utils.py`.

First, `max_count_in_window()` computes the diagnostic `nmax200` value:

```python
nmax200 = max_count_in_window(time[search_idx0], args.nmax200_window_ns)
```

Then `dense_time_window_keep_mask()` removes only hits inside over-threshold
local dense windows.

This is important: the code does not reject the full 300 us delayed search when
there is a dense 200 ns burst. It removes the burst hits and keeps the rest of
the delayed-search region.

The relevant counters are:

- `delayed_searches_with_nmax200_veto`
- `hits_removed_by_nmax200_veto`
- `delayed_searches_emptied_by_nmax200_veto`

## 13. Same-PMT Continuous-Noise Cleaning

After the local burst veto, the code applies same-PMT cleaning with:

```python
continuous_noise_keep_mask(...)
```

The PMT identity is made by `make_pmt_key()` in `preselection_utils.py`.

The preferred PMT key is:

```text
slot * 1000 + pos
```

If a PMT has repeated hits closer than `continuous_noise_ns`, both hits in that
close pair are removed. This suppresses afterpulse-like or unstable-channel
activity that can fake compact time clusters.

This cleaning is prompt-specific. Hits in another prompt's delayed-search
region should not veto the current prompt's candidates.

## 14. Time-Of-Flight Correction

After cleaning, the delayed-search hits are time-of-flight corrected relative
to the fixed prompt vertex:

```python
c_water = C_LIGHT_CM_PER_NS / args.water_refractive_index
tcorr_s = time_s - distance(PMT, prompt_vertex_cm) / c_water
```

The water refractive index default is in `config.py`.

This corrected time array is used to find compact delayed clusters with Nn.

## 15. Greedy Nn Candidate Finding

Delayed candidates are first found by `greedy_nn_candidates()` in
`observables/vertex.py`.

`Nn` is the number of hits in the best configured-width corrected-time window,
where the width is `nn_window_ns`.
With the default:

```text
DEFAULT_NN_WINDOW_NS = 10
DEFAULT_NN_CUT = 5
```

The condition is:

```text
Nn > 5
```

So the raw candidate must have at least 6 hits in the best configured-width
corrected-time window. With the default settings, that window is 10 ns.

The greedy algorithm works like this:

- Find the densest `nn_window_ns` corrected-time window.
- Keep it if `Nn > nn_cut`.
- Remove those selected hits from the remaining pool.
- Repeat until no more candidates pass or `max_candidates_per_prompt` is reached.

The default maximum is:

```text
DEFAULT_MAX_CANDIDATES_PER_PROMPT = 50
```

This protects runtime and prevents a noisy delayed region from producing an
unbounded number of raw candidates.

## 16. Fit Context Around The Candidate

Before the vertex scan, the code selects a local timing context around the raw
candidate center:

```python
context = abs(tcorr_s - cand_center) <= args.fit_context_ns
```

The default is:

```text
DEFAULT_FIT_CONTEXT_NS = 150
```

This keeps the local vertex scan focused on hits near the selected delayed
candidate. Without this, unrelated hits elsewhere in the 300 us delayed region
could pull the refit or slow it down.

There is a fallback. If too few hits survive the context cut, the code uses all
delayed-search hits so the refit remains defined:

```python
if np.sum(context) < max(args.nn_cut + 1, 6):
    context = np.ones_like(tcorr_s, dtype=bool)
```

## 17. Local Vertex Refit

The vertex fit is performed by `refit_vertex_by_multilateration_grid()` in
`observables/vertex.py`.

The fit uses the same Nn seed burst selected at the prompt/source vertex:

- Run a coarse cubic grid centered on the prompt vertex.
- Refine with a fine cubic grid around the best coarse point.
- Score trial vertices with the median-centered corrected-time RMS of the fixed
  Nn seed hits, with the optional fine-stage `dt` cut.
- At the fitted vertex, find the new best Nn window using `best_window_indices()`.
- Use that recomputed final window for all downstream observables and hit-set
  de-duplication.

If a `WallEstimator` is available, vertices outside the simple detector proxy
are skipped.

The refit returns:

- `xfit`: fitted delayed-candidate vertex.
- `nn_refit`: Nn after the refit.
- `best_context_loc`: hits selected by the best refit window.
- `trmsp`: tRMS after the refit.

The final candidate hit set is:

```python
final_global = search_idx[context_idx[best_context_loc]]
```

These final hits are the ones used for topology, noise, truth labeling, and the
output candidate row.

## 18. Candidate De-Duplication

The function `candidate_hitset_key()` in `candidate_preselection.py` builds a
stable identity for a candidate hit set.

The preferred key is:

```text
sorted((source_entry, source_hit_index) for each final hit)
```

The first de-duplication happens before the expensive vertex refit. The raw Nn
hit set returned by `greedy_nn_candidates()` is keyed and compared against
`seen_raw_candidate_hitsets`. If the same raw cluster has already been processed,
the candidate is skipped and `raw_capture_candidates_skipped_by_hitset` is
incremented.

The second de-duplication happens after the vertex refit. The final fitted hit
set is keyed and compared against `seen_candidate_hitsets`. This remains as a
safety net for cases where different raw Nn seeds collapse onto the same final
refit candidate.

Together these catch repeated candidates caused by:

- The same delayed cluster being found under multiple prompt candidates.
- Different greedy Nn seeds refitting to the same final hit set.
- Neighboring anchor windows seeing the same following-window hits.

Both tracking sets are created once in `extract_candidates()` and passed into
every `event_to_candidates()` call, so de-duplication is global over the
extraction run.

When a duplicate is found, the candidate row is not written. Repeated raw
clusters are counted by `raw_capture_candidates_skipped_by_hitset`; repeated
final fitted clusters are counted by `capture_candidates_deduplicated_by_hitset`.

## 19. Vertex-Determination Observables

Some vertex observables are computed directly in `event_to_candidates()`, using
helpers from `observables/vertex.py`.

The vertex-determination observable columns are defined in `config.py`:

- `Nn`: initial raw Nn before the vertex refit.
- `trms`: initial raw tRMS before the vertex refit.
- `fpdist`: distance between the fixed prompt vertex and fitted delayed vertex.
- `delta_trms`: initial tRMS minus refit tRMS.
- `delta_Nn`: refit Nn minus initial Nn.
- `fwall`: distance from fitted vertex to the current wall proxy.
- `trms3`: smallest tRMS among any 3 final corrected hit times.
- `trms6`: smallest tRMS among any 6 final corrected hit times.
- `Bpdist`: Euclidean separation between the BONSAI vertex and `xfit`.
- `Bwall`: shortest distance from the BONSAI vertex to the wall proxy.

`fwall` is computed by `WallEstimator.distance_to_wall()` in `geometry.py`.
This is currently a simple cylindrical proxy inferred from PMT coordinates, not
a full detector-solid calculation.

BONSAI is called only after final hit-set de-duplication. The adapter shifts each
candidate's time origin to 200 ns, maps readout `(slot, pos)` pairs to WCSim tube
IDs, and swaps BONSAI's internal WCTE `(x, z, y)` ordering back to `(x, y, z)`.
If the fit does not converge, `Bpdist`, `Bwall`, and the BONSAI vertex columns
are `NaN`, while `bonsai_fit_success` is zero.

## 20. Cherenkov Event Topology Observables

Topology features are computed by:

```python
calculate_cherenkov_topology_observables(...)
```

in `observables/topology.py`.

The topology feature columns are:

- `theta_mean`: mean angular spread of final-hit PMT directions around the mean axis.
- `theta_rms`: RMS of that angular spread.
- `phi_rms`: azimuthal non-uniformity around the mean hit axis.
- `Nlowtheta`: hits close to the mean axis.
- `Nback`: hits far behind the mean axis.
- `Nlow`: hits with low geometric light-collection proxy.

These use PMT positions, PMT directions, the fitted vertex, and configurable
angular/light-proxy thresholds from `config.py`.

## 21. Noise Characterization Observables

Noise features are computed by:

```python
calculate_noise_observables(...)
```

in `observables/noise.py`.

The noise feature columns are:

- `Qmean`: mean charge of final candidate hits.
- `Qrms`: RMS of final-hit charges.
- `NhighQ`: number of final hits above `high_charge_pe`.
- `Nclus`: number of final hits in compact angular clusters.
- `N300`: number of delayed-search hits in a wider corrected-time window around the original candidate peak.

`N300` is computed in `event_to_candidates()` before calling the noise module,
because it uses the original delayed-search corrected-time array around the raw
candidate center.

## 22. Truth Labels

Truth label production starts only after all observables have been computed.

The relevant code is in `event_to_candidates()` after the topology and noise
observable calls.

For the final candidate hit set, the code counts:

- `n_capture_hits`
- `n_prompt_hits`
- `n_background_hits`
- `dominant_capture_hits`
- `dominant_capture_fraction`
- `dominant_capture_source_idx`
- `n_capture_sources`

The candidate is labeled signal if either condition passes:

```text
dominant_capture_hits >= min_capture_hits
and dominant_capture_fraction >= min_capture_fraction
```

or:

```text
dominant_capture_hits >= min_capture_hits_absolute
```

The default label thresholds are in `config.py`:

- `DEFAULT_MIN_CAPTURE_HITS = 3`
- `DEFAULT_MIN_CAPTURE_FRACTION = 0.50`
- `DEFAULT_MIN_CAPTURE_HITS_ABSOLUTE = 4`

These truth quantities are saved for diagnostics and training labels. They are
not BDT inputs unless deliberately added to `FEATURE_COLUMNS` in
`../bdt_model/config.py`, which they currently are not.

## 23. Output Candidate Row

Each kept delayed candidate becomes one dictionary row in `event_to_candidates()`.

Important bookkeeping columns include:

- `event_number`: anchor readout window event number.
- `candidate_id`: sequential id for kept candidates inside that anchor window.
- `prompt_id`: prompt candidate id inside that anchor window.
- `prompt_time_ns`: prompt candidate mean time.
- `prompt_window_start_ns`, `prompt_window_end_ns`: prompt sliding-window boundaries.
- `prompt_nhits`: prompt hit count.
- `prompt_trms_ns`: prompt hit-time RMS after fixed-source TOF correction.
- `prompt_tmean_ns`: mean corrected prompt-hit time relative to the earliest corrected hit.
- `candidate_tcorr_center_ns`: raw candidate corrected-time center.
- `candidate_raw_mean_time_ns`: raw mean time of final candidate hits.
- `candidate_time_from_prompt_ns`: `candidate_raw_mean_time_ns - prompt_time_ns`;
  also included in `OBSERVABLE_COLUMNS` as a timing observable.
- `candidate_first_window_offset`, `candidate_last_window_offset`: whether final hits came from anchor or following windows.
- `candidate_first_source_entry`, `candidate_last_source_entry`: original ROOT entries touched by final hits.

Important reconstruction columns include:

- `xfit_cm`, `yfit_cm`, `zfit_cm`
- `xbonsai_cm`, `ybonsai_cm`, `zbonsai_cm`
- `bonsai_fit_success`, `bonsai_n_fit`, `bonsai_n_input_hits`, `bonsai_n_selected`,
  `bonsai_n_window`, `bonsai_fit_goodness`, `bonsai_time_goodness`
- all extracted observable columns from `OBSERVABLE_COLUMNS`

Important truth-label diagnostics include:

- `label`
- `n_capture_hits`
- `dominant_capture_hits`
- `dominant_capture_fraction`
- `dominant_capture_source_idx`
- `n_prompt_hits`
- `n_background_hits`
- `capture_fraction`

## 24. Output Files

After all chunks are processed, `extract_candidates()` builds a Pandas
DataFrame from all candidate rows.

The candidate table is written to:

```text
ntag_bdt_out/candidates.parquet
```

If Parquet writing fails, it falls back to:

```text
ntag_bdt_out/candidates.csv
```

The summary file is written to:

```text
ntag_bdt_out/extraction_summary.json
```

The wall proxy is written to:

```text
ntag_bdt_out/wall_estimator.json
```

The selected hit information for each final candidate is written to:

```text
ntag_bdt_out/candidates.root
```

The ROOT file contains one `THits` entry per final candidate. Hit branches are
jagged arrays; scalar branches store the candidate label, fitted vertex,
`fpdist`, prompt time, and capture-hit counts. This layout can be used to train
point-cloud methods.

One overlaid probability-density histogram per feature column is written to:

```text
ntag_bdt_out/obs_plots/
```

This directory keeps the aggregate neutron-capture (`label == 1`) versus
accidental (`label == 0`) plots. The directory also contains
`observable_plot_summary.json` with finite component counts and histogram ranges.

## 25. Extraction Summary Counters

The counters are grouped by `grouped_counter_summary()` and printed by
`format_grouped_counter_summary()` in `candidate_preselection.py`.

Truth counters:

- `true_prompts_total`: direct truth prompt count from `TTrueInfo`.
- `true_prompts_trms_gt_100_ns`: truth prompt groups with fixed-source
  TOF-corrected hit-time RMS above 100 ns.
- `true_neutron_captures_total`: direct truth neutron-capture count.
- `windows_with_true_capture_hits`: anchor windows containing a truth capture.

Preselection counters:

- `windows_total`: anchor readout windows processed.
- `windows_with_prompt_candidates`: windows with at least one tagged prompt.
- `prompt_candidates_total`: prompt candidates after prompt de-duplication.
- `prompt_candidates_truth_checked`: prompt candidates checked using
  `hit_from_prompt`.
- `prompt_candidates_truth_matched`: checked prompt candidates containing at
  least one truth-tagged prompt hit.
- `prompt_candidates_truth_unmatched`: checked prompt candidates containing no
  truth-tagged prompt hits.
- `delayed_searches_with_nmax200_veto`: prompt-associated searches where local burst hits were removed.
- `hits_removed_by_nmax200_veto`: total hits removed by local burst cleaning.
- `delayed_searches_emptied_by_nmax200_veto`: searches left empty after burst cleaning.
- `prompt_candidates_with_capture_candidates`: prompt candidates that produced at least one raw capture candidate before final hit-set de-duplication.
- `windows_with_capture_candidates`: anchor windows with at least one final written capture candidate.
- `capture_candidates_total`: final de-duplicated candidate rows written.
- `raw_capture_candidates_skipped_by_hitset`: repeated raw Nn hit clusters skipped before vertex fitting.
- `capture_candidates_deduplicated_by_hitset`: exact repeated final hit sets removed.
- `capture_candidates_signal`: final candidates labeled signal by truth.
- `capture_candidates_background`: final candidates labeled background by truth.
- `windows_with_signal_capture_candidate`: windows with at least one signal-labeled candidate.

Sanity counters:

- `windows_failed_length_match`: readout arrays with inconsistent lengths.
- `windows_failed_truth_length_match`: truth arrays not matching readout hit counts.
- `windows_missing_hit_from_capture_branch`: windows without `hit_from_capture`.
- `windows_missing_hit_from_prompt_branch`: windows without `hit_from_prompt`.
- `windows_with_missing_geometry_hits`: windows with at least one unmatched `(slot, pos)`.
- `hits_missing_geometry`: total hits with missing PMT geometry.

## 26. How To Trace One Candidate Through The Code

A useful mental trace is:

```text
../wcte_ambe_neutron_bdt.py::main
  -> candidate_preselection.py::extract_candidates
    -> root_io.py::build_window_payload
    -> root_io.py::combine_window_payloads
    -> candidate_preselection.py::event_to_candidates
      -> prompt_tagging.py::find_prompt_candidates
      -> preselection_utils.py::dense_time_window_keep_mask
      -> preselection_utils.py::continuous_noise_keep_mask
      -> observables/vertex.py::greedy_nn_candidates
      -> candidate_preselection.py::candidate_hitset_key for raw Nn de-duplication
      -> observables/vertex.py::refit_vertex_by_multilateration_grid
      -> candidate_preselection.py::candidate_hitset_key for final hit-set de-duplication
      -> observables/topology.py::calculate_cherenkov_topology_observables
      -> observables/noise.py::calculate_noise_observables
      -> truth label assignment
    -> write candidates and summary
```

If the extraction finds too many candidates, the most informative places to
look first are:

- Prompt multiplicity and isolation cuts in `prompt_tagging.py`.
- `prompt_candidates_total` and `prompt_candidates_with_capture_candidates` in the summary.
- `raw_capture_candidates_skipped_by_hitset` in the summary.
- `capture_candidates_deduplicated_by_hitset` in the summary.
- Nn settings in `config.py`.
- `max_candidates_per_prompt` in `config.py`.
- `fit_context_ns` and the `multilateration_*` grid settings in `config.py`.
- `Nmax200` and continuous-noise counters in the summary.

If candidates are missing, first check:

- `hits_missing_geometry`
- `windows_failed_truth_length_match`
- prompt cuts in `config.py`
- `nn_cut`
- whether the input ROOT file uses the expected time and charge branches
