# Observable Definitions

This folder contains the candidate-level observables used by the WCTE AmBe
neutron-tagging BDT. The code is split by analysis intent:

- `vertex.py`: vertex determination and timing-compactness observables.
- `topology.py`: Cherenkov event topology observables.
- `noise.py`: noise, charge-tail, and local-clustering observables.
- `common.py`: shared numerical helpers.

The observable groups written by extraction are defined in
`extraction/config.py`. The BDT can use any subset selected independently in
`bdt_model/config.py`. The extracted list includes the BONSAI vertex observables
`Bpdist` and `Bwall`; the BONSAI energy proxy `bse` is not implemented.

## Candidate Context

One row in the training table is one delayed-capture candidate, not one ROOT
entry. A ROOT entry is a 270 us WCTE readout window containing whatever signal
and background hits fell in that acquisition window.

The candidate-building sequence is:

1. Find prompt-like scintillation peaks in the anchor readout window.
2. For each prompt, search delayed hits in the following 300 us, allowing the
   search to continue into following readout windows after the 7.75 us gap.
3. Locally veto dense delayed-time bursts using the `Nmax200` rule.
4. Remove close repeated same-PMT hits using the continuous-noise cleaning.
5. Find local delayed clusters using `Nn`.
6. Refit a local delayed vertex with a small WCTE-sized scan around the prompt
   vertex.
7. Fit an independent BONSAI vertex from the cleaned 1.3 us candidate
   neighborhood.
8. Compute observables from the two vertices and final candidate hits.
9. Only after all observables are computed, use truth flags to assign labels.

Truth information is never used to compute observables. It is only used for
training labels and diagnostics.

## Notation

For one candidate:

- `t_i` is the raw or calibrated hit time in ns.
- `q_i` is the hit charge in PE-like units.
- `p_i` is the PMT position in cm from the WCSim geofile.
- `d_i` is the PMT direction vector from the geofile.
- `v_prompt` is the configured prompt/source vertex.
- `xfit` is the fitted delayed-candidate vertex.
- `c_water = c / n_water` is the photon speed used for time-of-flight
  correction.

The corrected time of hit `i` for a trial vertex `v` is:

```text
t_corr_i(v) = t_i - |p_i - v| / c_water
```

Most timing observables use corrected times. Angular observables use the
direction from the fitted vertex to each hit PMT:

```text
u_i = (p_i - xfit) / |p_i - xfit|
```

All distances in the observables are in cm. All time widths and RMS values are
in ns. Angular thresholds and outputs are in degrees.

## Vertex Determination

These observables describe how well the delayed hit cluster can be localized in
space and time. They are implemented in `vertex.py` and assembled in
`candidate_preselection.py`.

### `Nn`

`Nn` is the number of hits in the densest half-open corrected-time window. The
window width is set by `nn_window_ns`:

```text
[t0, t0 + nn_window_ns)
```

The initial `Nn` candidate search is performed after correcting delayed-search
hits to the prompt vertex. A candidate is kept only if `Nn > nn_cut`. With the
default `nn_window_ns = 10` and `nn_cut = 5`, this means at least 6 hits in the
best 10 ns window.

Why it helps:

- Neutron-capture gamma cascades produce a compact burst of delayed light.
- Random backgrounds are less likely to pile many hits into the same compact
  time-of-flight-corrected window.

Subtleties:

- The window is half-open to avoid boundary double-counting.
- If two windows have the same hit count, the one with smaller `trms` is
  preferred.
- The stored `Nn` is the initial candidate multiplicity before the local
  vertex scan. The refitted multiplicity enters through `delta_Nn`; it is the
  maximum configured-window multiplicity after correcting the refit-context
  hits to the fitted vertex.

### `trms`

`trms` is the RMS of the corrected hit times in the initial best `Nn` window.
It is computed around the mean corrected time of those hits.

Why it helps:

- True captures should remain compact after a reasonable time-of-flight
  correction.
- Accidental coincidences can pass an `Nn` count cut but often have a wider
  corrected-time spread.

Subtleties:

- `trms` is evaluated on the same hit subset that defines the initial `Nn`
  candidate.
- A small `trms` is not sufficient by itself; localized electronics effects can
  also produce tight timing clusters, which is why topology and noise
  observables are also used.

### `fpdist`

`fpdist` is the distance between the prompt/source vertex and the fitted delayed
vertex:

```text
fpdist = |xfit - v_prompt|
```

The fitted vertex is obtained with a coarse+fine grid multilateration scan
centered on `v_prompt`. The fit is sent the fixed original `Nn` seed-hit
cluster, scores trial vertices with median-centered corrected-time RMS, and may
apply a fine-stage `dt` cut. After that vertex is chosen, the code recomputes
the best configured-width window in the local context at the fitted vertex to
obtain the refitted `Nn`.

Why it helps:

- The neutron capture point should be spatially related to the prompt source
  region, but it may be displaced by neutron transport.
- Very large displacements can indicate noise, unrelated background, or a
  vertex fit pulled toward detector boundaries.

Subtleties:

- This is not a full reconstruction. It is a local double-grid multilateration
  placeholder for WCTE-sized geometry.
- The coarse/fine scan is configured by the `multilateration_*` settings in
  `config.py`. The scan should remain modest for WCTE; scanning meters would
  let random peaks choose unphysical vertices in a small detector.

### `delta_trms`

`delta_trms` measures the improvement in timing compactness after the local
vertex scan:

```text
delta_trms = initial_trms - refit_trms
```

Why it helps:

- A real capture cluster should often become tighter when corrected to a better
  delayed vertex.
- Noise clusters may not improve, or may improve only accidentally.

Subtleties:

- Positive values mean the refit made the candidate more time-compact.
- Negative values are possible if the best configured-width window found after
  the vertex refit is broader than the original seed window. The scan minimizes
  the fixed seed-hit tRMS; it does not directly minimize the final-window tRMS.
- The refit is evaluated only on hits near the original candidate in corrected
  time, controlled by `fit_context_ns`. If too few hits are in that context, the
  code falls back to all delayed-search hits for that prompt.

### `delta_Nn`

`delta_Nn` measures how the best configured-window multiplicity changes after
the local vertex scan:

```text
refit_Nn = max over t0 of count(tcorr_i(xfit) in [t0, t0 + nn_window_ns))
delta_Nn = refit_Nn - initial_Nn
```

Here the maximum is over configured-width timing windows at the already
selected fitted vertex `xfit`, using the hits in the local refit context.

Why it helps:

- A better vertex can align photon time-of-flight corrections and collect more
  hits into the configured timing window.
- A candidate that cannot be improved spatially may be less capture-like.

Subtleties:

- The BDT also sees the original `Nn`, so `delta_Nn` should be interpreted as
  an improvement variable, not as the final cluster size by itself.
- The fitted vertex minimizes the original seed cluster's tRMS; it does not
  maximize `refit_Nn` across trial vertices. Therefore `delta_Nn` is signed
  and can be negative.

### `fwall`

`fwall` is the estimated distance from `xfit` to the detector wall:

```text
fwall = distance_to_wall(xfit)
```

The current wall model is a simple cylinder inferred from PMT positions. For
WCTE, the default cylinder axis is `y`, because `y` is vertical in the WCTE
coordinate convention used here.

Why it helps:

- Vertices near or outside the wall proxy are more likely to be poorly fitted,
  edge-like, or noise-induced.
- This also guards against a local scan choosing trial vertices that are
  geometrically implausible.

Subtleties:

- This is a placeholder geometry proxy, not a detailed detector solid.
- Replacing it with a true distance-to-wall calculation would improve this
  observable without changing the rest of the pipeline.
- PMT coordinates must be loaded with the correct units; the default geofile is
  already interpreted in cm by this project.

### `Bpdist`

`Bpdist` measures the agreement of the two independent delayed-vertex fits:

```text
Bpdist = |xBONSAI - xfit|
```

`xfit` comes from the local Nn scan. `xBONSAI` comes from hk-BONSAI using
calibrated hit times, charges, and WCSim tube IDs. Small values indicate that
the two reconstruction methods found compatible vertices.

### `Bwall`

`Bwall` is the shortest non-negative distance from the BONSAI vertex to the
finite cylindrical wall proxy:

```text
Bwall = |xBONSAI - xwall|
```

Unlike the local scan, BONSAI can return an outside-detector vertex. The wall
helper therefore calculates the geometric distance to the cylinder boundary
for both inside and outside points.

### `trms3`

`trms3` is the minimum timing RMS among any 3-hit subset of the final candidate
hits, using corrected times at `xfit`.

Why it helps:

- It captures the tightest possible small subcluster inside the candidate.
- A true compact burst may contain a very narrow core even if extra accidental
  hits broaden the full candidate.

Subtleties:

- The exact minimum can be found by sorting corrected times and scanning
  contiguous 3-hit windows; no combinatorial search is needed.
- If fewer than 3 final candidate hits exist, the value is `NaN`.

### `trms6`

`trms6` is the same idea as `trms3`, but for the best 6-hit subset.

Why it helps:

- It is less sensitive to tiny 3-hit accidental coincidences.
- It probes whether a substantial part of the candidate is time-compact.

Subtleties:

- If fewer than 6 final candidate hits exist, the value is `NaN`.
- `trms3` and `trms6` should be interpreted together: a low `trms3` with a high
  `trms6` can indicate a very small tight core plus broader accidental activity.

## Cherenkov Event Topology

These observables describe the angular pattern of hit PMTs as seen from the
fitted delayed vertex. They are implemented in `topology.py`.

### `theta_mean`

`theta_mean` is the mean angle between each hit direction `u_i` and the mean hit
axis:

```text
mean_axis = normalize(sum_i u_i)
theta_i   = arccos(u_i dot mean_axis)
theta_mean = mean(theta_i)
```

Why it helps:

- Capture gamma cascades should be relatively diffuse compared with strongly
  directional backgrounds.
- A small `theta_mean` means hits are concentrated around one direction.

Subtleties:

- If the hit directions cancel exactly, the mean axis is undefined and angular
  quantities can become `NaN`.
- This is a topology summary, not a Cherenkov-ring fit.

### `theta_rms`

`theta_rms` is the RMS spread of the `theta_i` angles around `theta_mean`.

Why it helps:

- It separates narrow directional clusters from broad angular patterns.
- It complements `theta_mean`: two candidates can have similar mean angle but
  different angular spread.

Subtleties:

- The RMS ignores non-finite values through the shared `safe_rms` helper.
- Low multiplicity candidates can have unstable angular RMS values.

### `phi_rms`

`phi_rms` measures azimuthal non-uniformity around the mean hit axis. The code
builds an orthonormal basis perpendicular to the mean axis, computes each hit's
azimuthal angle, sorts those angles, and compares the gaps to equal spacing.

Why it helps:

- An isotropic capture-like pattern should have relatively even azimuthal
  coverage.
- Directional or localized backgrounds tend to leave uneven azimuthal gaps.

Subtleties:

- The observable uses azimuthal gaps rather than raw `phi` values, so it is
  invariant under a global rotation around the mean axis.
- A stable fallback basis is used when the mean axis is nearly parallel to the
  default basis vector.

### `Nlowtheta`

`Nlowtheta` counts hits whose angle to the mean hit axis is below
`low_theta_deg`:

```text
Nlowtheta = count(theta_i < low_theta_deg)
```

The default threshold is 25 degrees.

Why it helps:

- A high value means many hits are tightly aligned with the main direction.
- This can indicate a directional topology rather than a diffuse capture
  cascade.

Subtleties:

- The threshold is exposed as a CLI/config parameter because the useful value
  depends on detector scale, PMT layout, and source location.

### `Nback`

`Nback` counts hits whose angle to the mean hit axis is larger than
`back_theta_deg`:

```text
Nback = count(theta_i > back_theta_deg)
```

The default threshold is 90 degrees.

Why it helps:

- It measures the backward or wide-angle population relative to the candidate's
  main direction.
- Capture-like events can have broader angular distributions than a single
  directional background.

Subtleties:

- `Nback` is only meaningful relative to the fitted vertex and mean axis.
- Mis-reconstructed vertices can change the angular pattern substantially.

### `Nlow`

`Nlow` counts hits with a low geometric light-collection proxy:

```text
weight_i = incidence_i * exp(-r_i / latt) / r_i^2
Nlow = count(weight_i < low_weight_cut)
```

where:

- `r_i = |p_i - xfit|`
- `incidence_i` is based on the PMT direction and the hit direction
- `latt` is the attenuation length

Why it helps:

- Hits that are geometrically unlikely for the fitted vertex can indicate a
  bad vertex, accidental background, or detector noise.
- The observable gives the BDT a rough consistency check between the fitted
  vertex and the PMT hit pattern.

Subtleties:

- This is not a calibrated PE prediction. It is only a relative geometric
  proxy.
- If PMT directions are unavailable, incidence is set to 1 so the observable
  remains defined but becomes less physically discriminating.
- The distance term is protected at 1 cm to avoid singular weights if a trial
  vertex lands too close to a PMT.

## Noise Characterization

These observables describe charge tails, local PMT clustering, and surrounding
time activity. They are implemented in `noise.py`.

### `Qmean`

`Qmean` is the mean charge of the final candidate hits.

Why it helps:

- Capture candidates should usually be dominated by low-PE hits.
- Large mean charge can indicate merged pulses, non-capture activity, or
  detector noise.

Subtleties:

- The interpretation depends on the input charge branch and calibration. By
  default, the pipeline reads `hit_pmt_charges`.
- For real-background-derived files, make sure the charge branch is already in
  the PE-like convention expected by the training sample.

### `Qrms`

`Qrms` is the RMS of the final candidate hit charges.

Why it helps:

- It measures whether the candidate has a broad charge distribution or a high
  charge tail.
- It complements `Qmean`: a candidate can have a modest mean but still contain
  a few anomalously large hits.

Subtleties:

- Non-finite charge values are ignored by the shared `safe_rms` helper.
- Same-PMT hit merging upstream can change charge RMS substantially.

### `NhighQ`

`NhighQ` counts final candidate hits with charge greater than or equal to
`high_charge_pe`:

```text
NhighQ = count(q_i >= high_charge_pe)
```

The default threshold is 3 PE.

Why it helps:

- High-charge hits are more common in some noise bursts, merged hits, or
  non-capture backgrounds than in low-energy delayed capture light.

Subtleties:

- The threshold should be retuned if the charge calibration or PE convention
  changes.
- This count is candidate-local; it does not count high-charge hits elsewhere
  in the 270 us readout window.

### `Nclus`

`Nclus` counts hits that belong to compact angular connected components around
the fitted vertex. Two hits are connected if their directions from `xfit` are
within `cluster_angle_deg`; connected components with at least 3 hits
contribute all their hits to `Nclus`.

Why it helps:

- Tight angular clusters can signal local detector effects, noisy modules, or
  PMT-neighborhood bursts.
- WCTE mPMTs have a small angular span from the source region, so this is a
  useful handle on local activity.

Subtleties:

- The default `cluster_angle_deg` is 14 degrees, chosen as a WCTE-sized
  placeholder near the angular span of one mPMT from the source region.
- This observable depends strongly on geometry and source position; it should
  be revisited if the detector geometry or source location changes.

### `N300`

`N300` counts delayed-search hits in a wider 300 ns corrected-time window around
the original candidate center:

```text
N300 = count(|t_corr_i(v_prompt) - candidate_center| <= 150 ns)
```

Why it helps:

- It measures nearby time activity around the compact `Nn` candidate.
- A real capture can have a compact core, while high accidental pileup can add
  many surrounding hits in the wider window.

Subtleties:

- `N300` is computed using the prompt-vertex corrected times from the delayed
  search, not the final refitted candidate hit list.
- It is intentionally placed in the noise group because it describes local
  time-environment activity rather than vertex quality.

## Cleaning And Preselection Quantities

The following quantities affect candidate production but are not BDT feature
columns by default.

### `Nmax200`

`Nmax200` is the maximum number of delayed-search hits in any 200 ns raw-time
window for one prompt search. If a 200 ns window contains at least
`nmax200_cut` hits, only that locally dense burst region is vetoed; the rest of
the 300 us delayed search remains available for capture candidate finding.

Why it exists:

- Very dense 200 ns bursts are usually not isolated delayed neutron-capture
  candidates.
- Vetoing them prevents pathological readout periods or prompt-like leakage
  from dominating candidate finding without deleting the full delayed search.

Subtleties:

- It is evaluated after prompt tagging and inside the prompt-specific delayed
  search region.
- The diagnostic column `nmax200` stores the pre-veto maximum for that prompt's
  delayed search, even if the final candidate is found elsewhere in the 300 us
  interval.
- The extraction summary records how often this local veto was used with
  `delayed_searches_with_nmax200_veto`, how many hits were removed with
  `hits_removed_by_nmax200_veto`, and how often the delayed search was emptied
  with `delayed_searches_emptied_by_nmax200_veto`.
- `nmax200` is not included in `OBSERVABLE_COLUMNS`.

### Continuous same-PMT noise removal

Before delayed candidate finding, the code removes close repeated hits on the
same PMT. If two hits on the same PMT are separated by less than
`continuous_noise_ns`, both hits in that close pair are discarded.

Why it exists:

- It reduces afterpulse-like and unstable-channel contributions.
- It prevents one PMT from creating an artificial compact time cluster.

Subtleties:

- The PMT identity is based on `(slot, pos)` when available, matching the WCTE
  readout convention and geometry lookup.
- This is conservative: it can remove true photons if multiple real hits occur
  on the same PMT inside the coincidence window.
- The default window is 6000 ns and should be treated as a tunable placeholder.

## Shared Helpers

`common.py` contains small numerical utilities used by multiple observable
groups:

- `safe_rms(values)`: RMS around the mean after dropping non-finite values.
- `unit_vectors(vectors)`: row-wise vector normalization with finite-safe
  handling of zero-length vectors.
- `angular_distance_deg(unit_dirs)`: pairwise angular distance matrix for
  already-normalized direction vectors.

Keeping these helpers centralized makes the behavior consistent across timing,
topology, and clustering observables.

## Tuning Parameters

The main observable thresholds and geometry placeholders live in `extraction/config.py`
and are exposed through the command-line interface:

- Timing: `nn_window_ns`, `nn_cut`, `n300_window_ns`, `nmax200_window_ns`,
  `nmax200_cut`, `continuous_noise_ns`.
- Local vertex fit: `fit_context_ns`, `fit_wall_margin_cm`,
  `multilateration_xyz_bounds_cm`, `multilateration_coarse_step_cm`,
  `multilateration_fine_step_cm`,
  `multilateration_refine_halfwidth_cm`, `multilateration_dt_cut_ns`,
  `multilateration_grid_chunk_size`, `multilateration_min_hits`, and
  `multilateration_earliest_per_channel`.
- BONSAI: `bonsai_window_ns`, `bonsai_param`, `bonsai_geometry_root`,
  `bonsai_dir`, and `wcsim_dir`.
- Detector optics/topology: `water_refractive_index`, `attenuation_length_cm`,
  `low_theta_deg`, `back_theta_deg`, `low_weight_cut`.
- Noise/charge: `cluster_angle_deg`, `high_charge_pe`.

These values are intentionally easy to locate because they are analysis-tuning
placeholders, especially for WCTE's compact geometry.

## BONSAI Fit Failures

If a BONSAI vertex fit fails, `Bpdist` and `Bwall` are stored as
`NaN` and `bonsai_fit_success` is zero.
