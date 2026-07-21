"""Central configuration for the WCTE AmBe neutron-tagging pipeline."""

from __future__ import annotations

import os
from pathlib import Path


C_LIGHT_CM_PER_NS = 29.9792458
DEFAULT_WATER_REFRACTIVE_INDEX = 1.333
REPO_DIR = Path(__file__).resolve().parents[1]

# WCTE geometry / reconstruction placeholders.
DEFAULT_GEOMETRY_FILE = REPO_DIR / "data" / "geofile_NuPRISMBeamTest_16cShort_mPMT.txt"
DEFAULT_GEOMETRY_SCALE_TO_CM = 1.0
DEFAULT_PMT_POSITION_BASE = "zero"
# Vertical tank axis used by the cylindrical wall approximation in geometry.py.
DEFAULT_WALL_AXIS = "y"
DEFAULT_PROMPT_VERTEX_CM = (0.0, 30.5, 0.0)

# hk-BONSAI / WCSim integration. Environment variables from ~/setup.sh take
# precedence, while the absolute fallbacks make batch jobs reproducible.
DEFAULT_BONSAI_DIR = Path(os.environ.get("BONSAIDIR", "/scratch/saborido/WCTE/hk-BONSAI"))
DEFAULT_BONSAI_PARAM = Path(
    os.environ.get("BONSAIPARAM", str(DEFAULT_BONSAI_DIR / "data" / "fit_param_wcte.dat"))
)
DEFAULT_BONSAI_GEOMETRY_ROOT = DEFAULT_BONSAI_DIR / "NiCf" / "wcsim_dummy.root"
DEFAULT_WCSIM_DIR = Path(
    os.environ.get("WCSIM_BUILD_DIR", "/scratch/saborido/wcsim-dev/wcsim-install")
)
DEFAULT_BONSAI_WINDOW_NS = 1300.0
DEFAULT_OBSERVABLE_PLOT_BINS = 60

# Readout-window handling. One ROOT entry is a 270 us readout window. The next
# entry is represented in a continuous delayed-search time axis after a 7.75 us
# gap with no hits.
DEFAULT_WINDOW_DURATION_NS = 270_000.0
DEFAULT_WINDOW_GAP_NS = 7_750.0
DEFAULT_CAPTURE_SEARCH_WINDOW_NS = 300_000.0
DEFAULT_CAPTURE_START_AFTER_PROMPT_NS = 5_000.0

# Prompt tagging: broad, high-multiplicity scintillation peaks. The sliding
# window is in raw readout time; tRMS and tmean are evaluated after correcting
# each hit for photon travel time from the fixed source position.
DEFAULT_PROMPT_WINDOW_NS = 1_000.0
DEFAULT_PROMPT_MIN_HITS = 80
DEFAULT_PROMPT_MAX_HITS = 300
DEFAULT_PROMPT_MIN_TRMS_NS = 200.0
DEFAULT_PROMPT_MAX_TRMS_NS = 500.0
DEFAULT_PROMPT_MIN_TMEAN_NS = 200.0
DEFAULT_PROMPT_MAX_TMEAN_NS = 400.0
DEFAULT_PROMPT_ISOLATION_NS = 200.0

# Candidate preselection / cleaning.
DEFAULT_NMAX200_WINDOW_NS = 200.0
DEFAULT_NMAX200_CUT = 50
DEFAULT_CONTINUOUS_NOISE_NS = 6000.0
DEFAULT_N10_WINDOW_NS = 10.0
DEFAULT_N10_CUT = 5
DEFAULT_MAX_CANDIDATES_PER_PROMPT = 50

# WCTE is much smaller than SK.
DEFAULT_FIT_CONTEXT_NS = 25.0
DEFAULT_FIT_WALL_MARGIN_CM = 10.0
DEFAULT_N300_WINDOW_NS = 300.0

# Double-grid multilateration used for the delayed-candidate local vertex.
# The coarse grid is centered on the prompt/source vertex; the fine grid is
# centered on the best coarse point.
DEFAULT_MULTILATERATION_XYZ_BOUNDS_CM = 120.0
DEFAULT_MULTILATERATION_COARSE_STEP_CM = 10.0
DEFAULT_MULTILATERATION_FINE_STEP_CM = 1.0
DEFAULT_MULTILATERATION_REFINE_HALFWIDTH_CM = 20.0
DEFAULT_MULTILATERATION_DT_CUT_NS = 10.0
DEFAULT_MULTILATERATION_GRID_CHUNK_SIZE = 4096
DEFAULT_MULTILATERATION_MIN_HITS = 6
DEFAULT_MULTILATERATION_EARLIEST_PER_CHANNEL = True

# Cherenkov/topology and noise-observable tuning.
DEFAULT_ATTENUATION_LENGTH_CM = 7000.0
DEFAULT_LOW_WEIGHT_CUT = 1e-5
DEFAULT_LOW_THETA_DEG = 20.0
DEFAULT_BACK_THETA_DEG = 90.0

# The full angular span of one WCTE mPMT is about 13 deg from the source
# region; 14 deg groups hits that are locally clustered on one module.
DEFAULT_CLUSTER_ANGLE_DEG = 14.0
DEFAULT_HIGH_CHARGE_PE = 3.0

# Truth-label thresholds. These are label-only cuts and must not enter any
# candidate observable.
DEFAULT_MIN_CAPTURE_HITS = 3
DEFAULT_MIN_CAPTURE_FRACTION = 0.50
DEFAULT_MIN_CAPTURE_HITS_ABSOLUTE = 4

# Vertex determination: timing compactness and quality of the local vertex scan.
VERTEX_OBSERVABLE_COLUMNS = [
    "N10",
    "trms",
    "fpdist",
    "delta_trms",
    "delta_N10",
    "fwall",
    "trms3",
    "trms6",
    "Bpdist",
    "Bwall",
]

# Cherenkov event topology: angular pattern and geometric light-yield proxy.
TOPOLOGY_OBSERVABLE_COLUMNS = [
    "theta_mean",
    "theta_rms",
    "phi_rms",
    "Nlowtheta",
    "Nback",
    "Nlow",
]

# Noise characterization: charge tails, local PMT clustering, and nearby time
# activity around the candidate peak.
NOISE_OBSERVABLE_COLUMNS = [
    "Qmean",
    "Qrms",
    "NhighQ",
    "Nclus",
    "N300",
]

OBSERVABLE_GROUPS = {
    "vertex_determination": VERTEX_OBSERVABLE_COLUMNS,
    "cherenkov_event_topology": TOPOLOGY_OBSERVABLE_COLUMNS,
    "noise_characterization": NOISE_OBSERVABLE_COLUMNS,
}

OBSERVABLE_COLUMNS = [col for columns in OBSERVABLE_GROUPS.values() for col in columns]
