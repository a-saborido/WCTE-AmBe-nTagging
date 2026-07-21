"""Feature selection for training the neutron-tagging BDT."""

# This list is independent of extraction/config.py. Remove entries here to
# train and evaluate the model with a subset of the extracted observables.
FEATURE_COLUMNS = [
    # Vertex determination
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
    # Cherenkov event topology
    "theta_mean",
    "theta_rms",
    "phi_rms",
    "Nlowtheta",
    "Nback",
    "Nlow",
    # Noise characterization
    "Qmean",
    "Qrms",
    "NhighQ",
    "Nclus",
    "N300",
]
