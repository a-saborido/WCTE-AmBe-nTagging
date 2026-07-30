"""Feature selection for training the neutron-tagging BDT."""

# This list is independent of extraction/config.py. Remove entries here to
# train and evaluate the model with a subset of the extracted observables.
FEATURE_COLUMNS = [
    # Vertex determination
    "Nn",
    "trms",
    "fpdist",
    "delta_trms",
    "delta_Nn",
    "fwall",
    "trms3",
    "trms6",
    "Bpdist",
    "Bwall",
    # Candidate timing
    #"candidate_time_from_prompt_ns", # don't use, at least for increased neutron yield
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
