import torch
import random
import numpy as np

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default paths
DATA_PATH = "/disk/raptor-array/SaeedLab-Data/Neuroimaging/ASD/fmri_rois_200"
PHENO_PATH = "/lclhome/falmu027/projects/Autoencoder-fMRI-Harmonization/sample_data/Phenotypic_V1_0b_preprocessed1.csv"
SAMPLE_PATH = "disk/raptor-array/SaeedLab-Data/Neuroimaging/ASD/10_CV_1035.csv"
PROP_SPLIT_PATH = "/lclhome/falmu027/projects/Autoencoder-fMRI-Harmonization/sample_data/PS_sample_file.csv"