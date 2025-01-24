# README

## Overcoming Site Variability in Multisite fMRI Studies: An Autoencoder Framework for Enhanced Generalizability of Machine Learning Models

### Abstract

Harmonizing multisite functional magnetic resonance imaging (fMRI) data is crucial for eliminating site-specific variability that hinders the generalizability of machine learning models. Traditional harmonization techniques such as ComBat rely on simple additive and multiplicative effects, failing to capture the complex non-linear interactions between scanner hardware, acquisition protocols, and biological signal variations across imaging sites. Additionally, ComBat requires data from all sites during model training, limiting its applicability to new, unseen scanning sites. Concerns such as data leakage are commonly observed with statistical harmonization methods which may result in low reproducibility of ML models across different sites. In this study, we propose Autoencoders (AEs) as an alternative for harmonizing multisite fMRI data. Our designed and developed framework leverages the non-linear representation learning capabilities of AEs to reduce site-specific effects while preserving biologically meaningful features. Our evaluation using Autism Brain Imaging Data Exchange (ABIDE-I) dataset, containing 1,035 subjects collected from 17 centers demonstrates statistically significant improvements in leave-one-site-out (LOSO) cross-validation evaluations. All autoencoder variants (AE, SAE, TAE, and DAE) significantly outperformed the baseline mode (p<0.01), with mean accuracy improvements ranging from 3.41% to 5.04%. Our findings demonstrate the potential of Autoencoders to harmonize multisite neuroimaging data effectively enabling robust downstream analyses across various neuroscience applications while reducing data-leakage, and preservation of neurobiological features.


## Overview

This repository contains the code and scripts to run experiments for ASD harmonization using Autoencoders (AEs). The experiments are designed to be submitted to a SLURM workload manager, with configurable settings to customize the runs.

## Requirements

The code requires the following dependencies:

- Python (via a Conda environment)
- `neuroCombat` Python package

## Setup Instructions

### Step 1: Create a Conda Environment

Create a new Conda environment and install the required dependencies.

```bash
conda create -n asd_env python=3.8 -y
conda activate asd_env
pip install neuroCombat
```

### Step 2: Update the Data Paths

The script file, `asd_harmonization_AEs.py`, requires paths to the raw fMRI data and phenotype (pheno) data. Update the following lines in the file:

- **Line 152**: Replace with the folder path to the raw fMRI data.
- **Line 162**: Replace with the folder path to the phenotype (pheno) data.

## Running the Experiments

### Step 1: Prepare the SLURM Script

The SLURM script, `run_experiments.sbatch`, contains multiple experimental runs with different settings. Each run is commented out by default. To execute specific experiments:

1. Open the file `script/run_experiments.sbatch`.
2. Uncomment the relevant `python` command(s) corresponding to the experiments you wish to run.

### Step 2: Submit the SLURM Script

Submit the SLURM script to the SLURM workload manager using the following command:

```bash
sbatch script/run_experiments.sbatch
```

### Notes:

- Ensure the data paths in `asd_harmonization_AEs.py` are correctly updated before running the script.
- If you need to run additional experiments, you can modify the settings in `run_experiments.sbatch` or add new commands as needed.

## File Structure

```
.
├── asd_harmonization_AEs.py  # Main Python script for experiments
├── script/
│   └── run_experiments.sbatch  # SLURM script to run experiments
├── data-files.md  # File containing details about the data directory
├── notebooks/  # Folder containing Jupyter notebooks to plot results
└── README.md  # Instructions to set up and run the code
```

## Contact

If you have any questions or encounter any issues, please reach out to the repository maintainer at [falmu027@fiu.edu](mailto\:falmu027@fiu.edu).

