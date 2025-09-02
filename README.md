# ASD Harmonization

## Overcoming Site Variability in Multisite fMRI Studies: An Autoencoder Framework for Enhanced Generalizability of Machine Learning Models

### Abstract
Multisite functional magnetic resonance imaging (fMRI) studies are vital for advancing our understanding of brain disorders. However, site variability poses a significant challenge to the generalizability of machine learning models. In this work, we propose an autoencoder (AE) framework to mitigate site-specific variations and improve model performance across diverse datasets. By leveraging AEs, we demonstrate enhanced generalizability and robustness in ASD classification tasks. This framework is evaluated on multisite fMRI data, highlighting its ability to overcome site variability and set a new standard for harmonization techniques in neuroimaging studies.

## Citation
If you use this framework in your work, please cite the associated publications.

## System Requirements
- A computer with Ubuntu 16.04 (or later) or CentOS 8.1 (or later).
- CUDA-enabled GPU with at least 12 GB of memory.

## Installation Guide

### Install Anaconda
[Step by Step Guide to Install Anaconda](https://docs.anaconda.com/anaconda/install/)


### Fork the Repository
- Fork this repository to your own account.
- Clone your fork to your machine.

### Create a Conda Environment
```bash
cd <repository_directory>
conda env create --file environment.yml
```

### Activate the Environment
```bash
conda activate ae_env
```

### Install Pytorch and pyprind
```bash
pip install pyprind
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Data Preparation
The script file, `asd_harmonization_AEs.py`, now uses a configuration file to set the paths for the required data. Update the `config.ini` file with the following paths:

- `DATA_PATH`: Path to the raw fMRI data.
- `PHENO_PATH`: Path to the phenotype (pheno) data.
- `SAMPLE_PATH`: Path to the input sample file.


## Usage

```bash
python asd_harmonization_AEs.py --p_method <p_method_value> [--ml_method <ml_method_value>] [--ae_type <ae_type_value>] [--run_combat] [--centers <center1,center2>] [--fold <fold_value>]
```

## Run trained models
You can download the trained models and data splits from the [latest release](https://github.com/pcdslab/Autoencoder-fMRI-Harmonization/releases).

Then, you can run the following command to run TAE for proportional split:

```bash
python run_trained_harmonization_AEs.py --p_method ASD-Standalone-AE-10-fold --model_dir models/ --ae_type TAE
```

### Options

- **`--p_method`** (Required)
  Specifies the processing method for the pipeline.\
  Example values: `ASD-Standalone-AE-10-fold`, `ASD-ml-combine-two-centers`, `ASD-ml-combine-two-centers-asd-vs-hc`, `ASD-ml`, `ASD-Standalone-AE`

- **`--ml_method`** (Optional)
  Specifies the machine learning method.\
  Default: `RF`.\
  Example values: `RF`, `SVM`, `NB`

- **`--ae_type`** (Optional)
  Specifies the autoencoder type to use.\
  Default: `AE`.\
  Example values: `AE`, `TAE`, `SAE`, `DAE`

- **`--run_combat`** (Optional)
  Enables ComBat harmonization in the pipeline.\
  When included, the pipeline applies ComBat. If not specified, ComBat is not used. Please note this is used when ComBat is in the experiment

- **`--centers`** (Optional)
  Comma-separated list of two centers to include in the analysis.\
  Default: `None`.\
  Example: `Pitt,Yale`. Used for center classification experiments.

- **`--fold`** (Optional)
  Specifies the fold number for cross-validation experiments. It only used in proportional split experiments not in the LOSO experiments\
  Default: 10.\
  Example values: 5, 10
---

## Running the Experiments

Below is an example command for running the paper experiments:

1. Autoencoder harmonization using proportional split
```bash
python asd_harmonization_AEs.py --p_method ASD-Standalone-AE-10-fold --ae_type AE --ml_method RF
```
---

2. ML classification using with ComBat and without ComBat - proportional split
```bash
python asd_harmonization_AEs.py --p_method ASD-ml --ml_method RF --run_combat
```
---

3. Center classification using an ML method, like RF, NB and SVM. It will take ASD of first-center, NT of second-center and vice versa
```bash
python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ml_method RF --run_combat --fold 5 --centers UCLA,KKI
```
---

4. Center classification using NT subjects only from two given centers, and the model will classify the center with and without ComBat. no need to pass --run_combat because the method will iterate through both of them
```bash
python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ml_method RF --run_combat --fold 5 --centers UCLA,KKI
```
---

4. Leave One Site Out experiment using AEs as harmonization method
```bash
python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF
```
---


5. Leave One Site Out experiment with and without ComBat
```bash
python asd_harmonization_AEs.py --p_method ASD-ml --ml_method RF --run_combat
```
---

## Results and Analysis
- Once the experiments are completed, Jupyter notebooks for result visualization and analysis can be found in the `notebooks` folder.
- Open the relevant notebook and follow the instructions for plotting and evaluating results.

## Additional Notes
This project uses the ComBat harmonization method, available at [https://github.com/Jfortin1/ComBatHarmonization](https://github.com/Jfortin1/ComBatHarmonization).

## Contact
If you have any questions or encounter any issues, please reach out at [falmu027@fiu.edu](mailto:falmu027@fiu.edu).

