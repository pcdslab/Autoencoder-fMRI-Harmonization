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
conda activate asd_env
```


## Data Preparation
The script file, `asd_harmonization_AEs.py`, now uses a configuration file to set the paths for the required data. Update the `config.ini` file with the following paths:

- `DATA_PATH`: Path to the raw fMRI data.
- `PHENO_PATH`: Path to the phenotype (pheno) data.
- `SAMPLE_PATH`: Path to the input sample file.

## Running the Experiments

### Example Command
Below is an example command for running the main Python script:

1. Autoencoder harmonization using proportional split
```bash
python asd_harmonization_AEs.py --p_method ASD-Standalone-AE-10-fold --ae_type AE --ml_method RF
```
- `--p_method`: Specifies the processing method (e.g., ASD-Standalone-AE-10-fold).
- `--ae_type`: Specifies the type of autoencoder to use (e.g., AE, TAE, SAE, DAE).
- `--ml_method`: Specifies the machine learning method (e.g., RF, SVM, NB).

2. ML classification using with ComBat and without ComBat - proportional split
```bash
python asd_harmonization_AEs.py --p_method ASD-ml --ml_method RF --run_combat
```
- `--p_method`: Specifies the processing method (e.g., ASD-ml).
- `--ml_method`: Specifies the machine learning method (e.g., RF, SVM, NB).
- `--run_combat`: it will run combat if provided the arg, if it is not provided, it will not run_combat.

2. Center classification using an ML method, like RF, NB and SVM. It will take ASD of first-center, NT of second-center and vice versa
```bash
python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers-asd-vs-hc --ml_method RF --run_combat --fold 5 --centers UCLA,KKI
```
- `--p_method`: Specifies the processing method (e.g., ASD-ml-combine-two-centers-asd-vs-hc).
- `--ml_method`: Specifies the machine learning method (e.g., RF).
- `--run_combat`: it will run combat if provided the arg, if it is not provided, it will not run_combat.
- `--fold`: Specifies the cross-validation fold.
- `--centers`: Comma-separated list of two centers to include in the analysis.

3. Center classification using NT subjects only from two given centers, and the model will classify the center with and without ComBat. no need to pass --run_combat because the method will iterate through both of them
```bash
python asd_harmonization_AEs.py --p_method ASD-ml-combine-two-centers --ml_method RF --run_combat --fold 5 --centers UCLA,KKI
```
- `--p_method`: Specifies the processing method (e.g., ASD-ml-combine-two-centers-asd-vs-hc).
- `--ml_method`: Specifies the machine learning method (e.g., RF).
- `--run_combat`: it will run combat if provided the arg, if it is not provided, it will not run_combat.
- `--fold`: Specifies the cross-validation fold.
- `--centers`: Comma-separated list of two centers to include in the analysis.

4. Leave One Site Out experiment using AEs as harmonization method
```bash
python asd_harmonization_AEs.py --p_method ASD-Standalone-AE --ae_type AE --ml_method RF
```
- `--p_method`: Specifies the processing method (e.g., ASD-ml-combine-two-centers-asd-vs-hc).
- `--ae_type`: Specifies the type of autoencoder to use (e.g., AE, TAE, SAE, DAE).
- `--ml_method`: Specifies the machine learning method (e.g., RF, SVM, NB).


5. Leave One Site Out experiment with and without ComBat
```bash
python asd_harmonization_AEs.py --p_method ASD-ml --ml_method RF --run_combat
```
- `--p_method`: Specifies the processing method (e.g., ASD-ml).
- `--ml_method`: Specifies the machine learning method (e.g., RF, SVM, NB).
- `--run_combat`: it will run combat if provided the arg, if it is not provided, it will not run_combat.

## Results and Analysis
- Once the experiments are completed, Jupyter notebooks for result visualization and analysis can be found in the `notebooks` folder.
- Open the relevant notebook and follow the instructions for plotting and evaluating results.

## Additional Notes
This project uses the ComBat harmonization method, available at [https://github.com/Jfortin1/ComBatHarmonization](https://github.com/Jfortin1/ComBatHarmonization).

## Contact
If you have any questions or encounter any issues, please reach out to the repository maintainer at [falmu027@fiu.edu](mailto:falmu027@fiu.edu).

