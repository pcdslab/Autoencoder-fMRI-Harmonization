from neuroCombat import neuroCombat, neuroCombatFromTraining
import pandas as pd
import numpy as np


def neuroCombatFromTraining1(dat, batch, estimates):
    """
    Combat harmonization with pre-trained ComBat estimates [UNDER DEVELOPMENT]
    Arguments
    ---------
    dat : a pandas data frame or numpy array for the new dataset to harmonize
        - rows must be identical to the training dataset

    batch : numpy array specifying scanner/batch for the new dataset
        - scanners/batches must also be present in the training dataset
    estimates : dictionary of ComBat estimates from a previously-harmonized dataset
        - should be in the same format as neuroCombat(...)['estimates']

    Returns
    -------
    A dictionary of length 2:
    - data: A numpy array with the same shape as `dat` which has now been ComBat-harmonized
    - estimates: A dictionary of the ComBat estimates used for harmonization
    """
    # print("[neuroCombatFromTraining local] In development ...\n")
    batch = np.array(batch, dtype="str")
    new_levels = np.unique(batch)
    old_levels = np.array(estimates["batches"], dtype="int")
    old_levels = np.array(old_levels, dtype="str")
    missing_levels = np.setdiff1d(new_levels, old_levels)
    if missing_levels.shape[0] != 0:
        raise ValueError(
            "The batches "
            + str(missing_levels)
            + " are not part of the training dataset"
        )

    wh = [int(np.where(old_levels == x)[0]) if x in old_levels else None for x in batch]

    var_pooled = estimates["var.pooled"]
    stand_mean = estimates["stand.mean"][:, 0]
    mod_mean = estimates["mod.mean"]
    gamma_star = estimates["gamma.star"]
    delta_star = estimates["delta.star"]
    n_array = dat.shape[1]
    stand_mean = stand_mean + mod_mean.mean(axis=1)

    stand_mean = np.transpose(
        [
            stand_mean,
        ]
        * n_array
    )
    bayesdata = np.subtract(dat, stand_mean) / np.sqrt(var_pooled)

    # gamma = np.transpose(np.repeat(gamma_star, repeats=2, axis=0))
    # delta = np.transpose(np.repeat(delta_star, repeats=2, axis=0))
    gamma = np.transpose(gamma_star[wh, :])
    delta = np.transpose(delta_star[wh, :])
    bayesdata = np.subtract(bayesdata, gamma) / np.sqrt(delta)
    bayesdata = bayesdata * np.sqrt(var_pooled) + stand_mean
    out = {"data": bayesdata, "estimates": estimates}
    return out


def getEstimator(all_corr, sample_files):
    print("getting estimator ...")
    # for testing
    # sample_files = sample_files[:100]

    scanners = {}
    id = 1
    # assign scanner ids
    for name in sample_files:
        if name.startswith("MaxMun") or name.startswith("CMU"):
            center = name.split("_")[0]
        else:
            center = "_".join(name.split("_")[:-1])
        if center not in scanners:
            scanners[center] = id
            id = id + 1
    print(scanners)

    # read data, add to matrix, assign scanner
    batch = []
    data = np.empty((19900, len(sample_files)))

    for i, sample in enumerate(sample_files):
        print(sample)
        data[:, i] = all_corr[sample][0]

        # data.append(all_corr[sample][0])
        # center = '_'.join(sample.split('_')[:-1])
        if sample.startswith("MaxMun") or sample.startswith("CMU"):
            center = sample.split("_")[0]
        else:
            center = "_".join(sample.split("_")[:-1])
        batch.append(scanners[center])
    # print('batches:',len(set(batch)))

    data = np.array(data)
    batch = np.array(batch)
    covars = {"batch": batch}
    covars = pd.DataFrame(covars)
    # To specify the name of the variable that encodes for the scanner/batch covariate:
    batch_col = "batch"

    output = neuroCombat(dat=data, covars=covars, batch_col=batch_col)

    est = output["estimates"]
    # print(output['data'][:, 0][:10])

    # sample = sample_files[0]
    # dat = np.empty((19900, 1))
    # dat[:, 0] = all_corr[sample][0]

    # bat = np.array([scanners[sample.split('_')[0]]])
    # print(bat)
    # print(dat.shape)
    # out = neuroCombatFromTraining(dat=dat, batch=bat, estimates=est)
    # print('data.shape: ', out['data'][:, 0].shape)
    # print('all_corr[sample][0]:', all_corr[sample][0].shape)
    # print('out[data].shape:', out['data'].shape)
    # print(dat[:10])

    # out = neuroCombatFromTraining1(dat=dat, batch=bat, estimates=est)
    # print('data.shape: ', out['data'][:, 0].shape)
    # print('all_corr[sample][0]:', all_corr[sample][0].shape)
    # print('out[data].shape:', out['data'].shape)
    # print(dat[:10])

    # return scanners and estmator
    return scanners, est


def getEstimator(all_corr, sample_files, pheno_data, add_gender=False, add_age=False):
    print("getting estimator ...")
    # for testing
    # sample_files = sample_files[:100]

    scanners = {}
    id = 1
    # assign scanner ids
    for name in sample_files:
        center = name.split("_")[0]
        # if name.startswith('MaxMun') or name.startswith('CMU'):
        #   center = name.split('_')[0]
        # else:
        #   center = '_'.join(name.split('_')[:-1])

        if center not in scanners:
            scanners[center] = id
            id = id + 1
    print(scanners)

    # read data, add to matrix, assign scanner
    batch = []
    age = []
    gender = []
    labels = []
    data = np.empty((19900, len(sample_files)))

    for i, sample in enumerate(sample_files):
        # print(sample)
        data[:, i] = all_corr[sample][0]
        if add_age:
            age.append(
                list(pheno_data[pheno_data["FILE_ID"] == sample]["AGE_AT_SCAN"])[0]
            )
        if add_gender:
            gender.append(list(pheno_data[pheno_data["FILE_ID"] == sample]["SEX"])[0])
        labels.append(list(pheno_data[pheno_data["FILE_ID"] == sample]["DX_GROUP"])[0])
        center = sample.split("_")[0]
        # data.append(all_corr[sample][0])
        # center = '_'.join(sample.split('_')[:-1])
        # if sample.startswith('MaxMun') or sample.startswith('CMU'):
        #   center = sample.split('_')[0]
        # else:
        #   center = '_'.join(sample.split('_')[:-1])
        batch.append(scanners[center])

    data = np.array(data)
    batch = np.array(batch)
    age = np.array(age)
    gender = np.array(gender)
    labels = np.array(labels)

    if add_age and add_gender:
        covars = {"batch": batch, "gender": gender, "age": age, "label": labels}
    elif add_age:
        covars = {"batch": batch, "age": age, "label": labels}
    elif add_gender:
        covars = {"batch": batch, "gender": gender, "label": labels}
    else:
        covars = {"batch": batch}
        # covars = {"batch": batch, "label": labels}

    covars = pd.DataFrame(covars)
    # To specify the name of the variable that encodes for the scanner/batch covariate:
    batch_col = "batch"
    continuous_cols = ["age"]

    if add_age and add_gender:
        print("both age and gender are preserved")
        categorical_cols = ["gender", "label"]
        output = neuroCombat(
            dat=data,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
        )
    elif add_age:
        print("only age is preserved")
        categorical_cols = ["label"]
        output = neuroCombat(
            dat=data,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
        )
    elif add_gender:
        print("only gender is preserved")
        categorical_cols = ["gender", "label"]
        output = neuroCombat(
            dat=data,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
        )
    else:
        print("label is not preserved new")
        # categorical_cols = ["label"]
        output = neuroCombat(
            dat=data,
            covars=covars,
            batch_col=batch_col,
            # categorical_cols=categorical_cols,
        )

    est = output["estimates"]
    # print(output['data'][:, 0][:10])

    # sample = sample_files[0]
    # dat = np.empty((19900, 1))
    # dat[:, 0] = all_corr[sample][0]

    # bat = np.array([scanners[sample.split('_')[0]]])
    # print(bat)
    # print(dat.shape)
    # out = neuroCombatFromTraining(dat=dat, batch=bat, estimates=est)
    # print('data.shape: ', out['data'][:, 0].shape)
    # print('all_corr[sample][0]:', all_corr[sample][0].shape)
    # print('out[data].shape:', out['data'].shape)
    # print(dat[:10])

    # out = neuroCombatFromTraining1(dat=dat, batch=bat, estimates=est)
    # print('data.shape: ', out['data'][:, 0].shape)
    # print('all_corr[sample][0]:', all_corr[sample][0].shape)
    # print('out[data].shape:', out['data'].shape)
    # print(dat[:10])

    # return scanners and estmator
    return scanners, est


def getEstimator2(
    all_corr, sample_files, pheno_data, df_labels, add_gender=False, add_age=False
):
    print("getting estimator ...")
    # for testing
    # sample_files = sample_files[:100]

    scanners = {}
    id = 1
    # assign scanner ids
    for name in sample_files:
        center = name.split("_")[0]
        # if name.startswith('MaxMun') or name.startswith('CMU'):
        #   center = name.split('_')[0]
        # else:
        #   center = '_'.join(name.split('_')[:-1])

        if center not in scanners:
            scanners[center] = id
            id = id + 1
    print(scanners)

    # read data, add to matrix, assign scanner
    batch = []
    age = []
    gender = []
    labels = []
    data = np.empty((17955, len(sample_files)))

    for i, sample in enumerate(sample_files):
        # print(sample)
        data[:, i] = all_corr[sample][0]
        if add_age:
            age.append(
                list(pheno_data[pheno_data["FILE_ID"] == sample]["AGE_AT_SCAN"])[0]
            )
        if add_gender:
            gender.append(list(pheno_data[pheno_data["FILE_ID"] == sample]["SEX"])[0])
        labels.append(df_labels[sample.split("_")[-1]])
        center = sample.split("_")[0]
        # data.append(all_corr[sample][0])
        # center = '_'.join(sample.split('_')[:-1])
        # if sample.startswith('MaxMun') or sample.startswith('CMU'):
        #   center = sample.split('_')[0]
        # else:
        #   center = '_'.join(sample.split('_')[:-1])
        batch.append(scanners[center])

    data = np.array(data)
    batch = np.array(batch)
    age = np.array(age)
    gender = np.array(gender)
    labels = np.array(labels)

    if add_age and add_gender:
        covars = {"batch": batch, "gender": gender, "age": age, "label": labels}
    elif add_age:
        covars = {"batch": batch, "age": age, "label": labels}
    elif add_gender:
        covars = {"batch": batch, "gender": gender, "label": labels}
    else:
        covars = {"batch": batch, "label": labels}

    covars = pd.DataFrame(covars)
    # To specify the name of the variable that encodes for the scanner/batch covariate:
    batch_col = "batch"
    continuous_cols = ["age"]

    if add_age and add_gender:
        print("both age and gender are preserved")
        categorical_cols = ["gender", "label"]
        output = neuroCombat(
            dat=data,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
        )
    elif add_age:
        print("only age is preserved")
        categorical_cols = ["label"]
        output = neuroCombat(
            dat=data,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
        )
    elif add_gender:
        print("only gender is preserved")
        categorical_cols = ["gender", "label"]
        output = neuroCombat(
            dat=data,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
        )
    else:
        print("label is preserved")
        categorical_cols = ["label"]
        output = neuroCombat(
            dat=data,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
        )

    est = output["estimates"]
    # print(output['data'][:, 0][:10])

    # sample = sample_files[0]
    # dat = np.empty((19900, 1))
    # dat[:, 0] = all_corr[sample][0]

    # bat = np.array([scanners[sample.split('_')[0]]])
    # print(bat)
    # print(dat.shape)
    # out = neuroCombatFromTraining(dat=dat, batch=bat, estimates=est)
    # print('data.shape: ', out['data'][:, 0].shape)
    # print('all_corr[sample][0]:', all_corr[sample][0].shape)
    # print('out[data].shape:', out['data'].shape)
    # print(dat[:10])

    # out = neuroCombatFromTraining1(dat=dat, batch=bat, estimates=est)
    # print('data.shape: ', out['data'][:, 0].shape)
    # print('all_corr[sample][0]:', all_corr[sample][0].shape)
    # print('out[data].shape:', out['data'].shape)
    # print(dat[:10])

    # return scanners and estmator
    return scanners, est
