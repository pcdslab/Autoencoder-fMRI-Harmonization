import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import reduce
from sklearn.impute import SimpleImputer
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pyprind
import sys
import pickle
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn import tree
import functools
import numpy.ma as ma  # for masked arrays
import pyprind
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import argparse
import csv
from applyCombat import getEstimator, neuroCombatFromTraining1
from neuroCombat import neuroCombat, neuroCombatFromTraining
from numpy import savetxt
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from config import *

# Set a random seed value
seed_value = 42

# PyTorch
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU.

# Python's `random` module
random.seed(seed_value)

# NumPy
np.random.seed(seed_value)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str, default=None, help="path to samples file")
ap.add_argument("-c", "--centers", type=str, default=None, help="centers to run")
ap.add_argument("-o", "--output", type=str, default=None, help="path to results")
ap.add_argument("-t", "--ae_type", type=str, default="AE", help="AE to use")
ap.add_argument("-m", "--ml_method", type=str, default="RF", help="ML to use")
ap.add_argument(
    "-f",
    "--fold",
    type=int,
    default=10,
    help="number of fold in the samples file",
)
ap.add_argument(
    "-g", "--gender", type=bool, default=False, help="to add gender to ComBat"
)

ap.add_argument(
    "-u", "--run_combat", type=bool, default=False, help="to run ComBat or not"
)

ap.add_argument("-a", "--age", type=bool, default=False, help="to add age to ComBat")

ap.add_argument(
    "-s", "--p_method", type=str, default="ASD-Standalone-AE", help="method to run"
)

ap.add_argument(
    "--age_min",
    type=int,
    default=0,
    help="number of fold in the samples file",
)
ap.add_argument(
    "--age_max",
    type=int,
    default=100,
    help="number of fold in the samples file",
)


args = vars(ap.parse_args())

path = SAMPLE_PATH
output = args["output"]
p_fold = args["fold"]
centers = args["centers"]

ml_method = args["ml_method"]
ae_type = args["ae_type"]

add_age = args["age"]
add_gender = args["gender"]
age_min = args["age_min"]
age_max = args["age_max"]

centers = np.array(centers.split(","))

print("path: ", path)
print("output: ", output)
print("add_age: ", add_age)
print("add_gender: ", add_gender)
print("p_fold: ", p_fold)
print("centers: ", centers)

# options: cc200, dosenbach160, aal
p_ROI = "cc200"
p_center = "Stanford"
p_mode = "percenter"
p_augmentation = False

# Change this to run combat or not
run_combat = args["run_combat"]

# p_Method = "ASD-Standalone-AE"
# p_Method = "ASD-Standalone-AE-4-centers"
# p_Method = "ASD-ml"
# p_Method = "ASD-Standalone-AE-k-fold"
p_Method = args["p_method"]

parameter_list = [p_ROI, p_fold, p_center, p_mode, p_augmentation, p_Method]
print("*****List of patameters****")
print("ROI atlas: ", p_ROI)
print("per Center or whole: ", p_mode)
if p_mode == "percenter":
    print("Center's name: ", p_center)
print("Method's name: ", p_Method)
if p_Method == "ASD-DiagNet":
    print("Augmentation: ", p_augmentation)
print("run_combat:", run_combat)

all_corr = {}


def get_key(filename):
    f_split = filename.split("_")
    if f_split[3] == "rois":
        key = "_".join(f_split[0:3])
    else:
        key = "_".join(f_split[0:2])
    return key


flist = os.listdir(DATA_PATH)
print(len(flist))

for f in range(len(flist)):
    flist[f] = get_key(flist[f])


df_labels = pd.read_csv(
    PHENO_PATH
)  # path

df_labels.DX_GROUP = df_labels.DX_GROUP.map({1: 1, 2: 0})
df_labels['AGE_AT_SCAN'] = pd.to_numeric(df_labels['AGE_AT_SCAN'], errors='coerce').fillna(0).astype(int)
print(len(df_labels))

labels = {}
for row in df_labels.iterrows():
    file_id = row[1]["FILE_ID"]
    y_label = row[1]["DX_GROUP"]
    if file_id == "no_filename":
        continue
    assert file_id not in labels
    labels[file_id] = y_label


def get_label(filename):
    assert filename in labels
    return labels[filename]


def get_corr_data(filename):
    # print(filename)
    for file in os.listdir(DATA_PATH):
        if file.startswith(filename):
            df = pd.read_csv(os.path.join(DATA_PATH, file), sep="\t")

    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(df.T))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()


def get_corr_matrix(filename):
    for file in os.listdir(DATA_PATH):
        if file.startswith(filename):
            df = pd.read_csv(os.path.join(DATA_PATH, file), sep="\t")
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(df.T))
        return corr


def confusion(g_turth, predictions):
    # x = confusion_matrix(g_turth, predictions).ravel()
    # print("X:", x)

    tn, fp, fn, tp = confusion_matrix(g_turth, predictions).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    sensitivity = (tp) / (tp + fn)
    specificty = (tn) / (tn + fp)
    return accuracy, sensitivity, specificty


if not os.path.exists("./correlations_file" + p_ROI + ".pkl"):
    pbar = pyprind.ProgBar(len(flist))
    all_corr = {}
    for f in flist:
        lab = get_label(f)
        all_corr[f] = (get_corr_data(f), lab)
        pbar.update()

    print("Corr-computations finished")

    pickle.dump(all_corr, open("./correlations_file" + p_ROI + ".pkl", "wb"))
    print("Saving to file finished")

else:
    all_corr = pickle.load(open("./correlations_file" + p_ROI + ".pkl", "rb"))


def get_combat_data(data, sample, num_corr=None, scanners=None, estimator=None):
    dat = np.empty((num_corr, 1))
    dat[:, 0] = data
    center = sample.split("_")[0]
    # if sample.startswith('MaxMun') or sample.startswith('CMU'):
    #   center = sample.split('_')[0]
    # else:
    #   center = '_'.join(sample.split('_')[:-1])

    bat = np.array([scanners[center]])
    # print(bat)
    # print(dat.shape)
    out = neuroCombatFromTraining1(dat=dat, batch=bat, estimates=estimator)
    # print('data.shape: ', out['data'][:, 0].shape)
    return out["data"][:, 0]


def get_regs_new(samplesnames, regnum, num_corr=None, scanners=None, estimator=None):
    # print(samplesnames)
    # print("---------------------")
    # print(all_corr)
    # print("------------------")
    datas = []
    for sn in samplesnames:
        # print(sn)
        if estimator is not None:
            data = get_combat_data(all_corr[sn][0], sn, num_corr, scanners, estimator)
        else:
            data = all_corr[sn][0]
        datas.append(data)
    datas = np.array(datas)
    avg = []
    for ie in range(datas.shape[1]):
        avg.append(np.mean(datas[:, ie]))
    avg = np.array(avg)
    highs = avg.argsort()[-regnum:][::-1]
    lows = avg.argsort()[:regnum][::-1]
    regions = np.concatenate((highs, lows), axis=0)
    return regions


def get_regs(
    new_data, samplesnames, regnum, num_corr=None, scanners=None, estimator=None
):
    # print(samplesnames)
    # print("---------------------")
    # print(new_data)
    # print("------------------")
    datas = []
    for sn in samplesnames:
        # print(sn)
        if estimator is not None:
            data = get_combat_data(new_data[sn][0], sn, num_corr, scanners, estimator)
        else:
            data = new_data[sn][0]
        datas.append(data)
    datas = np.array(datas)
    avg = []
    for ie in range(datas.shape[1]):
        avg.append(np.mean(datas[:, ie]))
    avg = np.array(avg)
    highs = avg.argsort()[-regnum:][::-1]
    lows = avg.argsort()[:regnum][::-1]
    regions = np.concatenate((highs, lows), axis=0)
    return regions


## TODO: commomt this if using ComBat
# if p_Method == "ASD-DiagNet":
#     eig_data = {}
#     pbar = pyprind.ProgBar(len(flist))
#     for f in flist:
#         d = get_corr_matrix(f)
#         eig_vals, eig_vecs = np.linalg.eig(d)

#         for ev in eig_vecs.T:
#             np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

#         sum_eigvals = np.sum(np.abs(eig_vals))
#         # Make a list of (eigenvalue, eigenvector, norm_eigval) tuples
#         eig_pairs = [
#             (np.abs(eig_vals[i]), eig_vecs[:, i], np.abs(eig_vals[i]) / sum_eigvals)
#             for i in range(len(eig_vals))
#         ]

#         # Sort the (eigenvalue, eigenvector) tuples from high to low
#         eig_pairs.sort(key=lambda x: x[0], reverse=True)

#         eig_data[f] = {
#             "eigvals": np.array([ep[0] for ep in eig_pairs]),
#             "norm-eigvals": np.array([ep[2] for ep in eig_pairs]),
#             "eigvecs": [ep[1] for ep in eig_pairs],
#         }
#         pbar.update()


def norm_weights(sub_flist):
    num_dim = len(eig_data[flist[0]]["eigvals"])
    norm_weights = np.zeros(shape=num_dim)
    for f in sub_flist:
        norm_weights += eig_data[f]["norm-eigvals"]
    return norm_weights


def cal_similarity(d1, d2, weights, lim=None):
    res = 0.0
    if lim is None:
        weights_arr = weights.copy()
    else:
        weights_arr = weights[:lim].copy()
        weights_arr /= np.sum(weights_arr)
    for i, w in enumerate(weights_arr):
        res += w * np.inner(d1[i], d2[i])
    return res


class CC200Dataset(Dataset):
    def __init__(
        self,
        pkl_filename=None,
        data=None,
        samples_list=None,
        augmentation=False,
        aug_factor=1,
        num_neighbs=5,
        eig_data=None,
        similarity_fn=None,
        verbose=False,
        regs=None,
        num_corr=None,
        scanners=None,
        estimator=None,
    ):
        self.num_corr = num_corr
        self.scanners = scanners
        self.estimator = estimator
        self.regs = regs
        if pkl_filename is not None:
            if verbose:
                print("Loading ..!", end=" ")
            self.data = pickle.load(open(pkl_filename, "rb"))
        elif data is not None:
            self.data = data.copy()

        else:
            sys.stderr.write("Eigther PKL file or data is needed!")
            return

        # if verbose:
        #    print ('Preprocess..!', end='  ')
        if samples_list is None:
            self.flist = [f for f in self.data]
        else:
            self.flist = [f for f in samples_list]
        self.labels = np.array([self.data[f][1] for f in self.flist])

        current_flist = np.array(self.flist.copy())
        current_lab0_flist = current_flist[self.labels == 0]
        current_lab1_flist = current_flist[self.labels == 1]
        # if verbose:
        #    print(' Num Positive : ', len(current_lab1_flist), end=' ')
        #    print(' Num Negative : ', len(current_lab0_flist), end=' ')

        if augmentation:
            self.num_data = aug_factor * len(self.flist)
            self.neighbors = {}
            pbar = pyprind.ProgBar(len(self.flist))
            weights = norm_weights(samples_list)  # ??
            for f in self.flist:
                label = self.data[f][1]
                candidates = (
                    set(current_lab0_flist) if label == 0 else set(current_lab1_flist)
                )
                candidates.remove(f)
                eig_f = eig_data[f]["eigvecs"]
                sim_list = []
                for cand in candidates:
                    eig_cand = eig_data[cand]["eigvecs"]
                    sim = similarity_fn(eig_f, eig_cand, weights)
                    sim_list.append((sim, cand))
                sim_list.sort(key=lambda x: x[0], reverse=True)
                self.neighbors[f] = [
                    item[1] for item in sim_list[:num_neighbs]
                ]  # list(candidates)#[item[1] for item in sim_list[:num_neighbs]]

        else:
            self.num_data = len(self.flist)

    def __getitem__(self, index):
        if index < len(self.flist):
            fname = self.flist[index]
            data = self.data[fname][0].copy()  # get_corr_data(fname, mode=cal_mode)
            if self.estimator is not None:
                data = get_combat_data(
                    data, fname, self.num_corr, self.scanners, self.estimator
                )
            data = data[self.regs].copy()
            label = (self.labels[index],)
            return torch.FloatTensor(data), torch.FloatTensor(label)
        else:
            f1 = self.flist[index % len(self.flist)]
            d1, y1 = self.data[f1][0], self.data[f1][1]
            if self.estimator is not None:
                d1 = get_combat_data(
                    d1, f1, self.num_corr, self.scanners, self.estimator
                )
            d1 = d1[self.regs]
            f2 = np.random.choice(self.neighbors[f1])
            d2, y2 = self.data[f2][0], self.data[f2][1]
            if self.estimator is not None:
                d2 = get_combat_data(
                    d2, f2, self.num_corr, self.scanners, self.estimator
                )
            d2 = d2[self.regs]
            assert y1 == y2
            r = np.random.uniform(low=0, high=1)
            label = (y1,)
            data = r * d1 + (1 - r) * d2
            return torch.FloatTensor(data), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data


def get_loader(
    pkl_filename=None,
    data=None,
    samples_list=None,
    batch_size=64,
    num_workers=1,
    mode="train",
    *,
    augmentation=False,
    aug_factor=1,
    num_neighbs=5,
    eig_data=None,
    similarity_fn=None,
    verbose=False,
    regions=None,
    num_corr=None,
    scanners=None,
    estimator=None,
):
    """Build and return data loader."""
    if mode == "train":
        shuffle = True
    else:
        shuffle = False
        augmentation = False

    dataset = CC200Dataset(
        pkl_filename=pkl_filename,
        data=data,
        samples_list=samples_list,
        augmentation=augmentation,
        aug_factor=aug_factor,
        eig_data=eig_data,
        similarity_fn=similarity_fn,
        verbose=verbose,
        regs=regions,
        num_corr=num_corr,
        scanners=scanners,
        estimator=estimator,
    )

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader


class MTAutoEncoder(nn.Module):
    def __init__(
        self,
        num_inputs=990,
        num_latent=200,
        tied=True,
        num_classes=2,
        use_dropout=False,
    ):
        super(MTAutoEncoder, self).__init__()
        self.tied = tied
        self.num_latent = num_latent

        self.fc_encoder = nn.Linear(num_inputs, num_latent)

        if not tied:
            self.fc_decoder = nn.Linear(num_latent, num_inputs)

        self.fc_encoder = nn.Linear(num_inputs, num_latent)

        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.num_latent, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.num_latent, 1),
            )

    def forward(self, x, eval_classifier=False):
        x = self.fc_encoder(x)
        x = torch.tanh(x)
        if eval_classifier:
            x_logit = self.classifier(x)
        else:
            x_logit = None

        if self.tied:
            x = F.linear(x, self.fc_encoder.weight.t())
        else:
            x = self.fc_decoder(x)

        return x, x_logit


class AE(nn.Module):
    def __init__(
        self,
        num_inputs=990,
        num_latent=200,
        tied=True,
    ):
        super(AE, self).__init__()
        self.tied = tied
        self.num_latent = num_latent

        self.fc_encoder = nn.Linear(num_inputs, num_latent)

        if not tied:
            self.fc_decoder = nn.Linear(num_latent, num_inputs)

        # self.fc_encoder = nn.Linear(num_inputs, num_latent)

    def forward(self, x):
        bottleneck = self.fc_encoder(x)
        bottleneck = torch.tanh(bottleneck)

        if self.tied:
            x = F.linear(bottleneck, self.fc_encoder.weight.t())
        else:
            x = self.fc_decoder(bottleneck)
        return bottleneck, x


class SAE(nn.Module):
    def __init__(self, n_features, n_lat):
        super().__init__()
        self.encoder = nn.Linear(n_features, n_lat)
        self.decoder = nn.Linear(n_lat, n_features)

    def forward(self, x):
        bottleneck = self.encoder(x)
        bottleneck = F.relu(bottleneck)
        x = self.decoder(bottleneck)
        return bottleneck, x


class DAE(nn.Module):
    def __init__(self, input_size):
        super(DAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, input_size),
            nn.Tanh(),
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        return bottleneck, x


class VAE(nn.Module):
    def __init__(self, input_size=19900, latent_size=9950, hidden_size=1000):
        super(VAE, self).__init__()
        self.input_size = input_size
        # Encoder
        self.fc1 = nn.Linear(
            input_size, hidden_size
        )  # Adjusted for variable hidden layer size
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(
            hidden_size, input_size
        )  # Adjusted for variable hidden layer size

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 1)
    rho = torch.tensor([rho] * len(rho_hat)).to(device)
    return torch.sum(
        rho * torch.log(rho / rho_hat)
        + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    )


def loss_function_sae(model, images):
    model_children = list(model.children())
    loss = 0
    p = 0.05
    beta = 2
    values = images
    for i in range(len(model_children)):
        values = model_children[i](values)
        loss += kl_divergence(p, values)
    return beta * loss


# Loss function
def loss_function_vae(recon_x, x, mu, logvar, input_size=19900):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_size), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_ae(model, epochs, train_loader, ae_type):
    for epoch in range(epochs):
        # print("epoch:", epoch)
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            train_loss = 0
            pass
        for i, (batch_x, batch_y) in enumerate(train_loader):
            if len(batch_x) != batch_size:
                continue
            data, target = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            # AE
            if ae_type == "AE" or ae_type == "TAE":
                bot, rec = model(data)
                loss_total = criterion_ae(rec, data)
            # SAE
            if ae_type == "SAE":
                bot, rec = model(data)
                loss_ae = criterion_ae(rec, data)
                loss_sae = loss_function_sae(model, data)
                loss_total = loss_ae + loss_sae
            # VAE
            if ae_type == "VAE":
                bot, recon_batch, mu, logvar = model(data)
                loss_total = loss_function_vae(recon_batch, data, mu, logvar)
            # DAE
            if ae_type == "DAE":
                # Add noise to the inputs
                noisy_inputs = data + torch.randn_like(data) * 0.5
                bot, outputs = model(noisy_inputs)
                loss_total = criterion_ae(outputs, data)

            loss_total.backward()
            optimizer.step()
    return model


def train(
    model,
    optimizer,
    batch_size,
    criterion_ae,
    criterion_clf,
    epoch,
    train_loader,
    p_bernoulli=None,
    mode="both",
    lam_factor=1.0,
):
    # def train(model, epoch, train_loader, p_bernoulli=None, mode="both", lam_factor=1.0):
    model.train()
    train_losses = []
    for i, (batch_x, batch_y) in enumerate(train_loader):
        if len(batch_x) != batch_size:
            continue
        if p_bernoulli is not None:
            if i == 0:
                p_tensor = torch.ones_like(batch_x).to(device) * p_bernoulli
            rand_bernoulli = torch.bernoulli(p_tensor).to(device)

        data, target = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()

        if mode in ["both", "ae"]:
            if p_bernoulli is not None:
                rec_noisy, _ = model(data * rand_bernoulli, False)
                loss_ae = criterion_ae(rec_noisy, data) / len(batch_x)
            else:
                rec, _ = model(data, False)
                loss_ae = criterion_ae(rec, data) / len(batch_x)

        if mode in ["both", "clf"]:
            rec_clean, logits = model(data, True)
            loss_clf = criterion_clf(logits, target)

        if mode == "both":
            loss_total = loss_ae + lam_factor * loss_clf
            train_losses.append(
                [loss_ae.detach().cpu().numpy(), loss_clf.detach().cpu().numpy()]
            )
        elif mode == "ae":
            loss_total = loss_ae
            train_losses.append([loss_ae.detach().cpu().numpy(), 0.0])
        elif mode == "clf":
            loss_total = loss_clf
            train_losses.append([0.0, loss_clf.detach().cpu().numpy()])

        loss_total.backward()
        optimizer.step()

    return train_losses


def test(model, criterion, test_loader, eval_classifier=False, num_batch=None):
    test_loss, n_test, correct = 0.0, 0, 0
    all_predss = []
    if eval_classifier:
        y_true, y_pred = [], []
    with torch.no_grad():
        model.eval()
        for i, (batch_x, batch_y) in enumerate(test_loader, 1):
            if num_batch is not None:
                if i >= num_batch:
                    continue
            data = batch_x.to(device)
            rec, logits = model(data, eval_classifier)

            test_loss += criterion(rec, data).detach().cpu().numpy()
            n_test += len(batch_x)
            if eval_classifier:
                proba = torch.sigmoid(logits).detach().cpu().numpy()
                preds = np.ones_like(proba, dtype=np.int32)
                preds[proba < 0.5] = 0
                all_predss.extend(preds)  ###????
                y_arr = np.array(batch_y, dtype=np.int32)

                correct += np.sum(preds == y_arr)
                y_true.extend(y_arr.tolist())
                y_pred.extend(proba.tolist())
        mlp_acc, mlp_sens, mlp_spef = confusion(y_true, all_predss)

    return mlp_acc, mlp_sens, mlp_spef  # ,correct/n_test


def update_data(all_corr, num_corr, scanners, estimator, samples=None):
    print("updating data ...")
    new_data = all_corr.copy()
    for sample in new_data.keys():
        if samples is not None and sample not in samples:
            continue
        data = get_combat_data(
            new_data[sample][0], sample, num_corr, scanners, estimator
        )
        label = new_data[sample][1]
        new_data[sample] = (data, label)
    print("finished updating!")
    return new_data


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        # target = torch.LongTensor(target)
        loss = (torch.log(((output - target) ** 2))).mean()
        # loss = (torch.lgamma((1 - (input-target)**2))).mean()
        return loss



if p_Method == "ASD-Standalone-AE" and p_mode == "whole":
    # ml_method = "RF"

    filter_regions = False
    eig_data = None
    # output_dir = "leave_site_out_experiments_standalone_ae/no_combat/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # output_dir = "four_centers_experiments_standalone_ae/no_combat/" + ml_method + "/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    output_dir = (
        "all_leave_site_out_experiments_standalone_ae_bottleneck/no_combat/"
        + ml_method
        + "/"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # output_dir = "combat_experiments_new/no_combat/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    num_corr = len(all_corr[flist[0]][0])
    print("num_corr:  ", num_corr)

    start = time.time()
    batch_size = 8
    learning_rate_ae, learning_rate_clf = 0.0001, 0.0001

    p_bernoulli = None
    augmentation = p_augmentation
    use_dropout = False

    aug_factor = 2
    num_neighbs = 5
    lim4sim = 2
    n_lat = int(num_corr / 4)
    print(n_lat)
    start = time.time()

    print("p_bernoulli: ", p_bernoulli)
    print(
        "augmentaiton: ",
        augmentation,
        "aug_factor: ",
        aug_factor,
        "num_neighbs: ",
        num_neighbs,
        "lim4sim: ",
        lim4sim,
    )
    print("use_dropout: ", use_dropout, "\n")
    print("filter_regions:", filter_regions)

    sim_function = functools.partial(cal_similarity, lim=lim4sim)

    samples = pd.read_csv(path)

    # ******** Here we change AE type *****#
    # ae_type = "AE"
    num_epochs = 25
    print("num_epochs:", num_epochs)

    file_name = (
        "leave_site_out_"
        + "_standalone_ae_bottleneck_type_"
        + ae_type
        + "_filter_regions_"
        + str(filter_regions)
        + "_epochs_"
        + str(num_epochs)
        + "_method_"
        + ml_method
        + ".csv"
    )
    print("results will be at:", output_dir + file_name)

    results = open(output_dir + file_name, "a")
    writer = csv.writer(results)
    fieldnames = [
        "iter",
        "ae_type",
        "filter_regions",
        "ml_method",
        "site",
        "train_size",
        "test_size",
        "acc",
        "sens",
        "sepf",
    ]
    writer = csv.DictWriter(results, fieldnames=fieldnames)
    writer.writeheader()

    crossval_res_kol = []

    # sites = ["NYU", "UCLA", "UM", "USM"]
    sites = [
        "NYU",
        "UCLA",
        "UM",
        "USM",
        "Caltech",
        "CMU",
        "KKI",
        "Leuven",
        "MaxMun",
        "OHSU",
        "Olin",
        "Pitt",
        "SBL",
        "SDSU",
        "Stanford",
        "Trinity",
        "Yale",
    ]
    overall_result = []

    for itr in range(10):
        res = []
        for site in sites:
            print("Run site:", site)
            new_sites = [item for item in sites if item != site]
            train_samples = [file for file in flist if file.split("_")[0] in new_sites]
            test_samples = [file for file in flist if file.split("_")[0] in [site]]
            train_samples = np.array(train_samples)
            test_samples = np.array(test_samples)
            print("len(train_samples):", len(train_samples))
            print("len(test_samples):", len(test_samples))

            # TODO: if we run all data all together and need to split data equaly
            # for i in range(p_fold):
            #     x = f"fold_{i}_train"
            #     y = f"fold_{i}_test"

            #     train_samples = samples[samples[x] == 1]["subject"].to_numpy()
            #     test_samples = samples[samples[y] == 1]["subject"].to_numpy()

            if filter_regions:
                regions_inds = get_regs(all_corr, train_samples, int(num_corr / 4))
            else:
                regions_inds = np.array([i for i in range(num_corr)])

            estimator = None
            scanners = None
            new_data = all_corr

            verbose = True

            # regions_inds = get_regs(
            #     train_samples, int(num_corr / 4), num_corr, scanners, estimator
            # )
            # regions_inds = get_regs(new_data, train_samples, int(num_corr / 4))

            # # TODO: save regions_inds
            # savetxt(
            #     f"{output_dir}regions_inds_iter_{itr}_fold{i}.csv",
            #     regions_inds,
            #     delimiter=",",
            # )
            print(len(regions_inds))
            # new_data = all_corr
            scanners = None
            # continue

            num_inpp = len(regions_inds)
            n_lat = int(num_inpp / 2)

            print("num_inpp:", num_inpp)
            print("n_lat:", n_lat)

            train_loader = get_loader(
                data=new_data,
                samples_list=train_samples,
                batch_size=batch_size,
                mode="train",
                augmentation=False,
                aug_factor=aug_factor,
                num_neighbs=num_neighbs,
                eig_data=eig_data,
                similarity_fn=sim_function,
                verbose=verbose,
                regions=regions_inds,
                num_corr=num_corr,
                scanners=scanners,
                estimator=None,
            )

            test_loader = get_loader(
                data=new_data,
                samples_list=test_samples,
                batch_size=batch_size,
                mode="test",
                augmentation=False,
                verbose=verbose,
                regions=regions_inds,
                num_corr=num_corr,
                scanners=scanners,
                estimator=None,
            )

            if ae_type == "AE":
                model = AE(
                    tied=False,
                    num_inputs=num_inpp,
                    num_latent=n_lat,
                )
            elif ae_type == "TAE":
                model = AE(
                    tied=True,
                    num_inputs=num_inpp,
                    num_latent=n_lat,
                )
            elif ae_type == "SAE":
                model = SAE(
                    n_features=num_inpp,
                    n_lat=n_lat,
                )
            elif ae_type == "VAE":
                model = VAE(input_size=num_inpp, latent_size=n_lat, hidden_size=1000)
            else:
                model = DAE(input_size=num_inpp)
                # model = VAE(input_size=num_inpp, latent_size=n_lat, hidden_size=1000)

            model.to(device)
            criterion_ae = nn.MSELoss(reduction="sum")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_ae)
            # optimizer = optim.SGD(
            #     [
            #         {"params": model.fc_encoder.parameters(), "lr": learning_rate_ae},
            #         {"params": model.classifier.parameters(), "lr": learning_rate_clf},
            #     ],
            #     momentum=0.9,
            # )

            trained_model = train_ae(model, num_epochs, train_loader, ae_type)

            print("trained model done ")
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []

            # loop through train loader and append the data after we pass it to the model
            trained_model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    if len(batch_x) != batch_size:
                        continue
                    data, target = batch_x.to(device), batch_y.to(device)
                    if ae_type == "VAE":
                        output, rec, _, _ = trained_model(data)
                    else:
                        output, rec = trained_model(data)
                    train_data.extend(output.detach().cpu().numpy().tolist())
                    train_labels.extend(target.detach().cpu().numpy().tolist())
            # print("train loader done")
            # loop through test loader and append the data after we pass it to the model
            trained_model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    if len(batch_x) != batch_size:
                        continue
                    data, target = batch_x.to(device), batch_y.to(device)
                    if ae_type == "VAE":
                        output, rec, _, _ = trained_model(data)
                    else:
                        output, rec = trained_model(data)
                    test_data.extend(output.detach().cpu().numpy().tolist())
                    test_labels.extend(target.detach().cpu().numpy().tolist())
            # run ML model
            # print("train loader done")

            # print("here .... 1")
            train_data = np.array(train_data)
            train_labels = np.array(train_labels).ravel()
            test_data = np.array(test_data)
            test_labels = np.array(test_labels).ravel()
            # print("here .... 2")
            # print("len(train_data):", len(train_data))
            # print("len(train_labels):", len(train_labels))
            # print("len(test_data):", len(test_data))
            # print("len(test_labels):", len(test_labels))
            # ML_method = "RF"

            if ml_method == "NB":
                clf = GaussianNB()
            if ml_method == "SVM":
                # Apply PCA
                # You can adjust n_components based on your requirement
                pca = PCA(n_components=2)
                train_data = pca.fit_transform(train_data)
                test_data = pca.transform(test_data)
                clf = SVC(gamma="auto")
            if ml_method == "RF":
                clf = RandomForestClassifier(n_estimators=100)
            # clf = (
            #     SVC(gamma="auto")
            #     if ML_method == "SVM"
            #     else RandomForestClassifier(n_estimators=100)
            # )

            clf.fit(train_data, train_labels)
            pr = clf.predict(test_data)
            print("site:", site)

            accuracy, sensitivity, specificty = confusion(test_labels, pr)

            print(accuracy, sensitivity, specificty)
            res.append(confusion(test_labels, pr))
            writer.writerow(
                {
                    "iter": itr,
                    "ae_type": ae_type,
                    "filter_regions": filter_regions,
                    "ml_method": ml_method,
                    "site": site,
                    "train_size": len(train_samples),
                    "test_size": len(test_samples),
                    "acc": accuracy,
                    "sens": sensitivity,
                    "sepf": specificty,
                }
            )
            # break
        print("repeat: ", itr, np.mean(res, axis=0).tolist())
        overall_result.append(np.mean(res, axis=0).tolist())
        # break
    print("---------------Result of repeating 10 times-------------------")
    print(np.mean(np.array(overall_result), axis=0).tolist())
    results.close()
    sys.exit(0)

# Leave site out for 4 centers
# ***** to do the 10 fold for all sites using AEs
if p_Method == "ASD-Standalone-AE-4-centers" and p_mode == "whole":
    filter_regions = False
    eig_data = None
    # output_dir = "leave_site_out_experiments_standalone_ae/no_combat/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    output_dir = (
        "four_centers_leave_site_out_experiments_standalone_ae_bottleneck/no_combat/"
        + ml_method
        + "/"
    )

    # output_dir = (
    #     "four_centers_leave_site_out_experiments_standalone_ae_bottleneck/no_combat/"
    # )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # output_dir = "combat_experiments_new/no_combat/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    num_corr = len(all_corr[flist[0]][0])
    print("num_corr:  ", num_corr)

    start = time.time()
    batch_size = 8
    learning_rate_ae, learning_rate_clf = 0.0001, 0.0001
    num_epochs = 25
    print("num_epochs:", num_epochs)

    p_bernoulli = None
    augmentation = p_augmentation
    use_dropout = False

    aug_factor = 2
    num_neighbs = 5
    lim4sim = 2
    n_lat = int(num_corr / 4)
    print(n_lat)
    start = time.time()

    print("p_bernoulli: ", p_bernoulli)
    print(
        "augmentaiton: ",
        augmentation,
        "aug_factor: ",
        aug_factor,
        "num_neighbs: ",
        num_neighbs,
        "lim4sim: ",
        lim4sim,
    )
    print("use_dropout: ", use_dropout, "\n")
    print("filter_regions:", filter_regions)

    sim_function = functools.partial(cal_similarity, lim=lim4sim)

    samples = pd.read_csv(path)

    # ******** Here we change AE type *****#
    # ae_type = "VAE"

    file_name = (
        "leave_site_out_"
        + "_standalone_ae_bottleneck_type_"
        + ae_type
        + "_filter_regions_"
        + str(filter_regions)
        + "_epochs_"
        + str(num_epochs)
        + "_method_"
        + ml_method
        + ".csv"
    )
    # file_name = (
    #     "leave_site_out_"
    #     + "_standalone_ae_bottleneck_type_"
    #     + ae_type
    #     + "_filter_regions_"
    #     + str(filter_regions)
    #     + "_epochs_"
    #     + str(num_epochs)
    #     + ".csv"
    # )
    print("results will be at:", output_dir + file_name)

    results = open(output_dir + file_name, "a")
    writer = csv.writer(results)
    fieldnames = [
        "iter",
        "ae_type",
        "filter_regions",
        "ml_method",
        "site",
        "train_size",
        "test_size",
        "acc",
        "sens",
        "sepf",
    ]
    writer = csv.DictWriter(results, fieldnames=fieldnames)
    writer.writeheader()

    crossval_res_kol = []

    sites = ["NYU", "UCLA", "UM", "USM"]
    # sites = [
    #     "NYU",
    #     "UCLA",
    #     "UM",
    #     "USM",
    #     "Caltech",
    #     "CMU",
    #     "KKI",
    #     "Leuven",
    #     "MaxMun",
    #     "OHSU",
    #     "Olin",
    #     "Pitt",
    #     "SBL",
    #     "SDSU",
    #     "Stanford",
    #     "Trinity",
    #     "Yale",
    # ]
    overall_result = []

    for itr in range(10):
        res = []
        site = "4centers_leave_site_out"
        for site in sites:
            print("Run site:", site)
            new_sites = [item for item in sites if item != site]
            train_samples = [file for file in flist if file.split("_")[0] in new_sites]
            test_samples = [file for file in flist if file.split("_")[0] in [site]]
            train_samples = np.array(train_samples)
            test_samples = np.array(test_samples)
            print("len(train_samples):", len(train_samples))
            print("len(test_samples):", len(test_samples))

            # TODO: if we run all data all together and need to split data equaly
            # for i in range(10):
            #     x = f"fold_{i}_train"
            #     y = f"fold_{i}_test"

            #     train_samples = samples[samples[x] == 1]["subject"].to_numpy()
            #     test_samples = samples[samples[y] == 1]["subject"].to_numpy()

            #     train_samples = [
            #         file for file in train_samples if file.split("_")[0] in sites
            #     ]
            #     test_samples = [
            #         file for file in test_samples if file.split("_")[0] in sites
            #     ]
            #     train_samples = np.array(train_samples)
            #     test_samples = np.array(test_samples)
            if filter_regions:
                regions_inds = get_regs(all_corr, train_samples, int(num_corr / 4))
            else:
                regions_inds = np.array([i for i in range(num_corr)])

            estimator = None
            scanners = None
            new_data = all_corr

            verbose = True

            # regions_inds = get_regs(
            #     train_samples, int(num_corr / 4), num_corr, scanners, estimator
            # )
            # regions_inds = get_regs(new_data, train_samples, int(num_corr / 4))

            # # TODO: save regions_inds
            # savetxt(
            #     f"{output_dir}regions_inds_iter_{itr}_fold{i}.csv",
            #     regions_inds,
            #     delimiter=",",
            # )
            print(len(regions_inds))
            # new_data = all_corr
            scanners = None
            # continue

            num_inpp = len(regions_inds)
            n_lat = int(num_inpp / 2)

            print("num_inpp:", num_inpp)
            print("n_lat:", n_lat)

            train_loader = get_loader(
                data=new_data,
                samples_list=train_samples,
                batch_size=batch_size,
                mode="train",
                augmentation=False,
                aug_factor=aug_factor,
                num_neighbs=num_neighbs,
                eig_data=eig_data,
                similarity_fn=sim_function,
                verbose=verbose,
                regions=regions_inds,
                num_corr=num_corr,
                scanners=scanners,
                estimator=None,
            )

            test_loader = get_loader(
                data=new_data,
                samples_list=test_samples,
                batch_size=batch_size,
                mode="test",
                augmentation=False,
                verbose=verbose,
                regions=regions_inds,
                num_corr=num_corr,
                scanners=scanners,
                estimator=None,
            )

            if ae_type == "AE":
                model = AE(
                    tied=False,
                    num_inputs=num_inpp,
                    num_latent=n_lat,
                )
            elif ae_type == "TAE":
                model = AE(
                    tied=True,
                    num_inputs=num_inpp,
                    num_latent=n_lat,
                )
            elif ae_type == "SAE":
                model = SAE(
                    n_features=num_inpp,
                    n_lat=n_lat,
                )
            elif ae_type == "VAE":
                model = VAE(input_size=num_inpp, latent_size=n_lat, hidden_size=1000)
            else:
                model = DAE(input_size=num_inpp)

            model.to(device)
            criterion_ae = nn.MSELoss(reduction="sum")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_ae)
            # optimizer = optim.SGD(
            #     [
            #         {"params": model.fc_encoder.parameters(), "lr": learning_rate_ae},
            #         {"params": model.classifier.parameters(), "lr": learning_rate_clf},
            #     ],
            #     momentum=0.9,
            # )

            trained_model = train_ae(model, num_epochs, train_loader, ae_type)

            print("trained model done ")
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []

            # loop through train loader and append the data after we pass it to the model
            trained_model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    if len(batch_x) != batch_size:
                        continue
                    data, target = batch_x.to(device), batch_y.to(device)
                    if ae_type == "VAE":
                        output, rec, _, _ = trained_model(data)
                    else:
                        output, rec = trained_model(data)
                    train_data.extend(output.detach().cpu().numpy().tolist())
                    train_labels.extend(target.detach().cpu().numpy().tolist())
            # print("train loader done")
            # loop through test loader and append the data after we pass it to the model
            trained_model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    if len(batch_x) != batch_size:
                        continue
                    data, target = batch_x.to(device), batch_y.to(device)
                    if ae_type == "VAE":
                        output, rec, _, _ = trained_model(data)
                    else:
                        output, rec = trained_model(data)
                    test_data.extend(output.detach().cpu().numpy().tolist())
                    test_labels.extend(target.detach().cpu().numpy().tolist())
            # run ML model
            # print("train loader done")

            # print("here .... 1")
            train_data = np.array(train_data)
            train_labels = np.array(train_labels).ravel()
            test_data = np.array(test_data)
            test_labels = np.array(test_labels).ravel()
            # print("here .... 2")
            # print("len(train_data):", len(train_data))
            # print("len(train_labels):", len(train_labels))
            # print("len(test_data):", len(test_data))
            # print("len(test_labels):", len(test_labels))
            if ml_method == "NB":
                clf = GaussianNB()
            if ml_method == "SVM":
                # Apply PCA
                # You can adjust n_components based on your requirement
                pca = PCA(n_components=2)
                train_data = pca.fit_transform(train_data)
                test_data = pca.transform(test_data)
                clf = SVC(gamma="auto")
            if ml_method == "RF":
                clf = RandomForestClassifier(n_estimators=100)

            clf.fit(train_data, train_labels)
            pr = clf.predict(test_data)
            print("site:", site)

            accuracy, sensitivity, specificty = confusion(test_labels, pr)

            print(accuracy, sensitivity, specificty)
            res.append(confusion(test_labels, pr))
            writer.writerow(
                {
                    "iter": itr,
                    "ae_type": ae_type,
                    "filter_regions": filter_regions,
                    "ml_method": ml_method,
                    "site": site,
                    "train_size": len(train_samples),
                    "test_size": len(test_samples),
                    "acc": accuracy,
                    "sens": sensitivity,
                    "sepf": specificty,
                }
            )
            # break
        print("repeat: ", itr, np.mean(res, axis=0).tolist())
        overall_result.append(np.mean(res, axis=0).tolist())
        # break
    print("---------------Result of repeating 10 times-------------------")
    print(np.mean(np.array(overall_result), axis=0).tolist())
    results.close()
    sys.exit(0)



if p_Method == "ASD-Standalone-AE-10-fold" and p_mode == "whole":
    filter_regions = False
    eig_data = None
    # output_dir = "leave_site_out_experiments_standalone_ae/no_combat/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    output_dir = (
        "all_centers_experiments_standalone_ae_bottleneck/no_combat/" + ml_method + "/"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # output_dir = "all_centers_experiments_standalone_ae_bottleneck/no_combat/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # output_dir = "all_centers_experiments_standalone_ae/no_combat/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    site = "all_centers_10_fold"
    print("site:", site)
    # output_dir = "combat_experiments_new/no_combat/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    num_corr = len(all_corr[flist[0]][0])
    print("num_corr:  ", num_corr)

    start = time.time()
    batch_size = 8
    learning_rate_ae, learning_rate_clf = 0.0001, 0.0001
    num_epochs = 25
    print("num_epochs:", num_epochs)

    p_bernoulli = None
    augmentation = p_augmentation
    use_dropout = False

    aug_factor = 2
    num_neighbs = 5
    lim4sim = 2
    n_lat = int(num_corr / 4)
    print(n_lat)
    start = time.time()

    print("p_bernoulli: ", p_bernoulli)
    print(
        "augmentaiton: ",
        augmentation,
        "aug_factor: ",
        aug_factor,
        "num_neighbs: ",
        num_neighbs,
        "lim4sim: ",
        lim4sim,
    )
    print("use_dropout: ", use_dropout, "\n")
    print("filter_regions:", filter_regions)

    sim_function = functools.partial(cal_similarity, lim=lim4sim)

    samples = pd.read_csv(path)

    # ******** Here we change AE type *****#
    # ae_type = "DAE"
    # ml_method = "RF"
    file_name = (
        "all_centers_10_fold_"
        + "standalone_ae_type_"
        + ae_type
        + "_filter_regions_"
        + str(filter_regions)
        + "_epochs_"
        + str(num_epochs)
        + "_method_"
        + ml_method
        + "_sute_"
        + site
        + ".csv"
    )
    print("results will be at:", output_dir + file_name)

    print("ml_method:", ml_method)
    print("ae_type:", ae_type)

    results = open(output_dir + file_name, "a")
    writer = csv.writer(results)
    fieldnames = [
        "iter",
        "ae_type",
        "filter_regions",
        "ml_method",
        "site",
        "train_size",
        "test_size",
        "acc",
        "sens",
        "sepf",
    ]
    writer = csv.DictWriter(results, fieldnames=fieldnames)
    writer.writeheader()

    crossval_res_kol = []

    # sites = ["NYU", "UCLA", "UM", "USM"]
    sites = [
        "NYU",
        "UCLA",
        "UM",
        "USM",
        "Caltech",
        "CMU",
        "KKI",
        "Leuven",
        "MaxMun",
        "OHSU",
        "Olin",
        "Pitt",
        "SBL",
        "SDSU",
        "Stanford",
        "Trinity",
        "Yale",
    ]
    overall_result = []

    for itr in range(10):
        res = []
        # for site in sites:
        #     print("Run site:", site)
        #     new_sites = [item for item in sites if item != site]
        #     train_samples = [file for file in flist if file.split("_")[0] in new_sites]
        #     test_samples = [file for file in flist if file.split("_")[0] in [site]]
        #     train_samples = np.array(train_samples)
        #     test_samples = np.array(test_samples)
        #     print("len(train_samples):", len(train_samples))
        #     print("len(test_samples):", len(test_samples))

        # TODO: if we run all data all together and need to split data equaly
        for i in range(10):
            x = f"fold_{i}_train"
            y = f"fold_{i}_test"

            train_samples = samples[samples[x] == 1]["subject"].to_numpy()
            test_samples = samples[samples[y] == 1]["subject"].to_numpy()

            train_samples = [
                file for file in train_samples if file.split("_")[0] in sites
            ]
            test_samples = [
                file for file in test_samples if file.split("_")[0] in sites
            ]
            train_samples = np.array(train_samples)
            test_samples = np.array(test_samples)
            if filter_regions:
                regions_inds = get_regs(all_corr, train_samples, int(num_corr / 4))
            else:
                regions_inds = np.array([i for i in range(num_corr)])

            estimator = None
            scanners = None
            new_data = all_corr

            verbose = True

            # regions_inds = get_regs(
            #     train_samples, int(num_corr / 4), num_corr, scanners, estimator
            # )
            # regions_inds = get_regs(new_data, train_samples, int(num_corr / 4))

            # # TODO: save regions_inds
            # savetxt(
            #     f"{output_dir}regions_inds_iter_{itr}_fold{i}.csv",
            #     regions_inds,
            #     delimiter=",",
            # )
            print(len(regions_inds))
            # new_data = all_corr
            scanners = None
            # continue

            num_inpp = len(regions_inds)
            n_lat = int(num_inpp / 2)

            print("num_inpp:", num_inpp)
            print("n_lat:", n_lat)

            train_loader = get_loader(
                data=new_data,
                samples_list=train_samples,
                batch_size=batch_size,
                mode="train",
                augmentation=False,
                aug_factor=aug_factor,
                num_neighbs=num_neighbs,
                eig_data=eig_data,
                similarity_fn=sim_function,
                verbose=verbose,
                regions=regions_inds,
                num_corr=num_corr,
                scanners=scanners,
                estimator=None,
            )

            test_loader = get_loader(
                data=new_data,
                samples_list=test_samples,
                batch_size=batch_size,
                mode="test",
                augmentation=False,
                verbose=verbose,
                regions=regions_inds,
                num_corr=num_corr,
                scanners=scanners,
                estimator=None,
            )

            if ae_type == "AE":
                model = AE(
                    tied=False,
                    num_inputs=num_inpp,
                    num_latent=n_lat,
                )
            elif ae_type == "TAE":
                model = AE(
                    tied=True,
                    num_inputs=num_inpp,
                    num_latent=n_lat,
                )
            elif ae_type == "SAE":
                model = SAE(
                    n_features=num_inpp,
                    n_lat=n_lat,
                )
            elif ae_type == "VAE":
                model = VAE(input_size=num_inpp, latent_size=n_lat, hidden_size=1000)
            else:
                model = DAE(input_size=num_inpp)

            model.to(device)
            criterion_ae = nn.MSELoss(reduction="sum")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_ae)
            # optimizer = optim.SGD(
            #     [
            #         {"params": model.fc_encoder.parameters(), "lr": learning_rate_ae},
            #         {"params": model.classifier.parameters(), "lr": learning_rate_clf},
            #     ],
            #     momentum=0.9,
            # )

            trained_model = train_ae(model, num_epochs, train_loader, ae_type)

            print("trained model done ")
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []

            # loop through train loader and append the data after we pass it to the model
            trained_model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    if len(batch_x) != batch_size:
                        continue
                    data, target = batch_x.to(device), batch_y.to(device)
                    if ae_type == "VAE":
                        output, rec, _, _ = trained_model(data)
                    else:
                        output, rec = trained_model(data)
                    train_data.extend(output.detach().cpu().numpy().tolist())
                    train_labels.extend(target.detach().cpu().numpy().tolist())
            # print("train loader done")
            # loop through test loader and append the data after we pass it to the model
            trained_model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    if len(batch_x) != batch_size:
                        continue
                    data, target = batch_x.to(device), batch_y.to(device)
                    if ae_type == "VAE":
                        output, rec, _, _ = trained_model(data)
                    else:
                        output, rec = trained_model(data)
                    test_data.extend(output.detach().cpu().numpy().tolist())
                    test_labels.extend(target.detach().cpu().numpy().tolist())
            # run ML model
            # print("train loader done")

            # print("here .... 1")
            train_data = np.array(train_data)
            train_labels = np.array(train_labels).ravel()
            test_data = np.array(test_data)
            test_labels = np.array(test_labels).ravel()
            # print("here .... 2")
            # print("len(train_data):", len(train_data))
            # print("len(train_labels):", len(train_labels))
            # print("len(test_data):", len(test_data))
            # print("len(test_labels):", len(test_labels))
            # ML_method = "RF"

            if ml_method == "NB":
                clf = GaussianNB()
            if ml_method == "SVM":
                # Apply PCA
                # You can adjust n_components based on your requirement
                pca = PCA(n_components=2)
                train_data = pca.fit_transform(train_data)
                test_data = pca.transform(test_data)
                clf = SVC(gamma="auto")
            if ml_method == "RF":
                clf = RandomForestClassifier(n_estimators=100)
            # clf = (
            #     SVC(gamma="auto")
            #     if ML_method == "SVM"
            #     else RandomForestClassifier(n_estimators=100)
            # )

            clf.fit(train_data, train_labels)
            pr = clf.predict(test_data)
            print("site:", site)

            accuracy, sensitivity, specificty = confusion(test_labels, pr)

            print(accuracy, sensitivity, specificty)
            res.append(confusion(test_labels, pr))
            writer.writerow(
                {
                    "iter": itr,
                    "ae_type": ae_type,
                    "filter_regions": filter_regions,
                    "ml_method": ml_method,
                    "site": site,
                    "train_size": len(train_samples),
                    "test_size": len(test_samples),
                    "acc": accuracy,
                    "sens": sensitivity,
                    "sepf": specificty,
                }
            )
            # break
        print("repeat: ", itr, np.mean(res, axis=0).tolist())
        overall_result.append(np.mean(res, axis=0).tolist())
        # break
    print("---------------Result of repeating 10 times-------------------")
    print(np.mean(np.array(overall_result), axis=0).tolist())
    results.close()
    sys.exit(0)

### Method to use and update the class for mix and match
def get_subjects(samples, centers, mode="all"):
    sub_sample = samples[samples["subject"].str.startswith(tuple(centers))]

    subjects = []
    fold_str = f"fold_0_train"
    subjects.extend(sub_sample[sub_sample[fold_str] == 1]["subject"].to_numpy())
    fold_str = f"fold_0_test"
    subjects.extend(sub_sample[sub_sample[fold_str] == 1]["subject"].to_numpy())
    subjects = np.array(subjects)

    final_subjects = []
    y_arr = []
    if mode == "all":
        final_subjects = subjects
        y_arr = np.array([get_label(f) for f in final_subjects])
    elif mode == "first-asd":
        for subject in subjects:
            if subject.startswith(tuple(centers[:1])) and get_label(subject) == 1:
                final_subjects.append(subject)
            if subject.startswith(tuple(centers[1:])) and get_label(subject) == 0:
                final_subjects.append(subject)
        y_arr = np.array([get_label(f) for f in final_subjects])
    elif mode == "second-asd":
        for subject in subjects:
            if subject.startswith(tuple(centers[1:])) and get_label(subject) == 1:
                final_subjects.append(subject)
            if subject.startswith(tuple(centers[:1])) and get_label(subject) == 0:
                final_subjects.append(subject)
        y_arr = np.array([get_label(f) for f in final_subjects])
    elif mode == "hc-center-as-class":
        y_arr = []
        for subject in subjects:
            if subject.startswith(tuple(centers[:1])) and get_label(subject) == 0:
                final_subjects.append(subject)
                y_arr.append(0)
            if subject.startswith(tuple(centers[1:])) and get_label(subject) == 0:
                final_subjects.append(subject)
                all_corr[subject] = (all_corr[subject][0], 1)
                y_arr.append(1)
    # TODO: you need to re-run this experiments
    elif mode == "asd-center-as-class":
        y_arr = []
        for subject in subjects:
            if subject.startswith(tuple(centers[:1])) and get_label(subject) == 1:
                final_subjects.append(subject)
                y_arr.append(1)
            if subject.startswith(tuple(centers[1:])) and get_label(subject) == 1:
                final_subjects.append(subject)
                all_corr[subject] = (all_corr[subject][0], 0)
                y_arr.append(0)
    else:
        print("wrong mode input")

    final_subjects = np.array(final_subjects)
    y_arr = np.array(y_arr)

    return final_subjects, y_arr


def get_subjects_age(samples, centers, mode="all", age_min=0, age_max=100):
    sub_sample = samples[samples["subject"].str.startswith(tuple(centers))]

    subjects = []
    fold_str = f"fold_0_train"
    subjects.extend(sub_sample[sub_sample[fold_str] == 1]["subject"].to_numpy())
    fold_str = f"fold_0_test"
    subjects.extend(sub_sample[sub_sample[fold_str] == 1]["subject"].to_numpy())
    subjects = np.array(subjects)
    
    # filter subject based on age
    filtered_subjects = df_labels[(df_labels['AGE_AT_SCAN'] >= age_min) & (df_labels['AGE_AT_SCAN'] <= age_max)]['FILE_ID'].tolist()
    
    final_subjects = []
    y_arr = []
    if mode == "all":
        final_subjects = np.array(list(set(subjects) & set(filtered_subjects)))
        # final_subjects = subjects
        y_arr = np.array([get_label(f) for f in final_subjects])
    elif mode == "first-asd":
        for subject in subjects:
            if subject.startswith(tuple(centers[:1])) and get_label(subject) == 1:
                final_subjects.append(subject)
            if subject.startswith(tuple(centers[1:])) and get_label(subject) == 0:
                final_subjects.append(subject)
        y_arr = np.array([get_label(f) for f in final_subjects])
    elif mode == "second-asd":
        for subject in subjects:
            if subject.startswith(tuple(centers[1:])) and get_label(subject) == 1:
                final_subjects.append(subject)
            if subject.startswith(tuple(centers[:1])) and get_label(subject) == 0:
                final_subjects.append(subject)
        y_arr = np.array([get_label(f) for f in final_subjects])
    elif mode == "hc-center-as-class":
        y_arr = []
        for subject in subjects:
            if subject.startswith(tuple(centers[:1])) and get_label(subject) == 0:
                final_subjects.append(subject)
                y_arr.append(0)
            if subject.startswith(tuple(centers[1:])) and get_label(subject) == 0:
                final_subjects.append(subject)
                all_corr[subject] = (all_corr[subject][0], 1)
                y_arr.append(1)
    # TODO: you need to re-run this experiments
    elif mode == "asd-center-as-class":
        y_arr = []
        for subject in subjects:
            if subject.startswith(tuple(centers[:1])) and get_label(subject) == 1:
                final_subjects.append(subject)
                y_arr.append(1)
            if subject.startswith(tuple(centers[1:])) and get_label(subject) == 1:
                final_subjects.append(subject)
                all_corr[subject] = (all_corr[subject][0], 0)
                y_arr.append(0)
    else:
        print("wrong mode input")

    final_subjects = np.array(final_subjects)
    y_arr = np.array(y_arr)

    return final_subjects, y_arr





def get_subject_score_ados(min=0, max=22):
    subjects = []

    hc_subjects = df_labels[
        (df_labels["DX_GROUP"] == 0) & (df_labels["FILE_ID"] != "no_filename")
    ]["FILE_ID"].to_numpy()
    asd_subjects = df_labels[
        (df_labels["ADOS_TOTAL"] >= min)
        & (df_labels["ADOS_TOTAL"] <= max)
        & (df_labels["FILE_ID"] != "no_filename")
    ]["FILE_ID"].to_numpy()
    hc_subjects = random.sample(list(hc_subjects), len(asd_subjects))
    subjects.extend(hc_subjects)
    subjects.extend(asd_subjects)

    subjects = np.array(subjects)
    y_arr = np.array([get_label(f) for f in subjects])
    return subjects, y_arr





if p_Method == "ASD-ml-combine-two-centers":

    print("p_Method:", p_Method)
    print("centers:", centers)

    output_dir = "ml_methods_mix_match_two_centers_mode_all/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = "ml_method_" + ml_method + "_" + "_".join(centers) + ".csv"
    print("results will be at:", output_dir + file_name)

    results = open(output_dir + file_name, "a")
    writer = csv.writer(results)

    modes = [
        "hc-center-as-class",
        "all",
        # "first-asd",
        # "second-asd",
        # "asd-center-as-class",
    ]
    # centers = ['CMU', 'OHSU']
    samples = pd.read_csv(path)

    center_text = "_".join(centers)
    num_corr = len(all_corr[flist[0]][0])
    print("num_corr:  ", num_corr)

    first_run = True
    for mode in modes[1:]:
        print("run mode:", mode)
        flist, y_arr = get_subjects(samples, centers, mode=mode)

        kk = 0
        crossval_res_kol_kol = []

        if first_run:
            fieldnames = [
                "iter",
                "centers",
                "mode",
                "use_ComBat",
                "method",
                "fold",
                "train_size",
                "test_size",
                "acc",
                "sens",
                "sepf",
            ]
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            writer.writeheader()
            first_run = False

        for run_combat in [False, True]:
            print("========================")
            print("run_combat: ", run_combat, "\n")
            all_rp_res = []
            overall_result = []
            for rp in range(10):
                res = []
                # kf = StratifiedKFold(n_splits=p_fold, shuffle=True, random_state=0)
                kf = StratifiedKFold(n_splits=p_fold)
                for kk, (train_index, test_index) in enumerate(kf.split(flist, y_arr)):
                    train_samples, test_samples = flist[train_index], flist[test_index]
                    print("train_samples: ", len(train_samples))
                    print("test_samples: ", len(test_samples))
                    verbose = True if (kk == 0) else False

                    if run_combat:
                        scanners, estimator = getEstimator(
                            all_corr, train_samples, df_labels
                        )
                        new_data = update_data(
                            all_corr,
                            num_corr,
                            scanners,
                            estimator,
                            np.concatenate((train_samples, test_samples)),
                        )
                    else:
                        estimator = None
                        scanners = None
                        new_data = all_corr
                    train_data = []
                    train_labels = []
                    test_data = []
                    test_labels = []

                    for i in train_samples:
                        train_data.append(new_data[i][0])
                        train_labels.append(new_data[i][1])

                    for i in test_samples:
                        test_data.append(new_data[i][0])
                        test_labels.append(new_data[i][1])

                    if ml_method == "NB":
                        clf = GaussianNB()
                    if ml_method == "SVM":
                        # Apply PCA
                        # You can adjust n_components based on your requirement
                        pca = PCA(n_components=2)
                        train_data = pca.fit_transform(train_data)
                        test_data = pca.transform(test_data)
                        clf = SVC(gamma="auto")

                    if ml_method == "RF":
                        clf = RandomForestClassifier(n_estimators=100)

                    clf.fit(train_data, train_labels)
                    pr = clf.predict(test_data)

                    # print("test_labels:", test_labels)
                    # print("pr:", pr)
                    accuracy, sensitivity, specificty = confusion(test_labels, pr)
                    print("fold:", kk)
                    print(accuracy, sensitivity, specificty)
                    res.append(confusion(test_labels, pr))
                    writer.writerow(
                        {
                            "iter": rp,
                            "centers": center_text,
                            "use_ComBat": run_combat,
                            "method": ml_method,
                            "fold": kk,
                            "train_size": len(train_samples),
                            "test_size": len(test_samples),
                            "acc": accuracy,
                            "sens": sensitivity,
                            "sepf": specificty,
                        }
                    )
                print("repeat: ", rp, np.mean(res, axis=0).tolist())
                overall_result.append(np.mean(res, axis=0).tolist())
            print("---------------Result of repeating 10 times-------------------")
            print(np.mean(np.array(overall_result), axis=0).tolist())
    results.close()
    sys.exit(0)



if p_Method == "ASD-ml-combine-two-centers-ae-harmonization":
    output_dir = "ml_methods_mix_match_two_centers_ae_harmonization/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ******** Here we change AE type *****#
    # ae_type = "DAE"
    num_epochs = 25
    print("num_epochs:", num_epochs)
    
    file_name = "ml_method_" + ml_method + "_ae_harmonization_" + ae_type + "_" + "_".join(centers) + ".csv"
    print("results will be at:", output_dir + file_name)

    results = open(output_dir + file_name, "a")
    writer = csv.writer(results)

    modes = [
        "hc-center-as-class",
        "all",
        # "first-asd",
        # "second-asd",
        # "asd-center-as-class",
    ]
    
    num_corr = len(all_corr[flist[0]][0])
    print("num_corr:  ", num_corr)
    filter_regions = False
    eig_data = None
    
    start = time.time()
    batch_size = 8
    learning_rate_ae, learning_rate_clf = 0.0001, 0.0001

    p_bernoulli = None
    augmentation = p_augmentation
    use_dropout = False

    aug_factor = 2
    num_neighbs = 5
    lim4sim = 2
    n_lat = int(num_corr / 4)
    print(n_lat)
    start = time.time()

    print("p_bernoulli: ", p_bernoulli)
    print(
        "augmentaiton: ",
        augmentation,
        "aug_factor: ",
        aug_factor,
        "num_neighbs: ",
        num_neighbs,
        "lim4sim: ",
        lim4sim,
    )
    print("use_dropout: ", use_dropout, "\n")
    print("filter_regions:", filter_regions)

    sim_function = functools.partial(cal_similarity, lim=lim4sim)
    
    # centers = ['CMU', 'OHSU']
    samples = pd.read_csv(path)

    center_text = "_".join(centers)
    num_corr = len(all_corr[flist[0]][0])
    print("num_corr:  ", num_corr)

    first_run = True
    for mode in modes[:1]:
        print("run mode:", mode)
        flist, y_arr = get_subjects(samples, centers, mode=mode)

        kk = 0
        crossval_res_kol_kol = []

        if first_run:
            fieldnames = [
                "iter",
                "centers",
                "mode",
                "use_ComBat",
                "method",
                "fold",
                "train_size",
                "test_size",
                "acc",
                "sens",
                "sepf",
            ]
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            writer.writeheader()
            first_run = False
            
       
        #TODO Here we run AEs
        
        run_combat = False
        # for run_combat in [False, True]:
        print("========================")
        print("run_combat: ", run_combat, "\n")
        all_rp_res = []
        overall_result = []
        for rp in range(10):
            res = []
            # kf = StratifiedKFold(n_splits=p_fold, shuffle=True, random_state=0)
            kf = StratifiedKFold(n_splits=p_fold)
            for kk, (train_index, test_index) in enumerate(kf.split(flist, y_arr)):
                train_samples, test_samples = flist[train_index], flist[test_index]
                print("train_samples: ", len(train_samples))
                print("test_samples: ", len(test_samples))
                verbose = True if (kk == 0) else False

                if run_combat:
                    scanners, estimator = getEstimator(
                        all_corr, train_samples, df_labels
                    )
                    new_data = update_data(
                        all_corr,
                        num_corr,
                        scanners,
                        estimator,
                        np.concatenate((train_samples, test_samples)),
                    )
                else:
                    estimator = None
                    scanners = None
                    new_data = all_corr
                regions_inds = np.array([i for i in range(num_corr)])
                # verbose = True
                num_inpp = len(regions_inds)
                
                n_lat = int(num_inpp / 2)

                print("num_inpp:", num_inpp)
                print("n_lat:", n_lat)
                
                
                train_loader = get_loader(
                    data=new_data,
                    samples_list=train_samples,
                    batch_size=batch_size,
                    mode="train",
                    augmentation=False,
                    aug_factor=aug_factor,
                    num_neighbs=num_neighbs,
                    eig_data=eig_data,
                    similarity_fn=sim_function,
                    verbose=verbose,
                    regions=regions_inds,
                    num_corr=num_corr,
                    scanners=scanners,
                    estimator=None,
                )

                test_loader = get_loader(
                    data=new_data,
                    samples_list=test_samples,
                    batch_size=batch_size,
                    mode="test",
                    augmentation=False,
                    verbose=verbose,
                    regions=regions_inds,
                    num_corr=num_corr,
                    scanners=scanners,
                    estimator=None,
                )
                
                
                if ae_type == "AE":
                    model = AE(
                        tied=False,
                        num_inputs=num_inpp,
                        num_latent=n_lat,
                    )
                elif ae_type == "TAE":
                    model = AE(
                        tied=True,
                        num_inputs=num_inpp,
                        num_latent=n_lat,
                    )
                elif ae_type == "SAE":
                    model = SAE(
                        n_features=num_inpp,
                        n_lat=n_lat,
                    )
                elif ae_type == "DAE":
                    model = DAE(input_size=num_inpp)
                else:
                    # model = DAE(input_size=num_inpp)
                    model = VAE(input_size=num_inpp, latent_size=n_lat, hidden_size=1000)
                    
                model.to(device)
                criterion_ae = nn.MSELoss(reduction="sum")
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_ae)
                # optimizer = optim.SGD(
                #     [
                #         {"params": model.fc_encoder.parameters(), "lr": learning_rate_ae},
                #         {"params": model.classifier.parameters(), "lr": learning_rate_clf},
                #     ],
                #     momentum=0.9,
                # )

                trained_model = train_ae(model, num_epochs, train_loader, ae_type)
                print("trained model done ")
                
                
                train_data = []
                train_labels = []
                test_data = []
                test_labels = []
                
                
                # loop through train loader and append the data after we pass it to the model
                trained_model.eval()
                with torch.no_grad():
                    for i, (batch_x, batch_y) in enumerate(train_loader):
                        if len(batch_x) != batch_size:
                            continue
                        data, target = batch_x.to(device), batch_y.to(device)
                        if ae_type == "VAE":
                            _, output, _, _ = trained_model(data)
                        else:
                            _, output = trained_model(data)
                        train_data.extend(output.detach().cpu().numpy().tolist())
                        train_labels.extend(target.detach().cpu().numpy().tolist())
                # print("train loader done")
                # loop through test loader and append the data after we pass it to the model
                trained_model.eval()
                with torch.no_grad():
                    for i, (batch_x, batch_y) in enumerate(test_loader):
                        if len(batch_x) != batch_size:
                            continue
                        data, target = batch_x.to(device), batch_y.to(device)
                        if ae_type == "VAE":
                            _, output, _, _ = trained_model(data)
                        else:
                            _, output = trained_model(data)
                        test_data.extend(output.detach().cpu().numpy().tolist())
                        test_labels.extend(target.detach().cpu().numpy().tolist())
                        
                        
                train_data = np.array(train_data)
                train_labels = np.array(train_labels).ravel()
                test_data = np.array(test_data)
                test_labels = np.array(test_labels).ravel()
                
                if ml_method == "NB":
                    clf = GaussianNB()
                if ml_method == "SVM":
                    # Apply PCA
                    # You can adjust n_components based on your requirement
                    pca = PCA(n_components=2)
                    train_data = pca.fit_transform(train_data)
                    test_data = pca.transform(test_data)
                    clf = SVC(gamma="auto")

                if ml_method == "RF":
                    clf = RandomForestClassifier(n_estimators=100)

                clf.fit(train_data, train_labels)
                pr = clf.predict(test_data)

                accuracy, sensitivity, specificty = confusion(test_labels, pr)
                print(accuracy, sensitivity, specificty)
                res.append(confusion(test_labels, pr))
                writer.writerow(
                    {
                        "iter": rp,
                        "centers": center_text,
                        "use_ComBat": ae_type,
                        "method": ml_method,
                        "fold": kk,
                        "train_size": len(train_samples),
                        "test_size": len(test_samples),
                        "acc": accuracy,
                        "sens": sensitivity,
                        "sepf": specificty,
                    }
                )
            print("repeat: ", rp, np.mean(res, axis=0).tolist())
            overall_result.append(np.mean(res, axis=0).tolist())
        print("---------------Result of repeating 10 times-------------------")
        print(np.mean(np.array(overall_result), axis=0).tolist())
        results.close()
        sys.exit(0)


if p_Method == "ASD-ml-combine-two-centers-asd-vs-hc":
    output_dir = "ml_methods_mix_match_two_centers_asd_vs_hc/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ******** Here we change AE type *****#
    # ae_type = "DAE"
    num_epochs = 25
    print("num_epochs:", num_epochs)
    
    file_name = "ml_method_" + ml_method + "_asd_vs_hc_" + "_".join(centers) + ".csv"
    print("results will be at:", output_dir + file_name)

    results = open(output_dir + file_name, "a")
    writer = csv.writer(results)

    modes = [
        # "hc-center-as-class",
        # "all",
        "first-asd",
        "second-asd",
        #"asd-center-as-class",
    ]

    num_corr = len(all_corr[flist[0]][0])
    print("num_corr:  ", num_corr)
    filter_regions = False
    eig_data = None
    
    start = time.time()
    batch_size = 8
    learning_rate_ae, learning_rate_clf = 0.0001, 0.0001

    p_bernoulli = None
    augmentation = p_augmentation
    use_dropout = False

    aug_factor = 2
    num_neighbs = 5
    lim4sim = 2
    n_lat = int(num_corr / 4)
    print(n_lat)
    start = time.time()

    print("p_bernoulli: ", p_bernoulli)
    print(
        "augmentaiton: ",
        augmentation,
        "aug_factor: ",
        aug_factor,
        "num_neighbs: ",
        num_neighbs,
        "lim4sim: ",
        lim4sim,
    )
    print("use_dropout: ", use_dropout, "\n")
    print("filter_regions:", filter_regions)

    sim_function = functools.partial(cal_similarity, lim=lim4sim)
    
    # centers = ['CMU', 'OHSU']
    samples = pd.read_csv(path)

    center_text = "_".join(centers)
    num_corr = len(all_corr[flist[0]][0])
    print("num_corr:  ", num_corr)

    first_run = True
    for mode in modes:
        print("run mode:", mode)
        flist, y_arr = get_subjects(samples, centers, mode=mode)

        kk = 0
        crossval_res_kol_kol = []

        if first_run:
            fieldnames = [
                "iter",
                "centers",
                "mode",
                "use_ComBat",
                "method",
                "fold",
                "train_size",
                "test_size",
                "acc",
                "sens",
                "sepf",
            ]
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            writer.writeheader()
            first_run = False
            
               
        # run_combat = False
        # for run_combat in [False, True]:
        print("========================")
        print("run_combat: ", run_combat, "\n")
        all_rp_res = []
        overall_result = []
        for rp in range(10):
            res = []
            # kf = StratifiedKFold(n_splits=p_fold, shuffle=True, random_state=0)
            kf = StratifiedKFold(n_splits=p_fold)
            for kk, (train_index, test_index) in enumerate(kf.split(flist, y_arr)):
                train_samples, test_samples = flist[train_index], flist[test_index]
                print("train_samples: ", len(train_samples))
                print("test_samples: ", len(test_samples))
                verbose = True if (kk == 0) else False

                if run_combat:
                    scanners, estimator = getEstimator(
                        all_corr, train_samples, df_labels
                    )
                    new_data = update_data(
                        all_corr,
                        num_corr,
                        scanners,
                        estimator,
                        np.concatenate((train_samples, test_samples)),
                    )
                else:
                    estimator = None
                    scanners = None
                    new_data = all_corr
                regions_inds = np.array([i for i in range(num_corr)])
                # verbose = True
                num_inpp = len(regions_inds)
                
                n_lat = int(num_inpp / 2)

                print("num_inpp:", num_inpp)
                print("n_lat:", n_lat)
                
                
                
                train_data = []
                train_labels = []
                test_data = []
                test_labels = []
                
                for i in train_samples:
                    train_data.append(new_data[i][0])
                    train_labels.append(new_data[i][1])

                for i in test_samples:
                    test_data.append(all_corr[i][0])
                    test_labels.append(all_corr[i][1])
                
                        
                train_data = np.array(train_data)
                train_labels = np.array(train_labels).ravel()
                test_data = np.array(test_data)
                test_labels = np.array(test_labels).ravel()
                
                if ml_method == "NB":
                    clf = GaussianNB()
                if ml_method == "SVM":
                    # Apply PCA
                    # You can adjust n_components based on your requirement
                    pca = PCA(n_components=2)
                    train_data = pca.fit_transform(train_data)
                    test_data = pca.transform(test_data)
                    clf = SVC(gamma="auto")

                if ml_method == "RF":
                    clf = RandomForestClassifier(n_estimators=100)

                clf.fit(train_data, train_labels)
                pr = clf.predict(test_data)

                accuracy, sensitivity, specificty = confusion(test_labels, pr)
                print(accuracy, sensitivity, specificty)
                res.append(confusion(test_labels, pr))
                writer.writerow(
                    {
                        "iter": rp,
                        "centers": center_text,
                        "use_ComBat": ae_type,
                        "method": ml_method,
                        "fold": kk,
                        "train_size": len(train_samples),
                        "test_size": len(test_samples),
                        "acc": accuracy,
                        "sens": sensitivity,
                        "sepf": specificty,
                    }
                )
            print("repeat: ", rp, np.mean(res, axis=0).tolist())
            overall_result.append(np.mean(res, axis=0).tolist())
        print("---------------Result of repeating 10 times-------------------")
        print(np.mean(np.array(overall_result), axis=0).tolist())
    results.close()
    sys.exit(0)



# ***** Here to use machine learning algorithms with ComBat and without
if p_Method == "ASD-ml" and p_mode == "whole" and run_combat == False:
    
    output_dir = "ml_leave_one_site_out_combat_False_/ML/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = "ml_leave_one_site_out_combat_False_" + ml_method + ".csv"
    print("results will be at:", output_dir + file_name)

    results = open(output_dir + file_name, "a")
    writer = csv.writer(results)
    fieldnames = [
        "iter",
        "combat",
        "method",
        "site",
        "train_size",
        "test_size",
        "acc",
        "sens",
        "sepf",
    ]
    writer = csv.DictWriter(results, fieldnames=fieldnames)
    writer.writeheader()

    # if ml_method == "NB":
    #     clf = GaussianNB()
    #     # nb_classifier.fit(X_train, y_train)
    #     pass

    # clf = (
    #     SVC(gamma="auto")
    #     if p_Method == "SVM"
    #     else RandomForestClassifier(n_estimators=100)
    # )
    overall_result = []
    for rp in range(10):
        kf = StratifiedKFold(n_splits=p_fold, random_state=1, shuffle=True)
        np.random.shuffle(flist)
        y_arr = np.array([get_label(f) for f in flist])
        res = []
        # sites = ["NYU", "UCLA", "UM", "USM"]
        sites = [
            "NYU",
            "UCLA",
            "UM",
            "USM",
            "Caltech",
            "CMU",
            "KKI",
            "Leuven",
            "MaxMun",
            "OHSU",
            "Olin",
            "Pitt",
            "SBL",
            "SDSU",
            "Stanford",
            "Trinity",
            "Yale",
        ]
        for site in sites:
            print("Run site:", site)
            new_sites = [item for item in sites if item != site]
            train_samples = [file for file in flist if file.split("_")[0] in new_sites]
            test_samples = [file for file in flist if file.split("_")[0] in [site]]
            train_samples = np.array(train_samples)
            test_samples = np.array(test_samples)
            print("len(train_samples):", len(train_samples))
            print("len(test_samples):", len(test_samples))
            # for kk,(train_index, test_index) in enumerate(kf.split(flist, y_arr)):
            #     train_samples, test_samples = np.array(flist)[train_index], np.array(flist)[test_index]
            train_data = []
            train_labels = []
            test_data = []
            test_labels = []

            for i in train_samples:
                train_data.append(all_corr[i][0])
                train_labels.append(all_corr[i][1])

            for i in test_samples:
                test_data.append(all_corr[i][0])
                test_labels.append(all_corr[i][1])

            if ml_method == "NB":
                clf = GaussianNB()
            if ml_method == "SVM":
                # Apply PCA
                # You can adjust n_components based on your requirement
                pca = PCA(n_components=2)
                train_data = pca.fit_transform(train_data)
                test_data = pca.transform(test_data)
                clf = SVC(gamma="auto")

            clf.fit(train_data, train_labels)
            pr = clf.predict(test_data)
            print("site:", site)

            accuracy, sensitivity, specificty = confusion(test_labels, pr)
            print(accuracy, sensitivity, specificty)
            res.append(confusion(test_labels, pr))
            writer.writerow(
                {
                    "iter": rp,
                    "combat": run_combat,
                    "method": ml_method,
                    "site": site,
                    "train_size": len(train_samples),
                    "test_size": len(test_samples),
                    "acc": accuracy,
                    "sens": sensitivity,
                    "sepf": specificty,
                }
            )
        print("repeat: ", rp, np.mean(res, axis=0).tolist())
        overall_result.append(np.mean(res, axis=0).tolist())
    print("---------------Result of repeating 10 times-------------------")
    print(np.mean(np.array(overall_result), axis=0).tolist())
    results.close()
    sys.exit(0)

if p_Method == "ASD-ml" and p_mode == "whole" and run_combat:

    # ml_method = "NB"

    output_dir = "all_centers_experiments/ML/"
    file_name = (
        "ml_leave_one_site_out_combat_True_"
        + ml_method
        + ".csv"
    )
    print("results will be at:", output_dir + file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = open(output_dir + file_name, "a")
    writer = csv.writer(results)
    fieldnames = [
        "iter",
        "combat",
        "method",
        "site",
        "train_size",
        "test_size",
        "acc",
        "sens",
        "sepf",
    ]
    writer = csv.DictWriter(results, fieldnames=fieldnames)
    writer.writeheader()

    # clf = (
    #     SVC(gamma="auto")
    #     if p_Method == "SVM"
    #     else RandomForestClassifier(n_estimators=100)
    # )
    overall_result = []
    for rp in range(10):
        kf = StratifiedKFold(n_splits=p_fold, random_state=1, shuffle=True)
        np.random.shuffle(flist)
        y_arr = np.array([get_label(f) for f in flist])
        res = []
        # sites = ["NYU", "UCLA", "UM", "USM"]
        sites = [
            "NYU",
            "UCLA",
            "UM",
            "USM",
            "Caltech",
            "CMU",
            "KKI",
            "Leuven",
            "MaxMun",
            "OHSU",
            "Olin",
            "Pitt",
            "SBL",
            "SDSU",
            "Stanford",
            "Trinity",
            "Yale",
        ]
        for site in sites:
            print("Run site:", site)
            new_sites = [item for item in sites if item != site]
            train_samples = [file for file in flist if file.split("_")[0] in new_sites]
            test_samples = [file for file in flist if file.split("_")[0] in [site]]
            train_samples = np.array(train_samples)
            test_samples = np.array(test_samples)
            print("len(train_samples):", len(train_samples))
            print("len(test_samples):", len(test_samples))

            all_subjects = np.concatenate((train_samples, test_samples))
            # for kk,(train_index, test_index) in enumerate(kf.split(flist, y_arr)):
            #     train_samples, test_samples = np.array(flist)[train_index], np.array(flist)[test_index]
            scanners, estimator = getEstimator(
                all_corr,
                all_subjects,
                df_labels,
                add_gender=add_gender,
                add_age=add_age,
            )
            num_corr = 19900
            new_data = update_data(
                all_corr, num_corr, scanners, estimator, all_subjects
            )
            print("new_data updated!")

            train_data = []
            train_labels = []
            test_data = []
            test_labels = []

            for i in train_samples:
                train_data.append(new_data[i][0])
                train_labels.append(new_data[i][1])

            for i in test_samples:
                test_data.append(all_corr[i][0])
                test_labels.append(all_corr[i][1])

            if ml_method == "NB":
                clf = GaussianNB()
            if ml_method == "SVM":
                # Apply PCA
                # You can adjust n_components based on your requirement
                pca = PCA(n_components=2)
                train_data = pca.fit_transform(train_data)
                test_data = pca.transform(test_data)
                clf = SVC(gamma="auto")
            if ml_method == "RF":
                clf = RandomForestClassifier(n_estimators=100)

            clf.fit(train_data, train_labels)
            pr = clf.predict(test_data)

            print("site:", site)
            accuracy, sensitivity, specificty = confusion(test_labels, pr)
            print(accuracy, sensitivity, specificty)
            res.append(confusion(test_labels, pr))
            writer.writerow(
                {
                    "iter": rp,
                    "combat": run_combat,
                    "method": p_Method,
                    "site": site,
                    "train_size": len(train_samples),
                    "test_size": len(test_samples),
                    "acc": accuracy,
                    "sens": sensitivity,
                    "sepf": specificty,
                }
            )

        print("repeat: ", rp, np.mean(res, axis=0).tolist())
        overall_result.append(np.mean(res, axis=0).tolist())
    print("---------------Result of repeating 10 times-------------------")
    print(np.mean(np.array(overall_result), axis=0).tolist())
    results.close()
    sys.exit(0)
if p_Method == "ASD-ml-combine-two-centers-permutation-test":

    print("p_Method:", p_Method)
    print("centers:", centers)

    output_dir = "ps_center_classification_comabt_none/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = "ml_method_" + ml_method + "_" + "_".join(centers) + ".csv"
    print("results will be at:", output_dir + file_name)

    results = open(output_dir + file_name, "a")
    writer = csv.writer(results)

    modes = [
        "hc-center-as-class",
        # "all",
        # "first-asd",
        # "second-asd",
        # "asd-center-as-class",
    ]
    # centers = ['CMU', 'OHSU']
    samples = pd.read_csv(path)

    center_text = "_".join(centers)
    num_corr = len(all_corr[flist[0]][0])
    print("num_corr:  ", num_corr)

    first_run = True
    for mode in modes:
        print("run mode:", mode)
        flist, y_arr = get_subjects(samples, centers, mode=mode)

        kk = 0
        crossval_res_kol_kol = []

        if first_run:
            fieldnames = [
                "iter",
                "centers",
                "mode",
                "use_ComBat",
                "method",
                "fold",
                "train_size",
                "test_size",
                "acc",
                "sens",
                "sepf",
            ]
            writer = csv.DictWriter(results, fieldnames=fieldnames)
            writer.writeheader()
            first_run = False

        for run_combat in [False, True][:1]:
            print("========================")
            print("run_combat: ", run_combat, "\n")
            all_rp_res = []
            overall_result = []
            # Permutation test
            null_distribution = []
            for rp in range(1000):
                res = []
                # kf = StratifiedKFold(n_splits=p_fold, shuffle=True, random_state=0)
                kf = StratifiedKFold(n_splits=p_fold)
                shuffled_labels = np.random.permutation(y_arr)
                for kk, (train_index, test_index) in enumerate(kf.split(flist, shuffled_labels)):
                    train_samples, test_samples = flist[train_index], flist[test_index]
                    train_y, test_y = shuffled_labels[train_index], shuffled_labels[test_index]
                    print("train_samples: ", len(train_samples))
                    print("test_samples: ", len(test_samples))
                    verbose = True if (kk == 0) else False

                    if run_combat:
                        scanners, estimator = getEstimator(
                            all_corr, train_samples, df_labels
                        )
                        new_data = update_data(
                            all_corr,
                            num_corr,
                            scanners,
                            estimator,
                            np.concatenate((train_samples, test_samples)),
                        )
                    else:
                        estimator = None
                        scanners = None
                        new_data = all_corr
                    train_data = []
                    train_labels = []
                    test_data = []
                    test_labels = []

                    for j, i in enumerate(train_samples):
                        train_data.append(new_data[i][0])
                        train_labels.append(train_y[j])

                    for j, i in enumerate(test_samples):
                        test_data.append(new_data[i][0])
                        test_labels.append(test_y[j])

                    if ml_method == "NB":
                        clf = GaussianNB()
                    if ml_method == "SVM":
                        # Apply PCA
                        # You can adjust n_components based on your requirement
                        pca = PCA(n_components=2)
                        train_data = pca.fit_transform(train_data)
                        test_data = pca.transform(test_data)
                        clf = SVC(gamma="auto")

                    if ml_method == "RF":
                        clf = RandomForestClassifier(n_estimators=100)

                    clf.fit(train_data, train_labels)
                    pr = clf.predict(test_data)

                    # print("test_labels:", test_labels)
                    # print("pr:", pr)
                    accuracy, sensitivity, specificty = confusion(test_labels, pr)
                    null_distribution.append(accuracy)
                    print("fold:", kk)
                    print(accuracy, sensitivity, specificty)
                    res.append(accuracy)
                #     writer.writerow(
                #         {
                #             "iter": rp,
                #             "centers": center_text,
                #             "use_ComBat": run_combat,
                #             "method": ml_method,
                #             "fold": kk,
                #             "train_size": len(train_samples),
                #             "test_size": len(test_samples),
                #             "acc": accuracy,
                #             "sens": sensitivity,
                #             "sepf": specificty,
                #         }
                #     )
                
                print("repeat: ", rp)
                print(np.mean(np.array(res)))
                # overall_result.append(np.mean(res, axis=0).tolist())
            print("---------------Result of repeating 1000 times-------------------")
            print(np.mean(np.array(null_distribution)))
    # results.close()
    sys.exit(0)
