# (c) Charles Le Losq 2024
# see embedded licence file

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn import model_selection
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import SplineTransformer
#from sklearn.linear_model import HuberRegressor
from sklearn.metrics import (
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from scipy.special import erf

import h5py

import matplotlib.pyplot as plt

import torch

# to load data in a library
_BASEDATAPATH = Path(os.path.dirname(__file__)) / "data"

########################################
# Functions for device and create dirs
########################################
def create_dir(dirName):
    """search and, if necessary, create a folder

    Parameters
    ----------
    dirName : str
        path of the new directory
    """
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

def get_device(verbose=True):
    """set device = cuda if a GPU is available, if not device = cpu"""

    if torch.cuda.is_available():
        torch.device("cuda:0")
        device = "cuda"
    else:
        torch.device("cpu")
        device = "cpu"
    
    if verbose == True:
        print("Will run on {}".format(device))

    return device
########################################
# FUNCTIONS FOR CHEMICAL PREPROCESSING
########################################
def list_oxides():
    """returns the list of oxydes handled by i-Melt codes

    Returns
    -------
    out : list
        list of the oxides in the good order that are handled in i-Melt codes
    """
    return [
        "sio2",
        "tio2",
        "al2o3",
        "feo",
        "fe2o3",
        "mno",
        "na2o",
        "k2o",
        "mgo",
        "cao",
        "p2o5",
        "h2o",
    ]


def molarweights():
    """returns a partial table of molecular weights for elements and oxides that can be used in other functions

    Uses molar weights from IUPAC Periodic Table 2016

    Returns
    =======
    w : dictionary
        containing the molar weights of elements and oxides:

        - si, ti, al, fe, h, li, na, k, mg, ca, ba, sr, ni, mn, p, o (no upper case, symbol calling)

        - sio2, tio2, al2o3, fe2o3, feo, h2o, li2o, na2o, k2o, mgo, cao, sro, bao, nio, mno, p2o5 (no upper case, symbol calling)

    """
    w = {"si": 28.085}

    # in g/mol
    w["ti"] = 47.867
    w["al"] = 26.982
    w["fe"] = 55.845
    w["h"] = 1.00794
    w["li"] = 6.94
    w["na"] = 22.990
    w["k"] = 39.098
    w["mg"] = 24.305
    w["ca"] = 40.078
    w["ba"] = 137.327
    w["sr"] = 87.62
    w["o"] = 15.9994

    w["ni"] = 58.6934
    w["mn"] = 54.938045
    w["p"] = 30.973762

    # oxides
    w["sio2"] = w["si"] + 2 * w["o"]
    w["tio2"] = w["ti"] + 2 * w["o"]
    w["al2o3"] = 2 * w["al"] + 3 * w["o"]
    w["fe2o3"] = 2 * w["fe"] + 3 * w["o"]
    w["feo"] = w["fe"] + w["o"]
    w["h2o"] = 2 * w["h"] + w["o"]
    w["li2o"] = 2 * w["li"] + w["o"]
    w["na2o"] = 2 * w["na"] + w["o"]
    w["k2o"] = 2 * w["k"] + w["o"]
    w["mgo"] = w["mg"] + w["o"]
    w["cao"] = w["ca"] + w["o"]
    w["sro"] = w["sr"] + w["o"]
    w["bao"] = w["ba"] + w["o"]

    w["nio"] = w["ni"] + w["o"]
    w["mno"] = w["mn"] + w["o"]
    w["p2o5"] = w["p"] * 2 + w["o"] * 5
    return w  # explicit return


def wt_mol(data):
    """convert weights in mol fraction

    Parameters
    ==========
    data: Pandas DataFrame
        containing the fields sio2, tio2, al2o3, feo, mno, na2o, k2o, mgo, cao, p2o5, h2o

    Returns
    =======
    chemtable: Pandas DataFrame
        contains the fields sio2, tio2, al2o3, feo, mno, na2o, k2o, mgo, cao, p2o5, h2o in mol%
    """
    chemtable = data.copy()
    w = molarweights()

    # conversion to mol in 100 grammes
    sio2 = chemtable["sio2"] / w["sio2"]
    tio2 = chemtable["tio2"] / w["tio2"]
    al2o3 = chemtable["al2o3"] / w["al2o3"]
    fe2o3 = chemtable["fe2o3"] / w["fe2o3"]
    feo = chemtable["feo"] / w["feo"]
    mno = chemtable["mno"] / w["mno"]
    na2o = chemtable["na2o"] / w["na2o"]
    k2o = chemtable["k2o"] / w["k2o"]
    mgo = chemtable["mgo"] / w["mgo"]
    cao = chemtable["cao"] / w["cao"]
    p2o5 = chemtable["p2o5"] / w["p2o5"]
    h2o = chemtable["h2o"] / w["h2o"]

    # renormalisation
    tot = sio2 + tio2 + al2o3 + fe2o3 + feo + mno + na2o + k2o + mgo + cao + p2o5 + h2o
    chemtable["sio2"] = sio2 / tot
    chemtable["tio2"] = tio2 / tot
    chemtable["al2o3"] = al2o3 / tot
    chemtable["fe2o3"] = fe2o3 / tot
    chemtable["feo"] = feo / tot
    chemtable["mno"] = mno / tot
    chemtable["na2o"] = na2o / tot
    chemtable["k2o"] = k2o / tot
    chemtable["mgo"] = mgo / tot
    chemtable["cao"] = cao / tot
    chemtable["p2o5"] = p2o5 / tot
    chemtable["h2o"] = h2o / tot

    return chemtable


def descriptors(X):
    """generate a X augmented dataframe with new descriptors
    
    Parameters
    ----------
    X : pandas dataframe
        input melt composition

    Returns
    -------
        X : pandas dataframe
            updated dataframe with new descriptors (as a copy).
    """

    # number of network formers (T), modifiers (M) and oxygen atoms
    # we consider Fe3+ as network former but this could be discussed (see Le Losq and Sossi 2023)
    Tetra = X.sio2 + X.tio2 + 2 * X.al2o3 + 2 * X.fe2o3
    Metals = (
        2 * X.na2o + 2 * X.k2o + X.mgo + X.cao + X.feo + X.mno + 2 * X.p2o5 + 2 * X.h2o
    )
    Oxygens = (
        2 * X.sio2
        + 2 * X.tio2
        + 3 * X.al2o3
        + 2 * X.fe2o3
        + X.feo
        + X.mno
        + X.na2o
        + X.k2o
        + X.mgo
        + X.cao
        + 5 * X.p2o5
        + X.h2o
    )
    X["Tetra"] = Tetra
    X["Metals"] = Metals
    X["Oxygens"] = Oxygens

    # electronegativity calculations
    # not an exact way to do it, just used as new features
    X["en_Tetra"] = 1.8 * X.sio2 + 1.5 * X.tio2 + 1.5 * 2 * X.al2o3 + 1.8 * 2 * X.fe2o3
    X["en_Metals"] = (
        0.93 * 2 * X.na2o
        + 0.82 * 2 * X.k2o
        + 1.31 * X.mgo
        + 1.0 * X.cao
        + 1.83 * X.feo
        + 1.55 * X.mno
        + 2.19 * 2 * X.p2o5
        + 2.20 * 2 * X.h2o
    )
    X["en_Oxygens"] = 3.44 * Oxygens
    # average anion-cation EN difference
    X["en_diff_average"] = X["en_Oxygens"] - (X["en_Metals"] + X["en_Tetra"])

    # calculation of NBO/T
    # we allow it to be negative, as it is the case for some samples
    # we will use it as a feature, not as a target
    # we count phosphorus here as a network former because well it can be a glass former.
    X["nbot"] = (2 * Oxygens - 4 * Tetra) / Tetra
    X.loc[Tetra == 0, "nbot"] = (
        4  # we put 4 if there is no network former in the mix, so if T=0
    )

    # calculation of optical basicity
    # partial oxyde values from Moretti 2005,
    # DOI 10.4401/ag-3221
    # Table 1
    X["optbas"] = (
        0.48 * X.sio2
        + 0.58 * X.tio2
        + 0.59 * X.al2o3
        + 0.48 * X.fe2o3
        + 0.51 * X.feo
        + 1.15 * X.na2o
        + 1.36 * X.k2o
        + 0.78 * X.mgo
        + 0.99 * X.cao
        + 0.59 * X.mno
        + 0.40 * X.p2o5
        + 0.39 * X.h2o
    )

    # calculation of the molecular weight of the glass
    mw = molarweights()
    X["mw"] = (
        mw["sio2"] * X.sio2
        + mw["tio2"] * X.tio2
        + mw["al2o3"] * X.al2o3
        + mw["fe2o3"] * X.fe2o3
        + mw["feo"] * X.feo
        + mw["mno"] * X.mno
        + mw["na2o"] * X.na2o
        + mw["k2o"] * X.k2o
        + mw["mgo"] * X.mgo
        + mw["cao"] * X.cao
        + mw["p2o5"] * X.p2o5
        + mw["h2o"] * X.h2o
    )

    # calculation of the ratio of each oxide
    # we input the ratios in a dict then concatenate things in the Pandas dataframe
    # this avoids slowness in trying to directly add Pandas columns
    list_ox = list_oxides()
    ratios = {}
    for i in list_ox:
        for j in list_ox:
            if i != j:
                ratios[i + "_" + j] = X.loc[:, i] / (X.loc[:, i] + X.loc[:, j])
    X = pd.concat((X, pd.DataFrame(ratios)), axis=1)
    # calculation of the ratio of aluminium over the sum of modifier cations
    X["al_m"] = X.al2o3 / (
        X.al2o3
        + X.loc[:, ["h2o", "na2o", "k2o", "feo", "mno", "mgo", "cao"]].sum(axis=1)
    )

    return X.fillna(value=0).copy()


def chimie_control(data):
    """check that all needed oxides are there and setup correctly the Pandas datalist.

    Parameters
    ----------
    data : Pandas dataframe
        the user input list.

    Returns
    -------
    out : Pandas dataframe
        the output list with all required oxides.
    """
    list_ox = list_oxides()
    datalist = data.copy()  # safety net

    for i in list_ox:
        try:
            oxd = datalist[i]
        except:
            datalist[i] = 0.0

    # we calculate the sum and store it
    sum_oxides = (
        datalist["sio2"]
        + datalist["tio2"]
        + datalist["al2o3"]
        + datalist["fe2o3"]
        + datalist["feo"]
        + datalist["mno"]
        + datalist["na2o"]
        + datalist["k2o"]
        + datalist["mgo"]
        + datalist["cao"]
        + datalist["p2o5"]
        + datalist["h2o"]
    )

    # renormalisation of each element
    datalist["sio2"] = datalist["sio2"] / sum_oxides
    datalist["tio2"] = datalist["tio2"] / sum_oxides
    datalist["al2o3"] = datalist["al2o3"] / sum_oxides
    datalist["fe2o3"] = datalist["fe2o3"] / sum_oxides
    datalist["feo"] = datalist["feo"] / sum_oxides
    datalist["mno"] = datalist["mno"] / sum_oxides
    datalist["na2o"] = datalist["na2o"] / sum_oxides
    datalist["k2o"] = datalist["k2o"] / sum_oxides
    datalist["mgo"] = datalist["mgo"] / sum_oxides
    datalist["cao"] = datalist["cao"] / sum_oxides
    datalist["p2o5"] = datalist["p2o5"] / sum_oxides
    datalist["h2o"] = datalist["h2o"] / sum_oxides

    # we calculate again the sum and store it in the dataframe
    datalist["sum"] = (
        datalist["sio2"]
        + datalist["tio2"]
        + datalist["al2o3"]
        + datalist["fe2o3"]
        + datalist["feo"]
        + datalist["mno"]
        + datalist["na2o"]
        + datalist["k2o"]
        + datalist["mgo"]
        + datalist["cao"]
        + datalist["p2o5"]
        + datalist["h2o"]
    )

    return datalist.copy()


########################################
# PREPROCESSING FUNCTIONS
########################################
def stratified_group_splitting(
    dataset, target, verbose=False, random_state=167, n_splits=5
):
    """performs a stratified group splitting of the dataset

    Parameters
    ----------
    dataset : pandas dataframe
        dataset to split
    target : str
        name of the target column
    verbose : bool
        if True, prints the number of samples in each set
    random_state : int
        random seed
    n_splits : int
        number of splits to perform

    Returns
    -------
    dataset_train : pandas dataframe
        training data subset
    dataset_valid : pandas dataframe
        validation data subset
    dataset_test : pandas dataframe
        testing data subset
    """

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )
    for i, (train_index, vt_index) in enumerate(
        sgkf.split(dataset, class_data(dataset), dataset[target])
    ):
        if i == 0:  # we grab the first fold
            t_i, tv_i = train_index, vt_index

    dataset_train = dataset.loc[t_i, :].reset_index()
    dataset_vt = dataset.loc[tv_i, :].reset_index()

    sgkf = StratifiedGroupKFold(n_splits=2, random_state=random_state, shuffle=True)
    for i, (valid_index, test_index) in enumerate(
        sgkf.split(dataset_vt, class_data(dataset_vt), dataset_vt.Name)
    ):
        if i == 0:  # we grab the first fold
            v_i, ts_i = valid_index, test_index

    dataset_valid = dataset_vt.loc[v_i, :].reset_index()
    dataset_test = dataset_vt.loc[ts_i, :].reset_index()

    if verbose == True:

        nb_train_compo = len(dataset_train["Name"].unique())
        nb_valid_compo = len(dataset_valid["Name"].unique())
        nb_test_compo = len(dataset_test["Name"].unique())
        nb_tot = nb_train_compo + nb_valid_compo + nb_test_compo
        print("Unique compositions in the train, valid and test subsets:")
        print(
            "train {}, valid {}, test {}".format(
                nb_train_compo, nb_valid_compo, nb_test_compo
            )
        )
        print("this makes:")
        print(
            "train {:.1f}%, valid {:.1f}%, test {:.1f}%".format(
                nb_train_compo / nb_tot * 100,
                nb_valid_compo / nb_tot * 100,
                nb_test_compo / nb_tot * 100,
            )
        )

        print("\nDetection of group (composition) leackage: between\n and train-test:")
        print(
            "{} leacked composition between train and valid subsets".format(
                np.sum(dataset_train.Name.isin(dataset_valid.Name).astype(int))
            )
        )
        print(
            "{} leacked composition between train and test subsets".format(
                np.sum(dataset_train.Name.isin(dataset_test.Name).astype(int))
            )
        )

    return dataset_train, dataset_valid, dataset_test


def class_data(chemical_set):
    """class data in different chemical systems

    Parameters
    ----------
    chemical_set : pandas dataframe
        a dataframe containing the chemical data

    Returns
    -------
    class_of_data : numpy array
        an array containing the class of each sample
    """
    # get only the relevant things from chemical_set

    chemical_set = chemical_set.loc[
        :,
        [
            "sio2",
            "tio2",
            "al2o3",
            "fe2o3",
            "feo",
            "mno",
            "na2o",
            "k2o",
            "mgo",
            "cao",
            "p2o5",
            "h2o",
        ],
    ].copy()
    # we regroup here Fe2O3 and FeO for simplicity
    chemical_set.feo = chemical_set.feo + chemical_set.fe2o3
    # dropping the Fe2O3 columns
    chemical_set = chemical_set.loc[
        :,
        [
            "sio2",
            "tio2",
            "al2o3",
            "feo",
            "mno",
            "na2o",
            "k2o",
            "mgo",
            "cao",
            "p2o5",
            "h2o",
        ],
    ].values 

    # an integer array to contain my classes, initialized to 0
    class_of_data = np.zeros(len(chemical_set), dtype=int)

    # sio2-al2o3 (sio2 included), class 1
    class_of_data[
        (chemical_set[:, [0, 2]] >= 0).all(axis=1)
        & (chemical_set[:, [1, 3, 4, 5, 6, 7, 8, 9, 10]] == 0).all(axis=1)
    ] = 1

    # all silicate systems, no Al2O3, more than ternary, class 2
    class_of_data[(chemical_set[:, 2] == 0)] = 2

    # sio2-na2o, class 3
    class_of_data[
        (chemical_set[:, [0, 5]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 3, 4, 6, 7, 8, 9, 10]] == 0).all(axis=1)
    ] = 3

    # sio2-k2o, class 4
    class_of_data[
        (chemical_set[:, [0, 6]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 3, 4, 5, 7, 8, 9, 10]] == 0).all(axis=1)
    ] = 4

    # sio2-mgo, class 5
    class_of_data[
        (chemical_set[:, [0, 7]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 3, 4, 5, 6, 8, 9, 10]] == 0).all(axis=1)
    ] = 5

    # sio2-cao, class 6
    class_of_data[
        (chemical_set[:, [0, 8]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 3, 4, 5, 6, 7, 9, 10]] == 0).all(axis=1)
    ] = 6

    # sio2-alkali ternary, class 7
    class_of_data[
        (chemical_set[:, [0, 5, 6]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 3, 4, 7, 8, 9, 10]] == 0).all(axis=1)
    ] = 7

    # sio2-alkaline-earth ternary, class 8
    class_of_data[
        (chemical_set[:, [0, 7, 8]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 3, 4, 5, 6, 9, 10]] == 0).all(axis=1)
    ] = 8

    # al2o3-cao, class 9
    class_of_data[
        (chemical_set[:, [2, 8]] > 0).all(axis=1)
        & (chemical_set[:, [0, 1, 3, 4, 5, 6, 7, 9, 10]] == 0).all(axis=1)
    ] = 9

    # sio2-al2o3-na2o, class 10
    class_of_data[
        (chemical_set[:, [0, 2, 5]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 4, 6, 7, 8, 9, 10]] == 0).all(axis=1)
    ] = 10

    # sio2-al2o3-k2o, class 11
    class_of_data[
        (chemical_set[:, [0, 2, 6]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 4, 5, 7, 8, 9, 10]] == 0).all(axis=1)
    ] = 11

    # sio2-al2o3-na2o-k2o, class 12
    class_of_data[
        (chemical_set[:, [0, 2, 5, 6]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 4, 7, 8, 9, 10]] == 0).all(axis=1)
    ] = 12

    # sio2-al2o3-mgo, class 13
    class_of_data[
        (chemical_set[:, [0, 2, 7]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 4, 5, 6, 8, 9, 10]] == 0).all(axis=1)
    ] = 13

    # sio2-al2o3-cao, class 14
    class_of_data[
        (chemical_set[:, [0, 2, 8]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 4, 5, 6, 7, 9, 10]] == 0).all(axis=1)
    ] = 14

    # sio2-al2o3-mgo-cao, class 15
    class_of_data[
        (chemical_set[:, [0, 2, 7, 8]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 4, 5, 6, 9, 10]] == 0).all(axis=1)
    ] = 15

    # sio2-al2o3-na2o-k2o-mgo-cao, class 16
    class_of_data[
        (chemical_set[:, [0, 2, 5, 6, 7, 8]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 4, 9, 10]] == 0).all(axis=1)
    ] = 16

    # sio2-feo, class 17
    class_of_data[
        (chemical_set[:, [0, 3]] > 0).all(axis=1)
        & (chemical_set[:, [1, 2, 4, 5, 6, 7, 8, 9, 10]] == 0).all(axis=1)
    ] = 17

    # sio2-al2o3-feo-na2o-k2o-mgo-cao, class 18
    class_of_data[
        (chemical_set[:, [0, 2, 4, 5, 6, 7, 8]] > 0).all(axis=1)
        & (chemical_set[:, [1, 3, 9, 10]] == 0).all(axis=1)
    ] = 18

    # Ti-bearing melts, anhydrous, class 19
    class_of_data[(chemical_set[:, 1] > 0) & (chemical_set[:, 10] == 0)] = 19

    # Hydrous melts, class 20
    class_of_data[(chemical_set[:, 10] > 0)] = 20

    # P-bearing melts, anhydrous, class 21
    class_of_data[(chemical_set[:, 9] > 0) & (chemical_set[:, 10] == 0)] = 21

    return class_of_data


def generate_token(dataset):
    """generate tokens for each unique composition in the dataset

    FeO and Fe2O3 are counted together

    We assume compositions are unique at 1 decimal precision

    Parameters
    ----------
    dataset : pandas dataframe
        The composition of the melt

    Returns
    -------
    token : list
        list of token names
    """

    # get the chemical composition
    compo = dataset.loc[:, list_oxides()].copy()

    # we sum FeO and Fe2O3
    compo["feo"] = compo["feo"] + 0.5 * compo["fe2o3"]
    # drop Fe2O3
    compo = compo.drop("fe2o3", axis=1).copy().to_numpy()
    # renormalise so that the composition is normalised again
    compo = compo / np.sum(compo, axis=1).reshape(-1, 1) * 100

    # we want differenty compositions at 1 decimal precision
    compo = np.round(compo, decimals=1)

    # token variable as a list
    token = []

    # loop to create the token for each entries in compo
    for i in range(len(compo)):

        token.append(
            "S"
            + str(compo[i, 0])
            + "T"
            + str(compo[i, 1])
            + "A"
            + str(compo[i, 2])
            + "F"
            + str(compo[i, 3])
            + "M"
            + str(compo[i, 4])
            + "N"
            + str(compo[i, 5])
            + "K"
            + str(compo[i, 6])
            + "Mg"
            + str(compo[i, 7])
            + "C"
            + str(compo[i, 8])
            + "P"
            + str(compo[i, 9])
            + "H"
            + str(compo[i, 10])
        )

    return token


def prepare_viscosity_hp(
    dataset, dataset_hp, output_file, savefig=True, rand_state=67, verbose=True
):
    """prepare the dataset of glass-forming melt viscosity for the ML model"""
    if verbose == True:
        print("Reading data...")

    # reading the Pandas dataframe
    dataset = chimie_control(dataset)
    dataset["Name"] = generate_token(dataset)
    dataset_hp = chimie_control(dataset_hp)
    dataset_hp["Name"] = generate_token(dataset_hp)

    ####
    # viscosity at room P
    # train-valid-test group stratified split
    # 80-10-10
    ####
    if verbose == True:
        print(
            "====================================================================================="
        )
        print("Preparing the 1 bar dataset with stratified group splitting...")
    train_lp, valid_lp, test_lp = stratified_group_splitting(
        dataset, "Name", verbose=verbose, random_state=rand_state
    )

    ####
    # viscosity at HP
    # we allow compositions to be in the different subsets
    # because we want to capture the effect of pressure mostly
    # 80-10-10
    ####
    if verbose == True:
        print(
            "====================================================================================="
        )
        print(
            "Preparing now the high pressure dataset, using train__test_split from Scikit-Learn..."
        )
    train_hp, tv_hp = model_selection.train_test_split(
        dataset_hp, test_size=0.20, random_state=rand_state
    )
    test_hp, valid_hp = model_selection.train_test_split(
        tv_hp, test_size=0.5, random_state=rand_state
    )
    if verbose == True:
        print(
            "The total number of unique compositions is {}.".format(
                len(dataset_hp["Name"].unique())
            )
        )

    # we join the datasets
    train_ = pd.concat((train_lp, train_hp), axis=0)
    valid_ = pd.concat((valid_lp, valid_hp), axis=0)
    test_ = pd.concat((test_lp, test_hp), axis=0)

    # grab good X columns and add descriptors
    X_train = descriptors(train_.loc[:, list_oxides()])
    X_columns = X_train.keys().to_list()
    X_train = X_train.values
    X_valid = descriptors(valid_.loc[:, list_oxides()]).values
    X_test = descriptors(test_.loc[:, list_oxides()]).values

    # temperature values
    T_train = train_.loc[:, "T"].values.reshape(-1, 1)
    T_valid = valid_.loc[:, "T"].values.reshape(-1, 1)
    T_test = test_.loc[:, "T"].values.reshape(-1, 1)

    # Names in subsets
    Name_train = train_.loc[:, "Name"].values.reshape(-1, 1)
    Name_valid = valid_.loc[:, "Name"].values.reshape(-1, 1)
    Name_test = test_.loc[:, "Name"].values.reshape(-1, 1)

    # Source of data in subsets
    Sciglass_train = train_.loc[:, "Sciglass"].values.astype('bool').reshape(-1, 1)
    Sciglass_valid = valid_.loc[:, "Sciglass"].values.astype('bool').reshape(-1, 1)
    Sciglass_test = test_.loc[:, "Sciglass"].values.astype('bool').reshape(-1, 1)

    # pressure values
    P_train = train_.loc[:, "P"].values.reshape(-1, 1)
    P_valid = valid_.loc[:, "P"].values.reshape(-1, 1)
    P_test = test_.loc[:, "P"].values.reshape(-1, 1)

    # grab the good y values
    y_train = train_.loc[:, ["viscosity"]].values.reshape(-1, 1)
    y_valid = valid_.loc[:, ["viscosity"]].values.reshape(-1, 1)
    y_test = test_.loc[:, ["viscosity"]].values.reshape(-1, 1)

    # Figure of the datasets
    if savefig == True:
        plt.figure()
        plt.subplot(121)
        plt.plot(10000 / T_train, y_train, "k.", label="train")

        plt.subplot(121)
        plt.plot(10000 / T_valid, y_valid, "b.", label="valid")

        plt.subplot(121)
        plt.plot(10000 / T_test, y_test, "r.", label="test")
        plt.savefig(output_file + ".pdf")
        plt.close()

    if verbose == True:
        print("Size of viscous training subsets:\n")
        print(X_train.shape)
        print(X_valid.shape)
        print(X_test.shape)

    # writing the data in HDF5 file for later call
    with h5py.File(output_file, "w") as f:
        f.create_dataset("X_columns", data=X_columns)

        f.create_dataset("X_train", data=X_train)
        f.create_dataset("T_train", data=T_train)
        f.create_dataset("P_train", data=P_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("Name_train", data=Name_train)
        f.create_dataset("Sciglass_train", data=Sciglass_train)

        f.create_dataset("X_valid", data=X_valid)
        f.create_dataset("T_valid", data=T_valid)
        f.create_dataset("P_valid", data=P_valid)
        f.create_dataset("y_valid", data=y_valid)
        f.create_dataset("Name_valid", data=Name_valid)
        f.create_dataset("Sciglass_valid", data=Sciglass_valid)

        f.create_dataset("X_test", data=X_test)
        f.create_dataset("T_test", data=T_test)
        f.create_dataset("P_test", data=P_test)
        f.create_dataset("y_test", data=y_test)
        f.create_dataset("Name_test", data=Name_test)
        f.create_dataset("Sciglass_test", data=Sciglass_test)


class data_loader:
    """custom data loader for batch training"""

    def __init__(
        self,
        pretreat_data=False,
        pretreat_path="./data/",
        rand_state=56,
        verbose=True,
    ):
        """
        Inputs
        ------
        pretreat_data : Bool
            If True, will pre-treat the data and create the datasets for training-validation and testing
            A file named all_viscosity.hdf5 will be saved in the folder indicated by the path variable.
        pretreat_path : string
            If pre-treatment, we path for the data
        rand_state : Int
            random seed
        verbose : Bool
            If using or not the verbose mode.
        """
        
        if pretreat_data == True:

            # path for the futur HDF5 dataset
            self.path_viscosity = pretreat_path+"all_viscosity.hdf5"

            # path of the low and high pressure CSV datasets
            self.dataset_lp = pd.read_csv(pretreat_path+"dataset+sciglass_lp.csv")
            self.dataset_hp = pd.read_csv(pretreat_path+"dataset_hp.csv")

            # we prepare the data
            prepare_viscosity_hp(
                self.dataset_lp,
                self.dataset_hp,
                self.path_viscosity,
                savefig=False,
                verbose=verbose,
                rand_state=rand_state,
            )
        else:
            self.path_viscosity = _BASEDATAPATH / "all_viscosity.hdf5"
            self.dataset_lp = pd.read_csv(_BASEDATAPATH / "dataset+sciglass_lp.csv")
            self.dataset_hp = pd.read_csv(_BASEDATAPATH / "dataset_hp.csv")

        f = h5py.File(self.path_viscosity, "r")

        # List all groups
        self.X_columns = f["X_columns"][()]

        # Viscosity dataset
        # the viscosity is set in deca Pa s
        # values will be closer to 1 this way
        self.X_train = f["X_train"][()]
        self.T_train = f["T_train"][()]
        self.P_train = f["P_train"][()]
        self.y_train = f["y_train"][()]
        self.Names_train = f["Name_train"][()]
        self.Sciglass_train = f["Sciglass_train"][()]

        self.X_valid = f["X_valid"][()]
        self.T_valid = f["T_valid"][()]
        self.P_valid = f["P_valid"][()]
        self.y_valid = f["y_valid"][()]
        self.Names_valid = f["Name_valid"][()]
        self.Sciglass_valid = f["Sciglass_valid"][()]

        self.X_test = f["X_test"][()]
        self.T_test = f["T_test"][()]
        self.P_test = f["P_test"][()]
        self.y_test = f["y_test"][()]
        self.Names_test = f["Name_test"][()]
        self.Sciglass_test = f["Sciglass_test"][()]

        f.close()

        # Scaling and preparing for blackbox
        self.TPX_train = np.concatenate(
            (self.T_train.reshape(-1, 1), self.P_train.reshape(-1, 1), self.X_train),
            axis=1,
        )
        self.TPX_valid = np.concatenate(
            (self.T_valid.reshape(-1, 1), self.P_valid.reshape(-1, 1), self.X_valid),
            axis=1,
        )
        self.TPX_test = np.concatenate(
            (self.T_test.reshape(-1, 1), self.P_test.reshape(-1, 1), self.X_test),
            axis=1,
        )

        self.TPX_train_valid = np.concatenate((self.TPX_train, self.TPX_valid), axis=0)
        self.y_train_valid = np.concatenate((self.y_train, self.y_valid))

        # Scaling and preparing for greybox
        self.PX_train = np.concatenate(
            (self.P_train.reshape(-1, 1), self.X_train), axis=1
        )
        self.PX_valid = np.concatenate(
            (self.P_valid.reshape(-1, 1), self.X_valid), axis=1
        )
        self.PX_test = np.concatenate((self.P_test.reshape(-1, 1), self.X_test), axis=1)

def Y_scale():
    """Y scale for GP
    
    standard deviation of Y data determined on the training dataset"""
    return 4.4408

def scale_for_blackbox(T, P, X, scaler):
    """scale a dataset for black box models

    temperature, pressure and compositions are combined"""

    # Handle different input types for T and P
    T = np.asarray(T).reshape(-1, 1)
    P = np.asarray(P).reshape(-1, 1)

    # Check array lengths
    if len(T) != len(P):
        raise ValueError("Temperature and pressure arrays must have the same length.")
    
    # Ensure C is a pandas DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    X = X.loc[:, list_oxides()].values

    # Handle cases where one variable has a single value and others have multiple
    if len(T) > 1 and len(X) == 1:
        X = np.tile(X, (len(T), 1))
    elif len(X) > 1 and len(T) == 1:
        T = np.tile(T, (len(X), 1))
        P = np.tile(P, (len(X), 1))

    # concatenate
    TPX = np.concatenate((T, P, X), 
                          axis=1)
    
    # Scale and return
    return scaler.transform(TPX).copy()

def scale_for_greybox(p, x, scaler):
    """scale a dataset for greybox model 1 (only pressure), for models using Tensorflow

    compositions and pressure are combined"""

    # concatenate the datasets
    px = np.concatenate((p.reshape(-1, 1), x), axis=1)

    return scaler.transform(px).copy()

def scale_for_gaussianprocess(T, P, C):
    """prepare a X dataset for gaussian process and their greybox ann mean function

    reciprocal temperature (1000/T),
    pressure (/30 GPa),
    and 12 oxide fractions are combined

    Parameters
    ----------
    T : n x 1 numpy darray
        Temperature
    P : n x 1 numpy array
        Pressure
    C : n x 12 columns Pandas DataFrame
        Composition in mole fractions (see utils.list_oxides() for the keys)

    Returns
    -------
    X : ndarray
        numpy array ready for GP inference
    """
    
    # Handle different input types for T and P
    T = np.asarray(T).reshape(-1, 1)
    P = np.asarray(P).reshape(-1, 1)

    # Check array lengths
    if len(T) != len(P):
        raise ValueError("Temperature and pressure arrays must have the same length.")
    
    # Ensure C is a pandas DataFrame
    if not isinstance(C, pd.DataFrame):
        raise ValueError("C must be a pandas DataFrame.")

    # Scaling
    T_scaled = 1000.0 / T
    P_scaled = P / 30.0
    C_scaled = C.loc[:, list_oxides()].values

    # Handle cases where one variable has a single value and others have multiple
    if len(T) > 1 and len(C_scaled) == 1:
        C_scaled = np.tile(C_scaled, (len(T), 1))
    elif len(C_scaled) > 1 and len(T) == 1:
        T_scaled = np.tile(T_scaled, (len(C_scaled), 1))
        P_scaled = np.tile(P_scaled, (len(C_scaled), 1))
        
    # Concatenate the data and return
    return np.concatenate((T_scaled, P_scaled, C_scaled), axis=1)
    
########################################
# ERROR CALCULATIONS
########################################


def evaluate_blackbox_model(ds, model, model_name="Don't forget this..."):

    y_train_pred = model.predict(ds.TPX_train_scaled)
    y_valid_pred = model.predict(ds.TPX_valid_scaled)
    y_test_pred = model.predict(ds.TPX_test_scaled)

    mse_train = mean_squared_error(ds.y_train, y_train_pred, squared=False)
    mse_valid = mean_squared_error(ds.y_valid, y_valid_pred, squared=False)
    mse_test = mean_squared_error(ds.y_test, y_test_pred, squared=False)

    r2_train = r2_score(ds.y_train, y_train_pred)
    r2_valid = r2_score(ds.y_valid, y_valid_pred)
    r2_test = r2_score(ds.y_test, y_test_pred)

    # Stats data
    print(model_name)
    print(
        "Root Mean Square Errors:",
        "\n",
        "Train    %.2f" % mse_train,
        "\n",
        "Valid    %.2f" % mse_valid,
        "\n",
        "Test     %.2f" % mse_test,
        "\n" "Coefficient of Determination:",
        "\n",
        "Train    %.2f" % r2_train,
        "\n",
        "Valid    %.2f" % r2_valid,
        "\n",
        "Test     %.2f" % r2_test,
    )


def residual_error_calc(y, y_pred, mode="BOTH"):
    """returns the root-mean-squared-error (RMSE) and median absolute error (MAE) between y an y_pred

    Parameters
    ----------
    y : array-like
        the true values
    y_pred : array-like
        the predicted values
    mode : str, optional
        the type of error to be calculated (RMSE, MAE or BOTH). By default "BOTH"

    Returns
    -------
    float or tuple
        the error or a tuple of errors

    """

    # check the type of input arrays and convert them to numpy if necessary
    # if type(y) == torch.Tensor:
    #    y = y.cpu().detach().numpy()
    # if type(y_pred) == torch.Tensor:
    #    y_pred = y_pred.cpu().detach().numpy()

    # now perform the relevant calculation
    if mode == "RMSE":  # root mean square error
        return mean_squared_error(y, y_pred, squared=False)
    elif mode == "MAE":  # median absolute deviation
        return median_absolute_error(y, y_pred)
    elif mode == "BOTH":
        rmse = mean_squared_error(y, y_pred, squared=False)
        mae = median_absolute_error(y, y_pred)
        return rmse, mae


########################################
# VISCOSITY MODELS
########################################
def G2008(data):  # for spectral fit
    """Viscosity model of Giordano et al. (2008):

    log viscosity(Pa s) = A + B / (T(K) - C),

    where A is a constant independent of composition and B and C are adjustable parameters

    Parameters
    ----------

    data : Pandas dataframe
        Input composition in mol fractions.
        Use the functions chimie_control() and then wt_mol() to properly setup the input data table.

    Returns
    -------

    visco : Numpy Array
        The viscosity in log Pa s calculated with the Giordano et al. 2008 model
    Tg : double
        The glass transition temperature (viscosity = 10^12 Pa s)
    A : double
        Parameter A of the TVF equation
    B : double
        Parameter B of the TVF equation
    C : double
        Parameter C of the TVF equation

    Reference
    ---------
    Giordano, D., Russell, J. K., & Dingwell, D. B. (2008). Viscosity of magmatic liquids: a model. Earth and Planetary Science Letters, 271(1), 123-134.

    """

    T = data.loc[:, "T"]

    # now we put things in %
    data = data * 100

    # fitting parameters:
    # see table 1 of Giordano et al. for parameter significance
    b1 = 159.560
    b2 = -173.340
    b3 = 72.130
    b4 = 75.690
    b5 = -38.980
    b6 = -84.080
    b7 = 141.540
    b11 = -2.43
    b12 = -0.91
    b13 = 17.62

    c1 = 2.75
    c2 = 15.720
    c3 = 8.320
    c4 = 10.20
    c5 = -12.290
    c6 = -99.540
    c11 = 0.300

    # calculate VFT paramters
    A = -4.55

    V = data.loc[:, "h2o"]  # we do not input F data such that it does not appear there.
    TA = data.loc[:, "tio2"] + data.loc[:, "al2o3"]
    FM = data.loc[:, "feo"] + data.loc[:, "mno"] + data.loc[:, "mgo"]
    NK = data.loc[:, "na2o"] + data.loc[:, "k2o"]

    bb1 = b1 * (data.loc[:, "sio2"] + data.loc[:, "tio2"])
    bb2 = b2 * data.loc[:, "al2o3"]
    bb3 = b3 * (data.loc[:, "feo"] + data.loc[:, "p2o5"])
    bb4 = data.loc[:, "mgo"] * b4
    bb5 = data.loc[:, "cao"] * b5
    bb6 = (data.loc[:, "na2o"] + V) * b6
    bb7 = (V + np.log(1 + data.loc[:, "h2o"])) * b7
    bb11 = (data.loc[:, "sio2"] + data.loc[:, "tio2"]) * FM * b11
    bb12 = (
        (data.loc[:, "sio2"] + TA + data.loc[:, "p2o5"])
        * (NK + data.loc[:, "h2o"])
        * b12
    )
    bb13 = (data.loc[:, "al2o3"] * NK) * b13

    B = bb1 + bb2 + bb3 + bb4 + bb5 + bb6 + bb7 + bb11 + bb12 + bb13

    cc1 = data.loc[:, "sio2"] * c1
    cc2 = TA * c2
    cc3 = FM * c3
    cc4 = data.loc[:, "cao"] * c4
    cc5 = NK * c5
    cc6 = np.log(1 + V) * c6
    cc11 = (
        (data.loc[:, "al2o3"] + FM + data.loc[:, "cao"] - data.loc[:, "p2o5"])
        * (NK + V)
        * c11
    )

    C = cc1 + cc2 + cc3 + cc4 + cc5 + cc6 + cc11

    visco = A + B / (T - C)

    Tg = B / (12 - A) + C

    return visco, Tg, A, B, C

def VFT(T,A,B,C):
    """VFT equation
    """
    return A + B / (T - C)

def C2009(phy, xi=0.1e-4, gamma=3.429, B=2.5, phy_star=0.622):
    """Calculate the relative viscosity increase generated by the presence of crystals in a melt. Default parameter values are for MORB.

    Parameters
    ----------
    phy : ndarray
        volumic fraction of crystals
    xi : float
        see paper
    gamma : float
        see paper
    B : float
        see paper
    phy_star : float
        rheological transition threshold

    Returns
    -------
    eta_r : ndarray
        relative viscosity increase generated by the presence of crystals, in units of log10 Pa.s

    source: Costa, A., Caricchi, L., Bagdassarov, N., 2009. A model for the rheology of particle-bearing suspensions and partially molten rocks. Geochemistry Geophysics Geosystems 10, 1–13.
    """

    delta = 13 - gamma
    r_phy = phy / phy_star

    F = (1 - xi) * erf(np.sqrt(np.pi) / (2 * (1 - xi)) * r_phy * (1 + r_phy**gamma))
    eta_r = (1 + r_phy**delta) / ((1 - F) ** (B * phy_star))

    return eta_r


def eta_mantle(T, phy, Ea=250.0e3, eta_0=10**21.0, T_0=1400.0, sigma_sil=21.0):
    """calculate the viscosity of a solid mantle depending on the fraction of solid/liquid

    Parameters
    ----------
    T : ndarray
        temperature in K
    phy : ndarray or float
        volumic fraction of crystals
    Ea : float
        activation energy for the viscous flow
    eta_0 : float
        viscosity of the solid at To
    T_0 : float
        reference temperature
    sigma_sil : float
        see paper

    Returns
    -------
    eta : ndarray
        relative viscosity increase generated by the presence of crystals, in units of Pa.s

    source: Scott, T., Kohlstedt, D.L., 2006. The effect of large melt fraction on the deformation behavior of peridotite. Earth and Planetary Science Letters 246, 177–187. https://doi.org/10.1016/j.epsl.2006.04.027
    """
    f_phy = np.exp(-sigma_sil * (1.0 - phy))
    term2 = np.exp(Ea / 8.314 * (1.0 / T - 1.0 / T_0))

    eta = eta_0 * f_phy * term2

    return eta

def viscosity_melt_ctx(
    n_start, T, phy, phy_m=0.65, Ea=250.0e3, eta_0=10**21.0, T_0=1400.0, sigma_sil=21.0
):
    """combines the models of Costa et al. 2009 and of Scott and Kohlstedt 2006 for
    the prediction of the viscosity of a melt-crystal mixture.

    n_start : ndarray
        viscosity of the melt phase
    T : ndarray
        temperature in K
    phy : ndarray
        volumic fraction of crystals

    Returns
    -------
    eta : ndarray
        the viscosity of the melt-crystal mixture in units of log10 Pa.s
    """

    # final viscosity vector
    eta = n_start.copy()

    # we get the effect of crystals on the melt viscosity before the rheological transition
    effet_ctx = np.log10(C2009(phy[phy < phy_m], phy_star=phy_m))
    eta[phy < phy_m] = eta[phy < phy_m].copy() + np.nan_to_num(effet_ctx)

    # after the rheological transition at phy_m, we use the good function
    eta[phy >= phy_m] = np.log10(
        eta_mantle(
            T[phy >= phy_m],
            phy[phy >= phy_m],
            Ea=Ea,
            eta_0=eta_0,
            T_0=T_0,
            sigma_sil=sigma_sil,
        )
    )

    # add the viscosity of the solid mantle affected by melt (after rheological transition)
    # ctx_ = 1/(1 + np.exp(slope*(-phy[phy>=phy_m]+phy_m)))
    # n_[phy>=phy_m] = np.log10(eta_mantle(T, phy)) #ctx_*(n_high-trans) + trans

    return eta

