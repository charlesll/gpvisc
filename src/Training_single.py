#!/usr/bin/env python
# coding: utf-8
# (c) Charles Le Losq, Clément Ferraina 2023
# see embedded licence file
# iVisc 1.0
# this script is meant to be loaded from the parent directory

#
# Library Loading
#
import pandas as pd # manipulate dataframes
import matplotlib.pyplot as plt # plotting
import numpy as np
np.random.seed = 167 # fix random seed for reproducibility

import time, os

# local imports
import src.utils as utils
import src.ivisc as ivisc

# deep learning libraries
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import torch

# import sklearn utils
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

from tqdm import tqdm

from joblib import dump, load
#
# First we check if CUDA is available
#
device= utils.get_default_device()

#
# Data Loading
#
    
# Data loading
print("Loading the viscosity datasets...")
ds = ivisc.data_loader(path_viscosity='./data/all_viscosity.hdf5')
ds.print_data()
print("Loaded.")

# reference architecture: 4 layers, 400 neurons per layer
# Parameters were tuned after the random search, & learning rate by Bayesian Optimization & patience by hand.

nb_neurons = [400,400,400,400]
nb_layers = len(nb_neurons)
p_drop = 0.1

print("Network architecture is: {} layers, {} neurons/layers, dropout {}".format(nb_layers,nb_neurons,p_drop))

# Create directories if they do not exist
utils.create_dir('./models/')
utils.create_dir('./figures/')
utils.create_dir('./outputs/')
utils.create_dir('./figures/single/')
utils.create_dir('./outputs/single/')

name = "./models/l"+str(nb_layers)+"_n"+str(nb_neurons)+"_p"+str(p_drop)+"_GELU_cpfree_test"+".pth"

# declaring model
neuralmodel = ivisc.ivisc(ds.x_visco_train.shape[1],
                          hidden_size=nb_neurons,
                          activation_function = torch.nn.GELU(), # activation units in FFN
                          shape = "rectangle", 
                          p_drop=p_drop)

# criterion for match
criterion = torch.nn.MSELoss(reduction='mean')
criterion.to(device) # sending criterion on device

# we initialize the output bias
neuralmodel.output_bias_init()
# float network
neuralmodel = neuralmodel.float()
neuralmodel.to(device)