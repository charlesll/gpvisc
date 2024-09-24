#!/usr/bin/env python
# coding: utf-8
# (c) Charles Le Losq, Clément Ferraina 2023-2024
# see embedded licence file
# GP-melt 1.0

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import gpvisc.utils as utils
import gpvisc.models as models
import torch
import gpytorch
import os
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CPU or GPU?
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Will run on {device}")

def load_data():
    """Load the viscosity datasets."""
    logging.info("Loading the viscosity datasets...")
    ds = utils.data_loader()
    logging.info("Loaded.")
    return ds

def prepare_data(ds):
    """Prepare training, validation, and test datasets."""
    X_train = torch.FloatTensor(np.concatenate((1000/ds.T_train, ds.P_train/30.0, ds.X_train[:,0:12]), axis=1).copy())
    X_valid = torch.FloatTensor(np.concatenate((1000/ds.T_valid, ds.P_valid/30.0, ds.X_valid[:,0:12]), axis=1).copy())
    X_test = torch.FloatTensor(np.concatenate((1000/ds.T_test, ds.P_test/30.0, ds.X_test[:,0:12]), axis=1).copy())

    Y_train = torch.FloatTensor(ds.y_train.ravel().copy()) / utils.Y_scale()
    Y_valid = torch.FloatTensor(ds.y_valid.ravel().copy()) / utils.Y_scale()
    Y_test = torch.FloatTensor(ds.y_test.ravel().copy()) / utils.Y_scale()

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

def train_mean_function(model_mean, 
                        X_train, Y_train, 
                        X_valid, Y_valid, 
                        gp_save_name, training_iter, 
                        early_criterion, lr=0.0003):
    """Train the ANN mean function."""
    mean_save_name = os.path.join(gp_save_name, 'mean.pth')
    if not os.path.exists(gp_save_name):
        os.makedirs(gp_save_name)
    
    optimizer = torch.optim.Adam(model_mean.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[21000,], gamma=0.5)
    loss_f = nn.MSELoss(reduction='mean').to(device)
    loss_train_record, loss_valid_record = [], []
    patience = 0
    best_loss = float('inf')
    
    for i in range(training_iter):
        model_mean.train()
        optimizer.zero_grad()
        output = model_mean(X_train.to(device))
        loss = loss_f(output, Y_train.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_value_(model_mean.parameters(), clip_value=0.1)
        optimizer.step()
        scheduler.step()
        
        model_mean.eval()
        with torch.no_grad():
            loss_v = loss_f(model_mean(X_valid.to(device)), Y_valid.to(device))
        
        loss_train_record.append(loss.item())
        loss_valid_record.append(loss_v.item())

        if i % 500 == 0:
            logging.info(f'Iter {i + 1}/{training_iter} - Train loss: {loss.item():.3f} - Valid loss: {loss_v.item():.3f}')
        
        if loss_v.item() < best_loss:
            torch.save(model_mean.state_dict(), mean_save_name)
            best_loss = loss_v.item()
            patience = 0
            logging.info(f'New best validation loss: {best_loss:.3f} at iteration {i + 1}')
        else:
            patience += 1
        
        if patience > early_criterion:
            logging.info(f'Early stopping at iteration {i + 1} with best validation loss: {best_loss:.3f}')
            break

    plt.figure()
    plt.plot(loss_train_record, label="Training Loss")
    plt.plot(loss_valid_record, label="Validation Loss")
    plt.xlabel("Iteration #")
    plt.ylabel("Loss value")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(gp_save_name, "mean_f_loss.pdf"))
    plt.close()

def train_gp_model(model, likelihood, X_train_valid, Y_train_valid, gp_save_name, training_iter_gp):
    """Train the GP model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600], gamma=0.5)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)
    loss_train_record = []

    for i in range(training_iter_gp):
        optimizer.zero_grad()
        output = model(X_train_valid.to(device))
        loss = -mll(output, Y_train_valid.to(device))
        loss.backward()
        
        logging.info(f'Iter {i + 1}/{training_iter_gp} - Loss: {loss.item():.3f} - Noise: {model.likelihood.noise.item():.3f}')
        loss_train_record.append(loss.item())
        
        optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(gp_save_name, "gp.pth"))
    torch.save(likelihood.state_dict(), os.path.join(gp_save_name, "likelihood.pth"))

    torch.save(X_train_valid, os.path.join(gp_save_name, "X_train_valid.pth"))
    torch.save(Y_train_valid, os.path.join(gp_save_name, "Y_train_valid.pth"))

    plt.figure()
    plt.plot(loss_train_record, label="Training Loss")
    plt.xlabel("Iteration #")
    plt.ylabel("Loss value")
    plt.legend()
    plt.savefig(os.path.join(gp_save_name, "gp_loss.pdf"))

def evaluate_model(model, likelihood, X_train_valid, X_test, Y_train_valid, Y_test, ds, gp_save_name):
    """Evaluate the trained model."""
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_train_preds = likelihood(model(X_train_valid.to(device)))
        y_test_preds = likelihood(model(X_test.to(device)))

    rmse_train = mean_squared_error(Y_train_valid.cpu().numpy() * utils.Y_scale(), y_train_preds.mean.cpu().numpy() * utils.Y_scale(), squared=False)
    diff_se_test = (Y_test.cpu().numpy() * utils.Y_scale() - y_test_preds.mean.cpu().numpy() * utils.Y_scale()) ** 2
    rmse_test = np.sqrt(np.mean(diff_se_test))
    diff_se_test_handdataset = diff_se_test[ds.Sciglass_test[:, 0] == False]
    rmse_test_handdataset = np.sqrt(np.mean(diff_se_test_handdataset))

    metrics = {"train": [rmse_train], "test": [rmse_test], "handtest": [rmse_test_handdataset]}
    metrics_pandas = pd.DataFrame(metrics)
    metrics_pandas.to_csv(os.path.join(gp_save_name, "metrics.csv"))

def main(gp_save_name, training_iter, early_criterion, training_iter_gp):

    # loading data
    ds = load_data()
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prepare_data(ds)

    # declaring and pretraining the mean function
    model_mean = models.mean_f(hidden_size=[200, 200], activation_function=torch.nn.GELU(), p_drop=0.1).to(device)
    train_mean_function(model_mean, X_train, Y_train, X_valid, Y_valid, gp_save_name, training_iter, early_criterion)

    # now we don't need the validation dataset, we glue things together for final training
    X_train_valid = torch.cat((X_train, X_valid)).clone()
    Y_train_valid = torch.cat((Y_train, Y_valid)).clone().ravel()
    
    # declaring the GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = models.ExactGPModel(X_train_valid.to(device), Y_train_valid.to(device), likelihood).to(device)
   
    # loading the pre-trained mean function parameters and freezing them
    model.mean_f.load_state_dict(torch.load(os.path.join(gp_save_name, 'mean.pth'), map_location=torch.device(device)))
    for param in model.mean_f.parameters():
        param.requires_grad = False

    # training and evaluating the GP
    train_gp_model(model, likelihood, X_train_valid, Y_train_valid, gp_save_name, training_iter_gp)
    evaluate_model(model, likelihood, X_train_valid, X_test, Y_train_valid, Y_test, ds, gp_save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GP-melt model training')
    parser.add_argument('--gp_save_name', type=str, default="./models/test/", help='Path to save the GP model')
    parser.add_argument('--training_iter', type=int, default=50000, help='Number of training iterations for the mean function')
    parser.add_argument('--early_criterion', type=int, default=10000, help='Early stopping criterion for the mean function')
    parser.add_argument('--training_iter_gp', type=int, default=1000, help='Number of training iterations for the GP')

    args = parser.parse_args()
    main(args.gp_save_name, 
         args.training_iter, 
         args.early_criterion, 
         args.training_iter_gp)
