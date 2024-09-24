import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import gpytorch
import torch
import torch.nn as nn

import os
from pathlib import Path

import gpvisc

# to load models in a library
_BASEMODELPATH = Path(os.path.dirname(__file__)) / "models"

################
### GPYTORCH ###
################

class mean_f(torch.nn.Module):
    """greybox artificial neural network for using as a mean function of the GP model
    
    Viscosity dependence to temperature is constrained by the use of the VFT equation

    Pytorch framework
    """
    def __init__(self, input_size=13, hidden_size = [200, 200],
                 p_drop=0.1, activation_function = torch.nn.GELU()):
        super(mean_f, self).__init__()
        # network related torch stuffs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.p_drop = p_drop
        self.dropout = torch.nn.Dropout(p=p_drop)
        self.num_layers = len(hidden_size)

        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.input_size, self.hidden_size[0])])
        self.linears.extend([torch.nn.Linear(self.hidden_size[i], self.hidden_size[i+1]) for i in range(self.num_layers-1)])
        
        # output layer
        self.out = torch.nn.Linear(self.hidden_size[-1],2)#
        self.out.bias = torch.nn.Parameter(data=torch.FloatTensor([7.0, # B
                                                                   3.0, # C
                                                                   ]))
        self.A = torch.nn.Parameter(torch.tensor([-4.5]), requires_grad=True)

    def forward(self, x):
        """foward pass in the neural network"""
        # get different variables
        k = torch.reshape(x[:,0], (x.shape[0],1)) # inverse of T
        compo = x[:,1:] # pressure and melt composition

        # calculation in the neural network
        for linear in self.linears:
            compo = linear(compo)
            compo  = self.dropout(self.activation_function(compo))
        output_ann = torch.exp(self.out(compo))
        #output_ann = self.out(compo)
        
        # get B and C in the good shape
        B  = torch.reshape(output_ann[:,0], (output_ann.shape[0],1))
        C  = torch.reshape(output_ann[:,1], (output_ann.shape[0],1))
        
        # calculate viscosity
        viscosity = (self.A + B/(1000.0/k-C))/gpvisc.Y_scale()
        
        return torch.reshape(viscosity, (len(viscosity),))
    
class mean_f_P(torch.nn.Module):
    """greybox artificial neural network for using as a mean function of the GP model
    
    Viscosity dependence to temperature is constrained by the use of the VFT equation

    Pytorch framework
    """
    def __init__(self, input_size=13, hidden_size = [200, 200],
                 p_drop=0.1, activation_function = torch.nn.GELU(),
                 hidden_size_2 = [10,]):
        super(mean_f_P, self).__init__()
        # network related torch stuffs

        # general network 1 : no pressure
        # provides B0 in B0 + B1*P + B2*P*P
        # and C
        self.input_size = input_size-1
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.p_drop = p_drop
        self.dropout = torch.nn.Dropout(p=p_drop)
        self.num_layers = len(hidden_size)

        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.input_size, self.hidden_size[0])])
        self.linears.extend([torch.nn.Linear(self.hidden_size[i], self.hidden_size[i+1]) for i in range(self.num_layers-1)])
        
        # output layer
        self.out = torch.nn.Linear(self.hidden_size[-1],2)#
        self.out.bias = torch.nn.Parameter(data=torch.FloatTensor([7.0, # B
                                                                   3.0, # C
                                                                   ]))
        
        # network 2: pressure network
        # provides B1 and B2
        # simple one layer network
        self.input_size_2 = input_size
        self.hidden_size_2 = hidden_size_2
        self.num_layers_2 = len(hidden_size_2)

        self.linears_2 = torch.nn.ModuleList([torch.nn.Linear(self.input_size_2, self.hidden_size_2[0])])
        self.linears_2.extend([torch.nn.Linear(self.hidden_size_2[i], self.hidden_size_2[i+1]) for i in range(self.num_layers_2-1)])

        self.out_2 = torch.nn.Linear(self.hidden_size_2[-1],2)
        self.out_2.bias = torch.nn.Parameter(data=torch.FloatTensor([-1.2, # dB
                                                                   -6.21, # B2
                                                                   ]))
        
        # common A and dP parameters
        self.A = torch.nn.Parameter(torch.tensor([-4.5]), requires_grad=True)
        self.dP = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=True)

    def forward_1(self, x):
        compo = x[:,2:] # melt composition
        # calculation in the first neural network for B0 and C
        for linear in self.linears:
            compo = linear(compo)
            compo  = self.dropout(self.activation_function(compo))
        return torch.exp(self.out(compo))

    def forward_2(self, x):
        compo_P = x[:,1:] # pressure and melt composition
        # get B1 and B2
        for linear in self.linears_2:
            compo_P = linear(compo_P)
            compo_P = self.activation_function(compo_P)
        return torch.exp(self.out_2(compo_P))

    def forward(self, x):
        """foward pass in the neural network"""
        # get different variables
        k = torch.reshape(x[:,0], (x.shape[0],1)) # inverse of T
        P = torch.reshape(x[:,1], (x.shape[0],1)) # pressure
        
        output_ann = self.forward_1(x)
        output_ann_2 = self.forward_2(x)

        # get B and C in the good shape
        B0  = torch.reshape(output_ann[:,0], (output_ann.shape[0],1))
        C  = torch.reshape(output_ann[:,1], (output_ann.shape[0],1))

        # get B1 and B2 in the good shape
        B1 = torch.reshape(output_ann_2[:,0], (output_ann_2.shape[0],1))
        B2 = torch.reshape(output_ann_2[:,1], (output_ann_2.shape[0],1))

        # calculate viscosity
        B = B0 + B1*P +B2*(P**2)
        viscosity = (self.A + B/(1000.0/k-C))/gpvisc.Y_scale()
        
        return torch.reshape(viscosity, (len(viscosity),))
    
class mean_fb(torch.nn.Module):
    """blackbox artificial neural network for using as a mean function of the GP model

    Pytorch framework
    """
    def __init__(self, 
                 hidden_size = [50,],
                 p_drop=0.1, 
                 activation_function = torch.nn.GELU()):
        super(mean_fb, self).__init__()
        # network related torch stuffs
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.p_drop = p_drop
        self.dropout = torch.nn.Dropout(p=p_drop)
        self.num_layers = len(hidden_size)

        self.linears = torch.nn.ModuleList([torch.nn.Linear(14, self.hidden_size[0])])
        self.linears.extend([torch.nn.Linear(self.hidden_size[i], self.hidden_size[i+1]) for i in range(self.num_layers-1)])
        
        # output layer
        self.out = torch.nn.Linear(self.hidden_size[-1],1)#

    def forward(self, x):
        """foward pass in the neural network"""
        for linear in self.linears:
            x = linear(x)
            x  = self.dropout(self.activation_function(x))
        output_ann = self.out(x)
        return torch.reshape(output_ann, (len(output_ann),))
        
class ExactGPModel(gpytorch.models.ExactGP):
    """Gaussian process model
    
    GPyTorch framework
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        # get the mean function
        self.mean_f = mean_f(hidden_size=[200, 200], 
                             activation_function = torch.nn.GELU(), 
                             p_drop=0.0)
        
        # A smooth RBF kernel for long trends (particularly P and T)
        # with priors (very vague on melt composition)
        # those prior values were determined with using first a constant mean function
        prior_l= torch.FloatTensor([0.7, 1.5, 0.66, 0.41, 0.29, 2.24,
         0.88, 0.41, 0.45, 0.41, 1.43, 0.92, 3.81, 0.19])
        prior_l_std = torch.FloatTensor([0.1, 0.1,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0,
                                         1.0,  1.0,  1.0,  1.0,  1.0,  1.0])
                                                 
        # A Matern kernel 5/2
        kernel_2 = gpytorch.kernels.MaternKernel(nu=5/2, 
                                                 active_dims=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], 
                                                 ard_num_dims=14,
                                                 lengthscale_prior=gpytorch.priors.NormalPrior(prior_l, 
                                                                                            prior_l_std))
        kernel_2.lengthscale = torch.tensor([0.7, 1.5, 0.66, 0.41, 0.29, 2.24,
         0.88, 0.41, 0.45, 0.41, 1.43, 0.92, 3.81, 0.19])
        
        # build the final kernel
        self.base_covar_module = gpytorch.kernels.ScaleKernel(kernel_2)

        # Get some inducing points
        # after testing several, the best method is by K-mean clustering
        k_means = KMeans(init="k-means++", n_clusters=1024, n_init=10)
        k_means.fit(train_x.cpu().detach().numpy())

        # build the covariance kernel with the inducing points
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, 
                                                inducing_points=torch.FloatTensor(k_means.cluster_centers_).clone(), 
                                                likelihood=likelihood)
            
    def forward(self, x):
        """forward function, for training"""
        mean_x = self.mean_f(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def load_gp_model(model_number = 1, device="cpu"):
        """load a pre-trained GP model.
        
        Parameters
        ----------
        model_number : Int
            Three models are available, choose between 1, 2 or 3.
        device : string
            "cpu" or "gpu", the device you will run the calculations on.
        default_path : bool
            True or False, if you are using the provided model or want to provide a custom path

        Returns
        -------
        gp_model : GPyTorch model object
            the GP model
        likelihood : GPyTorch likelihood object
            the GP likelihood
        """
        if model_number == 1:
            model_path = _BASEMODELPATH / "GP_model1"
        elif model_number == 2:
            model_path = _BASEMODELPATH / "GP_model2"
        elif model_number == 3:
            model_path = _BASEMODELPATH / "GP_model3"
        else:
            raise ValueError("Choose between models 1, 2 or 3")
        
        # load the data that were used for training
        X_train_valid = torch.load(model_path / "X_train_valid.pth")
        Y_train_valid = torch.load(model_path / "Y_train_valid.pth")
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = ExactGPModel(X_train_valid, Y_train_valid, likelihood)

        likelihood.load_state_dict(torch.load(model_path / "likelihood.pth", map_location=torch.device(device)))
        gp_model.load_state_dict(torch.load(model_path / "gp.pth", map_location=torch.device(device)))

        likelihood.eval()
        gp_model.eval()

        return gp_model, likelihood

def predict(x, gp_model, likelihood, model_to_use="gp", device="cpu"):
    """predicts mean viscosity and standard deviation given melt composition x, and a GP model and likelihood
    
    Internally, the function calls likelihood(gp_model(x.to(device)))

    Parameters
    ==========
    x : numpy array, pandas DataFrame, or torch.Tensor
        Model input containing the melt composition.
    gp_model : GPyTorch model
        The Gaussian Process model.
    likelihood : GPyTorch likelihood
        The likelihood function.
    model_to_use : string
        the model to use to make predictions. "ann" is for the greybox artificial
         neural network, and "gp" for the Gaussian process implementation. 
         If using "ann", standard deviations on prediction will NOT be provided.
    device : str, optional (default="cpu")
        Device to perform computations on ("cpu" or "cuda").

    Returns
    =======
    mean : ndarray
        Numpy array containing the mean viscosity.
    standard_deviation : ndarray
        Numpy array containing the standard deviation.
    """
    # Handle different input types for x
    if isinstance(x, np.ndarray):
        x = torch.FloatTensor(x)  # Convert to FloatTensor
    elif isinstance(x, pd.DataFrame):
        x = torch.FloatTensor(x.values)  # Extract values and convert to Tensor
    # If x is already a torch.Tensor, no conversion is needed

    # sending to device
    likelihood.to(device)
    gp_model.to(device)

    # evaluation model
    gp_model.eval()
    likelihood.eval()
    
    if model_to_use == "gp":
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_gp = likelihood(gp_model(x.to(device)))

            visco_mean = y_gp.mean.cpu().detach().numpy()*gpvisc.Y_scale()
            visco_std = y_gp.stddev.cpu().detach().numpy()*gpvisc.Y_scale()
            return visco_mean, visco_std
    elif model_to_use == "ann":
        with torch.no_grad():
            visco_mean = gp_model.mean_f(x.to(device)).cpu().detach().numpy()*gpvisc.Y_scale()
            return visco_mean
    else:
        raise ValueError("model_to_use should be set to gp or ann.")

###
# HEAT CAPACITY MODEL
###

def aCpl(x):
    """calculate term a in equation Cpl = aCpl + bCpl*T

    Partial molar Cp are from Richet 1985, etc.

    Parameters
    ----------
    x : compositional pandas dataframe (mole fractions)
    
    Returns
    -------
    aCpl: an array of the term a in equation Cpl = aCpl + bCpl*T

    References
    ----------
    Bouhifd, M.A., Whittington, A.G., Withers, A.C., Richet, P., 2013. Heat capacities of hydrous silicate glasses and liquids. Chemical Geology, 9th Silicate Melts Workshop 346, 125–134. https://doi.org/10.1016/j.chemgeo.2012.10.026

    Courtial, P., Richet, P., 1993. Heat capacity of magnesium aluminosilicate melts. Geochimica et Cosmochimica Acta 57, 1267–1275. https://doi.org/10.1016/0016-7037(93)90063-3

    Richet, P., Bottinga, Y., 1985. Heat capacity of aluminum-free liquid silicates. Geochimica et Cosmochimica Acta 49, 471–486. https://doi.org/10.1016/0016-7037(85)90039-0

    """
    # Richet 1985
    out = (81.37*x.sio2.values # Cp liquid SiO2
         + 75.21*x.tio2.values # Richet 1985 
         + 130.2*x.al2o3.values # Courtial R. 1993
         + 199.7*x.fe2o3.values # Richet 1985
         + 78.94*x.fe2o3.values # Richet 1985
         + 82.73*x.mno.values # Richet 1985
         + 100.6*x.na2o.values # Richet 1985
         + 50.13*x.k2o.values + x.sio2.values*(x.k2o.values*x.k2o.values)*151.7 # Cp liquid K2O (Richet 1985)
         + 85.78*x.mgo.values # Cp liquid MgO (Richet 1985)
         + 86.05*x.cao.values # Cp liquid CaO (Richet 1985)
         + 86.05*x.p2o5.values # Cp liquid P2O5 (Richet 1985)
         + 75.3*x.h2o.values # Cp H2O
           )

    return out

def bCpl(self, x):
    """calculate term b in equation Cpl = aCpl + bCpl*T

    only apply B terms on Al and K

    Parameters
    ----------
    x : compositional pandas dataframe (mole fractions)
    
    Returns
    -------
    bCpl: an array of the term a in equation Cpl = aCpl + bCpl*T

    References
    ----------
    Richet, P., Bottinga, Y., 1985. Heat capacity of aluminum-free liquid silicates. Geochimica et Cosmochimica Acta 49, 471–486. https://doi.org/10.1016/0016-7037(85)90039-0
    """
    return 0.09428*x.al2o3.values + 0.01578*x.k2o.values

def Cp_liquid(x, T):
    """calculate the liquid heat capacity at 1 atm
    
    It is calculated as Cpl = aCpl + bCpl*T
    see for details the docstrings of the aCpl and bCpl functions

    Parameters
    ----------
    x : compositional pandas dataframe (mole fractions)
    T : temperature, same length as x
    
    Returns
    -------
    Cpl: the liquid heat capacity given composition x and temperature T (at 1 atm)

    """
    return aCpl(x) + bCpl(x)*T

    