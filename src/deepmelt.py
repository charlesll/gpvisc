# (c) Charles Le Losq 2022
# see embedded licence file
# imelt V1.2

import numpy as np
import torch, time
import torch.nn.functional as F
import h5py
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.constants import Avogadro, Planck

###
### FUNCTIONS FOR HANDLNG DATA
###

class data_loader():
    """custom data loader for batch training

    """
    def __init__(self, 
                 path_viscosity = "./data/all_viscosity.hdf5", 
                 path_raman = "./data/NKCMAS_Raman.hdf5", 
                 path_density = "./data/NKCMAS_density.hdf5", 
                 path_ri = "./data/NKCMAS_optical.hdf5", 
                 path_cp = "./data/NKCMAS_cp.hdf5", 
                 path_elastic = "./data/NKCMAS_em.hdf5",
                 path_cte = "./data/NKCMAS_cte.hdf5",
                 path_abbe = "./data/NKCMAS_abbe.hdf5",
                 path_liquidus = "./data/NKCMAS_tl.hdf5",
                 scaling = False):
        """
        Inputs
        ------
        path_viscosity : string
            path for the viscosity HDF5 dataset
        path_raman : string
            path for the Raman spectra HDF5 dataset
        path_density : string
            path for the density HDF5 dataset
        path_ri : String
            path for the refractive index HDF5 dataset
        path_cp : String
            path for the liquid heat capacity HDF5 dataset
        path_elastic : String
            path for the elastic moduli HDF5 dataset
        path_cte : String
            path for the thermal expansion HDF5 dataset
        path_abbe : String
            path for the Abbe number HDF5 dataset
        path_liquidus : String
            path for the liquidus temperature HDF5 dataset
        scaling : False or True
            Scales the input chemical composition.
            WARNING : Does not work currently as this is a relic of testing this effect,
            but we chose not to scale inputs and Cp are calculated with unscaled values in the network.
        """
        
        f = h5py.File(path_viscosity, 'r')

        # List all groups
        self.X_columns = f['X_columns'][()]

        # Entropy dataset
        X_entropy_train = f["X_entropy_train"][()]
        y_entropy_train = f["y_entropy_train"][()]

        X_entropy_valid = f["X_entropy_valid"][()]
        y_entropy_valid = f["y_entropy_valid"][()]

        X_entropy_test = f["X_entropy_test"][()]
        y_entropy_test = f["y_entropy_test"][()]

        # Viscosity dataset
        X_train = f["X_train"][()]
        T_train = f["T_train"][()]
        P_train = f["P_train"][()]
        y_train = f["y_train"][()]

        X_valid = f["X_valid"][()]
        T_valid = f["T_valid"][()]
        P_valid = f["P_valid"][()]
        y_valid = f["y_valid"][()]

        X_test = f["X_test"][()]
        T_test = f["T_test"][()]
        P_test = f["P_test"][()]
        y_test = f["y_test"][()]

        f.close()

        # preparing data for pytorch

        # Scaler
        # We apply min-max scaling to the descriptors
        # that are not mole fractions
        # starting from column 12
        if scaling ==  True:
            self.X_scaler_min = np.min(X_train[:,12:], axis=0)
            self.X_scaler_max = np.max(X_train[:,12:], axis=0)
        else:
            self.X_scaler_min = 0.0
            self.X_scaler_max = 0.0

        # The following lines perform scaling (not needed, not active),
        # put the data in torch tensors and send them to device (GPU or CPU, as requested) not anymore

        # viscosity
        self.x_visco_train = self.generate_tensor(X_train, scaling=scaling, min=self.X_scaler_min, max=self.X_scaler_max)
        self.T_visco_train = self.generate_tensor(T_train.reshape(-1,1), scaling=False)
        self.P_visco_train = self.generate_tensor(P_train.reshape(-1,1), scaling=False)
        self.y_visco_train = self.generate_tensor(y_train[:,0].reshape(-1,1), scaling=False)

        self.x_visco_valid = self.generate_tensor(X_valid, scaling=scaling, min=self.X_scaler_min, max=self.X_scaler_max)
        self.T_visco_valid = self.generate_tensor(T_valid.reshape(-1,1), scaling=False)
        self.P_visco_valid = self.generate_tensor(P_valid.reshape(-1,1), scaling=False)
        self.y_visco_valid = self.generate_tensor(y_valid[:,0].reshape(-1,1), scaling=False)

        self.x_visco_test = self.generate_tensor(X_test, scaling=scaling, min=self.X_scaler_min, max=self.X_scaler_max)
        self.T_visco_test = self.generate_tensor(T_test.reshape(-1,1), scaling=False)
        self.P_visco_test = self.generate_tensor(P_test.reshape(-1,1), scaling=False)
        self.y_visco_test = self.generate_tensor(y_test[:,0].reshape(-1,1), scaling=False)

        # entropy
        self.x_entro_train = self.generate_tensor(X_entropy_train, scaling=scaling, min=self.X_scaler_min, max=self.X_scaler_max)
        self.y_entro_train = self.generate_tensor(y_entropy_train[:,0].reshape(-1,1), scaling=False)

        self.x_entro_valid = self.generate_tensor(X_entropy_valid, scaling=scaling, min=self.X_scaler_min, max=self.X_scaler_max)
        self.y_entro_valid = self.generate_tensor(y_entropy_valid[:,0].reshape(-1,1), scaling=False)

        self.x_entro_test = self.generate_tensor(X_entropy_test, scaling=scaling, min=self.X_scaler_min, max=self.X_scaler_max)
        self.y_entro_test = self.generate_tensor(y_entropy_test[:,0].reshape(-1,1), scaling=False)

    def recall_order(self):
        print("Order of chemical components is sio2, al2o3, na2o, k2o, mgo, cao, then descriptors")

    def generate_tensor(self, X, scaling, min=None, max=None):
        """ will put data in tensors and scale columns starting from 12 (those before are mole fractions)
        
        Parameters
        ----------
        X : nd array

        scaling : bool

        min : nd array

        max : nd array
        
        """
        if scaling == False:
            return torch.FloatTensor(X)
        elif scaling == True:
            X[:,12:] = (X[:,12:]-min)/(max-min)
            return torch.FloatTensor(X)

    def print_data(self):
        """print the specifications of the datasets"""

        print("################################")
        print("#### Dataset specifications ####")
        print("################################")

        # print splitting
        size_train = self.x_visco_train.unique(dim=0).shape[0]
        size_valid = self.x_visco_valid.unique(dim=0).shape[0]
        size_test = self.x_visco_test.unique(dim=0).shape[0]
        size_total = size_train+size_valid+size_test
        self.size_total_visco = size_total

        print("")
        print("Number of unique compositions (viscosity): {}".format(size_total))
        print("Number of unique compositions in training (viscosity): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(size_train/size_total,
                                                                                    size_valid/size_total,
                                                                                    size_test/size_total))

        # print splitting
        size_train = self.x_entro_train.unique(dim=0).shape[0]
        size_valid = self.x_entro_valid.unique(dim=0).shape[0]
        size_test = self.x_entro_test.unique(dim=0).shape[0]
        size_total = size_train+size_valid+size_test
        self.size_total_entro = size_total

        print("")
        print("Number of unique compositions (entropy): {}".format(size_total))
        print("Number of unique compositions in training (entropy): {}".format(size_train))
        print("Dataset separations are {:.2f} in train, {:.2f} in valid, {:.2f} in test".format(size_train/size_total,
                                                                                    size_valid/size_total,
                                                                                    size_test/size_total))

        # training shapes
        print("")
        print("This is for checking the shape consistency of the dataset:\n")

        print("Visco train shape")
        print(self.x_visco_train.shape)
        print(self.T_visco_train.shape)
        print(self.y_visco_train.shape)

        print("Entropy train shape")
        print(self.x_entro_train.shape)
        print(self.y_entro_train.shape)

###
### MODEL
###
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        # Compute positional encodings in advance
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register as buffer so that it's moved to GPU with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encodings to input
        x = x + self.pe[:, :x.size(1)]
        return x

class model(torch.nn.Module):
    """i-MELT model

    """
    def __init__(self, input_size, hidden_size = 300, num_layers = 4, nb_channels_raman = 800, 
                 p_drop=0.2, activation_function = torch.nn.ReLU(), 
                 shape="rectangle", 
                 dropout_pos_enc=0.01, n_heads=4, 
                 num_encoder_layers=1, dim_feedforward_encoder=128,
                 d_model = 32, d_output = 32):
        """Initialization of i-MELT model

        Parameters
        ----------
        input_size : int
            number of input parameters

        hidden_size : int
            number of hidden units per hidden layer

        num_layers : int
            number of hidden layers

        nb_channels_raman : int
            number of Raman spectra channels, typically provided by the dataset

        p_drop : float (optinal)
            dropout probability, default = 0.2

        activation_function : torch.nn activation function (optional)
            activation function for the hidden units, default = torch.nn.ReLU()
            choose here : https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

        shape : string (optional)
            either a rectangle network (same number of neurons per layer, or triangle (regularly decreasing number of neurons per layer))
            default = rectangle
            
        dropout_pos_enc & n_heads are experimental features, do not use...
        """
        super(model, self).__init__()

        # init parameters
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_layers  = num_layers
        self.nb_channels_raman = nb_channels_raman
        self.shape = shape

        # get constants
        #self.constants = constants()

        # network related torch stuffs
        self.activation_function = activation_function
        self.p_drop = p_drop
        self.dropout = torch.nn.Dropout(p=p_drop)

        # for transformer
        self.dropout_pos_enc = dropout_pos_enc
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward_encoder = dim_feedforward_encoder
        self.d_model = d_model
        self.d_output = d_output # number of input parameters

        # general shape of the network
        if self.shape == "rectangle":

            self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, self.hidden_size)])
            self.linears.extend([torch.nn.Linear(self.hidden_size, self.hidden_size) for i in range(1, self.num_layers)])

        if self.shape == "triangle":

            self.linears = torch.nn.ModuleList([torch.nn.Linear(self.input_size, int(self.hidden_size/self.num_layers))])
            self.linears.extend([torch.nn.Linear(int(self.hidden_size/self.num_layers*i),
                                                 int(self.hidden_size/self.num_layers*(i+1))) for i in range(1,
                                                                                                         self.num_layers)])
        if self.shape == "transformer":

            # Input Embedding Layer
            self.embedding = torch.nn.Linear(1, self.d_model)

            # Positional Encoding
            self.positional_encoding = PositionalEncoding(self.d_model, self.input_size)

            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward = self.dim_feedforward_encoder,
                dropout=self.p_drop,
                batch_first=True
                )

            # Stack the encoder layers in nn.TransformerDecoder
            # It seems the option of passing a normalization instance is redundant
            # in my case, because nn.TransformerEncoderLayer per default normalizes
            # after each sub-layer
            # (https://github.com/pytorch/pytorch/issues/24930).
            self.encoder = torch.nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=self.num_encoder_layers,
                norm=None
                )
                
            self.output_encoder = torch.nn.Linear(self.d_model, self.d_output)
            
            self.linears = torch.nn.ModuleList([torch.nn.Linear(self.d_output*self.input_size, self.hidden_size)])
            self.linears.extend([torch.nn.Linear(self.hidden_size, self.hidden_size) for i in range(1, self.num_layers)])
            
        ### 
        # output layers
        ###
        self.out_thermo = torch.nn.Linear(self.hidden_size, 3) # Linear output, 22 without Cp
        
        ###
        # General Ae
        ###
        self.ae = torch.nn.Parameter(torch.tensor([-3.5]),requires_grad=True)
            
    def output_bias_init(self):
        """bias initialisation for self.out_thermo
        """
        self.out_thermo.bias = torch.nn.Parameter(data=torch.tensor([np.log(1000.), # Tg
                                                                     np.log(8.), # ScTg
                                                                     0.0, # pressure effect
                                                                     ])
                                                  )

    def forward(self, x):
        """foward pass in core neural network"""
        if self.shape != "transformer":
            for layer in self.linears: # Feedforward
                x = self.dropout(self.activation_function(layer(x)))
            return x
        else:
            # Embedding and Positional Encoding
            x = self.embedding(x.unsqueeze(2))
            x = self.positional_encoding(x)
            # only the encoder part
            x = self.encoder(x)
            # output layer
            x = self.output_encoder(x)
            x = torch.flatten(x, start_dim=1)
            # then the feedforward part
            for layer in self.linears: # Feedforward
                x = self.dropout(self.activation_function(layer(x)))
            return x
        
    def at_gfu(self,x):
        """calculate atom per gram formula unit

        assumes first columns are 
                sio2 tio2 al2o3 feo fe2o3 mno na2o k2o mgo cao p2o5 h2o
        columns: 0     1    2     3   4    5    6   7   8   9   10   11
        indices: 3     3    5     2   5    2    3   3   2   2   5    3 
        """
        out = (3.0*x[:,0] + 3.0*x[:,1] + 5.0*x[:,2]
               + 2.0*x[:,3] + 5.0*x[:,4]     
               + 2.0*x[:,5] + 3.0*x[:,6]
               + 3.0*x[:,7] + 2.0*x[:,8]
               + 2.0*x[:,9] + 5.0*x[:,10]
               + 3.0*x[:,11])
        return torch.reshape(out, (out.shape[0], 1))

    def aCpl(self,x):
        """calculate term a in equation Cpl = aCpl + bCpl*T

        Partial molar Cp are calculated by the ANN

        assumes first columns are sio2 tio2 al2o3 feo fe2o3 mno na2o k2o mgo cao p2o5 h2o
        """
        out = (81.37*x[:,0]
        + 75.21*x[:,1]
        + 130.2*x[:,2]
        + 78.94*x[:,3]
        + 199.7*x[:,4]
        + 82.73*x[:,5]
        + 100.6*x[:,6]
        + 50.13*x[:,7]
        + 85.78*x[:,8]
        + 86.05*x[:,9]
        + 80.00*x[:,10]
        + 85.00*x[:,11])
        #a_cp = torch.exp(self.out_thermo(self.forward(x))[:,2:14])
        #out = torch.sum(a_cp*x[:,0:12],axis=1)
        
        return torch.reshape(out, (out.shape[0], 1))

    def bCpl(self,x):
        """calculate term b in equation Cpl = aCpl + bCpl*T

        assumes first columns are sio2 tio2 al2o3 feo fe2o3 mno na2o k2o mgo cao p2o5 h2o

        only apply b terms on Al and K, following Richet 1985, calculated by the ANN
        """
        #b_cp = torch.exp(self.out_thermo(self.forward(x))[:,14:16])
        #out = b_cp[:,0]*x[:,2] + b_cp[:,1]*x[:,7]
        out = 0.03*x[:,2] + 0.01578*x[:,7]

        return torch.reshape(out, (out.shape[0], 1))
        

    def cpg_tg(self,x):
        """Glass heat capacity at Tg calculated from Dulong and Petit limit
        """
        return 3.0*8.314462*self.at_gfu(x)
    
    def cpl(self,x,T):
        """Liquid heat capacity at T
        """
        out = torch.exp(self.out_thermo(self.forward(x))[:,3])
        
        return torch.reshape(out, (out.shape[0], 1))
    
    def partial_cpl(self,x):
        """partial molar values for Cpl
        11 values in order: SiO2 Al2O3 Na2O K2O MgO CaO
        2 last values are temperature dependence for Al2O3 and K2O
        """
        return torch.exp(self.out_thermo(self.forward(x))[:,2:14])

    def ap_calc(self,x):
        """calculate term ap in equation dS = ap ln(T/Tg) + b(T-Tg)
        """
        out = self.aCpl(x) - self.cpg_tg(x)
        return torch.reshape(out, (out.shape[0], 1))

    def dCp(self,x,T):
        out = self.ap_calc(x)*(torch.log(T)-torch.log(self.tg(x))) + self.bCpl(x)*(T-self.tg(x))
        return torch.reshape(out, (out.shape[0], 1))

    def tg(self,x):
        """glass transition temperature Tg"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,0])
        return torch.reshape(out, (out.shape[0], 1))

    def sctg(self,x):
        """configurational entropy at Tg"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,1])
        return torch.reshape(out, (out.shape[0], 1))

    def be(self,x):
        """Be term in Adam-Gibbs equation given Ae, Tg and Scong(Tg)"""
        return (12.0-self.ae(x))*(self.tg(x)*self.sctg(x))

    def ag(self,x,T,P):
        """viscosity from the Adam-Gibbs equation, given chemistry X and temperature T

        need for speed = we decompose the calculation as much as reasonable for a minimum amount of forward pass
        """
        # analyse the chemical dataset
        X_analysis = self.forward(x)
        
        # one forward pass to get thermodynamic output
        thermo_out = self.out_thermo(X_analysis)

        # get ScTg
        sctg = torch.exp(thermo_out[:,1])
        sctg = torch.reshape(sctg, (sctg.shape[0], 1))
        
        # effect of pressure
        P_eff = thermo_out[:,2]
        P_eff = torch.reshape(P_eff, (P_eff.shape[0], 1))
        P_effect = P*P_eff
        
        # final entropy
        sc = sctg + self.dCp(x, T) + P_effect

        # get Tg
        tg = torch.exp(thermo_out[:,0])
        tg = torch.reshape(tg, (tg.shape[0], 1))

        # get Be
        be = (12.0-self.ae)*(tg*sctg)
        
        return self.ae + be/(T*sc)

###
### TRAINING FUNCTIONS
###
def training(neuralmodel, ds, 
             criterion, optimizer, 
             save_switch=True, save_name="./temp", 
             nb_folds=1, train_patience=50, min_delta=0.1, 
             verbose=True,  mode="main", device='cuda'):
    """train neuralmodel given a dataset, criterion and optimizer

    Parameters
    ----------
    neuralmodel : model
        a neuravi model
    ds : dataset
        dataset from data_loader()
    criterion : pytorch criterion
        the criterion for goodness of fit
    optimizer : pytorch optimizer
        the optimizer to use
    

    Options
    -------
    save_switch : bool
        if True, the network will be saved in save_name
    save_name : string
        the path to save the model during training
    nb_folds : int, default = 10
        the number of folds for the K-fold training
    train_patience : int, default = 50
        the number of iterations
    min_delta : float, default = 0.1
        Minimum decrease in the loss to qualify as an improvement,
        a decrease of less than or equal to `min_delta` will count as no improvement.
    verbose : bool, default = True
        Do you want details during training?
    device : string, default = "cuda"
        the device where the calculations are made during training

    Returns
    -------
    neuralmodel : model
        trained model
    record_train_loss : list
        training loss (global)
    record_valid_loss : list
        validation loss (global)
    """

    if verbose == True:
        time1 = time.time()

    #put model in train mode
    neuralmodel.train()

    # for early stopping
    epoch = 0
    best_epoch = 0
    val_ex = 0

    # for recording losses
    record_train_loss = []
    record_valid_loss = []

    # new vectors for the K-fold training (each vector contains slices of data separated)
    
    # training dataset is not on device yet and needs to be sent there
    x_visco_train = ds.x_visco_train.to(device)
    y_visco_train = ds.y_visco_train.to(device)
    T_visco_train = ds.T_visco_train.to(device)
    P_visco_train = ds.P_visco_train.to(device)

    x_entro_train = ds.x_entro_train.to(device)
    y_entro_train = ds.y_entro_train.to(device)

    while val_ex <= train_patience:

        #
        # TRAINING
        #
        loss = 0 # initialize the sum of losses of each fold

        # Forward pass on training set
        y_pred_train = neuralmodel.ag(x_visco_train, T_visco_train, P_visco_train)
        y_entro_pred_train = neuralmodel.sctg(x_entro_train)
        
        # Compute Loss
        loss_ag = criterion(y_pred_train, y_visco_train)
        loss_entro = criterion(y_entro_pred_train, y_entro_train)
        
        loss = loss_ag + loss_entro
        
        # initialise gradient
        optimizer.zero_grad() 
        loss.backward() # backward gradient determination
        optimizer.step() # optimiser call and step    
            
        # record global loss (mean of the losses of the training folds)
        record_train_loss.append(loss.item())

        #
        # MONITORING VALIDATION SUBSET
        #
        with torch.set_grad_enabled(False):

            # on validation set
            y_pred_valid = neuralmodel.ag(ds.x_visco_valid.to(device), ds.T_visco_valid.to(device), ds.P_visco_valid.to(device))
            y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid.to(device))
        
            # validation loss
            loss_ag_v = criterion(y_pred_valid, ds.y_visco_valid.to(device))
            loss_entro_v = criterion(y_entro_pred_valid,ds.y_entro_valid.to(device))
            
            loss_v = (loss_ag_v + loss_entro_v)

            record_valid_loss.append(loss_v.item())

        #
        # Print info on screen
        #
        if verbose == True:
            if (epoch % 100 == 0):
                print('\nTRAIN -- S: {:.3f}, V: {:.3f}'.format(
                loss_entro,  loss_ag
                ))
                print('VALID -- S: {:.3f}, V: {:.3f}\n'.format(
                loss_entro_v,  loss_ag_v
                ))
            if (epoch % 20 == 0):
                print('Epoch {} => loss train {:.2f}, valid {:.2f}; reg A: {:.6f}'.format(epoch, loss.item(), loss_v.item(), 0))

        #
        # calculating ES criterion
        #
        if epoch == 0:
            val_ex = 0
            best_loss_v = loss_v.item()
        elif loss_v.item() <= best_loss_v - min_delta: # if improvement is significant, this saves the model
            val_ex = 0
            best_epoch = epoch
            best_loss_v = loss_v.item()

            if save_switch == True: # save best model
                torch.save(neuralmodel.state_dict(), save_name)
        else:
            val_ex += 1

        epoch += 1

    # print outputs if verbose is True
    if verbose == True:
        time2 = time.time()
        print("Running time in seconds:", time2-time1)
        print("Scaled loss values are:")
        print('\nTRAIN -- S: {:.3f}, V: {:.3f}'.format(
                loss_entro,  loss_ag
                ))
        print('VALID -- S: {:.3f}, V: {:.3f}\n'.format(
                loss_entro_v,  loss_ag_v
        ))

    return neuralmodel, record_train_loss, record_valid_loss

def training_lbfgs(neuralmodel, ds, 
             criterion, optimizer, 
             save_switch=True, save_name="./temp", 
             train_patience=50, min_delta=0.1, 
             verbose=True,  mode="main", device='cuda'):
    """train neuralmodel given a dataset, criterion and optimizer

    Parameters
    ----------
    neuralmodel : model
        a neuravi model
    ds : dataset
        dataset from data_loader()
    criterion : pytorch criterion
        the criterion for goodness of fit
    optimizer : pytorch optimizer
        the optimizer to use
    

    Options
    -------
    save_switch : bool
        if True, the network will be saved in save_name
    save_name : string
        the path to save the model during training
    train_patience : int, default = 50
        the number of iterations
    min_delta : float, default = 0.1
        Minimum decrease in the loss to qualify as an improvement,
        a decrease of less than or equal to `min_delta` will count as no improvement.
    verbose : bool, default = True
        Do you want details during training?
    device : string, default = "cuda"
        the device where the calculations are made during training

    Returns
    -------
    neuralmodel : model
        trained model
    record_train_loss : list
        training loss (global)
    record_valid_loss : list
        validation loss (global)
    """

    if verbose == True:
        time1 = time.time()

    #put model in train mode
    neuralmodel.train()

    # for early stopping
    epoch = 0
    best_epoch = 0
    val_ex = 0

    # for recording losses
    record_train_loss = []
    record_valid_loss = []

    x_visco_train = ds.x_visco_train.to(device)
    y_visco_train = ds.y_visco_train.to(device)
    T_visco_train = ds.T_visco_train.to(device)

    x_raman_train = ds.x_raman_train.to(device)
    y_raman_train = ds.y_raman_train.to(device)

    x_density_train = ds.x_density_train.to(device)
    y_density_train = ds.y_density_train.to(device)

    x_elastic_train = ds.x_elastic_train.to(device)
    y_elastic_train = ds.y_elastic_train.to(device)

    x_entro_train = ds.x_entro_train.to(device)
    y_entro_train = ds.y_entro_train.to(device)

    x_ri_train = ds.x_ri_train.to(device)
    y_ri_train = ds.y_ri_train.to(device)
    lbd_ri_train = ds.lbd_ri_train.to(device)

    x_cpl_train = ds.x_cpl_train.to(device)
    y_cpl_train = ds.y_cpl_train.to(device)
    T_cpl_train = ds.T_cpl_train.to(device)

    x_cte_train = ds.x_cte_train.to(device)
    y_cte_train = ds.y_cte_train.to(device)

    x_abbe_train = ds.x_abbe_train.to(device)
    y_abbe_train = ds.y_abbe_train.to(device)

    x_liquidus_train = ds.x_liquidus_train.to(device)
    y_liquidus_train = ds.y_liquidus_train.to(device)

    while val_ex <= train_patience:

        #
        # TRAINING
        #
        def closure(): # closure condition for LBFGS

            # Forward pass on training set
            y_ag_pred_train = neuralmodel.ag(x_visco_train,T_visco_train)
            y_myega_pred_train = neuralmodel.myega(x_visco_train,T_visco_train)
            y_am_pred_train = neuralmodel.am(x_visco_train,T_visco_train)
            y_cg_pred_train = neuralmodel.cg(x_visco_train,T_visco_train)
            y_tvf_pred_train = neuralmodel.tvf(x_visco_train,T_visco_train)
            y_raman_pred_train = neuralmodel.raman_pred(x_raman_train)
            y_density_pred_train = neuralmodel.density_glass(x_density_train)
            y_elastic_pred_train = neuralmodel.elastic_modulus(x_elastic_train)
            y_entro_pred_train = neuralmodel.sctg(x_entro_train)
            y_ri_pred_train = neuralmodel.sellmeier(x_ri_train,lbd_ri_train)
            y_cpl_pred_train = neuralmodel.cpl(x_cpl_train, T_cpl_train)
            y_cte_pred_train = neuralmodel.cte(x_cte_train)
            y_abbe_pred_train = neuralmodel.abbe(x_abbe_train)
            y_liquidus_pred_train = neuralmodel.liquidus(x_liquidus_train)
            
            # Get precisions
            precision_visco = torch.exp(-neuralmodel.log_vars[0])
            precision_raman = torch.exp(-neuralmodel.log_vars[1])
            precision_density = torch.exp(-neuralmodel.log_vars[2])
            precision_entro = torch.exp(-neuralmodel.log_vars[3])
            precision_ri = torch.exp(-neuralmodel.log_vars[4])
            precision_cpl = torch.exp(-neuralmodel.log_vars[5])
            precision_elastic = torch.exp(-neuralmodel.log_vars[6])
            precision_cte = torch.exp(-neuralmodel.log_vars[7])
            precision_abbe = torch.exp(-neuralmodel.log_vars[8])
            precision_liquidus = torch.exp(-neuralmodel.log_vars[9])

            # Compute Loss
            loss_ag = precision_visco * criterion(y_ag_pred_train, y_visco_train)
            loss_myega = precision_visco * criterion(y_myega_pred_train, y_visco_train)
            loss_am = precision_visco * criterion(y_am_pred_train, y_visco_train)
            loss_cg = precision_visco * criterion(y_cg_pred_train, y_visco_train)
            loss_tvf = precision_visco * criterion(y_tvf_pred_train, y_visco_train)
            loss_raman = precision_raman * criterion(y_raman_pred_train,y_raman_train)
            loss_density = precision_density * criterion(y_density_pred_train,y_density_train)
            loss_entro = precision_entro * criterion(y_entro_pred_train,y_entro_train)
            loss_ri = precision_ri * criterion(y_ri_pred_train,y_ri_train)
            loss_cpl = precision_cpl * criterion(y_cpl_pred_train,y_cpl_train) 
            loss_elastic = precision_elastic * criterion(y_elastic_pred_train,y_elastic_train)
            loss_cte = precision_cte * criterion(y_cte_pred_train,y_cte_train)
            loss_abbe = precision_abbe * criterion(y_abbe_pred_train,y_abbe_train)
            loss_liquidus = precision_liquidus * criterion(y_liquidus_pred_train,y_liquidus_train)

            loss_fold = (loss_ag + loss_myega + loss_am + loss_cg + loss_tvf
                        + loss_raman + loss_density + loss_entro + loss_ri 
                        + loss_cpl +loss_elastic + loss_cte + loss_abbe + loss_liquidus
                        + neuralmodel.log_vars[0] + neuralmodel.log_vars[1] + neuralmodel.log_vars[2] 
                        + neuralmodel.log_vars[3] + neuralmodel.log_vars[4] + neuralmodel.log_vars[5] 
                        + neuralmodel.log_vars[6] + neuralmodel.log_vars[7] + neuralmodel.log_vars[8]
                        + neuralmodel.log_vars[9])

            # initialise gradient
            optimizer.zero_grad() 
            loss_fold.backward() # backward gradient determination
            
            return loss_fold
        
        # Update weights
        optimizer.step(closure) # update weights

        # update the running loss
        loss = closure().item()

        # record global loss (mean of the losses of the training folds)
        record_train_loss.append(loss)

        #
        # MONITORING VALIDATION SUBSET
        #
        with torch.set_grad_enabled(False):

            # # Precisions
            precision_visco = torch.exp(-neuralmodel.log_vars[0])
            precision_raman = torch.exp(-neuralmodel.log_vars[1])
            precision_density = torch.exp(-neuralmodel.log_vars[2])
            precision_entro = torch.exp(-neuralmodel.log_vars[3])
            precision_ri = torch.exp(-neuralmodel.log_vars[4])
            precision_cpl = torch.exp(-neuralmodel.log_vars[5])
            precision_elastic = torch.exp(-neuralmodel.log_vars[6])
            precision_cte = torch.exp(-neuralmodel.log_vars[7])
            precision_abbe = torch.exp(-neuralmodel.log_vars[8])
            precision_liquidus = torch.exp(-neuralmodel.log_vars[9])

            # on validation set
            y_ag_pred_valid = neuralmodel.ag(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))
            y_myega_pred_valid = neuralmodel.myega(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))
            y_am_pred_valid = neuralmodel.am(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))
            y_cg_pred_valid = neuralmodel.cg(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))
            y_tvf_pred_valid = neuralmodel.tvf(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))
            y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid.to(device))
            y_density_pred_valid = neuralmodel.density_glass(ds.x_density_valid.to(device))
            y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid.to(device))
            y_ri_pred_valid = neuralmodel.sellmeier(ds.x_ri_valid.to(device), ds.lbd_ri_valid.to(device))
            y_clp_pred_valid = neuralmodel.cpl(ds.x_cpl_valid.to(device), ds.T_cpl_valid.to(device))
            y_elastic_pred_valid = neuralmodel.elastic_modulus(ds.x_elastic_valid.to(device))
            y_cte_pred_valid = neuralmodel.cte(ds.x_cte_valid.to(device))
            y_abbe_pred_valid = neuralmodel.abbe(ds.x_abbe_valid.to(device))
            y_liquidus_pred_valid = neuralmodel.liquidus(ds.x_liquidus_valid.to(device))

            # validation loss
            loss_ag_v = precision_visco * criterion(y_ag_pred_valid, ds.y_visco_valid.to(device))
            loss_myega_v = precision_visco * criterion(y_myega_pred_valid, ds.y_visco_valid.to(device))
            loss_am_v = precision_visco * criterion(y_am_pred_valid, ds.y_visco_valid.to(device))
            loss_cg_v = precision_visco * criterion(y_cg_pred_valid, ds.y_visco_valid.to(device))
            loss_tvf_v = precision_visco * criterion(y_tvf_pred_valid, ds.y_visco_valid.to(device))
            loss_raman_v = precision_raman * criterion(y_raman_pred_valid,ds.y_raman_valid.to(device))
            loss_density_v = precision_density * criterion(y_density_pred_valid,ds.y_density_valid.to(device))
            loss_entro_v = precision_entro * criterion(y_entro_pred_valid,ds.y_entro_valid.to(device))
            loss_ri_v = precision_ri * criterion(y_ri_pred_valid,ds.y_ri_valid.to(device))
            loss_cpl_v = precision_cpl * criterion(y_clp_pred_valid,ds.y_cpl_valid.to(device))
            loss_elastic_v = precision_elastic * criterion(y_elastic_pred_valid,ds.y_elastic_valid.to(device))
            loss_cte_v = precision_cte * criterion(y_cte_pred_valid,ds.y_cte_valid.to(device))
            loss_abbe_v = precision_abbe * criterion(y_abbe_pred_valid,ds.y_abbe_valid.to(device))
            loss_liquidus_v = precision_liquidus * criterion(y_liquidus_pred_valid,ds.y_liquidus_valid.to(device))

            loss_v = (loss_ag_v + loss_myega_v + loss_am_v + loss_cg_v + loss_tvf_v
                     + loss_raman_v + loss_density_v + loss_entro_v + loss_ri_v 
                     + loss_cpl_v + loss_elastic_v + loss_cte_v + loss_abbe_v + loss_liquidus_v
                     + neuralmodel.log_vars[0] + neuralmodel.log_vars[1] + neuralmodel.log_vars[2] 
                     + neuralmodel.log_vars[3] + neuralmodel.log_vars[4] + neuralmodel.log_vars[5] 
                     + neuralmodel.log_vars[6] + neuralmodel.log_vars[7] + neuralmodel.log_vars[8]
                     + neuralmodel.log_vars[9])

            record_valid_loss.append(loss_v.item())

        #
        # Print info on screen
        #
        if verbose == True:
            if (epoch % 100 == 0):
                #print('\nTRAIN -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}'.format(
                #loss_raman, loss_density, loss_entro,  loss_ri, loss_ag, loss_cpl, loss_elastic, loss_cte, loss_abbe, loss_liquidus
                #))
                print('VALID -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}\n'.format(
                loss_raman_v, loss_density_v, loss_entro_v,  loss_ri_v, loss_ag_v, loss_cpl_v, loss_elastic_v, loss_cte_v, loss_abbe_v, loss_liquidus_v
                ))
            if (epoch % 20 == 0):
                print('Epoch {} => loss train {:.2f}, valid {:.2f}'.format(epoch, loss, loss_v))

        #
        # calculating ES criterion
        #
        if epoch == 0:
            val_ex = 0
            best_loss_v = loss_v.item()
        elif loss_v.item() <= best_loss_v - min_delta: # if improvement is significant, this saves the model
            val_ex = 0
            best_epoch = epoch
            best_loss_v = loss_v.item()

            if save_switch == True: # save best model
                torch.save(neuralmodel.state_dict(), save_name)
        else:
            val_ex += 1

        epoch += 1

    # print outputs if verbose is True
    # if verbose == True:
    #     time2 = time.time()
    #     print("Running time in seconds:", time2-time1)
    #     print("Scaled loss values are:")
    #     print('\nTRAIN -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}'.format(
    #             loss_raman, loss_density, loss_entro,  loss_ri, loss_ag, loss_cpl, loss_elastic, loss_cte, loss_abbe, loss_liquidus
    #             ))
    #     print('VALID -- Raman: {:.3f}, d: {:.3f}, S: {:.3f}, RI: {:.3f}, V: {:.3f}, Cp: {:.3f}, Em: {:.3f}, CTE: {:.3f}, Ab: {:.3f}, Tl: {:.3f}\n'.format(
    #             loss_raman_v, loss_density_v, loss_entro_v,  loss_ri_v, loss_ag_v, loss_cpl_v, loss_elastic_v, loss_cte_v, loss_abbe_v, loss_liquidus_v
    #             ))

    return neuralmodel, record_train_loss, record_valid_loss

def record_loss_build(path, list_models, ds, shape='rectangle'):
    """build a Pandas dataframe with the losses for a list of models at path

    """
    # scaling coefficients for global loss function
    # viscosity is always one
    # check lines 578-582 in imelt.py
    entro_scale = 1.
    raman_scale = 20.
    density_scale = 1000.
    ri_scale = 10000.

    nb_exp = len(list_models)

    record_loss = pd.DataFrame()

    record_loss["name"] = list_models

    record_loss["nb_layers"] = np.zeros(nb_exp)
    record_loss["nb_neurons"] = np.zeros(nb_exp)
    record_loss["p_drop"] = np.zeros(nb_exp)

    record_loss["loss_ag_train"] = np.zeros(nb_exp)
    record_loss["loss_ag_valid"] = np.zeros(nb_exp)

    record_loss["loss_am_train"] = np.zeros(nb_exp)
    record_loss["loss_am_valid"] = np.zeros(nb_exp)

    record_loss["loss_am_train"] = np.zeros(nb_exp)
    record_loss["loss_am_valid"] = np.zeros(nb_exp)

    record_loss["loss_Sconf_train"] = np.zeros(nb_exp)
    record_loss["loss_Sconf_valid"] = np.zeros(nb_exp)

    record_loss["loss_d_train"] = np.zeros(nb_exp)
    record_loss["loss_d_valid"] = np.zeros(nb_exp)

    record_loss["loss_raman_train"] = np.zeros(nb_exp)
    record_loss["loss_raman_valid"] = np.zeros(nb_exp)

    record_loss["loss_train"] = np.zeros(nb_exp)
    record_loss["loss_valid"] = np.zeros(nb_exp)

    # Loss criterion
    criterion = torch.nn.MSELoss()

    # Load dataset
    for idx,name in enumerate(list_models):

        # Extract arch
        nb_layers = int(name[name.find("l")+1:name.find("_")])
        nb_neurons = int(name[name.find("n")+1:name.find("p")-1])
        p_drop = float(name[name.find("p")+1:name.find("s")-1])

        # Record arch
        record_loss.loc[idx,"nb_layers"] = nb_layers
        record_loss.loc[idx,"nb_neurons"] = nb_neurons
        record_loss.loc[idx,"p_drop"] = p_drop

        # Declare model
        neuralmodel = imelt.model(6,nb_neurons,nb_layers,ds.nb_channels_raman,p_drop=p_drop, shape=shape)
        neuralmodel.load_state_dict(torch.load(path+'/'+name, map_location='cpu'))
        neuralmodel.eval()

        # PREDICTIONS

        with torch.set_grad_enabled(False):
            # train
            y_ag_pred_train = neuralmodel.ag(ds.x_visco_train,ds.T_visco_train)
            y_myega_pred_train = neuralmodel.myega(ds.x_visco_train,ds.T_visco_train)
            y_am_pred_train = neuralmodel.am(ds.x_visco_train,ds.T_visco_train)
            y_cg_pred_train = neuralmodel.cg(ds.x_visco_train,ds.T_visco_train)
            y_tvf_pred_train = neuralmodel.tvf(ds.x_visco_train,ds.T_visco_train)
            y_raman_pred_train = neuralmodel.raman_pred(ds.x_raman_train)
            y_density_pred_train = neuralmodel.density_glass(ds.x_density_train)
            y_entro_pred_train = neuralmodel.sctg(ds.x_entro_train)
            y_ri_pred_train = neuralmodel.sellmeier(ds.x_ri_train, ds.lbd_ri_train)

            # valid
            y_ag_pred_valid = neuralmodel.ag(ds.x_visco_valid,ds.T_visco_valid)
            y_myega_pred_valid = neuralmodel.myega(ds.x_visco_valid,ds.T_visco_valid)
            y_am_pred_valid = neuralmodel.am(ds.x_visco_valid,ds.T_visco_valid)
            y_cg_pred_valid = neuralmodel.cg(ds.x_visco_valid,ds.T_visco_valid)
            y_tvf_pred_valid = neuralmodel.tvf(ds.x_visco_valid,ds.T_visco_valid)
            y_raman_pred_valid = neuralmodel.raman_pred(ds.x_raman_valid)
            y_density_pred_valid = neuralmodel.density_glass(ds.x_density_valid)
            y_entro_pred_valid = neuralmodel.sctg(ds.x_entro_valid)
            y_ri_pred_valid = neuralmodel.sellmeier(ds.x_ri_valid, ds.lbd_ri_valid)

            # Compute Loss

            # train
            record_loss.loc[idx,"loss_ag_train"] = np.sqrt(criterion(y_ag_pred_train, ds.y_visco_train).item())
            record_loss.loc[idx,"loss_myega_train"]  = np.sqrt(criterion(y_myega_pred_train, ds.y_visco_train).item())
            record_loss.loc[idx,"loss_am_train"]  = np.sqrt(criterion(y_am_pred_train, ds.y_visco_train).item())
            record_loss.loc[idx,"loss_cg_train"]  = np.sqrt(criterion(y_cg_pred_train, ds.y_visco_train).item())
            record_loss.loc[idx,"loss_tvf_train"]  = np.sqrt(criterion(y_tvf_pred_train, ds.y_visco_train).item())
            record_loss.loc[idx,"loss_raman_train"]  = np.sqrt(criterion(y_raman_pred_train,ds.y_raman_train).item())
            record_loss.loc[idx,"loss_d_train"]  = np.sqrt(criterion(y_density_pred_train,ds.y_density_train).item())
            record_loss.loc[idx,"loss_Sconf_train"]  = np.sqrt(criterion(y_entro_pred_train,ds.y_entro_train).item())
            record_loss.loc[idx,"loss_ri_train"]  = np.sqrt(criterion(y_ri_pred_train,ds.y_ri_train).item())

            # validation
            record_loss.loc[idx,"loss_ag_valid"] = np.sqrt(criterion(y_ag_pred_valid, ds.y_visco_valid).item())
            record_loss.loc[idx,"loss_myega_valid"] = np.sqrt(criterion(y_myega_pred_valid, ds.y_visco_valid).item())
            record_loss.loc[idx,"loss_am_valid"] = np.sqrt(criterion(y_am_pred_valid, ds.y_visco_valid).item())
            record_loss.loc[idx,"loss_cg_valid"]  = np.sqrt(criterion(y_cg_pred_valid, ds.y_visco_valid).item())
            record_loss.loc[idx,"loss_tvf_valid"]  = np.sqrt(criterion(y_tvf_pred_valid, ds.y_visco_valid).item())
            record_loss.loc[idx,"loss_raman_valid"] = np.sqrt(criterion(y_raman_pred_valid,ds.y_raman_valid).item())
            record_loss.loc[idx,"loss_d_valid"] = np.sqrt(criterion(y_density_pred_valid,ds.y_density_valid).item())
            record_loss.loc[idx,"loss_Sconf_valid"] = np.sqrt(criterion(y_entro_pred_valid,ds.y_entro_valid).item())
            record_loss.loc[idx,"loss_ri_valid"]  = np.sqrt(criterion(y_ri_pred_valid,ds.y_ri_valid).item())

            record_loss.loc[idx,"loss_train"] = (record_loss.loc[idx,"loss_ag_train"] +
                                                 record_loss.loc[idx,"loss_myega_train"] +
                                                 record_loss.loc[idx,"loss_am_train"] +
                                                 record_loss.loc[idx,"loss_cg_train"] +
                                                 record_loss.loc[idx,"loss_tvf_train"] +
                                                 raman_scale*record_loss.loc[idx,"loss_raman_train"] +
                                                 density_scale*record_loss.loc[idx,"loss_d_train"] +
                                                 entro_scale*record_loss.loc[idx,"loss_Sconf_train"] +
                                                 ri_scale*record_loss.loc[idx,"loss_ri_train"])

            record_loss.loc[idx,"loss_valid"] = (record_loss.loc[idx,"loss_ag_valid"] +
                                                 record_loss.loc[idx,"loss_myega_valid"] +
                                                 record_loss.loc[idx,"loss_am_valid"] +
                                                 record_loss.loc[idx,"loss_cg_valid"] +
                                                 record_loss.loc[idx,"loss_tvf_valid"] +
                                                 raman_scale*record_loss.loc[idx,"loss_raman_valid"] +
                                                 density_scale*record_loss.loc[idx,"loss_d_valid"] +
                                                 entro_scale*record_loss.loc[idx,"loss_Sconf_valid"] +
                                                 ri_scale*record_loss.loc[idx,"loss_ri_valid"])

    return record_loss

###
### BAGGING
###

class bagging_models:
    """custom class for bagging models and making predictions

    Parameters
    ----------
    path : str
        path of models

    name_models : list of str
        names of models

    device : str
        cpu or gpu

    activation_function : torch.nn.Module
        activation function to be used, default is ReLU

    Methods
    -------
    predict : function
        make predictions

    """
    def __init__(self, path, name_models, ds, device, activation_function=torch.nn.ReLU()):

        self.device = device
        self.n_models = len(name_models)
        self.models = [None for _ in range(self.n_models)]

        for i in range(self.n_models):
            name = name_models[i]

            # Extract arch
            nb_layers = int(name[name.find("l")+1:name.find("_n")])
            nb_neurons = int(name[name.find("n")+1:name.rfind("_p")])
            p_drop = float(name[name.find("p")+1:name.rfind("_m")])

            self.models[i] = model(ds.x_visco_train.shape[1],nb_neurons,nb_layers,ds.nb_channels_raman,
                                   p_drop=p_drop, activation_function=activation_function)
            
            state_dict = torch.load(path+name,map_location='cpu')
            if len(state_dict) == 2:
                self.models[i].load_state_dict(state_dict[0])
            else:
                self.models[i].load_state_dict(state_dict)
            self.models[i].eval()

    def predict(self, method, X, T=[1000.0], lbd= [500.0], sampling=False, n_sample = 10):
        """returns predictions from the n models

        Parameters
        ----------
        method : str
            the property to predict. See imelt code for possibilities. Basically it is a string handle that will be converted to an imelt function.
            For instance, for tg, enter 'tg'.
        X : pandas dataframe
            chemical composition for prediction
        T : list of floats
            temperatures for predictions, default = [1000.0]
        lbd : list of floats
            lambdas for Sellmeier equation, default = [500.0]
        sampling : Bool
            if True, dropout is activated and n_sample random samples will be generated per network. 
            This allows performing MC Dropout on the ensemble of models.
        """

        # sending data to device
        X = torch.Tensor(X).to(self.device)
        T = torch.Tensor(T).to(self.device)
        lbd = torch.Tensor(lbd).to(self.device)

        #
        # sending models to device also.
        # and we activate dropout if necessary for error sampling
        #
        for i in range(self.n_models):
            self.models[i].to(self.device)
            if sampling == True:
                self.models[i].train()

        with torch.no_grad():
            if method == "raman_pred":
                #
                # For Raman spectra generation
                #
                if sampling == True:
                    out = np.zeros((len(X),850,self.n_models,n_sample)) # problem is defined with a X raman shift of 850 values
                    for i in range(self.n_models):
                        for j in range(n_sample):
                            out[:,:,i,j] = getattr(self.models[i],method)(X).cpu().detach().numpy()        
                    
                    # reshaping for 3D outputs
                    out = out.reshape((out.shape[0], out.shape[1], out.shape[2]*out.shape[3]))
                else:
                    out = np.zeros((len(X),850,self.n_models)) # problem is defined with a X raman shift of 850 values
                    for i in range(self.n_models):
                        out[:,:,i] = getattr(self.models[i],method)(X).cpu().detach().numpy()
            else:
                #
                # Other parameters (latent or real)
                #

                # sampling activated
                if sampling == True:
                    out = np.zeros((len(X),self.n_models, n_sample))
                    if method in frozenset(('ag', 'myega', 'am', 'cg', 'tvf','density_melt','cpl','dCp')):
                        for i in range(self.n_models):
                            for j in range(n_sample):
                                out[:,i,j] = getattr(self.models[i],method)(X,T).cpu().detach().numpy().reshape(-1)
                    elif method == "sellmeier":
                        for i in range(self.n_models):
                            for j in range(n_sample):
                                out[:,i,j] = getattr(self.models[i],method)(X,lbd).cpu().detach().numpy().reshape(-1)
                    elif method == 'vm_glass':
                        # we must create a new out tensor because we have a multi-output 
                        out = np.zeros((len(X),6, self.n_models, n_sample))
                        for i in range(self.n_models):
                            for j in range(n_sample):
                                out[:,:,i,j] = getattr(self.models[i],method)(X).cpu().detach().numpy()
                    elif method == "partial_cpl":
                        # we must create a new out tensor because we have a multi-output 
                        out = np.zeros((len(X),8, self.n_models, n_sample))
                        for i in range(self.n_models):
                            for j in range(n_sample):
                                out[:,:,i,j] = getattr(self.models[i],method)(X).cpu().detach().numpy()
                    else:
                        for i in range(self.n_models):
                            for j in range(n_sample):
                                out[:,i,j] = getattr(self.models[i],method)(X).cpu().detach().numpy().reshape(-1)
                    
                    # reshaping for 2D outputs
                    if method in frozenset(('vm_glass', 'partial_cpl')):
                        out = out.reshape((out.shape[0], out.shape[1], out.shape[2]*out.shape[3]))
                    else:
                        out = out.reshape((out.shape[0], out.shape[1]*out.shape[2]))
                
                # no sampling
                else:
                    out = np.zeros((len(X),self.n_models))
                    if method in frozenset(('ag', 'myega', 'am', 'cg', 'tvf','density_melt','cpl', 'dCp')):
                        for i in range(self.n_models):
                            out[:,i] = getattr(self.models[i],method)(X,T).cpu().detach().numpy().reshape(-1)
                    elif method == "sellmeier":
                        for i in range(self.n_models):
                            out[:,i] = getattr(self.models[i],method)(X,lbd).cpu().detach().numpy().reshape(-1)
                    elif method == 'vm_glass':
                        # we must create a new out tensor because we have a multi-output 
                        out = np.zeros((len(X),6, self.n_models))
                        for i in range(self.n_models):
                            out[:,:,i] = getattr(self.models[i],method)(X).cpu().detach().numpy()
                    elif method == "partial_cpl":
                        # we must create a new out tensor because we have a multi-output 
                        out = np.zeros((len(X),8, self.n_models))
                        for i in range(self.n_models):
                            out[:,:,i] = getattr(self.models[i],method)(X).cpu().detach().numpy()
                    else:
                        for i in range(self.n_models):
                            out[:,i] = getattr(self.models[i],method)(X).cpu().detach().numpy().reshape(-1)
            
        # Before leaving this function, we make sure we freeze again the dropout
        for i in range(self.n_models):
            self.models[i].eval() # we make sure we freeze dropout if user does not activate sampling
        
        # returning our sample
        if sampling == False:
            return np.median(out, axis= out.ndim-1)
        else:
            return out

def load_pretrained_bagged(path_viscosity = "./data/NKCMAS_viscosity.hdf5", 
                            path_raman = "./data/NKCMAS_Raman.hdf5", 
                            path_density = "./data/NKCMAS_density.hdf5", 
                            path_ri = "./data/NKCMAS_optical.hdf5", 
                            path_cp = "./data/NKCMAS_cp.hdf5", 
                            path_elastic = "./data/NKCMAS_em.hdf5",
                            path_cte = "./data/NKCMAS_cte.hdf5",
                            path_abbe = "./data/NKCMAS_abbe.hdf5",
                            path_liquidus = "./data/NKCMAS_tl.hdf5", 
                            path_models = "./model/best/", 
                            device=torch.device('cpu'),
                            activation_function=torch.nn.GELU()):
    """loader for the pretrained bagged i-melt models

    Parameters
    ----------
    path_viscosity : string
            path for the viscosity HDF5 dataset (optional)
    path_raman : string
        path for the Raman spectra HDF5 dataset (optional)
    path_density : string
        path for the density HDF5 dataset (optional)
    path_ri : String
        path for the refractive index HDF5 dataset (optional)
    path_cp : String
        path for the liquid heat capacity HDF5 dataset (optional)
    path_elastic : String
        path for the elastic moduli HDF5 dataset (optional)
    path_cte : String
        path for the thermal expansion HDF5 dataset (optional)
    path_abbe : String
        path for the Abbe number HDF5 dataset (optional)
    path_liquidus : String
        path for the liquidus temperature HDF5 dataset (optional)
    path_models : str
        Path for the models (optional)
    device : torch.device()
        CPU or GPU device, default = 'cpu' (optional)

    Returns
    -------
    bagging_models : object
        A bagging_models object that can be used for predictions
    """
    import pandas as pd
    ds = data_loader(path_viscosity,path_raman,path_density,path_ri,path_cp,path_elastic,path_cte,path_abbe,path_liquidus)
    name_list = pd.read_csv(path_models+"best_list.csv").loc[:,"name"]
    return bagging_models(path_models, name_list, ds, device, activation_function=activation_function)

def R_Raman(x,y, lb = 670, hb = 870):
    """calculates the R_Raman parameter of a Raman signal y sampled at x.

    y can be an NxM array with N samples and M Raman shifts.
    """
    A_LW =  np.trapz(y[:,x<lb],x[x<lb],axis=1)
    A_HW =  np.trapz(y[:,x>hb],x[x>hb],axis=1)
    return A_LW/A_HW

class constants():
    def __init__(self):
        self.V_g_sio2 = (27+2*16)/2.2007
        self.V_g_al2o3 = (26*2+3*16)/3.009
        self.V_g_na2o = (22*2+16)/2.686
        self.V_g_k2o = (44*2+16)/2.707
        self.V_g_mgo = (24.3+16)/3.115
        self.V_g_cao = (40.08+16)/3.140
        
        self.V_m_sio2 = 27.297 # Courtial and Dingwell 1999, 1873 K
        self.V_m_al2o3 = 36.666 # Courtial and Dingwell 1999
        #self.V_m_SiCa = -7.105 # Courtial and Dingwell 1999
        self.V_m_na2o = 29.65 # Tref=1773 K (Lange, 1997; CMP)
        self.V_m_k2o = 47.28 # Tref=1773 K (Lange, 1997; CMP)
        self.V_m_mgo = 12.662 # Courtial and Dingwell 1999
        self.V_m_cao = 20.664 # Courtial and Dingwell 1999
        
        #dV/dT values
        self.dVdT_SiO2 = 1.157e-3 # Courtial and Dingwell 1999 
        self.dVdT_Al2O3 = -1.184e-3 # Courtial and Dingwell 1999
        #self.dVdT_SiCa = -2.138 # Courtial and Dingwell 1999
        self.dVdT_Na2O = 0.00768 # Table 4 (Lange, 1997)
        self.dVdT_K2O = 0.01208 # Table 4 (Lange, 1997)
        self.dVdT_MgO = 1.041e-3 # Courtial and Dingwell 1999
        self.dVdT_CaO = 3.786e-3 # Courtial and Dingwell 1999
        
        # melt T reference
        self.Tref_SiO2 = 1873.0 # Courtial and Dingwell 1999
        self.Tref_Al2O3 = 1873.0 # Courtial and Dingwell 1999
        self.Tref_Na2O = 1773.0 # Tref=1773 K (Lange, 1997; CMP)
        self.Tref_K2O = 1773.0 # Tref=1773 K (Lange, 1997; CMP)
        self.Tref_MgO = 1873.0 # Courtial and Dingwell 1999
        self.Tref_CaO = 1873.0 # Courtial and Dingwell 1999
        
        # correction constants between glass at Tambient and melt at Tref
        self.c_sio2 = self.V_m_sio2 - self.V_g_sio2
        self.c_al2o3 = self.V_m_al2o3 - self.V_g_al2o3
        self.c_na2o = self.V_m_na2o - self.V_g_na2o
        self.c_k2o = self.V_m_k2o - self.V_g_k2o
        self.c_mgo = self.V_m_mgo - self.V_g_mgo
        self.c_cao = self.V_m_cao - self.V_g_cao

class density_constants():

    def __init__(self):
        #Partial Molar Volumes
        self.MV_SiO2 = 27.297 # Courtial and Dingwell 1999
        self.MV_TiO2 = 28.32 # TiO2 at Tref=1773 K (Lange and Carmichael, 1987)
        self.MV_Al2O3 = 36.666 # Courtial and Dingwell 1999
        self.MV_Fe2O3 = 41.50 # Fe2O3 at Tref=1723 K (Liu and Lange, 2006)
        self.MV_FeO = 12.68 # FeO at Tref=1723 K (Guo et al., 2014)
        self.MV_MgO = 12.662 # Courtial and Dingwell 1999
        self.MV_CaO = 20.664 # Courtial and Dingwell 1999
        self.MV_SiCa = -7.105 # Courtial and Dingwell 1999
        self.MV_Na2O = 29.65 # Tref=1773 K (Lange, 1997; CMP)
        self.MV_K2O = 47.28 # Tref=1773 K (Lange, 1997; CMP)
        self.MV_H2O = 22.9 # H2O at Tref=1273 K (Ochs and Lange, 1999)

        #Partial Molar Volume uncertainties
        #value = 0 if not reported
        self.unc_MV_SiO2 = 0.152 # Courtial and Dingwell 1999
        self.unc_MV_TiO2 = 0.0
        self.unc_MV_Al2O3 = 0.196 # Courtial and Dingwell 1999
        self.unc_MV_Fe2O3 = 0.0
        self.unc_MV_FeO = 0.0
        self.unc_MV_MgO = 0.181 # Courtial and Dingwell 1999
        self.unc_MV_CaO = 0.123 # Courtial and Dingwell 1999
        self.unc_MV_SiCa = 0.509 # Courtial and Dingwell 1999
        self.unc_MV_Na2O = 0.07
        self.unc_MV_K2O = 0.10
        self.unc_MV_H2O = 0.60

        #dV/dT values
        #MgO, CaO, Na2O, K2O Table 4 (Lange, 1997)
        #SiO2, TiO2, Al2O3 Table 9 (Lange and Carmichael, 1987)
        #H2O from Ochs & Lange (1999)
        #Fe2O3 from Liu & Lange (2006)
        #FeO from Guo et al (2014)
        self.dVdT_SiO2 = 1.157e-3 # Courtial and Dingwell 1999 
        self.dVdT_TiO2 = 0.00724
        self.dVdT_Al2O3 = -1.184e-3 # Courtial and Dingwell 1999
        self.dVdT_Fe2O3 = 0.0
        self.dVdT_FeO = 0.00369
        self.dVdT_MgO = 1.041e-3 # Courtial and Dingwell 1999
        self.dVdT_CaO = 3.786e-3 # Courtial and Dingwell 1999
        self.dVdT_SiCa = -2.138 # Courtial and Dingwell 1999
        self.dVdT_Na2O = 0.00768
        self.dVdT_K2O = 0.01208
        self.dVdT_H2O = 0.0095

        #dV/dT uncertainties
        #value = 0 if not reported
        self.unc_dVdT_SiO2 = 0.0007e-3 # Courtial and Dingwell 1999
        self.unc_dVdT_TiO2 = 0.0
        self.unc_dVdT_Al2O3 = 0.0009e-3 # Courtial and Dingwell 1999
        self.unc_dVdT_Fe2O3 = 0.0
        self.unc_dVdT_FeO = 0.0
        self.unc_dVdT_MgO = 0.0008 # Courtial and Dingwell 1999
        self.unc_dVdT_CaO = 0.0005e-3 # Courtial and Dingwell 1999
        self.unc_dVdT_SiCa = 0.002e-3 # Courtial and Dingwell 1999
        self.unc_dVdT_Na2O = 0.0
        self.unc_dVdT_K2O = 0.0
        self.unc_dVdT_H2O = 0.0008

        #dV/dP values
        #Anhydrous component data from Kess and Carmichael (1991)
        #H2O data from Ochs & Lange (1999)
        self.dVdP_SiO2 = -0.000189
        self.dVdP_TiO2 = -0.000231
        self.dVdP_Al2O3 = -0.000226
        self.dVdP_Fe2O3 = -0.000253
        self.dVdP_FeO = -0.000045
        self.dVdP_MgO = 0.000027
        self.dVdP_CaO = 0.000034
        self.dVdP_Na2O = -0.00024
        self.dVdP_K2O = -0.000675
        self.dVdP_H2O = -0.00032

        #dV/dP uncertainties
        self.unc_dVdP_SiO2 = 0.000002
        self.unc_dVdP_TiO2 = 0.000006
        self.unc_dVdP_Al2O3 = 0.000009
        self.unc_dVdP_Fe2O3 = 0.000009
        self.unc_dVdP_FeO = 0.000003
        self.unc_dVdP_MgO = 0.000007
        self.unc_dVdP_CaO = 0.000005
        self.unc_dVdP_Na2O = 0.000005
        self.unc_dVdP_K2O = 0.000014
        self.unc_dVdP_H2O = 0.000060

        #Tref values
        self.Tref_SiO2 = 1873.0 # Courtial and Dingwell 1999
        self.Tref_TiO2 = 1773.0
        self.Tref_Al2O3 = 1873.0 # Courtial and Dingwell 1999
        self.Tref_Fe2O3 = 1723.0
        self.Tref_FeO = 1723.0
        self.Tref_MgO = 1873.0 # Courtial and Dingwell 1999
        self.Tref_CaO = 1873.0 # Courtial and Dingwell 1999
        self.Tref_Na2O = 1773.0
        self.Tref_K2O = 1773.0
        self.Tref_H2O = 1273.0


###
### Functions for ternary plots (not really needed with mpltern)
###
def plot_loss(ax, loss, legends, scale="linear"):
    for count,i in enumerate(loss):
        ax.plot(i,label=legends[count])

    ax.legend()
    ax.set_yscale(scale)
    ax.set_xlabel("Epoch")

###
### Molecular weights (duplicate of utils function to avoid path issues)
###
def molarweights():
	"""returns a partial table of molecular weights for elements and oxides that can be used in other functions

    Returns
    =======
    w : dictionary
        containing the molar weights of elements and oxides:

        - si, ti, al, fe, li, na, k, mg, ca, ba, o (no upper case, symbol calling)

        - sio2, tio2, al2o3, fe2o3, feo, li2o, na2o, k2o, mgo, cao, sro, bao (no upper case, symbol calling)

    """
	w = {"si": 28.085}

    # From IUPAC Periodic Table 2016, in g/mol
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
	w["sio2"] = w["si"] + 2* w["o"]
	w["tio2"] = w["ti"] + 2* w["o"]
	w["al2o3"] = 2*w["al"] + 3* w["o"]
	w["fe2o3"] = 2*w["fe"] + 3* w["o"]
	w["feo"] = w["fe"] + w["o"]
	w["h2o"] = 2*w["h"] + w["o"]
	w["li2o"] = 2*w["li"] +w["o"]
	w["na2o"] = 2*w["na"] + w["o"]
	w["k2o"] = 2*w["k"] + w["o"]
	w["mgo"] = w["mg"] + w["o"]
	w["cao"] = w["ca"] + w["o"]
	w["sro"] = w["sr"] + w["o"]
	w["bao"] = w["ba"] + w["o"]

	w["nio"] = w["ni"] + w["o"]
	w["mno"] = w["mn"] + w["o"]
	w["p2o5"] = w["p"]*2 + w["o"]*5
	return w # explicit return
