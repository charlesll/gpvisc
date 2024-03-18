import numpy as np
import torch, time
import torch.nn.functional as F
import h5py
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.constants import Avogadro, Planck

from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import keras
import tensorflow as tf


###
### FUNCTIONS FOR HANDLNG DATA
###

class data_loader_old():
    """custom data loader for batch training

    """
    def __init__(self, 
                 path_viscosity = "./data/all_viscosity.hdf5", 
                 scaling = False):
        """
        Inputs
        ------
        path_viscosity : string
            path for the viscosity HDF5 dataset
        scaling : False or True
            Scales the input chemical composition.
            WARNING : Does not work currently as this is a relic of testing this effect,
            but we chose not to scale inputs and Cp are calculated with unscaled values in the network.
        """
        
        f = h5py.File(path_viscosity, 'r')

        # List all groups
        self.X_columns = f['X_columns'][()]

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

        # training shapes
        print("")
        print("This is for checking the shape consistency of the dataset:\n")

        print("Visco train shape")
        print(self.x_visco_train.shape)
        print(self.T_visco_train.shape)
        print(self.y_visco_train.shape)


class ivisc(torch.nn.Module):
    """greybox model leveraging the MYEGA equation

    """
    def __init__(self, input_size, hidden_size = 300, num_layers = 4, nb_channels_raman = 800,
                 p_drop=0.2, activation_function = torch.nn.ReLU(),
                 shape="rectangle",
                 dropout_pos_enc=0.01, n_heads=4,
                 num_encoder_layers=1, dim_feedforward_encoder=128,
                 d_model = 32, d_output = 32):
        """Initialization of the model

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
        super(ivisc, self).__init__()

        # init parameters
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_layers  = len(hidden_size)
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

            self.linears = torch.nn.ModuleList([torch.nn.Linear(self.d_output*self.input_size, self.hidden_size[0])])
            self.linears.extend([torch.nn.Linear(self.hidden_size[i], self.hidden_size[i+1]) for i in range(1, self.num_layers-1)])
        else:
            # general shape of the network
            self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, self.hidden_size[0])])
            self.linears.extend([torch.nn.Linear(self.hidden_size[i], self.hidden_size[i+1]) for i in range(0, self.num_layers-1)])

        ###
        # output layers
        ###
        self.out_thermo = torch.nn.Linear(self.hidden_size[-1], 4) # Linear output, Ae, Tg, m

        ###
        # General Ae
        ###
        self.ae = torch.nn.Parameter(torch.tensor([-3.5]),requires_grad=True)

    def output_bias_init(self):
        """bias initialisation for self.out_thermo
        """
        self.out_thermo.bias = torch.nn.Parameter(data=torch.tensor([np.log(1000.), # Tg
                                                                     np.log(21.), # m
                                                                     0., # Peff Tg
                                                                     0., # Peff m
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

    def tg(self,x):
        """glass transition temperature Tg"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,0])
        return torch.reshape(out, (out.shape[0], 1))

    def fragility(self,x):
        """fragility"""
        out = torch.exp(self.out_thermo(self.forward(x))[:,1])
        return torch.reshape(out, (out.shape[0], 1))

    def predict(self,x,T,P):
        """viscosity from the myega equation, given chemistry X and temperature T

        need for speed = we decompose the calculation as much as reasonable for a minimum amount of forward pass
        """
        # analyse the chemical dataset
        X_analysis = self.forward(x)

        # one forward pass to get thermodynamic output
        thermo_out = self.out_thermo(X_analysis)

        # get pressure effects
        Peff_tg = torch.reshape(thermo_out[:,2], (thermo_out[:,2].shape[0], 1))
        Peff_m = torch.reshape(thermo_out[:,3], (thermo_out[:,3].shape[0], 1))

        # get Tg
        out = torch.exp(thermo_out[:,1])
        tg = torch.reshape(out, (out.shape[0], 1)) + P*Peff_tg

        # get fragility
        out = torch.exp(thermo_out[:,2])
        frag = torch.reshape(out, (out.shape[0], 1)) + P*Peff_m

        return self.ae + (12.0 - self.ae)*(tg/T)*torch.exp((frag/(12.0-self.ae)-1.0)*(tg/T-1.0))
    
    
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

    while val_ex <= train_patience:

        #
        # TRAINING
        #
        loss = 0 # initialize the sum of losses of each fold

        # Forward pass on training set
        y_pred_train = neuralmodel.predict(x_visco_train, T_visco_train, P_visco_train)

        # Compute Loss
        loss = criterion(y_pred_train, y_visco_train)

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
            y_pred_valid = neuralmodel.predict(ds.x_visco_valid.to(device), ds.T_visco_valid.to(device), ds.P_visco_valid.to(device))

            # validation loss
            loss_v = criterion(y_pred_valid, ds.y_visco_valid.to(device))

            record_valid_loss.append(loss_v.item())

        #
        # Print info on screen
        #
        if verbose == True:
            if (epoch % 100 == 0):
                print('\nTRAIN -- V: {:.3f}'.format(loss))
                print('VALID -- V: {:.3f}\n'.format(loss_v))
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
        print('\nTRAIN -- V: {:.3f}'.format(loss))
        print('VALID -- V: {:.3f}\n'.format(loss_v))

    return neuralmodel, record_train_loss, record_valid_loss


###
### KERAS
###

class MyegaCalculationLayer_model1(Layer):
    def __init__(self, **kwargs):
        super(MyegaCalculationLayer_model1, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ae = self.add_weight(name='ae',
                                  shape=(),
                                  initializer=keras.initializers.Constant(-3.5579),
                                  trainable=True)
        super(MyegaCalculationLayer_model1, self).build(input_shape)

    def call(self, inputs):
        x, t, tg, frag = inputs
        n_myega = self.ae + (12.0 - self.ae) * (tg / t) * tf.math.exp((frag / (12.0 - self.ae) - 1.0) * (tg / t - 1.0))
        return n_myega
    
    def get_config(self):
        config = super().get_config()
        return config
    
class MyegaCalculationLayer_model2(Layer):
    def __init__(self, **kwargs):
        super(MyegaCalculationLayer_model2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ae = self.add_weight(name='ae',
                                  shape=(),
                                  initializer=keras.initializers.Constant(-3.5579),
                                  trainable=True)
        super(MyegaCalculationLayer_model2, self).build(input_shape)

    def call(self, inputs):
        x, t, p, tg, frag, peff_tg, peff_m = inputs
        frag_P = frag + peff_m*p
        tg_P = tg + peff_tg*p
        n_myega = self.ae + (12.0 - self.ae) * (tg_P / t) * tf.math.exp((frag_P / (12.0 - self.ae) - 1.0) * (tg_P / t - 1.0))
        return n_myega
    
    def get_config(self):
        config = super().get_config()
        return config
    
def build_greybox_1(
    input_shape,
    mlp_units,
    mlp_dropout=0,
    activation="swish"
):
    inputs = [keras.Input(shape=input_shape),keras.Input(1)]
    x, t = inputs

    for dim in mlp_units:
        x = layers.Dense(dim)(x)
        x = layers.Activation(activation)(x)
        x = layers.Dropout(mlp_dropout)(x)

    tg = tf.math.exp(layers.Dense(1, 
                                  bias_initializer=keras.initializers.Constant(6.2), 
                                  kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), 
                                  name="Tg")(x)) # linear output for regression
    frag = tf.math.exp(layers.Dense(1, 
                                    bias_initializer=keras.initializers.Constant(3.2), 
                                    kernel_initializer=keras.initializers.RandomNormal(stddev=0.01), 
                                    name="frag")(x)) # linear output for regression
    
    # Use custom layer for n_myega calculation
    n_myega_layer = MyegaCalculationLayer_model1()
    n_myega = n_myega_layer([x, t, tg, frag])

    return keras.Model(inputs, n_myega), n_myega_layer

def build_greybox_2(
    input_shape,
    mlp_units,
    mlp_dropout=0,
):
    inputs = [keras.Input(shape=input_shape),keras.Input(1),keras.Input(1)]
    x, t, p = inputs

    for dim in mlp_units:
        x = layers.Dense(dim)(x)
        x = layers.Activation("swish")(x)
        x = layers.Dropout(mlp_dropout)(x)

    tg = tf.math.exp(layers.Dense(1, bias_initializer=keras.initializers.Constant(6.2), name="Tg")(x)) # linear output for regression
    frag = tf.math.exp(layers.Dense(1, bias_initializer=keras.initializers.Constant(3.2), name="frag")(x)) # linear output for regression
    peff_tg = layers.Dense(1, bias_initializer=keras.initializers.Constant(3.2), name="peff_tg")(x) # linear output for regression
    peff_m = layers.Dense(1, bias_initializer=keras.initializers.Constant(3.2), name="peff_m")(x) # linear output for regression
    
    # Use custom layer for n_myega calculation
    n_myega_layer = MyegaCalculationLayer_model2()
    n_myega = n_myega_layer([x, t, p, tg, frag, peff_tg, peff_m])

    return keras.Model(inputs, n_myega), n_myega_layer

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = [keras.Input(shape=input_shape),keras.Input(1)]
    x, t = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="swish")(x)
        x = layers.Dropout(mlp_dropout)(x)

    ae = -3.5#layers.Dense(1, bias_initializer=keras.initializers.Constant(-3.5), name='Ae')(x) # linear output for regression
    tg = tf.math.exp(layers.Dense(1, bias_initializer=keras.initializers.Constant(6.2), name="Tg")(x)) # linear output for regression
    frag = tf.math.exp(layers.Dense(1, bias_initializer=keras.initializers.Constant(3.2), name="frag")(x)) # linear output for regression
    
    a_tvf = -4.5#layers.Dense(1, bias_initializer=keras.initializers.Constant(-4.5), name='A')(x) # linear output for regression
    c_tvf = tf.math.exp(layers.Dense(1, bias_initializer=keras.initializers.Constant(6.2), name="C")(x)) # linear output for regression
    b_tvf = tf.math.exp(layers.Dense(1, bias_initializer=keras.initializers.Constant(7.7), name="B")(x))#((12.0-a_tvf)*(tg-c_tvf))
    
    # viscosity calculations
    n_myega = ae + (12.0 - ae)*(tg/t)*tf.math.exp((frag/(12.0-ae)-1.0)*(tg/t-1.0))
    n_tvf = a_tvf + b_tvf/(t - c_tvf)
    outputs = [n_myega, n_tvf]
    return keras.Model(inputs, outputs)

def build_FFN_FV(
    input_shape,
    mlp_units,
    mlp_dropout=0,
):
    inputs = [keras.Input(shape=input_shape),keras.Input(1)]
    x, t = inputs

    for dim in mlp_units:
        x = layers.Dense(dim, activation="gelu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    #a_cg = layers.Dense(1, bias_initializer=keras.initializers.Constant(-3.5), name='A')(x) # linear output for regression
    a_cg = -4.5
    tg = tf.math.exp(layers.Dense(1, bias_initializer=keras.initializers.Constant(6.9), name="Tg")(x)) # linear output for regression
    to_cg = tf.math.exp(layers.Dense(1, bias_initializer=keras.initializers.Constant(3.2), name="To")(x)) # linear output for regression
    c_cg = tf.math.exp(layers.Dense(1, bias_initializer=keras.initializers.Constant(3.2), name="C")(x)) # linear output for regression
    b_cg = tf.math.exp(layers.Dense(1, bias_initializer=keras.initializers.Constant(7.1), name='B')(x)) # linear output for regression
    # Ae + tf.math.exp(B)/(t-tf.math.exp(C))
    #b_cg = 0.5*(12.0 - a_cg) * (tg - to_cg + tf.math.sqrt((tg - to_cg)**2 + c_cg*tg))
    outputs = a_cg + 2.0*b_cg/(t - to_cg + tf.math.sqrt((t-to_cg)**2 + c_cg*t))
    return keras.Model(inputs, outputs)