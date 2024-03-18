import ivisc, utils, torch, os, numpy as np

from sklearn.metrics import mean_squared_error

from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.air.checkpoint import Checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Calculation will be performed on {}".format(device))

cluster = False

if cluster == True:
    path = "/gpfs/users/lelosq/ivisc/"
else:
    path = "/home/charles/ownCloud/ivisc/"

def train(neuralmodel, ds, criterion, optimizer, device='cuda'):
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
    nb_folds : int, default = 1
        the number of folds for the K-fold training
    device : string, default = "cuda"
        the device where the calculations are made during training

    """
    # set model to train mode
    neuralmodel.train()

    # data tensors are sent to device
    x_visco_train = ds.x_visco_train.to(device)
    y_visco_train = ds.y_visco_train.to(device)
    T_visco_train = ds.T_visco_train.to(device)
    P_visco_train = ds.P_visco_train.to(device)

    # Forward pass on training set
    y_pred_train = neuralmodel.predict(x_visco_train,T_visco_train,P_visco_train)

    # initialise gradient
    optimizer.zero_grad() 
        
    # Compute Loss
    loss_train = criterion(y_pred_train, y_visco_train)

    loss_train.backward() # backward gradient determination
    optimizer.step() # optimiser call and step
    
    return loss_train.item()

def valid(neuralmodel, ds, criterion, device='cuda'):

    # Set model to evaluation mode
    neuralmodel.eval()

    # MONITORING VALIDATION SUBSET
    with torch.no_grad():

        # on validation set
        y_pred_valid = neuralmodel.predict(ds.x_visco_valid.to(device),ds.T_visco_valid.to(device))

        # validation loss
        loss_valid = criterion(y_pred_valid, ds.y_visco_valid.to(device))

    return loss_valid.item()

# 1. Wrap a PyTorch model in an objective function.
def objective(config):

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set dtype
    dtype = torch.float

    # custom data loader
    ds = ivisc.data_loader(path_viscosity = path+"data/all_viscosity.hdf5")

    # model declaration
    net = ivisc.model(ds.x_visco_train.shape[1],
                        hidden_size=config["nb_neurons"],
                        num_layers=config["nb_layers"],
                        p_drop=config["dropout"], 
                        activation_function = torch.nn.GELU()) # declaring model
    
    net.output_bias_init() # we initialize the output bias
    net.to(dtype=dtype, device=device) # set dtype and send to device

    # Define loss and optimizer
    criterion = torch.nn.MSELoss() # the criterion : MSE
    criterion.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), 
                                 lr=config["lr"], # 
                                 weight_decay=0.00) # L2 loss

    epoch = 0
    val_ex = 0
    patience = 200 # 250
    min_delta = 0.03

    # training with
    # early stopping criterion
    while val_ex < patience: 
        # train and valid
        train_loss = train(net, ds, criterion, optimizer, device=device)
        valid_loss = valid(net, ds, criterion, device=device)

        diff_loss = np.sqrt((train_loss - valid_loss)**2)

        # calculating ES criterion
        if epoch == 0:
            best_loss_v = valid_loss
            best_diff_v = diff_loss
        elif valid_loss <= (best_loss_v - min_delta): # if improvement is significant, this saves the model
            val_ex = 0
            best_loss_v = valid_loss
            best_diff_v = diff_loss

            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and can be accessed through `session.get_checkpoint()`
            # API in future iterations.
            os.makedirs(path+"models/ray_model", exist_ok=True)
            torch.save(net.state_dict(), 
                path+"models/ray_model/checkpoint.pt")
        else:
            val_ex += 1

        epoch += 1
                
    # Report the current loss to Tune.
    checkpoint = Checkpoint.from_directory(path+"models/ray_model/")
    session.report({"mean_loss": best_loss_v+best_diff_v,
                   "valid_loss": best_loss_v, 
                   "diff_loss": best_diff_v,
                   "epoch": epoch}, 
                   checkpoint=checkpoint)

# 2. Define a search space and initialize the search algorithm.
search_space = {"lr": tune.loguniform(5e-5, 1e-3), 
                "nb_neurons": tune.randint(200, 600),
                "nb_layers": tune.randint(2, 5),
                "dropout": tune.uniform(0.1, 0.4)}
                #"wd": tune.loguniform(1e-5, 1e-2),}
 
# Using OptunaSearch
# we set initial parameters
initial_params = [
    {"lr": 0.0003, "nb_neurons": 450, "nb_layers": 2, "dropout": 0.2},
    {"lr": 0.0003, "nb_neurons": 450, "nb_layers": 3, "dropout": 0.3},
    {"lr": 0.0003, "nb_neurons": 450, "nb_layers": 4, "dropout": 0.2},
    {"lr": 0.0003, "nb_neurons": 450, "nb_layers": 5, "dropout": 0.3},
    {"lr": 0.0002, "nb_neurons": 450, "nb_layers": 6, "dropout": 0.2},
    {"lr": 0.0002, "nb_neurons": 450, "nb_layers": 6, "dropout": 0.3},
]

algo = OptunaSearch(points_to_evaluate=initial_params)

# AsyncHyperBand enables aggressive early stopping of bad trials.
scheduler = ASHAScheduler(time_attr="epoch", grace_period=100, max_t=6000)

tuner = tune.Tuner(
   tune.with_resources(objective, {"gpu": 4}),
   tune_config=tune.TuneConfig(
       num_samples=100,
       metric="valid_loss",
       mode="min",
       search_alg=algo,
       #scheduler=scheduler
   ),
#   run_config=air.RunConfig(
#       stop={"training_iteration": 5},
#   ),
   param_space=search_space,
)

# 4. Run the optimization.
results = tuner.fit()

# 5. Get the best configuration.
# print("Best config is:", results.get_best_result(metric=["valid_loss","diff_loss"], mode=["min","min"]).config)

