import os

##################
## Model config ##
##################

PLAN = [500, 300, 200, 100, 50]
PLAN_DECODER = [50, 100, 300, 500]
LATENT_DIM = 20
EPSILON = 1e-6
LAMBDA = 10e+8      # range [1, +inf]
NUMBER_OF_NEAREST_NEIGHBORS = 4
N_FORWARD_PASS = 8
#####################
## Training config ##
#####################

LEARNING_RATE = 1e-5
EPOCHS = 200
BATCH_SIZE = 128
USER_INPUT_LOGVAR = -10    # Make this a function of Epochs Deterministci -> Stochastic

########################
## Environment config ##
########################

# Only let lightning 'see' one GPU, but can be overridden by setting the environment variable
# CUDA_VISIBLE_DEVICES from outside the script.
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
# MLFLOW_TRACKING_URI = "/data/projects/SVAE/mlruns/"
MLFLOW_TRACKING_URI = "mlruns/"

DATA_ROOT = "/data/datasets/"
