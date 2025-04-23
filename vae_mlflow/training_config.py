import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 784
Z_DIM = 20
H1_DIM = 500
H2_DIM = 400
H3_DIM = 300
H4_DIM = 200

NUM_EPOCHS = 100
BATCH_SIZE = 128
LR_RATE = 3e-4
BETA = 1
LATENT_DIM = 20

DATA_ROOT = "/data/datasets/"

