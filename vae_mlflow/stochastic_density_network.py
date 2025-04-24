import torch
import torch.nn as nn
from training_config import PLAN_DECODER, LATENT_DIM
import torch.nn.functional as F


class Stochastic_Density_NN(nn.Module):

    def __init__(self, input_dim, z_dim, user_input_logvar=-2.5):
        super(Stochastic_Density_NN, self).__init__()

        self.fc5 = nn.Linear(LATENT_DIM, PLAN_DECODER[0])
        self.fc6 = nn.Linear(PLAN_DECODER[0], PLAN_DECODER[1])
        self.fc7 = nn.Linear(PLAN_DECODER[1], PLAN_DECODER[2])
        self.fc8 = nn.Linear(PLAN_DECODER[2], PLAN_DECODER[3])
        self.fc9 = nn.Linear(PLAN_DECODER[3], input_dim)

        # TIME BEING - Add a diagonal covariance in pixel space (Unnecessary; remove later)
        self.logvar_x = nn.Parameter(torch.zeros(input_dim))

    def reparameterization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std

        return z

    def log_likelihood_gaussian(self, x, mu_z, logvar_z):
        return -0.5 * (logvar_z + (x - mu_z) ** 2 / logvar_z.exp()).sum(dim=-1)

    def log_likelihood(self, x, recon_x, flatten_dim: int = 1):
        """Calculate p( x|mu,Sigma) for a gaussian with diagonal covariance."""
        # Flatten everything
        x = torch.flatten(x, start_dim=flatten_dim)
        recon_x = torch.flatten(recon_x, start_dim=flatten_dim)
        return self.log_likelihood_gaussian(x, recon_x, self.logvar_x)

    def forward(self, z):
        h5 = F.relu(self.fc5(z))
        h6 = F.relu(self.fc6(h5))
        h7 = F.relu(self.fc7(h6))
        h8 = F.relu(self.fc8(h7))
        # h9 = F.relu(self.fc9(h8))

        recon_x = torch.sigmoid(self.fc9(h8))

        return recon_x
