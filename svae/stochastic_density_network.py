import torch
import torch.nn as nn

from probability import log_prob_diagonal_gaussian
import torch.nn.functional as F


class Stochastic_Density_NN(nn.Module):

    def __init__(self, input_dim, latent_dim, plan):
        super(Stochastic_Density_NN, self).__init__()
        plan_with_inputs_and_outputs = [latent_dim] + plan + [input_dim]

        self.layers = nn.ModuleList()
        for i in range(1, len(plan_with_inputs_and_outputs)):
            in_size = plan_with_inputs_and_outputs[i - 1]
            out_size = plan_with_inputs_and_outputs[i]
            self.layers.append(nn.Linear(in_size, out_size))

        # TIME BEING - Add a diagonal covariance in pixel space (Unnecessary; remove later)
        self.logvar_x = nn.Parameter(torch.zeros(input_dim))

    def log_likelihood(self, x, recon_x, flatten_dim: int = 1):
        """Calculate p( x|mu,Sigma) for a gaussian with diagonal covariance."""
        # Flatten everything
        x = torch.flatten(x, start_dim=flatten_dim)
        recon_x = torch.flatten(recon_x, start_dim=flatten_dim)
        return log_prob_diagonal_gaussian(x, recon_x, self.logvar_x)

    def forward(self, z):
        for layer in self.layers[:-1]:
            z = F.relu(layer(z))
        return torch.sigmoid(self.layers[-1](z))
