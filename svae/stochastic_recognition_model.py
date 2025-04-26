import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from probability import reparameterization_trick


class Stochastic_Recognition_NN(nn.Module):

    def __init__(self, input_dim, latent_dim, plan, user_input_logvar=-20):
        super(Stochastic_Recognition_NN, self).__init__()

        self.d = latent_dim
        self.user_input_logvar = user_input_logvar
        self.deterministic = np.isinf(user_input_logvar)

        self.weights_mean = nn.ParameterList()
        self.weights_logvar = nn.ParameterList()
        self.bias_mean = nn.ParameterList()
        self.bias_logvar = nn.ParameterList()
        self.norms = nn.ModuleList()

        plan_with_inputs_and_outputs = [input_dim] + plan + [latent_dim]

        for plan_idx in range(1, len(plan_with_inputs_and_outputs)):
            in_size = plan_with_inputs_and_outputs[plan_idx - 1]
            out_size = plan_with_inputs_and_outputs[plan_idx]

            # Special case for last layer (2 heads)
            if plan_idx == len(plan_with_inputs_and_outputs) - 1:
                # head 1 - mean
                self.weights_mean.append(torch.Tensor(out_size, in_size))
                self.weights_logvar.append(torch.Tensor(out_size, in_size))
                self.bias_mean.append(torch.Tensor(out_size))
                self.bias_logvar.append(torch.Tensor(out_size))

                # head 2 - logvar
                self.weights_mean.append(torch.Tensor(out_size, in_size))
                self.weights_logvar.append(torch.Tensor(out_size, in_size))
                self.bias_mean.append(torch.Tensor(out_size))
                self.bias_logvar.append(torch.Tensor(out_size))
            else:
                self.weights_mean.append(torch.Tensor(out_size, in_size))
                self.weights_logvar.append(torch.Tensor(out_size, in_size))
                self.bias_mean.append(torch.Tensor(out_size))
                self.bias_logvar.append(torch.Tensor(out_size))

                # Layer normalization for everything except the last layer
                self.norms.append(nn.LayerNorm(out_size))

        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in self.weights_mean:
            init.kaiming_normal_(layer, mode="fan_in", nonlinearity="relu")
        for layer in self.weights_logvar:
            init.constant_(layer, self.user_input_logvar)
        for layer in self.bias_mean:
            init.constant_(layer, 0)
        for layer in self.bias_logvar:
            init.constant_(layer, self.user_input_logvar)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for i in range(len(self.weights_mean) - 2):
            weights_ipl = reparameterization_trick(self.weights_mean[i], self.weights_logvar[i])
            bias_ipl = reparameterization_trick(self.bias_mean[i], self.bias_logvar[i])
            x = torch.matmul(x, weights_ipl.T) + bias_ipl.view(1, -1)
            x = self.norms[i](x)
            x = F.relu(x)

        # head 1 - mean prediction
        head_1_weights_ipl = reparameterization_trick(
            self.weights_mean[-2], self.weights_logvar[-2]
        )
        head_1_bias_ipl = reparameterization_trick(self.bias_mean[-2], self.bias_logvar[-2])
        mean_z = torch.matmul(x, head_1_weights_ipl.T) + head_1_bias_ipl.view(1, -1)

        # head 2- logvar prediction
        head_2_weights_ipl = reparameterization_trick(
            self.weights_mean[-1], self.weights_logvar[-1]
        )
        head_2_bias_ipl = reparameterization_trick(self.bias_mean[-1], self.bias_logvar[-1])
        logvar_z = torch.matmul(x, head_2_weights_ipl.T) + head_2_bias_ipl.view(1, -1)

        return mean_z, logvar_z

    @torch.no_grad()
    def params_stats(self):
        stats = {}
        for i, (mu_w, mu_b, logvar_w, logvar_b) in enumerate(
            zip(self.weights_mean, self.bias_mean, self.weights_logvar, self.bias_logvar)
        ):
            stats[f"layer_{i}_weights_mean_mean"] = mu_w.mean().item()
            stats[f"layer_{i}_weights_logvar_mean"] = logvar_w.mean().item()
            stats[f"layer_{i}_weights_mean_std"] = mu_w.std().item()
            stats[f"layer_{i}_weights_logvar_std"] = logvar_w.std().item()
            stats[f"layer_{i}_bias_mean_mean"] = mu_b.mean().item()
            stats[f"layer_{i}_bias_logvar_mean"] = logvar_b.mean().item()
            stats[f"layer_{i}_bias_mean_std"] = mu_b.std().item()
            stats[f"layer_{i}_bias_logvar_std"] = logvar_b.std().item()

        return stats
