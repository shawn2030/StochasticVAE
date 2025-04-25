import torch
from torch import nn
from training_config import LATENT_DIM


class VAE(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, beta: float = 1.0):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def reparameterize(self, mu_z, logvar_z):
        """Sample from the approximate posterior using the reparameterization trick."""

        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)

        z = mu_z + eps * std

        return z
   

    def loss(self, x):

        mu_z, logvar_z = self.encoder(x)

        z = self.reparameterize(mu_z, logvar_z)
        z = z.view((-1, LATENT_DIM))

        x_recon = self.decoder(z)

        reconstruction_term = self.decoder.log_likelihood(x, x_recon).sum()
        # print(reconstruction_term.shape)
        kl_term = self.encoder.kl(mu_z, logvar_z)  # shape is []
        # elbo = reconstruction_term - kl_term
        elbo =  reconstruction_term - self.beta * kl_term
        loss = -elbo.mean()

        return loss, x_recon, kl_term, reconstruction_term

    def get_encoder_num_layers(self):
        num_layers = 0
        for name, param in self.encoder.named_parameters():
            if "weight" in name:
                num_layers += 1

        return num_layers

    def get_encoder_layer_width(self):
        pass
