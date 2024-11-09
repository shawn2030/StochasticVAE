import torch
import torch.nn as nn
from training_config import LATENT_DIM, LEARNING_RATE
import lightning as lit
from torchvision.utils import make_grid


class Stochastic_VAE(lit.LightningModule):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(Stochastic_VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.lr = LEARNING_RATE
        self.opt = self.sched = None

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
        kl_term = self.encoder.kl(mu_z, logvar_z)  # shape is []
        # elbo = reconstruction_term - kl_term
        loss = kl_term - reconstruction_term

        return loss.sum(), kl_term, reconstruction_term, x_recon

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=self.lr / 32
        )
        self.opt = optimizer
        self.sched = lr_scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            },
        }

    def on_train_epoch_start(self) -> None:
        self.log("lr", self.sched.get_last_lr()[0])

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, kl_term, reconstruction_term, _ = self.loss(x)
        self.log("train_loss", loss)
        self.log("train_kl", kl_term)
        self.log("train_reconstruction", reconstruction_term)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss, kl_term, reconstruction_term, x_recon = self.loss(x)
        self.log("val_loss", loss)
        self.log("val_kl", kl_term)
        self.log("val_reconstruction", reconstruction_term)
        if batch_idx == 0:
            # Log parameter stats
            self.log_dict(self.encoder.params_stats())

            # Log images
            x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=8)
            recon_grid = make_grid(x_recon.view(-1, 1, 28, 28), nrow=8)
            # TODO - these are not visible in the UI for some reason
            # TODO - will try using log_artifacts if it is provided
            self.logger.experiment.log_image(
                key="inputs",
                image=x_grid.cpu().permute(1, 2, 0).numpy(),
                run_id=self.logger.run_id,
            )
            self.logger.experiment.log_image(
                key="reconstructions",
                image=recon_grid.cpu().permute(1, 2, 0).numpy(),
                run_id=self.logger.run_id,
            )

        return loss
