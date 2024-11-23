import torch
import torch.nn as nn
from training_config import LATENT_DIM, LEARNING_RATE
import lightning as lit
from torchvision.utils import make_grid, save_image
from entropy import entropy_singh_2003_up_to_constants
import os


class Stochastic_VAE(lit.LightningModule):

    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, k_neighbor: int = 1, n_forward: int = 4
    ):
        super(Stochastic_VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.lr = LEARNING_RATE
        self.opt = self.sched = None
        self.n_forward = n_forward
        self.k_neighbor = k_neighbor

        self.vmap_encoder = torch.vmap(self.encoder, in_dims=0, out_dims=0, randomness="different")
        self.vmap_decoder = torch.vmap(self.decoder, in_dims=0, out_dims=0, randomness="error")

    def reparameterize(self, mu_z, logvar_z):
        """Sample from the approximate posterior using the reparameterization trick."""

        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)

        z = mu_z + eps * std

        return z
    
    def log_det_fisher(self,multi_mu_z, multi_logvar_z):
        return -2 * torch.sum(multi_logvar_z, dim=-1) + torch.tensor(LATENT_DIM) * torch.log(torch.tensor(2))

    def loss(self, x):
        batch_size = x.size(0)
        # mu_z, logvar_z = self.encoder(x)
        x_stack = torch.stack([x] * self.n_forward, dim=0)

        # Run the encoder (recognition model)
        multi_mu_z, multi_logvar_z = self.vmap_encoder(x_stack)
        kl_term = self.encoder.kl(multi_mu_z, multi_logvar_z).mean(dim=0)
        entropy_term = entropy_singh_2003_up_to_constants(
            torch.cat([multi_mu_z, multi_logvar_z], dim=-1),
            k=self.k_neighbor,
            dim_samples=0,
            dim_features=-1,
        )
        fim_term = 0.5 * self.log_det_fisher(multi_mu_z, multi_logvar_z).mean()

        # Sample from the approximate posterior
        z = self.reparameterize(multi_mu_z, multi_logvar_z)

        # Run the decoder (generative model)
        x_recon = self.vmap_decoder(z)
        reconstruction_term = self.decoder.log_likelihood(x_stack, x_recon, flatten_dim=2).mean(
            dim=0
        )

        assert kl_term.shape == (batch_size,)
        assert entropy_term.shape == (batch_size,)
        assert reconstruction_term.shape == (batch_size,)

        # TODO - average (rather than sum) the loss over the batch dimension. In all parts of the
        #  loss calculation (recon, kl, entropy, etc.)
        fancy_stochastic_elbo = entropy_term - kl_term + 1 / 2 * fim_term + reconstruction_term
        loss = -fancy_stochastic_elbo.mean()

        return loss, kl_term.mean(), reconstruction_term.mean(), entropy_term.mean(), x_recon[0], multi_mu_z.mean(), multi_logvar_z.mean()

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
        loss, kl_term, reconstruction_term, entropy_term, _, multi_mu_z_mean, multi_logvar_z_mean = self.loss(x)
        self.log("train_loss", loss)
        self.log("train_kl", kl_term)
        self.log("train_reconstruction", reconstruction_term)
        self.log("train_entropy", entropy_term)

        return loss

    def validation_step(self, batch, batch_idx):
        temp_dir = "svae/output"
        x, _ = batch
        loss, kl_term, reconstruction_term, entropy_term, x_recon, multi_mu_z_mean, multi_logvar_z_mean = self.loss(x)
        self.log("val_loss", loss)
        self.log("val_kl", kl_term)
        self.log("val_reconstruction", reconstruction_term)
        self.log("val_entropy", entropy_term)
        self.log("multi_mu_z_mean", multi_mu_z_mean)
        self.log("multi_logvar_z_mean", multi_logvar_z_mean)
        if batch_idx % 10 == 0:
            # Log parameter stats
            self.log_dict(self.encoder.params_stats())

            # Log images
            x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=8)
            recon_grid = make_grid(x_recon.view(-1, 1, 28, 28), nrow=8)

            input_image_path = os.path.join(temp_dir, "inputs.png")
            recon_image_path = os.path.join(temp_dir, "reconstructions.png")
            save_image(x_grid, input_image_path)
            save_image(recon_grid, recon_image_path)

            run_id = self.logger.run_id 
            self.logger.experiment.log_artifacts(local_dir=temp_dir, artifact_path="validation_images", run_id = run_id)

            os.remove(input_image_path)
            os.remove(recon_image_path)

        return loss
    
    def test_step(self, batch, batch_idx):
        temp_dir = "svae/output"
        x, _ = batch
        loss, kl_term, reconstruction_term, entropy_term, x_recon, multi_mu_z_mean, multi_logvar_z_mean = self.loss(x)
        self.log("test_loss", loss)
        self.log("test_kl", kl_term)
        # self.log("val_reconstruction", reconstruction_term)
        # self.log("val_entropy", entropy_term)
        self.log("test_multi_mu_z_mean", multi_mu_z_mean)
        self.log("test_multi_logvar_z_mean", multi_logvar_z_mean)
        
        # Log parameter stats
        self.log_dict(self.encoder.params_stats())

        # Log images
        x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=8)
        gen_img_grid = make_grid(x_recon.view(-1, 1, 28, 28), nrow=8)

        gen_image_path = os.path.join(temp_dir, "generated_output.png")
        save_image(gen_img_grid, gen_image_path)

        run_id = self.logger.run_id 
        self.logger.experiment.log_artifacts(local_dir=temp_dir, artifact_path="gen_images", run_id = run_id)

        # os.remove(gen_image_path)

        
