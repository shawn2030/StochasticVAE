import os
import tempfile

import lightning as lit
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from entropy import entropy_singh_2003_up_to_constants, entropy_singh_2003
from probability import (
    log_det_fisher,
    reparameterization_trick,
    kl_q_prior,
    log_prob_diagonal_gaussian,
)


class Stochastic_VAE(lit.LightningModule):

    # Note: batch and n_forward dims must be 0 and 1 so that the flattening of pixels starts at 2
    batch_dim = 0
    n_forward_dim = 1
    z_dim = -1

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        k_neighbor: int = 1,
        n_forward: int = 4,
        lr:float = 1e-5,
        lambda_: float = 2.0,
        ablate_entropy: bool = False,
        ablate_fim: bool = False,
    ):
        super(Stochastic_VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.lambda_ = lambda_
        self.opt = self.sched = None
        self.n_forward = n_forward
        self.k_neighbor = k_neighbor

        self.vmap_encoder = torch.vmap(
            self.encoder,
            in_dims=self.n_forward_dim,
            out_dims=self.n_forward_dim,
            randomness="different",
        )
        self.vmap_decoder = torch.vmap(
            self.decoder,
            in_dims=self.n_forward_dim,
            out_dims=self.n_forward_dim,
            randomness="error",
        )

        self.ablate_entropy = ablate_entropy
        self.ablate_fim = ablate_fim

    def loss(self, x):
        batch_size = x.size(0)
        x_stack = torch.stack([x] * self.n_forward, dim=self.n_forward_dim)

        # Run the encoder (recognition model)
        multi_mu_z, multi_logvar_z = self.vmap_encoder(x_stack)

        # Sample from the approximate posterior by drawing one sample from each gaussian component
        z = reparameterization_trick(multi_mu_z, multi_logvar_z)

        # Run the decoder (generative model)
        x_recon = self.vmap_decoder(z)

        # Terms from average of classic ELBO
        reconstruction_term = self.decoder.log_likelihood(x_stack, x_recon, flatten_dim=2).mean(
            dim=self.n_forward_dim
        )
        kl_term = kl_q_prior(multi_mu_z, multi_logvar_z, dim=self.z_dim).mean(
            dim=self.n_forward_dim
        )

        # Terms added for the stochastic ELBO (approximating mutual information)
        entropy_term = entropy_singh_2003_up_to_constants(
            torch.cat([multi_mu_z, multi_logvar_z], dim=self.z_dim),
            k=self.k_neighbor,
            dim_samples=self.n_forward_dim,
            dim_features=self.z_dim,
        )
        fim_term = log_det_fisher(multi_mu_z, multi_logvar_z, dim=self.z_dim).mean(
            dim=self.n_forward_dim
        )

        assert kl_term.shape == (batch_size,)
        assert entropy_term.shape == (batch_size,)
        assert reconstruction_term.shape == (batch_size,)
        assert fim_term.shape == (batch_size,)

        if self.ablate_entropy:
            entropy_term = entropy_term.detach()

        if self.ablate_fim:
            fim_term = fim_term.detach()

        # The fancy stochastic ELBO
        avg_classic_elbo = reconstruction_term - kl_term
        if self.encoder.deterministic:
            elbo = avg_classic_elbo
        else:
            approx_mi = entropy_term + fim_term / 2
            elbo = avg_classic_elbo + approx_mi / self.lambda_

        loss = -elbo.mean(dim=0)

        return {
            "loss": loss,
            "reconstruction_term": reconstruction_term.mean(),
            "kl_term": kl_term.mean(),
            "entropy_term": entropy_term.mean(),
            "fim_term": fim_term.mean(),
            "reconstruction": x_recon[:, 0],
            "avg_multi_mu_z": multi_mu_z.mean(),
            "avg_multi_logvar_z": multi_logvar_z.mean(),
        }

    def inference_entropy_gap_m_p(self, x):
        batch_size = x.size(0)
        x_stack = torch.stack([x] * self.n_forward, dim=self.n_forward_dim)

        # Run the encoder (recognition model)
        multi_mu_z, multi_logvar_z = self.vmap_encoder(x_stack)

        # Sample from the approximate posterior by drawing one sample from each gaussian component
        z = reparameterization_trick(multi_mu_z, multi_logvar_z)

        # Run the decoder (generative model)
        x_recon = self.vmap_decoder(z)

        # Terms from average of classic ELBO
        reconstruction_term = self.decoder.log_likelihood(x_stack, x_recon, flatten_dim=2).mean(
            dim=self.n_forward_dim
        )
        log_p_z = log_prob_diagonal_gaussian(
            z, torch.zeros_like(z), torch.zeros_like(z), dim=self.z_dim
        ).mean(dim=self.n_forward_dim)

        singh_entropy_term = entropy_singh_2003(
            z,
            k=self.k_neighbor,
            dim_samples=self.n_forward_dim,
            dim_features=self.z_dim,
        )

        assert reconstruction_term.shape == (batch_size,)
        assert log_p_z.shape == (batch_size,)
        assert singh_entropy_term.shape == (batch_size,)

        entropy_gap = -(reconstruction_term + log_p_z) - singh_entropy_term

        mean = entropy_gap.mean()
        second_moment = (entropy_gap**2).mean()

        return {
            "moment1": mean,
            "moment2": second_moment
        }

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
        terms = self.loss(x)
        self.log("train_loss", terms["loss"])
        self.log("train_kl", terms["kl_term"])
        self.log("train_reconstruction", terms["reconstruction_term"])
        self.log("train_entropy", terms["entropy_term"])
        self.log("train_fischer_information_matrix", terms["fim_term"])

        return terms["loss"]

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        terms = self.loss(x)
        self.log("val_loss", terms["loss"])
        self.log("val_kl", terms["kl_term"])
        self.log("val_reconstruction", terms["reconstruction_term"])
        self.log("val_entropy", terms["entropy_term"])
        self.log("val_fischer_information_matrix", terms["fim_term"])
        self.log("val_multi_mu_z_mean", terms["avg_multi_mu_z"])
        self.log("val_multi_logvar_z_mean", terms["avg_multi_logvar_z"])

        if batch_idx % 10 == 0:
            # Log parameter stats
            self.log_dict(self.encoder.params_stats())

            # Log images
            x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=8)
            recon_grid = make_grid(terms["reconstruction"].view(-1, 1, 28, 28), nrow=8)

            with tempfile.TemporaryDirectory() as temp_dir:
                input_image_path = os.path.join(temp_dir, "inputs.png")
                recon_image_path = os.path.join(temp_dir, "reconstructions.png")
                save_image(x_grid, input_image_path)
                save_image(recon_grid, recon_image_path)

                run_id = self.logger.run_id
                self.logger.experiment.log_artifacts(
                    local_dir=temp_dir, artifact_path="validation_images", run_id=run_id
                )

        return terms["loss"]

    def test_step(self, batch, batch_idx):
        x, _ = batch
        terms = self.loss(x)
        self.log("test_loss", terms["loss"])
        self.log("test_kl", terms["kl_term"])
        self.log("test_reconstruction", terms["reconstruction_term"])
        self.log("test_entropy", terms["entropy_term"])
        self.log("test_fischer_information_matrix", terms["fim_term"])
        self.log("test_multi_mu_z_mean", terms["avg_multi_mu_z"])
        self.log("test_multi_logvar_z_mean", terms["avg_multi_logvar_z"])

        # Log parameter stats
        self.log_dict(self.encoder.params_stats())

        # Log "inference goodness"
        terms = self.inference_entropy_gap_m_p(x)
        self.log("goodness_moment1", terms["moment1"])
        self.log("goodness_moment2", terms["moment2"])

        # Log images generated from the prior
        z = torch.randn(64, 1, self.encoder.d, device=self.device)
        gen_images = self.decoder(z)
        gen_img_grid = make_grid(gen_images.view(-1, 1, 28, 28), nrow=8)
        with tempfile.TemporaryDirectory() as temp_dir:
            gen_image_path = os.path.join(temp_dir, "generated_output.png")
            save_image(gen_img_grid, gen_image_path)

            run_id = self.logger.run_id
            self.logger.experiment.log_artifacts(
                local_dir=temp_dir, artifact_path="gen_images", run_id=run_id
            )
