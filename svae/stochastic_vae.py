import torch
import torch.nn as nn
from training_config import LATENT_DIM, LEARNING_RATE, LAMBDA
import lightning as lit
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
from torchvision.utils import make_grid, save_image
from entropy import entropy_singh_2003_up_to_constants
import os


class Stochastic_VAE(lit.LightningModule):


    # Note: batch and n_forward dims must be 0 and 1 so that the flattening of pixels starts at 2
    batch_dim = 0
    n_forward_dim = 1
    z_dim = -1

    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, k_neighbor: int = 1, n_forward: int = 4, ablate_entropy: bool = False, ablate_fim: bool = False
    ):
        super(Stochastic_VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.lr = LEARNING_RATE
        self.opt = self.sched = None
        self.n_forward = n_forward
        self.k_neighbor = k_neighbor

        self.vmap_encoder = torch.vmap(self.encoder, in_dims=self.n_forward_dim, out_dims=self.n_forward_dim, randomness="different")
        self.vmap_decoder = torch.vmap(self.decoder, in_dims=self.n_forward_dim, out_dims=self.n_forward_dim, randomness="error")

        self.ablate_entropy = ablate_entropy
        self.ablate_fim = ablate_fim

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
        x_stack = torch.stack([x] * self.n_forward, dim=self.n_forward_dim)

        # Run the encoder (recognition model)
        multi_mu_z, multi_logvar_z = self.vmap_encoder(x_stack)
        kl_term = self.encoder.kl(multi_mu_z, multi_logvar_z).mean(dim=self.n_forward_dim)
        entropy_term = entropy_singh_2003_up_to_constants(
            torch.cat([multi_mu_z, multi_logvar_z], dim=self.z_dim),
            k=self.k_neighbor,
            dim_samples=self.n_forward_dim,
            dim_features=self.z_dim,
        )
        fim_term = 0.5 * self.log_det_fisher(multi_mu_z, multi_logvar_z).mean(dim=self.n_forward_dim)


        # Sample from the approximate posterior
        z = self.reparameterize(multi_mu_z, multi_logvar_z)

        # Run the decoder (generative model)
        x_recon = self.vmap_decoder(z)
        reconstruction_term = self.decoder.log_likelihood(x_stack, x_recon, flatten_dim=2).mean(
            dim=self.n_forward_dim
        )

        assert kl_term.shape == (batch_size,)
        assert entropy_term.shape == (batch_size,)
        assert reconstruction_term.shape == (batch_size,)
        assert fim_term.shape == (batch_size,)

        # TODO - average (rather than sum) the loss over the batch dimension. In all parts of the
        #  loss calculation (recon, kl, entropy, etc.)
        if self.ablate_entropy:
            entropy_term = entropy_term.detach()

        if self.ablate_fim:
            fim_term = fim_term.detach()
            
        fancy_stochastic_elbo = entropy_term - kl_term + 1 / 2 * fim_term * (1/LAMBDA) + reconstruction_term
        loss = -fancy_stochastic_elbo.mean()

        return loss, kl_term.mean(), reconstruction_term.mean(), entropy_term.mean(), x_recon[0], multi_mu_z.mean(), multi_logvar_z.mean(), fim_term.mean()

    def inference_entropy_gap_m_p(self, x):
        x_stack = torch.stack([x] * self.n_forward, dim=self.n_forward_dim)

        # Run the encoder (recognition model)
        multi_mu_z, multi_logvar_z = self.vmap_encoder(x_stack)


        # Sample from the approximate posterior
        z = self.reparameterize(multi_mu_z, multi_logvar_z)
        # print("Shape of multi z here is: ", str(z.shape))
        
        singh_entropy_term = entropy_singh_2003_up_to_constants(
            z,
            k=self.k_neighbor,
            dim_samples=self.n_forward_dim,
            dim_features=self.z_dim,
        )

        # Run the decoder (generative model)
        x_recon = self.vmap_decoder(z)
        reconstruction_term = self.decoder.log_likelihood(x_stack, x_recon, flatten_dim=2).mean(
            dim=self.n_forward_dim
        )

        log_p_z = self.decoder.log_likelihood_gaussian(z, torch.zeros(1, device = z.device), torch.zeros(1, device = z.device)).mean(
            dim=self.n_forward_dim
        )

        entropy_gap = (reconstruction_term + log_p_z) - singh_entropy_term   #shape [batch_size]

        mean = entropy_gap.mean()
        second_moment = (entropy_gap**2).mean()

        return mean, second_moment

    
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


    def create_batch_mog(self, means: torch.Tensor, logvar: torch.Tensor):
        """Given a set of (batch, k, d) of means and logvars, create a batch of Mixture of Gaussians
        distributions. Essentially, we get back a set of 'batch' distributions, each of which is a
        mixture of k Gaussians over a d-dimensional space.
        """
        batch, k, d = means.shape
        assert self.batch_dim == 0 and self.n_forward_dim == 1, "create_batch_mog does not support other dimension orders"
        assert logvar.shape == means.shape
        assert d == self.encoder.d

        # Create a batch of MultivariateNormal distributions. Start with a set of covariance matrices
        # stored in a (batch, k, d, d) tensor such that each (d,d) matrix is diagonal.
        components = MultivariateNormal(means, covariance_matrix=torch.diag_embed(torch.exp(logvar)))

        # Create a batch of Categorical distributions. Note from torch docs: the 'rightmost batch dimension', as in (batch, k) shape,
        # must be the dim containing the k mixture components.
        weights = torch.ones(batch, k) / k
        cat = Categorical(weights.to(device='cuda'))

        # Create a batch of MixtureSameFamily distributions. This now behaves as a (batch,) sized set
        # of distributions, each over a d-dim space.
        mog = MixtureSameFamily(cat, components)

        return mog
    

    def loss_mog(self, x, n_components_per_mixture, n_monte_carlo_elbo):
        batch_size = x.size(self.batch_dim)
        # mu_z, logvar_z = self.encoder(x)
        x_stack_multiple_forward = torch.stack([x] * n_components_per_mixture, dim=self.n_forward_dim)

        # Run the encoder (recognition model)
        multi_mu_z, multi_logvar_z = self.vmap_encoder(x_stack_multiple_forward)

        # Turn the (batch, k) set of gaussians into a (batch,) set of mixtures-of-gaussians
        m = self.create_batch_mog(multi_mu_z, multi_logvar_z)
        z = m.sample(sample_shape=(n_monte_carlo_elbo,)).permute((1, 0, 2))

        assert z.shape == (batch_size, n_monte_carlo_elbo, self.encoder.d), "Gotta fix sample dims"
        
        x_recon = self.vmap_decoder(z)
        x_stack_per_sample = torch.stack([x] * n_monte_carlo_elbo, dim=self.n_forward_dim)
        reconstruction_term = self.decoder.log_likelihood(x_stack_per_sample, x_recon, flatten_dim=2)

        prior_mu_z = torch.zeros((batch_size, n_monte_carlo_elbo, self.encoder.d), device=x.device, dtype=x.dtype)
        prior_logvar_z = torch.zeros((batch_size, n_monte_carlo_elbo, self.encoder.d), device=x.device, dtype=x.dtype)
        log_p_z_prior = self.decoder.log_likelihood_gaussian(z, mu_z=prior_mu_z, logvar_z=prior_logvar_z)
        log_m_z = torch.zeros_like(log_p_z_prior)
        for s in range(n_monte_carlo_elbo):
            log_m_z[: , s] = m.log_prob(z[:, s, :])
        kl_term = log_m_z - log_p_z_prior

        print(reconstruction_term.shape, kl_term.shape)
        assert reconstruction_term.shape == (batch_size, n_monte_carlo_elbo)
        assert kl_term.shape == (batch_size, n_monte_carlo_elbo)

        # First mean: monte carlo 1/S estimate of ELBO
        elbo = (reconstruction_term - kl_term).mean(dim=1)

        # Second reduction over the batch:
        loss = -elbo.mean()

        return loss, kl_term.mean(), reconstruction_term.mean(), x_recon[0]
    

    def on_train_epoch_start(self) -> None:
        self.log("lr", self.sched.get_last_lr()[0])

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, kl_term, reconstruction_term, entropy_term, _, multi_mu_z_mean, multi_logvar_z_mean, fim_term = self.loss(x)
        self.log("train_loss", loss)
        self.log("train_kl", kl_term)
        self.log("train_reconstruction", reconstruction_term)
        self.log("train_entropy", entropy_term)
        self.log("train_fischer_information_matrix", fim_term)

        return loss

    def validation_step(self, batch, batch_idx):
        temp_dir = "svae/output"
        x, _ = batch
        loss, kl_term, reconstruction_term, entropy_term, x_recon, multi_mu_z_mean, multi_logvar_z_mean, fim_term = self.loss(x)
        # try:
        #     loss_mog, _kl_term, __reconstruction_term, __x_recon = self.loss_mog(x, 30, 2)
        # except ValueError:
        #     loss_mog = torch.nan


        self.log("val_loss", loss)
        # self.log("validation_sanity_check_on_elbo", loss_mog)
        self.log("val_kl", kl_term)
        self.log("val_reconstruction", reconstruction_term)
        self.log("val_entropy", entropy_term)
        self.log("multi_mu_z_mean", multi_mu_z_mean)
        self.log("multi_logvar_z_mean", multi_logvar_z_mean)
        self.log("val_fischer_information_matrix", fim_term)
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

            # os.remove(input_image_path)
            # os.remove(recon_image_path)

        return loss
    
    def test_step(self, batch, batch_idx):
        temp_dir = "svae/output"
        x, _ = batch
        loss, kl_term, reconstruction_term, entropy_term, x_recon, multi_mu_z_mean, multi_logvar_z_mean, _ = self.loss(x)
        entropy_gap_mean, entropy_gap_moment2 = self.inference_entropy_gap_m_p(x)
        self.log("test_loss", loss)
        self.log("test_kl", kl_term)
        self.log("entropy gap--m_p", entropy_gap_mean)
        self.log("entropy gap second moment", entropy_gap_moment2)
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

        
