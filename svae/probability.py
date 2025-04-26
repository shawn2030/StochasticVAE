import torch


def reparameterization_trick(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_q_prior(mu_q, logvar_q, dim=-1):
    """Calculate KL divergence between a diagonal gaussian and a standard normal."""
    return -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=dim)


def log_det_fisher(mu_q, logvar_q, dim=-1):
    d = mu_q.size(dim)
    return -2 * torch.sum(logvar_q, dim=dim) + torch.tensor(d) * torch.log(torch.tensor(2))


def log_prob_diagonal_gaussian(x, mu, logvar, dim=-1):
    return -0.5 * (logvar + (x - mu) ** 2 / logvar.exp()).sum(dim=dim)
