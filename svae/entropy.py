import numpy as np
import torch

LOGPI = np.log(np.pi)


def kth_nearest_neighbor_dist(x, k=1):
    """Given x, a (n x d) tensor of n points in d dimensions, calculate the nxn pairwise distances
    between rows of x, then get the kth smallest distance for each point in x.

    Returns a (n,) tensor of the kth nearest neighbor distance for each point in x.
    """
    xxT = torch.einsum("i...,j...->ij", x, x)
    sq_pair_dist = torch.diagonal(xxT, 0)[:, None] + torch.diagonal(xxT, 0)[None, :] - 2 * xxT
    return torch.kthvalue(sq_pair_dist, k + 1, dim=1).values ** 0.5


def entropy_singh_2003(
    x: torch.Tensor,
    k: int,
    dim_samples: int = 1,
    dim_features: int = 2,
    nearest_neighbor_epsilon: float = 1e-9,
):
    """Estimate the entropy of a set of samples along the 'dim'th dimension using the kth nearest
    neighbor method (Singh et al., 2003).
    """
    value_up_to_constants = entropy_singh_2003_up_to_constants(
        x, k, dim_samples, dim_features, nearest_neighbor_epsilon
    )
    k = torch.tensor(k, device=x.device)
    d = torch.tensor(x.size(dim_features), device=x.device)
    bias_correction = -torch.digamma(k) - torch.lgamma(d / 2 + 1)
    return value_up_to_constants + bias_correction


def entropy_singh_2003_up_to_constants(
    x: torch.Tensor,
    k: int,
    dim_samples: int = 1,
    dim_features: int = 2,
    nearest_neighbor_epsilon: float = 1e-9,
):
    """Estimate the entropy of a set of samples along the 'dim'th dimension using the kth nearest
    neighbor method (Singh et al., 2003).
    """
    if dim_samples < 0:
        raise ValueError("Negative indexing for samples dimension not supported")
    n = x.size(dim_samples)
    d = x.size(dim_features)
    if n <= 1:
        raise ValueError("Cannot compute entropy with only one sample.")
    if k >= n:
        raise ValueError("k must be less than the number of samples.")
    # TODO - cache the vmap compilation (if it isn't already... see torch docs)
    knn_dist = torch.vmap(kth_nearest_neighbor_dist, in_dims=(0, None), out_dims=0)(x, k)
    log_denominator = d * torch.log(torch.clip(knn_dist, min=nearest_neighbor_epsilon, max=None))
    # TODO - adjust this mean() call for negative dim_samples and remove the valueerror above
    return log_denominator.mean(dim=dim_samples)
