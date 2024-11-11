import torch
import numpy as np
from training_config import EPSILON


LOGPI = np.log(np.pi)


def kth_nearest_neighbor_dist(x, k=1):
    """Given x, a (n x d) tensor of n points in d dimensions, calculate the nxn pairwise distances
    between rows of x, then get the kth smallest distance for each point in x.

    Returns a (n,) tensor of the kth nearest neighbor distance for each point in x.
    """
    xxT = torch.einsum("i...,j...->ij", x, x)
    sq_pair_dist = torch.diagonal(xxT, 0)[:, None] + torch.diagonal(xxT, 0)[None, :] - 2 * xxT
    return torch.kthvalue(sq_pair_dist, k + 1, dim=1).values ** 0.5


def entropy_singh_2003(x: torch.Tensor, k: int, dim_samples: int = -2, dim_features: int = -1):
    """Estimate the entropy of a set of samples along the 'dim'th dimension using the kth nearest
    neighbor method (Singh et al., 2003).
    """
    n = x.size(dim_samples)
    d = x.size(dim_features)
    if n <= 1:
        raise ValueError("Cannot compute entropy with only one sample.")
    if k >= n:
        raise ValueError("k must be less than the number of samples.")
    # TODO - make n, k, and d tensors and fix device errors / any cpu->gpu bottlenecks
    # TODO - cache the vmap compilation (if it isn't already... see torch docs)
    knn_dist = torch.vmap(kth_nearest_neighbor_dist, in_dims=(1, None), out_dims=0)(x, k)
    log_numerator = torch.log(k) + torch.lgamma(torch.tensor([d / 2 + 1]))
    log_denominator = torch.log(n) + d / 2 * LOGPI + d * torch.log(knn_dist)
    bias_correction = torch.log(k) - torch.digamma(k)
    terms = log_numerator - log_denominator
    return -terms.mean() + bias_correction


def entropy_singh_2003_up_to_constants(
    x: torch.Tensor, k: int, dim_samples: int = -2, dim_features: int = -1
):
    """Estimate the entropy of a set of samples along the 'dim'th dimension using the kth nearest
    neighbor method (Singh et al., 2003).
    """
    n = x.size(dim_samples)
    d = x.size(dim_features)
    if n <= 1:
        raise ValueError("Cannot compute entropy with only one sample.")
    if k >= n:
        raise ValueError("k must be less than the number of samples.")
    # TODO - cache the vmap compilation (if it isn't already... see torch docs)
    # TODO - the '1' is a magic number meaning 'dim_batch'. Make this more explicit/an argument/something.
    knn_dist = torch.vmap(kth_nearest_neighbor_dist, in_dims=(1, None), out_dims=1)(x, k)
    log_denominator = d * torch.log(torch.clip(knn_dist, min=EPSILON, max=None))
    return log_denominator.mean(dim=dim_samples)
