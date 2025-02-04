# utility functions
import torch
import numpy as np
import math


def sigmoid(z):
    return 1. / (1 + torch.exp(-z))


def logit(z):
    return torch.log(z/(1.-z))


def gumbel_softmax(logits, U, temperature, hard=False, eps=1e-20):
    """
        gumbel-softmax-approximation
    """
    z = logits + torch.log(U + eps) - torch.log(1 - U + eps)
    y = 1 / (1 + torch.exp(- z / temperature))

    # @akanksha: please understand the following steps and explain here
    if not hard: return y
    y_hard = (y > 0.5).double()

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y #If we apply back propogation on y_hard then it should give error

    return y_hard


def log_gaussian(x, mu, sigma):
    """
        log pdf of one-dimensional gaussian
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma)
    return float(-0.5 * np.log(2 * np.pi)) - torch.log(sigma) - (x - mu)**2 / (2 * sigma**2)


def cross_entropy(y, outputs):
    m = y.shape[0]
    loss = torch.log(outputs[range(m), y])
    return torch.sum(loss)


def truncated_gaussian(theta, sigma, epsilon, hard=False):
    """
        truncated-gaussian-approximation
    """
    z = theta + sigma * epsilon
    y = min_max(z)
    if not hard: return y
    y_hard = (y > 0).double()
    
    y_hard = (y_hard - y).detach() + y
    return y_hard


def min_max(x):
    return torch.clamp(x, 0.0, 1.0)

def gaussian_cdf(x):
    ''' Gaussian CDF. '''
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
