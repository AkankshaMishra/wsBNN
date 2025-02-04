# Modified: 25 May 2024
# Author: Akanksha Mishra
import torch
import torch.nn as nn
import numpy as np
from layer import SpikeNSlabLayer, NormalLayer, SpikeNSlabLayer1
# from layer_with_laplace import SpikeNLaplaceSlabLayer, NormalLayer
from tools import cross_entropy

class SparseBNNClassification(nn.Module):

    """
    Sparse Bayesian Neural Network (BNN)
    """

    def __init__(self, data_dim, device, target_dim=3, hidden_dim=[16,8,4], sigma_noise=1.):

        # initialize the network using the MLP layer
        super(SparseBNNClassification, self).__init__()
        self.rho_prior = torch.Tensor([np.log(np.exp(1.3) - 1)]).to(device)
        self.device = device

        self.l1 = SpikeNSlabLayer(data_dim, hidden_dim[0], self.rho_prior, self.device, flag=False)
        self.l1_relu = nn.ReLU()
        self.l2 = SpikeNSlabLayer(hidden_dim[0], hidden_dim[1], self.rho_prior, self.device, flag=False)
        self.l2_relu = nn.ReLU()
        self.l3 = SpikeNSlabLayer(hidden_dim[1], hidden_dim[2], self.rho_prior, self.device, flag=False)
        self.l3_relu = nn.ReLU()
        self.l4 = SpikeNSlabLayer(hidden_dim[2], target_dim, self.rho_prior, self.device, flag=False)
        self.l4_softmax = nn.Softmax(dim = -1)

        # self.layers = nn.ModuleList()
        # self.relus = nn.ModuleList()
        # self.softmax = nn.ModuleList()

        # for i in range(min(target_dim, len(hidden_dim))):
        #     layer = SpikeNSlabLayer(hidden_dim[i-1] if i > 0 else data_dim, hidden_dim[i] if i < target_dim else target_dim, self.rho_prior, self.device, flag=False)
        #     setattr(self, f'l{i}', layer)
        #     self.layers.append(layer)
        #     if i == target_dim:
        #         softmax = nn.Softmax(dim = -1)
        #         setattr(self, f'l{i}_softmax', softmax)
        #         self.softmax.append(softmax)
        #     else:
        #         relu = nn.ReLU()
        #         setattr(self, f'l{i}_relu', relu)
        #         self.relus.append(relu)

        self.target_dim = target_dim
        # self.log_sigma_noise = torch.log(torch.Tensor([sigma_noise])).to(device)

    def forward(self, X, temp, phi_prior, flag=False):
        """
            output of the BNN for one Monte Carlo sample

            :param X: [batch_size, data_dim]
            :return: [batch_size, target_dim]
        """
        # for i in range(target_dim):
        output = self.l1_relu(self.l1(X, temp, phi_prior, flag=False))
        output = self.l2_relu(self.l2(output, temp, phi_prior, flag=False))
        output = self.l3_relu(self.l3(output, temp, phi_prior, flag=False))
        output = self.l4_softmax(self.l4(output, temp, phi_prior, flag=False))
        return output

    def kl(self):
        """
        Calculate the kl divergence over all four BNN layers 
        """
        kl = self.l1.kl + self.l2.kl + self.l3.kl + self.l4.kl
        return kl

    def sample_elbo(self, X, y, n_samples, temp, phi_prior, num_batches):
        """
            calculate the loss function - negative elbo

            :param X: [batch_size, data_dim]
            :param y: [batch_size]
            :param n_samples: number of MC samples
            :param temp: temperature
            :return:
        """

        # initialization
        outputs = torch.zeros(n_samples, y.shape[0], self.target_dim).to(self.device)
        kls = 0.
        log_likes = 0.

        # make predictions and calculate prior, posterior, and likelihood for a given number of MC samples
        for i in range(n_samples):  # ith mc sample
            # make predictions, (batch_size, target_dim)
            outputs[i] = self(X, temp, phi_prior)
            sample_kl = self.kl()  # get kl (a number)
            kls += sample_kl
            log_likes += torch.sum(cross_entropy(y, outputs[i].squeeze()))

        # calculate MC estimates of log prior, vb and likelihood
        kl_MC = kls / float(n_samples)
        # calculate negative loglikelihood
        nll_MC = - log_likes / float(n_samples)
        # calculate negative elbo
        loss = kl_MC / num_batches + nll_MC
        # print("Loss: {}, First: {}, Second: {}".format(loss, kl_MC / num_batches, nll_MC))
        return loss, torch.argmax(outputs, dim=-1).squeeze()



class FeatureSelectionBNNClassification(nn.Module):
    """
    Feature Selection Bayesian Neural Network (BNN)
    """

    def __init__(self, data_dim, device, target_dim=3, hidden_dim=[16,8,4], sigma_noise=1.):

        # initialize the network using the MLP layer
        super(FeatureSelectionBNNClassification, self).__init__()
        self.rho_prior = torch.Tensor([np.log(np.exp(1.3) - 1)]).to(device)
        self.device = device

        self.l1 = SpikeNSlabLayer1(data_dim, hidden_dim[0], self.rho_prior, self.device, flag=False)
        self.l1_relu = nn.ReLU()
        self.l2 = NormalLayer(hidden_dim[0], hidden_dim[1], self.rho_prior, self.device)
        self.l2_relu = nn.ReLU()
        self.l3 = NormalLayer(hidden_dim[1], hidden_dim[2], self.rho_prior, self.device)
        self.l3_relu = nn.ReLU()
        self.l4 = NormalLayer(hidden_dim[2], target_dim, self.rho_prior, self.device)
        self.l4_softmax = nn.Softmax(dim=-1)

        self.target_dim = target_dim
        self.log_sigma_noise = torch.log(torch.Tensor([sigma_noise])).to(device)

    def forward(self, X, temp, phi_prior, flag=False):
        """
            output of the BNN for one Monte Carlo sample

            :param X: [batch_size, data_dim]
            :return: [batch_size, target_dim]
        """
        output = self.l1_relu(self.l1(X, temp, phi_prior, flag=False))
        output = self.l2_relu(self.l2(output))
        output = self.l3_relu(self.l3(output))
        output = self.l4_softmax(self.l4(output))
        return output

    def kl(self):
        """
        Calculate the kl divergence over all four BNN layers 
        """
        kl = self.l1.kl + self.l2.kl + self.l3.kl + self.l4.kl
        return kl

    def sample_elbo(self, X, y, n_samples, temp, phi_prior, num_batches):
        """
            calculate the loss function - negative elbo

            :param X: [batch_size, data_dim]
            :param y: [batch_size]
            :param n_samples: number of MC samples
            :param temp: temperature
            :return:
        """

        # initialization
        outputs = torch.zeros(n_samples, y.shape[0], self.target_dim).to(self.device)
        kls = 0.
        log_likes = 0.

        # make predictions and calculate prior, posterior, and likelihood for a given number of MC samples
        for i in range(n_samples):  # ith mc sample
            # make predictions, (batch_size, target_dim)
            outputs[i] = self(X, temp, phi_prior)
            sample_kl = self.kl()  # get kl (a number)
            kls += sample_kl
            log_likes += torch.sum(cross_entropy(y, outputs[i].squeeze()))

        # calculate MC estimates of log prior, vb and likelihood
        kl_MC = kls / float(n_samples)
        # calculate negative loglikelihood
        nll_MC = - log_likes / float(n_samples)

        # calculate negative elbo
        loss = kl_MC / num_batches + nll_MC
        # print("Loss: {}, First: {}, Second: {}".format(loss, kl_MC / num_batches, nll_MC))
        return loss, torch.argmax(outputs, dim=-1).squeeze()
