# Modified: 25 May 2024
# Author: Akanksha Mishra
# Layer classes 
import torch
import torch.nn as nn
from torch.distributions import Normal
from tools import sigmoid, logit, gumbel_softmax, min_max, truncated_gaussian, gaussian_cdf



class SpikeNSlabLayer(nn.Module):
    """
    Layer with Spike and Slab prior and flag to select gumbel-softmax or truncated-gaussian
    """

    def __init__(self, input_dim, output_dim, rho_prior, device, rho0=-6., lambda0=0.99, flag=False):
        # initialize layers
        super(SpikeNSlabLayer, self).__init__()

        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # initialize mu, rho and theta parameters for layer's weights
        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-0.6, 0.6))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(rho0, rho0))

        # initialize mu, rho and theta parameters for layer's biases, theta = logit(phi)
        self.b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim).uniform_(rho0, rho0))

        # w_theta, w_sigma, b_theta, b_sigma are used for the mixing variable
        if flag:
            # This is adapted from STG
            # requires_grad = True; by default True
            self.w_theta = nn.Parameter(torch.rand(input_dim, output_dim))
            # self.w_sigma = nn.Parameter(torch.rand(input_dim, output_dim))
            self.b_theta = nn.Parameter(torch.rand(output_dim))
            # self.b_sigma = nn.Parameter(torch.rand(output_dim))
        else:
            self.w_theta = nn.Parameter(logit(torch.Tensor(input_dim, output_dim).uniform_(lambda0, lambda0)))
            self.b_theta = nn.Parameter(logit(torch.Tensor(output_dim).uniform_(lambda0, lambda0)))

        self.rho_prior = rho_prior
        self.device = device

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.gamma_w = None
        self.gamma_b = None

        self.w = None
        self.b = None

        # initialize log pdf of prior and vb distributions
        self.kl = 0

    def forward(self, X, temp, phi_prior, flag=False):
        """
        For one Monte Carlo sample

        :param X: [batch_size, input_dim]
        :return: output for one MC sample, size = [batch_size, output_dim]
        """
        # sample weights and biases
        # to keep variance term nonnegative
        # @akanksha: you can do the same for self.w_sigma, self.b_sigma
        sigma_w = torch.log(1 + torch.exp(self.w_rho))
        sigma_b = torch.log(1 + torch.exp(self.b_rho))
        sigma_prior = torch.log(1 + torch.exp(self.rho_prior))
        w_sigma_ve = 1
        b_sigma_ve = 1

        if flag:
            epsilon_wt = Normal(0, 1).sample(self.w_theta.shape)
            epsilon_bt = Normal(0, 1).sample(self.b_theta.shape)
            epsilon_wt = epsilon_wt.to(self.device)
            epsilon_bt = epsilon_bt.to(self.device)
            self.gamma_w = truncated_gaussian(self.w_theta, w_sigma_ve, epsilon_wt, hard=True)
            self.gamma_b = truncated_gaussian(self.b_theta, b_sigma_ve, epsilon_bt, hard=True)

            # for w_phi computation the orginal code doesn't use epsilon 
            w_phi = gaussian_cdf(self.w_theta / w_sigma_ve)
            b_phi = gaussian_cdf(self.b_theta / b_sigma_ve)
        else:
            u_w = torch.rand(self.w_theta.shape)
            u_b = torch.rand(self.b_theta.shape)
            u_w = u_w.to(self.device)
            u_b = u_b.to(self.device)
            self.gamma_w = gumbel_softmax(self.w_theta, u_w, temp, hard=True)
            self.gamma_b = gumbel_softmax(self.b_theta, u_b, temp, hard=True)

            w_phi = sigmoid(self.w_theta)
            b_phi = sigmoid(self.b_theta)

        # print("gamma_w={0} gamma_b={1} w_phi={0} b_phi={1}".format(self.gamma_w, self.gamma_b, w_phi, b_phi))
        # print(w_phi)
        # print("grad_fn attribute of the tensor gamma_w::", self.gamma_w.grad_fn)
        # print("is_leaf attribute of the tensor gamma_w::", self.gamma_w.is_leaf)
        # print("requires_grad attribute of the tensor gamma_w::", self.gamma_w.requires_grad)
        # Record KL at sampled weight and bias

        kl_w = w_phi * (torch.log(w_phi) - torch.log(phi_prior)) + \
            (1 - w_phi) * (torch.log(1 - w_phi) - torch.log(1 - phi_prior)) + \
            w_phi * (torch.log(sigma_prior) - torch.log(sigma_w) +
                     0.5 * (sigma_w ** 2 + self.w_mu ** 2) / sigma_prior ** 2 - 0.5)

        kl_b = b_phi * (torch.log(b_phi) - torch.log(phi_prior)) + \
            (1 - b_phi) * (torch.log(1 - b_phi) - torch.log(1 - phi_prior)) + \
            b_phi * (torch.log(sigma_prior) - torch.log(sigma_b) +
                     0.5 * (sigma_b ** 2 + self.b_mu ** 2) / sigma_prior ** 2 - 0.5)

        # Sums over all weights and biases
        self.kl = torch.sum(kl_w) + torch.sum(kl_b)

        # Sample weights and biases via reparametrization trick 

        epsilon_w = Normal(0, 1).sample(self.w_mu.shape)
        epsilon_b = Normal(0, 1).sample(self.b_mu.shape)
        epsilon_w = epsilon_w.to(self.device)
        epsilon_b = epsilon_b.to(self.device)

        self.w = self.gamma_w * (self.w_mu + sigma_w * epsilon_w)
        self.b = self.gamma_b * (self.b_mu + sigma_b * epsilon_b)
        output = torch.mm(X, self.w) + self.b.expand(X.size()[0], self.output_dim)

        return output

class SpikeNSlabLayer1(nn.Module):
    """
    Layer with Spike and Slab prior and flag to select gumbel-softmax or truncated-gaussian
    """

    def __init__(self, input_dim, output_dim, rho_prior, device, rho0=-6., lambda0=0.99, flag=False):
        # initialize layers
        super(SpikeNSlabLayer1, self).__init__()

        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # initialize mu, rho and theta parameters for layer's weights
        self.w_mu = nn.Parameter(torch.Tensor(input_dim).uniform_(-0.6, 0.6))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim).uniform_(rho0, rho0))

        # initialize mu, rho and theta parameters for layer's biases, theta = logit(phi)
        self.b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim).uniform_(rho0, rho0))

        self.w_theta = nn.Parameter(logit(torch.Tensor(input_dim).uniform_(lambda0, lambda0)))
        self.b_theta = nn.Parameter(logit(torch.Tensor(output_dim).uniform_(lambda0, lambda0)))

        self.rho_prior = rho_prior
        self.device = device

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.gamma_w = None
        self.gamma_b = None

        self.w = None
        self.b = None

        # initialize log pdf of prior and vb distributions
        self.kl = 0

    def forward(self, X, temp, phi_prior, flag=False):
        """
        For one Monte Carlo sample

        :param X: [batch_size, input_dim]
        :return: output for one MC sample, size = [batch_size, output_dim]
        """
        # sample weights and biases
        # to keep variance term nonnegative
        # @akanksha: you can do the same for self.w_sigma, self.b_sigma
        sigma_w = torch.log(1 + torch.exp(self.w_rho))
        sigma_b = torch.log(1 + torch.exp(self.b_rho))
        sigma_prior = torch.log(1 + torch.exp(self.rho_prior))

        u_w = torch.rand(self.w_theta.shape)
        u_b = torch.rand(self.b_theta.shape)
        u_w = u_w.to(self.device)
        u_b = u_b.to(self.device)
        self.gamma_w = gumbel_softmax(self.w_theta, u_w, temp, hard=True)
        self.gamma_b = gumbel_softmax(self.b_theta, u_b, temp, hard=True)

        w_phi = sigmoid(self.w_theta)
        b_phi = sigmoid(self.b_theta)

        # print("gamma_w={0} gamma_b={1} w_phi={0} b_phi={1}".format(self.gamma_w, self.gamma_b, w_phi, b_phi))
        # print(w_phi)
        # print("grad_fn attribute of the tensor gamma_w::", self.gamma_w.grad_fn)
        # print("is_leaf attribute of the tensor gamma_w::", self.gamma_w.is_leaf)
        # print("requires_grad attribute of the tensor gamma_w::", self.gamma_w.requires_grad)
        # Record KL at sampled weight and bias

        kl_w = w_phi * (torch.log(w_phi) - torch.log(phi_prior)) + \
            (1 - w_phi) * (torch.log(1 - w_phi) - torch.log(1 - phi_prior)) + \
            w_phi * (torch.log(sigma_prior) - torch.log(sigma_w) +
                     0.5 * (sigma_w ** 2 + self.w_mu ** 2) / sigma_prior ** 2 - 0.5)

        kl_b = b_phi * (torch.log(b_phi) - torch.log(phi_prior)) + \
            (1 - b_phi) * (torch.log(1 - b_phi) - torch.log(1 - phi_prior)) + \
            b_phi * (torch.log(sigma_prior) - torch.log(sigma_b) +
                     0.5 * (sigma_b ** 2 + self.b_mu ** 2) / sigma_prior ** 2 - 0.5)

        # Sums over all weights and biases
        self.kl = torch.sum(kl_w) + torch.sum(kl_b)

        # Sample weights and biases via reparametrization trick 

        epsilon_w = Normal(0, 1).sample(self.w_mu.shape)
        epsilon_b = Normal(0, 1).sample(self.b_mu.shape)
        epsilon_w = epsilon_w.to(self.device)
        epsilon_b = epsilon_b.to(self.device)

        self.w = self.gamma_w * (self.w_mu + sigma_w * epsilon_w)
        self.b = self.gamma_b * (self.b_mu + sigma_b * epsilon_b)
        
        self.w = self.w.reshape([X.size()[1],1])
        self.w = self.w.expand(-1, self.output_dim)
        
        output = torch.mm(X, self.w) + self.b.expand(X.size()[0], self.output_dim)

        return output


class NormalLayer(nn.Module):
    """
    Layer with Normal prior   
    """

    def __init__(self, input_dim, output_dim, rho_prior, device, rho0=-6.):
        # initialize layers
        super(NormalLayer, self).__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # initialize mu and rho parameters for layer's weights
        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-0.6, 0.6))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(rho0, rho0))
        # initialize mu and rho parameters for layer's biases, theta = logit(phi)
        self.b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim).uniform_(rho0, rho0))

        self.rho_prior = rho_prior
        self.device = device

        self.w = None
        self.b = None

        # initialize log pdf of prior and vb distributions
        self.kl = 0

    def forward(self, X):
        """
        For one Monte Carlo sample

        :param X: [batch_size, input_dim]
        :return: output for one MC sample, size = [batch_size, output_dim]
        """
        # sample weights and biases
        sigma_w = torch.log(1 + torch.exp(self.w_rho)) # to keep variance term nonnegative 
        sigma_b = torch.log(1 + torch.exp(self.b_rho))
        sigma_prior = torch.log(1 + torch.exp(self.rho_prior))

        epsilon_w = Normal(0, 1).sample(self.w_mu.shape)
        epsilon_b = Normal(0, 1).sample(self.b_mu.shape)
        epsilon_w = epsilon_w.to(self.device)
        epsilon_b = epsilon_b.to(self.device)

        self.w = (self.w_mu + sigma_w * epsilon_w) # reparamterization trick 
        self.b = (self.b_mu + sigma_b * epsilon_b)
        output = torch.mm(X, self.w) + self.b.expand(X.size()[0], self.output_dim)

        kl_w = (torch.log(sigma_prior) - torch.log(sigma_w) +
                     0.5 * (sigma_w ** 2 + self.w_mu ** 2) / sigma_prior ** 2 - 0.5)

        kl_b = (torch.log(sigma_prior) - torch.log(sigma_b) +
                     0.5 * (sigma_b ** 2 + self.b_mu ** 2) / sigma_prior ** 2 - 0.5)

        # Sums over all weights and biases 
        self.kl = torch.sum(kl_w) + torch.sum(kl_b) 

        return output
