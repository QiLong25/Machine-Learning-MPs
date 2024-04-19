import hw4_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt

class VAE(torch.nn.Module):
    def __init__(self, lam,lrate,latent_dim,loss_fn):
        """
        Initialize the layers of your neural network

        @param lam: Hyperparameter to scale KL-divergence penalty
        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param latent_dim: The dimension of the latent space

        The network should have the following architecture (in terms of hidden units):
        Encoder Network:
        2 -> 50 -> ReLU -> 50 -> ReLU -> 50 -> ReLU -> (6,6) (mu_layer,logstd2_layer)

        Decoder Network:
        6 -> 50 -> ReLU -> 50 -> ReLU -> 2 -> Sigmoid

        See set_parameters() function for the exact shapes for each weight
        """
        super(VAE, self).__init__()

        self.lrate = lrate
        self.lam = lam
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim

        ## encoder
        self.Encoder = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(50, self.latent_dim)
        self.logstd2_layer = nn.Linear(50, self.latent_dim)

        ## decoder
        self.Decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.Sigmoid(),
        )

        self.optimzer = torch.optim.Adam([{'params': self.Encoder.parameters()},
                                          {'params': self.mu_layer.parameters()},
                                          {'params': self.logstd2_layer.parameters()},
                                          {'params': self.Decoder.parameters()}],
                                         lr=self.lrate)

    def set_parameters(self, We1,be1, We2, be2, We3, be3, Wmu, bmu, Wstd, bstd, Wd1, bd1, Wd2, bd2, Wd3, bd3):
        """ Set the parameters of your network

        # Encoder weights:
        @param We1: an (50,2) torch tensor
        @param be1: an (50,) torch tensor
        @param We2: an (50,50) torch tensor
        @param be2: an (50,) torch tensor
        @param We3: an (50,50) torch tensor
        @param be3: an (50,) torch tensor
        @param Wmu: an (6,50) torch tensor
        @param bmu: an (6,) torch tensor
        @param Wstd: an (6,50) torch tensor
        @param bstd: an (6,) torch tensor

        # Decoder weights:
        @param Wd1: an (50,6) torch tensor
        @param bd1: an (50,) torch tensor
        @param Wd2: an (50,50) torch tensor
        @param bd2: an (50,) torch tensor
        @param Wd3: an (2,50) torch tensor
        @param bd3: an (2,) torch tensor

        """
        with torch.no_grad():
            ## encoder
            for name, param in self.Encoder.named_parameters():
                if name == "0.weight":
                    param.data = We1
                elif name == "0.bias":
                    param.data = be1
                elif name == "2.weight":
                    param.data = We2
                elif name == "2.bias":
                    param.data = be2
                elif name == "4.weight":
                    param.data = We3
                elif name == "4.bias":
                    param.data = be3

            ## self.mu_layer
            self.mu_layer.weight.data = Wmu
            self.mu_layer.bias.data = bmu

            ## self.mu_layer
            self.logstd2_layer.weight.data = Wstd
            self.logstd2_layer.bias.data = bstd

            ## decoder
            for name, param in self.Decoder.named_parameters():
                if name == "0.weight":
                    param.data = Wd1
                elif name == "0.bias":
                    param.data = bd1
                elif name == "2.weight":
                    param.data = Wd2
                elif name == "2.bias":
                    param.data = bd2
                elif name == "4.weight":
                    param.data = Wd3
                elif name == "4.bias":
                    param.data = bd3

    def forward(self, x):
        """ A forward pass of your autoencoder

        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return y: an (N, 50) torch tensor of output from the encoder network
        @return mean: an (N,latent_dim) torch tensor of output mu layer
        @return stddev_p: an (N,latent_dim) torch tensor of output stddev layer
        @return z: an (N,latent_dim) torch tensor sampled from N(mean,exp(stddev_p/2)
        @return xhat: an (N,D) torch tensor of outputs from f_dec(z)
        """
        y = self.Encoder(x)
        # print("y:", y.shape)
        mean = self.mu_layer(y)
        # print("mean:", self.mean.shape)
        stddev_p = self.logstd2_layer(y)
        # print("stddev_p:", self.stddev_p.shape)
        z = torch.randn_like(mean) * torch.exp(stddev_p / 2) + mean
        # print("z:", z.shape)
        x_hat = self.Decoder(z)
        # print("xhat:", self.xhat.shape)
        return y, mean, stddev_p, z, x_hat

    def step(self, x, mean, stddev_p, x_hat):
        """
        Performs one gradient step through a batch of data x
        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return L_rec: float containing the reconstruction loss at this time step
        @return L_kl: kl divergence penalty at this time step
        @return L: total loss at this time step
        """

        self.optimzer.zero_grad()

        ## compute loss
        L_rec = self.loss_fn(x, x_hat)
        L_kr_dim = - (self.latent_dim + torch.sum(stddev_p - mean**2 - torch.exp(stddev_p), dim=1)) / 2
        # print("L_kr_dim:", L_kr_dim.shape)
        L_kr = torch.sum(L_kr_dim) / x.shape[0]
        L = L_rec + self.lam * L_kr

        L.backward()
        self.optimzer.step()

        return L_rec, L_kr, L

def fit(net,X,n_iter):
    """ Fit a VAE.  Use the full batch size.
    @param net: the VAE
    @param X: an (N, D) torch tensor
    @param n_iter: int, the number of iterations of training

    # return all of these from left to right:

    @return losses_rec: Array of reconstruction losses at the beginning and after each iteration. Ensure len(losses_rec) == n_iter
    @return losses_kl: Array of KL loss penalties at the beginning and after each iteration. Ensure len(losses_kl) == n_iter
    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return Xhat: an (N,D) NumPy array of approximations to X
    @return gen_samples: an (N,D) NumPy array of N samples generated by the VAE
    """

    losses_rec = np.zeros(n_iter)
    losses_kl = np.zeros(n_iter)
    losses = np.zeros(n_iter)

    ## iteration
    for epoch in range(n_iter):
        y, mean, stddev_p, z, x_hat = net.forward(X)
        L_rec, L_kr, L = net.step(X, mean, stddev_p, x_hat)
        losses_rec[epoch] = L_rec
        losses_kl[epoch] = L_kr
        losses[epoch] = L

    ## generate samples
    Z = torch.randn(X.shape[0], net.latent_dim) * torch.exp(stddev_p / 2) + mean
    gen_samples = net.Decoder(Z)

    return losses_rec, losses_kl, losses, x_hat, gen_samples


# temp = VAE(1, 1, 1, 1)
# # print(temp.mu_layer.weight.data)
# # temp.mu_layer.weight.data = torch.zeros_like(temp.mu_layer.weight.data)
# # print(temp.mu_layer.weight.data)
# for name, param in temp.Decoder.named_parameters():
#     print(name, param.data.shape)

# ### Generate Output ###
# KL_scale = 5e-5
# lr = 0.01
# lat_dim = 6
# loss_fn = nn.MSELoss()
# epochs = 8000
# model = VAE(KL_scale, lr, lat_dim, loss_fn)
# X = hw4_utils.generate_data()
# losses_rec, losses_kl, losses, x_hat, gen_samples = fit(model, X, epochs)
#
# # print(X)
# # print(gen_samples)
#
# ## save model
# torch.save(model.cpu().state_dict(), "vae.pb")

# ## plot training process
# x = range(len(losses))

# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.plot(x, losses_rec, marker='o', markersize=3)
# plt.plot(x, losses_kl, marker='o', markersize=3)
# plt.plot(x, losses, marker='o', markersize=3)
# plt.legend(["Reconstruction", "KL-divergence", "Total-Loss"])
# plt.show()

# # plot training curve
# plt.plot(x, losses, marker='o', markersize=3)
# plt.show()
#
# # plot X and X_hat
# plt.scatter(X.detach().numpy()[:, 0], X.detach().numpy()[:, 1], c='blue', marker='o')
# plt.scatter(x_hat.detach().numpy()[:, 0], x_hat.detach().numpy()[:, 1], c='red', marker='o')
# plt.show()
#
# # plot X, X_hat and gen_samples
# plt.scatter(X.detach().numpy()[:, 0], X.detach().numpy()[:, 1], c='blue', marker='o')
# plt.scatter(x_hat.detach().numpy()[:, 0], x_hat.detach().numpy()[:, 1], c='red', marker='o')
# plt.scatter(gen_samples.detach().numpy()[:, 0], gen_samples.detach().numpy()[:, 1], c='green', marker='o')
# plt.show()
#
# # test case
# print(losses_rec[-1])
# print(losses_kl[-1])
