import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from hw5_utils import BASE_URL, download, GANDataset

class DNet(nn.Module):
    """This is discriminator network."""

    def __init__(self):
        super(DNet, self).__init__()
        
        # TODO: implement layers here
        self.D = nn.Sequential(
            nn.Conv2d(1, 2, 3, 1, 1),       # in_ch, out_ch, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # kernel_size, stride, padding
            nn.Conv2d(2, 4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 8, 3, 1, 0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(200, 1),
        )

        self._weight_init()

    def _weight_init(self):
        # TODO: implement weight initialization here
        for _, layer in enumerate(self.children()):
            print(layer)
            for idx in range(len(layer)):
                try:
                    nn.init.kaiming_normal_(layer[idx].weight.data)
                except:
                    continue
                try:
                    nn.init.constant_(layer[idx].bias, 0)
                except:
                    continue

    def forward(self, x):
        # TODO: complete forward function
        y = self.D(x)
        return y


class GNet(nn.Module):
    """This is generator network."""

    def __init__(self, zdim):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        super(GNet, self).__init__()

        # TODO: implement layers here
        self.G = nn.Sequential(
            nn.Linear(64, 1568),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (32, 7, 7)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

        self._weight_init()

    def _weight_init(self):
        # TODO: implement weight initialization here
        for _, layer in enumerate(self.children()):
            print(layer)
            for idx in range(len(layer)):
                try:
                    nn.init.kaiming_normal_(layer[idx].weight.data)
                except:
                    continue
                try:
                    nn.init.constant_(layer[idx].bias, 0)
                except:
                    continue

    def forward(self, z):
        """
        Parameters
        ----------
            z: latent variables used to generate images.
        """
        # TODO: complete forward function
        y = self.G(z)
        return y


class GAN:
    def __init__(self, zdim=64):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        torch.manual_seed(2)
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._zdim = zdim
        self.disc = DNet().to(self._dev)
        self.gen = GNet(self._zdim).to(self._dev)

    def _get_loss_d(self, batch_size, batch_data, z):
        """This function computes loss for discriminator.

        Parameters
        ----------
            batch_size: #data per batch.
            batch_data: data from dataset.
            z: random latent variable.
        """
        # TODO: implement discriminator's loss function

        ## generate fake data batch
        fake_images = self.gen(z).to(self._dev)

        ## calculate discriminator output
        real_score = self.disc(batch_data)
        fake_score = self.disc(fake_images)

        ## compute discriminator loss
        loss_real = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score), reduction="mean")
        loss_fake = F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score), reduction="mean")
        loss = loss_real + loss_fake

        return loss

    def _get_loss_g(self, batch_size, z):
        """This function computes loss for generator.

        Parameters
        ----------
            batch_size: #data per batch.
            z: random latent variable.
        """
        # TODO: implement generator's loss function

        ## generate fake data batch
        fake_images = self.gen(z).to(self._dev)

        ## calculate discriminator output
        fake_score = self.disc(fake_images)

        ## compute generator loss
        loss = F.binary_cross_entropy_with_logits(fake_score, torch.ones_like(fake_score), reduction="mean")

        return loss

    def train(self, iter_d=1, iter_g=1, n_epochs=100, batch_size=256, lr=0.0002):

        # first download
        f_name = "train-images-idx3-ubyte.gz"
        download(BASE_URL + f_name, f_name)

        print("Processing dataset ...")
        train_data = GANDataset(
            f"./data/{f_name}",
            self._dev,
            transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]),
        )
        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        dopt = optim.Adam(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self._zdim, device=self._dev) - 1

                if batch_idx == 0 and epoch == 0:
                    plt.imshow(data[0, 0, :, :].detach().cpu().numpy())
                    plt.savefig("goal.pdf")

                if batch_idx == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        tmpimg = self.gen(z)[0:64, :, :, :].detach().cpu()
                    save_image(
                        tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True
                    )

                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self._get_loss_d(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self._get_loss_g(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()

            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")


if __name__ == "__main__":
    gan = GAN()
    gan.train()

