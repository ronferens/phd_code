"""NICE model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


class Model(nn.Module):
    def __init__(self, latent_dim: int, device: str):
        """
        Initialize a VAE
        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        # -----------------------------
        # Encoder
        # -----------------------------
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.log_var = nn.Linear(64 * 7 * 7, latent_dim)

        # -----------------------------
        # Decoder
        # -----------------------------
        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )

    def sample(self, sample_size: int, mu: float = None, log_var=None) -> typing.List:
        """
        Sampled images from the model
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param log_var: z log-STD, None for prior (init with zeros)
        :return:
        """
        if mu is None:
            mu = torch.zeros((sample_size, self.latent_dim)).to(self.device)
        if log_var is None:
            log_var = torch.zeros((sample_size, self.latent_dim)).to(self.device)

        # Making sure the model is not trained while sampling and no gradients are accumulated
        samples = []
        with torch.no_grad():
            z = self.z_sample(mu=mu, log_var=log_var)
            output = self.decoder(self.upsample(z).view(-1, 64, 7, 7))
            for i in range(sample_size):
                samples.append(output[i].squeeze().data.cpu().numpy())

        return samples

    def z_sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Applying the reparameterization trick to generate Z
        :param mu: Z mean
        :param log_var: z log-STD
        :return: samples of Z
        """
        # Getting the STD from log−variance input
        std = torch.exp(0.5 * log_var)

        # Sampling epsilon from Normal (0, 1)
        eps = torch.rand_like(std).to(self.device)

        # Applying the reparameterization trick for Gaussian
        z = mu + eps * std
        return z

    @staticmethod
    def loss(x: torch.Tensor, recon: torch.Tensor, mu: float, log_var: float):
        """
        Calculating the model's loss composed of BCE and KL elements
        :param x: input data
        :param recon: reconstructed data
        :param mu: Z mean
        :param log_var: Z log-STD
        :return:
        """
        # Calculating the binary cross entropy loss
        bce_loss = F.binary_cross_entropy(recon, x, reduction='sum')

        # Calculating the KL Divergence loss
        kl_loss = torch.sum(1 + log_var - torch.exp(log_var) - torch.pow(mu, 2)) / 2.0

        # Calculating the ELBO(p, q, x)
        elbo = bce_loss - kl_loss
        return elbo

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """"
        Running the model's forward pass
        :param x: input data
        """
        # Encoder forward pass
        x = self.encoder(x)
        x = x.reshape(-1, 64 * 7 * 7)

        # Generating the mean and variance values
        mu = self.mu(x)
        log_var = self.log_var(x)

        # Decoder forward pass
        z = self.z_sample(mu=mu, log_var=log_var)
        output = self.decoder(self.upsample(z).reshape(-1, 64, 7, 7))
        return output, mu, log_var
