import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    def __init__(self, latent_dim, batch_size, hidden_dim, output_dim=1):
        super(Generator, self).__init__()

        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._batch_size = batch_size

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self._latent_dim, self._hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self._hidden_dim * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self._hidden_dim * 8, self._hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._hidden_dim * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self._hidden_dim * 4, self._hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._hidden_dim * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self._hidden_dim * 2, self._hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._hidden_dim),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self._hidden_dim, self._output_dim, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input=None):
        if input is None:
            input = torch.randn(self._batch_size, self._latent_dim).cuda()
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self._input_dim, self._hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self._hidden_dim, self._hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self._hidden_dim * 2, self._hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self._hidden_dim * 4, self._hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self._hidden_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
