"""NICE model
References:
    (1) Paper at arXiv - https://arxiv.org/pdf/1410.8516v6.pdf
    (2) ICLR 2015 lecture - https://www.youtube.com/watch?v=7hKul_tOfsI&t=1s
    (3) Glow: Generative Flow with Invertible 1×1 Convolutions - https://arxiv.org/pdf/1807.03039v2.pdf
"""
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

"""Additive coupling layer
"""


class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim: int, mid_dim: int, hidden: int, mask_config: bool) -> None:
        """Initialize an additive coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()

        # Making sure that the input/output dimension can be split by half
        assert in_out_dim % 2 == 0
        self._mask_config = mask_config

        # Defining the coupling model based on the number of requested hidden layers
        self.input_layer = nn.Sequential(nn.Linear(in_out_dim // 2, mid_dim), nn.ReLU())
        self.hidden_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU()) for _ in range(hidden)])
        self.output_layer = nn.Linear(mid_dim, in_out_dim // 2)

    def forward(self, x: torch.Tensor, log_det_J: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # Split the input tensor into to X1 and X2 (two halves)
        x1 = x[:, ::2] if (self._mask_config % 2) == 0 else x[:, 1::2]
        x2 = x[:, 1::2] if (self._mask_config % 2) == 0 else x[:, ::2]

        m_x2 = self.input_layer(x2)
        for i in range(len(self.hidden_layers)):
            m_x2 = self.hidden_layers[i](m_x2)
        m_x2 = self.output_layer(m_x2)
        if reverse:
            x1 = x1 - m_x2
        else:
            x1 = x1 + m_x2

        out = torch.zeros_like(x)
        if (self._mask_config % 2) == 0:
            out[:, ::2] = x1
            out[:, 1::2] = x2
        else:
            out[:, ::2] = x2
            out[:, 1::2] = x1
        return out, log_det_J


"""Affine coupling layer
"""


class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim: int, mid_dim: int, hidden: int, mask_config: bool) -> None:
        """Initialize an affine coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()

        # Making sure that the input/output dimension can be split by half
        assert in_out_dim % 2 == 0
        self._mask_config = mask_config

        # Defining the coupling model based on the number of requested hidden layers
        self.input_layer = nn.Sequential(nn.Linear(in_out_dim // 2, mid_dim), nn.ReLU())
        self.hidden_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU()) for _ in range(hidden)])
        self.output_layer = nn.Linear(mid_dim, in_out_dim)

    def forward(self, x: torch.Tensor, log_det_J: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # Split the input tensor into to X1 and X2 (two halves)
        x1 = x[:, ::2] if (self._mask_config % 2) == 0 else x[:, 1::2]
        x2 = x[:, 1::2] if (self._mask_config % 2) == 0 else x[:, ::2]

        m_x2 = self.input_layer(x2)
        for i in range(len(self.hidden_layers)):
            m_x2 = self.hidden_layers[i](m_x2)
        m_x2 = self.output_layer(m_x2)

        log_s, t = m_x2[:, ::2, ...], m_x2[:, 1::2, ...]
        s = torch.exp(log_s)
        if reverse:
            x1 = (x1 - t) / s
        else:
            x1 = (s * x1) + t

        log_det_J = torch.sum(torch.log(torch.abs(s)))

        out = torch.zeros_like(x)
        if (self._mask_config % 2) == 0:
            out[:, ::2] = x1
            out[:, 1::2] = x2
        else:
            out[:, ::2] = x2
            out[:, 1::2] = x1
        return out, log_det_J


"""Log-scaling layer.
"""


class Scaling(nn.Module):
    def __init__(self, dim: int) -> None:
        """Initialize a (log-)scaling layer.
        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        # Calculating the scale factor
        scale = torch.exp(self.scale) + self.eps
        if reverse:
            # h_reverse = h_prev / exp(s)
            x = x / scale
        else:
            # h = h_prev * exp(s)
            x = x * scale

        # Calculating the log-determinant of Jacobian
        log_det_J = torch.sum(self.scale) + self.eps
        return x, log_det_J


"""Standard logistic distribution.
"""


class StandardLogistic(torch.distributions.Distribution):
    def __init__(self):
        super(StandardLogistic, self).__init__()

    @staticmethod
    def log_prob(x: torch.Tensor) -> torch.Tensor:
        """Computes data log-likelihood.
        Args:
            x: input tensor.
        Returns:
            log-likelihood.
        """
        return -(F.softplus(x) + F.softplus(-x))

    @staticmethod
    def sample(size: typing.Tuple) -> torch.Tensor:
        """Samples from the distribution.
        Args:
            size: number of samples to generate.
        Returns:
            samples.
        """
        z = torch.distributions.Uniform(0., 1.).sample(size).cuda()
        return torch.log(z) - torch.log(1. - z)


"""NICE main model.
"""


class NICE(nn.Module):
    def __init__(self, prior: str, coupling: int, coupling_type: str, in_out_dim: int, mid_dim: int, hidden: int,
                 device: str):
        """Initialize a NICE.
        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = StandardLogistic()
        else:
            raise ValueError('Prior not implemented.')

        self.in_out_dim = in_out_dim
        self.num_coupling_layers = coupling
        self.coupling_type = coupling_type
        mask_config = 1

        if self.coupling_type.lower() == 'additive':
            self.coupling_layers = nn.ModuleList([AdditiveCoupling(in_out_dim=in_out_dim,
                                                                   mid_dim=mid_dim,
                                                                   hidden=hidden,
                                                                   mask_config=(mask_config + i) % 2) for i in
                                                  range(coupling)])
        elif self.coupling_type.lower() == 'affine':
            self.coupling_layers = nn.ModuleList([AffineCoupling(in_out_dim=in_out_dim,
                                                                 mid_dim=mid_dim,
                                                                 hidden=hidden,
                                                                 mask_config=(mask_config + i) % 2) for i in
                                                  range(coupling)])
        else:
            raise ValueError('coupling_type not implemented.')

        self.scaling = Scaling(in_out_dim)

    def f_inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(self.num_coupling_layers)):
            x, _ = self.coupling_layers[i](x, 0, reverse=True)
        return x

    def f(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        log_det_J = 0.

        # Forward pass over the coupling layers
        for i in range(self.num_coupling_layers):
            x, log_det_j_layer = self.coupling_layers[i](x, log_det_J)
            log_det_J += log_det_j_layer

        # Scaling the latent tensors and Jacobian
        x, log_det_j_scale = self.scaling(x)
        log_det_J += log_det_j_scale
        return x, log_det_J

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)

        #  Log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_det_J -= np.log(256) * self.in_out_dim

        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size: int) -> torch.Tensor:
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        x = self.f_inverse(z)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)
