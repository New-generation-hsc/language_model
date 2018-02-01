"""
In this module, we implement the HighWay Network
y = H(x, W_h) * T(x, W_t) + x * (1 - T(x, W_t))
reference : https://github.com/c0nn3r/pytorch_highway_networks
"""

import torch.nn as nn
import torch.nn.functional as F


class HighWayCell(nn.Module):
    def __init__(self, size, gate_bias=-2, activation_function=F.relu, gate_activation=F.sigmoid):
        """
        :param size: the input size and output size
        :param gate_bias: `the transform gate` initialized bias b_t = -2
        :param activation_function: nonlinear activation g
        :param gate_activation: gate activation function t
        """
        super(HighWayCell, self).__init__()
        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.affine = nn.Linear(size, size)
        self.gate_linear = nn.Linear(size, size)

        # initialize the the transform gate bias
        self.gate_linear.bias.data.fill_(gate_bias)

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        applies t ⨀ g + (1 - t) ⨀ y transformation
        t is non-linear transformation, σ(x) is affine transformation with sigmoid non-linear
        and ⨀ is element-wise multiplication
        """
        g = self.activation_function(self.affine(x))
        t = self.gate_activation(self.gate_linear(x))

        z = t * g + (1. - t) * x
        return z


class HighWay(nn.Module):
    """
    implement a multiple layers highway network
    """
    def __init__(self, size, num_layers):
        super(HighWay, self).__init__()

        self.layers = nn.ModuleList([HighWayCell(size) for _ in range(num_layers)])

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]
        """
        for lay in self.layers:
            x = lay(x)
        return x
