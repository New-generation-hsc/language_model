"""
In this module, we will implement the basic model for constructing Character level
language model;
which includes a CNN, a HighWay, a lstm
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNNCell(nn.Module):
    """
    using the char as input, the output through a highway net
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        :param in_channels: the character embedding size
        :param out_channels: the number of filters for a given width
        :param kernel_size: [1, width] where the height is fixed to 1
        For a standard convolution network, the height is 1, the width is a given width
        the channel is the embedding size or filters number
        """
        super(CharCNNCell, self).__init__()
        # convolution input size: <batch_size, channel_in, H_in, W_in>
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=[1, kernel_size], stride=1)
        # convolution output size: <batch_size, channel_out, H_out, W_out>
        # H_in and H_out equals to 1, the W_out = W_in - width + 1

    def forward(self, x, reduce_length):
        # print("CharCNN x input shape", x.shape)
        output = F.tanh(self.convolution(x))

        # we want to fina the max feature in one filter
        # the reduce_length = W_in - width + 1, so the result will only 1
        # the pool operation input size : <batch_size, channel_out, H_out, W_out>
        # which is <batch_size, channel_out, 1, reduce_length>
        output = F.max_pool2d(output, kernel_size=[1, reduce_length], stride=[1, 1])
        # the output dimension size is like : <batch_size, channel_size, 1, 1>
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 2)

        return output


class CharCNN(nn.Module):
    """
    implement a multiple features CNN
    concatenate every result of different filter size
    """
    def __init__(self, input_channel, kernels, kernel_features):
        """
        :param input_channel: the first layer input channel
        :param kernels: array of kernel size, in the paper this is [1, 2, 3, 4, 5, 6, 7]
        :param kernel_features: array if kernel feature size [50, 100, 150, 200, 200, 200, 200]
        """
        super(CharCNN, self).__init__()
        assert len(kernels) == len(kernel_features), "The kernel size must coincide with the kernel features"
        self.kernels = kernels
        self.kernel_features = kernel_features

        input_size = [input_channel] * len(self.kernel_features)
        self.layers = nn.ModuleList([CharCNNCell(in_channel, out_channel, kernel_size) for
                         (in_channel, out_channel, kernel_size) in zip(input_size, self.kernel_features, self.kernels)])

    def forward(self, x):
        """
        :param x: tensor with shape [batch, channel_in, H_in, W_in]
        where concretely the channel_in is the char embedding size, H_in is equals to 1
        W_in is equals to the max_word_length
        :return: tensor with concatenate features with shape <batch, channel_out>
        """
        outputs = []
        max_word_length = x.data.shape[-1]
        for (index, layer) in enumerate(self.layers):
            reduce_length = max_word_length - self.kernels[index] + 1
            lay_output = layer(x, reduce_length)
            outputs.append(lay_output)

        if len(self.kernels) > 1:
            output = torch.cat(outputs, 1) # Concatenates all the features together
        else:
            output = outputs[1]
        return output
