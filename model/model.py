"""
In this module, we accomplish the last model
and will combine the char CNN model and HighWay and LSTM model
to finish the final model
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from model import charcnn
from model import highway


class Model(nn.Module):
    """
    implement the Character-Aware Neural Language Models
    1. character embedding 2. Char CNN 3. Multiple layer HighWay 4. LSTM 5. softmax
    """
    def __init__(self, char_vocab_size,
                 word_vocab_size,
                 char_embedding_size=15,
                 num_highway_layers=2,
                 num_rnn_layers=2,
                 hidden_dimension=650,
                 max_word_length=65,
                 kernels=None,
                 kernel_features=None,
                 drop=0.5
                 ):
        super(Model, self).__init__()

        self.max_word_length = max_word_length
        self.kernels = kernels or [1, 2, 3, 4, 5, 6, 7]
        self.kernel_features = kernel_features or [50, 100, 150, 200, 200, 200, 200]
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_dimension
        self.drop = drop

        # character embedding
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_size)
        # clear embedding vector of the first symbol(symbol at position 0)
        zeros = torch.zeros([char_embedding_size])
        self.char_embedding.weight.data[0] = zeros

        # Construct the model
        self.convolution = charcnn.CharCNN(char_embedding_size, self.kernels, self.kernel_features)
        # the highway input dimension the number of filters
        cnn_size = sum(self.kernel_features)
        self.highway = highway.HighWay(cnn_size, num_highway_layers)
        self.rnn = nn.LSTM(cnn_size, hidden_dimension, self.num_rnn_layers, dropout=drop)

        # linear projection onto output (word) vocab
        self.projection = nn.Linear(hidden_dimension, word_vocab_size)

    def init_hidden(self, batch):
        """
        for the first time, we should initialize the hidden unit and memory unit
        h_0 = (num_layers , batch, hidden_size)
        c_0 = (num_layers , batch, hidden_size)
        """
        return (Variable(torch.zeros([self.num_rnn_layers, batch, self.hidden_size])),
                Variable(torch.zeros([self.num_rnn_layers, batch, self.hidden_size])))

    def forward(self, x, hidden):
        """
        :param x: a tensor shape <batch_size, num_unroll_steps, max_word_length>
        :param hidden: the rnn hidden layer
        :return:
        """
        batch, num_unroll_steps, max_word_length = x.shape
        # loop up table for embedding
        # the Embedding input shape is (N, W) N = mini-batch, W = indices
        x = x.view(-1, max_word_length)
        input_embedded = self.char_embedding(x)
        # input_embedded shape is <batch * num_unroll_steps, max_word_length, char_embedding_size>

        # for character- level cnn
        # the CNN model input shape (N, C_{in}, H, W)
        cnn_input = torch.transpose(input_embedded, 1, 2)
        cnn_input = torch.unsqueeze(cnn_input, 2)
        # print("The CNN input shape is", cnn_input.shape)
        cnn_output = self.convolution(cnn_input)
        # cnn_output shape is <batch * num_unroll_steps, cnn_size>

        # for HighWay Network
        # the HighWay Network doesn't change the the dimension of the input
        rnn_input = self.highway(cnn_output)

        # for RNN Network
        # the lstm input shape is (seq_len, batch, input_size)
        rnn_input = rnn_input.view(batch, num_unroll_steps, -1)
        rnn_input = torch.transpose(rnn_input, 0, 1)
        output, hidden = self.rnn(rnn_input, hidden)

        # the output shape is (seq_len, batch, hidden_size)
        output = output.view(-1, output.shape[-1])
        output = self.projection(output)
        # the output is <batch_size * seq_len, vocabulary_size>
        return output, hidden
