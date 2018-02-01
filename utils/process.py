"""In this module, we load the `train.txt`, `valid.txt`, `test.txt`.
Also, we will convert the word in the text file into structure vocabulary.
for every word, we store the word index in the vocabulary.
Besides, we can custom define the word transformation, like adding some prefix and postfix"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.utils.data import Dataset
import collections
import numpy as np
import torch
import os

global_constant = {
    'BLANK': ' ', 'START' : '{', 'END': '}', 'UNK': '|', 'EOS': '+'
}


class Vocabulary(object):
    """
    A convenient data structure to store the index of every word in the vocabulary.
    """
    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        """
        if the token is in the vocabulary, ignore it, else add it into vocabulary
        :param token: current word in the corpus
        :return: the index of current word
        """
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        """
        :return: the vocabulary size
        """
        return len(self._token2index)

    def token(self, index):
        """
        get the token vocabulary with index
        :param index:
        :return: word, token --> str
        """
        return self._index2token[index]

    def __getitem__(self, token):
        """
        get the index from vocabulary with token
        :param token:
        :return: the index
        """
        assert token in self._token2index, "KeyError: unexpected token {0}".format(token)
        return self.get(token)

    def get(self, token, default=None):
        """get the index for token in the vocabulary
        default: None
        """
        return self._token2index.get(token, default)


class DataTransform(object):
    """Deal with the line, transform it into a proper representation"""
    def __init__(self, max_word_length):
        self.max_word_length = max_word_length

    @staticmethod
    def truncate(x, length):
        x = x[:length - 2]
        return x

    def __call__(self, line, eos=global_constant['EOS']):
        line = line.strip()
        line = line.replace(global_constant['START'], '').replace(global_constant['END'],
                                                                  '').replace(global_constant['UNK'], '')
        line = line.replace('<unk>', ' | ')
        if eos:
            line = line.replace(eos, '')
        words = [self.truncate(x, self.max_word_length) for x in line.split()]
        return words


class DataReader(object):
    """
    Read content from  text file and extract all proper words
    return the vocabulary of the text file and its word tensor and char tensor
    accept a `Transform` class that transform the word
    """
    def __init__(self, path='data', filename='train', max_word_length=65, transform=None):
        self.path = path
        self.filename = filename
        self.transform = transform
        self.word_vocab = Vocabulary()
        self.char_vocab = Vocabulary()

        self.word_tokens = collections.defaultdict(list)
        self.char_tokens = collections.defaultdict(list)

        self.max_word_length = max_word_length
        self.actual_max_length = 0 # record the max word length in the corpus
        self._initial()

    def _initial(self):
        print("Initialize the special character")
        self.char_vocab.feed(global_constant['BLANK'])  # blank is at index 0 in char vocab
        self.char_vocab.feed(global_constant['START'])  # start is at index 1 in char vocab
        self.char_vocab.feed(global_constant['END'])  # end is at index 2 in char vocab

        self.word_vocab.feed(global_constant['UNK'])  # <unk> is at index 0 in word vocab

    def _load(self, filename, extension=".txt", eos=global_constant['EOS']):
        """load data from one file, convert the data into structure data"""
        file_path = os.path.join(self.path, filename + extension)
        assert os.path.exists(file_path), "PathError: {0} does't exist.".format(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                words = self.transform(line, eos)
                for word in words:
                    self.word_tokens[filename].append(self.word_vocab.feed(word))
                    char_array = [self.char_vocab.feed(c) for c in global_constant['START'] + word + global_constant['END']]
                    self.char_tokens[filename].append(char_array)
                    self.actual_max_length = max(len(char_array), self.actual_max_length)
                if eos:
                    self.word_tokens[filename].append(self.word_vocab.feed(eos))
                    char_array = [self.char_vocab.feed(c) for c in '{' + eos + '}']
                    self.char_tokens[filename].append(char_array)

    def load(self):
        """load data from multiple file or single file"""
        if isinstance(self.filename, (list, tuple)):
            for name in self.filename:
                self._load(name)
        else:
            self._load(self.filename)

        return self._get(self.filename)

    def _get(self, filename):
        """change the list to numpy array, and pad 0 if the word is not the actual_max_length"""
        word_tensor = {}
        char_tensor = {}
        filename = [filename] if not isinstance(filename, (list, tuple)) else filename
        for name in list(filename):
            assert len(self.char_tokens[name]) == len(self.word_tokens[name])

            word_tensor[name] = np.array(self.word_tokens[name], dtype=np.int32)
            char_tensor[name] = np.zeros([len(self.char_tokens[name]), self.actual_max_length], dtype=np.int32)
            for i, char_array in enumerate(self.char_tokens[name]):
                char_tensor[name][i, :len(char_array)] = char_array
        return self.actual_max_length, self.word_vocab, self.char_vocab, word_tensor, char_tensor


class WordDataSet(Dataset):
    """inherit the Dataset, then we can use pytorch dataset loader, and batch the samples"""
    def __init__(self, word_tensor, char_tensor, batch, num_unroll_steps):
        self.x_data = char_tensor
        self.y_data = np.zeros_like(word_tensor)
        self.y_data[:-1] = word_tensor[1:].copy()
        self.y_data[-1] = word_tensor[0]

        self.num_unroll_steps = num_unroll_steps
        self.batch = batch
        self._initial()

    def _initial(self):
        size = self.x_data.shape[0]
        reduce_length = (size // (self.num_unroll_steps * self.batch)) * self.num_unroll_steps * self.batch
        self.x_data = self.x_data[:reduce_length]
        self.y_data = self.y_data[:reduce_length]
        print("Original shape:", self.x_data.shape)
        self.x_data = self.x_data.reshape([-1, self.num_unroll_steps, self.x_data.shape[1]])
        print("After shape:", self.x_data.shape)
        self.y_data = self.y_data.reshape([-1, self.num_unroll_steps])
        print(self.y_data.shape)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return torch.LongTensor(self.x_data[index]), torch.LongTensor(self.y_data[index])