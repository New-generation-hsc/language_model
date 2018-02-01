from utils import process
from torch.utils.data import DataLoader
import numpy as np


transform = process.DataTransform(65)
reader = process.DataReader("../data", ["train", "valid", "test"], transform=transform)

max_length, word_vocab, char_vocab, \
 word_tensor, char_tensor = reader.load()

# print(word_tensor.keys())
# print(word_tensor['test'][:50])
# print("word tensor length:", word_tensor['test'].shape)
dataset = process.WordDataSet(word_tensor["test"], char_tensor["test"], 35)
dataloader = DataLoader(dataset, batch_size=20, shuffle=False)

print(len(dataloader))
count = 0
for source, target in dataloader:
    print(source.shape, target.shape)
    print("source:")
    print(source)
    print("target:")
    print(target)
    if count >= 1:
        break
    count += 1

# num_unroll_steps = 5
# max_word_length = 6
# x = np.arange(0, 20)
# y = np.arange(0, 120).reshape([-1, max_word_length])
#
# dataset = process.WordDataSet(x, y, num_unroll_steps)
# dataloader = DataLoader(dataset, 3, shuffle=False)
#
# for source, target in dataloader:
#     print(source.shape, target.shape)
#     print("source:")
#     print(source)
#     print("target:")
#     print(target)