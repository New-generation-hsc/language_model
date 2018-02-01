import torch
import time
import math

from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from model import model
from utils import process

transform = process.DataTransform(65)
reader = process.DataReader("data", ["smalltrain", "smallvalid", "smalltest"], transform=transform)

max_length, word_vocab, char_vocab, \
 word_tensor, char_tensor = reader.load()
print("finish reading data")

model_ = model.Model(char_vocab.size, word_vocab.size)
optimizer = SGD(model_.parameters(), lr=1.0)

train_set = process.WordDataSet(word_tensor["smalltrain"], char_tensor["smalltrain"], 20, 35)
train_loader = DataLoader(train_set, batch_size=20, shuffle=False)

valid_set = process.WordDataSet(word_tensor["smallvalid"], char_tensor["smallvalid"], 20, 35)
valid_loader = DataLoader(valid_set, batch_size=20, shuffle=False)

test_set = process.WordDataSet(word_tensor["smalltest"], char_tensor["smalltest"], 20, 35)
test_loader = DataLoader(valid_set, batch_size=20, shuffle=False)


def repackage_hidden(h):
    """
    It is a step that need to understand
    :param h:
    :return:
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def train():

    model_.train()
    average_loss = 0.
    average_train_loss = 0.
    batch_count = 0
    batch_size = len(train_loader)

    start_time = time.time()
    hidden_state = model_.init_hidden(20)

    for x, y in train_loader:

        batch_count += 1
        source, target = Variable(x), Variable(y.view(-1))
        hidden_state = repackage_hidden(hidden_state)
        model_.zero_grad()
        output, hidden_state = model_(source, hidden_state)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem
        clip_grad_norm(model_.parameters(), 5)
        for p in model_.parameters():
            p.data.add_(-lr, p.grad.data)

        average_loss += loss.data[0]

        if batch_count % 10 == 0:
            cur_loss = average_loss / 100
            elapsed = time.time() - start_time
            print('[TRAIN]| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format( epoch, batch_count, batch_size, lr,
                  elapsed * 1000 / 20, cur_loss, math.exp(cur_loss)))
            average_loss = 0
            start_time = time.time()


def evaluate(data_loader):

    model_.eval()
    total_loss = 0
    count = 0
    hidden = model_.init_hidden(20)

    for x, y in data_loader:
        source, target = Variable(x), Variable(y.view(-1))
        output, hidden = model_(source, hidden)
        size = output.shape[0]
        total_loss += size * F.cross_entropy(output, target).data[0]
        hidden = repackage_hidden(hidden)
        count += size
    return total_loss / count


best_val_loss = None
lr = 1.0
steps = 0

try:
    for epoch in range(1, 26):
        epoch_start_time = time.time()
        print("EPOCH [{0}] : {1}".format(epoch, epoch_start_time))
        train()
        val_loss = evaluate(valid_loader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print("Existing from training early")


# Run on test data
test_loss = evaluate(test_loader)
print('-' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('-' * 89)
