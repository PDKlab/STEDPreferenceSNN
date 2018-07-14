import copy
import time

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from .losses import HingeLoss
from .batcher import batch_it


def train(model, train_data, valid_data, nb_epochs, batch_size, margin, cuda=False, early_stop=10):
    optimizer = optim.Adam(model.parameters()) ## TODO: more flexible
    criterion = HingeLoss(margin)
    train_losses, valid_losses = [], []
    min_valid_loss = np.inf
    early_stop_cnt = 0
    if cuda:
        model.cuda()
    start = time.time()
    for i_epoch in range(nb_epochs):
        tlosses = []
        ## training
        model.train()
        for i_batch, batch in batch_it(train_data, batch_size, random=True):
            if cuda:
                batch = batch.cuda()
            vbatch = Variable(batch)
            y_right, y_left = model(vbatch[:,0]), model(vbatch[:,1])
            optimizer.zero_grad()
            loss = criterion(y_right, y_left)
            loss.backward()
            optimizer.step()
            tlosses.append(loss.data.cpu().numpy()[0])
        vlosses = []
        ## validation
        model.eval()
        for i_batch, batch in batch_it(valid_data, batch_size):
            if cuda:
                batch = batch.cuda()
            vbatch = Variable(batch, volatile=True)
            y_right, y_left = model(vbatch[:,0]), model(vbatch[:,1])
            loss = criterion(y_right, y_left)
            vlosses.append(loss.data.cpu().numpy()[0])
        if np.mean(vlosses) < min_valid_loss:
            early_stop_cnt = 0
            effective_epochs = i_epoch
            min_train_loss = np.mean(tlosses)
            min_valid_loss = np.mean(vlosses)
            best_weights = copy.deepcopy(model.state_dict())
        elif early_stop_cnt == early_stop - 1:
            print(' [-]: * Early stopping * min_train:{:.6f}, min_valid:{:.6f}'.format(
                                                         min_train_loss,
                                                         min_valid_loss))
            break
        else:
            early_stop_cnt += 1

        end = time.time() - start
        print(' [-] epoch {}, train loss: {:.10f}, valid loss: {:.10f} in {}min{}'.format(i_epoch,
                                                              np.mean(tlosses),
                                                              np.mean(vlosses),
                                                              int(end // 60), int(end % 60)))
        train_losses.append(np.mean(tlosses))
        valid_losses.append(np.mean(vlosses))

    model.load_state_dict(best_weights)
    training_record = {
        'effective_epochs': effective_epochs,
        'min_train_loss': min_train_loss,
        'min_valid_loss': min_valid_loss,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
    }

    model.cpu()
    return best_weights, training_record


