import os

import argparse
import random

import matplotlib
matplotlib.use('qt5agg')
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.experiment import Experiment
from src.models import PrefNet
from src.dataset.load import load_dataset
from src.dataset.utils import train_valid_test_split, flatten_dataset
from src.loops import train

VALID_SIZE = 0.1
TEST_SIZE = 0.1


def graph_results(model, subset, location, y):
    ## real
    max_idx_real = y
    fig, (ax1, ax2) = plt.subplots(1,2)
    for i, p in enumerate(subset):
        if i == max_idx_real:
            ax1.scatter(*p[:2], color='green')
            if len(p) == 3:
                ax1.annotate(p[2], xy=p[:2], xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        else:
            ax1.scatter(*p[:2], color='red')
    ax1.set_title('real')
    ax1.set_xlabel('quality')
    ax1.set_ylabel('bleach')
    ## predict
    max_idx_predict = np.argmax(model.predict(subset))
    for i, p in enumerate(subset):
        if i == max_idx_predict:
            ax2.scatter(*p[:2], color='green')
            if len(p) == 3:
                ax2.annotate(p[2], xy=p[:2], xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        else:
            ax2.scatter(*p[:2], color='red')
    ax2.set_title('predict')
    ax2.set_xlabel('quality')
    ax2.set_ylabel('bleach')
    plt.savefig(location)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a prefNet')

    parser.add_argument('-m', '--margin', help='margin of the loss', type=float, 
                        default=0.0)
    parser.add_argument('-bs', '--batch-size', type=int, help='SGD batch size',
                        default=16)
    parser.add_argument('-ep', '--nb-epochs', type=int, help='SGD number of epochs',
                        default=5)
    parser.add_argument('-rs', '--random-state', type=int, help='random state of train/valid/test split',
                        default=42)    
    parser.add_argument('--cuda', help='use GPU or not', action="store_true",
                        default=False)
    parser.add_argument('data_path', help='path to the data to use', type=str)
    parser.add_argument('results_path', help='where to save the results', type=str)
    
    args = parser.parse_args()

    X, y = load_dataset(args.data_path)

    nb_obj = X[0].shape[-1]

    trainset, validset, testset = train_valid_test_split(X, y, valid_size=VALID_SIZE,
                                                test_size=TEST_SIZE, 
                                                random_state=args.random_state)

    train_flatten = flatten_dataset(trainset[0], trainset[1], random_state=args.random_state)
    valid_flatten = flatten_dataset(validset[0], validset[1], random_state=args.random_state)
    test_flatten = flatten_dataset(testset[0], testset[1], random_state=args.random_state)

    train_mean = np.mean(train_flatten)
    train_std = np.std(train_flatten)

    train_flatten = (train_flatten - train_mean) / train_std
    valid_flatten = (valid_flatten - train_mean) / train_std

    early_stop = 10

    experiments_config = {
        'nb_epochs': args.nb_epochs,
        'batch_size': args.batch_size,
        'margin': args.margin,
        'random_state': args.random_state,
        'cuda': args.cuda,
        'train_mean': float(train_mean),
        'train_std': float(train_std),
        'early_stop': early_stop,
        }

    xp = Experiment(experiments_config, args.results_path)

    print(experiments_config)

    model = PrefNet(nb_obj=nb_obj)
    min_model = np.min(model.predict(train_flatten))
    max_model = np.max(model.predict(train_flatten))
    while (max_model - min_model) < 0.1:
        model = PrefNet(nb_obj=nb_obj)
        min_model = np.min(model.predict(train_flatten))
        max_model = np.max(model.predict(train_flatten))

    print(' [-] min_model {:.3f}, max_model {:.3f}'.format(min_model, max_model))

    best_weights, record = train(model, train_flatten, \
                                 valid_flatten, \
                                 args.nb_epochs, \
                                 args.batch_size, \
                                 margin=args.margin, \
                                 cuda=args.cuda, \
                                 early_stop=early_stop)

    xp.save_model(torch.save, best_weights)
    xp.save_record(record)

    if nb_obj == 2:
        b, q = np.mgrid[0.0:1.0:0.01, 0.0:1.0:0.01]
        obj = np.concatenate((b.flatten().reshape(-1, 1), q.flatten().reshape(-1, 1)), axis=1)
        values = model.predict((obj-train_mean)/train_std).reshape(len(b), len(q))

        plt.close('all')
        plt.imshow(values, origin='lower', cmap='rainbow')
        plt.ylabel('quality')
        plt.xlabel('bleach')
        plt.colorbar()
        plt.show()

    test_sets_idx = list(range(len(testset[0])))
    random.shuffle(test_sets_idx)
    for i in test_sets_idx[:10]:
        location = os.path.join(xp.get_figuredir(), 'test-set-{}.png'.format(i))
        graph_results(model, (testset[0][i] - train_mean) / train_std, location, testset[1][i])

