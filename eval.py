import os
import json
import argparse
import random

import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt

import torch

from src.models import PrefNet
from src.dataset.load import load_dataset
from src.dataset.utils import train_valid_test_split, flatten_dataset



def graph_results(model, subset, location, y, mean, std):
    ## real
    max_idx_real = y
    fig, (ax1, ax2) = plt.subplots(1,2)
    for i, p in enumerate(subset):
        #p *= std
        #p += mean
        if i == max_idx_real:
            ax1.scatter(*p[:2], color='green')
            if len(p) == 3:
                ax1.annotate('{:.3f} us'.format(p[2]*1e6), xy=p[:2], xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        else:
            ax1.scatter(*p[:2], color='red')
    ax1.set_title('real')
    ax1.set_xlabel('quality')
    ax1.set_ylabel('bleach')
    ## predict
    max_idx_predict = np.argmax(model.predict((subset-mean)/std))
    for i, p in enumerate(subset):
        #p *= std
        #p += mean
        if i == max_idx_predict:
            ax2.scatter(*p[:2], color='green')
            if len(p) == 3:
                ax2.annotate('{:.3f} us'.format(p[2]*1e6), xy=p[:2], xytext=(-20, 20),
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
    parser = argparse.ArgumentParser(description='eval a prefNet')
    parser.add_argument('-rs', '--random-state', type=int, help='random state of train/valid/test split',
                        default=42) 
    parser.add_argument('data_path', help='path to the data to use', type=str)
    parser.add_argument('results_path', help='where to save the results', type=str)
    
    args = parser.parse_args()

    X, y = load_dataset(args.data_path)

    nb_obj = X[0].shape[-1]

    trainset, validset, testset = train_valid_test_split(X, y, valid_size=0.1,
                                                test_size=0.1, 
                                                random_state=args.random_state)

    train_flatten = flatten_dataset(trainset[0], trainset[1], random_state=args.random_state)
    valid_flatten = flatten_dataset(validset[0], validset[1], random_state=args.random_state)
    test_flatten = flatten_dataset(testset[0], testset[1], random_state=args.random_state)

    xp_folder = args.results_path
    with open(os.path.join(xp_folder, 'config.json'), 'r') as f:
        config = json.load(f)

    train_mean = config['train_mean']
    train_std = config['train_std']

    figuredir = os.path.join(xp_folder, 'figures')

    model = PrefNet(nb_obj=nb_obj)
    model.load_state_dict(torch.load(os.path.join(xp_folder, 'weights.t7')))
    model.eval()

    if nb_obj == 2:
        b, q = np.mgrid[0.0:1.0:0.01, 0.0:1.0:0.01]
        obj = np.concatenate((b.flatten().reshape(-1, 1), q.flatten().reshape(-1, 1)), axis=1)
        values = model.predict((obj-train_mean)/train_std).reshape(len(b), len(q))
        plt.close('all')
        plt.imshow(values, origin='lower', cmap='rainbow')
        plt.ylabel('quality')
        plt.xlabel('bleach')
        plt.colorbar()
        plt.savefig(os.path.join(figuredir, 'map.pdf'))

    test_sets_idx = list(range(len(testset[0])))
    random.shuffle(test_sets_idx)
    for i in test_sets_idx[:70]:
        location = os.path.join(figuredir, 'test-set-{}.png'.format(i))
        graph_results(model, testset[0][i], location, testset[1][i], train_mean, train_std)
