"""
dataset.load: to load the data
"""

import os
import pickle as pkl


def load_dataset(path):
    with open(path, 'rb') as f:
        X, y = pkl.load(f)
    return X, y


if __name__ == '__main__':
    print(load_dataset('/gel/usr/lerob17/data/bio/xp1_pref.pkl'))
