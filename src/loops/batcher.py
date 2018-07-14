import numpy as np

import torch


def batch_it(data, batch_size, random=False):
    idx = np.arange(len(data))
    if random:
        idx = np.random.choice(idx, len(idx), replace=False)
    for i in range(0,len(data),batch_size):
        torch_data = torch.from_numpy(data[idx[i:i+batch_size]]).float()
        yield i, torch_data