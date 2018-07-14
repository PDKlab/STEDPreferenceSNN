import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, y_right, y_left):
        return torch.mean(F.relu(y_left - y_right + self.margin))