import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules import Module


class LMK_Loss(Module):
    def __init__(self):
        super(LMK_Loss, self).__init__()

    def forward(self, gt, pred, weight=None):
        pred = pred.requires_grad_()
        dot_sum = (pred * gt).sum(axis=1)
        predm = torch.sqrt((pred * pred).sum(axis=1))
        gtm = torch.sqrt((gt * gt).sum(axis=1))
        if weight is None:
            loss = (1 - dot_sum / (predm * gtm)).sum() / pred.shape[0]
        else:
            loss = ((1 - dot_sum / (predm * gtm)) * weight).sum() / pred.shape[0]

        return loss


if __name__ == '__main__':
    criterion = LMK_Loss()
    a = torch.abs(torch.randn(2, 2, 16, 16))
    b = torch.abs(torch.randn(2, 2, 16, 16))
    c = torch.abs(torch.randn(2, 16, 16))
    loss = criterion(a, b, c)
    loss.backward()
    print(loss)
