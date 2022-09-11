import torch
import torch.nn as nn
from torch.nn.modules import Module


class L1_Mask_Loss(Module):
    def __init__(self, weight=1):
        super(L1_Mask_Loss, self).__init__()
        assert weight >= 1, "the input weight must be larger than 1"
        if weight == 1:
            self.add_ele = 1
        else:
            self.add_ele = weight / (weight - 1)
        self.factor = 1.0 / self.add_ele
        self.weight = weight
        self.criterion = nn.L1Loss(reduction="mean")

    def forward(self, gt, pred, mask, loss_weight=1):
        pred = pred.requires_grad_()
        mask = (mask * self.weight + self.add_ele) * self.factor
        gt = gt * mask
        pred = pred * mask
        loss = self.criterion(gt, pred) * loss_weight

        return loss


if __name__ == '__main__':
    criterion = L1_Mask_Loss(weight=10)
    a = torch.abs(torch.randn(2, 1, 16, 16))
    b = torch.abs(torch.randn(2, 1, 16, 16))
    c = torch.abs(torch.randn(2, 1, 16, 16))
    loss = criterion(a, b, c)
    loss.backward()
    print(loss)
