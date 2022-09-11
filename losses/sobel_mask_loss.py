import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules import Module


class Sobel_Mask_Loss(Module):
    def __init__(self, weight=1):
        super(Sobel_Mask_Loss, self).__init__()
        assert weight >= 1, "the input weight must be larger than 1"
        if weight == 1:
            self.add_ele = 1
        else:
            self.add_ele = weight / (weight - 1)
        self.factor = 1.0 / self.add_ele
        self.weight = weight
        x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1], ], dtype=np.float32).reshape(1, 1, 3, 3)
        y = x.copy().T.reshape(1, 1, 3, 3)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        self.kernelx = Variable(x.contiguous())
        self.kernely = Variable(y.contiguous())
        self.criterion = torch.nn.L1Loss(reduction="mean")

    def forward(self, gt, pred, mask, direction="x", loss_weight=1):
        pred = pred.requires_grad_()
        mask = (mask * self.weight + self.add_ele) * self.factor
        gt = gt * mask
        pred = pred * mask
        if direction == "x":
            tx = gt
            px = pred
            sobel_tx = F.conv2d(tx, self.kernelx, padding=1)
            sobel_px = F.conv2d(px, self.kernelx, padding=1)
            loss = self.criterion(sobel_tx, sobel_px)
        else:
            ty = gt
            py = pred
            sobel_ty = F.conv2d(ty, self.kernely, padding=1)
            sobel_py = F.conv2d(py, self.kernely, padding=1)
            loss = self.criterion(sobel_ty, sobel_py) * loss_weight

        return loss


if __name__ == '__main__':
    criterion = Sobel_Mask_Loss()
    a = torch.abs(torch.randn(2, 1, 16, 16))
    b = torch.abs(torch.randn(2, 1, 16, 16))
    c = torch.abs(torch.randn(2, 1, 16, 16))
    loss = criterion(a, b, c)
    loss.backward()
    print(loss)
