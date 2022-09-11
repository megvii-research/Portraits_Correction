import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules import Module


class Sobel_Loss(Module):
    def __init__(self):
        super(Sobel_Loss, self).__init__()
        x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32).reshape(1, 1, 3, 3)
        y = x.copy().T.reshape(1, 1, 3, 3)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        self.kernelx = Variable(x.contiguous())
        self.kernely = Variable(y.contiguous())
        self.criterion = torch.nn.L1Loss(reduction="mean")

    def forward(self, target, prediction, direction="x"):
        if direction == "x":
            tx = target
            px = prediction
            sobel_tx = F.conv2d(tx, self.kernelx, padding=1)
            sobel_px = F.conv2d(px, self.kernelx, padding=1)
            loss = self.criterion(sobel_tx, sobel_px)
        else:
            ty = target
            py = prediction
            sobel_ty = F.conv2d(ty, self.kernely, padding=1)
            sobel_py = F.conv2d(py, self.kernely, padding=1)
            loss = self.criterion(sobel_ty, sobel_py)

        return loss


if __name__ == '__main__':
    criterion = Sobel_Loss()
    a = torch.abs(torch.randn(2, 1, 16, 16))
    b = torch.abs(torch.randn(2, 1, 16, 16))
    loss = criterion(a, b)
    loss.backward()
    print(loss)
