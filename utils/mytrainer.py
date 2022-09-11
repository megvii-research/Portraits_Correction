import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
import torchvision.transforms as transforms
from losses import *
from models import *
from utils.Calculator import Calculator
from utils.dataset import DataProvider, LabeledData, UnLabeledData
from utils.myutils import map_to_mask, progress_bar

PARTIAL_LOAD = True


def get_weight_mask(mask, gt, pred, weight=10):
    mask = (mask * (weight - 1)) + 1
    gt = gt.mul(mask)
    pred = pred.mul(mask)
    return gt, pred


def adjust_weight(epoch, total_epoch, weight):
    return (1 - 0.9 * (epoch / total_epoch)) * weight


class MyTrainer():
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        self.device = torch.device('cuda:{}'.format(args.local_rank))
        self.downsample_ratio = args.downsample_ratio
        labeled_data_dir = args.labeled_data_dir
        unlabeled_data_dir = args.unlabeled_data_dir

        torch.cuda.manual_seed(args.seed)
        self.model = MS_UNet()
        self.model = self.model.cuda()

        print("Using DistributedDataParallel")
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        self.model = torch.nn.DataParallel(self.model)
        cudnn.benchmark = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.L1Loss(reduction="mean").to(self.device)
        self.criterion1 = Sobel_Loss().to(self.device)
        self.criterion_cls = nn.CrossEntropyLoss(size_average=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.start_epoch = 0

        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if PARTIAL_LOAD:
                model_dict = self.model.state_dict()
                if suf == 'tar':
                    print("The model load 'tar' file!")
                    checkpoint = torch.load(args.resume, self.device)
                    pretrained_dict = checkpoint['model_state_dict']
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    # update model_dict
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict, False)
                elif suf == 'pth':
                    print("The model load 'pth' file!")
                    self.model.load_state_dict(torch.load(args.resume, self.device), False)
            else:
                if suf == 'tar':
                    print("The model load 'tar' file!")
                    checkpoint = torch.load(args.resume, self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'], False)
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.start_epoch = checkpoint['epoch'] + 1
                elif suf == 'pth':
                    print("The model load 'pth' file!")
                    self.model.load_state_dict(torch.load(args.resume, self.device), False)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        self.save_dir = args.save_dir
        self.batch_size = 10
        print('==> Preparing data..')
        labled_train_dataset = LabeledData(labeled_data_dir, 512, 384, down_sample_factor=1)
        self.labled_train_loader = DataProvider(labled_train_dataset, batch_size=self.batch_size, shuffle=True,
                                                num_workers=8)
        unlabled_train_dataset = UnLabeledData(unlabeled_data_dir, 512, 384, down_sample_factor=1)
        self.unlabled_train_loader = DataProvider(unlabled_train_dataset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=8)
        self.labled_step = 4
        self.unlabled_step = 1
        self.batch_nums = int(1000 / self.batch_size / self.labled_step)

        self.best_mae = 0
        self.best_mse = np.inf
        self.best_epoch = np.inf
        print("setup successfully!")

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            print(" ")
            print('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()

    def train_eopch(self):
        epoch_loss = Calculator()
        epoch_sobel_loss = Calculator()
        epoch_l_loss = Calculator()
        epoch_loss_cls = Calculator()
        epoch_start = time.time()
        self.model.train()
        for batch_idx in range(self.batch_nums):

            # utilize the labled samples to train the network
            for i in range(self.labled_step):
                inputs, map_x, map_y, facemask, weight = self.labled_train_loader.next()
                inputs, map_x, map_y = inputs.to(self.device), map_x.to(self.device), map_y.to(self.device)
                facemask, weight = facemask.to(self.device), weight.to(self.device)
                # forward propagation
                outputs = self.model(inputs)
                weight = adjust_weight(self.epoch, self.args.max_epoch, weight)
                est_map_x, est_map_y = outputs[:, 0, :, :].unsqueeze(1), outputs[:, 1, :, :].unsqueeze(1)
                est_mask_x, est_mask_y = outputs[:, 2:5, :, :], outputs[:, 5:8, :, :]

                # Compute the segmentation loss
                gt_mask_x = map_to_mask(map_x, -5, 5)
                gt_mask_y = map_to_mask(map_y, -5, 5)
                loss_cls_x = self.criterion_cls(est_mask_x, gt_mask_x)
                loss_cls_y = self.criterion_cls(est_mask_y, gt_mask_y)
                loss_cls = (loss_cls_y + loss_cls_x) * 15

                map_x, est_map_x = get_weight_mask(facemask, map_x, est_map_x, weight=weight)
                map_y, est_map_y = get_weight_mask(facemask, map_y, est_map_y, weight=weight)
                """calculate loss"""
                # Calculate L1 loss

                l_loss_x = self.criterion(est_map_x, map_x)
                l_loss_y = self.criterion(est_map_y, map_y)
                l_loss = (l_loss_x + l_loss_y)

                # Compute sobel Loss
                sobel_loss_x = self.criterion1(map_x, est_map_x, direction='x')
                sobel_loss_y = self.criterion1(map_y, est_map_y, direction='y')
                sobel_loss = (sobel_loss_x + sobel_loss_y) * 10
                loss = l_loss + sobel_loss + loss_cls

                # Update the parameters and loss record
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                del inputs, map_x, map_y, facemask, outputs

                # utilize the unlabled samples to train the network
            for j in range(self.unlabled_step):
                inputs1, inputs2 = self.unlabled_train_loader.next()
                # print(step)
                inputs1, inputs2 = inputs1.to(self.device), inputs2.to(self.device)
                # forward propagation
                outputs1 = self.model(inputs1)
                outputs2 = self.model(inputs2)

                est_map_x1, est_map_y1 = outputs1[:, 0, :, :].unsqueeze(1), outputs1[:, 1, :, :].unsqueeze(1)
                est_mask_x1, est_mask_y1 = outputs1[:, 2:5, :, :], outputs1[:, 5:8, :, :]

                est_map_x2, est_map_y2 = outputs2[:, 0, :, :].unsqueeze(1), outputs2[:, 1, :, :].unsqueeze(1)
                est_mask_x2, est_mask_y2 = outputs2[:, 2:5, :, :], outputs2[:, 5:8, :, :]

                # Compute the segmentation loss
                gt_mask_x1 = map_to_mask(est_map_x1, -5, 5)
                gt_mask_y1 = map_to_mask(est_map_y1, -5, 5)
                loss_cls_x1 = self.criterion_cls(est_mask_x1, gt_mask_x1)
                loss_cls_y1 = self.criterion_cls(est_mask_y1, gt_mask_y1)
                gt_mask_x2 = map_to_mask(est_map_x2, -5, 5)
                gt_mask_y2 = map_to_mask(est_map_y2, -5, 5)
                loss_cls_x2 = self.criterion_cls(est_mask_x2, gt_mask_x2)
                loss_cls_y2 = self.criterion_cls(est_mask_y2, gt_mask_y2)
                loss_cls = (loss_cls_y1 + loss_cls_x1 + loss_cls_y2 + loss_cls_x2)

                # Compute the L1 loss
                l_loss_x = self.criterion(est_map_x1, est_map_x2)
                l_loss_y = self.criterion(est_map_y1, est_map_y2)
                l_loss = (l_loss_x + l_loss_y)

                # Compute sobel Loss
                sobel_loss_x = self.criterion1(est_map_x1, est_map_x2, direction='x')
                sobel_loss_y = self.criterion1(est_map_y1, est_map_y2, direction='y')
                sobel_loss = (sobel_loss_x + sobel_loss_y) * 10

                loss = (l_loss + sobel_loss + loss_cls) * 0.2

                # Update the parameters and loss record
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                del inputs1, inputs2, outputs1, outputs2

            epoch_loss.update(loss.item())
            epoch_sobel_loss.update(sobel_loss.item())
            epoch_l_loss.update(l_loss.item())
            epoch_loss_cls.update(loss_cls.item())
            progress_bar(batch_idx, self.batch_nums,
                         'L1 Loss: %.3f, Sobel Loss: %.3f, CLS Loss: %.3f' % (
                             l_loss.item(), sobel_loss.item(), loss_cls.item()))
        print('Epoch {} Train, L Loss: {:.4f}, Sobel Loss: {:.4f}, CLS loss:{:.4f}, Total Loss: {:.4f}, Cost {:.1f} sec'
              .format(self.epoch, epoch_l_loss.get_avg(), epoch_sobel_loss.get_avg(), epoch_loss_cls.get_avg(),
                      epoch_loss.get_avg(), time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, 'latest_epoch.tar')
        if self.epoch % 1 == 0:
            save_path = os.path.join(self.save_dir, '{}_epoch.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        epoch_loss = Calculator()
        correct = 0
        total = 0
        test_loss = 0
        # Iterate over data.
        for batch_idx, (inputs, targets) in enumerate(self.val_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # forward propagation
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            epoch_loss.update(loss.item())
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(self.val_loader),
                         'Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
        print('Epoch {} Train, Loss: {:.4f}, Cost {:.1f} sec, ACC: {:.2f}%'
              .format(self.epoch, epoch_loss.get_avg(), time.time() - epoch_start, 100. * correct / total))
        # Save checkpoint.
        model_state_dic = self.model.state_dict()
        acc = 100. * correct / total
        if acc > self.best_mae:
            self.best_mae = acc
            self.best_epoch = self.epoch
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
        print("[** Current best accuracy is: {:.2f}% **]".format(self.best_mae))
