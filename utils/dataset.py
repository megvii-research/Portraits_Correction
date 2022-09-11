import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

from PIL import Image
from imgaug import augmenters as iaa
from torchvision import transforms
from .randaugment import RandAugmentMC


def data_aug(img):
    oplist = []
    if random.random() > 0.5:
        oplist.append(iaa.GaussianBlur(sigma=(0.0, 1.0)))
    elif random.random() > 0.5:
        oplist.append(iaa.WithChannels(0, iaa.Add((1, 15))))
    elif random.random() > 0.5:
        oplist.append(iaa.WithChannels(1, iaa.Add((1, 15))))
    elif random.random() > 0.5:
        oplist.append(iaa.WithChannels(2, iaa.Add((1, 15))))
    elif random.random() > 0.5:
        oplist.append(iaa.AdditiveGaussianNoise(scale=(0, 10)))
    elif random.random() > 0.5:
        oplist.append(iaa.Sharpen(alpha=0.15))
    elif random.random() > 0.5:
        oplist.append(iaa.Clouds())
    elif random.random() > 0.5:
        oplist.append(iaa.Rain(speed=(0.1, 0.3)))

    seq = iaa.Sequential(oplist)
    images_aug = seq.augment_images([img])
    return images_aug[0]


class TransformFixMatch(object):
    def __init__(self):
        self.common = transforms.Compose([transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([RandAugmentMC(n=2, m=9)])

    def __call__(self, x):
        weak = self.common(x)
        strong = self.strong(weak)
        return weak, strong


class LabeledData():
    def __init__(self, data_dir, input_w, input_h, down_sample_factor=8):
        self.input_w = input_w
        self.input_h = input_h
        self.datas_infos = []
        self.down_sample_factor = down_sample_factor

        self.datas_infos = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) \
                            if os.path.isfile(os.path.join(data_dir, filename)) and filename.endswith("p.jpg")]
        self.datas_num = len(self.datas_infos)
        print("number of datas:", self.datas_num)
        print("init data finished")

    def __len__(self):
        return self.datas_num

    def __getitem__(self, index):
        # Process the input image
        img_path = self.datas_infos[index]
        origimg = cv2.imread(img_path)

        img = cv2.resize(origimg, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        img = data_aug(img)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # process the input mask
        img_mask_path = img_path.replace('.jpg', '_line_facemask.jpg')
        facemask = cv2.imread(img_mask_path, 0)  # read original mask
        facemask = cv2.resize(facemask,
                              (self.input_w // self.down_sample_factor, self.input_h // self.down_sample_factor),
                              interpolation=cv2.INTER_AREA)
        facemask = (facemask / 255.0)
        facemask = torch.from_numpy(facemask).float().unsqueeze(0)

        # Process the flow map
        img_map_x_path = img_path.replace('.jpg', '_ori2shape_mapx.exr')
        img_map_y_path = img_path.replace('.jpg', '_ori2shape_mapy.exr')

        flow_map_x = cv2.imread(img_map_x_path, cv2.IMREAD_ANYDEPTH)  # read flow map x direction
        flow_map_y = cv2.imread(img_map_y_path, cv2.IMREAD_ANYDEPTH)  # read flow map y direction
        flow_map_h, flow_map_w = flow_map_x.shape[:2]

        scale_x = self.input_w // self.down_sample_factor / flow_map_w
        scale_y = self.input_h // self.down_sample_factor / flow_map_h

        flow_map_x = cv2.resize(flow_map_x,
                                (self.input_w // self.down_sample_factor, self.input_h // self.down_sample_factor),
                                interpolation=cv2.INTER_AREA)
        flow_map_y = cv2.resize(flow_map_y,
                                (self.input_w // self.down_sample_factor, self.input_h // self.down_sample_factor),
                                interpolation=cv2.INTER_AREA)
        flow_map_x *= scale_x
        flow_map_y *= scale_y

        flow_map_x = flow_map_x[np.newaxis, :, :]
        flow_map_y = flow_map_y[np.newaxis, :, :]
        flow_map_x = torch.from_numpy(flow_map_x).float()
        flow_map_y = torch.from_numpy(flow_map_y).float()

        # Compute the weight
        mask_sum = torch.sum(facemask)
        weight = (self.input_w // self.down_sample_factor) * (self.input_h // self.down_sample_factor) / mask_sum - 1
        weight = torch.max(weight / 3, torch.ones(1))
        weight = weight.unsqueeze(-1).unsqueeze(-1)

        return img, flow_map_x, flow_map_y, facemask, weight


class UnLabeledData():
    def __init__(self, data_dir, input_w, input_h, down_sample_factor=8):
        self.input_w = input_w
        self.input_h = input_h
        self.datas_infos = []
        self.down_sample_factor = down_sample_factor
        self.transform = TransformFixMatch()
        self.datas_infos = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) \
                            if os.path.isfile(os.path.join(data_dir, filename)) and filename.endswith(".jpg")]
        self.datas_num = len(self.datas_infos)
        print("number of datas:", self.datas_num)
        print("init data finished")

    def __len__(self):
        return self.datas_num

    def __getitem__(self, index):
        # Process the input image
        img_path = self.datas_infos[index]
        origimg = cv2.imread(img_path)
        img = cv2.resize(origimg, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(img)
        img1_tr, img2_tr = self.transform(img)
        img1 = torch.from_numpy(np.array(img1_tr)).permute(2, 0, 1).float()
        img2 = torch.from_numpy(np.array(img2_tr)).permute(2, 0, 1).float()

        return img1, img2


class DataProvider():

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_iter = None
        self.iter = 0
        self.epoch = 0
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.drop_last = drop_last

    def build(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                 num_workers=self.num_workers,
                                                 pin_memory=self.pin_memory,
                                                 drop_last=self.drop_last)
        self.data_iter = _MultiProcessingDataLoaderIter(dataloader)

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iter += 1
            return batch

        except StopIteration:
            self.epoch += 1
            self.build()
            self.iter = 1
            batch = self.data_iter.next()
            return batch


if __name__ == "__main__":
    data_dir = "/data/undistortion/datas/cvpr_0705/train_4_3"
    train_dataset = UndistortData(data_dir, 512, 384)
    for index in range(len(train_dataset)):
        input_img, flow_map_x, flow_map_y, facemask, weight = train_dataset.__getitem__(index)
        input_img = input_img.permute(1, 2, 0).numpy()
        flow_map_x = flow_map_x[0].numpy()
        flow_map_y = flow_map_y[0].numpy()
        facemask = facemask.permute(1, 2, 0).numpy()

        print(input_img.shape, flow_map_x.shape, flow_map_y.shape, facemask.shape, weight.shape)

        input_h, input_w = input_img.shape[:2]
        flow_map_h, flow_map_w = flow_map_x.shape[:2]
        scale_x = input_w / flow_map_w
        scale_y = input_h / flow_map_h
        print(scale_x, scale_y)
        flow_map_x = cv2.resize(flow_map_x, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        flow_map_y = cv2.resize(flow_map_y, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        flow_map_x *= scale_x
        flow_map_y *= scale_y

        ys, xs = np.mgrid[:input_h, :input_w]
        flow_map_x = (flow_map_x + xs).astype(np.float32)
        flow_map_y = (flow_map_y + ys).astype(np.float32)
        print(input_img.shape, flow_map_x.shape, flow_map_y.shape)
        output_img = cv2.remap(input_img, flow_map_x, flow_map_y, interpolation=cv2.INTER_LINEAR)

        img = np.concatenate([input_img, output_img], axis=1)
        print(img.shape)

        cv2.imshow("img", img.astype(np.uint8))
        cv2.waitKey()
