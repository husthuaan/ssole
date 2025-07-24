# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random

from PIL import ImageFilter, ImageOps
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
        return_target=False,
        color_distortion_scale=1.0,
        center_crop_small=1.0,
        solar_prob=0.0,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        self.nmb_crops = nmb_crops
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index
        self.return_target = return_target

        color_transform = [get_color_distortion(s=color_distortion_scale), PILRandomGaussianBlur(), Solarization(p=solar_prob)]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            if i>0 and center_crop_small<1.0:
                randomresizedcrop = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(int(256 * center_crop_small)),
                    randomresizedcrop,
                ])
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        stked_m_crops = []
        start_idx = 0
        for end_idx in np.cumsum(self.nmb_crops):
            if end_idx == start_idx:
                continue
            stked_m_crops.append(torch.stack(multi_crops[start_idx:end_idx], dim=0))
            start_idx=end_idx

        if self.return_index:
            return index, stked_m_crops
        elif self.return_target:
            return stked_m_crops, target
        return stked_m_crops

    def _data_loader(self, batch_size, workers):
        data_loader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            drop_last=True
        )
        return data_loader

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
