import os
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision import transforms as v2
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, train_img_dir, mask_img_dir, transform=None):
        self.train_img_dir = train_img_dir
        self.mask_img_dir = mask_img_dir
        self.train_img_files = os.listdir(train_img_dir)
        self.mask_img_files = os.listdir(mask_img_dir)
        self.transform = transform

    def __len__(self):
        return min(len(self.train_img_files), len(self.mask_img_files))

    def __getitem__(self, idx):
        train_img_path = os.path.join(self.train_img_dir, self.train_img_files[idx])
        mask_img_path = os.path.join(self.mask_img_dir, self.mask_img_files[idx])

        train_img = cv2.imread(train_img_path)
        train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)

        mask_image = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        _, thresholded_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

        if self.transform:
            augmented = self.transform(image=train_img, mask=thresholded_mask)
            train_img = augmented['image']
            thresholded_mask = augmented['mask']

        train_img = train_img.float() / 255.0
        thresholded_mask = thresholded_mask.float() / 255.0

        return train_img, thresholded_mask
