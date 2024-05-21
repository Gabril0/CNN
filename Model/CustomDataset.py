import os
import torch
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, train_img_dir, mask_img_dir, transform=None, mask_transform=None):
        self.train_img_dir = train_img_dir
        self.mask_img_dir = mask_img_dir
        self.train_img_files = os.listdir(train_img_dir)
        self.mask_img_files = os.listdir(mask_img_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_images = 0

    def __len__(self):
        return min(len(self.train_img_files), len(self.mask_img_files))

    def __getitem__(self, idx):
        self.num_images += 1
        train_img_path = os.path.join(self.train_img_dir, self.train_img_files[idx])
        mask_img_path = os.path.join(self.mask_img_dir, self.mask_img_files[idx])

        train_image = read_image(train_img_path)  # Getting the image
        mask_image = read_image(mask_img_path)

        if self.transform:
            train_image = self.transform(train_image)  # Applying the effects
            mask_image = self.transform(mask_image)

        if self.mask_transform:
            mask_image = self.mask_transform(mask_image)

        if mask_image.shape[0] > 1:
            mask_image = mask_image[0, :, :]

        return train_image, mask_image
