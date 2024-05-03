import os
from torchvision.io import read_image
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

        train_image = read_image(train_img_path)
        mask_image = read_image(mask_img_path)

        if self.transform:
            train_image = self.transform(train_image)
            mask_image = self.transform(mask_image)

        return train_image, mask_image
