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
        self.num_images = 0

    def __len__(self): 
        return min(len(self.train_img_files), len(self.mask_img_files))

    def __getitem__(self, idx):
        self.num_images = self.num_images + 1
        print(f"Getting item at index: {idx}, total number of images: {self.num_images}")
        train_img_path = os.path.join(self.train_img_dir, self.train_img_files[idx])
        mask_img_path = os.path.join(self.mask_img_dir, self.mask_img_files[idx])

        train_image = read_image(train_img_path) #getting the image
        mask_image = read_image(mask_img_path)

        if self.transform:
            train_image = self.transform(train_image) #applying the effects
            mask_image = self.transform(mask_image)

        return train_image, mask_image

