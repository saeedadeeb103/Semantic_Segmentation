import os 
from PIL import Image 
from torch.utils.data import Dataset
import numpy as np 



class Enhancer_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, subset='train'):
        self.root_dir = os.path.join(root_dir, subset)
        self.transform = transform
        self.subset = subset

        if self.subset == 'train':
            self.clean_dir = os.path.join(self.root_dir, 'clean')
            self.noisy_dir = os.path.join(self.root_dir, 'noisy')
            self.image_list = os.listdir(self.clean_dir)
        elif self.subset == 'test':
            # If you have a separate test subset, handle it here
            self.image_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]

        if self.subset == 'train':
            clean_img = Image.open(os.path.join(self.clean_dir, image_name))
            noisy_img = Image.open(os.path.join(self.noisy_dir, image_name))
        elif self.subset == 'test':
            image_path = os.path.join(self.root_dir, image_name)
            noisy_img = Image.open(image_path)
            clean_img = None  # You may not have clean images for testing

        if self.transform:
            noisy_img = self.transform(noisy_img)
            if clean_img:
                clean_img = self.transform(clean_img)

        if self.subset == 'train':
            return noisy_img, clean_img
        elif self.subset == 'test':
            return noisy_img
        
class HumanDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform= None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.png"))

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        mask[mask== 255.0] = 1.0

        if self.transform != None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        return image, mask
