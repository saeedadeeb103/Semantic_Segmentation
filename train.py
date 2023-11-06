import os
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import get_loader

# Import your U-Net model (replace with the appropriate import)
from unet import Unet

# Create the training dataset
TRAIN_IMG_DIR = 'HumanDataset/train/images'
TRAIN_MASK_DIR = 'HumanDataset/train/masks'
VAL_IMG_DIR = 'HumanDataset/val/images'
VAL_MASK_DIR = 'HumanDataset/val/masks'
batch_size = 16
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


class Segment(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(Segment, self).__init__()
        self.model= Unet(in_channels= in_channels, out_channels=out_channels)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x , y = batch 

        y = y.float().unsqueeze(1)
        y_pred = self(x)
        # Ensure that y has a single channel
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        
        return loss 
    def validation_step(self, batch):
        x, y = batch 
        y = y.float().unsqueeze(1)
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 3e-3)
    
    def on_validation_epoch_end(self):
        avg_loss = self.trainer.logged_metrics['val_loss_epoch']

        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True)

        return {'Average Loss:': avg_loss}



def main():


    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0), 
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.1),
        A.RandomContrast(),
        A.Normalize(
            mean=(0.0, 0.0, 0.0), 
            std = [1.0, 1.0, 1.0], 
            max_pixel_value=255.0,
        ), 
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=(0.0, 0.0, 0.0), 
            std = [1.0, 1.0, 1.0], 
            max_pixel_value=255.0,
        ), 
        ToTensorV2(),
    ])


    train_loader, val_loader  = get_loader(
        train_dir=TRAIN_IMG_DIR,
        train_mask_dir=TRAIN_MASK_DIR,
        val_dir=VAL_IMG_DIR,
        val_mask_dir=VAL_MASK_DIR,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=2)
    
    model = Segment(in_channels=3, out_channels= 1)
    trainer = pl.Trainer(
        max_epochs=400, 
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
        check_val_every_n_epoch=5,
    )
    trainer.fit(
        model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    state_dict = model.state_dict()
    model_name = 'model.pth'
    torch.save(state_dict, model_name)

if __name__ == "__main__":
    main()


