import os
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision.transforms import transforms
from utils import get_loader
from torchmetrics import Accuracy

# Import your U-Net model (replace with the appropriate import)
from unet import Unet

# Define hyperparameters and paths
BATCH_SIZE = 8
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
TRAIN_IMG_DIR = 'HumanDataset/train/images'
TRAIN_MASK_DIR = 'HumanDataset/train/masks'
VAL_IMG_DIR = 'HumanDataset/val/images'
VAL_MASK_DIR = 'HumanDataset/val/masks'


class Segment(pl.LightningModule):
    def __init__(self, in_channels, out_channels, encoder):
        super(Segment, self).__init__()
        self.model = Unet(in_channels=in_channels, out_channels=out_channels, encoder=encoder)
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task='binary', num_classes=1, threshold=0.5)
        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        x , y = batch 

        y = y.float().unsqueeze(1)
        y_pred = self(x)
        # Ensure that y has a single channel
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)

        accuracy = self.accuracy(y_pred, y)
        self.log("train_acc", accuracy, prog_bar=True, on_epoch=True)
        
        return loss 
    def validation_step(self, batch):
        x, y = batch 
        y = y.float().unsqueeze(1)
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)

        accuracy = self.accuracy(y_pred, y)
        self.log("val_acc", accuracy, prog_bar=True, on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr = 3e-3)
    
    def on_validation_epoch_end(self):
        avg_loss = self.trainer.logged_metrics['val_loss_epoch']
        avg_acc = self.trainer.logged_metrics['val_acc']

        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', avg_acc, prog_bar=True, on_epoch=True)

        return {'Average Loss:': avg_loss, 
                'Average Accuracy:': avg_acc}



def main():


    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0), 
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.1),
        A.RandomContrast(),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std = [0.229, 0.224, 0.225], 
            max_pixel_value=255.0,
        ), 
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std = [0.229, 0.224, 0.225], 
            max_pixel_value=255.0,
        ), 
        ToTensorV2(),
    ])


    train_loader, val_loader  = get_loader(
        train_dir=TRAIN_IMG_DIR,
        train_mask_dir=TRAIN_MASK_DIR,
        val_dir=VAL_IMG_DIR,
        val_mask_dir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=2)
    
    # Create the model
    model = Segment(in_channels=3, out_channels= 1, encoder='Encoder')
    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
        check_val_every_n_epoch=1,
    )
    trainer.fit(
        model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    state_dict = model.state_dict()
    model_name = 'model.pth'
    torch.save(state_dict, model_name)
    state_dict = model.state_dict()
    model_name = 'model.pth'
    torch.save(state_dict, model_name)

if __name__ == "__main__":
    main()


