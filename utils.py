import torch 
import torchvision
from Dataset import HumanDataset
from torch.utils.data import DataLoader

def get_loader(
        train_dir, 
        train_mask_dir, 
        val_dir, 
        val_mask_dir, 
        batch_size, 
        train_transform, 
        val_transform,
        num_workers=2,
        pin_memory = True,
):
    train_dataset = HumanDataset(image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers= num_workers, pin_memory=pin_memory, shuffle=True)

    val_dataset = HumanDataset(image_dir=val_dir, mask_dir=val_mask_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader 

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y, in loader:
            x = x.to(device)
            y = y.to(device)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds * y ).sum()) / ((preds+y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels *100:.2f}")
    

