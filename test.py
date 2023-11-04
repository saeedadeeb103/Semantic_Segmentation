import torch
import torchvision.transforms as transforms
from PIL import Image
from unet import Unet  # Import your U-Net model
import numpy as np
from train import Segment
import albumentations as A 

def load_image(image_path):
    image = Image.open(image_path)
    image = transforms.Resize((256, 256))(image)
    image = transforms.ToTensor()(image) 
    return image

def test_model(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = Segment(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess the test image
    image = load_image(image_path)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    # Post-process the output if needed
    # For binary segmentation, you can apply a threshold to obtain the mask
    threshold = 0.7
    predicted_mask = (output > threshold).float()

    # Convert the tensor back to a NumPy array for visualization or further processing
    predicted_mask = predicted_mask.cpu().numpy()

    return predicted_mask

if __name__ == "__main__":
    # Define the paths to the test image and the trained model
    test_image_path = 'dataset/test/ahmed.jpeg'
    model_path = 'model.pth'

    predicted_mask = test_model(test_image_path, model_path)

    # You can now use the predicted_mask for visualization or further analysis
    # For example, you can save the predicted mask as an image:
    predicted_mask_image = Image.fromarray((predicted_mask[0, 0] * 255).astype(np.uint8))
    predicted_mask_image.save('predicted_mask.jpg')
