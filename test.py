import torch
import torchvision.transforms as transforms
from PIL import Image
from unet import Unet  # Import your U-Net model
import numpy as np
from train import Segment
import albumentations as A 
import cv2

def load_image(image_path):
    image = Image.open(image_path)
    image = transforms.Resize((256, 256))(image)
    image = transforms.ToTensor()(image) 
    return image

def overlay_mask_outline_on_image(image_path, model_path):
    test_image = Image.open(image_path)
    test_image = transforms.Resize((256, 256))(test_image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = Segment(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        test_image_tensor = transforms.ToTensor()(test_image).unsqueeze(0).to(device)
        output = model(test_image_tensor)

    # Apply a threshold to the model's output to get the mask
    threshold = 0.7
    predicted_mask = (output > threshold).float().cpu().numpy()[0, 0]

    # Convert the predicted mask to a binary image (0 and 255 values)
    predicted_mask_binary = (predicted_mask * 255).astype(np.uint8)

    # Resize the binary mask to the original image size
    predicted_mask_resized = cv2.resize(predicted_mask_binary, (test_image.width, test_image.height))

    # Find contours in the binary mask
    contours, _ = cv2.findContours(predicted_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black mask with the same size as the original image
    mask = np.zeros_like(test_image)

    # Draw only the contours on the mask with a thickness of 2
    cv2.drawContours(mask, contours, -1, (0, 255, 0), 2)  # White color for contours, thickness of 2

    # Overlay the outlined mask on the original image
    overlay = cv2.addWeighted(np.array(test_image), 1, mask, 2, 0)

    # Convert the overlay from BGR to RGB
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay_rgb


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
    threshold = 0.5
    predicted_mask = (output > threshold).float()

    # Convert the tensor back to a NumPy array for visualization or further processing
    predicted_mask = predicted_mask.cpu().numpy()

    return predicted_mask

if __name__ == "__main__":
    # Define the paths to the test image and the trained model
    # test_image_path = 'dataset/test/ahmed.jpeg'
    # model_path = 'model.pth'

    # predicted_mask = test_model(test_image_path, model_path)



    # # You can now use the predicted_mask for visualization or further analysis
    # # For example, you can save the predicted mask as an image:
    # predicted_mask_image = Image.fromarray((predicted_mask[0, 0] * 255).astype(np.uint8))
    # predicted_mask_image.save('predicted_mask.jpg')

    test_image_path = 'HumanDataset/test/76_png_jpg.rf.8eb59f3f96b7f582d44e6e21f21a1f97.jpg'
    model_path = 'model.pth'

    overlay = overlay_mask_outline_on_image(test_image_path, model_path)

    # Save or display the overlay image
    cv2.imwrite('overlay.jpg', overlay)

