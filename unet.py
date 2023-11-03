# unet.py
# This script defines a U-Net architecture for semantic segmentation using PyTorch.

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """
    DoubleConv defines a double convolution block used in the U-Net architecture.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        conv (nn.Sequential): A sequence of two 2D convolutional layers with batch normalization and ReLU activation.

    Methods:
        forward(x): Forward pass through the DoubleConv block.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through the DoubleConv block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the double convolution operations.
        """
        return self.conv(x)

class Unet(nn.Module):
    """
    Unet defines a U-Net architecture for semantic segmentation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        levels (list): List of integers specifying the number of feature channels at each level of the U-Net.

    Attributes:
        levels (list): List of integers specifying the number of feature channels at each level of the U-Net.
        down_sample (nn.ModuleList): List of DoubleConv blocks for downsampling.
        up_sample (nn.ModuleList): List of convolution and DoubleConv blocks for upsampling.
        out_channels (int): Number of output channels.
        max_pool (nn.MaxPool2d): Max pooling layer for downsampling.
        bottleneck (DoubleConv): DoubleConv block at the bottleneck of the U-Net.
        final_conv (nn.Conv2d): Final convolutional layer for producing segmentation output.

    Methods:
        forward(x): Forward pass through the U-Net model.
    """
    def __init__(self, in_channels, out_channels, levels=[64, 128, 256, 512]):
        super(Unet, self).__init()

        self.levels = levels
        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        self.out_channels = out_channels
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for level in levels:
            self.down_sample.append(DoubleConv(in_channels, level))
            in_channels = level

        for level in levels[::-1]:
            self.up_sample.append(
                nn.ConvTranspose2d(
                    level * 2, level, kernel_size=2, stride=2,
                )
            )
            self.up_sample.append(DoubleConv(level * 2, level))
        self.bottleneck = DoubleConv(levels[-1], levels[-1] * 2)
        self.final_conv = nn.Conv2d(levels[0], self.out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the U-Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Segmentation output.
        """
        encoder_output = []

        for down in self.down_sample:
            x = down(x)
            encoder_output.append(x)
            x = self.max_pool(x)

        x = self.bottleneck(x)
        encoder_output = encoder_output[::-1]
        for idx in range(0, len(self.up_sample), 2):
            x = self.up_sample[idx](x)
            skip = encoder_output[idx // 2]
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])

            concat = torch.cat((skip, x), dim=1)
            x = self.up_sample[idx + 1](concat)

        return self.final_conv(x)

def test():
    """
    Function to test the U-Net model with random input data.

    Prints the model architecture and the segmentation prediction.
    """
    x = torch.randn((3, 1, 161, 161))
    model = Unet(in_channels=1, out_channels=1)
    print(model)
    prediction = model(x)
    print(prediction)

if __name__ == "__main__":
    test()
