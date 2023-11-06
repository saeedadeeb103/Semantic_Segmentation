# unet.py
# This script defines a U-Net architecture for semantic segmentation using PyTorch.

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """
    A block with two sequential convolutional layers with batch normalization and ReLU activation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
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
        Perform a forward pass through the DoubleConv block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after the double convolution operations.
        """
        return self.conv(x)

class Encoder(nn.Module):
    """
    The encoder part of the U-Net architecture.
    
    Args:
        in_channels (int): Number of input channels.
        levels (list): List of integers specifying the number of feature channels at each level.
    """
    def __init__(self, in_channels, levels):
        super(Encoder, self).__init__()
        self.levels = levels
        self.down_sample = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for level in levels:
            self.down_sample.append(DoubleConv(in_channels, level))
            in_channels = level

    def forward(self, x):
        """
        Perform a forward pass through the Encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the encoder.
            list: List of intermediate encoder outputs.
        """
        encoder_output = []
        for down in self.down_sample:
            x = down(x)
            encoder_output.append(x)
            x = self.max_pool(x)
        return x, encoder_output

class Decoder(nn.Module):
    """
    The decoder part of the U-Net architecture.
    
    Args:
        out_channels (int): Number of output channels.
        levels (list): List of integers specifying the number of feature channels at each level.
    """
    def __init__(self, levels):
        super(Decoder, self).__init__()
        self.levels = levels
        self.up_sample = nn.ModuleList()
        for level in levels[::-1]:
            self.up_sample.append(
                nn.ConvTranspose2d(level * 2, level, kernel_size=2, stride=2)
            )
            self.up_sample.append(DoubleConv(level * 2, level))

    def forward(self, x, encoder_output):
        """
        Perform a forward pass through the Decoder.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (list): List of intermediate encoder outputs.

        Returns:
            torch.Tensor: Output tensor from the decoder.
        """
        encoder_output = encoder_output[::-1]
        for idx in range(0, len(self.up_sample), 2):
            x = self.up_sample[idx](x)
            skip = encoder_output[idx // 2]
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])
            concat = torch.cat((skip, x), dim=1)
            x = self.up_sample[idx + 1](concat)
        return x

class Unet(nn.Module):
    """
    U-Net architecture for semantic segmentation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        levels (list): List of integers specifying the number of feature channels at each level.
    """
    def __init__(self, in_channels, out_channels, levels=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.encoder = Encoder(in_channels, levels)
        self.decoder = Decoder(levels)
        self.bottleneck = DoubleConv(levels[-1], levels[-1] * 2)
        self.final_conv = nn.Conv2d(levels[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Perform a forward pass through the U-Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Segmentation output.
        """
        x, encoder_output = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, encoder_output)
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
