import torch.nn as nn
from .image_reconstructor import *


class Shallow_Feature_Extractor(nn.Module):
    def __init__(self, num_in_ch: int, embed_dim: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, upsample: bool = False, pan_low_size_ratio: int = None):
        """_summary_

        Args:
            num_in_ch (int): Number of input channels
            embed_dim (int): Dimension of embedding
            kernel_size (int, optional): Kernel size. Defaults to 3.
            stride (int, optional): . Stride size. Defaults to 1.
            padding (int, optional): Padding size. Defaults to 1.
            upsample (bool, optional): Upsample input image. Defaults to False.
            pan_low_size_ratio (int, optional): Ratio between Pan and LR. Defaults to None.
        """
        super().__init__()
        self.upsample = upsample
        if self.upsample == True:
            self.upsampler = Image_Reconstruction(
                num_in_ch, embed_dim, embed_dim, pan_low_size_ratio)
        else:
            self.conv_first = nn.Conv2d(
                num_in_ch, embed_dim, kernel_size, stride, padding)

    def forward(self, x):
        if self.upsample == True:
            x = self.upsampler(x)
        else:
            x = self.conv_first(x)
        return x
