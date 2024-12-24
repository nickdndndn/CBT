import torch.nn as nn
import math


class Image_Reconstruction(nn.Module):
    def __init__(self, embed_dim, num_feat, num_out_ch, upscale) -> None:
        super().__init__()

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        """Upsample module.

        Args:
            scale (int): Scale factor. Supported scales: 2^n and 3.
            num_feat (int): Channel number of intermediate features.
        """
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super().__init__(*m)
