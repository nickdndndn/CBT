import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from .utils import retrieve_2d_tuple
from einops import rearrange
from .torch_wavelets import DWT_2D, IDWT_2D

import matplotlib.pyplot as plt
import time

class Deep_Feature_Extractor(nn.Module):
    def __init__(self,
                img_size=64,
                patch_size=1,
                in_chans=3,
                embed_dim=96,
                depths=(6, 6, 6, 6),
                num_heads=(6, 6, 6, 6),
                window_size=7,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm,
                ape=False,
                patch_norm=True,
                upscale=2,
                upsampler='',
                resi_connection='1conv',
                hab_wav = True,
                scbab_wav = True,
                ocab_wav = True,
                ocbab_wav = True,
                 **kwargs):
        super().__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        self.upscale = upscale
        self.upsampler = upsampler

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA',
                             relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA',
                             relative_position_index_OCA)
        # =======
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.pan_patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.mslr_patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.pan_patch_embed.num_patches
        patches_resolution = self.pan_patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.pan_patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.mslr_patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build Multi Band Attention Groups (MBAG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MBAG(
                dim=embed_dim,
                input_resolution=(
                    patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(
                    depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                hab_wav = hab_wav,
                scbab_wav = scbab_wav,
                ocab_wav = ocab_wav,
                ocbab_wav = ocbab_wav
                )
            self.layers.append(layer)

        self.hab_wav = hab_wav

        self.pan_norm = norm_layer(self.num_features)
        self.mslr_norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.pan_conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.mslr_conv_after_body = nn.Conv2d(
                embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.pan_conv_after_body = nn.Identity()
            self.mslr_conv_after_body = nn.Identity()

    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + \
            int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing='ij'))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing='ij'))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        # 2, ws*ws, wse*wse
        relative_coords = coords_ext_flatten[:,
                                             None, :] - coords_ori_flatten[:, :, None]

        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - \
            window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # nw, window_size, window_size, 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def calculate_mask_wav(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = tuple(size // 2 for size in x_size)
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # nw, window_size, window_size, 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, pan, mslr):
        x_size = (pan.shape[2], pan.shape[3])

        # Calculate attention mask and relative position index in advance to speed up inference.
        # The original code is very time-cosuming for large window size.

        attn_mask_wav = self.calculate_mask_wav(x_size).to(pan.device)
        attn_mask = self.calculate_mask(x_size).to(pan.device)
        

        params = {'attn_mask_wav': attn_mask_wav, 'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA,
                  'rpi_oca': self.relative_position_index_OCA}

        pan_forward = self.pan_patch_embed(pan)
        mslr_forward = self.mslr_patch_embed(mslr)
        if self.ape:
            pan_forward = pan_forward + self.absolute_pos_embed
            mslr_forward = mslr_forward + self.absolute_pos_embed
        pan_forward = self.pos_drop(pan_forward)
        mslr_forward = self.pos_drop(mslr_forward)

        for layers in self.layers:
            pan_forward, mslr_forward = layers(
                pan_forward, mslr_forward, x_size, params)

        pan_forward = self.pan_norm(pan_forward)  # b seq_len c
        mslr_forward = self.pan_norm(mslr_forward)

        pan_forward = self.pan_patch_unembed(pan_forward, x_size)
        mslr_forward = self.mslr_patch_unembed(mslr_forward, x_size)

        pan_forward = self.pan_conv_after_body(pan_forward) + pan
        mslr_forward = self.mslr_conv_after_body(mslr_forward) + mslr
        return pan_forward, mslr_forward


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

        From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        """Channel attention used in RCAN.
        Args:
            num_feat (int): Channel number of intermediate features.
            squeeze_factor (int): Channel squeeze factor. Default: 16.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        attn = self.attention(x)
        return x * attn


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        """Channel Attention Block

        Args:
            num_feat (int): Feat number
            compress_ratio (int, optional): Compress ratio. Defaults to 3.
            squeeze_factor (int, optional): Squeeze factor. Defaults to 30.
        """
        super().__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer: nn.Module = nn.GELU, drop=0.):
        """Multi Layer Perceptron

        Args:
            in_features (int): Input features
            hidden_features (int, optional): Hidden features.
            out_features (int, optional): Output features. Defaults to None.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            drop (float, optional): Drop. Defaults to 0..
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size,
               w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """ Window reverse

    Args:
        windows : Windows
        window_size (tuple): Window size
        h : Height
        w : Width

    Returns:
        _type_: _description_
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """ Window based multi-head self attention (W-MSA) module with relative position bias.
        It supports both of shifted and non-shifted window.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # Start measuring time
        #start_time = time.time()


        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n,
                             n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)


        # End measuring time
        #end_time = time.time()

        # Calculate elapsed time
        #elapsed_time = end_time - start_time

        #print("Time taken:", elapsed_time, "seconds")
        
        return x

class WindowAttentionWav(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """ Window based multi-head self attention (W-MSA) module with relative position bias.
        It supports both of shifted and non-shifted window.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1]- 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qk = nn.Linear(dim * 4, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim * 4, dim * 4, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * 4, dim * 4)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # Start measuring time
        #start_time = time.time()

        b_, n, c = x.shape
        qk = self.qk(x).reshape(b_, n, 2, self.num_heads, self.dim //
                                self.num_heads).permute(2, 0, 3, 1, 4) # 1, b_, heads, n, c/heads
        v = self.v(x).reshape(b_, n, 1, self.num_heads, self.dim * 4// 
                               self.num_heads).permute(2, 0, 3, 1, 4) # 2, b_, heads, n, c/heads
        q, k, v = qk[0], qk[1], v[0]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n,
                             n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        ## End measuring time
        #end_time = time.time()

        ## Calculate elapsed time
        #elapsed_time = end_time - start_time#
        #print("Time taken:", elapsed_time, "seconds")

        return x

class CrossBandWindowAttentionWav(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """ Cross Band Multi Headed Self Attention module with relative position bias.
        It supports both of shifted and non-shifted window.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): Window size of window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.q = nn.Linear(dim * 4, dim, bias=qkv_bias)
        self.k = nn.Linear(dim * 4, dim, bias=qkv_bias)
        self.v = nn.Linear(dim * 4, dim * 4, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * 4, dim * 4)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, cross_x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape

        #x = x.view(b_, n, 4, c // 4)
        q = self.q(x).reshape(b_, n, 1, self.num_heads, self.dim //
                              self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(cross_x).reshape(b_, n, 1, self.num_heads, self.dim //
                                      self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(cross_x).reshape(b_, n, 1, self.num_heads, c //
                                      self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], k[0], v[0]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n,
                             n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossBandWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """ Cross Band Multi Headed Self Attention module with relative position bias.
        It supports both of shifted and non-shifted window.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): Window size of window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, cross_x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        q = self.q(x).reshape(b_, n, 1, self.num_heads, c //
                              self.num_heads).permute(2, 0, 3, 1, 4)

        kv = self.kv(cross_x).reshape(b_, n, 2, self.num_heads, c //
                                      self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n,
                             n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAB(nn.Module):
    def __init__(self,
                 dim: int,
                 input_resolution: tuple,
                 num_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 compress_ratio: int = 3,
                 squeeze_factor: int = 30,
                 conv_scale: float = 0.01,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path=0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 wav = True):
        """Hybrid Attention Block

        Args:
            dim (int): Number of input channels
            input_resolution (tuple[int]): Input resolution.
            num_heads (int): Number of attention heads.
            window_size (int, optional): Window size Defaults to 7.
            shift_size (int, optional): Shift size for WindowAttention. Defaults to 0.
            compress_ratio (int, optional): Compression ration. Defaults to 3.
            squeeze_factor (int, optional): Squeeze factor. Defaults to 30.
            conv_scale (float, optional): Conv scale. Defaults to 0.01.
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4..
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (_type_, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        # self.conv_scale = conv_scale
        # self.conv_block = CAB(
        #     num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
            
        self.wav = wav
        if self.wav:
            self.dwt = DWT_2D(wave='haar')
            self.idwt = IDWT_2D(wave='haar')

        if self.wav:
           
            self.attn = WindowAttentionWav(
                dim,
                window_size=retrieve_2d_tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        else:
            self.attn = WindowAttention(
                dim,
                window_size=retrieve_2d_tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)

        

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        assert _ == h * w, "input feature in HAB has wrong size"

        # Residual connection
        shortcut = x # [b, h * w, c]
        
        # Layer Normalization
        x = self.norm1(x) 
        x = x.view(b, h, w, c) # [b, h, w, c]

        # # CAB
        # conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        # conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c) # [b, h * w, c]

        # Optional cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # attn_mask
        else:
            shifted_x = x
            attn_mask = None

        if self.wav:
            shifted_x = self.dwt(shifted_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Partitioning
        x_windows = window_partition(shifted_x, self.window_size)
        if self.wav:
            x_windows = x_windows.view(-1, self.window_size * self.window_size, c * 4) # [crops * b, win * win, c * 4]
        else:
            x_windows = x_windows.view(-1, self.window_size * self.window_size, c) # [crops * b, win * win, c]


        # (S)W-MSA
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        # merge windows
        if self.wav:
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c * 4)
            shifted_x = window_reverse(attn_windows, self.window_size, h // 2, w // 2).permute(0, 3, 1, 2)  # b, c, h, w
        else:
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
            shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        if self.wav:
            #IWT
            shifted_x = self.idwt(shifted_x).permute(0, 2, 3, 1) # b, h, w, c
        
        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # Addition
        x = shortcut + self.drop_path(attn_x) #+ conv_x * self.conv_scale
        
        # LN and MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SCBAB(nn.Module):
    def __init__(self,
                 dim: int,
                 input_resolution: tuple,
                 num_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 compress_ratio: int = 3,
                 squeeze_factor: int = 30,
                 conv_scale: float = 0.01,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 wav = True
                 ):
        """ Shifted Cross Band Attention Block

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resolution.
            num_heads (int): Number of attention heads.
            window_size (int, optional): Window size. Defaults to 7.
            shift_size (int, optional): Shift size for CrossBandWindowAttention. Defaults to 0.
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4..
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

    
        self.norm1 = norm_layer(dim)
        self.norm_cross = norm_layer(dim)

        
        self.wav = wav

        if self.wav:
            self.conv_scale = conv_scale
            # self.conv_block = CAB(
            #     num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
            
            self.dwt = DWT_2D(wave='haar')
            self.idwt = IDWT_2D(wave='haar')
            

        if self.wav:
            self.cross_band_attn = CrossBandWindowAttentionWav(
                dim,
                window_size=retrieve_2d_tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        else:
            self.cross_band_attn = CrossBandWindowAttention(
                dim,
                window_size=retrieve_2d_tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, cross_x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape

        # Residual connection
        shortcut = x
        x = self.norm1(x)
        cross_x = self.norm_cross(cross_x)
        x = x.view(b, h, w, c)
        cross_x = cross_x.view(b, h, w, c)

        #if self.wav:
            # CAB
            # conv_x = self.conv_block(x.permute(0, 3, 1, 2))
            # conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c) # [b, h * w, c]

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
            shifted_cross_x = torch.roll(
                cross_x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            shifted_cross_x = cross_x
            attn_mask = None

        if self.wav:
            # DWT
            shifted_x = self.dwt(shifted_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            shifted_cross_x = self.dwt(shifted_cross_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
        if self.wav:
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size) 
            x_windows = x_windows.view(-1, self.window_size * self.window_size, c * 4)
            
            cross_x_windows = window_partition(shifted_cross_x, self.window_size)
            cross_x_windows = cross_x_windows.view(-1, self.window_size * self.window_size, c * 4)
        else:
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
            
            cross_x_windows = window_partition(shifted_cross_x, self.window_size)
            cross_x_windows = cross_x_windows.view(-1, self.window_size * self.window_size, c)

        attn_windows = self.cross_band_attn(
            x_windows, cross_x_windows, rpi=rpi_sa, mask=attn_mask)

        if self.wav:
            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size , self.window_size, c * 4)
            shifted_x = window_reverse(attn_windows, self.window_size, h // 2, w // 2).permute(0, 3, 1, 2) # b, c, h, w
        else:
            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
            shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        if self.wav:
            #IWT
            shifted_x = self.idwt(shifted_x).permute(0, 2, 3, 1)

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)


        # FFN
        if self.wav:
            x = shortcut + attn_x #+ conv_x * self.conv_scale
            x = x + self.mlp(self.norm2(x))
            return x
        else:
            x = shortcut + attn_x
            x = x + self.mlp(self.norm2(x))
            return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        """ Patch Merging Layer

        Args:
            input_resolution (tuple[int]): Input Resolution
            dim (int): Number of input channels
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x


class OCAB(nn.Module):
    def __init__(self, dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 compress_ratio: int = 3,
                 squeeze_factor: int = 30,
                 conv_scale: float = 0.01,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=2,
                 norm_layer=nn.LayerNorm,
                 wav = True
                 ):
        """Overlapping Cross Attention Block

        Args:
            dim (int): Input dimension
            input_resolution (tuple): Resolution of input 
            window_size (int): Window size
            overlap_ratio (float): Overlapping ration
            num_heads (int): Number of Attention heads
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set.
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        
        self.norm1 = norm_layer(dim)

        self.wav = wav
        if self.wav:
            self.conv_scale = conv_scale
            # self.conv_block = CAB(
            #     num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
            

            self.dwt = DWT_2D(wave='haar')
            self.idwt = IDWT_2D(wave='haar')
            
            self.qk = nn.Linear(dim * 4, dim * 2, bias=qkv_bias)
            self.v = nn.Linear(dim * 4, dim * 4, bias=qkv_bias)

            self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size),
                                stride=window_size, padding=(self.overlap_win_size - window_size))
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size),
                                    stride=window_size, padding=(self.overlap_win_size - window_size) // 2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        if self.wav:
            # CAB
            #conv_x = self.conv_block(x.permute(0, 3, 1, 2))
            #conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c) # [b, h * w, c]

            # DWT
            x_dwt = self.dwt(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            assert x_dwt.shape == (b, h // 2, w // 2, c * 4)
            _, h_, w_, c_ = x_dwt.shape

            # Linear Mapping
            qk = self.qk(x_dwt).reshape(b, h_, w_, 2, self.dim).permute(3, 0, 4, 1, 2) # 2, b, c, h, w
            v = self.v(x_dwt).reshape(b, h_, w_, 1, c_).permute(3, 0, 4, 1, 2) # 1, b, c, h, w
            
            q = qk[0].permute(0, 2, 3, 1)  # b, h, w, c
            k = qk[1]
            v = v[0]

            # partition windows
            q_windows = window_partition(q, self.window_size) # nw*b, window_size, window_size, c
            q_windows = q_windows.view(-1, self.window_size * self.window_size, c) # nw*b, window_size*window_size, c

            k_windows = self.unfold(k)
            k_windows = rearrange(k_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=1,
                                ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() 
            v_windows = self.unfold(v)
            v_windows = rearrange(v_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=1,
                                ch=c_, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
            
            k_windows, v_windows = k_windows[0], v_windows[0]  # nw*b, ow*ow, c

            b_, nq, _ = q_windows.shape
            _, n, _ = k_windows.shape
            q = q_windows.reshape(b_, nq, self.num_heads, self.dim // self.num_heads).permute(
                0, 2, 1, 3)  # nw*b, nH, nq, d
            k = k_windows.reshape(b_, n, self.num_heads, self.dim // self.num_heads).permute(
                0, 2, 1, 3)  # nw*b, nH, n, d
            v = v_windows.reshape(b_, n, self.num_heads, c_ // self.num_heads).permute(
                0, 2, 1, 3)  # nw*b, nH, n, d

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
            attn = attn + relative_position_bias.unsqueeze(0)

            attn = self.softmax(attn)
            attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, c_)

            # merge windows
            attn_windows = attn_windows.view(-1,
                                            self.window_size, self.window_size, c_)
            x = window_reverse(attn_windows, self.window_size, h // 2, w // 2).permute(0, 3, 1, 2)
            assert x.shape == (b,  c * 4, h // 2,  w // 2)
            
            x_idwt = self.idwt(x).permute(0, 2, 3, 1)
            
            x = x_idwt.view(b, h * w, self.dim)

            x = self.proj(x) + shortcut #+ conv_x * self.conv_scale
            x = x + self.mlp(self.norm2(x))
            return x
        else:
            qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(
                3, 0, 4, 1, 2)  # 3, b, c, h, w
            q = qkv[0].permute(0, 2, 3, 1)  # b, h, w, c
            kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2*c, h, w

            # partition windows
            # nw*b, window_size, window_size, c
            q_windows = window_partition(q, self.window_size)
            # nw*b, window_size*window_size, c
            q_windows = q_windows.view(-1, self.window_size * self.window_size, c)

            kv_windows = self.unfold(kv)  # b, c*w*w, nw
            kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2,
                                ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()  # 2, nw*b, ow*ow, c
            k_windows, v_windows = kv_windows[0], kv_windows[1]  # nw*b, ow*ow, c

            b_, nq, _ = q_windows.shape
            _, n, _ = k_windows.shape
            d = self.dim // self.num_heads
            q = q_windows.reshape(b_, nq, self.num_heads, d).permute(
                0, 2, 1, 3)  # nw*b, nH, nq, d
            k = k_windows.reshape(b_, n, self.num_heads, d).permute(
                0, 2, 1, 3)  # nw*b, nH, n, d
            v = v_windows.reshape(b_, n, self.num_heads, d).permute(
                0, 2, 1, 3)  # nw*b, nH, n, d

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
            attn = attn + relative_position_bias.unsqueeze(0)

            attn = self.softmax(attn)
            attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

            # merge windows
            attn_windows = attn_windows.view(-1,
                                            self.window_size, self.window_size, self.dim)
            x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
            x = x.view(b, h * w, self.dim)

            x = self.proj(x) + shortcut
            x = x + self.mlp(self.norm2(x))
            return x
                


class OCBAB(nn.Module):
    def __init__(self, dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 compress_ratio: int = 3,
                 squeeze_factor: int = 30,
                 conv_scale: float = 0.01,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=2,
                 norm_layer=nn.LayerNorm,
                 wav = True
                 ):
        """ Overlapping Cross Band Attention Block

        Args:
            dim (int): Input dimension
            input_resolution (tuple): Resolution of input 
            window_size (int): Window size
            overlap_ratio (float): Overlapping ration
            num_heads (int): Number of Attention heads
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set.
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.norm_cross = norm_layer(dim)

        
        self.wav = wav
        if self.wav:
            self.conv_scale = conv_scale
            # self.conv_block = CAB(
            #     num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
            
            self.dwt = DWT_2D(wave='haar')
            self.idwt = IDWT_2D(wave='haar')

        if self.wav:
            self.q = nn.Linear(dim * 4, dim, bias=qkv_bias)
            self.k = nn.Linear(dim * 4, dim, bias=qkv_bias)
            self.v = nn.Linear(dim * 4, dim  * 4, bias=qkv_bias)
            self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size),
                                stride=window_size, padding=(self.overlap_win_size) - (window_size))
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size),
                                    stride=window_size, padding=(self.overlap_win_size - window_size) // 2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, cross_x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        cross_x = self.norm_cross(cross_x)
        cross_x = cross_x.view(b, h, w, c)

        if self.wav:
            # CAB
            # conv_x = self.conv_block(x.permute(0, 3, 1, 2))
            # conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c) # [b, h * w, c]

            # WT
            x_dwt = self.dwt(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            assert x_dwt.shape == (b, h // 2, w // 2, c * 4)
            x_dwt_cross = self.dwt(cross_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            assert x_dwt_cross.shape == (b, h // 2, w // 2, c * 4)
            
            # Linear Mapping
            q = self.q(x_dwt).reshape(b, h // 2, w // 2, 1, c).permute(
                3, 0, 4, 1, 2)  # 1, b, c, h, w
            k = self.k(x_dwt).reshape(b, h // 2, w // 2, 1, c).permute(
                3, 0, 4, 1, 2)  # 1, b, c, h, w
            v = self.v(x_dwt).reshape(b, h // 2, w // 2, 1, c * 4).permute(
                3, 0, 4, 1, 2)  # 1, b, c, h, w
            
            q = q.squeeze(0).permute(0, 2, 3, 1)  # b, h, w, c
            k = k.squeeze(0)
            v = v.squeeze(0)

            # partition windows
            q_windows = window_partition(q, self.window_size)
            # nw*b, window_size*window_size, c
            q_windows = q_windows.view(-1, (self.window_size * self.window_size), c)

            k_windows = self.unfold(k)
            k_windows = rearrange(k_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=1,
                                ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() 
            v_windows = self.unfold(v)
            v_windows = rearrange(v_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=1,
                                ch=c * 4, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
            
            k_windows, v_windows = k_windows[0], v_windows[0]  # nw*b, ow*ow, c


            b_, nq, _ = q_windows.shape
            _, n, _ = k_windows.shape
            q = q_windows.reshape(b_, nq, self.num_heads, self.dim // self.num_heads).permute(
                0, 2, 1, 3)  # nw*b, nH, nq, d
            k = k_windows.reshape(b_, n, self.num_heads, self.dim // self.num_heads).permute(
                0, 2, 1, 3)  # nw*b, nH, n, d
            v = v_windows.reshape(b_, n, self.num_heads, (self.dim * 4) // self.num_heads).permute(
                0, 2, 1, 3)  # nw*b, nH, n, d

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                (self.window_size) * (self.window_size), (self.overlap_win_size) * (self.overlap_win_size), -1)  # ws*ws, wse*wse, nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
            attn = attn + relative_position_bias.unsqueeze(0)

            attn = self.softmax(attn)
            attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim * 4)

            # merge windows
            attn_windows = attn_windows.view(-1,
                                            self.window_size, self.window_size, self.dim * 4)
            x = window_reverse(attn_windows, self.window_size, h // 2, w // 2).permute(0, 3, 1, 2)  # b h w c
            assert x.shape == (b,  c * 4, h // 2,  w // 2)

            x_idwt = self.idwt(x).permute(0, 2, 3, 1)     
            
            x = x_idwt.view(b, h * w, self.dim)

            if self.wav:
                x = self.proj(x) + shortcut #+ conv_x * self.conv_scale
                x = x + self.mlp(self.norm2(x))
                return x
            else:
                x = self.proj(x) + shortcut
                x = x + self.mlp(self.norm2(x))
                return x
        else:
        
            q = self.q(x).reshape(b, h, w, 1, c).permute(
                3, 0, 4, 1, 2)  # 1, b, c, h, w
            kv = self.kv(cross_x).reshape(b, h, w, 2, c).permute(
                3, 0, 4, 1, 2)  # 2, b, c, h, w
            q = q.squeeze(0).permute(0, 2, 3, 1)  # b, h, w, c
            kv = torch.cat((kv[0], kv[1]), dim=1)

            q = q * self.scale  # partition windows
            # nw*b, window_size, window_size, c
            q_windows = window_partition(q, self.window_size)
            # nw*b, window_size*window_size, c
            q_windows = q_windows.view(-1, self.window_size * self.window_size, c)

            kv_windows = self.unfold(kv)  # b, c*w*w, nw
            kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2,
                                ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()  # 2, nw*b, ow*ow, c
            k_windows, v_windows = kv_windows[0], kv_windows[1]  # nw*b, ow*ow, c

            b_, nq, _ = q_windows.shape
            _, n, _ = k_windows.shape
            d = self.dim // self.num_heads
            q = q_windows.reshape(b_, nq, self.num_heads, d).permute(
                0, 2, 1, 3)  # nw*b, nH, nq, d
            k = k_windows.reshape(b_, n, self.num_heads, d).permute(
                0, 2, 1, 3)  # nw*b, nH, n, d
            v = v_windows.reshape(b_, n, self.num_heads, d).permute(
                0, 2, 1, 3)  # nw*b, nH, n, d

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
            attn = attn + relative_position_bias.unsqueeze(0)

            attn = self.softmax(attn)
            attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

            # merge windows
            attn_windows = attn_windows.view(-1,
                                            self.window_size, self.window_size, self.dim)
            x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
            x = x.view(b, h * w, self.dim)

            x = self.proj(x) + shortcut

            x = x + self.mlp(self.norm2(x))
            return x


class MBAG(nn.Module):
    """Multi Band Attention Group (MBAG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                dim,
                input_resolution,
                depth,
                num_heads,
                window_size,
                compress_ratio,
                squeeze_factor,
                conv_scale,
                overlap_ratio,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                downsample=None,
                img_size=224,
                patch_size=4,
                resi_connection='1conv',
                hab_wav = True,
                scbab_wav = True,
                ocab_wav = True,
                ocbab_wav = True,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.hab_wav = hab_wav
        self.scbab_wav = scbab_wav

        # HAB blocks
        self.pan_hab_blocks = nn.ModuleList([
            HAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer,
                wav = hab_wav) for i in range(depth)
        ])
        self.mslr_hab_blocks = nn.ModuleList([
            HAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer,
                wav = hab_wav) for i in range(depth)
        ])

        #comment for ablation study
        # SCBAB blocks
        self.pan_scbab_blocks = nn.ModuleList([
            SCBAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                wav = scbab_wav) for i in range(depth)
        ])
        self.mslr_scbab_blocks = nn.ModuleList([
            SCBAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                wav = scbab_wav) for i in range(depth)
        ])

        # OCAB block
        self.pan_overlap_attn = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            wav = ocab_wav
        )
        self.mslr_overlap_attn = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            wav = ocab_wav
        )

        # OCBAB block
        self.pan_cbab = OCBAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            num_heads=num_heads,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer,
            wav = ocbab_wav
            )
        self.mslr_cbab = OCBAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            num_heads=num_heads,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer,
            wav = ocbab_wav
            )

        # patch merging layer
        if downsample is not None:
            self.pan_downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
            self.mslr_downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.pan_downsample = None
            self.mslr_downsample = None

        if resi_connection == '1conv':
            self.pan_conv = nn.Conv2d(dim, dim, 3, 1, 1)
            self.mslr_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.pan_conv = nn.Identity()
            self.mslr_conv = nn.Identity()

        self.pan_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.mslr_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.pan_patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.mslr_patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, pan, mslr, x_size, params):

        pan_forward = pan
        mslr_forward = mslr

        if self.hab_wav:
            # Multiple HAB
            for pan_blk, mslr_blk in zip(self.pan_hab_blocks, self.mslr_hab_blocks):
                pan_forward = pan_blk(pan_forward, x_size, params['rpi_sa'], params['attn_mask_wav'])
                mslr_forward = mslr_blk( mslr_forward, x_size, params['rpi_sa'], params['attn_mask_wav'])
        else:
            # Multiple HAB
            for pan_blk, mslr_blk in zip(self.pan_hab_blocks, self.mslr_hab_blocks):
                pan_forward = pan_blk(pan_forward, x_size, params['rpi_sa'], params['attn_mask'])
                mslr_forward = mslr_blk( mslr_forward, x_size, params['rpi_sa'], params['attn_mask'])
        
        if self.scbab_wav:
        # Multiple SCBAB
            for pan_blk, mslr_blk in zip(self.pan_scbab_blocks, self.mslr_scbab_blocks):
                pan_forward_temp = pan_blk(pan_forward, mslr_forward, x_size, params['rpi_sa'], params['attn_mask_wav'])
                mslr_forward_temp = mslr_blk(mslr_forward, pan_forward, x_size, params['rpi_sa'], params['attn_mask_wav'])
        else:
            for pan_blk, mslr_blk in zip(self.pan_scbab_blocks, self.mslr_scbab_blocks):
                pan_forward_temp = pan_blk(pan_forward, mslr_forward, x_size, params['rpi_sa'], params['attn_mask'])
                mslr_forward_temp = mslr_blk(mslr_forward, pan_forward, x_size, params['rpi_sa'], params['attn_mask'])

        pan_forward = pan_forward_temp
        mslr_forward = mslr_forward_temp

        # OCAB
        pan_forward = self.pan_overlap_attn(
            pan_forward, x_size, params['rpi_oca'])
        mslr_forward = self.mslr_overlap_attn(
            mslr_forward, x_size, params['rpi_oca'])
        
        # OCBAB
        pan_forward_ = self.pan_cbab(
            pan_forward, mslr_forward, x_size, params['rpi_oca'])
        mslr_forward_ = self.mslr_cbab(
            mslr_forward, pan_forward, x_size, params['rpi_oca'])

        if self.pan_downsample is not None:
            pan_forward_ = self.pan_downsample(pan_forward_)
            mslr_forward_ = self.mslr_downsample(mslr_forward_)

        pan_forward = self.pan_patch_unembed(pan_forward_, x_size)
        mslr_forward = self.mslr_patch_unembed(mslr_forward_, x_size)
        pan_forward = self.pan_conv(pan_forward)
        mslr_forward = self.mslr_conv(mslr_forward)
        pan_forward = self.pan_patch_embed(pan_forward) + pan
        mslr_forward = self.mslr_patch_embed(mslr_forward) + mslr

        return pan_forward, mslr_forward


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        """ Image to Patch Embedding

        Args:
            img_size (int): Image size.  Default: 224.
            patch_size (int): Patch token size. Default: 4.
            in_chans (int): Number of input image channels. Default: 3.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            norm_layer (nn.Module, optional): Normalization layer. Default: None
        """
        super().__init__()
        img_size = retrieve_2d_tuple(img_size)
        patch_size = retrieve_2d_tuple(patch_size)
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        """ Image to Patch Unembedding

        Args:
            img_size (int): Image size.  Default: 224.
            patch_size (int): Patch token size. Default: 4.
            in_chans (int): Number of input image channels. Default: 3.
            embed_dim (int): Number of linear projection output channels. Default: 96.
            norm_layer (nn.Module, optional): Normalization layer. Default: None
        """
        super().__init__()
        img_size = retrieve_2d_tuple(img_size)
        patch_size = retrieve_2d_tuple(patch_size)
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x
