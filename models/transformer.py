import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=F.gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------PatchEmbed---------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=(384, 512), patch_size=(4, 4), in_chans=3, embed_dim=48, norm_layer=nn.LayerNorm,
                 activation=F.gelu):
        """
        Image to Patch Embedding
        :param img_size:  Image size.  Default: 384 x 512.
        :param patch_size:  Patch token size. Default: [4, 4].
        :param in_chans:  Number of input image channels. Default: 3.
        :param embed_dim: Number of linear projection output channels. Default: 48.
        :param norm_layer: ormalization layer. Default: None
        """
        super().__init__()
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # 直接大kernel卷, 粗暴嗷

        self.activation = activation

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], "Input image size doesn't match model input requirement."
        x = self.layers(x)

        # N,C,PH,PW -> N,C,PH*PW -> N, PH*PW, C
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------WindowAttention---------------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        Window based multi-head self attention (W-MSA) module with relative position bias. It supports both of shifted
        and non-shifted window.
        :param dim: Number of input channels.
        :param window_size: The height and width of the window.
        :param num_heads: Number of attention heads.
        :param qkv_bias: If True, add a learnable bias to query, key, value. Default: True
        :param qk_scale: Override default qk scale of head_dim ** -0.5 if set
        :param attn_drop: Dropout ratio of attention weight. Default: 0.0
        :param proj_drop:  Dropout ratio of output. Default: 0.0
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.computeq = nn.Linear(dim, dim, bias=qkv_bias)
        self.computekv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, local_x, mask=None):
        """
        :param x: input features with shape of (num_windows*B, N, C)
        :param mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.computeq(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.computekv(local_x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        x = self.pointwise(x)
        x = self.activation(self.bn2(x))
        return x


class Local_InfExtractor(nn.Module):
    def __init__(self, dim, input_resolution, compre=2, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.compre = compre
        channel = dim // compre
        self.norm = nn.BatchNorm2d(dim)
        self.compress_layer = nn.Conv2d(dim, channel, kernel_size=1, bias=False)
        self.stage1 = SeparableConv2d(channel, channel, kernel_size=3, stride=1, padding=dilation[0],
                                      dilation=dilation[0])
        self.stage2 = SeparableConv2d(channel * 2, channel, kernel_size=3, stride=1, padding=dilation[1],
                                      dilation=dilation[1])
        self.stage3 = SeparableConv2d(channel * 3, channel, kernel_size=3, stride=1, padding=dilation[2],
                                      dilation=dilation[2])
        self.uplayer = nn.Conv2d(channel * 4, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.permute(0, 3, 1, 2)
        x0 = self.compress_layer(x)
        x1 = self.stage1(x0)
        x2 = torch.cat([x1, x0], dim=1)
        x2 = self.stage2(x2)
        x3 = torch.cat([x2, x1, x0], dim=1)
        x3 = self.stage3(x3)
        out = torch.cat([x3, x2, x1, x0], dim=1)
        out = self.uplayer(out)
        out = self.norm(out)
        out = out.permute(0, 2, 3, 1)

        return out


# --------------------------------Swin Transformer Block------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=F.gelu, norm_layer=nn.LayerNorm):
        """
        Introducion: Swin Transformer Block.
        :param dim: Number of input channels.
        :param input_resolution:  Input resulotion.
        :param num_heads: Number of attention heads.
        :param window_size: Window size.
        :param shift_size: Shift size for SW-MSA.
        :param mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        :param qkv_bias: If True, add a learnable bias to query, key, value. Default: True
        :param qk_scale: Override default qk scale of head_dim ** -0.5 if set.
        :param drop:  Dropout rate. Default: 0.0
        :param attn_drop: Attention dropout rate. Default: 0.0
        :param drop_path:  Stochastic depth rate. Default: 0.0
        :param act_layer:  Activation layer. Default: nn.GELU
        :param norm_layer:  Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) < self.window_size[0]:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size[0] = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size[0], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.local_mscale_extractor = Local_InfExtractor(dim, input_resolution)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        local_shifted_x = self.local_mscale_extractor(shifted_x)

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        local_x_windows = window_partition(local_shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        local_x_windows = local_x_windows.view(-1, self.window_size[0] * self.window_size[1],
                                               C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, local_x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            local_x = torch.roll(local_shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            local_x = local_shifted_x
        x = x.view(B, H * W, C)
        local_x = local_x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x) + local_x
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ---------------------------Patch Merging-----------------------------------------
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        """
        Patch Merging Layer
        :param input_resolution:   (tuple[int])Resolution of input feature.
        :param dim:  Number of input channels.
        :param norm_layer:  (nn.Module, optional) Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, "x size are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# ---------------------------------------seqtoimg-------------------------------
def Seqtoimg(input_resolution, x):
    B, L, C = x.shape[0], x.shape[1], x.shape[2]
    H, W = input_resolution[0], input_resolution[1]
    x = x.view(B, H, W, C)
    x = x.permute(0, 3, 1, 2)
    return x


def ImgtoSeq(x):
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    x = x.view(N, C, H * W)
    x = x.permute(0, 2, 1)
    return x


class Expanding_layer(nn.Module):
    def __init__(self, input_resolution, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 groups=1, bias=True, up_factor=2):
        super().__init__()
        self.factor = up_factor
        self.H, self.W = input_resolution[0], input_resolution[1]
        self.upsample_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                                 output_padding=output_padding,
                                                 groups=groups, bias=bias)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.H, self.W, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.upsample_layer(x)
        x = x.view(B, -1, self.H * self.W * self.factor * self.factor)
        x = x.permute(0, 2, 1)
        return x


def window_partition(x, window_size):
    """
    :param x: (B, H, W, C)
    :param window_size: (int )window size
    :return: windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


# --------------------Window reverse---------------------------
def window_reverse(windows, window_size, H, W):
    """
    :param windows: (num_windows*B, window_size, window_size, C)
    :param window_size:  (int) Window size
    :param H: Height of image
    :param W: Width of image
    :return: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ------------------------MyTransformer--------------------------------
class MS_UNet(nn.Module):
    def __init__(self, img_size=(384, 512), patch_size=(4, 4), in_chans=3, embed_dim=48, num_heads=[3, 6, 12, 24],
                 window_size=(6, 8), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True, **kwargs):
        """
        Learning the flow map which is utilized for Distorted Image Rectification
        :param img_size: (int | tuple(int)) Input image size. Default: 384 x 512
        :param patch_size:  (int | tuple(int)) Patch size. Default: 4 x 4
        :param in_chans: (int) Number of input image channels. Default: 3
        :param embed_dim: (int) Patch embedding dimension. Default: 48
        :param num_heads:  (int) Patch embedding dimension. Default: 48
        :param window_size: (int) Window size. Default: (12,16)
        :param mlp_ratio:  (int) Ratio of mlp hidden dim to embedding dim. Default: 4
        :param qkv_bias: (bool) If True, add a learnable bias to query, key, value. Default: True
        :param qk_scale: (float) Override default qk scale of head_dim ** -0.5 if set. Default: None
        :param drop_rate: (float) Dropout rate. Default: 0
        :param norm_layer:  (nn.Module) Normalization layer. Default: nn.LayerNorm.
        :param ape: (bool) If True, add absolute position embedding to the patch embedding. Default: False
        :param patch_norm: (bool) If True, add normalization after patch embedding. Default: True
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.pos_drop = nn.Dropout(p=drop_rate)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        # absolute position embedding
        if self.ape:
            # Gaussian distribution: mean = 0, delta = 1, Now: mean =
            parameter = torch.randn(1, self.patch_embed.num_patches, embed_dim)
            parameter = (parameter - parameter.mean()) / parameter.std()
            parameter = parameter * 0.02
            self.absolute_pos_embed = nn.Parameter(parameter)

        # The core structure of our Transformer
        # Stage 0
        input_resolution0 = patches_resolution
        self.stage00 = TransformerBlock(embed_dim, input_resolution0, num_heads=3, window_size=window_size,
                                        shift_size=0,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)
        self.stage01 = TransformerBlock(embed_dim, input_resolution0, num_heads=3, window_size=window_size,
                                        shift_size=3,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)

        # Stage 1
        self.patch_merge1 = PatchMerging(input_resolution0, embed_dim, norm_layer=nn.LayerNorm)
        self.input_resolution1 = (input_resolution0[0] // 2, input_resolution0[1] // 2)
        self.stage10 = TransformerBlock(embed_dim * 2, self.input_resolution1, num_heads=6, window_size=window_size,
                                        shift_size=0,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)
        self.stage11 = TransformerBlock(embed_dim * 2, self.input_resolution1, num_heads=6, window_size=window_size,
                                        shift_size=3,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)
        # Stage 2
        self.patch_merge2 = PatchMerging(self.input_resolution1, embed_dim * 2, norm_layer=nn.LayerNorm)
        self.input_resolution2 = (self.input_resolution1[0] // 2, self.input_resolution1[1] // 2)
        self.stage20 = TransformerBlock(embed_dim * 4, self.input_resolution2, num_heads=12, window_size=window_size,
                                        shift_size=0,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)
        self.stage21 = TransformerBlock(embed_dim * 4, self.input_resolution2, num_heads=12, window_size=window_size,
                                        shift_size=3,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)
        # Stage 3
        self.patch_merge3 = PatchMerging(self.input_resolution2, embed_dim * 4, norm_layer=nn.LayerNorm)
        self.input_resolution3 = (self.input_resolution2[0] // 2, self.input_resolution2[1] // 2)
        self.stage30 = TransformerBlock(embed_dim * 8, self.input_resolution3, num_heads=24, window_size=window_size,
                                        shift_size=0,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)
        self.stage31 = TransformerBlock(embed_dim * 8, self.input_resolution3, num_heads=24, window_size=window_size,
                                        shift_size=3,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)
        # Usample stage4
        self.upsample_layer4 = Expanding_layer(self.input_resolution3, embed_dim * 8, embed_dim * 4, 2, stride=2)
        self.input_resolution4 = (self.input_resolution3[0] * 2, self.input_resolution3[1] * 2)
        self.stage40 = TransformerBlock(embed_dim * 4, self.input_resolution4, num_heads=12, window_size=window_size,
                                        shift_size=0,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)
        self.stage41 = TransformerBlock(embed_dim * 4, self.input_resolution4, num_heads=12, window_size=window_size,
                                        shift_size=3,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)

        # Usample stage5
        self.upsample_layer5 = Expanding_layer(self.input_resolution4, embed_dim * 4, embed_dim * 2, 2, stride=2)
        self.input_resolution5 = (self.input_resolution4[0] * 2, self.input_resolution4[1] * 2)
        self.stage50 = TransformerBlock(embed_dim * 2, self.input_resolution5, num_heads=6, window_size=window_size,
                                        shift_size=0,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)
        self.stage51 = TransformerBlock(embed_dim * 2, self.input_resolution5, num_heads=6, window_size=window_size,
                                        shift_size=3,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)

        # Usample stage6
        self.upsample_layer6 = Expanding_layer(self.input_resolution5, embed_dim * 2, embed_dim, 2, stride=2)
        self.input_resolution6 = (self.input_resolution5[0] * 2, self.input_resolution5[1] * 2)
        self.stage60 = TransformerBlock(embed_dim, self.input_resolution6, num_heads=3, window_size=window_size,
                                        shift_size=0,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)
        self.stage61 = TransformerBlock(embed_dim, self.input_resolution6, num_heads=3, window_size=window_size,
                                        shift_size=3,
                                        mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0., attn_drop=0.,
                                        drop_path=0.,
                                        act_layer=F.gelu, norm_layer=nn.LayerNorm)

        self.fusion_layer1 = nn.Conv1d(embed_dim * 8, embed_dim * 4, kernel_size=9, stride=1, padding=9 // 2)
        self.fusion_layer2 = nn.Conv1d(embed_dim * 4, embed_dim * 2, kernel_size=9, stride=1, padding=9 // 2)
        self.fusion_layer3 = nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=9, stride=1, padding=9 // 2)

        # Regression layer
        self.regress1 = nn.ConvTranspose2d(embed_dim, out_channels=embed_dim // 2, kernel_size=2, stride=2)
        self.regress2 = nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1)
        self.regress3 = nn.ConvTranspose2d(embed_dim // 2, out_channels=embed_dim // 4, kernel_size=2, stride=2)
        self.regress4 = nn.Conv2d(embed_dim // 4, 8, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Let the features pass the core structure of Transformer
        # Forward stage 0
        x0 = self.stage00(x)
        x0 = self.stage01(x0)

        # Forward stage 1
        x1 = self.patch_merge1(x0)
        x1 = self.stage10(x1)
        x1 = self.stage11(x1)

        # Forward stage 2
        x2 = self.patch_merge2(x1)
        x2 = self.stage20(x2)
        x2 = self.stage21(x2)

        # Forward stage 3
        x3 = self.patch_merge3(x2)
        x3 = self.stage30(x3)
        x3 = self.stage31(x3)

        # Upsample stage 4
        x4 = self.upsample_layer4(x3)
        x4 = torch.cat([x4, x2], dim=2)
        x4 = x4.permute(0, 2, 1)
        x4 = self.fusion_layer1(x4).permute(0, 2, 1)
        x4 = self.stage41(x4)

        # Upsample stage 5
        x5 = self.upsample_layer5(x4)
        x5 = torch.cat([x5, x1], dim=2)
        x5 = x5.permute(0, 2, 1)
        x5 = self.fusion_layer2(x5).permute(0, 2, 1)
        x5 = self.stage51(x5)

        # Upsample stage 5
        x6 = self.upsample_layer6(x5)
        x6 = torch.cat([x6, x0], dim=2)
        x6 = x6.permute(0, 2, 1)
        x6 = self.fusion_layer3(x6).permute(0, 2, 1)
        x6 = self.stage61(x6)

        # Flow map regression module
        output = Seqtoimg(self.input_resolution6, x6)
        output = self.regress1(output)
        output = self.regress2(output)
        output = self.regress3(output)
        output = self.regress4(output)
        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MS_UNet().to(device)
    input = torch.ones((10, 3, 384, 512)).to(device)
    out = model(input)
    print(out.shape)
