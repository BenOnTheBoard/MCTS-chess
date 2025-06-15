import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        if mask.shape != (self.weight.shape[2], self.weight.shape[3]):
            raise ValueError("Mask shape must match kernel spatial dimensions (H, W)")

        mask = mask[None, None, :, :].expand(self.weight.shape)
        self.register_buffer("mask", mask)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.conv2d(
            x,
            masked_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class KnightConv2d(MaskedConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        mask = torch.tensor(
            [
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
            ],
            dtype=torch.float32,
        )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=5,
            mask=mask,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )


class BishopConv2d(MaskedConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        mask = torch.tensor(
            [
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=5,
            mask=mask,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )


class RookConv2d(MaskedConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        mask = torch.tensor(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=torch.float32,
        )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=5,
            mask=mask,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )


class ChessConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        knight_channels = out_channels // 4
        bishop_channels = (out_channels * 3) // 8
        rook_channels = out_channels - knight_channels - bishop_channels

        self.knight = KnightConv2d(
            in_channels, knight_channels, stride, padding, dilation, groups, bias
        )
        self.bishop = BishopConv2d(
            in_channels, bishop_channels, stride, padding, dilation, groups, bias
        )
        self.rook = RookConv2d(
            in_channels, rook_channels, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        knight_out = self.knight(x)
        bishop_out = self.bishop(x)
        rook_out = self.rook(x)
        return torch.cat([knight_out, bishop_out, rook_out], dim=1)
