"""Models.

PaperUNet is a layer-for-layer PyTorch port of the Keras `unet()` in the original
notebook: single Conv3x3 -> ReLU -> BatchNorm -> Dropout per block (in that order),
four encoder levels (32/64/128/256 filters), a densely connected bridge of four such
blocks at 512 filters each (concatenating, as Keras `dense_block` does - not a fully
connected layer, as the manuscript said), nearest-neighbour upsampling with skip
concatenation, and a 1x1 sigmoid head. 25,749,633 trainable parameters plus 6,016
BatchNorm statistics = the 25,755,649 total Keras printed for a 1-channel input.

Everything else is segmentation_models_pytorch with an ImageNet-pretrained encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, cin, cout, p):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01)   # Keras defaults
        self.drop = nn.Dropout(p) if p else nn.Identity()

    def forward(self, x):
        return self.drop(self.bn(self.act(self.conv(x))))


class _DenseBridge(nn.Module):
    def __init__(self, cin, filters, layers, p):
        super().__init__()
        self.blocks = nn.ModuleList()
        c = cin
        for _ in range(layers):
            self.blocks.append(_Block(c, filters, p))
            c += filters
        self.out_channels = c

    def forward(self, x):
        feats = [x]
        for b in self.blocks:
            feats.append(b(torch.cat(feats, 1) if len(feats) > 1 else feats[0]))
        return torch.cat(feats, 1)


class PaperUNet(nn.Module):
    def __init__(self, in_ch=1, base=32, dropout=0.01):
        super().__init__()
        f = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList()
        c = in_ch
        for k in f:
            self.enc.append(_Block(c, k, dropout))
            c = k
        self.pool = nn.MaxPool2d(2)
        self.bridge = _DenseBridge(c, base * 16, 4, dropout)
        c = self.bridge.out_channels
        self.dec = nn.ModuleList()
        for k in reversed(f):
            self.dec.append(_Block(c + k, k, dropout))
            c = k
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bridge(x)
        for d, s in zip(self.dec, reversed(skips)):
            x = d(torch.cat([self.up(x), s], 1))
        return self.head(x)       # logits; sigmoid lives in the loss / at inference


IMAGENET_GREY_MEAN, IMAGENET_GREY_STD = 0.449, 0.226


def build(cfg) -> nn.Module:
    if cfg.arch == "paper_unet":
        return PaperUNet(cfg.in_channels, cfg.paper_filters, cfg.paper_dropout)
    if cfg.arch == "unet":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name=cfg.encoder, encoder_weights=cfg.encoder_weights,
                        in_channels=cfg.in_channels, classes=1)
    raise ValueError(f"unknown arch {cfg.arch}")


def normalise_input(x: torch.Tensor, cfg) -> torch.Tensor:
    """Windowed [0, 1] inputs go to ImageNet grey statistics for pretrained encoders.
    z-scored inputs are already standardised."""
    if cfg.window == "zscore" or cfg.arch == "paper_unet" or not cfg.encoder_weights:
        return x
    return (x - IMAGENET_GREY_MEAN) / IMAGENET_GREY_STD


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def encoder_parameters(model):
    return list(model.encoder.parameters()) if hasattr(model, "encoder") else []
