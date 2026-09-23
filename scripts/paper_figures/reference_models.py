"""
Reference segmentation architectures, used ONLY to measure compute cost
(parameters, FLOPs, CPU latency) on the same hardware as the proposed model.
None of these are trained here, so no accuracy is claimed for them.

- UNet: Ronneberger et al., 2015 (the widely used milesial/Pytorch-UNet layout,
  transposed-conv upsampling) -- 31,037,633 params at 3->1 channels.
- DLinkNet34: Zhou et al., 2018, winner of the DeepGlobe 2018 road challenge,
  following the authors' public implementation (ResNet-34 encoder, cascaded
  dilated centre block, LinkNet decoder).
- DeepLabV3 / LR-ASPP with MobileNetV3-Large: torchvision implementations
  (the standard lightweight segmentation references).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class _DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=1):
        super().__init__()
        ch = [64, 128, 256, 512, 1024]
        self.inc = _DoubleConv(in_ch, ch[0])
        self.downs = nn.ModuleList([_DoubleConv(ch[i], ch[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.up_convs = nn.ModuleList([_DoubleConv(ch[i + 1], ch[i]) for i in reversed(range(4))])
        self.outc = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(F.max_pool2d(skips[-1], 2)))
        x = skips.pop()
        for up, conv in zip(self.ups, self.up_convs):
            x = conv(torch.cat([skips.pop(), up(x)], dim=1))
        return self.outc(x)


class _DBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dilates = nn.ModuleList([nn.Conv2d(ch, ch, 3, dilation=d, padding=d) for d in (1, 2, 4, 8)])

    def forward(self, x):
        out, h = x, x
        for conv in self.dilates:
            h = F.relu(conv(h), inplace=True)
            out = out + h
        return out


class _LinkDecoder(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cin // 4, 1), nn.BatchNorm2d(cin // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(cin // 4, cin // 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(cin // 4), nn.ReLU(inplace=True),
            nn.Conv2d(cin // 4, cout, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class DLinkNet34(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        r = torchvision.models.resnet34(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.e1, self.e2, self.e3, self.e4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.dblock = _DBlock(512)
        self.d4, self.d3 = _LinkDecoder(512, 256), _LinkDecoder(256, 128)
        self.d2, self.d1 = _LinkDecoder(128, 64), _LinkDecoder(64, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 3, padding=1),
        )

    def forward(self, x):
        e1 = self.e1(self.stem(x)); e2 = self.e2(e1); e3 = self.e3(e2)
        e4 = self.dblock(self.e4(e3))
        d4 = self.d4(e4) + e3; d3 = self.d3(d4) + e2; d2 = self.d2(d3) + e1
        return self.final(self.d1(d2))


class _TorchvisionSeg(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)["out"]


def reference_models():
    """name -> (constructor, short citation) for every compute-cost reference."""
    seg = torchvision.models.segmentation
    return {
        "U-Net": (lambda: UNet(), "Ronneberger et al., 2015"),
        "D-LinkNet34": (lambda: DLinkNet34(), "Zhou et al., 2018 (DeepGlobe winner)"),
        "DeepLabV3-MBv3": (lambda: _TorchvisionSeg(seg.deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=1, aux_loss=False)), "Chen et al., 2017"),
        "LR-ASPP-MBv3": (lambda: _TorchvisionSeg(seg.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=1)), "Howard et al., 2019"),
    }
