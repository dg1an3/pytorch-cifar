"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.filter_utils import make_oriented_map


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_oriented_phasemap=None):
        super(BasicBlock, self).__init__()
        assert not use_oriented_phasemap
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_oriented_phasemap=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if use_oriented_phasemap:

            conv2_planes_out, self.conv2 = make_oriented_map(
                inplanes=planes,
                kernel_size=7, 
                directions=9, 
                stride=stride,
                dstack_phases=True
            )

        else:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            conv2_planes_out = planes

        self.bn2 = nn.BatchNorm2d(conv2_planes_out)
        self.conv3 = nn.Conv2d(
            conv2_planes_out, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def train_oriented_maps(self, train):
        self.conv2.weight.requires_grad = train

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut_x = self.shortcut(x)
        # print(f"{out.shape} vs {x_shortcut.shape}")
        out += shortcut_x
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, num_classes=10, use_oriented_maps=Union[str, None]
    ):
        """_summary_

        Args:
            block (_type_): _description_
            num_blocks (_type_): _description_
            num_classes (int, optional): _description_. Defaults to 10.
            use_oriented_maps (_type_, optional): _description_. Defaults to None.
        """
        super(ResNet, self).__init__()

        if use_oriented_maps:

            self.in_planes, self._conv1_real, self._conv1_imag = make_oriented_map(
                inplanes=3,
                kernel_size=7, 
                directions=9,
                stride=1
            )

            self.conv1 = lambda x: self._conv1_real(x) ** 2 + self._conv1_imag(x) ** 2

        else:

            self.in_planes = 64
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=1, padding=1, bias=False
            )

        # turn this off for the bottlenecks
        # use_oriented_maps = None

        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            use_oriented_phasemap=use_oriented_maps,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            use_oriented_phasemap=use_oriented_maps,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            use_oriented_phasemap=use_oriented_maps,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            use_oriented_phasemap=use_oriented_maps,
        )
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # if the use_oriented_maps parameter is 'init' then we should set to train
        self.train_oriented_maps(use_oriented_maps == "init")

    def train_oriented_maps(self, train=False):
        """_summary_

        Args:
            train (bool, optional): _description_. Defaults to False.
        """
        self._conv1_real.weight.requires_grad = train
        self._conv1_imag.weight.requires_grad = train

        # recursively set train_oriented_maps
        for child in self.children():
            if hasattr(child, "train_oriented_maps"):
                child.train_oriented_maps(train)

    def _make_layer(self, block, planes, num_blocks, stride, use_oriented_phasemap):
        """ """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    use_oriented_phasemap=use_oriented_phasemap,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(use_oriented_maps=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], use_oriented_maps=use_oriented_maps)


def ResNet101(use_oriented_maps=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], use_oriented_maps=use_oriented_maps)


def ResNet152(use_oriented_maps=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], use_oriented_maps=use_oriented_maps)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
