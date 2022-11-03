'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.filter_utils import make_meshgrid, make_gabor_bank, kernels2weights

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, v1_oripowermap=None):
        super(ResNet, self).__init__()

        if v1_oripowermap:
            kernel_size = 7
            xs, ys = make_meshgrid(sz=kernel_size)
            phi = (5**0.5 + 1) / 2  # golden ratio
            freqs = [phi**n for n in range(2, -4, -1)]
            kernels_complex = (list)(make_gabor_bank(xs, ys, directions=9, freqs=freqs))
            kernels_real, kernels_imag = np.real(kernels_complex), np.imag(kernels_complex)
            weights_real, weights_imag = kernels2weights(kernels_real, 3), kernels2weights(kernels_imag, 3),
            print(f"ResNet: weights_real.shape = {weights_real.shape}")

            self.in_planes = len(kernels_complex)

            # NOTE: these need to be children to make sure they are handled correctly (i.e. assigning to device)
            self._conv1_real = nn.Conv2d(3, len(kernels_complex), kernel_size=kernel_size, stride=1, padding=3, bias=False)
            self._conv1_real.weight = torch.nn.Parameter(weights_real)
            
            self._conv1_imag = nn.Conv2d(3, len(kernels_complex), kernel_size=kernel_size, stride=1, padding=3, bias=False)
            self._conv1_imag.weight = torch.nn.Parameter(weights_imag)

            if v1_oripowermap == 'fixed':
                self._conv1_real.weight.requires_grad = False
                self._conv1_imag.weight.requires_grad = False

            elif v1_oripowermap == 'init':
                self._conv1_real.weight.requires_grad = True
                self._conv1_imag.weight.requires_grad = True

            self.conv1 = lambda x: self._conv1_real(x)**2 + self._conv1_imag(x)**2
            
        else:
            self.in_planes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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


def ResNet50(v1_oripowermap):
    return ResNet(Bottleneck, [3, 4, 6, 3], v1_oripowermap=v1_oripowermap)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
