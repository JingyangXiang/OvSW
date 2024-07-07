import torch.nn as nn

__all__ = ['ResNet18_1w1a', 'ResNet34_1w1a']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.act1 = nn.Hardtanh(inplace=True)
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes)
        self.act2 = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act1(out)
        residual = out

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.act2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, builder, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = builder.batchnorm(64)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0])
        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1e-8)
                m.bias.data.zero_()

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                builder.batchnorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(builder, self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.fc(x)

        return x


def ResNet18_1w1a(builder, num_classes: int = 1000):
    """Constructs a ResNet-18 model. """
    model = ResNet(builder, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model


def ResNet34_1w1a(builder, num_classes: int = 1000):
    """Constructs a ResNet-34 model. """
    model = ResNet(builder, BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model
