'''VGG for CIFAR10.
(c) YANG, Wei
'''

import math

import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, builder, num_classes):
        super(VGG, self).__init__()
        self.builder = builder
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = builder.batchnorm(128)
        self.nonlinear0 = builder.activation(num_parameters=128, affine=True)
        self.conv1 = builder.conv3x3(128, 128)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = builder.batchnorm(128, affine=True)
        self.nonlinear1 = builder.activation(num_parameters=128)
        self.conv2 = builder.conv3x3(128, 256)
        self.bn2 = builder.batchnorm(256, affine=True)
        self.nonlinear2 = builder.activation(num_parameters=256)
        self.conv3 = builder.conv3x3(256, 256)
        self.bn3 = builder.batchnorm(256, affine=True)
        self.nonlinear3 = builder.activation(num_parameters=256)
        self.conv4 = builder.conv3x3(256, 512)
        self.bn4 = builder.batchnorm(512, affine=True)
        self.nonlinear4 = builder.activation(num_parameters=512)
        self.conv5 = builder.conv3x3(512, 512)
        self.bn5 = builder.batchnorm(512, affine=True)
        # last block just use normal activation
        self.nonlinear5 = nn.PReLU(num_parameters=512)
        self.fc = nn.Linear(512 * 4 * 4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, self.builder.conv_layer):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear0(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear4(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def VGGSmall_1w1a(builder, num_classes: int = 100):
    model = VGG(builder, num_classes)
    return model
