import math

import torch
import torch.nn as nn

import model
from args import args


class Builder(object):
    def __init__(self, args, conv_layer, bn_layer):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.args = args

    def conv(self, kernel_size, in_planes, out_planes, stride=1, groups=1, bias=False):
        conv_layer = self.conv_layer

        if kernel_size == 3:
            conv = conv_layer(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)
        elif kernel_size == 1:
            conv = conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, groups=groups)
        elif kernel_size == 5:
            conv = conv_layer(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=bias, groups=groups)
        elif kernel_size == 7:
            conv = conv_layer(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=bias, groups=groups)
        else:
            return ValueError(f"kernel_size={kernel_size} is not support")

        self._init_conv(conv)

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1, bias=False):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, bias=bias)
        return c

    def dwconv3x3(self, planes, stride=1):
        """3x3 dw convolution with padding"""
        c = self.conv(3, planes, planes, stride=stride, groups=planes)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride)
        return c

    def batchnorm(self, planes, last_bn=False, affine=True):
        return self.bn_layer(planes, affine=affine)

    def activation(self, **kwargs):
        nonlinearity = args.nonlinearity.lower()
        name_to_func = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "prelu": nn.PReLU(kwargs.get("num_parameters")),
            "relu6": nn.ReLU6(inplace=True),
            "hardtanh": nn.Hardtanh(inplace=True),
            "leakyrelu_hardtanh": nn.Sequential(nn.LeakyReLU(negative_slope=0.9), nn.Hardtanh(inplace=True))
        }
        return name_to_func[nonlinearity]

    def _init_conv(self, conv):
        nonlinearity = "leaky_relu"
        if args.init == "signed_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
            gain = nn.init.calculate_gain(nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

        elif args.init == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
            gain = nn.init.calculate_gain(nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif args.init == "kaiming_normal":
            nn.init.kaiming_normal_(conv.weight, mode=args.mode, nonlinearity=nonlinearity)

        elif args.init == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                conv.weight, mode=args.mode, nonlinearity=nonlinearity
            )
        elif args.init == "xavier_normal":
            nn.init.xavier_normal_(conv.weight)
        elif args.init == "xavier_constant":

            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            conv.weight.data = conv.weight.data.sign() * std

        elif args.init == "standard":

            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))

        else:
            raise ValueError(f"{args.init} is not an initialization option!")


def get_builder(args, logger):
    logger.info("==> Conv Type: {}".format(args.conv_type))
    logger.info("==> BN Type: {}".format(args.bn_type))
    logger.info("==> Act Type:{}".format(args.nonlinearity))

    conv_layer = getattr(model.modules, args.conv_type)
    bn_layer = getattr(model.modules, args.bn_type)

    builder = Builder(args=args, conv_layer=conv_layer, bn_layer=bn_layer)

    return builder
