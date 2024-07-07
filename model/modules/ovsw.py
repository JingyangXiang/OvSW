import torch
from torch import nn as nn, Tensor

from .binary_act import BinaryAct
from .binary_weight import HardTanhSTE, HardTanhSTEV2, STEWeight, STEWeightV2

str2weight_binary = {
    "ste": STEWeight,
    "hardtanh": HardTanhSTE,
    'ste_v2': STEWeightV2,
    "hardtanhv2": HardTanhSTEV2,
}
str2act_binary = {
    "binary": BinaryAct,
}


class OvSWConv2d(nn.Conv2d):
    enable_dampen = False
    enable_ags = False

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(OvSWConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=bias)

    def init_scaling_factor(self, scaling_factor=False):
        if scaling_factor:
            self.register_parameter('scaling_factor', nn.Parameter(self.weight.abs().mean(dim=[1, 2, 3])))

    def init_binary_activation_and_weight(self, act_a, act_w):
        self.act_a = str2act_binary[act_a]()
        self.act_w = str2weight_binary[act_w]()
        self.logger.info("OvSWConv2d (act_a, act_w) is ({}, {})".format(self.act_a, self.act_w))

    def init_forward_type(self, forward_type):
        self.forward_type = forward_type
        assert forward_type == "xnor", f"{forward_type} is not supported by OvSWConv2d"
        self.logger.info(f"OvSWConv2d (forward_type) is {self.forward_type}")

    def init_logger(self, logger):
        self.logger = logger

    def init_adaptive_gradient_scale(self, args):
        self.enable_ags = args.enable_ags
        self.delta = args.delta
        if self.enable_ags:
            self.logger.info(f"OvSWConv2d set (adaptive_gradient_scale, delta) to ({self.enable_ags}, {self.delta:.5f})")
        else:
            self.logger.info(f"OvSWConv2d set adaptive_gradient_scale to {self.enable_ags}")

    def init_dampen(self, args):
        self.enable_dampen = args.enable_dampen
        self.dampen_weight = args.dampen_weight
        if self.enable_dampen:
            self.logger.info(
                f"OvSWConv2d (enable_dampen, dampen_weight) to ({self.enable_dampen}, {self.dampen_weight:.5f})")
            self.track_momentum = args.track_momentum
            self.track_threshold = args.track_threshold
            self.logger.info(
                f"OvSWConv2d (track_momentum, track_threshold) to ({self.track_momentum:.3f}, {self.track_threshold:.5f})")
            self.register_buffer("weight_prev", self.weight.clone())
            if args.track_ones:
                self.register_buffer("tracker", torch.ones_like(self.weight))
            else:
                self.register_buffer("tracker", torch.zeros_like(self.weight))
        else:
            self.logger.info(
                f"OvSWConv2d set enable_dampen to {self.enable_dampen}")

    def init_epoch(self, start_epoch, end_epoch, epoch_num):
        self.start_epoch = 0 if start_epoch is None else start_epoch
        self.end_epoch = end_epoch
        self.epoch_num = epoch_num
        self.logger.info(
            f"OvSWConv2d (start_epoch, end_epoch, epoch_num) is ({self.start_epoch}, {end_epoch}, {epoch_num})")

    def update_epoch(self, current_epoch):
        self.current_epoch = current_epoch
        self.logger.info(f"OvSWConv2d (current_epoch, epoch_num) is ({current_epoch}, {self.epoch_num})")

    @torch.no_grad()
    def adaptive_gradient_scale(self):
        if self.enable_ags:
            delta = self.delta
            grad_norm = torch.norm(self.weight.grad.data.flatten(1), dim=-1).clamp(min=1e-6).reshape(-1, 1, 1, 1)
            param_norm = torch.norm(self.weight.data.flatten(1), dim=-1).clamp(min=1e-3).reshape(-1, 1, 1, 1)
            judge = torch.less(grad_norm / param_norm, delta)
            grad = torch.where(judge, delta * param_norm / grad_norm * self.weight.grad, self.weight.grad)
            self.weight.grad = grad
            return torch.mean(grad_norm / param_norm)

    @torch.no_grad()
    def conditional_dampening(self):
        if self.enable_dampen:
            dampen_weight = self.dampen_weight
            # 记录权重震荡, 使用ema进行更新
            self.tracker *= self.track_momentum
            weight_sign = torch.sign(self.weight)
            weight_prev_sign = torch.sign(self.weight_prev)
            self.tracker += (1 - self.track_momentum) * torch.not_equal(weight_sign, weight_prev_sign)
            # 对权重施加 受条件限制权重衰减
            # 1. 判断震荡小于翻转阈值的部分信息
            condition_track = torch.less(self.tracker, self.track_threshold)
            # 2. 满足条件的部分权重需要权重衰减
            grad = self.weight.grad + dampen_weight * condition_track * self.weight.data
            # 3. 更新梯度
            self.weight.grad = grad

            # uodate weight_prev
            self.weight_prev.data = self.weight.data.clone()
            return torch.mean(condition_track * 1.)

    @torch.no_grad()
    def get_scale(self, weight):
        w0 = weight - weight.mean([1, 2, 3], keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var([1, 2, 3], keepdim=True) + 1e-5))
        scaling_factor_no_grad = w1.abs().mean([1, 2, 3], keepdim=True).detach().reshape(1, -1, 1, 1)
        return scaling_factor_no_grad

    def xnor(self, input, weight):
        if self.training:
            input = input / torch.sqrt(input.var(dim=[1, 2, 3], keepdim=True) + 1e-5).detach()
        input = self.act_a(input)
        if hasattr(self, 'scaling_factor'):
            scaling_factor = self.scaling_factor.reshape(1, -1, 1, 1)
        else:
            scaling_factor = self.get_scale(self.weight).reshape(1, -1, 1, 1)
        w1 = self.act_w(weight)
        return self._conv_forward(input, w1, self.bias) * scaling_factor

    @torch.no_grad()
    def scale_weights(self, scale):
        self.weight.data.mul_(scale)
        self.logger.info("OvSWConv2d scaled weights: {}".format(scale))

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight
        func = getattr(self, self.forward_type)
        return func(input, weight)
