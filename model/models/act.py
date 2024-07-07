import torch
import torch.nn as nn


def name_func(name, **kwargs):
    name_to_func = {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "swish": Swish(),
        "prelu": nn.PReLU(kwargs.get("num_parameters")),
        "relu6": nn.ReLU6(inplace=True),
        "hardtanh": nn.Hardtanh(inplace=True),
        "leakyrelu_hardtanh": nn.Sequential(nn.LeakyReLU(negative_slope=kwargs.get("negative_slope")),
                                            nn.Hardtanh(inplace=True))
    }
    return name_to_func[name]


def swish(x):
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)
