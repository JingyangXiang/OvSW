import torch.nn as nn

from .gradient_estimator import BinaryQuantizedEstimator, StraightThroughEstimator


# for weight and activation binary
class ActBinaryBase(nn.Module):
    def __init__(self, func):
        super(ActBinaryBase, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func.apply(x)


class BinaryAct(ActBinaryBase):
    def __init__(self, **kwargs):
        super(BinaryAct, self).__init__(func=BinaryQuantizedEstimator)


class STEAct(ActBinaryBase):
    def __init__(self, **kwargs):
        super(STEAct, self).__init__(func=StraightThroughEstimator)
