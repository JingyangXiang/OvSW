import torch.nn as nn

from model.modules.gradient_estimator import HtanhSTEThroughEstimator, HtanhSTEThroughEstimatorV2, \
    StraightThroughEstimator, StraightThroughEstimatorV2


# for weight and activation binary
class WeightBinaryBase(nn.Module):
    def __init__(self, func):
        super(WeightBinaryBase, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func.apply(x)


class STEWeight(WeightBinaryBase):
    def __init__(self, **kwargs):
        super(STEWeight, self).__init__(func=StraightThroughEstimator)


class STEWeightV2(WeightBinaryBase):
    def __init__(self, **kwargs):
        super(STEWeightV2, self).__init__(func=StraightThroughEstimatorV2)


class HardTanhSTE(WeightBinaryBase):
    def __init__(self, **kwargs):
        super(HardTanhSTE, self).__init__(func=HtanhSTEThroughEstimator)


class HardTanhSTEV2(WeightBinaryBase):
    def __init__(self, **kwargs):
        super(HardTanhSTEV2, self).__init__(func=HtanhSTEThroughEstimatorV2)
