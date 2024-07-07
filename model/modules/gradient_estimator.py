import torch


## SignQuantizedEstimator
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class StraightThroughEstimatorV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.sign(input)
        output = output * input.abs().mean(dim=[1, 2, 3], keepdim=True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


## BinaryQuantizedEstimator
class BinaryQuantizedEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output * ((2 - 2 * abs(input)).clamp(min=0))
        return grad_input


## HtanhSTEThroughEstimator
class HtanhSTEThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone() * (torch.abs(input) < 1)
        return grad_input


class HtanhSTEThroughEstimatorV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        scale = torch.abs(input).mean(dim=[1, 2, 3], keepdim=True)
        grad_input = grad_output.clone() * (torch.abs(input) < scale)
        return grad_input
