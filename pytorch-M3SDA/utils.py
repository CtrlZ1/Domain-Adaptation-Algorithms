import torch
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Any, Tuple
import torch.nn as nn
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def L1_discrepancy(out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return ReverseLayerF.apply(*input)