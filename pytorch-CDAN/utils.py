from typing import Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradientReverseFunction(Function):

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
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start
        The forward and backward behaviours are:
        .. math::
            \mathcal{R}(x) = x,
            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.
        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:
        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo
        where :math:`i` is the iteration step.
        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct
def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:
    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c
    where C is number of classes.
    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``
    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class ConditionalDomainAdversarialLoss(nn.Module):
    r"""The Conditional Domain Adversarial Loss used in `Conditional Adversarial Domain Adaptation (NIPS 2018) <https://arxiv.org/abs/1705.10667>`_
    Conditional Domain adversarial loss measures the domain discrepancy through training a domain discriminator in a
    conditional manner. Given domain discriminator :math:`D`, feature representation :math:`f` and
    classifier predictions :math:`g`, the definition of CDAN loss is
    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) &= \mathbb{E}_{x_i^s \sim \mathcal{D}_s} log[D(T(f_i^s, g_i^s))] \\
        &+ \mathbb{E}_{x_j^t \sim \mathcal{D}_t} log[1-D(T(f_j^t, g_j^t))],\\
    where :math:`T` is a :class:`MultiLinearMap`  or :class:`RandomizedMultiLinearMap` which convert two tensors to a single tensor.
    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of
          features. Its input shape is (N, F) and output shape is (N, 1)
        entropy_conditioning (bool, optional): If True, use entropy-aware weight to reweight each training example.
          Default: False
        randomized (bool, optional): If True, use `randomized multi linear map`. Else, use `multi linear map`.
          Default: False
        num_classes (int, optional): Number of classes. Default: -1
        features_dim (int, optional): Dimension of input features. Default: -1
        randomized_dim (int, optional): Dimension of features after randomized. Default: 1024
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    .. note::
        You need to provide `num_classes`, `features_dim` and `randomized_dim` **only when** `randomized`
        is set True.
    Inputs:
        - g_s (tensor): unnormalized classifier predictions on source domain, :math:`g^s`
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
    Shape:
        - g_s, g_t: :math:`(minibatch, C)` where C means the number of classes.
        - f_s, f_t: :math:`(minibatch, F)` where F means the dimension of input features.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, )`.
    Examples::
        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
        >>> import torch
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim * num_classes, hidden_size=1024)
        >>> loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(g_s, f_s, g_t, f_t)
    """

    def __init__(self, domain_discriminator: nn.Module, entropy_conditioning: Optional[bool] = False,
                 randomized: Optional[bool] = False, num_classes: Optional[int] = -1,
                 features_dim: Optional[int] = -1, randomized_dim: Optional[int] = 1024,
                 reduction: Optional[str] = 'mean'):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.entropy_conditioning = entropy_conditioning

        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, randomized_dim)
        else:
            self.map = MultiLinearMap()

        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight,
                                                                        reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)
        d_label = torch.cat((
            torch.ones((g_s.size(0), 1)).to(g_s.device),
            torch.zeros((g_t.size(0), 1)).to(g_t.device),
        ))
        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size
        self.domain_discriminator_accuracy = binary_accuracy(d, d_label)
        return self.bce(d, d_label, weight.view_as(d))


class RandomizedMultiLinearMap(nn.Module):
    """Random multi linear map
    Given two inputs :math:`f` and :math:`g`, the definition is
    .. math::
        T_{\odot}(f,g) = \dfrac{1}{\sqrt{d}} (R_f f) \odot (R_g g),
    where :math:`\odot` is element-wise product, :math:`R_f` and :math:`R_g` are random matrices
    sampled only once and ﬁxed in training.
    Args:
        features_dim (int): dimension of input :math:`f`
        num_classes (int): dimension of input :math:`g`
        output_dim (int, optional): dimension of output tensor. Default: 1024
    Shape:
        - f: (minibatch, features_dim)
        - g: (minibatch, num_classes)
        - Outputs: (minibatch, output_dim)
    """

    def __init__(self, features_dim: int, num_classes: int, output_dim: Optional[int] = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    """Multi linear map
    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)
