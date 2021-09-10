import numpy as np
import torch
import torch.autograd as autograd


def getGradientPenalty(critic,real_samples,fake_samples,args):
    alpha=torch.rand(real_samples.size(0), 1)
    if not args.noCuda and torch.cuda.is_available():
        alpha=alpha.cuda()
    interpolates=alpha*real_samples+(1-alpha)*fake_samples
    interpolates=torch.stack([interpolates,real_samples,fake_samples]).requires_grad_()
    D_interpolates=critic(interpolates)
    gradients=autograd.grad(
        inputs=interpolates,
        outputs=D_interpolates,
        grad_outputs=torch.ones_like(D_interpolates),
        retain_graph=True,create_graph=True,only_inputs=True
    )[0]

    gradient_penalty= ((gradients.norm(2,dim=1)-1)**2).mean()

    return gradient_penalty