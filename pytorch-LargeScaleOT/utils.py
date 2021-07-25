import torch
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def l2_distance(x: torch.Tensor, y: torch.Tensor) \
        -> torch.Tensor:
    """Compute the Gram matrix holding all ||.||_2 distances."""
    xTy = 2 * x.matmul(y.transpose(0, 1))
    x2 = torch.sum(x ** 2, dim=1)[:, None]
    y2 = torch.sum(y ** 2, dim=1)[None, :]
    K = x2 + y2 - xTy
    return K