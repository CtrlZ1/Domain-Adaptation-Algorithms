import torch.nn as nn

import torch

from utils import l2_distance


class KantorovichPotential(nn.Module):
    def __init__(self):
        super(KantorovichPotential, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )
    def forward(self,sourcefeature,targetfeature,C,args):
        loss=torch.mean(-1*args.Epsilon*torch.log(torch.mean(torch.exp((self.net(targetfeature).T-C)/args.Epsilon),dim=1)))+\
               torch.mean(self.net(targetfeature))-\
               args.Epsilon
        return loss

class Vector(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.v = nn.Parameter(initial)

    def forward(self, idx):
        return self.v[idx]
class Unflatten(nn.Module):
    def __init__(self, *spatial):
        super().__init__()
        self.spatial = spatial

    def forward(self, x):
        return x.reshape(len(x), 1, *self.spatial)

class source_dual_net(nn.Module):
    def __init__(self):
        super(source_dual_net, self).__init__()
        self.net= nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1),
            nn.Flatten(start_dim=0)
        )
    def forward(self,data):
        return self.net(data)
class target_dual_net(nn.Module):
    def __init__(self):
        super(target_dual_net, self).__init__()
        self.net= nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1),
            nn.Flatten(start_dim=0)
        )
    def forward(self,data):
        return self.net(data)

class source_to_target_net(nn.Module):
    def __init__(self):
        super(source_to_target_net, self).__init__()
        self.net= nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 28 * 28),
            Unflatten(28, 28),
            nn.Tanh()
        )
    def forward(self,data):
        return self.net(data)






class NeuralOT(nn.Module):
    def __init__(self, source_dual_net, target_dual_net, source_to_target_net, regularization_parameter=1e-1,
                 regularization_mode='entropy', from_discrete=False, to_discrete=False):
        super().__init__()
        self.u = source_dual_net
        self.v = target_dual_net
        self.f = source_to_target_net
        self.eps = regularization_parameter
        if regularization_mode not in ['entropy', 'l2']:
            raise ValueError("``regularization_mode`` must be ``'entropy'`` or ``'l2'``.")
        self.mode = regularization_mode
        self.from_discrete = from_discrete
        self.to_discrete = to_discrete

    @staticmethod
    def squared_l2_distances(x, y):
        """
        Parameters
        ----------
        x : torch.tensor
            Batch of images from source domain of shape ``(N_s, C, H, W)``.
        y : torch.tensor
            Batch of images from target domain of shape ``(N_t, C, H, W)``.

        Returns
        -------
        distances : torch.tensor
            Tensor of shape ``(N_s, N_t)`` with pairwise l2 distances between source and target images.
        """
        dims = [-(i+1) for i in range(len(x.shape)-1)]
        return torch.sum((x[:, None] - y) ** 2, dim=dims)

    def plan_criterion(self, x_idx, x, y_idx, y):
        """
        Parameters
        ----------
        x_idx : torch.tensor
            Batch of indices of shape ``(N,)``.
        x : torch.tensor
            Batch of images from source domain of shape ``(N, C, H, W)``.
        y_idx : torch.tensor
            Batch of indices of shape ``(N,)``.
        y : torch.tensor
            Batch of images from target domain of shape ``(N, C, H, W)``.

        Returns
        -------
        loss : torch.tensor
            Loss for training dual neural networks.
        """
        self.u.train()
        self.v.train()
        u = self.u(x_idx) if self.from_discrete else self.u(x)
        v = self.v(y_idx) if self.to_discrete else self.v(y)
        c = self.squared_l2_distances(x, y)

        if self.mode == 'entropy':
            regularization_term = -self.eps * torch.exp((u[:, None] + v - c) / self.eps)
        else:
            regularization_term = -torch.relu(u[:, None] + v - c) ** 2 / (4 * self.eps)

        return -torch.mean(u[:, None] + v + regularization_term)

    def mapping_criterion(self, x_idx, x, y_idx, y):
        """
        Parameters
        ----------
        x_idx : torch.tensor
            Batch of indices of shape ``(N,)``.
        x : torch.tensor
            Batch of images from source domain of shape ``(N, C, H, W)``.
        y_idx : torch.tensor
            Batch of indices of shape ``(N,)``.
        y : torch.tensor
            Batch of images from target domain of shape ``(N, C, H, W)``.

        Returns
        -------
        loss : torch.tensor
            Loss for training mapping neural network.
        """
        self.u.eval()
        self.v.eval()
        self.f.train()
        u = self.u(x_idx).detach() if self.from_discrete else self.u(x).detach()
        v = self.v(y_idx).detach() if self.to_discrete else self.v(y).detach()
        c = self.squared_l2_distances(x, y)
        mapped = self.f(x)  # shape ``(N, C, H, W)``

        d = self.squared_l2_distances(mapped, y)
        if self.mode == 'entropy':
            h = torch.exp((u[:, None] + v - c) / self.eps)
        else:
            h = torch.relu(u[:, None] + v - c) / (2 * self.eps)

        return torch.mean(d * h)

    def map(self, x):
        self.f.eval()
        return self.f(x)
