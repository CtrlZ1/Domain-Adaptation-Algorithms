import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from model import source_dual_net, target_dual_net, source_to_target_net, Vector, Unflatten
import dataLoder
from CommenSetting import pathAndImageSize,args
from model import NeuralOT
from train import common_train
import tqdm
datasetRootAndImageSize=pathAndImageSize

plt.set_cmap("Greys")
N_BATCHES_PER_EPOCH = 10
N_SAMPLES = 1000
# CPU
DEVICE=torch.device('cpu')
kwargs={}
# 全局取消证书验证

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# if use cuda
if args.isCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

if __name__ == '__main__':
    distr = MultivariateNormal(torch.zeros(2), 0.4 * torch.eye(2))
    distr_dset = dataLoder.DistributionDataset(distr)
    circle_dset = dataLoder.CircleDataset(N_SAMPLES, n_centers=9, sigma=0.08)

    # A dataset that consists only of the centers of the given gaussians
    # circle_dset = CentersDataset(9)

    gauss_loader = dataLoder.ZipLoader(distr_dset, circle_dset, batch_size=args.batchSize,
                                       n_batches=N_BATCHES_PER_EPOCH,return_idx=True, **kwargs)

    plt.scatter(circle_dset.data[:, 0], circle_dset.data[:, 1], marker="+", lw=0.6)
    plt.show()
    for (x_idx, x), (y_idx, y) in gauss_loader:
        print(x_idx.shape, x.shape, y_idx.shape, y.shape)
        break
    source_dual_net = nn.Sequential(nn.Linear(2, 200),
                                    nn.SELU(),
                                    nn.BatchNorm1d(200),
                                    nn.Linear(200, 500),
                                    nn.SELU(),
                                    nn.BatchNorm1d(500),
                                    nn.Linear(500, 1)
                                    )
    target_dual_net = Vector(initial=1e-2 * torch.randn(len(circle_dset)))

    source_to_target_net = nn.Sequential(nn.Linear(2, 200),
                                         nn.SELU(),
                                         nn.BatchNorm1d(200),
                                         nn.Linear(200, 500),
                                         nn.SELU(),
                                         nn.BatchNorm1d(500),
                                         nn.Linear(500, 2)
                                         )

    ot = NeuralOT(source_dual_net, target_dual_net, source_to_target_net,
                  regularization_mode='l2', regularization_parameter=5e-3,
                  from_discrete=False, to_discrete=True).to(DEVICE)

    plan_optimizer = Adam(ot.parameters(), lr=1e-3)
    plan_scheduler = None  # MultiStepLR(plan_optimizer, [5, 15])

    losses = common_train(ot.plan_criterion, plan_optimizer, gauss_loader, args=args, device=DEVICE,
                   scheduler=plan_scheduler)

    plt.plot(losses)
    plt.show()
    mapping_optimizer = Adam(ot.parameters(), lr=1e-4)
    mapping_scheduler = None

    mapping_losses = common_train(ot.mapping_criterion, mapping_optimizer, gauss_loader, args=args, device=DEVICE,
                           scheduler=mapping_scheduler)

    plt.plot(mapping_losses)
    plt.show()

    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    n_points = 100000
    n_show = 500
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5

    idx = np.random.choice(np.arange(n_points, dtype=np.int), n_show,
                           replace=False)

    ax = axes[0]
    # 从多元分布中采样
    ps = distr.sample([n_points])
    with torch.no_grad():
        ps_mapped = ot.cpu().map(ps).detach().numpy()
    ps = ps.numpy()
    # 直方图
    H, xedges, yedges = np.histogram2d(ps[:, 0], ps[:, 1],
                                       range=[[xmin, xmax], [ymin, ymax]])
    # 等高线图
    ax.contour(H.transpose(), extent=[xedges.min(), xedges.max(),
                                      yedges.min(), yedges.max()])
    sc = ax.scatter(circle_dset.data[:, 0], circle_dset.data[:, 1],
                    marker="+", lw=0.6, label="Target dist.")
    ax.set_title("Target and Source Distribution")
    custom_lines = [sc, Line2D([0], [0], color=plt.cm.hsv(0.2), lw=2.)]
    ax.legend(custom_lines, ["Target dist.", "Source dist."])

    xs = np.linspace(xmin, xmax, num=10)
    ys = np.linspace(ymin, ymax, num=10)


    ax = axes[1]
    # 点的x坐标和y坐标集合
    XX, YY = np.meshgrid(xs, ys)
    # 扁平化
    XX = torch.tensor(XX.ravel(), dtype=torch.float)
    YY = torch.tensor(YY.ravel(), dtype=torch.float)
    # 合并
    X = torch.stack([XX, YY], dim=1)
    with torch.no_grad():
        Z = ot.cpu().map(X).detach()
    offsets = Z - X

    ax.scatter(circle_dset.data[:, 0], circle_dset.data[:, 1],
               marker="+", lw=0.6)
    ax.scatter(XX, YY,
               marker=".", lw=0.6)
    # 矢量箭头
    ax.quiver(XX, YY, offsets[:, 0], offsets[:, 1], angles="xy", units="width",
              width=0.002)
    ax.set_title("Displacement Field")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax = axes[2]
    ax.set_title("Generated Samples")
    ax.scatter(circle_dset.data[:, 0], circle_dset.data[:, 1],
               marker="+", lw=0.6, label="Target samples")
    ax.scatter(ps_mapped[idx, 0], ps_mapped[idx, 1],
               marker="+", lw=0.6, label="Generated samples")
    ax.legend()

    ax = axes[3]
    ax.set_title("Generated Density")
    H, xedges, yedges = np.histogram2d(ps_mapped[:, 0], ps_mapped[:, 1],
                                       range=[[xmin, xmax], [ymin, ymax]])

    ax.contour(H.transpose(), extent=[xedges.min(), xedges.max(),
                                      yedges.min(), yedges.max()])
    plt.tight_layout()
    fig.savefig("gauss.pdf")
    plt.show()