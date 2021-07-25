import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from model import source_dual_net,target_dual_net,source_to_target_net
import dataLoder
from CommenSetting import pathAndImageSize,args
from model import NeuralOT
from train import common_train

datasetRootAndImageSize=pathAndImageSize

plt.set_cmap("Greys")
# CPU
DEVICE=torch.device('cpu')
kwargs={}

# if use cuda
if args.isCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(datasetRootAndImageSize[args.datasetIndex][2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    # prepare data
    mnist_train = datasets.MNIST(
        root=datasetRootAndImageSize[args.datasetIndex][1],
        transform=transform,
        train=True,
        download=True
    )
    mnist_test = datasets.MNIST(
        root=datasetRootAndImageSize[args.datasetIndex][1],
        transform=transform,
        train=False,
        download=True
    )
    MNIST = ConcatDataset([mnist_train, mnist_test])

    MNIST_MEAN, MNIST_COV = dataLoder.get_mean_covariance(MNIST)

    N_BATCHES_PER_EPOCH = 100

    gauss = dataLoder.CustomGaussian(MNIST_MEAN, MNIST_COV)
    # 采样得到[1,784]，转化成[1,28,28]
    gauss_dset = dataLoder.DistributionDataset(gauss, transform=lambda x: x.reshape(1, 28, 28))

    batch_generator = dataLoder.ZipLoader(gauss_dset, MNIST, batch_size=args.batchSize, n_batches=N_BATCHES_PER_EPOCH,
                                          return_idx=True,**kwargs)

    for (x_idx, x), (y_idx, y) in batch_generator:
        # y是MNIST，x是高斯分布
        print(x_idx.shape, x.shape, y_idx.shape, y.shape)
        break
    source_dual_net=source_dual_net().to(DEVICE)
    target_dual_net=target_dual_net().to(DEVICE)
    source_to_target_net=source_to_target_net().to(DEVICE)
    # ot = torch.load('generative_model.pth')
    ot = NeuralOT(source_dual_net, target_dual_net, source_to_target_net,
                  regularization_mode='l2', regularization_parameter=0.05,
                  from_discrete=False, to_discrete=False).to(DEVICE)

    plan_optimizer = Adam(ot.parameters(), lr=1e-3)
    plan_scheduler = MultiStepLR(plan_optimizer, [20, 75])

    losses = common_train(ot.plan_criterion, plan_optimizer, batch_generator, device=DEVICE,args=args,
                   scheduler=plan_scheduler)

    plt.plot(losses)
    plt.show()

    mapping_optimizer = Adam(ot.parameters(), lr=1e-4)
    mapping_scheduler = None  # MultiStepLR(plan_optimizer, [10])

    mapping_losses = common_train(ot.mapping_criterion, mapping_optimizer, batch_generator, device=DEVICE,
                                  args=args,scheduler=mapping_scheduler)

    tmp_loader = dataLoder.ZipLoader(gauss_dset, batch_size=args.batchSize, n_batches=N_BATCHES_PER_EPOCH,
                           return_idx=False, **kwargs)
    for x in tmp_loader:
        x = x[0]
        x = x.to(DEVICE)
        mapped = ot.map(x)
    imgs = mapped[:, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
    for i, img in enumerate(imgs):
        ax = axes[i // 10, i % 10]
        ax.imshow(img)
        ax.axis('off')
    fig.savefig("../img/generated.png")