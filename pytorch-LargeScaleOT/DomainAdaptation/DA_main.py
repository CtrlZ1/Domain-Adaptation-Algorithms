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
from model import source_dual_net, target_dual_net, source_to_target_net, Vector, Unflatten
import dataLoder
from CommenSetting import pathAndImageSize,args
from model import NeuralOT
from train import common_train
import tqdm
datasetRootAndImageSize=pathAndImageSize

plt.set_cmap("Greys")
N_BATCHES_PER_EPOCH = 10
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
    # prepare data
    sourceTrain, targetTrain = dataLoder.loadTrainData(datasetRootAndImageSize[args.datasetIndex],
                                                                   args.batchSize,
                                                                   args.datasetIndex,
                                                                   datasetRootAndImageSize[args.datasetIndex][2],
                                                                   kwargs)
    paired_loader = dataLoder.ZipLoader(sourceTrain, targetTrain, batch_size=args.batchSize, n_batches=N_BATCHES_PER_EPOCH,
                                        return_idx=True, **kwargs)

    ## Parametrization via vector ,discrete
    source_dual_net = Vector(initial=1e-2 * torch.randn(len(sourceTrain)))

    ## Parametrization via vector ,discrete
    target_dual_net = Vector(initial=1e-2 * torch.randn(len(targetTrain)))
    h=w=datasetRootAndImageSize[args.datasetIndex][2]
    source_to_target_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(h * w, 200),
        nn.ReLU(),
        nn.BatchNorm1d(200),
        nn.Linear(200, 500),
        nn.ReLU(),
        nn.BatchNorm1d(500),
        nn.Linear(500, h * w),
        Unflatten(h, w),
        nn.Tanh()
    )

    ## In the case we use vectors, we are working in the discrete setting
    ot = NeuralOT(source_dual_net, target_dual_net, source_to_target_net,
                  regularization_mode='l2', regularization_parameter=0.05,
                  from_discrete=True, to_discrete=True).to(DEVICE)

    # # The training procedure is changed too
    plan_optimizer = Adam(ot.parameters(), lr=1.)
    plan_scheduler = MultiStepLR(plan_optimizer, [100, 400, 800])

    losses = common_train(ot.plan_criterion, plan_optimizer, paired_loader, args=args, device=DEVICE,
               scheduler=plan_scheduler)

    plt.plot(losses)
    plt.show()
    mapping_optimizer = Adam(ot.parameters(), lr=1e-4)
    mapping_scheduler = MultiStepLR(plan_optimizer, [100, 400, 800])

    mapping_losses = common_train(ot.mapping_criterion, mapping_optimizer, paired_loader, args=args, device=DEVICE,
                           scheduler=mapping_scheduler)

    plt.plot(mapping_losses)
    plt.show()

    n_samples = 10
    idx = torch.multinomial(torch.ones(len(sourceTrain)), n_samples)
    ot.eval().cpu()
    # 2*n_samples,第一行是源域图，第二行是map之后的图，看上去像是两个人的笔记，23333
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 6))

    for i in range(n_samples):
        img = sourceTrain[idx[i]][0]
        axes[0, i].imshow(img.squeeze(), cmap="Greys")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        mapped = ot.map(img.reshape(1, 1, h, w))
        axes[1, i].imshow(mapped.squeeze().detach().numpy(), cmap="Greys")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.tight_layout()
    fig.savefig("../img/mappings.png")



    X_source, y_source = [], []
    for i in tqdm.tqdm(range(len(sourceTrain)), "Source"):
        X, y = sourceTrain[i]
        X_source.append(X)
        y_source.append(y)

    X_source = torch.cat(X_source).reshape(-1, h * w).numpy()
    y_source = np.array(y_source)

    X_target, y_target = [], []
    for i in tqdm.tqdm(range(len(targetTrain)), "Target"):
        X, y = targetTrain[i]
        X_target.append(X)
        y_target.append(y)

    X_target = torch.cat(X_target).reshape(-1, h * w).numpy()
    y_target = np.array(y_target)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_source, y_source)

    y_pred = clf.predict(X_target)
    print("1-KNN accuracy: {:.3f}".format(accuracy_score(y_target, y_pred)))

    X_source_mapped, y_source_mapped = [], []
    for i in tqdm.tqdm(range(len(sourceTrain)), "Source -> Target"):
        X, y = sourceTrain[i]
        mapped = ot.map(X.reshape(1, 1, h, w))
        X_source_mapped.append(mapped.squeeze())
        y_source_mapped.append(y)

    X_source_mapped = torch.cat(X_source_mapped).reshape(-1, h * w).detach().numpy()
    y_source_mapped = np.array(y_source_mapped)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_source_mapped, y_source_mapped)

    y_pred = clf.predict(X_target)
    print("Mapped 1-KNN accuracy: {:.3f}".format(accuracy_score(y_target, y_pred)))