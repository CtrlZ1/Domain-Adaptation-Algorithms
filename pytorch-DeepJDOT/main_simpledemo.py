import torch
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

import model

import matplotlib.pyplot as plt
import numpy as np
import os
from CommonSettings import args,datasetRootAndImageSize
from dataLoader import generateBiModGaussian


plt.ioff()
np.random.seed(1976)
np.set_printoptions(precision=3)

# CPU
DEVICE=torch.device('cpu')
kwargs={}

# if use cuda
if not args.noCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':

    # generate data center
    source_traindata, source_trainlabel = make_blobs(1200, centers=[[0, -1], [0, 0], [0, 1]], cluster_std=0.2)
    target_traindata, target_trainlabel = make_blobs(1200, centers=[[1, 0], [1, 1], [1, 2]], cluster_std=0.2)
    plt.figure()
    plt.scatter(source_traindata[:, 0], source_traindata[:, 1], c=source_trainlabel, marker='o', alpha=0.4)
    plt.scatter(target_traindata[:, 0], target_traindata[:, 1], c=target_trainlabel, marker='x', alpha=0.4)
    plt.legend(['source train data', 'target train data'])
    plt.title("2D blobs visualization (shape=domain, color=class)")
    plt.show()

    batchSize=128

    # use network model trained in source to predict target
    source_dataset = TensorDataset(torch.from_numpy(source_traindata).float(), torch.from_numpy(source_trainlabel).long())
    target_dataset = TensorDataset(torch.from_numpy(target_traindata).float(), torch.from_numpy(target_trainlabel).long())
    source_loader = torch.utils.data.DataLoader(dataset=source_dataset, batch_size=batchSize, shuffle=True)
    target_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=batchSize, shuffle=True)

    n_class = len(np.unique(source_trainlabel))
    n_dim = 0

    lrf=0.002
    lrc=0.002
    epoch=600
    mymodel= model.DeepJDOT(n_class,DEVICE)
    # mymodel.sourcemodel_usedin_target(mymodel.feature_ext_demo,mymodel.classifier_demo,source_loader,target_loader,lrf,lrc,epoch,n_dim)



    # use DeepJDOT model to predict target
    alpha=5
    lam=10
    mymodel.train_process(mymodel.feature_ext_demo, mymodel.classifier_demo, source_loader, target_loader, lrf, lrc, epoch, alpha,lam,n_dim,
                  method='sinkhorn', metric='deep', reg_sink=1)


