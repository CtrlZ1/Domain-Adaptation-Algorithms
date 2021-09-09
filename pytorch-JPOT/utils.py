import numpy as np
import random
import torch
from typing import Optional, Sequence
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.colors as col
# Nomalize alphas

from torchvision import datasets,transforms

from dataLoader import sourceClassWiseDataset


def euclidean_dist(x, y, square=False):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m = x.size(0)
    n = y.size(0)

    # 方式1
    a1 = torch.sum((x ** 2), 1, keepdim=True).expand(m, n)
    b2 = (y ** 2).sum(1).expand(m, n)
    if square:
        dist = (a1 + b2 - 2 * (x @ y.T)).float()
    else:
        dist = (a1 + b2 - 2 * (x @ y.T)).float().sqrt()
    return dist



def expanddata(data,label,n_dim,device):
    if n_dim == 0:
        data, label = data.view(len(data), -1).to(device), label.to(
            device)
    elif n_dim > 0:
        imgSize = torch.sqrt(
            (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
        data = data.expand(len(data), n_dim, imgSize.item(), imgSize.item()).to(
            device)
        label = label.to(device)

    return data,label


def model_feature_tSNE(model,sourceDataLoader,targetDataLoader,image_name,device):
    source_feature = collect_feature(sourceDataLoader, model, device)
    target_feature = collect_feature(targetDataLoader, model, device)

    tSNE_filename = os.path.join('images', image_name+'.png')
    tSNE(source_feature,target_feature,tSNE_filename)
def tSNE(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=2)
    plt.savefig(filename)
def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                                   device: torch.device, max_num_features=None) -> torch.Tensor:
    """
        Fetch data from `data_loader`, and then use `feature_extractor` to collect features

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader.
            feature_extractor (torch.nn.Module): A feature extractor.
            device (torch.device)
            max_num_features (int): The max number of features to return

        Returns:
            Features in shape (min(len(data_loader), max_num_features), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    print("start to extract features...")
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(device)
            fc6,fc7,fc8=feature_extractor.backbone(images)
            Output = fc8
            feature = feature_extractor.bottleneck(Output).cpu()
            all_features.append(feature)
            if max_num_features is not None and i >= max_num_features:
                break
    return torch.cat(all_features, dim=0)


def computeD(fea_target,mu_A,sigma_A,device):
    '''
    :param fea_target: [batchsize,feature_dim]
    :param mu_A: [n_labels+1,feature_dim]
    :param sigma_A: [n_labels+1,feature_dim，feature_dim]
    :return: [1]
    '''
    # [n_labels+1,batchsize]
    D = torch.zeros(len(mu_A),fea_target.size(0)).to(device)
    temp=[]
    sumD=torch.zeros(1,fea_target.size(0)).to(device)
    for index,i in enumerate(mu_A):
        d=compute_Mahalanobis_distance(fea_target,mu_A[index],sigma_A[index])
        element=torch.exp(-1*d)# [1,batchsize]
        temp.append(element)
        sumD+=element
    for i in range(len(mu_A)):
        D[i]=temp[i]/sumD
    return D
def compute_Mahalanobis_distance(fea_target,mu_A_z,sigma_A_z):
    '''
    :param fea_target:[batchsize,feature_dim]
    :param mu_A_z: [1,feature_dim]
    :param sigma_A_z:[feature_dim,feature_dim]
    :return:[1,batchsize]
    '''

    return torch.transpose(torch.sqrt(torch.sum(torch.matmul(fea_target-mu_A_z.view(1,-1),(1/sigma_A_z))*(fea_target-mu_A_z.view(1,-1)),dim=1,keepdim=True)),0,1)


def get_r2(r,C,Rou):
    r2=r.clone()
    C2=C-Rou
    a=torch.zeros(r2.size(0),r2.size(1))
    for i in range(C2.size(0)):
        for j in range(C2.size(1)):
            if C2[i][j]>0:
                r2[i][j]=0
    index=r2.data.max(1)[1]
    for i in range(a.size(0)):
        a[i][index[i]]=r2[i][index[i]]
    return a

def getsourceDataByClass(model,sourceData,sourceLabel,n_labels,DEVICE):
    sourceDataByClass=[[] for i in range(n_labels)]
    for index,i in enumerate(sourceLabel):
        sourceDataByClass[i].append(sourceData[index].detach().cpu().numpy())
    for index in range(len(sourceDataByClass)):
        sourceDataByClass[index]=torch.tensor(sourceDataByClass[index]).to(DEVICE)
    # for data in sourceDataByClass:
    #     print(data.size())
    return [model.bottleneck(model.backbone(data)) for data in sourceDataByClass[:-1]]


def changeSourceData(dataIndex,path,batchsize,kwargs):
    if dataIndex==6:
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_src = datasets.MNIST(
            root=path,
            train=True,
            download=False,
            transform=transform
        )
        newSourceData=[]
        newSourceLabel=[]
        label=trainData_src.targets
        data=trainData_src.data
        for index,i in enumerate(label):
            if i<5:
                newSourceData.append(data[index].numpy())
                newSourceLabel.append(i.numpy())
        newSourceDataLoader=DataLoader(
            sourceClassWiseDataset(newSourceData, newSourceLabel,
            transform=transform),
            batch_size=batchsize,
            shuffle=True,
            drop_last=True,
            **kwargs)

        return newSourceDataLoader

