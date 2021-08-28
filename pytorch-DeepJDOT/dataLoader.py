import numpy as np
import torch
from torchvision import datasets,transforms
from PIL import Image
import os
import torch.utils.data as data


def generateBiModGaussian(centers, sigma, n, A, b, h,args):
    result = {}
    temp_x=[centers[i]+sigma * np.random.standard_normal((round(h[i] * n), args.length_of_feature)) for i in range(args.n_labels)]
    xtmp = np.concatenate(tuple(temp_x))
    result['X'] = torch.tensor(xtmp.dot(A) + b)
    temp_y=[np.ones(round(h[i] * n))*i for i in range(args.n_labels)]
    temp_y = np.concatenate(tuple(temp_y))
    # result['y'] = np.concatenate(tuple(temp_y))
    y_onehot=np.zeros((len(temp_y),args.n_labels))

    for index,y in enumerate(temp_y):
        y_onehot[index][int(y)]=1
    result['y']=torch.tensor(y_onehot).float()
    return result

# **加载源域数据**
def loadSourceAndTargetData(path,datasetIndex,imageSize):
    if datasetIndex==5:
        transform = transforms.Compose([
            transforms.Resize((imageSize,imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        # 按照文件夹不同为训练数据加入类别标签
        trainData_src = datasets.ImageFolder(root=path[0], transform=transform)
        trainData_tar = datasets.ImageFolder(root=path[1], transform=transform)



        return trainData_src,trainData_tar
    if datasetIndex==4:
        transform = transforms.Compose([
            transforms.Resize((imageSize,imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_src = datasets.USPS(
            root=path[0],
            train=True,
            download=False,
            transform=transform
        )
        trainData_tar = datasets.MNIST(
            root=path[1],
            transform=transform,
            train=True,
            download=False
        )

        return trainData_src, trainData_tar