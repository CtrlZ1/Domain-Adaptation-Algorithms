import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets,transforms
from PIL import Image
import os
import torch.utils.data as data
class sourceClassWiseDataset(Dataset):
    def __init__(self,sourcedata,sourcelabel,transform=None):
        #   传入参数
        #   ndarray 类型的，可以是任何类型的
        self.sourcedata = sourcedata
        self.labels = sourcelabel
        self.lens = len(sourcedata)
        self.transform=transform
    def __getitem__(self, index):
        # index是方法自带的参数，获取相应的第index条数据
        label = torch.tensor(self.labels[index])
        img=torch.tensor(self.sourcedata[index])

        if self.transform is not None:
            img = self.transform(transforms.ToPILImage()(img))
        return img,label

    def __len__(self):
        return self.lens


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


    if datasetIndex==6:
        transform = transforms.Compose([
            transforms.Resize((imageSize,imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_tar = datasets.USPS(
            root=path[1],
            train=True,
            download=False,
            transform=transform
        )
        trainData_src = datasets.MNIST(
            root=path[0],
            transform=transform,
            train=True,
            download=False
        )

        return trainData_src, trainData_tar