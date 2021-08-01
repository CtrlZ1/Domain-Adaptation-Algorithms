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
    result['X'] = xtmp.dot(A) + b
    temp_y=[np.ones(round(h[i] * n))*i for i in range(args.n_labels)]
    result['y'] = np.concatenate(tuple(temp_y))
    return result

# **加载源域数据**
def loadSourceAndTargetData(path,batch_size,datasetIndex,imageSize,kwargs):
    if datasetIndex==7:
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        Data_tar = datasets.MNIST(
            root=path[1],
            transform=transform,
            train=True,
            download=False
        )
        Data_src2=datasets.SVHN(
            root=path[0][0],
            transform=transform,
            split='train',
            download=False
        )

        Data_src1 = datasets.USPS(
            root=path[0][1],
            transform=transform,
            train=True,
            download=False
        )

        DataLoder_src1 = torch.utils.data.DataLoader(Data_src1, batch_size=batch_size, shuffle=True,
                                                         drop_last=True, **kwargs)

        DataLoder_src2 = torch.utils.data.DataLoader(Data_src2, batch_size=batch_size, shuffle=True,
                                                     drop_last=True, **kwargs)
        DataLoder_tar = torch.utils.data.DataLoader(Data_tar, batch_size=batch_size, shuffle=True,
                                                         drop_last=True, **kwargs)

        return DataLoder_src1,DataLoder_src2,DataLoder_tar