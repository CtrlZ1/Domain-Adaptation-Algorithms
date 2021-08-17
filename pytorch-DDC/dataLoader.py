import numpy as np
import torch
from torchvision import datasets,transforms



def generateBiModGaussian(centers, sigma, n, A, b, h,args):
    result = {}
    temp_x=[centers[i]+sigma * np.random.standard_normal((round(h[i] * n), args.length_of_feature)) for i in range(args.n_labels)]
    xtmp = np.concatenate(tuple(temp_x))
    result['X'] = xtmp.dot(A) + b
    temp_y=[np.ones(round(h[i] * n))*i for i in range(args.n_labels)]
    result['y'] = np.concatenate(tuple(temp_y))
    return result

# **load training data**
def loadSourceAndTargetData(path,batch_size,datasetIndex,imageSize,kwargs):
    if datasetIndex==0:
        transform = transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.RandomCrop(imageSize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        trainData_src = datasets.ImageFolder(root=path[0], transform=transform)
        trainData_tar = datasets.ImageFolder(root=path[1], transform=transform)

        trainDataLoder_src = torch.utils.data.DataLoader(trainData_src, batch_size=batch_size, shuffle=True,
                                                         drop_last=True, **kwargs)
        trainDataLoder_tar = torch.utils.data.DataLoader(trainData_tar, batch_size=batch_size, shuffle=True,
                                                         drop_last=True, **kwargs)
        return trainDataLoder_src, trainDataLoder_tar

def loadTestData(path,batch_size,datasetIndex,imageSize,kwargs):
    if datasetIndex == 0 or datasetIndex==5:
        transform=transforms.Compose(
            [
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, .406], [0.229, 0.224, 0.225])
            ]
        )
        testData=datasets.ImageFolder(root=path,transform=transform)
        testDataLoder = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True, drop_last=False,
                                                    **kwargs)
        return testDataLoder