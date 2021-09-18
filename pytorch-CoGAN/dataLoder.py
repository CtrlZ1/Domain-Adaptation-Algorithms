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
def loadTrainData(path,batch_size,datasetIndex,imageSize,kwargs):
    if datasetIndex == 0 or datasetIndex==5:
        transform = transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.RandomCrop((imageSize,imageSize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        trainData_src = datasets.ImageFolder(root=path[0], transform=transform)
        trainData_tar = datasets.ImageFolder(root=path[1], transform=transform)


    # USPS->MNIST
    elif datasetIndex == 4:
        transform = transforms.Compose([
            transforms.Resize((imageSize,imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_tar = datasets.MNIST(
            root=path[1],
            transform=transform,
            train=True,
            download=True
        )
        trainData_src = datasets.USPS(
            root=path[0],
            transform=transform,
            train=True,
            download=True
        )
    # SVHN→MNIST
    elif datasetIndex==1:
        transform = transforms.Compose([
            transforms.Resize((imageSize,imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_tar = datasets.MNIST(
            root=path[1],
            transform=transform,
            train=True,
            download=True
        )
        trainData_src=datasets.SVHN(
            root=path[0],
            transform=transform,
            split='train',
            download=True
        )

    # MNIST->USPS
    elif datasetIndex == 8:
        transform = transforms.Compose([
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_tar = datasets.USPS(
            root=path[1],
            transform=transform,
            train=True,
            download=True
        )
        trainData_src = datasets.MNIST(
            root=path[0],
            transform=transform,
            train=True,
            download=True
        )
    trainDataLoder_src = torch.utils.data.DataLoader(trainData_src, batch_size=batch_size, shuffle=True,
                                                     drop_last=True, **kwargs)
    trainDataLoder_tar = torch.utils.data.DataLoader(trainData_tar, batch_size=batch_size, shuffle=True,
                                                     drop_last=True, **kwargs)
    return trainDataLoder_src, trainDataLoder_tar
def loadTestData(path,batch_size,datasetIndex,imageSize,kwargs):
    if datasetIndex == 0 or datasetIndex==5:
        transform=transforms.Compose(
            [
                transforms.Resize((imageSize,imageSize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, .406], [0.229, 0.224, 0.225])
            ]
        )
        testData_src=datasets.ImageFolder(root=path[0],transform=transform)
        testData_tar=datasets.ImageFolder(root=path[1],transform=transform)

    elif datasetIndex==4:
        transform = transforms.Compose([
            transforms.Resize((imageSize,imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        testData_src = datasets.USPS(
            root=path[0],
            train=False,
            transform=transform,
            download=True
        )
        testData_tar = datasets.MNIST(
            root=path[1],
            transform=transform,
            train=False,
            download=True
        )
    elif datasetIndex == 1:
        transform = transforms.Compose([
            transforms.Resize((imageSize,imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        testData_src = datasets.SVHN(
            root=path[0],
            transform=transform,
            split='test',
            download=True
        )
        testData_tar = datasets.MNIST(
            root=path[1],
            transform=transform,
            train=False,
            download=True
        )

    # MNIST->USPS
    elif datasetIndex == 8:
        transform = transforms.Compose([
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        testData_tar = datasets.USPS(
            root=path[1],
            transform=transform,
            train=False,
            download=True
        )
        testData_src = datasets.MNIST(
            root=path[0],
            transform=transform,
            train=False,
            download=True
        )
    testDataLoder_src = torch.utils.data.DataLoader(testData_src, batch_size=batch_size, shuffle=True,
                                                     drop_last=False, **kwargs)
    testDataLoder_tar = torch.utils.data.DataLoader(testData_tar, batch_size=batch_size, shuffle=True,
                                                     drop_last=False, **kwargs)
    return testDataLoder_src, testDataLoder_tar