import torch
from torchvision import datasets,transforms
from PIL import Image
import os
import torch.utils.data as data
class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


# **加载训练数据**
def loadTrainData(path,batch_size,datasetIndex,imageSize,kwargs):
    if datasetIndex == 0 or datasetIndex==5:
        transform=transforms.Compose(
            [
                transforms.Resize([256,256]),
                # 图片随机剪切，所写IRC，可以达到扩充数据，提高模型精度，增强模型稳定性，得到的是[224,224]图片
                transforms.RandomCrop(imageSize),
                # 按照一定的概率进行图片水平翻转，默认p=0.5，即一半的概率翻转，一半不翻转
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # BN操作
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        # 按照文件夹不同为训练数据加入类别标签
        trainData_src=datasets.ImageFolder(root=path[0],transform=transform)
        trainData_tar=datasets.ImageFolder(root=path[1],transform=transform)
    elif datasetIndex==1:
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_src = datasets.MNIST(
            root=path[1],
            transform=transform,
            train=True,
            download=True
        )
        trainData_tar=datasets.SVHN(
            root=path[0],
            transform=transform,
            split='train',
            download=True
        )
    elif datasetIndex==2:
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_src = datasets.MNIST(
            root=path[0],
            transform=transform,
            train=True,
            download=True
        )
        test_list = path + '/mnist_m_test_labels.txt'
        trainData_tar = GetLoader(
            data_root=path[1] + '/mnist_m_test',
            data_list=test_list,
            transform=transform
        )
    #USPS->MNIST
    elif datasetIndex==4:
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_src = datasets.MNIST(
            root=path[0],
            transform=transform,
            train=True,
            download=True
        )
        trainData_tar = datasets.ImageFolder(root=path[1], transform=transform)
    elif datasetIndex==6:
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_tar = datasets.MNIST(
            root=path[0],
            transform=transform,
            train=True,
            download=True
        )
        trainData_src = datasets.ImageFolder(root=path[1], transform=transform)
    # 得到训练数据集
    trainDataLoder_src=torch.utils.data.DataLoader(trainData_src,batch_size=batch_size,shuffle=True,drop_last=True,**kwargs)
    trainDataLoder_tar=torch.utils.data.DataLoader(trainData_tar,batch_size=batch_size,shuffle=True,drop_last=True,**kwargs)
    return trainDataLoder_src,trainDataLoder_tar

# **加载测试数据**
def loadTestData(path,batch_size,datasetIndex,imageSize,kwargs):
    if datasetIndex == 0 or datasetIndex==5:
        transform=transforms.Compose(
            [
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                # BN操作
                transforms.Normalize([0.485, 0.456, .406], [0.229, 0.224, 0.225])
            ]
        )
        testData=datasets.ImageFolder(root=path,transform=transform)
    elif datasetIndex==1 or datasetIndex==4:
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        testData = datasets.MNIST(
            root=path,
            transform=transform,
            train=False,
            download=True
        )
    elif datasetIndex==2:
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        test_list = path + '/mnist_m_test_labels.txt'
        testData = GetLoader(
            data_root=path + '/mnist_m_test',
            data_list=test_list,
            transform=transform
        )
    elif datasetIndex==6:
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        testData = datasets.ImageFolder(root=path, transform=transform)
    testDataLoder=torch.utils.data.DataLoader(testData,batch_size=batch_size,shuffle=True,drop_last=False,**kwargs)
    return testDataLoder
