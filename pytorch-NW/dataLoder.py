import torch
from torchvision import datasets,transforms
from PIL import Image
import os
import torch.utils.data as data
from torch.utils.data import Dataset,DataLoader

from folder import ImageFolder
from utils import form_samples_classwise, read_file
import os.path as osp
import torchvision.transforms as T
import numpy as np
# Datasets for MNIST -> MNIST-M experiments
def form_mnist_dataset(sourcepath,targetpath,kwargs,args):

    s_train = read_file(osp.join(sourcepath, 'source_train.txt'), data_root='E:/transferlearning/data/digits/')
    s_val = read_file(osp.join(sourcepath, 'source_val.txt'), data_root='E:/transferlearning/data/digits/')
    t_train = read_file(osp.join(targetpath, 'target_train.txt'), data_root='E:/transferlearning/data/digits/')
    t_val = read_file(osp.join(targetpath, 'target_val.txt'), data_root='E:/transferlearning/data/digits/')
    data_samples = {
        'source_train': s_train,
        'source_val': s_val,
        'target_train': t_train,
        'target_val': t_val
    }

    all_labels = np.array([int(s[1]) for s in s_train])
    nclasses = len(np.unique(all_labels))
    print('Number of classes: {}'.format(nclasses))

    dataset_mean = (0.5, 0.5, 0.5)
    dataset_std = (0.5, 0.5, 0.5)

    source_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=dataset_mean,
                    std=dataset_std)
    ])
    target_transform_train = T.Compose([
        T.RandomCrop(28),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean,
                    std=dataset_std)
    ])
    target_transform_test = T.Compose([
        T.CenterCrop(28),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean,
                    std=dataset_std)
    ])

    dat_s_train = ImageFolder(data_samples['source_train'],
                              transform=source_transform)
    dat_s_val = ImageFolder(data_samples['source_val'],
                            transform=source_transform)
    dat_t_train = ImageFolder(data_samples['target_train'],
                              transform=target_transform_train)
    dat_t_val = ImageFolder(data_samples['target_val'],
                            transform=target_transform_test)

    s_trainloader = DataLoader(dataset=dat_s_train,
                               batch_size=args.batchSize,
                               shuffle=True,
                               drop_last=True,
                               **kwargs)
    s_valloader = DataLoader(dataset=dat_s_val,
                             batch_size=args.batchSize,
                             shuffle=False,
                             drop_last=False,
                             **kwargs)
    t_trainloader = DataLoader(dataset=dat_t_train,
                               batch_size=args.batchSize,
                               shuffle=True,
                               drop_last=True,
                               **kwargs)
    t_valloader = DataLoader(dataset=dat_t_val,
                             batch_size=args.batchSize,
                             shuffle=False,
                             drop_last=False,
                             **kwargs)

    data_samples_classwise = form_samples_classwise(data_samples['source_train'],nclasses,8)
    s_trainloader_classwise = [
        DataLoader(
            ImageFolder(data_samples_classwise[cl],
                        transform=source_transform),
            batch_size=args.batchSize,
            shuffle=True,
            drop_last=True,
            **kwargs) for cl in range(nclasses)
    ]

    dataloaders = {
        's_train': s_trainloader,
        's_val': s_valloader,
        't_train': t_trainloader,
        't_val': t_valloader,
        's_classwise': s_trainloader_classwise,
        'nclasses': nclasses
    }
    return dataloaders


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
            transforms.Grayscale(),
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


def get_ClassWiseDataLoader(path,batch_size,datasetIndex,imageSize,n_labels,kwargs):
    if datasetIndex == 1:
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        trainData_src = datasets.SVHN(
            root=path[0],
            transform=transform,
            split='train',
            download=True
        )
    if datasetIndex == 4:
        transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        trainData_src = datasets.USPS(
            root=path[0],
            transform=transform,
            train=True,
            download=True
        )
    sourcedata_samples_classwise, sourcelabels_samples_classwise = form_samples_classwise(trainData_src, n_labels,datasetIndex)
    s_trainloader_classwise = [
        DataLoader(
            sourceClassWiseDataset(sourcedata_samples_classwise[cl], sourcelabels_samples_classwise[cl],
                                   transform=transform),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **kwargs) for cl in range(10)
    ]

    return s_trainloader_classwise