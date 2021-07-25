import torch
from torchvision import datasets,transforms
from PIL import Image
import os
import torch.utils.data as data

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
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
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        trainData_src = datasets.USPS(
            root=path[0],
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ]),
            train=True,
            download=True
        )
        testData_src = datasets.USPS(
            root=path[0],
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ]),
            train=False,
            download=True
        )
        trainData_tar = datasets.MNIST(
            root=path[1],
            transform=transform,
            train=True,
            download=False
        )
        testData_tar = datasets.MNIST(
            root=path[1],
            transform=transform,
            train=False,
            download=False
        )
        trainData_src=ConcatDataset([trainData_src, testData_src])
        trainData_tar=ConcatDataset([trainData_tar, testData_tar])
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

    return trainData_src,trainData_tar

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
            download=False
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
    return testData





class UniformSampler:
    """
    UniformSampler allows to sample batches in random manner without splitting the original data.
    """

    def __init__(self, *datasets, batch_size=1, n_batches=1):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.weights = [torch.ones(len(ds)) for ds in datasets]

    def __iter__(self):
        for i in range(self.n_batches):
            # torch.multinomial作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标。
            # 输入是一个input张量，一个取样数量，和一个布尔值replacement。
            idx = [torch.multinomial(w, self.batch_size, replacement=True) for w in self.weights]
            yield torch.stack(idx, dim=1)

    def __len__(self):
        return self.batch_size * self.n_batches


class ZipDataset(Dataset):
    """
    ZipDataset represents a dataset that stores several other datasets zipped together.
    """

    def __init__(self, *datasets, return_targets=False, return_idx=True):
        super().__init__()
        self.datasets = datasets
        self.return_targets = return_targets
        self.return_idx = return_idx

    def __getitem__(self, idx):
        items = []
        for i, ds in zip(idx, self.datasets):
            cur_items = []
            if self.return_idx:
                cur_items.append(i)
            cur_items.append(ds[i][0])
            if self.return_targets:
                cur_items.append(ds[i][1])  # gaussian返回None，MNIST返回标签
            items.append(cur_items)

        if len(items) == 1:
            items = items[0]

        return items

    def __len__(self):
        return np.prod([len(ds) for ds in self.datasets])


class ZipLoader(DataLoader):
    def __init__(self, *datasets, batch_size, n_batches, return_targets=False, return_idx=True, **kwargs):
        """
        ZipLoader allows to sample batches from zipped datasets with possibly different number of elements.
        """
        # 返回的是下标index
        us = UniformSampler(*datasets, batch_size=batch_size, n_batches=n_batches)
        dl = ZipDataset(*datasets, return_targets=return_targets, return_idx=return_idx)
        super().__init__(dl, batch_sampler=us, **kwargs)


def get_mean_covariance(mnist):
    def rescale(data):  # 放缩到-1到1。
        return 2 * (data.type(torch.float) / 255 - .5)

    if hasattr(mnist, 'data'):
        rescaled_data = rescale(mnist.data)
    elif hasattr(mnist, 'datasets'):
        rescaled_data = torch.cat([rescale(ds.data) for ds in mnist.datasets])
    else:
        raise ValueError('Argument ``mnist`` is invalid.')
    # 28*28=784，(n,784)
    rescaled_data = rescaled_data.reshape(len(rescaled_data), -1)
    # for ds in mnist.datasets:
    #     print(ds.data[0])
    # print(rescaled_data[0])
    # 返回平均数(784，每个像素的平均值）和协方差矩阵
    return torch.mean(rescaled_data, 0), torch.from_numpy(np.cov(rescaled_data.T).astype(np.float32))


def gaussian_sampler(mean, covariance, batch_size, n_batches, min_eigval=1e-3):
    eigval, eigvec = torch.symeig(covariance, eigenvectors=True)
    eigval, eigvec = eigval[eigval > min_eigval], eigvec[:, eigval > min_eigval]
    height = width = int(np.sqrt(len(mean)))

    for i in range(n_batches):
        samples = torch.randn(batch_size, len(eigval))
        samples = mean + (torch.sqrt(eigval) * samples) @ eigvec.T
        yield None, samples.reshape(-1, 1, height, width)


class DistributionDataset():
    def __init__(self, distribution, transform=None):
        super().__init__()
        self.distribution = distribution
        self.transform = transform

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.distribution.sample()), None
        else:
            return self.distribution.sample(), None

    def __len__(self):
        return 1


def get_rotation(theta):
    rad = np.radians(theta)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s],
                  [s, c]])
    return R


class CircleDataset():
    def __init__(self, n_samples, n_centers=9, sigma=0.02):
        super().__init__()
        self.nus = [torch.zeros(2)]
        self.sigma = sigma
        for i in range(n_centers - 1):
            R = get_rotation(i * 360 / (n_centers - 1))
            self.nus.append(torch.tensor([1, 0] @ R, dtype=torch.float))
        classes = torch.multinomial(torch.ones(n_centers), n_samples,
                                    replacement=True)

        data = []
        for i in range(n_centers):
            n_samples_class = torch.sum(classes == i)
            if n_samples_class == 0:
                continue
            dist = MultivariateNormal(self.nus[i],
                                      torch.eye(2) * self.sigma ** 2)
            data.append(dist.sample([n_samples_class.item()]))
        self.data = torch.cat(data)

    def __getitem__(self, idx):
        return self.data[idx], None

    def __len__(self):
        return self.data.shape[0]


class CentersDataset(Dataset):
    def __init__(self, n_centers=9):
        super().__init__()
        self.nus = [torch.zeros(2)]
        for i in range(n_centers - 1):
            R = get_rotation(i * 360 / (n_centers - 1))
            self.nus.append(torch.tensor([1, 0] @ R, dtype=torch.float))
        self.data = torch.stack(self.nus)

    def __getitem__(self, idx):
        return self.data[idx], None

    def __len__(self):
        return self.data.shape[0]


class CustomGaussian:
    def __init__(self, mean, covariance, min_eigval=1e-3):
        # 传入平均值和协方差矩阵
        self.mean = mean  # 平均值是每个样本每一列的平均值，[784]
        # 返回实对称矩阵covariance的特征值和特征向量，eigval，size:[n]，即一维数组，eigvec，size:[n,n]，即n维矩阵
        eigval, eigvec = torch.symeig(covariance, eigenvectors=True)
        self.eigval, self.eigvec = eigval[eigval > min_eigval], eigvec[:, eigval > min_eigval]
        # 正常没有筛选掉的话，就是[784],[784,784]
        self.height = self.width = int(np.sqrt(len(mean)))

    def sample(self):  # 开始随机取样
        x = torch.randn(1, len(self.eigval))
        # np.matmul(a, b)就是a @ b，查看@源代码就是矩阵相乘
        x = self.mean + (torch.sqrt(self.eigval) * x) @ self.eigvec.T
        return x