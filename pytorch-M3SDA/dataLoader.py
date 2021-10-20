import numpy as np
import torch
from torchvision import datasets,transforms
transform_office = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

transform_digits = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

data_name2path={
    'SVHN':r'E:\transferlearning\data\SVHN',
    'MNIST':r'E:\transferlearning\data\MNIST',
    'USPS':r'E:\transferlearning\data\usps',
    'MNIST-M':r'E:\transferlearning\data\MNIST-M\mnist_m',
    'Office31-amazon':r"E:\transferlearning\data\office-31\Original_images\amazon",
    'Office31-webcam':r"E:\transferlearning\data\office-31\Original_images\webcam",
    'Office31-dslr':r"E:\transferlearning\data\office-31\Original_images\dslr",
    'Officehome-Art':r'E:\transferlearning\data\office-home\Art',
    'Officehome-Clipart':r'E:\transferlearning\data\office-home\Clipart',
    'Officehome-Product':r'E:\transferlearning\data\office-home\Product',
    'Officehome-Real World':r'E:\transferlearning\data\office-home\Real World',
    'Office_celtech10-amazon':r"E:\transferlearning\data\office_caltech_10\amazon",
    'Office_celtech10-caltech':r"E:\transferlearning\data\office_caltech_10\caltech",
    'Office_celtech10-webcam':r"E:\transferlearning\data\office_caltech_10\webcam",
    'Office_celtech10-dslr':r"E:\transferlearning\data\office_caltech_10\dslr",
    'ImageCLEF_2014-b':r'E:\transferlearning\data\ImageCLEF 2014\b',
    'ImageCLEF_2014-c':r'E:\transferlearning\data\ImageCLEF 2014\c',
    'ImageCLEF_2014-i':r'E:\transferlearning\data\ImageCLEF 2014\i',
    'ImageCLEF_2014-p':r'E:\transferlearning\data\ImageCLEF 2014\p',

}

def data_name2dataset(name):
    if name=='SVHN':
        return [
            datasets.SVHN(
                root=data_name2path['SVHN'],
                transform=transform_digits,
                split='train',
                download=True
            ),
            datasets.SVHN(
                root=data_name2path['SVHN'],
                transform=transform_digits,
                split='test',
                download=True
            )
        ]
    if name=='MNIST':
        return [
            datasets.MNIST(
                root=data_name2path['MNIST'],
                transform=transform_digits,
                train=True,
                download=True
            ),
            datasets.MNIST(
                root=data_name2path['MNIST'],
                transform=transform_digits,
                train=False,
                download=True
            )
        ]
    if name=='USPS':
        return [
            datasets.USPS(
                root=data_name2path['USPS'],
                transform=transform_digits,
                train=True,
                download=True
            ),
            datasets.USPS(
                root=data_name2path['USPS'],
                transform=transform_digits,
                train=False,
                download=True
            )
        ]
    if name=='MNIST-M':
        return None
    if name=='Office31-amazon':
        return[
            datasets.ImageFolder(root=data_name2path['Office31-amazon'], transform=transform_office),
            datasets.ImageFolder(root=data_name2path['Office31-amazon'], transform=transform_office),
        ]
    if name=='Office31-webcam':
        return [
            datasets.ImageFolder(root=data_name2path['Office31-webcam'], transform=transform_office),
            datasets.ImageFolder(root=data_name2path['Office31-webcam'], transform=transform_office),
        ]
    if name=='Office31-dslr':
        return [
            datasets.ImageFolder(root=data_name2path['Office31-dslr'], transform=transform_office),
            datasets.ImageFolder(root=data_name2path['Office31-dslr'], transform=transform_office),
        ]
    if name=='Officehome-Art':
        return [
            datasets.ImageFolder(root=data_name2path['Officehome-Art'], transform=transform_office),
            datasets.ImageFolder(root=data_name2path['Officehome-Art'], transform=transform_office),
        ]
    if name=='Officehome-Clipart':
        return [
            datasets.ImageFolder(root=data_name2path['Officehome-Clipart'], transform=transform_office),
            datasets.ImageFolder(root=data_name2path['Officehome-Clipart'], transform=transform_office),
        ]
    if name=='Officehome-Product':
        return [
            datasets.ImageFolder(root=data_name2path['Officehome-Product'], transform=transform_office),
            datasets.ImageFolder(root=data_name2path['Officehome-Product'], transform=transform_office),
        ]
    if name=='Officehome-Real World':
        return [
            datasets.ImageFolder(root=data_name2path['Officehome-Real World'], transform=transform_office),
            datasets.ImageFolder(root=data_name2path['Officehome-Real World'], transform=transform_office),
        ]


class PairedData(object):
    def __init__(self, sourcesDataLoader, targetDataLoader,max_dataset_size=None):
        self.sourcesDataLoader = sourcesDataLoader
        self.targetDataLoader = targetDataLoader
        self.stop=[False for i in range(len(sourcesDataLoader)+len(targetDataLoader))]
        self.max_dataset_size = max_dataset_size


    def __iter__(self):


        self.sourcesDataLoader_iter = [iter(i)for i in self.sourcesDataLoader]
        self.targetDataLoader_iter = iter(self.targetDataLoader)
        self.stop = [False * (len(self.sourcesDataLoader) + len(self.targetDataLoader))]
        self.iter=0
        return self

    def __next__(self):
        Datas=[]
        Labels=[]
        for index,i in enumerate(self.sourcesDataLoader_iter):
            try:
                sourceData,sourceLebel=next(i)
            except StopIteration:
                if sourceData is None or sourceLebel is None:

                    self.sourcesDataLoader_iter[index] = iter(self.sourcesDataLoader[index])
                    sourceData, sourceLebel = next(self.sourcesDataLoader_iter[index])
                    self.stop[index]=True
            Datas.append(sourceData)
            Labels.append(sourceLebel)
        try:
            targetData, targetLebel = next(self.targetDataLoader_iter)
        except StopIteration:
            if targetData is None or targetLebel is None:
                self.targetDataLoader_iter = iter(self.targetDataLoader)
                targetData, targetLebel = next(self.targetDataLoader_iter)
                self.stop[-1]=True

        if sum(self.stop)==len(self.sourcesDataLoader)+len(self.targetDataLoader) or self.iter > self.max_dataset_size:
            self.stop = [False for i in range(len(self.sourcesDataLoader) + len(self.targetDataLoader))]
            self.iter=0
            raise StopIteration()
        else:
            self.iter+=1
            Datas.append(targetData)
            Labels.append(targetLebel)


            return Datas,Labels



def generateBiModGaussian(centers, sigma, n, A, b, h,args):
    result = {}
    temp_x=[centers[i]+sigma * np.random.standard_normal((round(h[i] * n), args.length_of_feature)) for i in range(args.n_labels)]
    xtmp = np.concatenate(tuple(temp_x))
    result['X'] = xtmp.dot(A) + b
    temp_y=[np.ones(round(h[i] * n))*i for i in range(args.n_labels)]
    result['y'] = np.concatenate(tuple(temp_y))
    return result


def multipleDataLoader(batch_size,sources,target,kwargs):
    sourceTrainDataLoaders=[]
    sourceTestDataLoaders=[]

    for i in sources:
        DataSets=data_name2dataset(i)
        sourceTrainDataLoaders.append(torch.utils.data.DataLoader(DataSets[0], batch_size=batch_size, shuffle=True,
                                                     drop_last=True, **kwargs))
        sourceTestDataLoaders.append(torch.utils.data.DataLoader(DataSets[1], batch_size=batch_size, shuffle=True,
                                                                  drop_last=False, **kwargs))
    targetDataSets = data_name2dataset(target)
    targetTrainDataLoaders=torch.utils.data.DataLoader(targetDataSets[0], batch_size=batch_size, shuffle=True,
                                                              drop_last=True, **kwargs)
    targetTestDataLoader=torch.utils.data.DataLoader(targetDataSets[1], batch_size=batch_size, shuffle=True,
                                                             drop_last=False, **kwargs)

    paired_data_train = PairedData(sourceTrainDataLoaders,targetTrainDataLoaders,120)
    paired_data_test = PairedData(sourceTestDataLoaders,targetTestDataLoader,120)


    return paired_data_train,paired_data_test
# **load training data**
def singleSourceDataLoader(batch_size,source,target,kwargs):
    DataSets = data_name2dataset(source)
    sourceTrainDataLoader=(torch.utils.data.DataLoader(DataSets[0], batch_size=batch_size, shuffle=True,
                                                              drop_last=True, **kwargs))
    sourceTestDataLoader=(torch.utils.data.DataLoader(DataSets[1], batch_size=batch_size, shuffle=True,
                                                             drop_last=False, **kwargs))
    DataSets = data_name2dataset(target)
    targetTrainDataLoader = (torch.utils.data.DataLoader(DataSets[0], batch_size=batch_size, shuffle=True,
                                                         drop_last=True, **kwargs))
    targetTestDataLoader = (torch.utils.data.DataLoader(DataSets[1], batch_size=batch_size, shuffle=True,
                                                        drop_last=False, **kwargs))

    return sourceTrainDataLoader, sourceTestDataLoader,targetTrainDataLoader,targetTestDataLoader