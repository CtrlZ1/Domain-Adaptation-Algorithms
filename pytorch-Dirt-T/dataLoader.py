import numpy as np
import torch
from torch.utils.data import Dataset
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
# data_name2path={
#     'SVHN':'/home/lyy/datas/digits/SVHN',
#     'MNIST':'/home/lyy/datas/digits/MNIST',
#     'USPS':'/home/lyy/datas/digits/usps',
#     'MNIST-M':'/home/lyy/datas/digits/mnist_m',
#     'syn':'/home/lyy/datas/digits/syn_number.mat',
#     'Office31-amazon':"/home/lyy/datas/office31/amazon",
#     'Office31-webcam':"/home/lyy/datas/office31/webcam",
#     'Office31-dslr':"/home/lyy/datas/office31/dslr",
#     'Officehome-Art':'/home/lyy/datas/office-home/Art',
#     'Officehome-Clipart':'/home/lyy/datas/office-home/Clipart',
#     'Officehome-Product':'/home/lyy/datas/office-home/Product',
#     'Officehome-Real World':'/home/lyy/datas/office-home/Real World',
#     'Office_celtech10-amazon':"/home/lyy/datas/office_caltech_10/amazon",
#     'Office_celtech10-caltech':"/home/lyy/datas/office_caltech_10/caltech",
#     'Office_celtech10-webcam':"/home/lyy/datas/office_caltech_10/webcam",
#     'Office_celtech10-dslr':"/home/lyy/datas/office_caltech_10/dslr"
# }
data_name2path={
    'SVHN':r'E:\transferlearning\data\SVHN',
    'MNIST':r'E:\transferlearning\data\MNIST',
    'USPS':r'E:\transferlearning\data\usps',
    'MNIST-M':r'E:\transferlearning\data\digits\mnist_m',
    'syn':r'E:\transferlearning\data\digit5\digit5\syn_number.mat',
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

class DefineDataset(Dataset):
    def __init__(self,sourcedata,sourcelabel,transform=None):
        self.sourcedata = sourcedata
        self.labels = sourcelabel
        self.lens = len(sourcedata)
        self.transform=transform
    def __getitem__(self, index):
        label = self.labels[index]
        img = self.sourcedata[index]
        if not torch.is_tensor(img):
            img=torch.from_numpy(img)
            label=torch.from_numpy(label)

        if self.transform is not None:
            img = self.transform(transforms.ToPILImage()(img))

        return img,label

    def __len__(self):
        return self.lens

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
        return [
            datasets.ImageFolder(root=data_name2path['MNIST-M']+r'/trainset', transform=transform_digits),
            datasets.ImageFolder(root=data_name2path['MNIST-M']+r'/testset', transform=transform_digits),
        ]

    if name=='syn':
        import scipy.io as scio
        train_data = torch.from_numpy(scio.loadmat(data_name2path['syn'])['train_data'])
        test_data = torch.from_numpy(scio.loadmat(data_name2path['syn'])['test_data'])
        train_label = torch.from_numpy(scio.loadmat(data_name2path['syn'])['train_label']).view((-1,))
        test_label = torch.from_numpy(scio.loadmat(data_name2path['syn'])['test_label']).view((-1,))

        train_data=torch.transpose(train_data, 1, 3)
        train_data=torch.transpose(train_data, 2, 3)
        test_data=torch.transpose(test_data, 1, 3)
        test_data=torch.transpose(test_data, 2, 3)

        return [
            DefineDataset(train_data,train_label,transform=transform_digits),
            DefineDataset(test_data,test_label,transform=transform_digits)
        ]
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
        self.stop = [False for i in range(len(self.sourcesDataLoader)+len(self.targetDataLoader))]
        self.iter=0
        return self

    def __next__(self):
        Datas=[]
        Labels=[]
        for index,i in enumerate(self.sourcesDataLoader_iter):
            try:
                sourceData,sourceLebel=next(i)
            except StopIteration:
                self.sourcesDataLoader_iter[index] = iter(self.sourcesDataLoader[index])
                sourceData, sourceLebel = next(self.sourcesDataLoader_iter[index])
                self.stop[index]=True
            Datas.append(sourceData)
            Labels.append(sourceLebel)
        try:
            targetData, targetLebel = next(self.targetDataLoader_iter)
        except StopIteration:
            # if targetData is None or targetLebel is None:
            self.targetDataLoader_iter = iter(self.targetDataLoader)
            targetData, targetLebel = next(self.targetDataLoader_iter)
            self.stop[-1]=True

        if sum(self.stop)==len(self.sourcesDataLoader)+len(self.targetDataLoader) or self.iter > self.max_dataset_size:
            self.stop = [False for i in range(len(self.sourcesDataLoader) + len(self.targetDataLoader))]
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
    # paired_data_test = PairedData(sourceTestDataLoaders,targetTestDataLoader,120)


    return paired_data_train,targetTestDataLoader
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