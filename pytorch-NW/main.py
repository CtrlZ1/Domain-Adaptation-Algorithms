import argparse
import torch
import model
import dataLoder
from train import train
from tes1t import tes1t
import matplotlib.pyplot as plt
import os

# parameter setting
parser=argparse.ArgumentParser()
parser.add_argument('--batchSize',type=int,default=256,metavar='batchSize',help='input the batch size of training process.(default=64)')
parser.add_argument('--epoch',type=int, default=200,metavar='epoch',help='the number of epochs for training.(default=100)')
parser.add_argument('--lr',type=float,default=2e-4,metavar='LR',help='the learning rate for training.(default=1e-2)')
parser.add_argument('--lrPi', type=float, default=0.0005, help='Learning rate.')
parser.add_argument('--momentum',type=float,default=0.9,metavar='M',help='SGD momentum.(default=0.5)')
parser.add_argument('--n_critic',type=int,default=5,help='the number of training critic before training others one time')
parser.add_argument('--n_labels',type=int,default=3,help='the number of labels for classifier task')
parser.add_argument('--n_clf',type=int,default=1,help='the number of training classifier after training critic one time')
parser.add_argument('--lambda_wd_clf',type=float,default=1,help='the lambda of wd_loss in the total loss of classifier')
parser.add_argument('--noCuda',action='store_true',default=False,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=233,metavar='S',help='random seed.(default:1)')
parser.add_argument('--n_dim',type=int,default=3)
parser.add_argument('--adam',type=bool,default=True)
parser.add_argument('--logInterval', type=int,default=10,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--l2Decay',type=float,default=5e-4,help='the L2 weight decay.(default=5e-4')
parser.add_argument('--diffLr',type=bool,default=True,help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--gamma',type=int,default=1,help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--numClass',default=10,type=int,help='the number of classes')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use')
parser.add_argument('--savePath', default='E:/毕设/model', type=str,help='the file to save models')
parser.add_argument('--saveSteps', default=50, type=int,help='the internal to save models')
parser.add_argument('--haveModel', default=False, type=bool,help='use the pretrained model')
parser.add_argument('--datasetIndex',type=int,default=8,help='chose the index of dataset by it')
parser.add_argument('--lambda_gp',type=int,default=10)


args=parser.parse_args()
datasetRootAndImageSize=[
    # office-31 a-w 0
    [r"E:\transferlearning\data\office-31\Original_images\amazon",r"E:\transferlearning\data\office-31\Original_images\webcam",224],
    # svhn->mnist 1
    [r'E:\transferlearning\data\SVHN',r'E:\transferlearning\data\MNIST',28],
    #mnist-mnist-m 2
    [r'E:\transferlearning\data\MNIST',r'E:\transferlearning\data\MNIST-M\mnist_m',28],
    #ImageCLEF 2014 3
    [r'E:\transferlearning\data\ImageCLEF 2014\b',r'E:\transferlearning\data\ImageCLEF 2014\c'],
    # usps-mnist 4
    [r'E:\transferlearning\data\usps',r'E:\transferlearning\data\MNIST',16],
    # office_caltech_10 a-w 5
    [r"E:\transferlearning\data\office_caltech_10\amazon",r"E:\transferlearning\data\office_caltech_10\webcam",224],
    # mnist-usps 6
    [r'E:\transferlearning\data\MNIST', r'E:\transferlearning\data\usps',16],
    # SVHN、USPS->MNIST 7
    [[r'E:\transferlearning\data\SVHN',r'E:\transferlearning\data\usps'], r'E:\transferlearning\data\MNIST',28],
    # MNIST -> MNIST-M 8
    ['data/MNIST-MNISTM/3mode/','data/MNIST-MNISTM/3mode/',28],

]
DEVICE=torch.device('cpu')
kwargs={}

if not args.noCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':

    if args.datasetIndex==8:
        allDataLoader= dataLoder.form_mnist_dataset(datasetRootAndImageSize[args.datasetIndex][0], datasetRootAndImageSize[args.datasetIndex][1], kwargs, args)
        sourceTrainLoader, sourceTestLoader,targetTrainLoader,targetTestLoader,s_trainloader_classwise=\
            allDataLoader['s_train'],allDataLoader['s_val'],allDataLoader['t_train'],allDataLoader['t_val'],allDataLoader['s_classwise']
    else:
        sourceTrainLoader, targetTrainLoader = dataLoder.loadTrainData(datasetRootAndImageSize[args.datasetIndex], args.batchSize,
                                                                       args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2],
                                                                       kwargs)
        targetTestLoader = dataLoder.loadTestData(datasetRootAndImageSize[args.datasetIndex][1], args.batchSize, args.datasetIndex,
                                                  datasetRootAndImageSize[args.datasetIndex][2], kwargs)
        s_trainloader_classwise= dataLoder.get_ClassWiseDataLoader(datasetRootAndImageSize[args.datasetIndex], args.batchSize, args.datasetIndex,
                                                                   datasetRootAndImageSize[args.datasetIndex][2], args.n_labels, kwargs)

    if args.haveModel == True:
        print(list(os.walk(args.savePath)))
        print(list(os.walk(args.savePath))[0][0] + '/' + list(os.walk(args.savePath))[0][2][0])
        model = torch.load(list(os.walk(args.savePath))[0][0] + '/' + list(os.walk(args.savePath))[0][2][0])
    else:
        model=model.NWModel(args).to(DEVICE)



    if args.datasetIndex==8:
        train(s_trainloader_classwise, model, [sourceTrainLoader,sourceTestLoader], [targetTrainLoader,targetTestLoader],targetTestLoader,
              datasetRootAndImageSize[args.datasetIndex][2], DEVICE, args)

    else:

        train(s_trainloader_classwise, model, sourceTrainLoader, targetTrainLoader,datasetRootAndImageSize[args.datasetIndex][2],DEVICE,args)
    tes1t(model, targetTestLoader,args,datasetRootAndImageSize[args.datasetIndex][2],DEVICE)

