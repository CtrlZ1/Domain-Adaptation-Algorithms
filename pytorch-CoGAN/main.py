import argparse
import torch
import model
import dataLoder

# parameter setting
parser=argparse.ArgumentParser()
# model parameter
parser.add_argument('--batchSize',type=int,default=64,metavar='batchSize',help='input the batch size of training process.(default=64)')
parser.add_argument('--epoch',type=int, default=1000,metavar='epoch',help='the number of epochs for training.(default=100)')
parser.add_argument('--lr',type=float,default=0.0002,metavar='LR',help='the learning rate for training.(default=1e-2)')
parser.add_argument('--n_critic',type=int,default=1,help='the number of training critic before training others one time.(default=5)')
parser.add_argument('--n_clf',type=int,default=10,help='the number of training classifier after training critic one time.(default=1)')
parser.add_argument('--n_preclf',type=int,default=10,help='the number of training classifier after training critic one time.(default=1)')
parser.add_argument('--n_labels',type=int,default=10,help='the number of sources and target labels')
parser.add_argument('--n_dim',type=int,default=1,help='the channels of images.(default=1)')
parser.add_argument('--latent_dims',type=int,default=100,help='the Number of neurons in generator.(default=100)')

# hyperparameter
parser.add_argument('--beta1',type=float,default=0.5,help='for adam.(default=0.5)')
parser.add_argument('--beta2',type=float,default=0.999,help='for adam.(default=0.999)')
parser.add_argument('--scale',type=float,default=2.0,help='for save images.(default=2.0)')
parser.add_argument('--bias',type=float,default=0.5,help='for save images.(default=0.5)')
parser.add_argument('--cls_weight',type=float,default=10.0,help='the Hyperparameter to weight cls and mse.(default=10.0)')
parser.add_argument('--mse_weight',type=float,default=0.01,help='the Hyperparameter to weight cls and mse.(default=0.01)')
# setting parameter
parser.add_argument('--noCuda',action='store_true',default=False,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=1999,metavar='S',help='random seed.(default:1)')
parser.add_argument('--logInterval', type=int,default=50,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use.(default=0)')
parser.add_argument('--data_name',type=str,default='Digits',help='the model')
parser.add_argument('--savePath',type=str,default='../checkpoints/',help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifsave',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--if_saveall',default=False,type=bool,help='if save all or just state.(default=False)')
parser.add_argument('--loadPath',default='../checkpoints/',type=str,help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifload',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--usecheckpoints', default=False, type=bool,help='use the pretrained model.(default=False)')
parser.add_argument('--datasetIndex',type=int,default=8,help='chose the index of dataset by it')


args=parser.parse_args()
# the root of dataset and the size of images,[source data root, target data root, size of images]
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
    [r'E:\transferlearning\data\usps',r'E:\transferlearning\data\MNIST',28],
    # office_caltech_10 a-w 5
    [r"E:\transferlearning\data\office_caltech_10\caltech",r"E:\transferlearning\data\office_caltech_10\webcam",224],
    # mnist-usps 6
    [r'E:\transferlearning\data\MNIST', r'E:\transferlearning\data\usps',16],
    # SVHN、USPS->MNIST 7
    [[r'E:\transferlearning\data\SVHN',r'E:\transferlearning\data\usps'], r'E:\transferlearning\data\MNIST',28],
    # MNIST->USPS 8
    [r'E:\transferlearning\data\MNIST', r'E:\transferlearning\data\usps',28]

]

# 默认CPU
DEVICE=torch.device('cpu')
kwargs={}

# 如果Cuda可用就用Cuda
if not args.noCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':

    sourceTrainLoader, targetTrainLoader = dataLoder.loadTrainData(datasetRootAndImageSize[args.datasetIndex], args.batchSize,
                                                                   args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2],
                                                                   kwargs)
    sourceTestLoader, targetTestLoader = dataLoder.loadTestData(datasetRootAndImageSize[args.datasetIndex], args.batchSize, args.datasetIndex,
                                              datasetRootAndImageSize[args.datasetIndex][2], kwargs)


    addamodel=model.CoGANModel(args).to(DEVICE)

    model.train_process(addamodel, sourceTrainLoader, targetTrainLoader, sourceTestLoader,targetTestLoader,DEVICE, datasetRootAndImageSize[args.datasetIndex][2], args)

    model.test_process(addamodel, sourceTestLoader, targetTestLoader, DEVICE, args)

