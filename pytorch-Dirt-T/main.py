import argparse
import torch
import model
import dataLoader

# parameter setting
parser=argparse.ArgumentParser()
# model parameter
parser.add_argument('--batchSize',type=int,default=128,metavar='batchSize',help='input the batch size of training process.(default=64)')
parser.add_argument('--epoch',type=int, default=100,metavar='epoch',help='the number of epochs for training.(default=100)')
parser.add_argument('--lr',type=float,default=2e-4,metavar='LR',help='the learning rate for training.(default=1e-2)')
parser.add_argument('--n_critic',type=int,default=1,help='the number of training critic before training others one time.(default=5)')
parser.add_argument('--n_clf',type=int,default=1,help='the number of training classifier after training critic one time.(default=1)')
parser.add_argument('--n_preclf',type=int,default=10,help='the number of training classifier after training critic one time.(default=1)')
parser.add_argument('--n_labels',type=int,default=10,help='the number of sources and target labels')
parser.add_argument('--n_dim',type=int,default=3,help='the channels of images.(default=1)')

# hyperparameter

parser.add_argument('--radius', type=float, default=3.5,
                    help="Perturbation 2-norm ball radius")
parser.add_argument('--n_power', type=int, default=1,
                    help="gradient iterations")
parser.add_argument('--beta1', default=0.5, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help=', beta2 for adam')
parser.add_argument('--beta_t', default=10.0, type=float)
parser.add_argument('--lambda_t', default=0.1, type=float)
parser.add_argument('--large', default=False, type=bool,
                    help=', use large network')
parser.add_argument('--gaussian_noise', type=float, default=1.0,
                    help="noise for feature extractor")
# setting parameter
parser.add_argument('--lr_gamma', default=5, type=float, help='parameter for lr scheduler.(default=3e-4)')
parser.add_argument('--lr_decay', default=0.75, type=float, help='parameter for lr scheduler.(default=0.75)')
parser.add_argument('--logPath',type=str,default='logs/log.txt',help='the path of log files')

parser.add_argument('--gamma',type=float,default=10.0)
parser.add_argument('--noCuda',action='store_true',default=False,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=1999,metavar='S',help='random seed.(default:1)')
parser.add_argument('--logInterval', type=int,default=10,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use.(default=0)')
parser.add_argument('--data_name',type=str,default='Digits',help='the model')
parser.add_argument('--savePath',type=str,default='../checkpoints/',help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifsave',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--ifDirtt',default=True,type=bool)
parser.add_argument('--if_saveall',default=False,type=bool,help='if save all or just state.(default=False)')
parser.add_argument('--loadPath',default='../checkpoints/',type=str,help='the file to save models.(default=checkpoints/)')
parser.add_argument('--ifload',default=False,type=bool,help='the file to save models.(default=False)')
parser.add_argument('--usecheckpoints', default=False, type=bool,help='use the pretrained model.(default=False)')
parser.add_argument('--datasetIndex',type=int,default=1,help='chose the index of dataset by it')


args=parser.parse_args()
# the root of dataset and the size of images,[source data root, target data root, size of images]
datasetRootAndImageSize=[
    # office-31 a-w 0
    ["Office31-amazon","Office31-webcam",224],
    # svhn->mnist 1
    ['SVHN','MNIST',32],
    #mnist-mnist-m 2
    ['MNIST','MNIST-M',32],
    #ImageCLEF 2014 3
    ['ImageCLEF_2014-b','ImageCLEF_2014-c',32],
    # usps-mnist 4
    ['USPS','MNIST',32],
    # office_caltech_10 a-w 5
    ["Office_celtech10-caltech","Office_celtech10-webcam",224],
    # mnist-usps 6
    ['MNIST', 'USPS',32],
    # syn、MNIST-M、SVHN、USPS->MNIST 7
    [['syn', 'MNIST-M', 'SVHN', 'USPS'], 'MNIST', 32],

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
    sourceTrainDataLoader, sourceTestDataLoader, targetTrainDataLoader, targetTestDataLoader = \
        dataLoader.singleSourceDataLoader(args.batchSize, datasetRootAndImageSize[args.datasetIndex][0],
                                          datasetRootAndImageSize[args.datasetIndex][1], kwargs)

    VADA=model.VADAModel(args).to(DEVICE)

    model.train_process(VADA, sourceTrainDataLoader, targetTestDataLoader, targetTestDataLoader, DEVICE, datasetRootAndImageSize[args.datasetIndex][2], args)


