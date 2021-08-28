import argparse


# parameter setting
parser=argparse.ArgumentParser()
parser.add_argument('--batchSize',type=int,default=500,metavar='batchSize',help='input the batch size of training process.(default=64)')
parser.add_argument('--sample_size',type=int,default=50,metavar='sample_size',help='the number of every class to chose in samples in every source for one batch')
parser.add_argument('--epoch',type=int, default=400,metavar='epoch',help='the number of epochs for training.(default=100)')
parser.add_argument('--lr',type=float,default=1e-5,metavar='LR',help='the learning rate for training.(default=1e-2)')
parser.add_argument('--n_source_simples',type=int,default=300,help='the number of each source_simples')
parser.add_argument('--n_target_simples',type=int,default=200,help='the number of each target_simples')
parser.add_argument('--length_of_feature',type=int,default=2,help='the length of feature')
parser.add_argument('--n_sources',type=int,default=5,help='the number of sources')
parser.add_argument('--n_labels',type=int,default=10,help='the number of sources and target labels')
parser.add_argument('--n_dim',type=int,default=1)
parser.add_argument('--sigma',type=float,default=0.1,help='the parameter used in the process of generating sources and target datas')

parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--lrf',type=float,default=0.0002,help='the lr of feature_extractor')
parser.add_argument('--lrc',type=float,default=0.0002,help='the lr of classifier')
parser.add_argument('--noCuda',action='store_true',default=False,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=233,metavar='S',help='random seed.(default:1)')
parser.add_argument('--logInterval', type=int,default=10,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--alpha',type=float,default=0.001)
parser.add_argument('--alpha2',type=float,default=0.0001)
parser.add_argument('--train_par',type=float,default=1.0)
parser.add_argument('--lam',type=float,default=1.0)
parser.add_argument('--l2_Decay',type=float,default=5e-4)
parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use')
parser.add_argument('--savePath', default='', type=str,help='the file to save models')
parser.add_argument('--loadPath', default='', type=str,help='the file to save models')
parser.add_argument('--saveSteps', default=50, type=int,help='the internal to save models')
parser.add_argument('--haveModel', default=False, type=bool,help='use the pretrained model')
parser.add_argument('--datasetIndex',type=int,default=4,help='chose the index of dataset by it')


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
    # usps->mnist 4
    [r'E:\transferlearning\data\usps',r'E:\transferlearning\data\MNIST',28],
    # office_caltech_10 w、a、d->c 5
    [r"E:\transferlearning\data\office_caltech_10\amazon",r"E:\transferlearning\data\office_caltech_10\webcam",224],
    # mnist-usps 6
    [r'E:\transferlearning\data\MNIST', r'E:\transferlearning\data\usps',16],
    # SVHN、USPS->MNIST 7
    [[r'E:\transferlearning\data\SVHN',r'E:\transferlearning\data\usps'], r'E:\transferlearning\data\MNIST',28],

]