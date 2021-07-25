import argparse
# parameter setting
parser=argparse.ArgumentParser()
parser.add_argument('--batchSize',type=int,default=32,metavar='batchSize',help='input the batch size of training process.(default=64)')
parser.add_argument('--epoch',type=int, default=300,metavar='epoch',help='the number of epochs for training.(default=100)')
parser.add_argument('--lr',type=float,default=1e-5,metavar='LR',help='the learning rate for training.(default=1e-2)')
parser.add_argument('--n_pretrain',type=int,default=1,help='the number of pre-training for classifier')
parser.add_argument('--n_g',type=int,default=5,help='the number of KantorovichPotential train every batch')
parser.add_argument('--Epsilon',type=float,default=1.0,help='the hyperparameter')
parser.add_argument('--isCuda',action='store_true',default=True,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=233,metavar='S',help='random seed.(default:1)')
parser.add_argument('--logInterval', type=int,default=50,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--diffLr',type=bool,default=True,help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--opt_lambda',type=float,default=2.5)
parser.add_argument('--beta',type=float,default=1e-1)
parser.add_argument('--numClass',default=10,type=int,help='the number of classes')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use')
parser.add_argument('--savePath', default='', type=str,help='the file to save models')
parser.add_argument('--loadPath', default='', type=str,help='the file to save models')
parser.add_argument('--saveSteps', default=50, type=int,help='the internal to save models')
parser.add_argument('--haveModel', default=False, type=bool,help='use the pretrained model')
parser.add_argument('--datasetIndex',type=int,default=4,help='chose the index of dataset by it')


args=parser.parse_args()

pathAndImageSize=[
    # 源域，目标域 office-31目录 a-w
    [r"E:\transferlearning\data\office-31\Original_images\amazon",r"E:\transferlearning\data\office-31\Original_images\webcam",224],
    # svhn->mnist
    [r'E:\transferlearning\data\SVHN',r'E:\transferlearning\data\MNIST',28],
    #mnist-mnist-m
    [r'E:\transferlearning\data\MNIST',r'E:\transferlearning\data\MNIST-M\mnist_m',28],
    #ImageCLEF 2014
    [r'E:\transferlearning\data\ImageCLEF 2014\b',r'E:\transferlearning\data\ImageCLEF 2014\c'],
    # usps-mnist
    [r'E:\transferlearning\data\usps',r'E:\transferlearning\data\MNIST',16],
    # 源域，目标域 office_caltech_10目录 a-w
    [r"E:\transferlearning\data\office_caltech_10\amazon",r"E:\transferlearning\data\office_caltech_10\webcam",224],
    # mnist-usps
    [r'E:\transferlearning\data\MNIST', r'E:\transferlearning\data\usps',16],

]