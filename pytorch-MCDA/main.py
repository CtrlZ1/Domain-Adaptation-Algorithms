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
parser.add_argument('--batchSize',type=int,default=128,metavar='batchSize',help='input the batch size of training process.(default=64)')
parser.add_argument('--epoch',type=int, default=1000,metavar='epoch',help='the number of epochs for training.(default=100)')
parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',help='the learning rate for training.(default=1e-2)')
parser.add_argument('--was_dim',type=int,default=100)
parser.add_argument('--momentum',type=float,default=0.9,metavar='M',help='SGD momentum.(default=0.5)')
parser.add_argument('--alpha',type=float,default=1e-4)
parser.add_argument('--n_critic',type=int,default=5,help='the number of training critic before training others one time')
parser.add_argument('--n_labels',type=int,default=10)
parser.add_argument('--n_dim',type=int,default=1)
parser.add_argument('--n_clf',type=int,default=1,help='the number of training classifier after training critic one time')
parser.add_argument('--lambda_wd_clf',type=float,default=1,help='the lambda of wd_loss in the total loss of classifier')
parser.add_argument('--noCuda',action='store_true',default=False,help='use cude in training process or do not.(default=Flase)')
parser.add_argument('--seed',type=int,default=233,metavar='S',help='random seed.(default:1)')
parser.add_argument('--logInterval', type=int,default=10,metavar='log',help='the interval to log for one time.(default=10)')
parser.add_argument('--l2Decay',type=float,default=5e-4,help='the L2 weight decay.(default=5e-4')
parser.add_argument('--diffLr',type=bool,default=True,help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--gamma',type=int,default=1,help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--numClass',default=10,type=int,help='the number of classes')
parser.add_argument('--gpu', default=0, type=int,help='the index of GPU to use')
parser.add_argument('--theta', default=0.9, type=float)
parser.add_argument('--savePath', default='E:/毕设/model', type=str,help='the file to save models')
parser.add_argument('--saveSteps', default=50, type=int,help='the internal to save models')
parser.add_argument('--haveModel', default=False, type=bool,help='use the pretrained model')
parser.add_argument('--imageSizeIndex',type=int,default=2,help='chose the index of imagesizes by it')
parser.add_argument('--datasetIndex',type=int,default=4,help='chose the index of dataset by it')
parser.add_argument('--lambda_gp',type=int,default=10)


args=parser.parse_args()
imageSize=[224,32,28]
datasetRoot=[
    # 源域，目标域 office-31目录 a-w
    [r"E:\transferlearning\data\office-31\Original_images\amazon",r"E:\transferlearning\data\office-31\Original_images\webcam"],
    # svhn->mnist
    [r'E:\transferlearning\data\SVHN',r'E:\transferlearning\data\MNIST'],
    #mnist-mnist-m
    [r'E:\transferlearning\data\MNIST',r'E:\transferlearning\data\MNIST-M\mnist_m'],
    #ImageCLEF 2014
    [r'E:\transferlearning\data\ImageCLEF 2014\b',r'E:\transferlearning\data\ImageCLEF 2014\c'],
    # usps-mnist
    [r'E:\transferlearning\data\usps',r'E:\transferlearning\data\MNIST'],
    # 源域，目标域 office_caltech_10目录 a-w
    [r"E:\transferlearning\data\office_caltech_10\amazon",r"E:\transferlearning\data\office_caltech_10\webcam"],
    # mnist-usps
    [r'E:\李沂洋毕设\data\MNIST', r'E:\李沂洋毕设\data\usps'],

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

    # 准备数据
    sourceTrainLoader, targetTrainLoader = dataLoder.loadTrainData(datasetRoot[args.datasetIndex], args.batchSize,
                                                                   args.datasetIndex, imageSize[args.imageSizeIndex],
                                                                   kwargs)
    targetTestLoader = dataLoder.loadTestData(datasetRoot[args.datasetIndex][1], args.batchSize, args.datasetIndex,
                                              imageSize[args.imageSizeIndex], kwargs)

    # 准备模型
    if args.haveModel == True:
        print(list(os.walk(args.savePath)))
        print(list(os.walk(args.savePath))[0][0] + '/' + list(os.walk(args.savePath))[0][2][0])

        model = torch.load(list(os.walk(args.savePath))[0][0] + '/' + list(os.walk(args.savePath))[0][2][0])



    else:


        if args.datasetIndex==4:
            mcdamodel=model.MCDAModel(args).to(DEVICE)


    # 用于测试
    correct=0

    for epoch in range(1,args.epoch+1):
        #plt.figure(figsize=(10, 10))
        # plt.figure(figsize=(10, 10))
        W=train(epoch, mcdamodel, sourceTrainLoader, targetTrainLoader,DEVICE,args)

        t_correct = tes1t(mcdamodel, targetTestLoader,DEVICE)
        if t_correct > correct:
            correct = t_correct
        print("max correct:" , correct)
        print(datasetRoot[args.datasetIndex][0], "to", datasetRoot[args.datasetIndex][1])
        # if epoch % args.saveSteps==0:
        #     torch.save(model, '{}/model_office_caltech_10{}.pth'.format(args.savePath,epoch))
        # plt.show()