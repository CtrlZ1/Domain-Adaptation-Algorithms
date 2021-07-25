
import torch
import model
import dataLoder
from CommenSetting import pathAndImageSize,args
from train import train, pre_train
from tes1t import tes1t
import matplotlib.pyplot as plt
import os

datasetRootAndImageSize=pathAndImageSize
# CPU
DEVICE=torch.device('cpu')
kwargs={}

# if use cuda
if args.isCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':

    # prepare data
    sourceTrainLoader, targetTrainLoader = dataLoder.loadTrainData(datasetRootAndImageSize[args.datasetIndex], args.batchSize,
                                                                   args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2],
                                                                   kwargs)
    targetTestLoader = dataLoder.loadTestData(datasetRootAndImageSize[args.datasetIndex][1], args.batchSize, args.datasetIndex,
                                              datasetRootAndImageSize[args.datasetIndex][2], kwargs)

    # prepare model
    if args.haveModel == True:
        net_W = torch.load(args.loadPath)
        net_g = torch.load(args.loadPath)
    else:


        if args.datasetIndex==1 or args.datasetIndex==2:
            net_W=model.Wnet_digistSMNet(args.numClass).to(DEVICE)

        net_g = model.KantorovichPotential().to(DEVICE)

    # for test precess
    correct=0
    # pre train
    pre_train(net_W,sourceTrainLoader,DEVICE,args)
    for epoch in range(1,args.epoch+1):
        #plt.figure(figsize=(10, 10))
        # plt.figure(figsize=(10, 10))
        W=train(epoch, net_W,net_g, sourceTrainLoader, targetTrainLoader,DEVICE,args)

        t_correct = tes1t(net_W, targetTestLoader,DEVICE)
        if t_correct > correct:
            correct = t_correct
        print("max correct:" , correct)
        print(datasetRootAndImageSize[args.datasetIndex][0], "to", datasetRootAndImageSize[args.datasetIndex][1])
        # if epoch % args.saveSteps==0:
        #     torch.save(model, '{}/model_office_caltech_10{}.pth'.format(args.savePath,epoch))
        # plt.show()