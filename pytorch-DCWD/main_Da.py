
import torch


import model
import dataLoader
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from CommonSettings import args,datasetRootAndImageSize

np.random.seed(2021)
torch.manual_seed(2021)

# CPU
from backBone import network_dict


DEVICE=torch.device('cpu')
kwargs={}

# if use cuda
if not args.noCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())


if __name__ == '__main__':

    if args.datasetIndex==5:
        # prepare data
        Data_src1, Data_src2,Data_src3,Data_tar = dataLoader.loadSourceAndTargetData(datasetRootAndImageSize[args.datasetIndex],
                                                                       args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2])

        trainDataLoder_src1 = torch.utils.data.DataLoader(Data_src1, batch_size=args.batchSize, shuffle=True,
                                                         drop_last=True, **kwargs)
        trainDataLoder_src2 = torch.utils.data.DataLoader(Data_src2, batch_size=args.batchSize, shuffle=True,
                                                         drop_last=True, **kwargs)
        trainDataLoder_src3 = torch.utils.data.DataLoader(Data_src3, batch_size=args.batchSize, shuffle=True,
                                                         drop_last=True, **kwargs)
        trainDataLoder_tar = torch.utils.data.DataLoader(Data_tar, batch_size=args.batchSize, shuffle=True,
                                                         drop_last=True, **kwargs)


        # extracting features
        sourcedata1=[]
        sourcelabel1=[]
        sourcedata2=[]
        sourcelabel2 = []
        sourcedata3=[]
        sourcelabel3 = []
        targetdata=[]
        targetlabel = []
        print("extracting features...")
        feature_ext=network_dict['ResNet50']().to(DEVICE)
        lenSourceDataLoader=len(trainDataLoder_src1)
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(trainDataLoder_src1), total=lenSourceDataLoader,
                                                              desc='source id = 1', ncols=80,
                                                              leave=False):
            sourceData=sourceData.float().to(DEVICE)
            sourcedata1.append(feature_ext(sourceData).detach().cpu().numpy())
            sourcelabel1.append(sourceLabel.numpy())
        sourcedata1=np.concatenate(np.array(sourcedata1))
        sourcelabel1=np.concatenate(np.array(sourcelabel1))

        lenSourceDataLoader = len(trainDataLoder_src2)
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(trainDataLoder_src2), total=lenSourceDataLoader,
                                                              desc='source id = 2', ncols=80,
                                                              leave=False):
            sourceData = sourceData.float().to(DEVICE)
            sourcedata2.append(feature_ext(sourceData).detach().cpu().numpy())
            sourcelabel2.append(sourceLabel.numpy())
        sourcedata2 = np.concatenate(np.array(sourcedata2))
        sourcelabel2 = np.concatenate(np.array(sourcelabel2))

        lenSourceDataLoader = len(trainDataLoder_src3)
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(trainDataLoder_src3), total=lenSourceDataLoader,
                                                              desc='source id = 3', ncols=80,
                                                              leave=False):
            sourceData = sourceData.float().to(DEVICE)
            sourcedata3.append(feature_ext(sourceData).detach().cpu().numpy())
            sourcelabel3.append(sourceLabel.numpy())
        sourcedata3 = np.concatenate(np.array(sourcedata3))
        sourcelabel3 = np.concatenate(np.array(sourcelabel3))

        lenTargetDataLoader = len(trainDataLoder_tar)
        for batch_idx, (targetData, targetLabel) in tqdm.tqdm(enumerate(trainDataLoder_tar), total=lenTargetDataLoader,
                                                              desc='target', ncols=80,
                                                              leave=False):
            targetData = targetData.float().to(DEVICE)
            targetdata.append(feature_ext(targetData).detach().cpu().numpy())
            targetlabel.append(targetLabel.numpy())
        targetdata = np.concatenate(np.array(targetdata))
        targetlabel = np.concatenate(np.array(targetlabel))

        print("feature down,then train...")
        model= model.WJDOTModel()
        model.modeltrain([sourcedata1,sourcedata2,sourcedata3],[sourcelabel1,sourcelabel2,sourcelabel3],
                         targetdata,targetlabel,args.rat_labeled,args.Omega)

