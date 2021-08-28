
import torch


import model
import dataLoader
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from CommonSettings import args,datasetRootAndImageSize



# CPU

DEVICE=torch.device('cpu')
kwargs={}

# if use cuda
if not args.noCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':
    # load data

    source, target = dataLoader.loadSourceAndTargetData(datasetRootAndImageSize[args.datasetIndex],
                                                                             args.datasetIndex,datasetRootAndImageSize[args.datasetIndex][2])

    source_loader = torch.utils.data.DataLoader(dataset=source, batch_size=64, shuffle=True)
    target_loader = torch.utils.data.DataLoader(dataset=target, batch_size=64, shuffle=True)

    mymodel = model.DeepJDOT(args.n_labels, DEVICE)

    # mymodel.sourcemodel_usedin_target(mymodel.feature_ext_digits,mymodel.classifier_digits,source_loader,target_loader,args.lrf,args.lrc,20,args.n_dim,args.logInterval)

    # use DeepJDOT model to predict target
    mymodel.train_process(mymodel.feature_ext_digits, mymodel.classifier_digits, source, target, args,method='emd',
                          metric='deep', reg_sink=1)
