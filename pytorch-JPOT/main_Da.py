
import torch


import model
import dataLoader
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from CommonSettings import args,datasetRootAndImageSize

# plt.ioff()
# np.random.seed(1976)
# np.set_printoptions(precision=3)

# CPU
from utils import changeSourceData

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

    source_loader = torch.utils.data.DataLoader(dataset=source, batch_size=args.batchSize, shuffle=True)
    target_loader = torch.utils.data.DataLoader(dataset=target, batch_size=args.batchSize, shuffle=True)

    newSourceDataLoader=changeSourceData(args.datasetIndex,datasetRootAndImageSize[args.datasetIndex][0],args.batchSize,kwargs)

    mymodel = model.JPOTModel(args.n_labels, DEVICE,n_dim=args.n_dim,backbone_name=args.backbone_name).to(DEVICE)

    model.train_process(mymodel, newSourceDataLoader, target_loader, args,DEVICE,args.datasetIndex, method='sinkhorn', metric='deep', reg_sink=1)
