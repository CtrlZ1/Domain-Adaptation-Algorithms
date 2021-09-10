
import torch
from model import train_process, tes1t, WDANModel
import dataLoader

from CommonSettings import args,datasetRootAndImageSize

# CPU
from utils import model_feature_tSNE

DEVICE=torch.device('cpu')
kwargs={}

# if use cuda
if not args.noCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':

    sourceTrainLoader, targetTrainLoader = dataLoader.loadSourceAndTargetData(datasetRootAndImageSize[args.datasetIndex], args.batchSize,
                                                                   args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2],
                                                                   kwargs)
    targetTestLoader = dataLoader.loadTestData(datasetRootAndImageSize[args.datasetIndex][1], args.batchSize, args.datasetIndex,
                                               datasetRootAndImageSize[args.datasetIndex][2], kwargs)
    model = WDANModel(DEVICE,args).to(DEVICE)
    train_process(model,sourceTrainLoader, targetTrainLoader,targetTestLoader ,DEVICE,datasetRootAndImageSize[args.datasetIndex][2],args)
    tes1t(model,targetTrainLoader,args.n_dim,DEVICE,args)
    model_feature_tSNE(model,sourceTrainLoader,targetTestLoader,'last',DEVICE)
