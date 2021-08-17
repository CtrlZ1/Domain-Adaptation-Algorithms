
import torch
import model
import dataLoader

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

    if args.datasetIndex==0:
        sourceTrainLoader, targetTrainLoader = dataLoader.loadSourceAndTargetData(datasetRootAndImageSize[args.datasetIndex], args.batchSize,
                                                                       args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2],
                                                                       kwargs)
        targetTestLoader = dataLoader.loadTestData(datasetRootAndImageSize[args.datasetIndex][1], args.batchSize, args.datasetIndex,
                                                   datasetRootAndImageSize[args.datasetIndex][2], kwargs)
        model = model.DDCNModel(args.n_labels,DEVICE).to(DEVICE)
        model.train_process(sourceTrainLoader,targetTrainLoader,args)
        model.tes1t(targetTrainLoader,args.n_dim,args)
