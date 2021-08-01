
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
DEVICE=torch.device('cpu')
kwargs={}

# if use cuda
if not args.noCuda and torch.cuda.is_available():
    DEVICE=torch.device('cuda:'+str(args.gpu))
    torch.cuda.manual_seed(args.seed)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':

    if args.datasetIndex==7:
        # prepare data
        DataLoder_src1, DataLoder_src2,DataLoder_tar = dataLoader.loadSourceAndTargetData(datasetRootAndImageSize[args.datasetIndex], args.batchSize,
                                                                       args.datasetIndex, datasetRootAndImageSize[args.datasetIndex][2],
                                                                       kwargs)
        all_Xt = []  # target data
        all_Yt = []  # target label
        targetdata=DataLoder_tar.dataset.data.unsqueeze(1)

        targetdata = targetdata.expand(len(targetdata), 3, 28, 28)

        targetlabel=DataLoder_tar.dataset.targets

        all_Xt.append(targetdata.reshape(len(targetdata),-1).numpy())
        all_Yt.append(targetlabel.numpy())

        proportion = []
        for i in range(args.n_labels):
            proportion.append(round(float((targetlabel == i).sum()) / len(targetlabel),4))
        print("h_target", proportion)




        batchid=0
        for batch_idx, ((sourceData1, sourceLabel1),(sourceData2, sourceLabel2)) in tqdm.tqdm(enumerate(zip(DataLoder_src1,DataLoder_src2)),
                                                                desc='Train batch = {}'.format(batchid), ncols=80,leave=False):
            batchid+=1
            sourceData1 = sourceData1.expand(len(sourceData1), 3, 28, 28)
            sourceData2 = sourceData2.expand(len(sourceData2), 3, 28, 28)

            all_Xs = []  # source data
            all_Ys = []  # source label
            flattened_data1=sourceData1.reshape(sourceData1.size(0),-1)
            flattened_data2=sourceData2.reshape(sourceData2.size(0),-1)

            all_Xs.append(flattened_data1.numpy())
            all_Xs.append(flattened_data2.numpy())


            all_Ys.append(sourceLabel1.numpy())
            all_Ys.append(sourceLabel2.numpy())

            jcpotModel = model.JCPOTTransport()
            sf = jcpotModel.fit(all_Xs, all_Ys, all_Xt[0], all_Yt[0])

            print("pre_h", sf.proportions_[0])

            propagateLabels = jcpotModel.transform_labels(all_Ys)
            propagateLabels = np.argmax(propagateLabels, axis=1)
            correct = (propagateLabels == all_Yt[0]).sum()
            proportion = []
            for i in range(args.n_labels):
                proportion.append(round((propagateLabels == i).sum() / len(propagateLabels),4))
            print("Label propagation", proportion)
            print("acc:", round(correct / len(propagateLabels),4))

            break


