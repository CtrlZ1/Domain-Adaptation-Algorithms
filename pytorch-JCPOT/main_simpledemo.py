import torch
import model

import matplotlib.pyplot as plt
import numpy as np
import os
from CommonSettings import args,datasetRootAndImageSize
from dataLoader import generateBiModGaussian

plt.ioff()
# np.random.seed(1976)
np.set_printoptions(precision=3)

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

    # generate data center
    # source
    center_s=[]
    for i in range(args.n_sources):
        center=[]
        for j in range(args.n_labels):
            center_ij=np.zeros(args.length_of_feature)
            # only define the first two elements
            center_ij[0]=i
            center_ij[1]=j
            center.append(center_ij)
        center_s.append(center)

    A=np.eye(args.length_of_feature)
    b = np.zeros(args.length_of_feature)
    all_Xs = [] # source data
    all_Ys = [] # source label
    for i in range(args.n_sources):
        h_source = np.random.rand(args.n_labels)
        h_source = h_source / np.sum(h_source)
        # bs = [b[np.random.randint(0,2)], b[np.random.randint(0,2)]]
        src = generateBiModGaussian(center_s[i], args.sigma, args.n_source_simples, A, b, h_source,args)
        all_Xs.append(src['X'])

        if args.length_of_feature==2:
            plt.scatter(src['X'][:, 0], src['X'][:, 1], marker="+", lw=0.6,c=plt.cm.Set3(i))

        all_Ys.append(src['y'].astype(int))

    # target
    center_t=[]
    for j in range(args.n_labels):
        center_ij = np.zeros(args.length_of_feature)
        # only define the first two elements
        center_ij[0] = float(args.n_sources)/2
        center_ij[1] = j+0.5
        center_t.append(center_ij)

    h_target = np.random.rand(args.n_labels)
    h_target = h_target / np.sum(h_target)
    print("h_target",h_target)
    all_Xt = []  # target data
    all_Yt = []  # target label
    target_data = generateBiModGaussian(center_t, args.sigma, args.n_target_simples, A, b, h_target,args)
    all_Xt.append(target_data['X'])
    all_Yt.append(target_data['y'].astype(int))



    jcpotModel = model.JCPOTTransport()
    sf = jcpotModel.fit(all_Xs, all_Ys, all_Xt[0], all_Yt[0])
    print("pre_h", sf.proportions_[0])
    # show the transform process
    if args.length_of_feature == 2:
        plt.scatter(target_data['X'][:, 0], target_data['X'][:, 1], marker="+", lw=0.6, c=plt.cm.Set3(args.n_sources))
        plt.title("original distribution of source data and target data")
        plt.show()

        transformed_data=jcpotModel.transform(all_Xs,all_Ys,all_Xt[0],all_Yt[0])

        for index,trdata in enumerate(transformed_data):
            plt.scatter(trdata[:, 0], trdata[:, 1], marker="+", lw=0.6, c=plt.cm.Set3(index))
        plt.scatter(target_data['X'][:, 0], target_data['X'][:, 1], marker="+", lw=0.6, c=plt.cm.Set3(args.n_sources))
        plt.title("transformed distribution of source data and target data")
        plt.show()



        Xs=np.concatenate(all_Xs)
        trans_Xs=np.concatenate(transformed_data)

        for index,data in enumerate(all_Xs):
            plt.scatter(data[:, 0], data[:, 1], marker="+", lw=0.6, c=plt.cm.Set3(index))
        for index,trdata in enumerate(transformed_data):
            plt.scatter(trdata[:, 0], trdata[:, 1], marker="+", lw=0.6, c=plt.cm.Set3(index))
        plt.scatter(target_data['X'][:, 0], target_data['X'][:, 1], marker="+", lw=0.6, c=plt.cm.Set3(args.n_sources))
        plt.quiver(Xs[:,0], Xs[:,1], trans_Xs[:, 0]-Xs[:,0], trans_Xs[:, 1]-Xs[:,1], angles="xy", scale_units='xy', scale=1,
                  width=0.0002)
        plt.title("transform map")
        plt.show()
        # print(pre_h.proportions_)


    # predict the proportion of target labels
    # Label propagation
    propagateLabels=jcpotModel.transform_labels(all_Ys)
    propagateLabels=np.argmax(propagateLabels,axis=1)
    correct = (propagateLabels==all_Yt[0]).sum()
    proportion=[]
    for i in range(args.n_labels):
        proportion.append((propagateLabels==i).sum()/len(propagateLabels))
    print("Label propagation",proportion)
    print("acc:",(correct/len(propagateLabels)))

