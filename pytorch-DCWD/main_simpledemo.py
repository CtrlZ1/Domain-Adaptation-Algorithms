import torch
import model

import matplotlib.pyplot as plt
import numpy as np
from CommonSettings import args
from dataLoader import generateBiModGaussian

np.random.seed(1999)
torch.manual_seed(1999)




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

        all_Ys.append(src['y'])

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
    all_Yt.append(target_data['y'])


    # show the transform process
    if args.length_of_feature == 2:
        plt.scatter(target_data['X'][:, 0], target_data['X'][:, 1], marker="+", lw=0.6, c=plt.cm.Set3(args.n_sources))
        plt.title("original distribution of source data and target data")
        plt.show()

        model = model.WJDOTModel()
        # model.modeltrain(np.array(all_Xs),np.array(all_Ys),np.array(all_Xt[0]), np.array(all_Yt[0]),args.rat_labeled,args.Omega)
        model.modeltrain(all_Xs,all_Ys,all_Xt[0], all_Yt[0],args.rat_labeled,args.Omega)



