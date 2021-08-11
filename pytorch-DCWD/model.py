import numpy as np


import ot

from ot import unif

import torch
import torch.nn as nn

from sklearn import svm
import scipy.spatial.distance as distance



class WJDOTModel(nn.Module):

    def __init__(self):
        super(WJDOTModel, self).__init__()


    def modeltrain(self,sources_data,sources_label,target_data,target_label,rat_labeled,Omega,method='sinkhorn',reg=1):

        xs=sources_data
        ys=sources_label
        xt=target_data
        yt=target_label




        xt_index = int(rat_labeled * len(xt))
        classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')  # ovr:一对多策略
        classifier.fit(xt[:xt_index,:], yt[:xt_index])

        yt_u_pre=classifier.predict(xt[xt_index:,:])
        n_labels = len(np.unique(yt))
        xt_byClass = [[] for i in range(n_labels)]
        yt_pre = np.append(yt[:xt_index], yt_u_pre)
        for i in range(len(yt_pre)):
            xt_byClass[int(yt_pre[i])].append(xt[i])

        nk = []
        ek = []
        for index,y in enumerate(ys):

            n_labels=len(np.unique(y))
            xs_byClass=[[] for i in range(n_labels)]

            for i in range(len(y)):
                xs_byClass[int(y[i])].append(xs[index][i])

            nk_class=0
            G = []
            for i in range(n_labels):

                C1=distance.cdist(np.array(xs_byClass[i]),np.array(xt_byClass[i]),metric='sqeuclidean')
                C1 = C1 / np.max(C1)
                ws = unif(len(xs_byClass[i]))
                wt = unif(len(xt_byClass[i]))
                if method == 'sinkhorn':
                    Gs = ot.sinkhorn(ws, wt, C1, reg)
                else:
                    Gs = ot.emd(ws, wt, C1)
                G.append(Gs)
                loss = np.multiply(C1, Gs).mean()
                nk_class+=loss
            nk_class=float(nk_class)/n_labels
            nk.append(nk_class)
            ek.append(-1 * Omega * (nk_class ** 2))


        softm = nn.Softmax(dim=0)
        alpha = softm(torch.tensor(ek).float())
        print("alpha:",alpha)
        new_xs = []
        new_ys = []
        for index, s in enumerate(xs):
            xs[index] = np.multiply(alpha[index],xs[index])
            new_xs.append(xs[index])
            new_ys.append(ys[index])

        xt_index=int(rat_labeled*len(xt))
        new_xs.append(xt[:xt_index,:])
        new_ys.append(yt[:xt_index])

        new_xs=np.concatenate(new_xs)# numpy
        new_ys=np.concatenate(new_ys)# numpy
        # train svm
        classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')  # ovr:一对多策略
        classifier.fit(new_xs, new_ys)
        print("acc：", classifier.score(xt[xt_index:,:], yt[xt_index:]))



