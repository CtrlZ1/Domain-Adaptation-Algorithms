import torch
import numpy as np
def set_requires_grad(model, requires_grad=True):
    for critic in model:
        for param in critic.parameters():
            param.requires_grad = requires_grad


def onehot_label(sourceLabel,n_labels):
    labels=[]
    for i in sourceLabel:
        l=np.zeros(n_labels,)
        l[i]=1
        labels.append(l)

    return torch.tensor(labels).float()