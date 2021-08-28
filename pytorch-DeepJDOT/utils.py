import numpy as np

import torch

# Nomalize alphas
def normalize_alphas_inplace(alphas):
    for alpha in alphas:
        alpha = torch.clamp_(alpha, min=0)
    sum=np.sum(alphas)
    for alpha in alphas:
        alpha = alpha.div_(sum)
    return alphas

# Calculate Wasserste Distance
def my_loss_custom(beta,G,C0,target_predict,y):
  # loss = torch.sum(torch.square(torch.cdist(y,target_predict))*G)
  C = beta*C0 + euclidean_dist(y,target_predict,square=False)

  lose = C*G
  return torch.sum(lose)

def euclidean_dist(x, y, square=False):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m = x.size(0)
    n = y.size(0)

    # 方式1
    a1 = torch.sum((x ** 2), 1, keepdim=True).expand(m, n)
    b2 = (y ** 2).sum(1).expand(m, n)
    if square:
        dist = (a1 + b2 - 2 * (x @ y.T)).float()
    else:
        dist = (a1 + b2 - 2 * (x @ y.T)).float().sqrt()
    return dist


def dataprocess_onehot(datalabel,n_labels):

    label = torch.zeros((len(datalabel),n_labels))
    # convert label to one-hot key
    for index, i in enumerate(datalabel):
        label[index][i]=1
    return label


def mini_batch_class_balanced(label, sample_size=20, shuffle=False):
    ''' sample the mini-batch with class balanced
    '''
    if shuffle:
        rindex = np.random.permutation(len(label))
        label = np.array(label)[rindex]

    n_class = len(np.unique(label))
    index = []
    for i in range(n_class):
        s_index = np.nonzero(label == i)
        s_ind = np.random.permutation(s_index[0])
        index = np.append(index, s_ind[0:sample_size])
        #          print(index)
    index = np.array(index, dtype=int)
    return index