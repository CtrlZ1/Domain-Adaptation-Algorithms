import torch
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os

from classif import NNClassifier
from jdot import jdot_nn_l2

np.random.seed(1999)


# CPU
DEVICE=torch.device('cpu')
kwargs={}

# if use cuda
if torch.cuda.is_available():
    DEVICE=torch.device('cuda:0')
    torch.cuda.manual_seed(1999)
    kwargs={'num_workers': 0, 'pin_memory': True}

print(DEVICE,torch.cuda.is_available())




if __name__ == '__main__':

    # generate data center
    source_traindata, source_trainlabel = make_blobs(1200, centers=[[0, -1], [0, 0], [0, 1]], cluster_std=0.2)
    target_traindata, target_trainlabel = make_blobs(1200, centers=[[1, 0], [1, 1], [1, 2]], cluster_std=0.2)
    plt.figure()
    plt.scatter(source_traindata[:, 0], source_traindata[:, 1], c=source_trainlabel, marker='o', alpha=0.4)
    plt.scatter(target_traindata[:, 0], target_traindata[:, 1], c=target_trainlabel, marker='x', alpha=0.4)
    plt.legend(['source train data', 'target train data'])
    plt.title("2D blobs visualization (shape=domain, color=class)")
    plt.show()

    # convert to one hot encoded vector
    def dataprocess_onehot(datalabel, n_labels):
        label = np.zeros((len(datalabel), n_labels))
        # convert label to one-hot key
        for index, i in enumerate(datalabel):
            label[index][i] = 1
        return label


    source_trainlabel_onehot,target_trainlabel_onehot=dataprocess_onehot(source_trainlabel,3),dataprocess_onehot(target_trainlabel,3)

    itermax = 10
    alpha = 1
    fit_params = {'epoch': 200}

    model, result = jdot_nn_l2(NNClassifier, source_traindata, source_trainlabel_onehot, target_traindata, ytest=target_trainlabel_onehot, fit_params=fit_params, numIterBCD=itermax,
                                  alpha=alpha)
    # chinese:
    # 下面的model acc其实是错误的，不应该用这种方式来预测目标标签，因为分类模型只是用来在训练过程中衡量每个源域样本分别与每个目标域样本的差异的
    # 而不是用来做预测的，目标预测应该使用label propagation方法，即通过转移矩阵G来预测标签。WJDOT也不能，而DeepJDOT是可以通过模型预测的，具体原因
    # 会在DeepJDOT代码中说明。
    # English:
    # The following model ACC is actually wrong and should not be used to predict the target label in this way,
    # because the classification model is only used to measure the difference between each source domain sample
    # and each target domain sample in the training process, not for prediction. The target prediction should
    # use the label propagation method, that is, predict the label through the transfer matrix G. Wjdot cannot,
    # and deepjdot can be predicted through the model. The specific reasons will be explained in the deepjdot code.
    ypred = model.predict(target_traindata).detach().cpu().numpy()
    ypred=np.argmax(ypred,axis=1)
    acc=np.sum(ypred==target_trainlabel)/len(ypred)

    print("model acc:",acc)

    G=result['G']
    label_propagation_correct=0
    propagate_mat = model.Label_propagation(target_traindata,source_trainlabel.reshape(len(source_trainlabel),), G)
    propagate_label = np.argmax(propagate_mat, axis=1)
    correct = (propagate_label == target_trainlabel).sum()
    label_propagation_correct += correct
    print('label propagation:',float(label_propagation_correct) / len(target_trainlabel))

    # Chinese:
    # 对于大规模数据，比如USPS->MNIST的训练我就不做了，因为JDOT设计的就是为小规模数据，大规模数据转移矩阵太大了，只能分batch，而这
    # 是DeepJDOT的内容了，放到DeepJDOT中去完成。
    # English:
    # For large-scale data, such as USPS-> MNIST training I do not do, because JDOT is designed
    # for small-scale data, large-scale data transfer matrix is too large, can only be divided into batch,
    # and this is the contents of the DeepJDOT, put deep JDOT to complete.





