import numpy as np

import tqdm
import ot
import torch.optim as optim
from ot import unif
from torch.optim.lr_scheduler import LambdaLR

from utils import normalize_alphas_inplace, my_loss_custom, euclidean_dist, mini_batch_class_balanced
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from utils import dataprocess_onehot


# One layer network (classifer needs to be optimized)
class DeepJDOT(nn.Module):

    def __init__(self, n_class, device):
        super(DeepJDOT, self).__init__()
        self.device = device
        self.n_class = n_class
        # self.fc1 = nn.Linear(2, 2)
        # self.fc1 = nn.Linear(12288, 10)# 12288，10
        self.feature_ext_demo = nn.Sequential(
            nn.Linear(2, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
        ).to(device)

        self.classifier_demo = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.n_class),
            nn.Softmax()
        ).to(device)

        self.feature_ext_digits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32768, 128),
            nn.ReLU(True),
        ).to(self.device)
        self.classifier_digits = nn.Sequential(
            nn.Linear(128, n_class),
        ).to(self.device)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return h

    def sourcemodel_usedin_target(self, feature_model, classifier_model, sourceDataLoader, targetDataLoader, lrf, lrc,
                                  epoch, n_dim, logInterval):
        optimizer_F = optim.SGD(feature_model.parameters(), lr=lrf)
        optimizer_C = optim.SGD(classifier_model.parameters(), lr=lrc)
        evaluate = nn.CrossEntropyLoss()
        lenSourceDataLoader = len(sourceDataLoader)
        for e in range(epoch):
            trainingloss = 0
            print("epoch:", e)
            for batch_idx, (sourceData, sourceLabel) in enumerate(sourceDataLoader):
                optimizer_F.zero_grad()
                optimizer_C.zero_grad()
                if n_dim == 0:
                    sourceData, sourceLabel = sourceData.view(len(sourceData), -1).to(self.device), sourceLabel.to(
                        self.device)
                elif n_dim > 0:
                    imgSize = torch.sqrt((torch.prod(torch.tensor(sourceData.size())) / (
                                sourceData.size(1) * len(sourceData))).float()).int()
                    sourceData = sourceData.expand(len(sourceData), n_dim, imgSize.item(), imgSize.item()).to(
                        self.device)
                    sourceLabel = sourceLabel.to(self.device)
                loss = evaluate(classifier_model(feature_model(sourceData)), sourceLabel)
                trainingloss = loss.item()
                loss.backward()
                optimizer_F.step()
                optimizer_C.step()

            if e % logInterval == 0:
                print("training loss:", trainingloss)
        print("training acc...")
        self.tes1t(sourceDataLoader, feature_model, classifier_model, n_dim, self.device)
        print("testing acc...")
        self.tes1t(targetDataLoader, feature_model, classifier_model, n_dim, self.device)

    # use DeepJDOT model
    def train_process(self, feature_model, classifier_model, source, target, args,method='sinkhorn', metric='deep', reg_sink=1):

        source_loader = torch.utils.data.DataLoader(dataset=source, batch_size=len(source), shuffle=True)
        target_loader = torch.utils.data.DataLoader(dataset=target, batch_size=len(target), shuffle=True)

        optimizer_F = optim.SGD(feature_model.parameters(), lr=args.lrf,momentum=args.momentum, weight_decay=args.l2_Decay, nesterov=True)
        optimizer_C = optim.SGD(classifier_model.parameters(), lr=args.lrc,momentum=args.momentum, weight_decay=args.l2_Decay, nesterov=True)



        evaluate = nn.CrossEntropyLoss()
        label_propagation_correct = 0

        sourcedata = iter(source_loader)
        (allsourceData, allsourceLabel) = next(sourcedata)

        targetdata = iter(target_loader)
        (alltargetData, alltargetLabel) = next(targetdata)

        source_loader = torch.utils.data.DataLoader(dataset=source, batch_size=64, shuffle=True)
        target_loader = torch.utils.data.DataLoader(dataset=target, batch_size=64, shuffle=True)
        leniter = int(len(allsourceData) / args.batchSize)
        for e in range(args.epoch):
            # print("epoch:",e)
            label_propagation_correct = 0
            for batch_idx in tqdm.tqdm(range(leniter),total=leniter,desc='Train epoch = {}'.format(e), ncols=80,
                                                                  leave=False):
                # all
                # for batch_idx, (sourceData, sourceLabel) in enumerate(sourceDataLoader):
                s_index=mini_batch_class_balanced(allsourceLabel.numpy(),args.sample_size,False)
                sourceData=allsourceData[s_index]
                sourceLabel=allsourceLabel[s_index]
                t_index = np.random.choice(len(alltargetData), args.batchSize)
                targetData = alltargetData[t_index]
                targetLabel = alltargetLabel[t_index]

                optimizer_F.zero_grad()
                optimizer_C.zero_grad()
                if args.n_dim == 0:
                    sourceData, sourceLabel = sourceData.view(len(sourceData), -1).float().to(self.device), sourceLabel.to(
                        self.device)
                elif args.n_dim > 0:
                    imgSize = torch.sqrt((torch.prod(torch.tensor(sourceData.size())) / (
                                sourceData.size(1) * len(sourceData))).float()).int()

                    sourceData = sourceData.expand(len(sourceData), args.n_dim, imgSize.item(), imgSize.item()).float().to(
                        self.device)
                    sourceLabel = sourceLabel.to(self.device)

                    if args.n_dim == 0:
                        targetData, targetLabel = targetData.view(len(targetData), -1).float().to(self.device), targetLabel.to(
                            self.device)
                    elif args.n_dim > 0:
                        imgSize = torch.sqrt(
                            (torch.prod(torch.tensor(targetData.size())) / (
                                        targetData.size(1) * len(targetData))).float()).int()
                        targetData = targetData.expand(len(targetData), args.n_dim, imgSize.item(), imgSize.item()).float().to(
                            self.device)
                        targetLabel = targetLabel.to(self.device)


                fea_source = feature_model(sourceData)
                fea_target = feature_model(targetData)

                if metric == 'original':
                    C0 = euclidean_dist(sourceData, targetData, square=True)
                elif metric == 'deep':
                    C0 = euclidean_dist(fea_source, fea_target, square=True)

                pre_targetlabel = classifier_model(fea_target)
                pre_sourcelabel = classifier_model(fea_source)
                one_hot_sourcelabel = dataprocess_onehot(sourceLabel, self.n_class).to(self.device)
                C1 = euclidean_dist(one_hot_sourcelabel, pre_targetlabel, square=True)

                C = args.alpha * C0 + args.alpha2 * C1
                # print(torch.sum(C0))
                # JDOT optimal coupling (gamma)
                if method == 'sinkhorn':
                    gamma = ot.sinkhorn(ot.unif(fea_source.size(0)), ot.unif(fea_target.size(0)),
                                        C.detach().cpu().numpy(), reg=reg_sink)
                elif method == 'emd':
                    gamma = ot.emd(ot.unif(fea_source.size(0)), ot.unif(fea_target.size(0)), C.detach().cpu().numpy())

                if e % args.logInterval == 0:
                    propagate_mat = self.Label_propagation(targetData.detach().cpu().numpy(),
                                                           sourceLabel.view(len(sourceLabel), ).detach().cpu().numpy(),
                                                           gamma, self.n_class)
                    propagate_label = np.argmax(propagate_mat, axis=1)
                    correct = (propagate_label == targetLabel.detach().cpu().numpy()).sum()
                    label_propagation_correct += correct
                    # print(label_propagation_correct)

                l_c = args.train_par * torch.sum(C * torch.tensor(gamma).float().to(self.device))
                l_t = args.lam * evaluate(pre_sourcelabel, sourceLabel)
                loss = l_c + l_t
                trainingloss = loss.item()
                trainingl_c = l_c.item()
                trainingl_t = l_t.item()
                loss.backward()

                optimizer_F.step()
                optimizer_C.step()


                if batch_idx % 100 == 0:
                    print("training loss:{:.4f},l_c:{:.4f},l_t:{:.4f}".format(trainingloss, trainingl_c, trainingl_t))
            if e % args.logInterval == 0:
                print("training acc...")
                self.tes1t(source_loader, feature_model, classifier_model, args.n_dim, self.device)
                print("testing acc...")
                self.tes1t(target_loader, feature_model, classifier_model, args.n_dim, self.device)
                # label propagation
                print("label propagation acc...")
                # print(float(label_propagation_correct),len(targetDataLoader.dataset))
                print(float(label_propagation_correct) / (leniter * args.batchSize))


        print("training acc...")
        self.tes1t(source_loader, feature_model, classifier_model, args.n_dim, self.device)
        print("testing acc...")
        self.tes1t(target_loader, feature_model, classifier_model, args.n_dim, self.device)
        # label propagation
        print("label propagation acc...")
        # print(float(label_propagation_correct),len(targetDataLoader.dataset))
        print(float(label_propagation_correct) / (leniter * args.batchSize))

    # Label propagation
    def Label_propagation(self, Xt, Ys, g, n_labels):
        ys = Ys
        xt = Xt
        yt = np.zeros((n_labels, xt.shape[0]))  # [n_labels,n_target_sample]
        # let labels start from a number
        ysTemp = np.copy(ys)  # ys、ysTemp:[n_source_samples,]
        # classes = np.unique(ysTemp)
        n = n_labels
        ns = len(ysTemp)

        # perform label propagation
        transp = g / np.sum(g, 1)[:, None]  # coupling_[i]:[n_source_samples,n_target_samples]

        # set nans to 0
        transp[~ np.isfinite(transp)] = 0

        D1 = np.zeros((n, ns))  # [n_labels,n_source_samples]

        for c in range(n_labels):
            D1[int(c), ysTemp == c] = 1

        # compute propagated labels
        # / len(ys)=/ k, means uniform sources transfering
        yt = yt + np.dot(D1, transp) / len(
            ys)  # np.dot(D1, transp):[n_labels,n_target_samples] show the mass of every class for transfering to target samples

        return yt.T  # n_samples,n_labels

    def tes1t(self, DataLoader, feature_model, classifier_model, n_dim, DEVICE):
        feature_model.eval()
        classifier_model.eval()
        testLoss = 0
        correct = 0
        num = 0
        with torch.no_grad():
            for data, targetLabel in DataLoader:
                if n_dim == 0:
                    data, targetLabel = data.to(DEVICE), targetLabel.to(DEVICE)
                elif n_dim > 0:
                    imgSize = torch.sqrt(
                        (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                    data = data.expand(len(data), n_dim, imgSize.item(), imgSize.item()).to(
                        self.device)
                    targetLabel = targetLabel.to(self.device)
                pre_label = classifier_model(feature_model(data))
                testLoss += F.nll_loss(F.log_softmax(pre_label, dim=1), targetLabel,
                                       size_average=False).item()  # sum up batch loss
                pred = pre_label.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()
            testLoss /= len(DataLoader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                testLoss, correct, len(DataLoader.dataset),
                100. * correct / len(DataLoader.dataset)))
        return correct

