
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
import tqdm
import math
import torch.optim as optim
import torch.nn.functional as F

from backBone import network_dict
from utils import JointMultipleKernelMaximumMeanDiscrepancy, GaussianKernel, model_feature_tSNE, Theta


def test_process(model,targetData):
    model.eval()
    targetOutput = model.classifier1(model.feature_extract(targetData))
    return targetOutput

def sort_rows(matrix):
    return torch.sort(matrix, descending=True,dim=0)[0]


def discrepancy_slice_wasserstein(p1, p2,device):
    s = p1.size(1)
    if s > 1:
        # For data more than one-dimensional, perform multiple random projection to 1-D
        theta = torch.rand(10, 128)
        theta = (theta / torch.sum(theta,dim=0)).to(device)

        p1 = torch.matmul(p1,theta)
        p2 = torch.matmul(p2,theta)
    p1 = sort_rows(p1)
    p2 = sort_rows(p2)
    wdist = (p1-p2)**2
    return torch.mean(wdist)

def discrepancy_mcd(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


def train_process(model,sourceDataLoader, targetDataLoader,taragetTestDataLoader, device,mode,args):

    parameters = [
        {'params': model.classifier1.parameters()},
        {'params': model.classifier2.parameters()},
    ]


    for epoch in range(1, args.epoch + 1):
        model.train()
        optimizer_cls = optim.Adam(model.parameters(), args.lr)
        optimizer_cls_dist = optim.Adam(parameters, args.lr)
        optimizer_dist = optim.Adam(model.feature_extract.parameters(), args.lr)


        clf_criterion = nn.CrossEntropyLoss()
        lenSourceDataLoader = len(sourceDataLoader)

        correct1 = 0
        correct2 = 0
        total_loss = 0
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):

            sourceData, sourceLabel = sourceData.to(device), sourceLabel.to(device)
            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.to(device), targetLabel.to(device)
                break

            optimizer_cls.zero_grad()
            clf_loss=0
            sourceOutput = model.classifier1(model.feature_extract(sourceData))
            source_pre = sourceOutput.data.max(1, keepdim=True)[1]
            correct1 += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
            clf_loss += clf_criterion(sourceOutput, sourceLabel)

            sourceOutput = model.classifier2(model.feature_extract(sourceData))
            source_pre = sourceOutput.data.max(1, keepdim=True)[1]
            correct2 += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
            clf_loss += clf_criterion(sourceOutput, sourceLabel)
            total_loss += clf_loss.item()
            clf_loss.backward()
            optimizer_cls.step()

            optimizer_cls_dist.zero_grad()
            p1=model.classifier1(model.feature_extract(targetData))
            p2=model.classifier2(model.feature_extract(targetData))

            if mode == 'adapt_swd':
                loss_dist = discrepancy_slice_wasserstein(p1, p2,device)
            elif mode == 'MCD':
                loss_dist = discrepancy_mcd(p1, p2)

            loss_clf_dist=0
            sourceOutput = model.classifier1(model.feature_extract(sourceData))
            loss_clf_dist += clf_criterion(sourceOutput, sourceLabel)

            sourceOutput = model.classifier2(model.feature_extract(sourceData))
            loss_clf_dist += clf_criterion(sourceOutput, sourceLabel)
            loss_clf_dist-=loss_dist
            total_loss += loss_clf_dist.item()
            loss_clf_dist.backward()

            optimizer_cls_dist.step()



            if mode == 'adapt_swd':#adapt_swd
                optimizer_dist.zero_grad()
                p1 = model.classifier1(model.feature_extract(targetData))
                p2 = model.classifier2(model.feature_extract(targetData))
                loss_dist = discrepancy_slice_wasserstein(p1, p2,device)
                total_loss += loss_dist.item()
                loss_dist.backward()
                optimizer_dist.step()
                # for i in range(args.num_k):
                #     optimizer_dist.zero_grad()
                #     p1 = model.classifier1(model.feature_extract(targetData))
                #     p2 = model.classifier2(model.feature_extract(targetData))
                #     loss_dist = discrepancy_slice_wasserstein(p1, p2,device)
                #     loss_dist.backward()
                #     optimizer_dist.step()
                # total_loss += loss_dist.item()
            elif mode=='MCD':
                for i in range(args.num_k):
                    optimizer_dist.zero_grad()
                    p1 = model.classifier1(model.feature_extract(targetData))
                    p2 = model.classifier2(model.feature_extract(targetData))
                    loss_dist = discrepancy_mcd(p1, p2)
                    loss_dist.backward()
                    optimizer_dist.step()
                total_loss += loss_dist.item()

            if batch_idx % args.logInterval == 0:
                print(
                    '\nclf_Loss: {:.4f},  loss_clf_dist: {:.4f}, loss_dist:{:.4f}'.format(
                         clf_loss.item(), loss_clf_dist.item(), loss_dist.item()))

        total_loss /= lenSourceDataLoader
        acc_train = float(correct1) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Average classification loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, correct1, (lenSourceDataLoader * args.batchSize), acc_train))
        tes1t(model, taragetTestDataLoader, args.n_dim, device, args)
        # if epoch % 10 == 0:
            # tes1t(model,taragetTestDataLoader, args.n_dim, device,args)
            # model_feature_tSNE(model, sourceDataLoader, taragetTestDataLoader, 'epoch'+str(epoch), device,args.backbone_name)


def tes1t(model,DataLoader, n_dim, device, args):
    correct = 0
    with torch.no_grad():
        for data, targetLabel in DataLoader:
            if n_dim == 0:
                data, targetLabel = data.to(args.device), targetLabel.to(args.device)
            elif n_dim > 0:
                imgSize = torch.sqrt(
                    (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                data = data.expand(len(data), n_dim, imgSize.item(), imgSize.item()).to(
                    device)
                targetLabel = targetLabel.to(device)
            pre_label = test_process(model,data)

            pred = pre_label.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()

        print('\nTest Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(DataLoader.dataset),
            100. * correct / len(DataLoader.dataset)))
    return correct



class SWDModel(nn.Module):
    def __init__(self,device,args):
        super(SWDModel,self).__init__()
        self.device=device
        self.args=args
        self.feature_extract=nn.Sequential(
            nn.Conv2d(args.n_dim, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(768, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, args.n_labels),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(768, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, args.n_labels),
        )

