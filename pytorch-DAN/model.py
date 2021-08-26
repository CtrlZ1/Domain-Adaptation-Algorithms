
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
import tqdm
import math
import torch.optim as optim
import torch.nn.functional as F

from backBone import network_dict
from utils import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel, model_feature_tSNE


def test_process(model,targetData):
    model.eval()
    fc6_s, fc7_s, fc8_s = model.backbone(targetData)
    targetOutput = fc8_s
    targetOutput = model.bottleneck(targetOutput)
    targetOutput = model.last_classifier(targetOutput)
    return targetOutput


def train_process(model,sourceDataLoader, targetDataLoader,taragetTestDataLoader, device,args):

    # define loss function
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=not args.non_linear
    )
    for epoch in range(1, args.epoch + 1):
        model.train()
        optimizer = optim.SGD([
            {'params': model.backbone.parameters()},
            {'params': model.bottleneck.parameters(), 'lr': args.lr},
            {'params': model.last_classifier.parameters(), 'lr': args.lr}
        ], lr=args.lr / 10, momentum=args.momentum, weight_decay=args.l2_Decay)

        # learningRate = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epoch), 0.75)
        learningRate = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
        clf_criterion = nn.CrossEntropyLoss()
        lenSourceDataLoader = len(sourceDataLoader)

        correct = 0
        total_loss = 0
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):

            sourceData, sourceLabel = sourceData.to(device), sourceLabel.to(device)
            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.to(device), targetLabel.to(device)
                break

            optimizer.zero_grad()
            sourceOutput, mmd_loss = model.forward(sourceData, targetData, mkmmd_loss)
            source_pre = sourceOutput.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
            clf_loss = clf_criterion(sourceOutput, sourceLabel)
            loss = clf_loss + args.lamb * mmd_loss
            total_loss += clf_loss.item()

            loss.backward()
            optimizer.step()
            learningRate.step(epoch)
            if batch_idx % args.logInterval == 0:
                print(
                    '\nLoss: {:.4f},  clf_Loss: {:.4f},  mmd_loss: {:.4f}, lamb*mmd_loss:{:.4f}'.format(
                        loss.item(), clf_loss.item(), mmd_loss.item(), args.lamb * mmd_loss.item()))

        total_loss /= lenSourceDataLoader
        acc_train = float(correct) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Average classification loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, correct, (lenSourceDataLoader * args.batchSize), acc_train))
        if epoch % 10 == 0:
            tes1t(model,taragetTestDataLoader, args.n_dim, device,args)
            model_feature_tSNE(model, sourceDataLoader, taragetTestDataLoader, 'epoch'+str(epoch), device)


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



class DANModel(nn.Module):
    def __init__(self,device,args,baseNet='AlexNet'):
        super(DANModel,self).__init__()
        self.backbone=network_dict[baseNet]()
        self.device=device

        self.bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(1000, args.bottleneck_dim),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.last_classifier = nn.Sequential(
            nn.Linear(args.bottleneck_dim, args.n_labels)
        )

    def forward(self,sourceData, targetData,mkmmd_loss):
        fc6_s, fc7_s, fc8_s = self.backbone(sourceData)
        sourceOutput = fc8_s
        sourceOutput = self.bottleneck(sourceOutput)

        mmd_loss = 0
        fc6_t, fc7_t, fc8_t = self.backbone(targetData)
        targetOutput = fc8_t
        targetOutput = self.bottleneck(targetOutput)
        # mmd_loss += mkmmd_loss(sourceOutput, targetOutput)
        mmd_loss += mkmmd_loss(fc8_s, fc8_t)
        mmd_loss += mkmmd_loss(sourceOutput, targetOutput)

        sourceOutput = self.last_classifier(sourceOutput)
        targetOutput = self.last_classifier(targetOutput)
        mmd_loss += mkmmd_loss(sourceOutput, targetOutput)

        return sourceOutput, mmd_loss

