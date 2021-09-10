
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
import tqdm
import math
import torch.optim as optim
import torch.nn.functional as F

from backBone import network_dict
from utils import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel, model_feature_tSNE, get_Omega, one_hot_label, \
    linear_kernel


def test_process(model,targetData):
    model.eval()
    fc6_s, fc7_s, fc8_s = model.backbone(targetData)
    targetOutput = fc8_s
    targetOutput = model.bottleneck(targetOutput)
    targetOutput = model.last_classifier(targetOutput)
    return targetOutput


def train_process(model,sourceDataLoader, targetDataLoader,taragetTestDataLoader, device,imageSize,args):

    # define loss function
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        # kernels=[linear_kernel()],
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

            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(
                device), sourceLabel.to(device)

            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.expand(len(targetData), args.n_dim, imageSize, imageSize).to(
                    device), targetLabel.to(device)
                break
            optimizer.zero_grad()
            sourceOutput, targetOutput, pseudo_label, mmd_loss = model.forward(sourceData, targetData, mkmmd_loss,
                                                                               sourceLabel)
            source_pre = sourceOutput.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
            clf_loss = clf_criterion(sourceOutput, sourceLabel)
            clf_t_loss = clf_criterion(targetOutput, pseudo_label)
            loss = args.lam_cls * clf_loss + args.lam * mmd_loss + args.gamma * clf_t_loss
            total_loss += clf_loss.item()

            loss.backward(retain_graph=True)
            optimizer.step()
            learningRate.step(epoch)
            if batch_idx % args.logInterval == 0:
                print(
                    '\nLoss: {:.4f},  clf_Loss: {:.4f},lam_clf*clf_Loss:{:.4f}, clf_t_loss:{:.4f},gamma*clf_t_loss:{:.4f}, mmd_loss: {:.4f}, lamb*mmd_loss:{:.4f}'.format(
                        loss.item(), clf_loss.item(), args.lam_cls * clf_loss.item(), clf_t_loss.item(),
                                                      args.gamma * clf_t_loss.item(), mmd_loss.item(),
                                                      args.lam * mmd_loss.item()))

        total_loss /= lenSourceDataLoader
        acc_train = float(correct) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Average classification loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, correct, (lenSourceDataLoader * args.batchSize), acc_train))
        if epoch % 1 == 0:
            tes1t(model,taragetTestDataLoader, args.n_dim, device,args)
            # model_feature_tSNE(model, sourceDataLoader, taragetTestDataLoader, 'epoch'+str(epoch), device)


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



class WDANModel(nn.Module):
    def __init__(self,device,args,baseNet='AlexNet'):
        super(WDANModel,self).__init__()
        self.backbone_name=baseNet
        self.backbone=network_dict[baseNet]()
        self.device=device

        self.bottleneck = nn.Sequential(
            nn.Linear(1000, args.bottleneck_dim),
            nn.ReLU(),
        )
        self.last_classifier = nn.Sequential(
            nn.Linear(args.bottleneck_dim, args.n_labels)
        )
        self.args=args

    def forward(self,sourceData, targetData,mkmmd_loss,sourceLabel):
        if self.backbone_name == 'AlexNet':
            fc6_s, fc7_s, fc8_s = self.backbone(sourceData)
            sourceOutput_neck = self.bottleneck(fc8_s)

            mmd_loss = 0
            fc6_t, fc7_t, fc8_t = self.backbone(targetData)
            targetOutput_neck = self.bottleneck(fc8_t)

            sourceOutput = self.last_classifier(sourceOutput_neck)
            targetOutput = self.last_classifier(targetOutput_neck)

            pseudo_label = (targetOutput.detach().data.max(1)[1]).view_as(sourceLabel)
            Omega_s = get_Omega(sourceLabel, self.args.n_labels)
            Omega_t = get_Omega(pseudo_label, self.args.n_labels)
            Omega = Omega_t / Omega_s
            # Omega/=torch.sum(Omega)
            Omega = Omega.view(len(Omega), 1).to(self.device)
            Omega = torch.autograd.Variable(Omega, requires_grad=True)
            sourceLabel_onehot = one_hot_label(sourceLabel, self.args.n_labels).to(self.device)
            source_Omega = torch.matmul(sourceLabel_onehot, Omega).float()
            # print(Omega_t)
            # print(Omega_s)
            # mmd_loss += mkmmd_loss(sourceOutput, targetOutput)
            mmd_loss += mkmmd_loss(fc8_s, fc8_t, source_Omega)
            # mmd_loss += mkmmd_loss(sourceOutput_neck, targetOutput_neck,source_Omega)
            # mmd_loss += mkmmd_loss(sourceOutput, targetOutput,source_Omega)
            mmd_loss += mkmmd_loss(fc6_s, fc6_t, source_Omega)
            mmd_loss += mkmmd_loss(fc7_s, fc7_t, source_Omega)
        # elif self.backbone_name=='LeNet':

        return sourceOutput, targetOutput,  pseudo_label, mmd_loss

