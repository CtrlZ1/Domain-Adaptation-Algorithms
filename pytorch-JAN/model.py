
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


def test_process(model,targetData,backbone_name):
    model.eval()
    if backbone_name=='AlexNet':
        fc6_s, fc7_s, fc8_s = model.backbone(targetData)
        targetOutput = fc8_s
        targetOutput = model.bottleneck(targetOutput)
        targetOutput = model.last_classifier(targetOutput)
    elif backbone_name=='ResNet50':
        fc = model.backbone(targetData)
        targetOutput = fc
        targetOutput = model.bottleneck(targetOutput)
        targetOutput = model.last_classifier(targetOutput)
    return targetOutput


def train_process(model,sourceDataLoader, targetDataLoader,taragetTestDataLoader, device,args):
    # define loss function
    if args.adversarial:
        if args.backbone_name=="AlexNet":

            thetas = [Theta(dim).to(device) for dim in (4096,4096,1000)]
        elif args.backbone_name=='ResNet50':
            thetas = [Theta(dim).to(device) for dim in (args.bottleneck_dim, args.n_labels)]
    else:
        thetas = None
    # define loss function
    if args.backbone_name == 'ResNet50':
        jmmd = JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=(
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                (GaussianKernel(sigma=0.92, track_running_stats=False),)
            ),
            linear=not args.non_linear,thetas=thetas
        ).to(device)
    elif args.backbone_name=='AlexNet':
        jmmd = JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=(
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            ),
            linear=not args.non_linear, thetas=thetas
        ).to(device)
    parameters = [
        {'params': model.backbone.parameters()},
        {'params': model.bottleneck.parameters(), 'lr': args.lr},
        {'params': model.last_classifier.parameters(), 'lr': args.lr}
    ]
    if thetas is not None:
        parameters += [{"params": theta.parameters(), 'lr': 0.1} for theta in thetas]
	optimizer = optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.l2_Decay, nesterov=True)
    # learningRate = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epoch), 0.75)
    learningRate = LambdaLR(optimizer, lambda x: (1. + args.lr_gamma * (float(x)/(args.epoch))) ** (-args.lr_decay))
	clf_criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epoch + 1):
        model.train()
        
        
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
            sourceOutput, jmmd_loss = model.forward(sourceData, targetData, jmmd)
            source_pre = sourceOutput.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
            clf_loss = clf_criterion(sourceOutput, sourceLabel)
            loss = clf_loss + args.lamb * jmmd_loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            learningRate.step(epoch)
            if batch_idx % args.logInterval == 0:
                print(
                    '\nLoss: {:.4f},  clf_Loss: {:.4f},  mmd_loss: {:.4f}, lamb*mmd_loss:{:.4f}'.format(
                        loss.item(), clf_loss.item(), jmmd_loss.item(), args.lamb * jmmd_loss.item()))

        total_loss /= lenSourceDataLoader
        acc_train = float(correct) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Average classification loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(
            total_loss, correct, (lenSourceDataLoader * args.batchSize), acc_train))
        if epoch % 10 == 0:
            tes1t(model,taragetTestDataLoader, args.n_dim, device,args)
            model_feature_tSNE(model, sourceDataLoader, taragetTestDataLoader, 'epoch'+str(epoch), device,args.backbone_name)


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
            pre_label = test_process(model,data,args.backbone_name)

            pred = pre_label.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()

        print('\nTest Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(DataLoader.dataset),
            100. * correct / len(DataLoader.dataset)))
    return correct



class JANModel(nn.Module):
    def __init__(self,device,args,baseNet='AlexNet'):
        super(JANModel,self).__init__()
        self.backbone=network_dict[baseNet]()
        self.device=device
        self.args=args
        if self.args.backbone_name=='AlexNet':
            self.bottleneck = nn.Sequential(
                nn.Linear(1000, args.bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        elif self.args.backbone_name=='ResNet50':
            self.bottleneck = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(2048, args.bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        self.last_classifier = nn.Sequential(
            nn.Linear(args.bottleneck_dim, args.n_labels)
        )

    def forward(self,sourceData, targetData,jmmd):
        if self.args.backbone_name=='AlexNet':
            fc6_s,fc7_s,fc8_s = self.backbone(sourceData)
            sourceOutput=fc8_s
            fc6_t, fc7_t, fc8_t = self.backbone(targetData)
            sourceOutput = self.bottleneck(sourceOutput)

            sourceOutput = self.last_classifier(sourceOutput)
            jmmd_loss = jmmd(
                (fc6_s, fc7_s, fc8_s),
                (fc6_t, fc7_t, fc8_t)
            )
        elif self.args.backbone_name=='ResNet50':

            sourceOutput=self.backbone(sourceData)
            targetOutput=self.backbone(targetData)
            sourceOutput = self.bottleneck(sourceOutput)
            targetOutput = self.bottleneck(targetOutput)
            f_s = sourceOutput
            f_t = targetOutput
            sourceOutput = self.last_classifier(sourceOutput)
            targetOutput = self.last_classifier(targetOutput)

            jmmd_loss = jmmd(
                (f_s, F.softmax(sourceOutput, dim=1)),
                (f_t, F.softmax(targetOutput, dim=1))
            )

        return sourceOutput, jmmd_loss

