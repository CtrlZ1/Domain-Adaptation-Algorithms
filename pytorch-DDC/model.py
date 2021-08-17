
import torch.nn as nn
import torch
from torchvision import models
import tqdm
import math
import torch.optim as optim
import torch.nn.functional as F

from utils import mmd


class DDCNModel(nn.Module):
    def __init__(self,n_classes,device):
        super(DDCNModel,self).__init__()
        modelAlexNet=models.alexnet(pretrained=True)
        self.device=device
        self.features=modelAlexNet.features
        self.classifier=modelAlexNet.classifier

        self.bottleneck = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True)
        )
        self.last_classifier = nn.Sequential(
            nn.Linear(256, n_classes)
        )

    def forward(self,sourceData, targetData):

        sourcefeature = self.features(sourceData).view(sourceData.size(0), -1)
        sourceOutput = self.classifier(sourcefeature)
        sourceOutput = self.bottleneck(sourceOutput)

        mmd_loss = 0
        if self.training:
            targetfeature = self.features(targetData).view(targetData.size(0), -1)
            targetOutput = self.classifier(targetfeature)
            targetOutput = self.bottleneck(targetOutput)
            mmd_loss += mmd(sourceOutput, targetOutput)

        sourceOutput = self.last_classifier(sourceOutput)

        return sourceOutput, mmd_loss

    def train_process(self,sourceDataLoader,targetDataLoader,args):
        for epoch in range(1,args.epoch+1):
            learningRate = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epoch), 0.75)
            print(f'Learning Rate: {learningRate}')
            optimizer = optim.SGD([
                {'params': self.features.parameters()},
                {'params': self.classifier.parameters()},
                {'params': self.bottleneck.parameters(), 'lr': learningRate},
                {'params': self.last_classifier.parameters(), 'lr': learningRate}
            ], lr=learningRate / 10, momentum=args.momentum, weight_decay=args.l2_Decay)

            clf_criterion = nn.CrossEntropyLoss()
            lenSourceDataLoader=len(sourceDataLoader)

            correct = 0
            total_loss = 0
            for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                                  desc='Train epoch = {}'.format(epoch), ncols=80,
                                                                  leave=False):


                sourceData, sourceLabel = sourceData.to(self.device), sourceLabel.to(self.device)
                for targetData, targetLabel in targetDataLoader:
                    targetData, targetLabel = targetData.to(self.device), targetLabel.to(self.device)
                    break

                optimizer.zero_grad()
                sourceOutput, mmd_loss=self.forward(sourceData,targetData)
                source_pre=sourceOutput.data.max(1, keepdim=True)[1]
                correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
                clf_loss = clf_criterion(sourceOutput, sourceLabel)
                loss = clf_loss + args.lamb * mmd_loss
                total_loss += clf_loss.item()

                loss.backward()
                optimizer.step()

                if batch_idx % args.logInterval == 0:
                    print(
                        '\nLoss: {:.4f},  clf_Loss: {:.4f},  mmd_loss: {:.4f}, lamb*mmd_loss:{:.4f}'.format(
                            loss.item(), clf_loss.item(), mmd_loss.item(), args.lamb * mmd_loss.item()))

            total_loss /= lenSourceDataLoader
            acc_train = float(correct) * 100. / (len(sourceDataLoader) * args.batchSize)

            print('Average classification loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                total_loss, correct, len(sourceDataLoader.dataset), acc_train))


    def tes1t(self,DataLoader ,n_dim,args):
            testLoss = 0
            correct = 0
            with torch.no_grad():
                for data, targetLabel in DataLoader:
                    if n_dim==0:
                        data, targetLabel = data.to(args.device), targetLabel.to(args.device)
                    elif n_dim>0:
                        imgSize = torch.sqrt(
                            (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                        data = data.expand(len(data), n_dim, imgSize.item(), imgSize.item()).to(
                            self.device)
                        targetLabel = targetLabel.to(self.device)
                    pre_label,mmd_loss = self.forward(data,data)
                    testLoss += F.nll_loss(F.log_softmax(pre_label, dim = 1), targetLabel, size_average=False).item() # sum up batch loss
                    testLoss+=args.lamb*mmd_loss
                    pred = pre_label.data.max(1)[1] # get the index of the max log-probability
                    correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()

                testLoss /= len(DataLoader.dataset)
                print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    testLoss, correct, len(DataLoader.dataset),
                    100. * correct / len(DataLoader.dataset)))
            return correct
