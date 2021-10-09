
from backBone import network_dict

import torch
import tqdm
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from utils import ConditionalDomainAdversarialLoss


def train_process(model, sourceDataLoader, targetDataLoader,sourceTestDataLoader,taragetTestDataLoader,DEVICE,imageSize,args):

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.l2_Decay, nesterov=True)

    learningRate = LambdaLR(optimizer, lambda x: (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        model.discriminator, entropy_conditioning=args.entropy,
        num_classes=args.n_labels, features_dim=model.classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(DEVICE)

    lenSourceDataLoader = len(sourceDataLoader)

    base_epoch = 0
    if args.ifload:
        path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2 = os.path.join(path, i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        base_epoch = checkpoint['epoch']
    t_correct=0
    for epoch in range(1 + base_epoch, base_epoch + args.epoch + 1):
        model.train()
        domain_adv.train()

        correct = 0
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):

            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(
                DEVICE), sourceLabel.to(DEVICE)

            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.expand(len(targetData), args.n_dim, imageSize, imageSize).to(
                    DEVICE), targetLabel.to(DEVICE)
                break

            # compute output
            x = torch.cat((sourceData, targetData), dim=0)

            y, f = model(x)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            source_pre = y_s.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()

            cls_loss = F.cross_entropy(y_s, sourceLabel)
            transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
            loss = cls_loss + transfer_loss * args.trade_off


            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learningRate.step(epoch)


            if batch_idx % args.logInterval == 0:
                print(
                    '\ncls_loss: {:.4f},  transfer_loss: {:.4f}'.format(
                        cls_loss.item(), transfer_loss.item()))


        acc_train = float(correct) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Train Accuracy: {}/{} ({:.2f}%)'.format(
            correct, (lenSourceDataLoader * args.batchSize), acc_train))

        test_correct=test_process(model, sourceTestDataLoader,taragetTestDataLoader, DEVICE, args)
        if test_correct > t_correct:
            t_correct = test_correct
        print("max correct:" , t_correct)
        # if epoch % args.logInterval == 0:
        #     model_feature_tSNE(model, sourceTestDataLoader, taragetTestDataLoader, 'epoch' + str(epoch), DEVICE,
        #                        args.model_name)

    if args.ifsave:
        path=args.savePath+args.model_name
        if not os.path.exists(path):
            os.makedirs(path)

        if args.if_saveall:
            state = {
                'epoch': args.epoch,
                'net': model,
                'optimizer': optimizer,

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),

            }
        path+='/'+args.model_name+'_epoch'+str(args.epoch)+'.pth'
        torch.save(state, path)


def test_process(model,sourceTestDataLoader,taragetTestDataLoader, device, args):
    model.eval()


    # source Test
    correct = 0
    testLoss = 0
    with torch.no_grad():
        for data, suorceLabel in sourceTestDataLoader:
            if args.n_dim == 0:
                data, suorceLabel = data.to(args.device), suorceLabel.to(args.device)
            elif args.n_dim > 0:
                imgSize = torch.sqrt(
                    (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                data = data.expand(len(data), args.n_dim, imgSize.item(), imgSize.item()).to(
                    device)
                suorceLabel = suorceLabel.to(device)
            Output = model(data)[0]
            testLoss += F.nll_loss(F.log_softmax(Output, dim=1), suorceLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = Output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(suorceLabel.data.view_as(pred)).cpu().sum()
        testLoss /= len(sourceTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, source Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(sourceTestDataLoader.dataset),
            100. * correct / len(sourceTestDataLoader.dataset)))


    if taragetTestDataLoader==None:
        return
    # target Test
    correct = 0
    testLoss = 0
    with torch.no_grad():
        for data, targetLabel in taragetTestDataLoader:
            if args.n_dim == 0:
                data, targetLabel = data.to(args.device), targetLabel.to(args.device)
            elif args.n_dim > 0:
                imgSize = torch.sqrt(
                    (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                data = data.expand(len(data), args.n_dim, imgSize.item(), imgSize.item()).to(
                    device)
                targetLabel = targetLabel.to(device)
            Output = model(data)[0]
            testLoss += F.nll_loss(F.log_softmax(Output, dim=1), targetLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = Output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()
        testLoss /= len(taragetTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(taragetTestDataLoader.dataset),
            100. * correct / len(taragetTestDataLoader.dataset)))
    return correct




class ADDAModel(nn.Module):

    def __init__(self, args):
        super(ADDAModel, self).__init__()
        self.args=args
        if args.data_name == 'Digits':
            self.backbone = network_dict['LeNet'](args.n_dim)
            if args.bottleneck:
                self.bottleneck = nn.Sequential(
                    nn.Linear(800, args.bottleneck_dim),
                    nn.BatchNorm1d(args.bottleneck_dim),
                    nn.ReLU()
                )
                self.classifier_feature_dim = args.bottleneck_dim
            else:
                self.classifier_feature_dim = 256
            # nn.Sequential(
            #     nn.Linear(self.bottleneck_dim, 500),
            #     nn.ReLU(),
            #     nn.Dropout(p=0.5),
            #     nn.Linear(500, self.num_classes)
            # )
            self.classifier = nn.Sequential(
                nn.Linear(self.classifier_feature_dim, args.n_labels)
            )

        elif args.data_name == 'office':
            self.backbone = network_dict['ResNet50']()
            if args.bottleneck:
                self.bottleneck = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    nn.Flatten(),
                    nn.Linear(self.backbone.out_features, args.bottleneck_dim),
                    nn.BatchNorm1d(args.bottleneck_dim),
                    nn.ReLU(),
                )
                self.classifier_feature_dim = args.bottleneck_dim
            else:
                self.classifier_feature_dim = self.backbone.out_features
            self.classifier = nn.Sequential(

                nn.Linear(self.classifier_feature_dim, args.n_labels)
            )
        # D
        if args.randomized:
            in_feature=args.randomized_dim
        else:
            in_feature =self.classifier_feature_dim * args.n_labels
        hidden_size=args.hidden_size

        if args.batch_norm:
            self.discriminator = nn.Sequential(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def forward(self,data):

        feature = self.backbone(data)
        if self.args.bottleneck:
            feature=self.bottleneck(feature)
        label=self.classifier(feature)


        return label,feature
