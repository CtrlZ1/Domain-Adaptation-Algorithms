import torch.nn as nn

from backBone import network_dict

import torch
import tqdm
import torch.nn as nn
import os
import torch.nn.functional as F

import torch.optim as optim

from utils import set_requires_grad


def pre_train(model, sourceDataLoader,sourceTestDataLoader,DEVICE,imageSize,args):
    model.train()
    backbone=model.backbone_s
    classifier=model.classifier
    parameters = [
        {'params': backbone.parameters()},
        {'params': classifier.parameters()},
    ]
    # setup criterion and optimizer
    optimizer = optim.Adam(parameters,lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.n_preclf):
        model.train()
        correct = 0
        lenSourceDataLoader = len(sourceDataLoader)
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):
            optimizer.zero_grad()
            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(
                DEVICE), sourceLabel.to(DEVICE)

            feature=backbone(sourceData)
            pred=classifier(feature)
            source_pre = pred.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()
            loss=criterion(pred,sourceLabel)
            loss.backward()
            optimizer.step()

            if batch_idx % args.logInterval == 0:
                print(
                    '\nclassifier_loss: {:.4f}'.format(
                        loss.item()))
        acc_train = float(correct) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Train Accuracy: {}/{} ({:.2f}%)'.format(
            correct, (lenSourceDataLoader * args.batchSize), acc_train))

        test_process(model, sourceTestDataLoader, None, DEVICE, args)

    return model




def train_process(model, sourceDataLoader, targetDataLoader,sourceTestDataLoader,taragetTestDataLoader,DEVICE,imageSize,args):

    model=pre_train(model,sourceDataLoader,sourceTestDataLoader,DEVICE,imageSize,args)

    backbone_s = model.backbone_s
    backbone_t = model.backbone_t
    classifier = model.classifier
    critic = model.critic

    backbone_s.eval()
    backbone_t.train()
    critic.train()
    classifier.train()


    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr,betas=(0.5, 0.999))
    clf_optim = torch.optim.Adam(backbone_t.parameters(), lr=args.lr,betas=(0.5, 0.999))

    criterion = nn.CrossEntropyLoss()#BCEWithLogitsLoss()
    lenSourceDataLoader = len(sourceDataLoader)

    base_epoch = 0
    if args.ifload:
        path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2 = os.path.join(path, i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        critic_optim.load_state_dict(checkpoint['critic_optim'])
        clf_optim.load_state_dict(checkpoint['clf_optim'])
        base_epoch = checkpoint['epoch']
    t_correct=0
    for epoch in range(1 + base_epoch, base_epoch + args.epoch + 1):
        model.train()
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

            sourceFeature, sourceLabel_pre = model(sourceData, targetData, True)
            targetFeature, targeteLabel_pre = model(sourceData, targetData, False)

            source_pre = sourceLabel_pre.data.max(1, keepdim=True)[1]
            correct += source_pre.eq(sourceLabel.data.view_as(source_pre)).sum()

            sourceFeature = sourceFeature.view(sourceFeature.shape[0], -1)
            targetFeature = targetFeature.view(targetFeature.shape[0], -1)

            for i in range(args.n_critic):
                discriminator_x = torch.cat([sourceFeature.detach(), targetFeature.detach()])
                discriminator_y = torch.cat([torch.ones(sourceFeature.shape[0], device=DEVICE).long(),
                                             torch.zeros(targetFeature.shape[0], device=DEVICE).long()])
                preds = critic(discriminator_x)#.squeeze()
                Critic_loss = criterion(preds, discriminator_y)
                # Training critic
                critic_optim.zero_grad()
                Critic_loss.backward()
                critic_optim.step()

            set_requires_grad(critic, requires_grad=False)
            for i in range(args.n_clf):
                targetFeature=backbone_t(targetData)
                targetFeature = targetFeature.view(targetFeature.shape[0], -1)
                discriminator_y = torch.ones(targetData.shape[0], device=DEVICE).long()
                preds = critic(targetFeature)#.squeeze()
                clf_loss = criterion(preds, discriminator_y)

                clf_optim.zero_grad()
                clf_loss.backward()
                clf_optim.step()

            set_requires_grad(critic, requires_grad=True)




            if batch_idx % args.logInterval == 0:
                print(
                    '\ncritic_loss: {:.4f},  classifer_loss: {:.4f}'.format(
                        Critic_loss.item(), clf_loss.item()))


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
                'critic_optim': critic_optim,
                'clf_optim': clf_optim,

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'critic_optim': critic_optim.state_dict(),
                'clf_optim': clf_optim.state_dict(),

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
            Output = model(data, data, True)[1]
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
            Output = model(data, data, False)[1]
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
            self.backbone_s = network_dict['LeNet'](args.n_dim)
            self.backbone_t = network_dict['LeNet'](args.n_dim)
            self.classifier = nn.Sequential(
                nn.Linear(256, args.n_labels),
            )
            # D
            self.critic = nn.Sequential(
                nn.Linear(256, 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500, 2),
            )
        elif args.data_name == 'office':
            self.backbone_s = network_dict['ResNet50']()
            self.backbone_t = network_dict['ResNet50']()
            self.classifier = nn.Sequential(
                nn.Linear(2048, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )
            # D
            self.critic = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 3072),
                nn.ReLU(),
                nn.Linear(3072, 2)
            )





    def forward(self,sourceData,targetData,source):

        # 提取数据特征
        sourceFeature = self.backbone_s(sourceData)
        targetFeature = self.backbone_t(targetData)

        sourceLabel=self.classifier(sourceFeature)
        targeteLabel=self.classifier(targetFeature)


        if source:
            return sourceFeature,sourceLabel
        else:
            return targetFeature,targeteLabel
