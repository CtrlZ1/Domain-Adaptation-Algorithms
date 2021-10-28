from torch.optim.lr_scheduler import LambdaLR

from backBone import network_dict

import torch
import tqdm
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F

import torch.optim as optim

from utils import set_requires_grad, GradientReverseLayer, \
    get_lam, binary_cross_entropyloss


def train_process(model, trainLoader,testLoader,DEVICE,imageSize,args):

    # model=pre_train(model,sourceDataLoader,sourceTestDataLoader,DEVICE,imageSize,args)

    model.train()
    backbone=model.backbone
    G = model.G
    classifier = model.classifier
    discriminator=model.discriminators



    pars = [
        {'params': classifier.parameters()},
        {'params': backbone.parameters()}
    ]

    if args.which_opt == 'momentum':
        opt_f_c = optim.SGD(pars,
                               lr=args.lr, weight_decay=args.l2_Decay,
                               momentum=args.momentum)

        opt_g = optim.SGD(G.parameters(),
                                lr=args.lr, weight_decay=args.l2_Decay,
                                momentum=args.momentum)

        opt_d = optim.SGD(discriminator.parameters(),
                          lr=args.lr, weight_decay=args.l2_Decay,
                          momentum=args.momentum)


    if args.which_opt == 'adam':
        opt_f_c = optim.Adam(pars,
                                lr=args.lr, weight_decay=args.l2_Decay)

        opt_g = optim.Adam(G.parameters(),
                                 lr=args.lr, weight_decay=args.l2_Decay)

        opt_d = optim.Adam(discriminator.parameters(),
                           lr=args.lr, weight_decay=args.l2_Decay)


    criterion = nn.CrossEntropyLoss().to(DEVICE)#BCEWithLogitsLoss()





    base_epoch = 0
    if args.ifload:
        path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2 = os.path.join(path, i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        opt_f_c.load_state_dict(checkpoint['opt_f_c'])
        opt_g.load_state_dict(checkpoint['opt_g'])
        opt_d.load_state_dict(checkpoint['opt_d'])
        base_epoch = checkpoint['epoch']
    t_correct=0

    learningRate_f_c = LambdaLR(opt_f_c,
                            lambda x: (1. + args.lr_gamma * (float(x) / (base_epoch + args.epoch))) ** (-args.lr_decay))
    learningRate_g = LambdaLR(opt_f_c,
                            lambda x: (1. + args.lr_gamma * (float(x) / (base_epoch + args.epoch))) ** (-args.lr_decay))
    learningRate_d = LambdaLR(opt_f_c,
                            lambda x: (1. + args.lr_gamma * (float(x) / (base_epoch + args.epoch))) ** (-args.lr_decay))

    for epoch in range(1 + base_epoch, base_epoch + args.epoch + 1):
        model.train()
        allnum = 0
        item=0
        print(learningRate_f_c.get_lr(),learningRate_g.get_lr(),learningRate_d.get_lr())
        correct_ave = 0

        lam=get_lam(epoch,base_epoch + args.epoch + 1,args.gamma)



        for batch_idx, data in tqdm.tqdm(enumerate(trainLoader),
                                                desc='Train epoch = {}'.format(epoch), ncols=80,
                                                leave=False):
            item+=1
            Datas, Labels=data

            for index in range(len(Datas)):
                Datas[index]=Datas[index].expand(len(Datas[index]), args.n_dim, imageSize, imageSize).to(DEVICE)
                Labels[index]=Labels[index].long().to(DEVICE)

            sourceDatas=Datas[0]
            sourceLabels=Labels[0]


            for i in range(1,len(Datas)-1):
                sourceDatas=torch.cat((sourceDatas,Datas[i]),dim=0)
                sourceLabels=torch.cat((sourceLabels,Labels[i]),dim=0)


            targetDatas=Datas[-1]


            source_feature=backbone(sourceDatas)
            target_feature=backbone(targetDatas)


            # Update G
            Gs = G(sourceDatas)
            set_requires_grad(discriminator,False)
            discriminate_source_g = discriminator(source_feature.detach())
            discriminate_target_g = discriminator(target_feature.detach())

            bce_weight = lambda input, target, weight: binary_cross_entropyloss(input, target, weight)
            bce_noweight = lambda input, target: binary_cross_entropyloss(input, target)
            # F.binary_cross_entropy_with_logits()
            d_label_one = torch.ones((discriminate_source_g.size(0), 1)).to(DEVICE)
            d_label_zero = torch.zeros((discriminate_target_g.size(0), 1)).to(DEVICE)

            weight = Gs
            l_wdom = -lam*(bce_weight(discriminate_source_g,d_label_one,weight)+bce_noweight(discriminate_target_g,d_label_zero))
            l_wdom_first=l_wdom.item()
            opt_g.zero_grad()
            l_wdom.backward()
            opt_g.step()
            learningRate_g.step(epoch)


            # update  discriminators
            set_requires_grad(discriminator, True)
            discriminate_source_d = discriminator(source_feature.detach())
            discriminate_target_d = discriminator(target_feature.detach())

            l_dom = lam * (bce_noweight(discriminate_source_d, d_label_one) + bce_noweight(discriminate_target_d,
                                                                                           d_label_zero))
            opt_d.zero_grad()
            l_dom.backward()
            opt_d.step()
            learningRate_d.step(epoch)

            # update  backbone and classifier
            Gs = G(sourceDatas)
            weight =Gs
            set_requires_grad(discriminator, False)
            grad_reverse=GradientReverseLayer()
            source_feature_reverse = grad_reverse(source_feature)
            target_feature_reverse = grad_reverse(target_feature)
            discriminate_source = discriminator(source_feature_reverse)
            discriminate_target = discriminator(target_feature_reverse)

            l_wdom = lam * (bce_weight(discriminate_source, d_label_one, weight) + bce_noweight(discriminate_target,
                                                                                               d_label_zero))

            source_output=classifier(source_feature)

            allnum+=len(source_output)
            pred = source_output.data.max(1)[1]  # get the index of the max log-probability
            correct_ave+=(pred.eq(sourceLabels.data.view_as(pred)).cpu().sum())

            l_clf=criterion(source_output,sourceLabels)
            loss=l_clf+l_wdom

            opt_f_c.zero_grad()
            loss.backward()
            opt_f_c.step()
            learningRate_f_c.step(epoch)



            if batch_idx % args.logInterval == 0:
                print(
                    '\nbatch_idx:{},l_wdom: {:.4f},  l_dom: {:.4f},l_clf:{:.4f}'.format(
                        batch_idx,l_wdom_first, l_dom.item(),l_clf.item()))





        print('Train Ave Accuracy in Sources: {}/{} ({:.2f}%)  '.format(
            correct_ave,allnum, 100.*correct_ave/allnum))

        test_correct=test_process(model, testLoader, DEVICE, args)
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
                'opt_f_c': opt_f_c,
                'opt_g': opt_g,
                'opt_d': opt_d

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'opt_f_c': opt_f_c.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d':opt_d.state_dict()

            }
        path+='/'+args.model_name+'_epoch'+str(args.epoch)+'.pth'
        torch.save(state, path)


def test_process(model,testLoader, device, args):
    model.eval()

    # target Test
    correct = 0
    testLoss = 0
    size = 0
    with torch.no_grad():
        for data in testLoader:
            targetData,targetLabel=data

            imgSize = torch.sqrt(
                (torch.prod(torch.tensor(targetData.size())) / (targetData.size(1) * len(targetData))).float()).int()

            targetData=targetData.expand(len(targetData), args.n_dim, imgSize, imgSize).to(device)
            targetLabel=targetLabel.to(device)
            size += targetLabel.data.size()[0]
            feat = model.backbone(targetData)

            output = model.classifier(feat)

            testLoss += F.nll_loss(F.log_softmax(output, dim=1), targetLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()
        testLoss /= size
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, size,
            100. * correct / size))
    return correct


class Feature(nn.Module):
    def __init__(self,n_dim):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(n_dim, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x

class G(nn.Module):
    def __init__(self,n_dim):
        super(G, self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(n_dim, 6, 5))
        layer1.add_module('bn1', nn.BatchNorm2d(6))
        layer1.add_module('relu1', nn.ReLU())
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5))
        layer2.add_module('bn2', nn.BatchNorm2d(16))
        layer2.add_module('relu2', nn.ReLU())
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400, 120))
        layer3.add_module('relu3', nn.ReLU())
        layer3.add_module('fc2', nn.Linear(120, 1))
        layer3.add_module('sf',nn.Softmax(dim=0))
        self.layer3 = layer3
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class CMSSModel(nn.Module):

    def __init__(self, args):
        super(CMSSModel,self).__init__()
        self.classifier_feature_dim=2048
        self.args=args
        if args.data_name == 'Digits':
            self.backbone = Feature(args.n_dim)
            self.classifier = nn.Sequential(
                nn.Linear(2048, 10)
            )

            self.discriminators=nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )
            self.G=G(args.n_dim)

        elif args.data_name == 'Office':
            self.backbone = network_dict['ResNet101']()
            self.classifier = nn.Sequential(
                nn.Linear(50, 31)
            )
            self.discriminators = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )
            self.G = G(args.n_dim)

