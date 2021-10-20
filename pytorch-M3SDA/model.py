import torch.nn as nn

from backBone import network_dict

import torch
import tqdm
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F

import torch.optim as optim

from utils import set_requires_grad, L1_discrepancy, GradientReverseLayer


def euclidean(x1,x2):
	return ((x1-x2)**2).sum().sqrt()

def k_moment(features_norm, k):

    features_norm_new = []
    for f in features_norm:
        features_norm_new.append((f**k).mean(0))
    moment = 0
    for i in range(len(features_norm_new) - 1):
        moment += euclidean(features_norm_new[i], features_norm_new[len(features_norm_new) - 1])
    for i in range(len(features_norm_new) - 1):
        for j in range(i + 1, len(features_norm_new)):
            moment += euclidean(features_norm_new[i], features_norm_new[j])
    return moment

def msda_regulizer(features, belta_moment):
    # print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))
    features_norm = []
    for f in features:
        features_norm.append(f - f.mean(0))
    moment1=0
    for i in range(len(features)-1):
        moment1+=euclidean(features_norm[i],features_norm[len(features)-1])
    for i in range(len(features)-1):
        for j in range(i+1,len(features)):
            moment1+=euclidean(features_norm[i],features_norm[j])

    reg_info = moment1
    # print(reg_info)
    for i in range(belta_moment - 1):
        reg_info += k_moment(features_norm, i + 2)

    return reg_info / 6


def get_Clfloss(Datas,Labels,backbone,classifier1,classifier2,criterion):
    features = []
    for data in Datas:
        features.append(backbone(data))

    correct_c1 = []
    correct_c2 = []

    c1_outputs = []
    c2_outputs = []
    for index,f in enumerate(features[:-1]):
        clf_output1=classifier1(f)
        c1_outputs.append(clf_output1)

        pred = clf_output1.data.max(1)[1]  # get the index of the max log-probability
        correct_c1.append(pred.eq(Labels[index].data.view_as(pred)).cpu().sum())

        clf_output2 = classifier2(f)
        c2_outputs.append(clf_output2)

        pred = clf_output2.data.max(1)[1]  # get the index of the max log-probability
        correct_c2.append(pred.eq(Labels[index].data.view_as(pred)).cpu().sum())



    loss_msda = 0.0005 * msda_regulizer(features, 5)

    loss1 = 0
    loss2 = 0
    for index, f1_o in enumerate(c1_outputs):
        loss1 += criterion(f1_o, Labels[index])

        loss2 += criterion(c2_outputs[index], Labels[index])


    return loss1,loss2,loss_msda,correct_c1,correct_c2

def train_process(model, trainLoader,testLoader,DEVICE,imageSize,n_source,args):

    # model=pre_train(model,sourceDataLoader,sourceTestDataLoader,DEVICE,imageSize,args)

    backbone = model.backbone
    classifier1 = model.classifier1
    classifier2 = model.classifier2
    backbone.train()
    classifier1.train()
    classifier2.train()

    if args.which_opt == 'momentum':
        opt_g = optim.SGD(backbone.parameters(),
                               lr=args.lr, weight_decay=args.l2_Decay,
                               momentum=args.momentum)

        opt_c1 = optim.SGD(classifier1.parameters(),
                                lr=args.lr, weight_decay=args.l2_Decay,
                                momentum=args.momentum)
        opt_c2 = optim.SGD(classifier2.parameters(),
                                lr=args.lr, weight_decay=args.l2_Decay,
                                momentum=args.momentum)

    if args.which_opt == 'adam':
        opt_g = optim.Adam(backbone.parameters(),
                                lr=args.lr, weight_decay=args.l2_Decay)

        opt_c1 = optim.Adam(classifier1.parameters(),
                                 lr=args.lr, weight_decay=args.l2_Decay)
        opt_c2 = optim.Adam(classifier2.parameters(),
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
        opt_g.load_state_dict(checkpoint['opt_g'])
        opt_c1.load_state_dict(checkpoint['opt_c1'])
        opt_c2.load_state_dict(checkpoint['opt_c2'])
        base_epoch = checkpoint['epoch']
    t_correct=0
    for epoch in range(1 + base_epoch, base_epoch + args.epoch + 1):
        model.train()
        correct = 0
        item=0
        correct_c1_iter = np.array([0 for i in range(n_source)])
        correct_c2_iter = np.array([0 for i in range(n_source)])
        for batch_idx, data in tqdm.tqdm(enumerate(trainLoader),
                                                desc='Train epoch = {}'.format(epoch), ncols=80,
                                                leave=False):
            item+=1
            Datas, Labels=data

            for index in range(len(Datas)):
                Datas[index]=Datas[index].expand(len(Datas[index]), args.n_dim, imageSize, imageSize).to(DEVICE)
                Labels[index]=Labels[index].to(DEVICE)

            # The first step
            opt_g.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            loss1,loss2,loss_msda,correct_c1,correct_c2=get_Clfloss(Datas, Labels, backbone, classifier1, classifier2, criterion)

            loss = loss1 + loss2 + loss_msda

            loss.backward()

            opt_g.step()
            opt_c1.step()
            opt_c2.step()

            # The second step: fix G, update C1,C2
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            loss1, loss2, loss_msda,correct_c1,correct_c2 = get_Clfloss(Datas, Labels, backbone, classifier1, classifier2, criterion)

            correct_c1_iter+=np.array(correct_c1)
            correct_c2_iter+=np.array(correct_c2)

            loss_s = loss1 + loss2 + loss_msda

            feat_t = backbone(Datas[-1])
            output_t1 = classifier1(feat_t)
            output_t2 = classifier2(feat_t)
            loss_dis = L1_discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            opt_c1.step()
            opt_c2.step()

            # The third step:fix classifiers, update G
            opt_g.zero_grad()

            for i in range(args.n_clf):
                feat_t = backbone(Datas[-1])
                output_t1 = classifier1(feat_t)
                output_t2 = classifier2(feat_t)
                loss_dis = L1_discrepancy(output_t1, output_t2)
                loss_dis.backward()
                opt_g.step()

            # # stop
            # if batch_idx > 500:
            #     return batch_idx



            if batch_idx % args.logInterval == 0:
                print(
                    '\nbatch_idx:{},ave_loss1: {:.4f},  ave_loss2: {:.4f},loss_msda:{:.4f},loss_dis:{:.4f}'.format(
                        batch_idx,loss1.item()/(len(Datas)-1), loss2.item()/(len(Datas)-1),loss_msda.item(),loss_dis.item()))



        for i in range(len(correct_c1_iter)):

            acc_train1 = float(correct_c1_iter[i]) * 100. / (item * args.batchSize)
            acc_train2 = float(correct_c2_iter[i]) * 100. / (item * args.batchSize)

            print('Train Accuracy C1 and C2 in S{}: {}/{} ({:.2f}%)  {}/{} ({:.2f}%) '.format(
                i+1,correct_c1_iter[i], (item * args.batchSize), acc_train1,correct_c2_iter[i], (item * args.batchSize), acc_train2))

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
                'opt_g': opt_g,
                'opt_c1': opt_c1,
                'opt_c2':opt_c2

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_c1': opt_c1.state_dict(),
                'opt_c2':opt_c2.state_dict()

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
            Datas, Labels = data
            targetData,targetLabel=Datas[-1],Labels[-1]

            imgSize = torch.sqrt(
                (torch.prod(torch.tensor(targetData.size())) / (targetData.size(1) * len(targetData))).float()).int()

            targetData=targetData.expand(len(targetData), args.n_dim, imgSize, imgSize).to(device)
            targetLabel=targetLabel.to(device)
            size += targetLabel.data.size()[0]
            feat = model.backbone(targetData)
            output = model.classifier1(feat)

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
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)

    def forward(self, x,reverse=False):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        if reverse:
            x = GradientReverseLayer(x)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x


class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        # self.fc1 = nn.Linear(8192, 3072)
        # self.bn1_fc = nn.BatchNorm1d(3072)
        # self.fc2 = nn.Linear(3072, 2048)
        # self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        # if reverse:
        #     x = grad_reverse(x, self.lambd)
        # x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x

class M3SDAModel(nn.Module):

    def __init__(self, args):
        super(M3SDAModel, self).__init__()
        self.args=args
        if args.data_name == 'Digits':
            self.backbone = Feature()
            self.classifier1 = Predictor()
            self.classifier2 = Predictor()

        elif args.data_name == 'office':
            self.backbone = network_dict['ResNet50']()
            self.classifier1 = nn.Sequential(
                nn.Linear(2048, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )
            self.classifier2 = nn.Sequential(
                nn.Linear(2048, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )

    def forward(self,sourceData,targetData,source):

        # 提取数据特征
        sourceFeature = self.backbone(sourceData)
        targetFeature = self.backbone(targetData)

        sourceLabel=self.classifier(sourceFeature)
        targeteLabel=self.classifier(targetFeature)


        if source:
            return sourceFeature,sourceLabel
        else:
            return targetFeature,targeteLabel
