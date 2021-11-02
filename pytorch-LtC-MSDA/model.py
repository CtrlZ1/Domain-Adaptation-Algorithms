from torch.optim.lr_scheduler import LambdaLR

from backBone import network_dict

import torch
import tqdm
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.nn.parameter import Parameter



def train_process(model, trainLoader,testLoader,DEVICE,imageSize,n_sources,args):

    # model=pre_train(model,sourceDataLoader,sourceTestDataLoader,DEVICE,imageSize,args)

    model.train()
    backbone=model.backbone
    gcn = model.GCN
    par = [
        {'params': backbone.parameters()},
        {'params': gcn.parameters()},
    ]

    if args.which_opt == 'momentum':

        opt = optim.SGD(par,lr=args.lr, weight_decay=args.l2_Decay,
                                momentum=args.momentum)


    if args.which_opt == 'adam':
        opt = optim.Adam(par,lr=args.lr, weight_decay=args.l2_Decay)



    criterion = nn.CrossEntropyLoss().to(DEVICE)#BCEWithLogitsLoss()





    base_epoch = 0
    if args.ifload:
        path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2 = os.path.join(path, i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['opt'])

        base_epoch = checkpoint['epoch']
    t_correct=0

    learningRate = LambdaLR(opt,
                            lambda x: (1. + args.lr_gamma * (float(x) / (base_epoch + args.epoch))) ** (-args.lr_decay))
    # learningRate_g = LambdaLR(opt_g,
    #                         lambda x: (1. + args.lr_gamma * (float(x) / (base_epoch + args.epoch))) ** (-args.lr_decay))


    for epoch in range(1 + base_epoch, base_epoch + args.epoch + 1):
        model.train()
        allnum = 0
        item=0
        print(learningRate.get_lr())
        correct_c_iter = np.array([0 for i in range(n_sources)])

        for batch_idx, data in tqdm.tqdm(enumerate(trainLoader),
                                                desc='Train epoch = {}'.format(epoch), ncols=80,
                                                leave=False):
            item+=1
            Datas, Labels=data

            for index in range(len(Datas)):
                Datas[index]=Datas[index].expand(len(Datas[index]), args.n_dim, imageSize, imageSize).to(DEVICE)
                Labels[index]=Labels[index].long().to(DEVICE)

            # Compute features
            features = []
            label_s = []
            for index, data in enumerate(Datas[:-1]):
                features.append(backbone(data))
                label_s.append(Labels[index])
            feature_t=backbone(Datas[-1])
            features.append(feature_t)

            feats = torch.cat(features, dim=0)
            labels = torch.cat(label_s, dim=0)

            # add query samples to the domain graph
            gcn_feats = torch.cat([model.mean, feats], dim=0)
            gcn_adj = model.construct_adj(feats)

            # output classification logit with GCN
            gcn_logit = gcn(gcn_feats, gcn_adj)

            # predict the psuedo labels for target domain
            feat_t_, label_t_ = model.pseudo_label(gcn_logit[-feature_t.shape[0]:, :], feature_t)# trusted feature of target
            features.pop()
            features.append(feat_t_)
            label_s.append(label_t_)

            # update the statistics for source and target domains
            loss_local = model.update_statistics(features, label_s)

            # define GCN classification losses
            domain_logit = gcn_logit[:model.mean.shape[0], :]
            domain_label = torch.cat([torch.arange(model.nclasses)] * model.ndomain, dim=0)
            domain_label = domain_label.long().to(DEVICE)
            loss_cls_dom = criterion(domain_logit, domain_label)

            query_logit = gcn_logit[model.mean.shape[0]:, :]
            loss_cls_src = criterion(query_logit[:-feature_t.shape[0]], labels)

            correct_c = []
            s_label_pre = torch.split(query_logit[:-feature_t.shape[0]], args.batchSize, dim=0)
            s_label_pre=list(s_label_pre)
            for index in range(n_sources):
                s_label_pre[index] = s_label_pre[index].data.max(1)[1]
                correct_c.append(s_label_pre[index].eq(label_s[index].data.view_as(s_label_pre[index])).cpu().sum())
            correct_c_iter += np.array(correct_c)

            target_logit = query_logit[-feature_t.shape[0]:]
            target_prob = F.softmax(target_logit, dim=1)
            loss_cls_tgt = (-target_prob * torch.log(target_prob + 1e-8)).mean()

            loss_cls = loss_cls_dom + loss_cls_src + loss_cls_tgt

            # define relation alignment losses
            loss_global = model.adj_loss() * args.Lambda_global
            loss_local = loss_local * args.Lambda_local
            loss_relation = loss_local + loss_global

            loss = loss_cls + loss_relation

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            learningRate.step(epoch)



            if batch_idx % args.logInterval == 0:
                print(
                    '\nbatch_idx:{},loss_cls_dom: {:.4f},  loss_cls_src: {:.4f},loss_cls_tgt:{:.4f},loss_local:{:.4f},loss_global:{:.4f}'.format(
                        batch_idx,loss_cls_dom.item(),loss_cls_src.item(), loss_cls_tgt.item(),loss_local.item(),loss_global.item()))





        for i in range(len(correct_c_iter)):

            acc_train = float(correct_c_iter[i]) * 100. / (item * args.batchSize)

            print('Train Accuracy in S{}: {}/{} ({:.2f}%)  '.format(
                i+1,correct_c_iter[i], (item * args.batchSize), acc_train))

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
                'opt': opt,

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'opt': opt.state_dict(),

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
            gcn_feats = torch.cat([model.mean, feat], dim=0)
            adj=model.construct_adj(feat)
            output = model.GCN(gcn_feats,adj)
            output = output[model.mean.shape[0]:, :]

            testLoss += F.nll_loss(F.log_softmax(output, dim=1), targetLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()
        testLoss /= size
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, size,
            100. * correct / size))
    return correct


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features,device, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).to(device))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nclasses, device,dropout = 0.5):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.conv_1 = GraphConvolution(nfeat, nfeat,device)
        self.classifier = GraphConvolution(nfeat, nclasses,device)

    def forward(self, x, adj):
        feat_1 = F.relu(self.conv_1(x, adj))
        feat_1 = F.dropout(feat_1, self.dropout, training=self.training)
        logit = self.classifier(feat_1, adj)

        return logit


class LtcMSDAModel(nn.Module):

    def __init__(self, args,n_sources,device):
        super(LtcMSDAModel,self).__init__()
        self.classifier_feature_dim=2048
        self.ndomain=n_sources+1
        self.device=device
        self.args=args
        self.nclasses=args.n_labels
        self.mean = torch.zeros(self.nclasses * self.ndomain, self.classifier_feature_dim).to(device)
        self.adj = torch.zeros(self.nclasses * self.ndomain, self.nclasses * self.ndomain).to(device)
        if args.data_name == 'Digits':
            self.backbone = network_dict['LeNet'](args.n_dim)
            self.GCN = GCN(nfeat=self.classifier_feature_dim, nclasses=self.nclasses,device=device)



        elif args.data_name == 'Office':
            self.backbone = network_dict['ResNet101']()
            self.GCN = GCN(nfeat=self.classifier_feature_dim, nclasses=self.nclasses,device=device)

    # construct the extended adjacency matrix
    def construct_adj(self, feats):
        dist = self.euclid_dist(self.mean, feats)
        sim = torch.exp(-dist / (2 * self.args.sigma ** 2))
        E = torch.eye(feats.shape[0]).float().to(self.device)

        A = torch.cat([self.adj, sim], dim=1)
        B = torch.cat([sim.t(), E], dim=1)
        gcn_adj = torch.cat([A, B], dim=0)

        return gcn_adj

    # assign pseudo labels to target samples
    def pseudo_label(self, logit, feat):
        pred = F.softmax(logit, dim=1)
        entropy = (-pred * torch.log(pred)).sum(-1)
        label = torch.argmax(logit, dim=-1).long()

        mask = (entropy < self.args.entropy_thr).float()
        index = torch.nonzero(mask).squeeze(-1)
        feat_ = torch.index_select(feat, 0, index)
        label_ = torch.index_select(label, 0, index)

        return feat_, label_

    # update prototypes and adjacency matrix
    def update_statistics(self, feats, labels, epsilon=1e-5):
        curr_mean = list()
        num_labels = 0

        for domain_idx in range(self.ndomain):
            tmp_feat = feats[domain_idx]
            tmp_label = labels[domain_idx]
            num_labels += tmp_label.shape[0]

            if tmp_label.shape[0] == 0:
                curr_mean.append(torch.zeros((self.nclasses, self.classifier_feature_dim)).to(self.device))
            else:
                onehot_label = torch.zeros((tmp_label.shape[0], self.nclasses)).scatter_(1,
                                                                                              tmp_label.unsqueeze(
                                                                                                  -1).cpu(),
                                                                                              1).float().cuda()
                domain_feature = tmp_feat.unsqueeze(1) * onehot_label.unsqueeze(-1)
                tmp_mean = domain_feature.sum(0) / (onehot_label.unsqueeze(-1).sum(0) + epsilon)

                curr_mean.append(tmp_mean)

        curr_mean = torch.cat(curr_mean, dim=0)
        curr_mask = (curr_mean.sum(-1) != 0).float().unsqueeze(-1)
        self.mean = self.mean.detach() * (1 - curr_mask) + (
                self.mean.detach() * self.args.beta + curr_mean * (1 - self.args.beta)) * curr_mask
        curr_dist = self.euclid_dist(self.mean, self.mean)
        self.adj = torch.exp(-curr_dist / (2 * self.args.sigma ** 2))

        # compute local relation alignment loss
        loss_local = ((((curr_mean - self.mean) * curr_mask) ** 2).mean(-1)).sum() / num_labels

        return loss_local

    # compute global relation alignment loss
    def adj_loss(self):
        adj_loss = 0

        for i in range(self.ndomain):
            for j in range(self.ndomain):
                adj_ii = self.adj[i * self.nclasses:(i + 1) * self.nclasses,
                         i * self.nclasses:(i + 1) * self.nclasses]
                adj_jj = self.adj[j * self.nclasses:(j + 1) * self.nclasses,
                         j * self.nclasses:(j + 1) * self.nclasses]
                adj_ij = self.adj[i * self.nclasses:(i + 1) * self.nclasses,
                         j * self.nclasses:(j + 1) * self.nclasses]

                adj_loss += ((adj_ii - adj_jj) ** 2).mean()
                adj_loss += ((adj_ij - adj_ii) ** 2).mean()
                adj_loss += ((adj_ij - adj_jj) ** 2).mean()

        adj_loss /= (self.ndomain * (self.ndomain - 1) / 2 * 3)

        return adj_loss

    # compute the Euclidean distance between two tensors
    def euclid_dist(self, x, y):
        x_sq = (x ** 2).mean(-1)
        x_sq_ = torch.stack([x_sq] * y.size(0), dim=1)
        y_sq = (y ** 2).mean(-1)
        y_sq_ = torch.stack([y_sq] * x.size(0), dim=0)
        xy = torch.mm(x, y.t()) / x.size(-1)
        dist = x_sq_ + y_sq_ - 2 * xy

        return dist