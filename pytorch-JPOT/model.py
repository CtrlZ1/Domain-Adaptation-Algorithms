import numpy as np

import tqdm
import ot
import torch.optim as optim
from ot import unif
from torch.optim.lr_scheduler import LambdaLR

from backBone import network_dict
from utils import euclidean_dist, expanddata, computeD, get_r2, getsourceDataByClass
import torch
import torch.nn as nn
import torch.nn.functional as F



# use DeepJDOT model
def train_process(model,sourceDataLoader,targetDataLoader,args,device,dataIndex,method='sinkhorn',metric='deep',reg_sink=1):

    optimizer = optim.SGD([
        {'params': model.backbone.parameters()},
        {'params': model.bottleneck.parameters(), 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ], args.lr, momentum=args.momentum, weight_decay=args.l2_Decay, nesterov=True)

    # learningRate = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epoch), 0.75)
    learningRate = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    evaluate_nomean = nn.CrossEntropyLoss(reduction='none')
    evaluate_mean = nn.CrossEntropyLoss()
    lenSourceDataLoader=len(sourceDataLoader)
    label_propagation_correct=0
    batchsize=0
    for e in range(args.epoch):

        model.train()
        correct=0
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(e), ncols=80,
                                                              leave=False):
            optimizer.zero_grad()
            sourceData, sourceLabel = expanddata(sourceData, sourceLabel, args.n_dim, device)

            for targetData, targetLabel in targetDataLoader:
                targetData,targetLabel=expanddata(targetData,targetLabel,args.n_dim,device)

                batchsize = len(targetData)
                break

            fea_source,pre_sourcelabel=model(sourceData)
            fea_target,pre_targetlabel=model(targetData)

            pre=pre_sourcelabel.data.max(1)[1]
            correct+=pre.eq(sourceLabel.data.view_as(pre)).cpu().sum()


            if metric=='original':
                C=euclidean_dist(sourceData,targetData,square=True)
            elif metric=='deep':
                C=euclidean_dist(fea_source,fea_target,square=True)

            if method == 'sinkhorn':
                r = ot.sinkhorn(ot.unif(fea_source.size(0)), ot.unif(fea_target.size(0)), C.detach().cpu().numpy(),
                                    reg=reg_sink)
            elif method == 'emd':
                r = ot.emd(ot.unif(fea_source.size(0)), ot.unif(fea_target.size(0)), C.detach().cpu().numpy())

            r=torch.tensor(r,dtype=torch.float).to(device)
            Rou = torch.mean(C * r)

            r2=get_r2(r,C,Rou).to(device)

            L_kno=torch.sum(C * r2)

            dirt_r=r-r2

            L_unk=(1.0/args.eta)*torch.sum(dirt_r*torch.log(1+torch.exp(-1*args.eta*C)))

            L_p=L_kno+L_unk

            U_j=torch.sum(dirt_r,dim=0).view(args.batchSize,1)# [batchsize,1]

            N_t_unk=torch.sum(U_j)

            # sourceDataByClass_sum=[torch.zeros(1,fea_source.size(1)) for i in range(args.n_labels)]
            sourcefeatureByClass=getsourceDataByClass(model,sourceData,sourceLabel,args.n_labels,device)

            # sourceNumberByClass=[0 for i in range(args.n_labels)]
            #
            # for index,i in enumerate(sourceLabel):
            #     sourceDataByClass_sum[i]+=(fea_source[index])
            #     sourceNumberByClass[i]+=1

            # [n_lebals,feature_dim]
            mu_A=[]
            for index in range(args.n_labels-1):
                mu_A.append((1.0/float(torch.abs(torch.tensor(len(sourcefeatureByClass[index])))))*torch.sum(sourcefeatureByClass[index],dim=0))
                
            
            L_dc=torch.zeros(1)
            for index in range(args.n_labels-1):
                L_dc+=torch.sum(euclidean_dist(sourcefeatureByClass[index],mu_A[index].view(1,-1),square=True))

            # [n_labels,feature_dim,feature_dim]
            sigma_s_zByClass = [torch.zeros(fea_source.size(1),fea_source.size(1)).to(device) for i in range(args.n_labels-1)]
            for index in range(args.n_labels-1):
                # print(index)
                # c=fea_source[index]-mu_A[i]
                # a=torch.transpose(fea_source[index]-mu_A[i],0,1)
                # b=fea_source[index]-mu_A[i]
                sigma_s_zByClass[index]+=torch.matmul(torch.transpose(sourcefeatureByClass[index]-mu_A[index],0,1),sourcefeatureByClass[index]-mu_A[index])

            # [n_labels,feature_dim，feature_dim]
            sigma_A=[]
            for index in range(args.n_labels-1):

                sigma_A.append((1.0/float(torch.abs(torch.tensor(len(sourcefeatureByClass[index])-1))))*sigma_s_zByClass[index])


            # [1,feature_dim]
            mu_t_unk=(1.0/float(N_t_unk))*torch.sum(fea_target*U_j)
            # [n_labels+1,feature_dim]
            mu_A.append(mu_t_unk)

            # [feature_dim,feature_dim]
            sigma_t_unk=(1.0/float(torch.abs(N_t_unk)))*torch.matmul(torch.transpose(U_j,0,1)*torch.transpose(fea_target-mu_t_unk,0,1),fea_target-mu_t_unk)
            # [n_labels+1,feature_dim，feature_dim]
            sigma_A.append(sigma_t_unk)

            L_dp=torch.zeros(1)
            # computeD:[n_labels+1,batchsize]
            for z in range(args.n_labels):
                L_dp+=computeD(fea_target,mu_A,sigma_A,device)[z].view(1,-1)*evaluate_nomean(
                    pre_targetlabel,torch.tensor(z).expand_as(sourceLabel).to(device)).view(-1,1)

            L_d=L_dc+L_dp

            L_cls=evaluate_mean(pre_sourcelabel,torch.tensor(args.n_labels).expand_as(pre_sourcelabel))

            L=L_cls+args.alpha*L_p+args.beta*L_d

            L.backward()

            # loss=l_t
            trainingloss = L.item()
            trainingl_cls = L_cls.item()
            trainingl_kno = L_kno.item()
            trainingl_unk = L_unk.item()
            trainingl_dc = L_dc.item()
            trainingl_dp = L_dp.item()
            optimizer.step()
            learningRate.step(e)

            if batch_idx%args.logInterval==0:
                print("training loss:{:.4f},l_cls:{:.4f},L_kno:{:.4f},L_unk{:.4f},L_dc:{:.4f},L_dp:{:.4f}，"
                      "args.alpha*L_p:{:.4f},args.beta*L_d:{:.4f}".
                      format(trainingloss,trainingl_cls,trainingl_kno,trainingl_unk,trainingl_dc,trainingl_dp,
                             args.alpha*L_p.item(),args.beta*L_d.item()))
        if e%args.logInterval==0:
            print("training acc...")
            allnumber=lenSourceDataLoader*args.batchSize
            print("training correct:"+correct+",all:"+allnumber+",rate:"+float(correct)/allnumber)

            print("testing acc...")
            tes1t(targetDataLoader, model, args.n_dim, device,dataIndex)
            # label propagation
            print("label propagation acc...")
            # print(float(label_propagation_correct),len(targetDataLoader.dataset))
            print(float(label_propagation_correct) / (lenSourceDataLoader * batchsize))

    print("testing acc...")
    tes1t(targetDataLoader, model, args.n_dim, device,dataIndex)
    # label propagation
    print("label propagation acc...")
    # print(float(label_propagation_correct),len(targetDataLoader.dataset))
    print(float(label_propagation_correct) / (lenSourceDataLoader*batchsize))
#Label propagation
def Label_propagation(Xt,Ys,g):
    ys = Ys
    xt=Xt
    yt = np.zeros((len(np.unique(ys)), xt.shape[0]))  # [n_labels,n_target_sample]
    # let labels start from a number
    ysTemp = np.copy(ys)  # ys、ysTemp:[n_source_samples,]
    classes = np.unique(ysTemp)
    n = len(classes)
    ns = len(ysTemp)

    # perform label propagation
    transp = g / np.sum(g, 1)[:,None]  # coupling_[i]:[n_source_samples,n_target_samples]

    # set nans to 0
    transp[~ np.isfinite(transp)] = 0


    D1 = np.zeros((n, ns))  # [n_labels,n_source_samples]

    for c in classes:
        D1[int(c), ysTemp == c] = 1

    # compute propagated labels
    # / len(ys)=/ k, means uniform sources transfering
    yt = yt + np.dot(D1, transp) / len(
        ys)  # np.dot(D1, transp):[n_labels,n_target_samples] show the mass of every class for transfering to target samples

    return yt.T #n_samples,n_labels
def tes1t(DataLoader,model,n_dim,DEVICE,dataIndex):
    model.eval()
    testLoss = 0
    correct = 0
    num=0
    with torch.no_grad():
        for data, targetLabel in DataLoader:
            if n_dim==0:
                data, targetLabel = data.to(DEVICE), targetLabel.to(DEVICE)
            elif n_dim>0:
                imgSize = torch.sqrt(
                    (torch.prod(torch.tensor(data.size())) / (data.size(1) * len(data))).float()).int()
                data = data.expand(len(data), n_dim, imgSize.item(), imgSize.item()).to(
                    DEVICE)
                targetLabel = targetLabel.to(DEVICE)
            newTargetLabel=targetLabel.clone()
            for index,i in enumerate(targetLabel):
                if dataIndex == 6:
                    if i>=5:
                        newTargetLabel[index]=5
            pre_label = model(data)
            testLoss += F.nll_loss(F.log_softmax(pre_label, dim = 1), newTargetLabel, size_average=False).item() # sum up batch loss
            pred = pre_label.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(newTargetLabel.data.view_as(pred)).cpu().sum()

        testLoss /= len(DataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(DataLoader.dataset),
            100. * correct / len(DataLoader.dataset)))
    return correct



# One layer network (classifer needs to be optimized)
class JPOTModel(nn.Module):

    def __init__(self,n_class,device,n_dim,backbone_name='LeNet'):
        super(JPOTModel, self).__init__()
        self.n_class=n_class
        self.backbone=network_dict[backbone_name](n_dim=n_dim) if backbone_name=='LeNet' else network_dict[backbone_name]()
        self.backbone_dim=10 if backbone_name=='LeNet' else 2048
        self.bottleneck=nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.LeakyReLU()
        ).to(device)
        self.classifier=nn.Sequential(
            nn.Linear(128, n_class),
        ).to(device)

    def forward(self,data):
        data_feature=self.backbone(data)
        data_feature=self.bottleneck(data_feature)

        ouput=self.classifier(data_feature)

        return data_feature,ouput




