import torch
import math
import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from getGradientPenalty import getGradientPenalty
from tes1t import tes1t
from utils import set_requires_grad, onehot_label


def train(epoch, model, sourceDataLoader, targetDataLoader,DEVICE,args):
    model.train()

    critics = [i.to(DEVICE).train() for i in model.critics]

    classifier=model.classifier
    par_critic = [
        {'params': c.parameters()} for c in critics
    ]
    critic_optim = torch.optim.Adam(par_critic, lr=args.lr)
    par_clf_wd=[
        {'params': model.feature_extractor.parameters()},
        {'params': model.classifier.parameters()},
    ]
    clf_optim = torch.optim.Adam(par_clf_wd, lr=args.lr)
    clf_criterion = nn.CrossEntropyLoss()
    lenSourceDataLoader = len(sourceDataLoader)
    for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader),total=lenSourceDataLoader,
                                                            desc='Train epoch = {}'.format(epoch), ncols=80,
                                                            leave=False):

        sourceData, sourceLabel = sourceData.to(DEVICE), sourceLabel.to(DEVICE)

        for targetData, targetLabel in targetDataLoader:
            targetData, targetLabel = targetData.to(DEVICE), targetLabel.to(DEVICE)
            break

        outPut = model(sourceData, targetData, True)
        sourceFeature, targetFeature, sourceLabel_pre, targeteLabel_pre=outPut[0],outPut[1],outPut[2],outPut[3]
		targeteLabel_soft = nn.Softmax(dim=1)(targeteLabel_pre)

        # h_s.requires_grad_(False)
        # h_t.requires_grad_(False)

        # Training critic
        set_requires_grad(critics,True)
        Critic_loss=torch.zeros(1).to(DEVICE)
        T=torch.ones(args.n_labels+1).to(DEVICE)
        ones=torch.ones((args.batchSize,1)).to(DEVICE)
        T=T/torch.sum(T)
        T[args.n_labels]=1
        sourceLabel_onehot=onehot_label(sourceLabel,args.n_labels).to(DEVICE)
        ys_weight=torch.cat([sourceLabel_onehot,ones],dim=1).to(DEVICE)
        yt_weight=torch.cat([targeteLabel_soft,ones],dim=1).to(DEVICE)
        ys_weight = ys_weight / (torch.mean(ys_weight, dim=0) + 1e-6)
        yt_weight = yt_weight / (torch.mean(yt_weight, dim=0) + 1e-6)
        gradient_weights = ys_weight * yt_weight * T
        ys_weight = ys_weight * T
        yt_weight = yt_weight * T

        for _ in range(args.n_critic):
            for i in range(args.n_labels+1):
                gp=getGradientPenalty(critics[i],sourceFeature,targetFeature,args)
                Critic_loss+=(-ys_weight[:,i].view(-1,1)*critics[i](sourceFeature.detach())).mean()+\
                             (yt_weight[:,i].view(-1,1)*critics[i](targetFeature.detach())).mean()+\
                             (gradient_weights[:,i].view(-1,1)*args.lambda_gp*gp).mean()

            critic_optim.zero_grad()
            Critic_loss.backward(retain_graph=True)

            critic_optim.step()
        # Training classifier
        set_requires_grad(critics, False)

        for _ in range(args.n_clf):
            clf_trust_target_loss=torch.zeros(1).to(DEVICE)
            num=0
            for i in targeteLabel_pre:
                if i.data.max()>args.theta:
                    num+=1
                    label=i.view(1,-1).data.max(1)[1]
                    clf_trust_target_loss+=clf_criterion(i.view(1,-1),label)
            clf_loss = clf_criterion(sourceLabel_pre, sourceLabel)
			
            wd_loss=torch.zeros(1).to(DEVICE)
            for i in range(args.n_labels+1):
                if i != args.n_labels:
                    alpha=args.alpha
                else:
                    alpha=alpha*args.n_labels
                wd_loss += alpha * ((ys_weight[:,i] * critics[i](sourceFeature)).mean() - (
                                yt_weight[:,i] * critics[i](targetFeature)).mean())

   
            classifer_and_wd_loss = (clf_loss*(len(sourceLabel)/(num+len(sourceLabel)))+
					(clf_trust_target_loss/num)*(num/(num+len(sourceLabel)))) + args.n_clf * wd_loss
            
			clf_optim.zero_grad()
            classifer_and_wd_loss.backward()
            clf_optim.step()
            #for par in model.feature_extractor.parameters():
            #    print(par.grad)

        if batch_idx % args.logInterval == 0:
            print(
                    '\ncritic_loss: {:.4f},  classifer_loss: {:.4f},  clf_trust_target_loss:{:.6f}, wd_Loss: {:.6f}'.format(
                        Critic_loss.item(), clf_loss.item(), clf_trust_target_loss.item(),wd_loss.item()))
        # if batch_idx>0 and batch_idx% 200==0:
        #     DEVICE = torch.device('cuda')
        #     t_correct = tes1t(model, targetDataLoader, DEVICE)
        #     print(t_correct)