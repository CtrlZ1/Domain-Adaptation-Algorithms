import torch
import math
import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from getGradientPenalty import getGradientPenalty
from tes1t import tes1t
from utils import set_requires_grad


def train(epoch, model, sourceDataLoader, targetDataLoader,DEVICE,args):
    model.train()
    critic = model.critic
    classifier=model.classifier
    feature_extractor=model.feature_extractor
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-4)
    clf_optim = torch.optim.Adam(model.parameters(), lr=1e-4)
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
        h_s,h_t=outPut[0],outPut[1]
        # h_s.requires_grad_(False)
        # h_t.requires_grad_(False)

        # Training critic
        set_requires_grad(critic,True)
        for _ in range(args.n_critic):
            gp=getGradientPenalty(critic,h_s,h_t,args)

            Critic_loss=-critic(h_s.detach()).mean()+critic(h_t.detach()).mean()+args.lambda_gp*gp

            critic_optim.zero_grad()
            Critic_loss.backward(retain_graph=True)

            critic_optim.step()
        # Training classifier
        set_requires_grad(critic, False)

        for _ in range(args.n_clf):
            source_features,target_features = model(sourceData,targetData,True)

            source_preds = classifier(source_features)
            clf_loss = clf_criterion(source_preds, sourceLabel)
            wasserstein_distance = critic(source_features).mean() - critic(target_features).mean()
            wd_loss=wasserstein_distance
            classifer_and_wd_loss = clf_loss + args.n_clf * wd_loss
            clf_optim.zero_grad()
            classifer_and_wd_loss.backward()
            clf_optim.step()
            #for par in model.feature_extractor.parameters():
            #    print(par.grad)

        if batch_idx % args.logInterval == 0:
            print(
                '\ncritic_loss: {:.4f},  classifer_loss: {:.4f},  wd_Loss: {:.4f}'.format(
                    Critic_loss.item(), clf_loss.item(), wd_loss.item()))
        # if batch_idx>0 and batch_idx% 200==0:
        #     DEVICE = torch.device('cuda')
        #     t_correct = tes1t(model, targetDataLoader, DEVICE)
        #     print(t_correct)