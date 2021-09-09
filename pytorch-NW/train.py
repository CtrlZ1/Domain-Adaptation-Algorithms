import torch
import math
import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from utils import getGradientPenalty
from tes1t import tes1t
from utils import set_requires_grad
import torch.optim as optim


def train(s_trainloader_classwise,model, sourceDataLoader, targetDataLoader,targetTestLoader,imageSize,DEVICE,args):

    if args.datasetIndex==8:
        sourceTrainLoader, sourceValiLoader=sourceDataLoader[0],sourceDataLoader[1]
        targetTrainLoader, targetValiLoader=targetDataLoader[0],targetDataLoader[1]

    else:
        sourceTrainLoader=sourceDataLoader
        targetTrainLoader=targetDataLoader

    netD = model.netD
    netC=model.netC
    netF=model.netF
    # Create optimizers
    if args.adam:
        optimizerF = optim.Adam(netF.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizerC = optim.Adam(netC.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
        pi = nn.Parameter(torch.FloatTensor(args.n_labels).fill_(1.0 / args.n_labels).cuda())
        optimizerPi = optim.Adam(iter([pi]), lr=args.lrPi, betas=(0.5, 0.999))
    else:
        optimizerF = optim.SGD(netF.parameters(), lr=args.lr, momentum=0.9)
        optimizerC = optim.SGD(netC.parameters(), lr=args.lr, momentum=0.9)
        optimizerD = optim.SGD(netD.parameters(), lr=args.lr, momentum=0.9)
        pi = nn.Parameter(torch.FloatTensor(args.n_labels).fill_(1.0 / args.n_labels).cuda())
        optimizerPi = optim.SGD(iter([pi]), lr=args.lrPi)
    def _zero_grad():
        optimizerF.zero_grad()
        optimizerC.zero_grad()
        optimizerD.zero_grad()
        optimizerPi.zero_grad()
    clf_criterion = nn.CrossEntropyLoss()
    lenSourceDataLoader = len(sourceTrainLoader)

    s_classwise_iterators = []
    for i in range(len(s_trainloader_classwise)):
        s_classwise_iterators.append(iter(s_trainloader_classwise[i]))

    def sample_classwise(class_id):
        try:
            batch = next(s_classwise_iterators[class_id])
        except StopIteration:
            s_classwise_iterators[class_id] = iter(s_trainloader_classwise[class_id])
            batch = next(s_classwise_iterators[class_id])
        return batch

    curr_iter=0
    for epoch in range(1, args.epoch + 1):
        model.train()

        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceTrainLoader),total=lenSourceDataLoader,
                                                                desc='Train epoch = {}'.format(epoch), ncols=80,
                                                                leave=False):
            curr_iter+=1
            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(DEVICE), sourceLabel.to(DEVICE)

            for targetData, targetLabel in targetTrainLoader:
                targetData, targetLabel = targetData.expand(len(targetData), args.n_dim, imageSize, imageSize).to(DEVICE), targetLabel.to(DEVICE)
                break

            # compute classifier loss of C and F
            source_features=netF(sourceData)
            source_outputs=netC(source_features)
            loss_cls=clf_criterion(source_outputs,sourceLabel)

            # compute Domain classifier loss of D
            target_features=netF(targetData)
            target_domain_output=netD(target_features)

            for mode in range(args.n_labels):
                s_inputs_classwise, s_lab_cl = sample_classwise(mode)
                s_inputs_classwise = s_inputs_classwise.to(DEVICE)

                if mode == 0:
                    source_domain_output = F.softmax(pi, dim=0)[mode] * \
                                   netD(netF(s_inputs_classwise))
                else:
                    source_domain_output = source_domain_output + F.softmax(pi, dim=0)[mode] * \
                                   netD(netF(s_inputs_classwise))
            loss_D=target_domain_output-source_domain_output

            # compute GradientPenalty loss of D
            loss_gp = getGradientPenalty(netD, source_features, target_features, args)

            # backward loss_D and loss_gp, then update parameters of D
            _zero_grad()
            loss_D_gp=loss_D+loss_gp
            loss_D_gp.backward(loss_D_gp.clone().detach(),retain_graph=True)
            optimizerD.step()
            # backward loss_pi and update parameters of pi
            optimizerPi.zero_grad()
            loss_pi=source_domain_output
            loss_pi.backward(loss_pi.clone().detach(),retain_graph=True)
            optimizerPi.step()



            # backward loss_cls and update parameters of F and C
            gamma = 1
            critic_iters = 10
            _zero_grad()
            if curr_iter % critic_iters == 0:
                loss_wd = gamma * source_domain_output-gamma * target_domain_output
                loss_wd.backward(loss_wd.clone().detach())
            loss_cls.backward()
            optimizerC.step()
            optimizerF.step()

            if batch_idx % args.logInterval == 0:
                print(
                    '\nD_loss: {:.4f},  gp_loss: {:.4f}, classifer_loss: {:.4f},  pi_Loss: {:.4f}'.format(
                        loss_D.mean().item(),loss_gp.item(), loss_cls.item(),loss_pi.mean().item()))


        if args.datasetIndex!=8:
            tes1t(model,targetTestLoader,args,imageSize,DEVICE)
        else:
            tes1t(model, [sourceValiLoader,targetValiLoader], args, imageSize, DEVICE)