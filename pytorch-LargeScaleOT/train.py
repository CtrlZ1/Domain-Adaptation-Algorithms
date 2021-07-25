import torch
import tqdm

import torch.nn as nn

from utils import set_requires_grad

def pre_train(model,sourceDataLoader,DEVICE,args):
    model.train()
    clf_criterion = nn.CrossEntropyLoss()
    lenSourceDataLoader = len(sourceDataLoader)
    clf_optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    # pre-train
    for _ in range(args.n_pretrain):
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='preTrain epoch = {}'.format(_), ncols=80,
                                                              leave=False):
            sourceData, sourceLabel = sourceData.to(DEVICE), sourceLabel.to(DEVICE)
            output = model.output_pretrain(sourceData)
            clf_optim.zero_grad()
            loss_s = clf_criterion(output, sourceLabel)
            loss_s.backward()
            clf_optim.step()

            if batch_idx % args.logInterval == 0:
                print(
                    '\nLoss_s: {:.4f}'.format(
                        loss_s.item()))

def common_train(criterion, optimizer, batch_generator, device, args,scheduler=None):
    losses = []
    for epoch in tqdm.tqdm(range(args.epoch)):
        epoch_loss = 0
        n_batches = 0
        for (x_idx, x), (y_idx, y) in batch_generator:
            optimizer.zero_grad()
            loss = criterion(x_idx, x.to(device), y_idx, y.to(device))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / n_batches)

        if scheduler is not None:
            scheduler.step()

    return losses
def train(epoch, net_W, net_g, sourceDataLoader, targetDataLoader,DEVICE,args):
    net_W.train()
    net_g.train()

    g_optim = torch.optim.Adam(net_g.parameters(), lr=args.lr)
    clf_optim = torch.optim.Adam(net_W.parameters(), lr=args.lr)
    clf_criterion = nn.CrossEntropyLoss()
    lenSourceDataLoader = len(sourceDataLoader)



    for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader),total=lenSourceDataLoader,
                                                            desc='Train epoch = {}'.format(epoch), ncols=80,
                                                            leave=False):

        sourceData, sourceLabel = sourceData.to(DEVICE), sourceLabel.to(DEVICE)

        for targetData, targetLabel in targetDataLoader:
            targetData, targetLabel = targetData.to(DEVICE), targetLabel.to(DEVICE)
            break

        sourceFeature, targetFeature, pre_s, pre_t, C = net_W(sourceData, targetData, True)#sourceFeature,targetFeature,pre_s,pre_t,C


        # Training KantorovichPotential network————>g
        set_requires_grad(net_g, True)
        for _ in range(args.n_g):
            g_optim.zero_grad()
            # only update the network g
            loss_opt=net_g(sourceFeature.detach(),targetFeature.detach(),C,args)
            loss_opt.backward(retain_graph=True)
            g_optim.step()
            if batch_idx % args.logInterval == 0 and _ % 2 == 0:
                print(
                    '\nloss_opt: {:.4f}'.format(
                        loss_opt.item()))
        # Training the network W
        set_requires_grad(net_g, False)
        clf_optim.zero_grad()
        loss_s=clf_criterion(pre_s,sourceLabel)
        pre_t=nn.Softmax(dim=1)(pre_t)
        loss_t=torch.mean(-pre_t*torch.log(pre_t))
        loss_opt = net_g(sourceFeature, targetFeature, C, args)
        loss=args.beta*loss_t+loss_s+args.opt_lambda*loss_opt
        loss.backward()
        clf_optim.step()


        if batch_idx % args.logInterval == 0:

            print(
                '\nloss: {:.4f},  loss_opt: {:.4f},  Loss_s: {:.4f},  Loss_t: {:.4f}'.format(
                    loss.item(), loss_opt.item(), loss_s.item(), loss_t.item()))
        # if batch_idx>0 and batch_idx% 200==0:
        #     DEVICE = torch.device('cuda')
        #     t_correct = tes1t(model, targetDataLoader, DEVICE)
        #     print(t_correct)