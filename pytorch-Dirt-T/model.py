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

from utils import ConditionalEntropyLoss, write_log
from vat import VAT


def train_process(model, sourceDataLoader, targetDataLoader, taragetTestDataLoader, DEVICE, imageSize, args):
    # model=pre_train(model,sourceDataLoader,sourceTestDataLoader,DEVICE,imageSize,args)

    model.train()
    # discriminator network
    discriminator = model.discriminator

    # classifier network.
    classifier = model.classifier

    # loss functions
    cent = ConditionalEntropyLoss().cuda()
    xent = nn.CrossEntropyLoss(reduction='mean').cuda()
    sigmoid_xent = nn.BCEWithLogitsLoss(reduction='mean').cuda()
    vat_loss = VAT(classifier, args).cuda()

    # optimizer.
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # loss params.
    dw = 1e-2
    cw = 1
    sw = 1
    tw = 1e-2
    bw = 1e-2

    ''' Exponential moving average (simulating teacher model) '''
    ema = EMA(0.998)
    ema.register(classifier)

    base_epoch = 0
    if args.ifload:
        path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2 = os.path.join(path, i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])
        optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])

        base_epoch = checkpoint['epoch']
    t_correct = 0

    learningRate_cls = LambdaLR(optimizer_cls,
                                lambda x: (1. + args.lr_gamma * (float(x) / (base_epoch + args.epoch))) ** (
                                    -args.lr_decay))
    learningRate_disc = LambdaLR(optimizer_disc,
                                 lambda x: (1. + args.lr_gamma * (float(x) / (base_epoch + args.epoch))) ** (
                                     -args.lr_decay))
    # learningRate_g = LambdaLR(opt_g,
    #                         lambda x: (1. + args.lr_gamma * (float(x) / (base_epoch + args.epoch))) ** (-args.lr_decay))
    lenSourceDataLoader = len(sourceDataLoader)

    for epoch in range(1 + base_epoch, base_epoch + args.epoch + 1):
        model.train()
        allnum = 0
        item = 0
        print(learningRate_cls.get_lr(), learningRate_disc.get_lr())
        lrlist = [str(i) for i in learningRate_cls.get_lr()] + [str(i) for i in learningRate_disc.get_lr()]
        lrlist = ' '.join(lrlist)  # 把列表中的元素放在空串中，元素间用空格隔开
        write_log(args.logPath, lrlist)
        correct = 0
        for batch_idx, (sourceData, sourceLabel) in tqdm.tqdm(enumerate(sourceDataLoader), total=lenSourceDataLoader,
                                                              desc='Train epoch = {}'.format(epoch), ncols=80,
                                                              leave=False):

            sourceData, sourceLabel = sourceData.expand(len(sourceData), args.n_dim, imageSize, imageSize).to(
                DEVICE), sourceLabel.long().to(DEVICE)

            for targetData, targetLabel in targetDataLoader:
                targetData, targetLabel = targetData.expand(len(targetData), args.n_dim, imageSize, imageSize).to(
                    DEVICE), targetLabel.long().to(DEVICE)
                break

            feats_source, pred_source = classifier(sourceData)
            feats_target, pred_target = classifier(targetData, track_bn=True)

            pred = pred_source.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(sourceLabel.data.view_as(pred)).cpu().sum()
            allnum += len(sourceLabel)

            ' Discriminator losses setup. '
            # discriminator loss.
            real_logit_disc = discriminator(feats_source.detach())
            fake_logit_disc = discriminator(feats_target.detach())

            loss_disc = 0.5 * (
                    sigmoid_xent(real_logit_disc, torch.ones_like(real_logit_disc, device='cuda')) +
                    sigmoid_xent(fake_logit_disc, torch.zeros_like(fake_logit_disc, device='cuda'))
            )
            # Update discriminator.
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            ' Classifier losses setup. '
            # supervised/source classification.
            loss_src_class = xent(pred_source, sourceLabel)

            # conditional entropy loss.
            loss_trg_cent = cent(pred_target)

            # virtual adversarial loss.
            loss_src_vat = vat_loss(sourceData, pred_source)
            loss_trg_vat = vat_loss(targetData, pred_target)

            # domain loss.
            real_logit = discriminator(feats_source)
            fake_logit = discriminator(feats_target)
            loss_domain = 0.5 * (
                    sigmoid_xent(real_logit, torch.zeros_like(real_logit, device='cuda')) +
                    sigmoid_xent(fake_logit, torch.ones_like(fake_logit, device='cuda'))
            )

            # combined loss.
            loss_main = (
                    dw * loss_domain +
                    cw * loss_src_class +
                    sw * loss_src_vat +
                    tw * loss_trg_cent +
                    tw * loss_trg_vat
            )

            ' Update network(s) '

            # Update classifier.
            optimizer_cls.zero_grad()
            loss_main.backward()
            optimizer_cls.step()

            # Polyak averaging.
            ema(classifier)  # TODO: move ema into the optimizer step fn.

            if batch_idx % args.logInterval == 0:
                seq = '\nbatch_idx:{},loss_domain: {:.4f},  loss_src_class: {:.4f},loss_src_vat:{:.4f},loss_trg_cent:{:.4f},loss_trg_vat:{:.4f}'.format(
                    batch_idx, loss_domain.item(), loss_src_class.item(), loss_src_vat.item(), loss_trg_cent.item(),
                    loss_trg_vat.item())
                print(seq)
                write_log(args.logPath, 'epoch:' + str(epoch) + ", batchindex:" + str(batch_idx) + "," + seq)

            # if batch_idx >= 1:
            #     break

        if args.ifDirtt:
            prev_tar_pred = None
            kl = nn.KLDivLoss(size_average=False, reduce=True)
            for targetData, targetLabel in tqdm.tqdm(targetDataLoader):
                targetData, targetLabel = targetData.expand(len(targetData), args.n_dim, imageSize, imageSize).to(
                    DEVICE), targetLabel.long().to(DEVICE)

                feats_target, pred_target = classifier(targetData, track_bn=True)
                # conditional entropy loss.
                loss_trg_cent = cent(pred_target)

                # virtual adversarial loss.
                loss_trg_vat = vat_loss(targetData, pred_target)

                if prev_tar_pred is not None and len(targetData) == args.batchSize:
                    lagrange_mult = kl(pred_target.detach(), prev_tar_pred)
                    prev_tar_pred = pred_target
                else:
                    lagrange_mult = torch.tensor(0)

                loss = args.lambda_t * (loss_trg_vat + loss_trg_cent)  +args.beta_t * lagrange_mult

                # Update classifier.
                optimizer_cls.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_cls.step()
                # Polyak averaging.
                ema(classifier)  # TODO: move ema into the optimizer step fn.

        print('Train Ave Accuracy in Sources: {}/{} ({:.2f}%)  '.format(
            correct, allnum, 100. * correct / allnum))
        write_log(args.logPath, 'Train Ave Accuracy in Sources: {}/{} ({:.2f}%)  '.format(
            correct, allnum, 100. * correct / allnum))

        test_correct = test_process(model, taragetTestDataLoader, DEVICE, args)
        if test_correct > t_correct:
            t_correct = test_correct
        print("max correct:", t_correct)
        write_log(args.logPath, "max correct:" + str(t_correct))

        # if epoch % args.logInterval == 0:
        #     model_feature_tSNE(model, sourceTestDataLoader, taragetTestDataLoader, 'epoch' + str(epoch), DEVICE,
        #                        args.model_name)

    if args.ifsave:
        path = args.savePath + args.model_name
        if not os.path.exists(path):
            os.makedirs(path)
        if args.if_saveall:
            state = {
                'epoch': args.epoch,
                'net': model,
                'optimizer_cls': optimizer_cls,
                'optimizer_disc': optimizer_disc,

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'optimizer_cls': optimizer_cls.state_dict(),
                'optimizer_disc': optimizer_disc.state_dict(),

            }
        path += '/' + args.model_name + '_epoch' + str(args.epoch) + '.pth'
        torch.save(state, path)


def test_process(model, testLoader, device, args):
    model.eval()

    # target Test
    correct = 0
    testLoss = 0
    size = 0
    with torch.no_grad():
        for data in testLoader:
            targetData, targetLabel = data

            imgSize = torch.sqrt(
                (torch.prod(torch.tensor(targetData.size())) / (targetData.size(1) * len(targetData))).float()).int()

            targetData = targetData.expand(len(targetData), args.n_dim, imgSize, imgSize).to(device)
            targetLabel = targetLabel.to(device)
            size += targetLabel.data.size()[0]
            _, output = model.classifier(targetData)

            testLoss += F.nll_loss(F.log_softmax(output, dim=1), targetLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()
        testLoss /= size
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, size,
            100. * correct / size))
        write_log(args.logPath, '\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, size,
            100. * correct / size))
    return correct


class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0).cuda()

    def forward(self, x):
        if self.training:
            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=0, std=self.sigma)
            x = x + sampled_noise
        return x


class Classifier(nn.Module):
    def __init__(self, args, large=False):
        super(Classifier, self).__init__()

        n_features = 192 if large else 64

        self.feature_extractor = nn.Sequential(
            nn.InstanceNorm2d(3, momentum=1, eps=1e-3),  # L-17
            nn.Conv2d(3, n_features, 3, 1, 1),  # L-16
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-16
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-16
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-15
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-15
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-15
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-14
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-14
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-14
            nn.MaxPool2d(2),  # L-13
            nn.Dropout(0.5),  # L-12
            GaussianNoise(args.gaussian_noise),  # L-11
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-10
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-10
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-10
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-9
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-9
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-9
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-8
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-8
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-8
            nn.MaxPool2d(2),  # L-7
            nn.Dropout(0.5),  # L-6
            GaussianNoise(args.gaussian_noise),  # L-5
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-4
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-4
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-4
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-3
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-3
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-3
            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-2
            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-2
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-2
            nn.AdaptiveAvgPool2d(1),  # L-1
            nn.Conv2d(n_features, args.n_labels, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.track_running_stats = False

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track

    def forward(self, x, track_bn=False):

        if track_bn:
            self.track_bn_stats(True)

        features = self.feature_extractor(x)
        logits = self.classifier(features)

        if track_bn:
            self.track_bn_stats(False)

        return features, logits.view(x.size(0), 10)


class Discriminator(nn.Module):
    def __init__(self, large=False):
        super(Discriminator, self).__init__()

        n_features = 192 if large else 64

        self.disc = nn.Sequential(
            nn.Linear(n_features * 1 * 8 * 8, 100),
            nn.ReLU(True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.disc(x).view(x.size(0), -1)


class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]


class VADAModel(nn.Module):

    def __init__(self, args):
        super(VADAModel, self).__init__()
        self.discriminator = Discriminator(large=args.large)
        self.classifier = Classifier(args, large=args.large)

