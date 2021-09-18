import torch.nn as nn
import torchvision

from backBone import network_dict

import torch
import tqdm
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from utils import set_requires_grad, model_feature_tSNE

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def train_process(model, sourceDataLoader, targetDataLoader,sourceTestDataLoader,taragetTestDataLoader,DEVICE,imageSize,args):


    G = model.generator
    D = model.discriminator

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.0005)

    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.0005)
    true_labels = Variable(torch.LongTensor(np.ones(args.batchSize * 2, dtype=np.int)).to(DEVICE))
    fake_labels = Variable(torch.LongTensor(np.zeros(args.batchSize * 2, dtype=np.int)).to(DEVICE))

    criterion = nn.CrossEntropyLoss()
    mse_loss_criterion = torch.nn.MSELoss()
    lenSourceDataLoader = len(sourceDataLoader)

    base_epoch = 0
    if args.ifload:
        path = args.savePath + args.model_name
        for i in os.listdir(path):
            path2 = os.path.join(path, i)
            break
        checkpoint = torch.load(path2)
        model.load_state_dict(checkpoint['net'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
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

            # train D

            optimizer_D.zero_grad()
            noise = Variable(torch.randn(args.batchSize, args.latent_dims)).cuda()
            # Adversarial part for true images
            true_outputs, true_feat_a, true_feat_b = D(sourceData, targetData)
            true_loss = criterion(true_outputs, true_labels)
            _, true_predicts = torch.max(true_outputs.data, 1)
            true_acc = float((true_predicts == 1).sum()) / (1.0 * true_predicts.size(0))

            # Adversarial part for fake images
            fake_images_a, fake_images_b = G(noise)
            fake_outputs, fake_feat_a, fake_feat_b = D(fake_images_a, fake_images_b)
            fake_loss = criterion(fake_outputs, fake_labels)
            _, fake_predicts = torch.max(fake_outputs.data, 1)
            fake_acc = float((fake_predicts == 0).sum()) / (1.0 * fake_predicts.size(0))
            dummy_tensor = Variable(torch.zeros(fake_feat_a.size(0), fake_feat_a.size(1), fake_feat_a.size(2), fake_feat_a.size(3))).cuda()
            mse_loss = mse_loss_criterion(fake_feat_a - fake_feat_b, dummy_tensor) * fake_feat_a.size(
                1) * fake_feat_a.size(2) * fake_feat_a.size(3)

            # Classification loss
            cls_outputs = D.classify_a(sourceData)
            cls_loss = criterion(cls_outputs, sourceLabel)
            _, cls_predicts = torch.max(cls_outputs.data, 1)
            cls_acc = float((cls_predicts == sourceLabel.data).sum()) / (1.0 * cls_predicts.size(0))
            correct+=(cls_predicts == sourceLabel.data).sum()

            d_loss = true_loss + fake_loss + args.mse_weight * mse_loss + args.cls_weight * cls_loss

            d_loss.backward()
            optimizer_D.step()

            ad_acc, mse_loss, cls_acc = 0.5 * (true_acc + fake_acc), mse_loss, cls_acc

            # train G
            optimizer_G.zero_grad()
            noise = Variable(torch.randn(args.batchSize, args.latent_dims)).cuda()
            fake_images_a, fake_images_b = G(noise)
            fake_outputs, fake_feat_a, fake_feat_b = D(fake_images_a, fake_images_b)
            fake_loss = criterion(fake_outputs, true_labels.cuda())
            fake_loss.backward()
            optimizer_G.step()


            if batch_idx % args.logInterval == 0:
                print(
                    '\nad_acc: {:.4f},  mse_loss: {:.4f}, cls_acc: {:.4f}, fake_loss:{:.4f}'.format(
                        ad_acc, mse_loss, cls_acc, fake_loss.item()))


        acc_train = float(correct) * 100. / (lenSourceDataLoader * args.batchSize)

        print('Train Accuracy: {}/{} ({:.2f}%)'.format(
            correct, (lenSourceDataLoader * args.batchSize), acc_train))

        test_correct=test_process(model, sourceTestDataLoader,taragetTestDataLoader, DEVICE, args)
        if test_correct > t_correct:
            t_correct = test_correct
        print("max correct:" , t_correct)


        if epoch % args.logInterval == 0:
            # model_feature_tSNE(model, sourceTestDataLoader, taragetTestDataLoader, 'epoch' + str(epoch), DEVICE,
            #                    args.model_name)
            # test_score_a = compute_test_score(trainer.dis.classify_a, train_loader_a)
            dirname = 'images'
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            img_filename = 'gen_%08d.jpg' % (epoch)
            fake_images = torch.cat((fake_images_a, fake_images_b), 3)
            torchvision.utils.save_image(args.scale * (fake_images.data - args.bias), img_filename)

    if args.ifsave:
        path=args.savePath+args.model_name
        if not os.path.exists(path):
            os.makedirs(path)

        if args.if_saveall:
            state = {
                'epoch': args.epoch,
                'net': model,
                'optimizer_G': optimizer_G,
                'optimizer_D': optimizer_D,

            }
        else:
            state = {
                'epoch': args.epoch,
                'net': model.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),

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
            Output = model.discriminator.classify_a(data)
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
            Output = model.discriminator.classify_b(data)
            testLoss += F.nll_loss(F.log_softmax(Output, dim=1), targetLabel,
                                   size_average=False).item()  # sum up batch loss
            pred = Output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()
        testLoss /= len(taragetTestDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, target Test Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(taragetTestDataLoader.dataset),
            100. * correct / len(taragetTestDataLoader.dataset)))
    return correct




# Discriminator Model
class CoDis28x28(nn.Module):
    def __init__(self):
        super(CoDis28x28, self).__init__()
        self.conv0_a = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv0_b = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.pool0 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(50, 500, kernel_size=4, stride=1, padding=0)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(500, 2, kernel_size=1, stride=1, padding=0)
        self.conv_cl = nn.Conv2d(500, 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x_a, x_b):
        h0_a = self.pool0(self.conv0_a(x_a))
        h0_b = self.pool0(self.conv0_b(x_b))
        h1_a = self.pool1(self.conv1(h0_a))
        h1_b = self.pool1(self.conv1(h0_b))
        h2_a = self.prelu2(self.conv2(h1_a))
        h2_b = self.prelu2(self.conv2(h1_b))
        h2 = torch.cat((h2_a, h2_b), 0)
        h3 = self.conv3(h2)
        return h3.squeeze(), h2_a, h2_b

    def classify_a(self, x_a):
        h0_a = self.pool0(self.conv0_a(x_a))
        h1_a = self.pool1(self.conv1(h0_a))
        h2_a = self.prelu2(self.conv2(h1_a))
        h3_a = self.conv_cl(h2_a)
        return h3_a.squeeze()

    def classify_b(self, x_b):
        h0_b = self.pool0(self.conv0_b(x_b))
        h1_b = self.pool1(self.conv1(h0_b))
        h2_b = self.prelu2(self.conv2(h1_b))
        h3_b = self.conv_cl(h2_b)
        return h3_b.squeeze()


# Generator Model
class CoGen28x28(nn.Module):
    def __init__(self, latent_dims):
        super(CoGen28x28, self).__init__()
        self.dconv0 = nn.ConvTranspose2d(latent_dims, 1024, kernel_size=4, stride=1)
        self.bn0 = nn.BatchNorm2d(1024, affine=False)
        self.prelu0 = nn.PReLU()
        self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512, affine=False)
        self.prelu1 = nn.PReLU()
        self.dconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256, affine=False)
        self.prelu2 = nn.PReLU()
        self.dconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.prelu3 = nn.PReLU()
        self.dconv4_a = nn.ConvTranspose2d(128, 1, kernel_size=6, stride=1, padding=1)
        self.dconv4_b = nn.ConvTranspose2d(128, 1, kernel_size=6, stride=1, padding=1)
        self.sig4_a = nn.Sigmoid()
        self.sig4_b = nn.Sigmoid()

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        h0 = self.prelu0(self.bn0(self.dconv0(z)))
        h1 = self.prelu1(self.bn1(self.dconv1(h0)))
        h2 = self.prelu2(self.bn2(self.dconv2(h1)))
        h3 = self.prelu3(self.bn3(self.dconv3(h2)))
        out_a = self.sig4_a(self.dconv4_a(h3))
        out_b = self.sig4_b(self.dconv4_b(h3))
        return out_a, out_b



class CoGANModel(nn.Module):

    def __init__(self, args):
        super(CoGANModel, self).__init__()
        self.args=args
        self.generator=CoGen28x28(args.latent_dims)
        self.discriminator=CoDis28x28()

