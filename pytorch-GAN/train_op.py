import torch
import numpy as np
from torchvision.utils import save_image
import tqdm
def train_op(G,D,opt,train_dataLoader):
    Tensor = torch.FloatTensor
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # Discriminator label
    valid = torch.ones(opt.batch_size).float()
    fake = torch.zeros(opt.batch_size).float()

    if opt.cuda:
        Tensor = torch.cuda.FloatTensor
        G.cuda()
        D.cuda()
        valid=valid.cuda()
        fake=fake.cuda()
        adversarial_loss.cuda()

        print("Cuda可用")
    else:
        print("Cuda不可用")


    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))







    # ----------
    #  Training
    # ----------

    for epoch in range(opt.epochs):
        # print("*"*10+"epoch:",epoch)
        for i, (imgs, _) in tqdm.tqdm(enumerate(train_dataLoader),total=len(train_dataLoader),desc='Train epoch = {}'.format(epoch), ncols=80,leave=False):
            # -----------------
            #  Train Generator
            # -----------------
            imgs=imgs.cuda()
            optimizer_G.zero_grad()
            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
            # Generate a batch of images
            gen_imgs = G(z)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(D(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(D(imgs), valid)
            fake_loss = adversarial_loss(D(gen_imgs.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % opt.log_interval==0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.epochs, i, len(train_dataLoader), d_loss.item(), g_loss.item())
                )

            batches_done = epoch * len(train_dataLoader) + i
            if batches_done % opt.save_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)





