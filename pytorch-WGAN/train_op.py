import torch
import numpy as np
from torchvision.utils import save_image
import tqdm
def train_op(G,D,opt,train_dataLoader):
    Tensor = torch.FloatTensor

    if opt.cuda:
        Tensor = torch.cuda.FloatTensor
        G.cuda()
        D.cuda()

        print("Cuda可用")
    else:
        print("Cuda不可用")


    # Optimizers
    optimizer_G = torch.optim.RMSprop(G.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.RMSprop(D.parameters(), lr=opt.lr)







    # ----------
    #  Training
    # ----------

    for epoch in range(opt.epochs):
        # print("*"*10+"epoch:",epoch)
        for i, (imgs, _) in tqdm.tqdm(enumerate(train_dataLoader),total=len(train_dataLoader),desc='Train epoch = {}'.format(epoch), ncols=80,leave=False):
            imgs = imgs.cuda()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))

            # Measure discriminator's ability to classify real from generated samples
            fake_img= G(z).detach()

            d_loss = -torch.mean(D(imgs))+torch.mean(D(fake_img))

            d_loss.backward()
            optimizer_D.step()


            # Clip weights of discriminator
            num_parameters=0
            num1 = 0
            num2 = 0
            for p in D.parameters():
                num_parameters+=(p.data.size(0)*p.data.size(-1))
                p.data.clamp_(-opt.clip_value, opt.clip_value)
                # print(num_parameters)
                num1+=float(p.data[p.data>(opt.clip_value*0.9)].size(0))
                num2+=float(p.data[p.data<(-opt.clip_value*0.9)].size(0))

            if i % opt.n_critic == 0:
                print(num_parameters, '%.2f' % (num1 / num_parameters), '%.2f' % (num2 / num_parameters),
                      '%.2f' % ((num1 + num2) / num_parameters))
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()
                fake_img = G(z)
                # Loss measures generator's ability to fool the discriminator
                g_loss = -torch.mean(D(fake_img))

                g_loss.backward()
                optimizer_G.step()

                if i % opt.log_interval==0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.epochs, i, len(train_dataLoader), d_loss.item(), g_loss.item())
                    )

            batches_done = epoch * len(train_dataLoader) + i
            if batches_done % opt.save_interval == 0:
                save_image(fake_img.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)





