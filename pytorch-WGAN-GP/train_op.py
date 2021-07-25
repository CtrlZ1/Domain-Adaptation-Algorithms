import torch
import numpy as np
from torchvision.utils import save_image
import tqdm
import torch.autograd as autograd



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
    optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0),1,1,1)))
        interpolates=(alpha * real_samples + (1-alpha) * fake_samples).requires_grad_(True)
        D_interprolates=D(interpolates)

        # Get gradient w.r.t interpolates
        gradients=autograd.grad(
            outputs=D_interprolates,
            inputs=interpolates,
            grad_outputs=torch.ones(interpolates.size(0),1).float().cuda(),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]
        gradients=gradients.view(gradients.size(0),-1)
        gradients_penalty=((gradients.norm(2,dim=1)-1)**2).mean()
        return gradients_penalty




    # ----------
    #  Training
    # ----------

    for epoch in range(opt.epochs):
        # print("*"*10+"epoch:",epoch)
        for i, (imgs, _) in tqdm.tqdm(enumerate(train_dataLoader),total=len(train_dataLoader),desc='Train epoch = {}'.format(epoch), ncols=80,leave=False):
            real_imgs = imgs.cuda()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (real_imgs.shape[0], opt.latent_dim)))

            # Measure discriminator's ability to classify real from generated samples
            fake_img= G(z).detach()

            d_loss = -torch.mean(D(real_imgs))+torch.mean(D(fake_img))+opt.lambda_gp *compute_gradient_penalty(D,real_imgs.data,fake_img.data)

            d_loss.backward()
            optimizer_D.step()


            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()
                fake_img = G(z)
                # Loss measures generator's ability to fool the discriminator
                g_loss = torch.mean(D(real_imgs))-torch.mean(D(fake_img))

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





