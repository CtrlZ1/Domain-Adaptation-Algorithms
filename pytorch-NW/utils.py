import torch
import numpy as np
import torch.autograd as autograd
import os
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad




def form_samples_classwise(samples, num_classes,dataIndex):
    if dataIndex!=8:
        datas=samples.data
        labels=samples.labels
        samplesdatas_cl = [[] for _ in range(num_classes)]
        sampleslabels_cl = [[] for _ in range(num_classes)]

        for index,label in enumerate(labels):
            samplesdatas_cl[label].append(datas[index])
            sampleslabels_cl[label].append(label)
        samplesdatas_cl=np.array(samplesdatas_cl)
        sampleslabels_cl=np.array(sampleslabels_cl)
        return samplesdatas_cl,sampleslabels_cl
    else:
        samples_cl = []
        for cl in range(num_classes):
            samples_cl.append([])

        for sample in samples:
            class_id = sample[1]
            samples_cl[class_id].append(sample)
        return samples_cl



def getGradientPenalty(critic,real_samples,fake_samples,args):
    alpha=torch.rand(real_samples.size(0), 1)
    if not args.noCuda and torch.cuda.is_available():
        alpha=alpha.cuda()
    interpolates=alpha*real_samples+(1-alpha)*fake_samples
    interpolates=torch.stack([interpolates,real_samples,fake_samples]).requires_grad_()
    D_interpolates=critic(interpolates)
    gradients=autograd.grad(
        inputs=interpolates,
        outputs=D_interpolates,
        grad_outputs=torch.ones_like(D_interpolates),
        retain_graph=True,create_graph=True,only_inputs=True
    )[0]

    gradient_penalty= ((gradients.norm(2,dim=1)-1)**2).mean()

    return gradient_penalty



def read_file(path, data_root=None):
    f = open(path, 'r')
    contents = f.readlines()
    samples = []
    for cnt in contents:
        cnt = cnt.rstrip()
        path, lab = cnt.split(',')
        if data_root is not None:
            path = os.path.join(data_root, path)
        lab = int(lab)
        tup = (path, lab)
        samples.append(tup)

    f.close()
    return samples




