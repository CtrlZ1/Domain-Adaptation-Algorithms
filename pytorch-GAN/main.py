import argparse
import torch
import models
from train_op import train_op
from dataLoader import getTrainDataLoader
import os

os.makedirs("images", exist_ok=True)

parser=argparse.ArgumentParser()
parameters=[
    ["--epochs",int,200,None],
    ["--batch_size",int,8,None],
    ["--lr",float,0.0002,None],
    ["--b1",float,0.5,"adam: decay of first order momentum of gradient"],
    ["--b2",float,0.999,"adam: decay of first order momentum of gradient"],
    ["--latent_dim",int,100,"dimensionality of the latent space"],
    ["--img_size",int,28,None],
    ["--channels",int,1,None],
    ["--save_interval",int,400,"the interval of saving images"],
    ["--cuda",bool,True if torch.cuda.is_available() else False,None],
    ["--train_data_path",str,r"E:\transferlearning\data\MNIST",None],
    ["--log_interval",int,5,None]
]
for par in parameters:
    parser.add_argument(par[0], type=par[1], default=par[2],help=par[3])
opt=parser.parse_args()
print(opt)
# dataLoader
train_dataLoader=getTrainDataLoader(opt)
# models
G=models.Generator(opt)
D=models.Discriminator(opt)



if __name__ == '__main__':
    train_op(G,D,opt,train_dataLoader)