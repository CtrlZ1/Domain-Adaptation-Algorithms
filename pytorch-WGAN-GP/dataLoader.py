import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms.transforms as tfs
def getTrainDataLoader(opt):
    train_dataLoader=data.DataLoader(
        dataset=datasets.MNIST(
            root=opt.train_data_path,
            train=True,
            download=True,
            transform=tfs.Compose(
                [tfs.Resize(opt.img_size),tfs.ToTensor(),tfs.Normalize([0.5],[0.5])]
            )
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )

    return train_dataLoader