import torch.nn as nn

class NWModel(nn.Module):

    def __init__(self, args):
        super(NWModel, self).__init__()
        self.n_labels=args.n_labels

        self.netF=nn.Sequential(
            nn.Conv2d(args.n_dim, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.netC = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.n_labels)
        )

        self.netD = nn.Sequential(
            nn.Linear(48 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )




    def forward(self,data,n_dim,imageSize):
        data = data.expand(len(data), n_dim, imageSize, imageSize)



        # 提取数据特征
        datafeature = self.netF(data)

        dataoutput = self.netC(datafeature)


        return dataoutput
