import torch.nn as nn



#USPS->MNIST
class MCDAModel(nn.Module):

    def __init__(self, args):
        super(MCDAModel, self).__init__()
        self.args=args
        # softmax层
        self.softmax = nn.Softmax(dim=1)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(args.n_dim, 32, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
            nn.Dropout2d(0)
        )


        self.classifier = nn.Sequential(
            nn.Linear(1024, args.n_labels),
            nn.Softmax()
        )

        # D
        self.critics=[nn.Sequential(
            nn.Linear(1024,args.was_dim),
            nn.LeakyReLU(),
            nn.Linear(args.was_dim,1)
        ) for i in range(args.n_labels+1)]



    def forward(self,sourceData,targetData,training):
        sourceData = sourceData.expand(len(sourceData), self.args.n_dim, 28, 28)
        targetData = targetData.expand(len(targetData), self.args.n_dim, 28, 28)



        # 提取数据特征
        sourceFeature = self.feature_extractor(sourceData)
        targetFeature = self.feature_extractor(targetData)

        sourceLabel=self.classifier(sourceFeature)
        targeteLabel=self.classifier(targetFeature)


        if training:

            return sourceFeature,targetFeature,sourceLabel,targeteLabel
        else:
            return targetFeature,targeteLabel
