import torch.nn as nn

#下面是用于数字SVHN->MNIST的模型结构
class digistSMNetByWDGRL(nn.Module):

    def __init__(self, numClass):
        super(digistSMNetByWDGRL, self).__init__()
        self.numClasses=numClass
        # softmax层
        self.softmax = nn.Softmax(dim=1)

        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module('f_conv1', nn.Conv2d(3, 32, kernel_size=3))
        self.feature_extractor.add_module('f_bn1', nn.BatchNorm2d(32))
        self.feature_extractor.add_module('f_pool1', nn.MaxPool2d(2,stride=2))
        self.feature_extractor.add_module('f_relu1', nn.ReLU(True))
        self.feature_extractor.add_module('f_conv2', nn.Conv2d(32, 64, kernel_size=3))
        self.feature_extractor.add_module('f_bn2', nn.BatchNorm2d(64))
        #self.feature_extractor.add_module('f_drop1', nn.Dropout2d())
        self.feature_extractor.add_module('f_pool2', nn.MaxPool2d(2,stride=2))
        self.feature_extractor.add_module('f_relu2', nn.ReLU(True))
        self.feature_extractor.add_module('f_conv3', nn.Conv2d(64, 128, kernel_size=3))
        self.feature_extractor.add_module('f_bn3', nn.BatchNorm2d(128))
        self.feature_extractor.add_module('f_pool3', nn.MaxPool2d(2, stride=2))
        self.feature_extractor.add_module('f_relu3', nn.ReLU(True))

        # 标签分类器Gy
        # 源数据全连接层
        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1', nn.Linear(128, 2048))
        self.classifier.add_module('c_bn1', nn.BatchNorm1d(2048))
        self.classifier.add_module('c_relu1', nn.ReLU(True))
        self.classifier.add_module('c_drop1', nn.Dropout2d())
        self.classifier.add_module('c_fc2', nn.Linear(2048, 2048))
        self.classifier.add_module('c_bn2', nn.BatchNorm1d(2048))
        self.classifier.add_module('c_relu2', nn.ReLU(True))
        # 划分十类
        self.classifier.add_module('c_fc3', nn.Linear(2048, self.numClasses))

        # WD
        self.critic=nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )


    def forward(self,sourceData,targetData,training):
        sourceData = sourceData.expand(len(sourceData), 3, 28, 28)
        targetData = targetData.expand(len(targetData), 3, 28, 28)



        # 提取数据特征
        sourceFeature = self.feature_extractor(sourceData)
        targetFeature = self.feature_extractor(targetData)

        h_s=sourceFeature.view(sourceFeature.size(0),-1)
        h_t=targetFeature.view(targetFeature.size(0),-1)

        if training:

            return h_s,h_t
        else:
            return h_t
