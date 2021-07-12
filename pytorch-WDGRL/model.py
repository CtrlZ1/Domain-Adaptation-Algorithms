import torch.nn as nn
import backBone
from ReverseLayer import ReverseLayerF
from getGradientPenalty import getGradientPenalty


class modelNet(nn.Module):
    def __init__(self,numClass,feature_extractor='ResNet50'):
        super(modelNet,self).__init__()
        # 获取基础模型，用于提取特征，为Gf层
        self.feature_extractor=backBone.network_dict[feature_extractor]()
        # 瓶颈层
        self.bottleNeck=nn.Linear(2048,256)
        # 标签分类器Gy
        # 源数据全连接层
        self.classifier=nn.Linear(256,numClass)
        # softmax层
        self.softmax=nn.Softmax(dim=1)
        self.classes=numClass

        # 全局域分类器Gg 此处可以考虑使用BN？
        self.globalDomainClassifier=nn.Sequential()
        self.globalDomainClassifier.add_module('fc1',nn.Linear(256,1024))
        self.globalDomainClassifier.add_module('relu1',nn.ReLU(True))
        self.globalDomainClassifier.add_module('dpt1', nn.Dropout())
        self.globalDomainClassifier.add_module('fc2', nn.Linear(1024, 1024))
        self.globalDomainClassifier.add_module('relu2', nn.ReLU(True))
        self.globalDomainClassifier.add_module('dpt2', nn.Dropout())
        self.globalDomainClassifier.add_module('fc3', nn.Linear(1024, 2))

        # 局部域分类器 Gl
        self.logalDomainClassifier = nn.Sequential()
        for i in range(numClass):
            self.ldc = nn.Sequential()
            self.ldc.add_module('fc1', nn.Linear(256, 1024))
            self.ldc.add_module('relu1', nn.ReLU(True))
            self.ldc.add_module('dpt1', nn.Dropout())
            self.ldc.add_module('fc2', nn.Linear(1024, 1024))
            self.ldc.add_module('relu2', nn.ReLU(True))
            self.ldc.add_module('dpt2', nn.Dropout())
            self.ldc.add_module('fc3', nn.Linear(1024, 2))
            self.logalDomainClassifier.add_module('ldc_' + str(i), self.ldc)

    def forward(self,sourceData,targetData,training,alpha=0.0):

        # 提取源数据特征
        sourceFeature=self.feature_extractor(sourceData)

        s_feature = sourceFeature#.view(sourceOutput.size(0), -1)
        sourceFeature=self.bottleNeck(sourceFeature)

        # 得到源数据分类结果
        sourceOutput=self.classifier(sourceFeature)

        sourceOutput_Softmax=self.softmax(sourceOutput)


        if training==True:

            # 提取目标域数据特征 Gf
            targetFeature = self.feature_extractor(targetData)
            t_feature = targetFeature
            targetFeature = self.bottleNeck(targetFeature)


            # 标签分类器 Gy
            targetOutput = self.classifier(targetFeature)

            targetOutput_Softmax = self.softmax(targetOutput)

            # 全局域分类器 Gg
            # 梯度翻转层
            sourceReverseFeature = ReverseLayerF.apply(sourceFeature, alpha)  # [batchsize,256] ，其实这就是一个全连接层
            targetReverseFeature = ReverseLayerF.apply(targetFeature, alpha)
            # 得到全局域分类器分类结果
            sourceDomainOutput = self.globalDomainClassifier(sourceReverseFeature)
            targetDomainOutput = self.globalDomainClassifier(targetReverseFeature)

            # 局部域分类器 Gl
            sourceTotalOut = []
            targetTotalOut = []
            for i in range(self.classes):
                pSource = sourceOutput_Softmax[:, i].reshape((targetFeature.shape[0],1))#[batchsize,1]
                # 梯度翻转层所有数据，整体与样本的某一类的可能性相乘
                fSource = pSource * sourceReverseFeature
                pTarget = targetOutput_Softmax[:, i].reshape((targetFeature.shape[0],1))
                fTarget = pTarget * targetReverseFeature

                # 域分类
                sourOut = self.logalDomainClassifier[i](fSource)
                sourceTotalOut.append(sourOut)
                targetOut = self.logalDomainClassifier[i](fTarget)
                targetTotalOut.append(targetOut)
            # 返回源数据分类器分类结果，源数据全局域分类器分类结果，目标数据全局分类器分类结果，源数据局部域分类器分类结果，目标数据局部域分类器分类结果
            return sourceOutput, targetOutput, sourceDomainOutput, targetDomainOutput, sourceTotalOut, targetTotalOut,s_feature,t_feature
        else:
            # 返回源数据分类器分类结果
            return sourceOutput





#下面是用于数字MNIST和USPS的模型结构
class digistMUNetByDANN(nn.Module):

    def __init__(self, numClass):
        super(digistMUNetByDANN, self).__init__()
        self.numClasses=numClass
        # softmax层
        self.softmax = nn.Softmax(dim=1)

        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module('f_conv1', nn.Conv2d(1, 32, kernel_size=5))
        self.feature_extractor.add_module('f_bn1', nn.BatchNorm2d(32))
        self.feature_extractor.add_module('f_pool1', nn.MaxPool2d(2,stride=2))
        self.feature_extractor.add_module('f_relu1', nn.ReLU(True))
        self.feature_extractor.add_module('f_conv2', nn.Conv2d(32, 48, kernel_size=5))
        self.feature_extractor.add_module('f_bn2', nn.BatchNorm2d(48))
        #self.feature_extractor.add_module('f_drop1', nn.Dropout2d())
        self.feature_extractor.add_module('f_pool2', nn.MaxPool2d(2,stride=2))
        self.feature_extractor.add_module('f_relu2', nn.ReLU(True))



        # 瓶颈层
        #self.bottleNeck = nn.Linear(48 * 4 * 4, 256)

        # 标签分类器Gy
        # 源数据全连接层
        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1', nn.Linear(48 * 4 * 4, 100))
        self.classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.classifier.add_module('c_relu1', nn.ReLU(True))
        self.classifier.add_module('c_drop1', nn.Dropout2d())
        self.classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.classifier.add_module('c_relu2', nn.ReLU(True))
        # 划分十类
        self.classifier.add_module('c_fc3', nn.Linear(100, self.numClasses))


        # 全局域分类器Gg 此处可以考虑使用BN？
        self.globalDomainClassifier = nn.Sequential()
        self.globalDomainClassifier.add_module('fc1', nn.Linear(768, 1024))
        self.globalDomainClassifier.add_module('relu1', nn.ReLU(True))
        self.globalDomainClassifier.add_module('dpt1', nn.Dropout())
        self.globalDomainClassifier.add_module('fc2', nn.Linear(1024, 1024))
        self.globalDomainClassifier.add_module('relu2', nn.ReLU(True))
        self.globalDomainClassifier.add_module('dpt2', nn.Dropout())
        self.globalDomainClassifier.add_module('fc3', nn.Linear(1024, 2))

        # 局部域分类器 Gl
        self.logalDomainClassifier = nn.Sequential()
        for i in range(numClass):
            self.ldc = nn.Sequential()
            self.ldc.add_module('fc1', nn.Linear(768, 1024))
            self.ldc.add_module('relu1', nn.ReLU(True))
            self.ldc.add_module('dpt1', nn.Dropout())
            self.ldc.add_module('fc2', nn.Linear(1024, 1024))
            self.ldc.add_module('relu2', nn.ReLU(True))
            self.ldc.add_module('dpt2', nn.Dropout())
            self.ldc.add_module('fc3', nn.Linear(1024, 2))
            self.logalDomainClassifier.add_module('ldc_' + str(i), self.ldc)

    def forward(self,sourceData,targetData,training,alpha=0.0):

        #sourceData = sourceData.expand(len(sourceData), 3, 28, 28)
        #targetData = targetData.expand(len(targetData), 3, 28, 28)

        # 提取源数据特征
        sourceFeature = self.feature_extractor(sourceData)
        #print(sourceFeature.size())
        # sourceFeature = self.bottleNeck(sourceFeature)

        # 得到源数据分类结果
        sourceOutput = self.classifier(sourceFeature)
        sourceOutput_Softmax = self.softmax(sourceOutput)

        if training == True:
            # 提取目标域数据特征 Gf
            targetFeature = self.feature_extractor(targetData)

            # targetFeature = self.bottleNeck(targetFeature)
            # 标签分类器 Gy
            targetOutput = self.classifier(targetFeature)

            targetOutput_Softmax = self.softmax(targetOutput)

            # 全局域分类器 Gg
            # 梯度翻转层
            sourceReverseFeature = ReverseLayerF.apply(sourceFeature, alpha)  # [batchsize,256] ，其实这就是一个全连接层
            targetReverseFeature = ReverseLayerF.apply(targetFeature, alpha)
            # 得到全局域分类器分类结果
            sourceDomainOutput = self.globalDomainClassifier(sourceReverseFeature)
            targetDomainOutput = self.globalDomainClassifier(targetReverseFeature)

            # 局部域分类器 Gl
            sourceTotalOut = []
            targetTotalOut = []
            for i in range(self.numClasses):
                pSource = sourceOutput_Softmax[:, i].reshape((targetFeature.shape[0], 1))  # [batchsize,1]
                # 梯度翻转层所有数据，整体与样本的某一类的可能性相乘
                fSource = pSource * sourceReverseFeature
                pTarget = targetOutput_Softmax[:, i].reshape((targetFeature.shape[0], 1))
                fTarget = pTarget * targetReverseFeature

                # 域分类
                sourOut = self.logalDomainClassifier[i](fSource)
                sourceTotalOut.append(sourOut)
                targetOut = self.logalDomainClassifier[i](fTarget)
                targetTotalOut.append(targetOut)
            # 返回源数据分类器分类结果，源数据全局域分类器分类结果，目标数据全局分类器分类结果，源数据局部域分类器分类结果，目标数据局部域分类器分类结果
            return sourceOutput, targetOutput, sourceDomainOutput, targetDomainOutput, sourceTotalOut, targetTotalOut,sourceFeature,targetFeature
        else:
            # 返回源数据分类器分类结果
            return sourceOutput



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
