import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import os
def tes1t(model, testDataLoader,DEVICE):
    model.eval()
    testLoss = 0
    correct = 0
    num=0
    with torch.no_grad():
        for data, targetLabel in testDataLoader:
            data, targetLabel = data.to(DEVICE), targetLabel.to(DEVICE)
            sourceOutput = model.classifier(model(data, data, False))
            testLoss += F.nll_loss(F.log_softmax(sourceOutput, dim = 1), targetLabel, size_average=False).item() # sum up batch loss
            pred = sourceOutput.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()

        testLoss /= len(testDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct, len(testDataLoader.dataset),
            100. * correct / len(testDataLoader.dataset)))
    return correct


def destAndReturnImg(model, testDataLoader,DEVICE,args):
    model.eval()
    testLoss = 0
    correct = 0
    num = 0
    wrongNum=0
    mean = [0.485, 0.456, 0.406]  # 自己设置的
    std = [0.229, 0.224, 0.225]  # 自己设置的
    with torch.no_grad():
        for data, target in testDataLoader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            sourceOutput = model(data, data, False)
            testLoss += F.nll_loss(F.log_softmax(sourceOutput, dim=1), target,
                                   size_average=False).item()  # sum up batch loss
            pred = sourceOutput.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # 分类错误的图片保存在一个文件夹里。
            ifchose = pred.eq(target.data.view_as(pred))

            for index, i in enumerate(ifchose):
                if not i:  # 即分类错误
                    wrongNum+=1
                    num+=1
                    # 先根据类别判断文件夹是否存在
                    path = 'E:/毕设/分类错误的数据/' + str(target.data.view_as(pred)[index].item()+1)#+1是因为它是从0开始算的
                    if not os.path.exists(path):
                        os.makedirs(path)
                    # 保存文件

                    # toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
                    # pic = toPIL(data[index].cpu())
                    # pic.save(path + '/' + str(num) + '.jpg')
                    #  反标准化
                    image_numpy=data[index].cpu().float().numpy()
                    for i in range(len(mean)):
                        image_numpy[i] = image_numpy[i] * std[i] + mean[i]
                    image_numpy = image_numpy * 255
                    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
                    image_numpy=image_numpy.astype(np.uint8)
                    im_array = Image.fromarray(image_numpy)
                    im_array.save(path + '/' + str(num) + '-'+str(pred[index].item()+1)+'.jpg')
                    print("已保存图片num:",num)

        testLoss /= len(testDataLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), wrongNum:{}\n'.format(
            testLoss, correct, len(testDataLoader.dataset),
            100. * correct / len(testDataLoader.dataset),wrongNum))
    return correct
