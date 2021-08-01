import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import os
def tes1t(model, pre_label,true_label,DEVICE):
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