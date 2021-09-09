import torch
import torch.nn.functional as F
import numpy as np

def tes1t(model, testDataLoader,args,imageSize,DEVICE):
    if args.datasetIndex == 8:
        sourceDataLoader,targetDataLoader=testDataLoader[0],testDataLoader[1]
    else:
        targetDataLoader=testDataLoader

    model.eval()

    correct = [0] * args.n_labels
    total = [0] * args.n_labels

    testLoss = 0
    correct_num = 0
    with torch.no_grad():
        for data, targetLabel in targetDataLoader:
            data, targetLabel = data.to(DEVICE), targetLabel.to(DEVICE)
            Output = model(data,args.n_dim,imageSize)
            testLoss += F.nll_loss(F.log_softmax(Output, dim = 1), targetLabel, size_average=False).item() # sum up batch loss
            pred = Output.data.max(1)[1] # get the index of the max log-probability
            correct_num += pred.eq(targetLabel.data.view_as(pred)).cpu().sum()

            for j in range(targetLabel.size(0)):
                lab = targetLabel[j]
                correct[lab] += (pred[j] == targetLabel[j]).sum()
                total[lab] += 1

        testLoss /= len(targetDataLoader.dataset)
        print('\nTest target set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correct_num, len(targetDataLoader.dataset),
            100. * correct_num / len(targetDataLoader.dataset)))

        train_acc_classwise = np.array([float(correct[j] * 100) / total[j] for j in range(args.n_labels)])
        print(train_acc_classwise)



        if args.datasetIndex==8:
            testLoss = 0
            correct_num = 0
            with torch.no_grad():
                for data, sourceLabel in sourceDataLoader:
                    data, sourceLabel = data.to(DEVICE), sourceLabel.to(DEVICE)
                    Output = model(data, args.n_dim, imageSize)
                    testLoss += F.nll_loss(F.log_softmax(Output, dim=1), sourceLabel,
                                           size_average=False).item()  # sum up batch loss
                    pred = Output.data.max(1)[1]  # get the index of the max log-probability
                    correct_num += pred.eq(sourceLabel.data.view_as(pred)).cpu().sum()

                testLoss /= len(sourceDataLoader.dataset)
                print('\nTest source set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    testLoss, correct_num, len(sourceDataLoader.dataset),
                    100. * correct_num / len(sourceDataLoader.dataset)))


    return correct_num