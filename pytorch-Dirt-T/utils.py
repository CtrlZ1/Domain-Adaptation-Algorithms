# author:ACoderlyy
# contact: ACoderlyy@163.com
# datetime:2021/11/11 22:55
# software: PyCharm
import torch
import torch.nn.functional as F
import os
class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


def write_log(filename,content):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            pass
    with open(filename, "a") as f:
        f.write(content+'\n')