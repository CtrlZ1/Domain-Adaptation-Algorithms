import torch


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram/核矩阵
       source: sample_size_1 * feature_size 的数据
       target: sample_size_2 * feature_size 的数据
       kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
       kernel_num: 表示的是多核的数量
       fix_sigma: 表示是否使用固定的标准差

           return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                           矩阵，表达形式:
                           [	K_ss K_st
                               K_ts K_tt ]
       """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)  # 按特征维进行求和
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # torch.sum 直接是所有元素相加到一个元素
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)  # 这里就是计算K(1/(m(m-1)))
    # 以 fix_sigma为中值，以kernel_mul为倍数取kernel_num个值
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  #combine all kernels

# In fact, it is the formula of multi-core MMD. The formula of ordinary Gaussian kernel
# function can also be easily found on the Internet. It can be rewritten here.
def mmd(source, target, kernel_mul=2.0, kernel_num=5,guass=True,fix_sigma=None):
    if guass:
        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target,
                                  kernel_mul=kernel_mul,
                                  kernel_num=kernel_num,
                                  fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]  # Source<->Source
        YY = kernels[batch_size:, batch_size:]  # Target<->Target
        XY = kernels[:batch_size, batch_size:]  # Source<->Target
        YX = kernels[batch_size:, :batch_size]  # Target<->Source
        loss = torch.mean(XX + YY - XY - YX)  # 这里是假定X和Y的样本数量是相同的
        # 当不同的时候，就需要乘上上面的M矩阵
    else:
        delta = source - target
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss