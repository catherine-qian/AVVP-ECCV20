"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
# from github: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
from __future__ import print_function

import torch
import torch.nn as nn
from utils.basics import *
import torch.nn.functional as F
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.plot(x.data.numpy()) 注意画图时要先colorbar一下，再plt.show()


def contrast(x1, x2, target, criterion2):
    # target [16,25], class label
    # x1, x2 [16, 10, 512], audio/video features


    # ----- 加dropout作为augmentation------

    # # ---把audio当做video的一种augmentation
    # em = x2.shape[-1]  # embedding size
    # target = target.unsqueeze(1).repeat(1, 10, 1) # [16, 10, 25]
    # feats = torch.cat([x1.reshape(-1, em).unsqueeze(1), x2.reshape(-1, em).unsqueeze(1)], dim=1)  # ([160, 2, 512])
    # target = target.reshape(-1, target.shape[-1]) # ([160, 25])

    # ----- concat to contrast
    # single modality
    # feats = torch.cat([x1, x2], dim=2) if len(x1) else x2  # av, video-only feature
    # feats = torch.cat([x1, x2], dim=2) if len(x2) else x1  # av, audio-only feature
    if len(x1) and len(x2):
        feats = torch.cat([x1, x2], dim=2)
    else:
        feats = x2 if len(x2) else x1

    _, T, em = feats.shape  # temporal

    # drop out as augmentation
    # feats_max = F.max_pool1d(feats.permute(0, 2, 1), T).permute(0, 2, 1) # max pooling
    # feats_avg = F.avg_pool1d(feats.permute(0, 2, 1), T).permute(0, 2, 1) # avg pooling
    # featmax = torch.cat([feats_max, F.dropout(feats_max, 0.1), F.dropout(feats_max, 0.3), F.dropout(feats_max, 0.5),
    #                      F.dropout(feats_max, 0.7)], dim=1)
    # featavg = torch.cat([feats_avg, F.dropout(feats_avg, 0.1), F.dropout(feats_avg, 0.3), F.dropout(feats_avg, 0.5),
    #                      F.dropout(feats_avg, 0.7)], dim=1)
    # feats = torch.cat([featmax, featavg], dim=1)

    # cut-off as augmentation
    Clen = round(em * 0.1)  # cut-off length
    cut = torch.randint(0, em - Clen, (1,))
    feats[:, :, cut:cut + Clen] = 0.0

    # compute loss
    labels = bin2dec(target, target.shape[-1]) # creat label

    feats = F.normalize(feats, dim=2)  # feature-level normalization
    loss = criterion2(feats, labels)

    return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`') # give either label or mask
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device) # identity mask
        elif labels is not None: # only provide the class labels
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) # diagonal mask [bs, bs] -> whether the same class
        else:
            mask = mask.float().to(device)  # only provide the mask

        contrast_count = features.shape[1]  # num of augmentations for each example
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # ubind in aug dim, then concat features with the same augmentation

        if self.contrast_mode == 'one':  # only one positive pair
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':  # same class-> all positive pairs
            anchor_feature = contrast_feature
            anchor_count = contrast_count # num of aug
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),  # anchor_feature=contrast_feature: ([bs, em]), 自己相乘
            self.temperature)  # numerator [bs, bs]

        # for numerical stability -> 把matrix压一压防止过大出现nan，最终的Logits就是要的
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # max value in each row
        logits = anchor_dot_contrast - logits_max.detach()  # 所有的similarity score

        # tile mask [bs, bs]
        mask = mask.repeat(anchor_count, contrast_count)  # same class label, 之前的mask没有aug的情况这一维度，这里把他加上去 [bs*au, bs*au]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),  # all one mask [bs*au, bs*au]
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),  # ([bs*au, 1])
            0
        )  # diagonal matrix, 对角线=0
        # 这个mask就是data之间的的最后的label，即为groudtruth，从这里往下的几行很难理解，需要把公式看懂
        mask = mask * logits_mask  # same class mask & 对角线=0

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # 非对角线所有元素score

        # logits就是公式分子上的内容，注意这里不止有正的还有负的，但是最后要求和所有的正样本，只要乘上mask求和就可以了，而log和exp抵消了
        # 后面的部分是公式分母的内容，所有的距离都求和取对数。
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) #  [bs*au, bs*au] - [bs*au, 1]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # [bs*au, 1]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # * scale [bs*au, 1]
        loss = loss.view(anchor_count, batch_size).mean() # ([au, bs]) -> compute mean

        return loss