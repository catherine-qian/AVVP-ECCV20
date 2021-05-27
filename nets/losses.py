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


def contrast(x, target, criterion2):
    # target [16,25], class label
    # x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)  # ([16, 10, 2, 512]) audio-visual aggregated features

    labels = bin2dec(target, target.shape[-1])

    feats = x.reshape(x.shape[0], -1, x.shape[3]) # re-shape to embedding size [16, 20, 512]
    feats = F.normalize(feats, dim=2)  # normalization

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
            mask = torch.eq(labels, labels.T).float().to(device) # diagonal mask [bs, bs]
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
            torch.matmul(anchor_feature, contrast_feature.T), # anchor_feature: ([160, 512]),
            self.temperature)   #  numerator

        # for numerical stability -> 把matrix压一压防止过大出现nan，最终的Logits就是要的
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # 之前的mask没有aug的情况这一维度，这里把他加上去
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )  #
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # negative pair losses

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss