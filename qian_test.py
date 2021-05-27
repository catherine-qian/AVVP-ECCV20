import scipy.io as sio
from nets.losses import *
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.get_backend() 无法画图可能是backend的问题


feats = sio.loadmat(os.getcwd()+'/data/qian/feats.mat')
x1, x2 = feats['x1'], feats['x2']


temp = 0.2
criterion = SupConLoss(temperature=temp)

# features: [bsz, n_views, f_dim]
# `n_views` is the number of crops from each image
# better be L2 normalized in f_dim dimension
features = torch.from_numpy(x1)
# labels: [bsz]
labels = torch.from_numpy(np.arange(features.shape[0]))

# SupContrast
features = F.normalize(features, dim=2) # normalization

loss = criterion(features, labels)
# or SimCLR
loss = criterion(features)
