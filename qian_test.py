import scipy.io as sio
from nets.losses import *
import os

feats = sio.loadmat('data/qian/feats.mat')
x1, x2 = feats['x1'], feats['x2']

temp = 0.2
criterion = SupConLoss(temperature=temp)

# features: [bsz, n_views, f_dim]
# `n_views` is the number of crops from each image
# better be L2 normalized in f_dim dimension
features = ...
# labels: [bsz]
labels = ...

# SupContrast
loss = criterion(features, labels)
# or SimCLR
loss = criterion(features)
