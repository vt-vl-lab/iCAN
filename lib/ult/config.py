# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao, modified from Zheqi he and Xinlei Chen's
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import divisions
from __future__ import print_function

import os
import os.path as osp
import numpy as np

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()

cfg = __C

#
# Training options
#
__C.TRAIN = edict()

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.96

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = 20000

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = None

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10

# The step interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 200


# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 20000


# ResNet options
#
__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize. 
# if true, the region will be resized to a square of 2xPOOLING_SIZE, 
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1



#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'Data'))











# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default pooling mode, only 'crop' is available
__C.POOLING_MODE = 'crop'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8,16,32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5,1,2]

# Number of filters for the RPN layer
__C.RPN_CHANNELS = 512
