# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Geonu Lee (lkw3139@gachon.ac.kr)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


# par_resnet related params
PAR_RESNET = CN()
PAR_RESNET.NUM_LAYERS = 50
PAR_RESNET.PRETRAINED_LAYERS = ['*']

MODEL_EXTRAS = {
    'par_resnet': PAR_RESNET,

}
