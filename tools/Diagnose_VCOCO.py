# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao
# --------------------------------------------------------

"""
Error diagnose of V-COCO detection. 
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import numpy as np
import pickle
import json
import sys

from ult.config import cfg
from ult.vsrl_eval import VCOCOeval
from ult.vcoco_diagnose import VCOCOdiagnose

if __name__ == '__main__':
    if len(sys.argv) is not 3:
        print("Usage: python Diagnose_VCOCO.py eval_mode vcoco_detection")
    else:
        eval_mode       = sys.argv[1]
        vcoco_detection = sys.argv[2]

        if eval_mode == 'eval':
            print('Evaluating...')
            vcocoeval = VCOCOeval(cfg.DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json', cfg.DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json', cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids')
            vcocoeval._do_eval(vcoco_detection, ovr_thresh=0.5) 
        else:
            if eval_mode == 'diagnose':
                print('Diagnosing...')
                vcocoeval = VCOCOdiagnose(cfg.DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json', cfg.DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json', cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids')
                vcocoeval._do_eval(vcoco_detection, ovr_thresh=0.5)   
            else:
                print('Please specify the eval_mode')

