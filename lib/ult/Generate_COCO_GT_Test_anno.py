# Generate VCOCO GT testing set object bounding box.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '/home/chengao/Dataset/v-coco/')

import copy
import numpy as np
import json
import vsrl_utils as vu
import pickle
import vsrl_utils as vu
import scipy.ndimage as ndimage
import ipdb

obj_index      = pickle.load(open('/home/chengao/Project/Network/data/object_index.pkl', "rb" ) )
coco_anno      = json.load(open('/home/chengao/Dataset/v-coco/coco/annotations/instances_val2014.json'))


def xyhw_to_xyxy(bbox):
    out = copy.deepcopy(bbox)
    out[2] = bbox[0] + bbox[2];
    out[3] = bbox[1] + bbox[3];
    return out

def bb_IOU(boxA, boxB):

    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps



coco = vu.load_coco()
vcoco_all = vu.load_vcoco('vcoco_test')
for x in vcoco_all:
    x = vu.attach_gt_boxes(x, coco)
    
classes = [x['action_name'] for x in vcoco_all]
np.random.seed(1)

test_list = []
for line in open('/home/chengao/Dataset/v-coco/data/splits/vcoco_test.ids', 'r'):

    image_id   = int(line.rstrip())
    test_list.append(image_id)
    
    
test_GT_dic = {}

for GT in coco_anno['annotations']:
    if GT['image_id'] in test_list:
        if GT['image_id'] not in test_GT_dic:
            tmp = []
            tmp.append(GT['image_id'])
            if GT['category_id'] == 1:
                tmp.append('Human')
            else:
                tmp.append('Object')
            tmp.append(np.array(np.round(xyhw_to_xyxy(GT['bbox']),2)))
            tmp.append(np.nan)
            tmp.append(obj_index[GT['category_id']] + 1)
            tmp.append(np.array(1))
            test_GT_dic[GT['image_id']] = [tmp]

        else:
            tmp = []
            tmp.append(GT['image_id'])
            if GT['category_id'] == 1:
                tmp.append('Human')
            else:
                tmp.append('Object')
            tmp.append(np.array(np.round(xyhw_to_xyxy(GT['bbox']),2)))
            tmp.append(np.nan)
            tmp.append(obj_index[GT['category_id']] + 1)
            tmp.append(np.array(1))
            test_GT_dic[GT['image_id']].append(tmp)