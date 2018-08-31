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

test_GT = []

count = 0

for ind in range(len(vcoco_all[0]['role_bbox'])):
    print (count)
    count += 1
    image_id  = vcoco_all[0]['image_id'][ind]
    human     = vcoco_all[0]['role_bbox'][ind][:4]

    
    tmp = []
    tmp.append(image_id)
    tmp.append('Human')
    tmp.append(human)
    tmp.append([])
    tmp.append(1)
    tmp.append([])
    test_GT.append(tmp)
    
    obj_set = set()
    
    
    for action in range(26):
    
        if (action == 1) or (action == 4) or (action == 15) or (action == 20):
            continue

        role_bbox = vcoco_all[action]['role_bbox'][ind]
        object1   = role_bbox[4:8]
        find_obj1 = 0
        
        
        if np.isnan(object1).all() == False:
            
            if tuple(object1) not in obj_set:
            
                obj_set.add(tuple(object1))

                for coco_idx, coco_ele in enumerate(coco_anno['annotations']):
                    if coco_ele['image_id'] == image_id:
                        if np.all(xyhw_to_xyxy(coco_ele['bbox']) == object1):
                            cato_id = coco_ele['category_id']
                            find_obj1 = 1
                            break

 
                if find_obj1 == 1:
                    tmp = []
                    tmp.append(image_id)
                    tmp.append('Object')
                    tmp.append(np.round(object1,2))
                    tmp.append(np.nan)
                    tmp.append(obj_index[cato_id] + 1)
                    tmp.append([])
                    test_GT.append(tmp)
                else:
                    print ('Error')

        if (action == 6) or (action == 7) or (action == 14):
            
            object2   = role_bbox[8:12]
            find_obj2 = 0
            
            if np.isnan(object2).all() == False:
                
                if tuple(object2) not in obj_set:
            
                    obj_set.add(tuple(object2))

                    for coco_idx, coco_ele in enumerate(coco_anno['annotations']):
                        if coco_ele['image_id'] == image_id:
                            if np.all(xyhw_to_xyxy(coco_ele['bbox']) == object2):
                                cato_id = coco_ele['category_id']
                                find_obj1 = 1
                                break


                    if find_obj1 == 1:
                        tmp = []
                        tmp.append(image_id)
                        tmp.append('Object')
                        tmp.append(np.round(object2,2))
                        tmp.append(np.nan)
                        tmp.append(obj_index[cato_id] + 1)
                        tmp.append([])
                        test_GT.append(tmp)

                    else:
                        print ('Error')


pickle.dump(test_GT, open( "/home/chengao/Project/Network/data/test_GT.pkl", "wb" ) )  


test_GT_dic = {}

for GT in test_GT:
    if GT[0][0] not in test_GT_dic:
        tmp = []
        tmp.append(GT[0][0])
        tmp.append(GT[1])
        tmp.append(GT[2])
        tmp.append(GT[3])
        tmp.append(GT[4])
        tmp.append(np.array(1))
        test_GT_dic[GT[0][0]] = [tmp]
        
    else:
        tmp = []
        tmp.append(GT[0][0])
        tmp.append(GT[1])
        tmp.append(GT[2])
        tmp.append(GT[3])
        tmp.append(GT[4])
        tmp.append(np.array(1))
        test_GT_dic[GT[0][0]].append(tmp)
        
    
pickle.dump( test_GT_dic, open( "/home/chengao/Project/Network/data/test_GT_dic.pkl", "wb" ) )


