import numpy as np
import json
from pprint import pprint
import xml.etree.ElementTree as ET
import re
import scipy.io as sio
import pickle
import ipdb

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


HICO = sio.loadmat('/home/chengao/Data/hico_20160224_det/anno_bbox.mat')

HICO_Train_GT = {}


for instance in HICO['bbox_train'][0]:
    name = int(instance[0][0][-9:-4].lstrip('0'))
    image_temp = []
    print (name)
    for HOI in instance[2][0]:
        if HOI[4][0][0] == 0 and HOI[1].shape != (0, 0): # vis
            HOI_id = HOI[0][0][0]
            for connect in HOI[3]: # connection
                image_instance = []
                image_instance.append(name) # image name
                image_instance.append(HOI_id) # action
                Human  = [HOI[1][0][connect[0] - 1][it][0][0] - 1 for it in [0,2,1,3]]
                Object = [HOI[2][0][connect[1] - 1][it][0][0] - 1 for it in [0,2,1,3]]
                image_instance.append(Human) # Human box
                image_instance.append(Object) # Object box
                image_temp.append(image_instance)
    HICO_Train_GT[name] = image_temp


dic = {}
for key, value in HICO_Train_GT.iteritems():
    for ele_idx, ele in enumerate(value):
        if ele_idx == 0:
            tmp = []
            tmp.append(ele[0])
            tmp.append([ele[1]-1])
            tmp.append(np.array(ele[2]))
            tmp.append(np.array(ele[3]))
            dic[ele[0]] = [tmp]
        else:
            UnknownFlag = 1
            for _, ele_in in enumerate(dic[ele[0]]):
                if bb_IOU(ele_in[2], np.array(ele[2])) > 0.25 and bb_IOU(ele_in[3], np.array(ele[3])) > 0.25:
                    ele_in[1].append(ele[1]-1)
                    UnknownFlag = 0
                    break
                    
            if (UnknownFlag == 1):
                tmp = []
                tmp.append(ele[0])
                tmp.append([ele[1]-1])
                tmp.append(np.array(ele[2]))
                tmp.append(np.array(ele[3]))
                dic[ele[0]].append(tmp)


pickle.dump( dic, open( "/home/chengao/Project/Network/data/trainval_GT_HO_dic_HICO.pkl", "wb" ) )


trainval_GT = []

for key, value in dic.iteritems():
    for ele in value:
        trainval_GT.append(ele)

pickle.dump( trainval_GT, open( "/home/chengao/Project/Network/data/trainval_GT_HO_HICO.pkl", "wb" ) )


# Generate mask

HICO = sio.loadmat('/home/chengao/Dataset/hico_20160224_det/anno.mat')
Mask = HICO['anno_train']


count = 0
image_id_2_mask = {}
for instance in HICO['list_train']:
    
    temp_mask_list = np.argwhere(~np.isnan(Mask[:,count]))
    mask_list = []
    for ele in temp_mask_list:
        mask_list.append(ele[0])
    image_id_2_mask[int(instance[0][0][-9:-4].lstrip('0'))] = mask_list
    
    count += 1
for ele in trainval_GT:
    ele.append(image_id_2_mask[ele[0]])
    
pickle.dump( trainval_GT, open( "/home/chengao/Project/Network/data/trainval_GT_HO_HICO.pkl", "wb" ) )   
