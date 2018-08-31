
import sys
sys.path.insert(0, '/home/chengao/Dataset/v-coco/')

import numpy as np
import json
import pickle
import copy
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

trainval_GT    = pickle.load( open( "/home/chengao/Project/Network/data/trainval_GT_HO_H.pkl", "rb" ) )
COCO_dic       = pickle.load( open( "/home/chengao/Project/Network/data/trainval_COCO_GT_dic.pkl", "rb" ) )

Trainval_COCO_N = {}
for COCO_imgID, COCO_box in COCO_dic.iteritems():
    
    for idx_out, HO_out in enumerate(COCO_box): 

        id = HO_out[0]
        if (int(HO_out[3]) == 1): # loop every human box
            COCO_H = HO_out[2]
            gt_list_w_obj  = []
            
            # for every COCO human, try to find the matched GT human
            for ele_idx, ele in enumerate(trainval_GT):
                if ele[0] == id: # find the same image
                    if np.all(ele[2][:2] == COCO_H[:2]): # find the human GT
                            gt_list_w_obj.append(ele[3]) # GT with object 
    
            if gt_list_w_obj: # gt_list is not empty. This COCO_H has match

                for idx_in, HO_in in enumerate(COCO_box):
                    COCO_O = HO_in[2]
                    
                    if (not np.all(COCO_O == COCO_H)) and (not (any((COCO_O == x).all() for x in gt_list_w_obj))): # COCO_H COCO_O is Neg
                        tmp = []
                        tmp.append(id)            # id
                        tmp.append(np.zeros(29))  # action class
                        tmp.append(COCO_H)
                        tmp.append(COCO_O) 
                        tmp.append(np.nan)        # pose
                        tmp.append(int(HO_in[3])) # object class
                        tmp.append(np.array(1))   # Object class score
                        if COCO_imgID not in Trainval_COCO_N:
                            Trainval_COCO_N[COCO_imgID] = [tmp]
                        else:
                            Trainval_COCO_N[COCO_imgID].append(tmp)

            
            else: # This COCO_H has no match
                
                for idx_in, HO_in in enumerate(COCO_box):
                    COCO_O = HO_in[2]
                    
                    if (not np.all(COCO_O == COCO_H)): # COCO_H COCO_O is Neg
                        tmp = []
                        tmp.append(id)            # id
                        tmp.append(np.zeros(29))  # action class
                        tmp.append(COCO_H)
                        tmp.append(COCO_O) 
                        tmp.append(np.nan)        # pose
                        tmp.append(int(HO_in[3])) # object class
                        tmp.append(np.array(1))   # Object class score
                        if COCO_imgID not in Trainval_COCO_N:
                            Trainval_COCO_N[COCO_imgID] = [tmp]
                        else:
                            Trainval_COCO_N[COCO_imgID].append(tmp)

                            
pickle.dump( Trainval_COCO_N, open( "/home/chengao/Project/Network/data/trainval_COCO_N_dic", "wb" ) )