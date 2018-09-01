# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao
# --------------------------------------------------------

"""
Generating training instance
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import pickle
import random
from random import randint
import tensorflow as tf
import cv2
from config import cfg

def bbox_trans(human_box_ori, object_box_ori, ratio, size = 64):

    human_box  = human_box_ori.copy()
    object_box = object_box_ori.copy()
    
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]    

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    
    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'
        
    # shift the top-left corner to (0,0)
    
    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]    
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1] 

    if ratio == 'height': # height is larger than width
        
        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width  - 1 - human_box[2]) / height
        human_box[3] = (size - 1)                  - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width  - 1 - object_box[2]) / height
        object_box[3] = (size - 1)                  - size * (height - 1 - object_box[3]) / height
        
        # Need to shift horizontally  
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1



        shift = size / 2 - (InteractionPattern[2] + 1) / 2 
        human_box += [shift, 0 , shift, 0]
        object_box += [shift, 0 , shift, 0]
     
    else: # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1)                  - size * (width  - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width
        

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1)                  - size * (width  - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width
        
        # Need to shift vertically 
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        
        
        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)
        

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2 
        
        human_box = human_box + [0, shift, 0 , shift]
        object_box = object_box + [0, shift, 0 , shift]

 
    return np.round(human_box), np.round(object_box)



def Get_next_sp(human_box, object_box):
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O  = bbox_trans(human_box, object_box, 'width')
    
    Pattern = np.zeros((64,64,2))
    Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
    Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = 1


    return Pattern

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

def Augmented_box(bbox, shape, image_id, augment = 15):

    thres_ = 0.7

    box = np.array([0, bbox[0],  bbox[1],  bbox[2],  bbox[3]]).reshape(1,5)
    box = box.astype(np.float64)
        
    count = 0
    time_count = 0
    while count < augment:
        
        time_count += 1
        height = bbox[3] - bbox[1]
        width  = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen  = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10,10) * 0.01

        height_shift = randint(-np.floor(height),np.floor(height)) * 0.1
        width_shift  = randint(-np.floor(width),np.floor(width)) * 0.1

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)
        
        
        if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1,5)
            box  = np.concatenate((box,     box_),     axis=0)
            count += 1
        if time_count > 150:
            return box
            
    return box


def Generate_action(action_list):
    action_ = np.zeros(29)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,29)
    return action_

def Get_Next_Instance_HO_Neg(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H = Augmented_HO_Neg(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes_solo']= Human_augmented_solo
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H']  = action_H
    blobs['Mask_HO']     = mask_HO
    blobs['Mask_H']      = mask_H
    blobs['sp']          = Pattern
    blobs['H_num']       = len(action_H)

    return blobs

def Augmented_HO_Neg(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    

    action_HO_ = Generate_action(GT[1])
    action_H_  = Generate_action(GT[4])
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    Human_augmented_solo = Human_augmented.copy()



    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)
    

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)


    action_HO = action_HO_
    action_H  = action_H_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1,29)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( num_pos, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H




def Get_Next_Instance_HO_spNeg(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H = Augmented_HO_spNeg(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['Hsp_boxes']   = Human_augmented_sp
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_sp'] = action_sp
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H']  = action_H
    blobs['Mask_sp']     = mask_sp
    blobs['Mask_HO']     = mask_HO
    blobs['Mask_H']      = mask_H
    blobs['sp']          = Pattern
    blobs['H_num']       = len(action_H)

    return blobs

def Augmented_HO_spNeg(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_sp_ = Generate_action(GT[1])
    action_HO_ = Generate_action(GT[1])
    action_H_  = Generate_action(GT[4])
    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    action_sp = action_sp_
    action_HO = action_HO_
    action_H  = action_H_
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_sp = np.concatenate((action_sp, action_sp_), axis=0)
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1,29)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 29)
    action_HO         = action_HO.reshape(num_pos, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_sp           = mask_sp.reshape(num_pos_neg, 29)
    mask_HO           = mask_HO.reshape(  num_pos, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H




def Generate_action_HICO(action_list):
    action_ = np.zeros(600)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,600)
    return action_

def Get_Next_Instance_HO_Neg_HICO(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos = Augmented_HO_Neg_HICO(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos

    return blobs

def Augmented_HO_Neg_HICO(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_HO_ = Generate_action_HICO(GT[1])
    action_HO  = action_HO_

    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
    
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos


