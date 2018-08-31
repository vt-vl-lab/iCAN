
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp
from ult.apply_prior import apply_prior

import cv2
import pickle
import numpy as np
import os
import sys
import glob
import time
import ipdb

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def get_blob(path, image_id):
    im_file  = path + image_id
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def im_detect(sess, net, path, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection):

    im_orig, im_shape = get_blob(path, image_id)
    
    blobs = {}
    blobs['H_num'] = 1
    
    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'): # This is a valid human
            
            # Predict actrion using human appearance only
            blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
            prediction_H  = net.test_image_H(sess, im_orig, blobs)

            # save image information
            dic = {}
            dic['image_id']   = image_id
            dic['person_box'] = Human_out[2]

            # Predict action using human and object appearance 
            Score_obj     = np.empty((0, 5 + 29), dtype=np.float32) 

            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])): # This is a valid object
                #if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])) and (Object[2][3] - Object[2][1]) *  (Object[2][2] - Object[2][0]) > 10000: # This is a valid object

                    blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
                    blobs['sp']      = Get_next_sp(Human_out[2], Object[2]).reshape(1, 64, 64, 2)
                    prediction_HO  = net.test_image_HO(sess, im_orig, blobs)

                    if prior_flag == 1:
                        prediction_HO  = apply_prior(Object, prediction_HO)
                    if prior_flag == 2:
                        prediction_HO  = prediction_HO * prior_mask[:,Object[4]].reshape(1,29)
                    if prior_flag == 3:
                        prediction_HO  = apply_prior(Object, prediction_HO)
                        prediction_HO  = prediction_HO * prior_mask[:,Object[4]].reshape(1,29)

                    This_Score_obj = np.concatenate((Object[2].reshape(1,4), np.array(Object[4]).reshape(1,1), prediction_HO[0] * np.max(Object[5])), axis=1)
                    Score_obj      = np.concatenate((Score_obj, This_Score_obj), axis=0)

            # There is only a single human detected in this image. I just ignore it. Might be better to add Nan as object box.
            if Score_obj.shape[0] == 0:
                continue

            # Find out the object box associated with highest action score
            max_idx = np.argmax(Score_obj,0)[5:]

            # agent mAP
            for i in range(29):
                #'''
                # walk, smile, run, stand

                agent_name      = Action_dic_inv[i] + '_agent'
                dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]

            # role mAP
            for i in range(29):
                # walk, smile, run, stand. Won't contribute to role mAP
                if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                    dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(Human_out[5]) * prediction_H[0][0][i]) 
                    continue

                # Impossible to perform this action
                if np.max(Human_out[5]) * Score_obj[max_idx[i]][5 + i] == 0:
                   dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(Human_out[5]) * Score_obj[max_idx[i]][5 + i])

                # Action with >0 score
                else:
                   dic[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:5], np.max(Human_out[5]) * Score_obj[max_idx[i]][5 + i])

            detection.append(dic)


def test_net(sess, net, Test_RCNN, prior_mask, Action_dic_inv, img_dir, output_dir, object_thres, human_thres, prior_flag):


    np.random.seed(cfg.RNG_SEED)
    detection = []
    count = 0

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}


    for im_file in glob.glob(img_dir + "*.png"):


        _t['im_detect'].tic()
 
        image_id   = im_file.split('/')[-1]
        
        im_detect(sess, net, img_dir, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection)

        _t['im_detect'].toc()

        print('im_detect: {:d} {:.3f}s'.format(count + 1, _t['im_detect'].average_time))
        count += 1

    pickle.dump( detection, open( output_dir, "wb" ) )




