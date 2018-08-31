from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import ipdb
import os
import os.path as osp

from ult.config import cfg
from models.test_demo import test_net

def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on VCOCO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=300000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='iCAN_ResNet50_VCOCO', type=str)
    parser.add_argument('--prior_flag', dest='prior_flag',
            help='whether use prior_flag',
            default=3, type=int)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.4, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.8, type=float)
    parser.add_argument('--img_dir', dest='img_dir',
            help='Please specify the img folder',
            default='/', type=str)
    parser.add_argument('--Demo_RCNN', dest='Demo_RCNN',
            help='The object detection .pkl file',
            default='/', type=str)
    parser.add_argument('--HOI_Detection', dest='HOI_Detection',
            help='Where to save the final HOI_Detection',
            default='/', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    
    prior_mask     = pickle.load( open( cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb" ) )
    Action_dic     = json.load(   open( cfg.DATA_DIR + '/' + 'action_index.json'))
    Action_dic_inv = {y:x for x,y in Action_dic.iteritems()}
    Demo_RCNN      = pickle.load( open( args.Demo_RCNN, "rb" ) )


    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'
    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight ) 
  
    
    # init session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)


    if args.model == 'iCAN_ResNet50_VCOCO':
        from networks.iCAN_ResNet50_VCOCO import ResNet50
    if args.model == 'iCAN_ResNet50_VCOCO_Early':
        from networks.iCAN_ResNet50_VCOCO_Early import ResNet50
        
    net = ResNet50()
    net.create_architecture(False)
    
    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    
    test_net(sess, net, Demo_RCNN, prior_mask, Action_dic_inv, args.img_dir, args.HOI_Detection, args.object_thres, args.human_thres, args.prior_flag)
    sess.close()
