# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao, based on code from Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops

from ult.config import cfg
from ult.visualization import draw_bounding_boxes_HOI

import numpy as np
import ipdb

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
        weights_initializer = slim.variance_scaling_initializer(),
        biases_regularizer  = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), 
        biases_initializer  = tf.constant_initializer(0.0),
        trainable           = is_training,
        activation_fn       = tf.nn.relu,
        normalizer_fn       = slim.batch_norm,
        normalizer_params   = batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc



class ResNet50():
    def __init__(self):
        self.visualize = {}
        self.intermediate = {}
        self.predictions = {}
        self.score_summaries = {}
        self.event_summaries = {}
        self.train_summaries = []
        self.losses = {}

        self.image       = tf.placeholder(tf.float32, shape=[1, None, None, 3], name = 'image')
        self.spatial     = tf.placeholder(tf.float32, shape=[None, 64, 64, 2], name = 'sp')
        self.H_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'H_boxes')
        self.O_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'O_boxes')
        self.gt_class_HO = tf.placeholder(tf.float32, shape=[None, 600], name = 'gt_class_HO')
        self.H_num       = tf.placeholder(tf.int32)
        self.num_classes = 600
        self.num_fc      = 1024
        self.scope       = 'resnet_v1_50'
        self.stride      = [16, ]
        self.lr          = tf.placeholder(tf.float32)
        if tf.__version__ == '1.1.0':
            self.blocks     = [resnet_utils.Block('block1', resnet_v1.bottleneck,[(256,   64, 1)] * 2 + [(256,   64, 2)]),
                               resnet_utils.Block('block2', resnet_v1.bottleneck,[(512,  128, 1)] * 3 + [(512,  128, 2)]),
                               resnet_utils.Block('block3', resnet_v1.bottleneck,[(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
                               resnet_utils.Block('block4', resnet_v1.bottleneck,[(2048, 512, 1)] * 3),
                               resnet_utils.Block('block5', resnet_v1.bottleneck,[(2048, 512, 1)] * 3)]
        else:
            from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
            self.blocks = [resnet_v1_block('block1', base_depth=64,  num_units=3, stride=2),
                           resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                           resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                           resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                           resnet_v1_block('block5', base_depth=512, num_units=3, stride=1)]

    def build_base(self):
        with tf.variable_scope(self.scope, self.scope):
            net = resnet_utils.conv2d_same(self.image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

    def image_to_head(self, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net    = self.build_base()
            net, _ = resnet_v1.resnet_v1(net,
                                         self.blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                         global_pool=False,
                                         include_root_block=False,
                                         scope=self.scope)
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            head, _ = resnet_v1.resnet_v1(net,
                                          self.blocks[cfg.RESNET.FIXED_BLOCKS:-2],
                                          global_pool=False,
                                          include_root_block=False,
                                          scope=self.scope)
        return head


    def sp_to_head(self):
        with tf.variable_scope(self.scope, self.scope):
            conv1_sp      = slim.conv2d(self.spatial, 64, [5, 5], padding='VALID', scope='conv1_sp')
            pool1_sp      = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
            conv2_sp      = slim.conv2d(pool1_sp,     32, [5, 5], padding='VALID', scope='conv2_sp')
            pool2_sp      = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
            pool2_flat_sp = slim.flatten(pool2_sp)

        return pool2_flat_sp

    def res5(self, pool5_H, pool5_O, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

            fc7_H, _ = resnet_v1.resnet_v1(pool5_H,
                                           self.blocks[-2:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_H = tf.reduce_mean(fc7_H, axis=[1, 2])


            fc7_O, _ = resnet_v1.resnet_v1(pool5_O,
                                       self.blocks[-1:],
                                       global_pool=False,
                                       include_root_block=False,
                                       reuse=False,
                                       scope=self.scope)

            fc7_O = tf.reduce_mean(fc7_O, axis=[1, 2])
        
        return fc7_H, fc7_O

    def head_to_tail(self, fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])
            fc7_SO = tf.reduce_mean(pool5_SO, axis=[1, 2])

            Concat_SH     = tf.concat([fc7_H, fc7_SH], 1)
            fc8_SH        = slim.fully_connected(Concat_SH, self.num_fc, scope='fc8_SH')
            fc8_SH        = slim.dropout(fc8_SH, keep_prob=0.5, is_training=is_training, scope='dropout8_SH')
            fc9_SH        = slim.fully_connected(fc8_SH, self.num_fc, scope='fc9_SH')
            fc9_SH        = slim.dropout(fc9_SH,    keep_prob=0.5, is_training=is_training, scope='dropout9_SH')  

            Concat_SO     = tf.concat([fc7_O, fc7_SO], 1)
            fc8_SO        = slim.fully_connected(Concat_SO, self.num_fc, scope='fc8_SO')
            fc8_SO        = slim.dropout(fc8_SO, keep_prob=0.5, is_training=is_training, scope='dropout8_SO')
            fc9_SO        = slim.fully_connected(fc8_SO, self.num_fc, scope='fc9_SO')
            fc9_SO        = slim.dropout(fc9_SO,    keep_prob=0.5, is_training=is_training, scope='dropout9_SO')  

            Concat_SHsp   = tf.concat([fc7_H, sp], 1)
            Concat_SHsp   = slim.fully_connected(Concat_SHsp, self.num_fc, scope='Concat_SHsp')
            Concat_SHsp   = slim.dropout(Concat_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout6_SHsp')
            fc7_SHsp      = slim.fully_connected(Concat_SHsp, self.num_fc, scope='fc7_SHsp')
            fc7_SHsp      = slim.dropout(fc7_SHsp,  keep_prob=0.5, is_training=is_training, scope='dropout7_SHsp')

        return fc9_SH, fc9_SO, fc7_SHsp
        

    def crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:

            batch_ids    = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            bottom_shape = tf.shape(bottom)
            height       = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.stride[0])
            width        = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            bboxes        = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE], name="crops")
        return crops

    def attention_pool_layer_H(self, bottom, fc7_H, is_training, name):
        with tf.variable_scope(name) as scope:

            fc1         = slim.fully_connected(fc7_H, 512, scope='fc1_b')
            fc1         = slim.dropout(fc1, keep_prob=0.8, is_training=is_training, scope='dropout1_b')
            fc1         = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att         = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keep_dims=True)
        return att


    def attention_norm_H(self, att, name):
        with tf.variable_scope(name) as scope:

            att         = tf.transpose(att, [0, 3, 1, 2])
            att_shape   = tf.shape(att)
            att         = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att         = tf.nn.softmax(att)
            att         = tf.reshape(att, att_shape)
            att         = tf.transpose(att, [0, 2, 3, 1])
        return att

    def attention_pool_layer_O(self, bottom, fc7_O, is_training, name):
        with tf.variable_scope(name) as scope:

            fc1         = slim.fully_connected(fc7_O, 512, scope='fc1_b')
            fc1         = slim.dropout(fc1, keep_prob=0.8, is_training=is_training, scope='dropout1_b')
            fc1         = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att         = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keep_dims=True)
        return att


    def attention_norm_O(self, att, name):
        with tf.variable_scope(name) as scope:

            att         = tf.transpose(att, [0, 3, 1, 2])
            att_shape   = tf.shape(att)
            att         = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att         = tf.nn.softmax(att)
            att         = tf.reshape(att, att_shape)
            att         = tf.transpose(att, [0, 2, 3, 1])
        return att

    def region_classification(self, fc7_H, fc7_O, fc7_SHsp, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
 
            cls_score_H = slim.fully_connected(fc7_H, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_H')
            cls_prob_H  = tf.nn.sigmoid(cls_score_H, name='cls_prob_H') 
            tf.reshape(cls_prob_H, [1, self.num_classes]) 

            cls_score_O  = slim.fully_connected(fc7_O, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_O')
            cls_prob_O  = tf.nn.sigmoid(cls_score_O, name='cls_prob_O') 
            tf.reshape(cls_prob_O, [1, self.num_classes]) 

            cls_score_sp = slim.fully_connected(fc7_SHsp, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_sp')
            cls_prob_sp  = tf.nn.sigmoid(cls_score_sp, name='cls_prob_sp') 
            tf.reshape(cls_prob_sp, [1, self.num_classes]) 


            self.predictions["cls_score_H"]  = cls_score_H
            self.predictions["cls_prob_H"]   = cls_prob_H
            self.predictions["cls_score_O"]  = cls_score_O
            self.predictions["cls_prob_O"]   = cls_prob_O
            self.predictions["cls_score_sp"] = cls_score_sp
            self.predictions["cls_prob_sp"]  = cls_prob_sp

            self.predictions["cls_prob_HO"]  = cls_prob_sp * (cls_prob_O + cls_prob_H)

        return cls_prob_H, cls_prob_O, cls_prob_sp

    def bottleneck(self, bottom, is_training, name, reuse=False):
        with tf.variable_scope(name) as scope:

            if reuse:
                scope.reuse_variables()

            head_bottleneck = slim.conv2d(bottom, 1024, [1, 1], scope=name)

        return head_bottleneck

    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        # ResNet Backbone
        head       = self.image_to_head(is_training)
        sp         = self.sp_to_head()
        pool5_H    = self.crop_pool_layer(head, self.H_boxes, 'Crop_H')
        pool5_O    = self.crop_pool_layer(head, self.O_boxes[:self.H_num,:], 'Crop_O')

        fc7_H, fc7_O = self.res5(pool5_H, pool5_O, sp, is_training, 'res5')

        # Phi 
        head_phi = slim.conv2d(head, 512, [1, 1], scope='head_phi')

        # g 
        head_g   = slim.conv2d(head, 512, [1, 1], scope='head_g')

        Att_H      = self.attention_pool_layer_H(head_phi, fc7_H, is_training, 'Att_H')
        Att_H      = self.attention_norm_H(Att_H, 'Norm_Att_H')
        att_head_H = tf.multiply(head_g, Att_H)

        Att_O      = self.attention_pool_layer_O(head_phi, fc7_O, is_training, 'Att_O')
        Att_O      = self.attention_norm_O(Att_O, 'Norm_Att_O')
        att_head_O = tf.multiply(head_g, Att_O)

        pool5_SH     = self.bottleneck(att_head_H, is_training, 'bottleneck', False)
        pool5_SO     = self.bottleneck(att_head_O, is_training, 'bottleneck', True)


        fc7_SH, fc7_SO, fc7_SHsp = self.head_to_tail(fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, 'fc_HO')

        cls_prob_H, cls_prob_O, cls_prob_sp = self.region_classification(fc7_SH, fc7_SO, fc7_SHsp, is_training, initializer, 'classification')

        self.score_summaries.update(self.predictions)
        self.visualize["attention_map_H"] = (Att_H - tf.reduce_min(Att_H[0,:,:,:])) / tf.reduce_max((Att_H[0,:,:,:] - tf.reduce_min(Att_H[0,:,:,:])))
        self.visualize["attention_map_O"] = (Att_O - tf.reduce_min(Att_O[0,:,:,:])) / tf.reduce_max((Att_O[0,:,:,:] - tf.reduce_min(Att_O[0,:,:,:])))

        return cls_prob_H, cls_prob_O, cls_prob_sp



    def create_architecture(self, is_training):

        cls_prob_H, cls_prob_O, cls_prob_sp = self.build_network(is_training)

        for var in tf.trainable_variables():
            self.train_summaries.append(var)

        self.add_loss()
        layers_to_output = {}
        layers_to_output.update(self.losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            val_summaries.append(self.add_gt_image_summary_H())
            val_summaries.append(self.add_gt_image_summary_HO())
            tf.summary.image('ATTENTION_MAP_H',  self.visualize["attention_map_H"], max_outputs=1)
            tf.summary.image('ATTENTION_MAP_O',  self.visualize["attention_map_O"], max_outputs=1)
            for key, var in self.event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            #for key, var in self.score_summaries.items():
            #    self.add_score_summary(key, var)
            #for var in self.train_summaries:
            #    self.add_train_summary(var)
        
        val_summaries.append(tf.summary.scalar('lr', self.lr))
        self.summary_op     = tf.summary.merge_all()
        self.summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output


    def add_loss(self):

        with tf.variable_scope('LOSS') as scope:
            cls_score_H  = self.predictions["cls_score_H"]
            cls_score_O  = self.predictions["cls_score_O"]
            cls_score_sp = self.predictions["cls_score_sp"]
            label_HO     = self.gt_class_HO

            H_cross_entropy  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_HO[:self.H_num,:], logits = cls_score_H[:self.H_num,:]))
            O_cross_entropy  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_HO[:self.H_num,:], logits = cls_score_O[:self.H_num,:]))
            sp_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_HO, logits = cls_score_sp))

            self.losses['H_cross_entropy']  = H_cross_entropy
            self.losses['O_cross_entropy']  = O_cross_entropy
            self.losses['sp_cross_entropy'] = sp_cross_entropy

            loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy

            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)

        return loss


    def add_gt_image_summary_H(self):

        image = tf.py_func(draw_bounding_boxes_HOI, 
                      [tf.reverse(self.image+cfg.PIXEL_MEANS, axis=[-1]), self.H_boxes, self.gt_class_HO],
                      tf.float32, name="gt_boxes_H")
        return tf.summary.image('GROUND_TRUTH_H', image)

    def add_gt_image_summary_HO(self):

        image = tf.py_func(draw_bounding_boxes_HOI, 
                      [tf.reverse(self.image+cfg.PIXEL_MEANS, axis=[-1]), self.O_boxes, self.gt_class_HO],
                      tf.float32, name="gt_boxes_HO")
        return tf.summary.image('GROUND_TRUTH_HO)', image)



    def add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)


    def add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)



    def train_step(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'], self.gt_class_HO: blobs['gt_class_HO'],
                     self.spatial:blobs['sp'], self.lr: lr,
                     self.H_num: blobs['H_num']}
        
        loss, _ = sess.run([self.losses['total_loss'],
                            train_op],
                            feed_dict=feed_dict)
        return loss

    def train_step_with_summary(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'], self.gt_class_HO: blobs['gt_class_HO'],
                     self.spatial:blobs['sp'], self.lr: lr,
                     self.H_num: blobs['H_num']}

        loss, summary, _ = sess.run([self.losses['total_loss'],
                                     self.summary_op,
                                     train_op],
                                     feed_dict=feed_dict)
        return loss, summary

    
    def test_image_HO(self, sess, image, blobs):
        feed_dict = {self.image: image, self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'], self.spatial: blobs['sp'], self.H_num: blobs['H_num']}
        cls_prob_HO = sess.run([self.predictions["cls_prob_HO"]], feed_dict=feed_dict)

        return cls_prob_HO