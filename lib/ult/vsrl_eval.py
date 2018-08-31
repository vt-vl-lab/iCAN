# AUTORIGHTS
# ---------------------------------------------------------
# Copyright (c) 2017, Saurabh Gupta 
# 
# This file is part of the VCOCO dataset hooks and is available 
# under the terms of the Simplified BSD License provided in 
# LICENSE. Please retain this notice and LICENSE if you use 
# this file (or any portion of it) in your project.
# ---------------------------------------------------------

# vsrl_data is a dictionary for each action class:
# image_id       - Nx1
# ann_id         - Nx1
# label          - Nx1
# action_name    - string
# role_name      - ['agent', 'obj', 'instr']
# role_object_id - N x K matrix, obviously [:,0] is same as ann_id

import numpy as np
from pycocotools.coco import COCO
import os, json
import copy
import pickle
import ipdb

class VCOCOeval(object):

  def __init__(self, vsrl_annot_file, coco_annot_file, 
      split_file):
    """Input:
    vslr_annot_file: path to the vcoco annotations
    coco_annot_file: path to the coco annotations
    split_file: image ids for split
    """
    self.COCO = COCO(coco_annot_file)
    self.VCOCO = _load_vcoco(vsrl_annot_file)
    self.image_ids = np.loadtxt(open(split_file, 'r'))
    # simple check  

    assert np.all(np.equal(np.sort(np.unique(self.VCOCO[0]['image_id'])), np.sort(self.image_ids)))

    self._init_coco()
    self._init_vcoco()


  def _init_vcoco(self):
    actions = [x['action_name'] for x in self.VCOCO]
    roles = [x['role_name'] for x in self.VCOCO]
    self.actions = actions
    self.actions_to_id_map = {v: i for i, v in enumerate(self.actions)}
    self.num_actions = len(self.actions)
    self.roles = roles   


  def _init_coco(self):
    category_ids = self.COCO.getCatIds()
    categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
    self.category_to_id_map = dict(zip(categories, category_ids))
    self.classes = ['__background__'] + categories
    self.num_classes = len(self.classes)
    self.json_category_id_to_contiguous_id = {
        v: i + 1 for i, v in enumerate(self.COCO.getCatIds())}
    self.contiguous_category_id_to_json_id = {
        v: k for k, v in self.json_category_id_to_contiguous_id.items()}


  def _get_vcocodb(self):
    vcocodb = copy.deepcopy(self.COCO.loadImgs(self.image_ids.tolist()))
    for entry in vcocodb:
      self._prep_vcocodb_entry(entry)
      self._add_gt_annotations(entry)

    # print
    if 0:
      nums = np.zeros((self.num_actions), dtype=np.int32)
      for entry in vcocodb:
        for aid in range(self.num_actions):
          nums[aid] += np.sum(np.logical_and(entry['gt_actions'][:, aid]==1, entry['gt_classes']==1))
      for aid in range(self.num_actions):
        print('Action %s = %d'%(self.actions[aid], nums[aid]))

    return vcocodb


  def _prep_vcocodb_entry(self, entry):
    entry['boxes'] = np.empty((0, 4), dtype=np.float32)
    entry['is_crowd'] = np.empty((0), dtype=np.bool)
    entry['gt_classes'] = np.empty((0), dtype=np.int32)
    entry['gt_actions'] = np.empty((0, self.num_actions), dtype=np.int32)
    entry['gt_role_id'] = np.empty((0, self.num_actions, 2), dtype=np.int32)


  def _add_gt_annotations(self, entry):
    ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
    objs = self.COCO.loadAnns(ann_ids)
    # Sanitize bboxes -- some are invalid
    valid_objs = []
    valid_ann_ids = []
    width = entry['width']
    height = entry['height']
    for i, obj in enumerate(objs):
      if 'ignore' in obj and obj['ignore'] == 1:
          continue
      # Convert form x1, y1, w, h to x1, y1, x2, y2
      x1 = obj['bbox'][0]
      y1 = obj['bbox'][1]
      x2 = x1 + np.maximum(0., obj['bbox'][2] - 1.)
      y2 = y1 + np.maximum(0., obj['bbox'][3] - 1.)
      x1, y1, x2, y2 = clip_xyxy_to_image(
          x1, y1, x2, y2, height, width)
      # Require non-zero seg area and more than 1x1 box size
      if obj['area'] > 0 and x2 > x1 and y2 > y1:
        obj['clean_bbox'] = [x1, y1, x2, y2]
        valid_objs.append(obj)
        valid_ann_ids.append(ann_ids[i])
    num_valid_objs = len(valid_objs)
    assert num_valid_objs == len(valid_ann_ids)

    boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
    is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
    gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
    gt_actions = -np.ones((num_valid_objs, self.num_actions), dtype=entry['gt_actions'].dtype)
    gt_role_id = -np.ones((num_valid_objs, self.num_actions, 2), dtype=entry['gt_role_id'].dtype)

    for ix, obj in enumerate(valid_objs):
      cls = self.json_category_id_to_contiguous_id[obj['category_id']]
      boxes[ix, :] = obj['clean_bbox']
      gt_classes[ix] = cls
      is_crowd[ix] = obj['iscrowd']
      
      gt_actions[ix, :], gt_role_id[ix, :, :] = \
        self._get_vsrl_data(valid_ann_ids[ix],
            valid_ann_ids, valid_objs)

    entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
    entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
    entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
    entry['gt_actions'] = np.append(entry['gt_actions'], gt_actions, axis=0)
    entry['gt_role_id'] = np.append(entry['gt_role_id'], gt_role_id, axis=0)
     

  def _get_vsrl_data(self, ann_id, ann_ids, objs):
    """ Get VSRL data for ann_id."""
    action_id = -np.ones((self.num_actions), dtype=np.int32)
    role_id = -np.ones((self.num_actions, 2), dtype=np.int32)
    # check if ann_id in vcoco annotations
    in_vcoco = np.where(self.VCOCO[0]['ann_id'] == ann_id)[0]
    if in_vcoco.size > 0:
      action_id[:] = 0
      role_id[:] = -1
    else:
      return action_id, role_id
    for i, x in enumerate(self.VCOCO):
      assert x['action_name'] == self.actions[i]
      has_label = np.where(np.logical_and(x['ann_id'] == ann_id, x['label'] == 1))[0]
      if has_label.size > 0:
        action_id[i] = 1
        assert has_label.size == 1
        rids = x['role_object_id'][has_label]
        assert rids[0, 0] == ann_id
        for j in range(1, rids.shape[1]):
          if rids[0, j] == 0:
            # no role
            continue
          aid = np.where(ann_ids == rids[0, j])[0]
          assert aid.size > 0
          role_id[i, j - 1] = aid
    return action_id, role_id


  def _collect_detections_for_image(self, dets, image_id):

    agents = np.empty((0, 4 + self.num_actions), dtype=np.float32) # 4 + 26 = 30
    roles = np.empty((0, 5 * self.num_actions, 2), dtype=np.float32) # (5 * 26), 2
    for det in dets: # loop all detection instance
      if det['image_id'] == image_id:# might be several
        this_agent = np.zeros((1, 4 + self.num_actions), dtype=np.float32)
        this_role  = np.zeros((1, 5 * self.num_actions, 2), dtype=np.float32)
        this_agent[0, :4] = det['person_box']
        for aid in range(self.num_actions): # loop 26 actions
          for j, rid in enumerate(self.roles[aid]):
            if rid == 'agent':
                #if aid == 10:
                #  this_agent[0, 4 + aid] = det['talk_' + rid]
                #if aid == 16:
                #  this_agent[0, 4 + aid] = det['work_' + rid]
                #if (aid != 10) and (aid != 16):
                this_agent[0, 4 + aid] = det[self.actions[aid] + '_' + rid]
            else:
                this_role[0, 5 * aid: 5 * aid + 5, j-1] = det[self.actions[aid] + '_' + rid]
        agents = np.concatenate((agents, this_agent), axis=0)
        roles  = np.concatenate((roles, this_role), axis=0)
    return agents, roles


  def _do_eval(self, detections_file, ovr_thresh=0.5):

    vcocodb = self._get_vcocodb()
    self._do_agent_eval(vcocodb, detections_file, ovr_thresh=ovr_thresh)
    self._do_role_eval(vcocodb, detections_file, ovr_thresh=ovr_thresh, eval_type='scenario_1')
    self._do_role_eval(vcocodb, detections_file, ovr_thresh=ovr_thresh, eval_type='scenario_2')

  
  def _do_role_eval(self, vcocodb, detections_file, ovr_thresh=0.5, eval_type='scenario_1'):

    with open(detections_file, 'rb') as f:
      dets = pickle.load(f)
    
    tp = [[[] for r in range(2)] for a in range(self.num_actions)]
    fp = [[[] for r in range(2)] for a in range(self.num_actions)]
    sc = [[[] for r in range(2)] for a in range(self.num_actions)]

    npos = np.zeros((self.num_actions), dtype=np.float32)
   
    
    for i in range(len(vcocodb)):
      image_id = vcocodb[i]['id']
      gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]
      # person boxes
      gt_boxes = vcocodb[i]['boxes'][gt_inds]
      gt_actions = vcocodb[i]['gt_actions'][gt_inds]
      # some peorson instances don't have annotated actions
      # we ignore those instances
      ignore = np.any(gt_actions == -1, axis=1)
      assert np.all(gt_actions[np.where(ignore==True)[0]]==-1)

      for aid in range(self.num_actions):
        npos[aid] += np.sum(gt_actions[:, aid] == 1)

      pred_agents, pred_roles = self._collect_detections_for_image(dets, image_id)

      for aid in range(self.num_actions):
        if len(self.roles[aid])<2:
          # if action has no role, then no role AP computed
          continue

        for rid in range(len(self.roles[aid])-1): 

          # keep track of detected instances for each action for each role
          covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)

          # get gt roles for action and role
          gt_role_inds = vcocodb[i]['gt_role_id'][gt_inds, aid, rid]
          gt_roles = -np.ones_like(gt_boxes)
          for j in range(gt_boxes.shape[0]):
            if gt_role_inds[j] > -1:
              gt_roles[j] = vcocodb[i]['boxes'][gt_role_inds[j]]

          agent_boxes = pred_agents[:, :4]
          role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4, rid]
          agent_scores = pred_roles[:, 5 * aid + 4, rid]

          valid = np.where(np.isnan(agent_scores) == False)[0]
          #valid = np.where(agent_scores != 0)[0]


          agent_scores = agent_scores[valid]
          agent_boxes = agent_boxes[valid, :]
          role_boxes = role_boxes[valid, :]

          idx = agent_scores.argsort()[::-1]

          for j in idx:
            pred_box = agent_boxes[j, :]
            overlaps = get_overlap(gt_boxes, pred_box)

            # matching happens based on the person 
            jmax = overlaps.argmax()
            ovmax = overlaps.max()

            # if matched with an instance with no annotations
            # continue
            if ignore[jmax]:
              continue

            # overlap between predicted role and gt role
            if np.all(gt_roles[jmax, :] == -1): # if no gt role
              if eval_type == 'scenario_1':
                if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                  # if no role is predicted, mark it as correct role overlap
                  ov_role = 1.0
                else:
                  # if a role is predicted, mark it as false 
                  ov_role = 0.0
              elif eval_type == 'scenario_2':
                # if no gt role, role prediction is always correct, irrespective of the actual predition
                ov_role = 1.0   
              else:
                raise ValueError('Unknown eval type')    
            else:
              ov_role = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])

            is_true_action = (gt_actions[jmax, aid] == 1)
            sc[aid][rid].append(agent_scores[j])
            if is_true_action and (ovmax>=ovr_thresh) and (ov_role>=ovr_thresh):
              if covered[jmax]:
                fp[aid][rid].append(1)
                tp[aid][rid].append(0)
              else:
                fp[aid][rid].append(0)
                tp[aid][rid].append(1)
                covered[jmax] = True
            else:
              fp[aid][rid].append(1)
              tp[aid][rid].append(0)

    # compute ap for each action
    role_ap = np.zeros((self.num_actions, 2), dtype=np.float32)
    role_ap[:] = np.nan
    for aid in range(self.num_actions):
      if len(self.roles[aid])<2:
        continue
      for rid in range(len(self.roles[aid])-1):
        a_fp = np.array(fp[aid][rid], dtype=np.float32)
        a_tp = np.array(tp[aid][rid], dtype=np.float32)
        a_sc = np.array(sc[aid][rid], dtype=np.float32)
        # sort in descending score order
        idx = a_sc.argsort()[::-1]
        a_fp = a_fp[idx]
        a_tp = a_tp[idx]
        a_sc = a_sc[idx]

        a_fp = np.cumsum(a_fp)
        a_tp = np.cumsum(a_tp)
        rec = a_tp / float(npos[aid])
        #check
        assert(np.amax(rec) <= 1)
        prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
        role_ap[aid, rid] = voc_ap(rec, prec)

    print('---------Reporting Role AP (%)------------------')
    for aid in range(self.num_actions):
      if len(self.roles[aid])<2: continue
      for rid in range(len(self.roles[aid])-1):
        print('{: >23}: AP = {:0.2f} (#pos = {:d})'.format(self.actions[aid]+'-'+self.roles[aid][rid+1], role_ap[aid, rid]*100.0, int(npos[aid])))
  
    print('Average Role [%s] AP = %.2f'%(eval_type, np.nanmean(role_ap) * 100.00))  
    print('---------------------------------------------') 
    print('Average Role [%s] AP = %.2f, omitting the action "point"'%(eval_type, (np.nanmean(role_ap) * 25 - role_ap[-3][0]) / 24 * 100.00))  
    print('---------------------------------------------') 

  def _do_agent_eval(self, vcocodb, detections_file, ovr_thresh=0.5):

    with open(detections_file, 'rb') as f:
      dets = pickle.load(f)

    tp = [[] for a in range(self.num_actions)]
    fp = [[] for a in range(self.num_actions)]
    sc = [[] for a in range(self.num_actions)]

    npos = np.zeros((self.num_actions), dtype=np.float32)
    
    for i in range(len(vcocodb)):
      image_id = vcocodb[i]['id']# img ID, not the full name (e.g. id= 165, 'file_name' = COCO_train2014_000000000165.jpg )
      gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]# index of the person's box among all object boxes
      # person boxes
      gt_boxes = vcocodb[i]['boxes'][gt_inds] # all person's boxes in this image
      gt_actions = vcocodb[i]['gt_actions'][gt_inds] # index of Nx26 binary matrix indicating the actions
      # some peorson instances don't have annotated actions
      # we ignore those instances
      ignore = np.any(gt_actions == -1, axis=1)

      for aid in range(self.num_actions):
        npos[aid] += np.sum(gt_actions[:, aid] == 1)# how many actions are involved in this image(for all the human)

      pred_agents, _ = self._collect_detections_for_image(dets, image_id)
      # For each image, we have a pred_agents. For example, there are 2 people detected, then pred_agents is a 2x(4+26) matrix. Each row stands for a human, 0-3 human box, 4-25 the score for each action.
        
      for aid in range(self.num_actions):

        # keep track of detected instances for each action
        covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)# gt_boxes.shape[0] is the number of people in this image

        agent_scores = pred_agents[:, 4 + aid]# score of this action for all people in this image
        agent_boxes = pred_agents[:, :4] # predicted buman box for all people in this image
        # remove NaNs
        # If only use agent, there should be no NAN cause there is no object information provided. Just give a agent score.
        valid = np.where(np.isnan(agent_scores) == False)[0]
        agent_scores = agent_scores[valid] 
        agent_boxes = agent_boxes[valid, :]

        # sort in descending order
        idx = agent_scores.argsort()[::-1]# For this action, sort score of all people. A action cam be done by many people.
    
        for j in idx: # Each predicted person
          pred_box = agent_boxes[j, :]# It's predicted human box
          overlaps = get_overlap(gt_boxes, pred_box)# overlap between this predict human and all human gt_boxes

          jmax = overlaps.argmax()# Find the idx of gt human box that matches this predicted human
          ovmax = overlaps.max()

          # if matched with an instance with no annotations
          # continue
          if ignore[jmax]:
            continue
           
          is_true_action = (gt_actions[jmax, aid] == 1)# Is this person actually doing this action according to gt?

          sc[aid].append(agent_scores[j]) # The predicted score of this person doing this action. In descending order.
          if is_true_action and (ovmax>=ovr_thresh): # bounding box IOU is larger than 0.5 and this this person is doing this action.
            if covered[jmax]:
              fp[aid].append(1)
              tp[aid].append(0)
            else:# first time see this gt human
              fp[aid].append(0)
              tp[aid].append(1)
              covered[jmax] = True
          else:
            fp[aid].append(1)
            tp[aid].append(0)

    # compute ap for each action
    agent_ap = np.zeros((self.num_actions), dtype=np.float32)
    for aid in range(self.num_actions):

      a_fp = np.array(fp[aid], dtype=np.float32)
      a_tp = np.array(tp[aid], dtype=np.float32)
      a_sc = np.array(sc[aid], dtype=np.float32)
      # sort in descending score order
      idx = a_sc.argsort()[::-1]# For each action, sort the score of all predicted people in all images
      a_fp = a_fp[idx]
      a_tp = a_tp[idx]
      a_sc = a_sc[idx]

      a_fp = np.cumsum(a_fp)
      a_tp = np.cumsum(a_tp)
      rec = a_tp / float(npos[aid])
      #check
      
      assert(np.amax(rec) <= 1)
      prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
      agent_ap[aid] = voc_ap(rec, prec)

    print('---------Reporting Agent AP (%)------------------')
    for aid in range(self.num_actions):
      print('{: >20}: AP = {:0.2f} (#pos = {:d})'.format(self.actions[aid], agent_ap[aid]*100.0, int(npos[aid])))
    print('Average Agent AP = %.2f'%(np.nansum(agent_ap) * 100.00/self.num_actions))
    print('---------------------------------------------')

def _load_vcoco(vcoco_file):
  print('loading vcoco annotations...')
  with open(vcoco_file, 'r') as f:
    vsrl_data = json.load(f)
  for i in range(len(vsrl_data)):
    vsrl_data[i]['role_object_id'] = \
    np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']), -1)).T
    for j in ['ann_id', 'label', 'image_id']:
        vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1, 1))
  return vsrl_data


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
  x1 = np.minimum(width - 1., np.maximum(0., x1))
  y1 = np.minimum(height - 1., np.maximum(0., y1))
  x2 = np.minimum(width - 1., np.maximum(0., x2))
  y2 = np.minimum(height - 1., np.maximum(0., y2))
  return x1, y1, x2, y2


def get_overlap(boxes, ref_box):
  ixmin = np.maximum(boxes[:, 0], ref_box[0])
  iymin = np.maximum(boxes[:, 1], ref_box[1])
  ixmax = np.minimum(boxes[:, 2], ref_box[2])
  iymax = np.minimum(boxes[:, 3], ref_box[3])
  iw = np.maximum(ixmax - ixmin + 1., 0.)
  ih = np.maximum(iymax - iymin + 1., 0.)
  inters = iw * ih

  # union
  uni = ((ref_box[2] - ref_box[0] + 1.) * (ref_box[3] - ref_box[1] + 1.) +
         (boxes[:, 2] - boxes[:, 0] + 1.) *
         (boxes[:, 3] - boxes[:, 1] + 1.) - inters)

  overlaps = inters / uni
  return overlaps


def voc_ap(rec, prec):
  """ ap = voc_ap(rec, prec)
  Compute VOC AP given precision and recall.
  [as defined in PASCAL VOC]
  """
  # correct AP calculation
  # first append sentinel values at the end
  mrec = np.concatenate(([0.], rec, [1.]))
  mpre = np.concatenate(([0.], prec, [0.]))

  # compute the precision envelope
  for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

  # to calculate area under PR curve, look for points
  # where X axis (recall) changes value
  i = np.where(mrec[1:] != mrec[:-1])[0]

  # and sum (\Delta recall) * prec
  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

