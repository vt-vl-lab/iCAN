import numpy as np
import json
import vsrl_utils as vu
from pprint import pprint
import xml.etree.ElementTree as ET
import re
import pickle

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

def getOpenPose(image_id, human_box):
    tree = ET.parse('data/vcoco_train_keypoint/COCO_train2014_' + str(image_id).zfill(12) + '_pose.xml')
    root = tree.getroot()
    data = re.split('\s+', root[0][2].text)
    data.pop(0)
    if len(data) <= 0:
        return np.full(36, 0)# if no pose detected, return 0
    else:
        num_person = len(data) / 54
        pose_IOU = -1
        for per in range(num_person):
            xs = []
            ys = []
            coor = []
            for j in range(18):
                xx = float(data[54*per + 3*j])
                if xx > 0.001:#delete those undetected points. They will destroy IOU
                    xs.append(xx)
                yy = float(data[54*per + 3*j + 1])
                if yy > 0.001:
                    ys.append(yy)
                coor.append(xx)
                coor.append(yy)
            
            if bb_IOU([min(xs), min(ys),max(xs), max(ys)],human_box) > pose_IOU:# better one
                pose_IOU = bb_IOU([min(xs), min(ys),max(xs), max(ys)],human_box)
                pose = []
                for j in range(18):
                    pose.append(float(data[54*per + 3*j]))
                    pose.append(float(data[54*per + 3*j + 1]))
        return pose

    
with open('/home/chengao/Network/action_index.json') as json_data:
    Action_dic = json.load(json_data) 
    
coco = vu.load_coco()
vcoco_all = vu.load_vcoco('vcoco_trainval')
for x in vcoco_all:
    x = vu.attach_gt_boxes(x, coco)
    
classes = [x['action_name'] for x in vcoco_all]
np.random.seed(1)
trianval_GT = []
for i in range(26):
    print i
    vcoco = vcoco_all[i] # all instances in this action
    positive_index = np.where(vcoco['label'] == 1)[0]

    for id in positive_index:
        UnknownFlag = 1
        image_id = vcoco_all[i]['image_id'][id][0]
        
        if len(vcoco_all[i]['role_name']) == 1:# Those without roles (run, smile, stand, walk)
            action_name = vcoco_all[i]['action_name']
            human_box = vcoco_all[i]['bbox'][id]
            object_box = np.full(4, np.nan)
            pose = getOpenPose(image_id, human_box)   
            
            for index, element in enumerate(trianval_GT):
                if ((element[0] == image_id) & (all(human_box == element[2])) & (all(np.isnan(element[3])))):
                    trianval_GT[index][1].append(Action_dic[action_name])
                    UnknownFlag = 0
                    break
            if (UnknownFlag == 1):
                tmp = []
                tmp.append(image_id)
                tmp.append([Action_dic[action_name]])
                tmp.append(human_box)
                tmp.append(object_box)
                tmp.append(pose)
                trianval_GT.append(tmp)
                                 
        if len(vcoco_all[i]['role_name']) == 2:# Those with obj/instr
            action_name = vcoco_all[i]['action_name'] + '_' + vcoco_all[i]['role_name'][1]
            human_box = vcoco_all[i]['bbox'][id]
            object_box = vcoco_all[i]['role_bbox'][id][4:]
            pose = getOpenPose(image_id, human_box)      
            
            for index, element in enumerate(trianval_GT):## image is the same & human box is the same & (object box is the same or object box is all nan)
                if ((element[0] == image_id) and (all(human_box == element[2])) and ((all(object_box == element[3])) or ((all(np.isnan(element[3]))) and (all(np.isnan(object_box)))))):
                    
                    trianval_GT[index][1].append(Action_dic[action_name])
                    UnknownFlag = 0
                    break            
            if (UnknownFlag == 1):
                tmp = []
                tmp.append(image_id)
                tmp.append([Action_dic[action_name]])
                tmp.append(human_box)
                tmp.append(object_box)
                tmp.append(pose)
                trianval_GT.append(tmp)

            
        if len(vcoco_all[i]['role_name']) == 3:# Those with obj and instr
            action_name = vcoco_all[i]['action_name'] + '_' + vcoco_all[i]['role_name'][1]
            human_box = vcoco_all[i]['bbox'][id]
            object_box = vcoco_all[i]['role_bbox'][id][4:8]
            pose = getOpenPose(image_id, human_box)  
                    
            for index, element in enumerate(trianval_GT):
                    if ((element[0] == image_id) and (all(human_box == element[2])) and ((all(object_box == element[3])) or ((all(np.isnan(element[3]))) and (all(np.isnan(object_box)))))):
                        trianval_GT[index][1].append(Action_dic[action_name])
                        UnknownFlag = 0
                        break 
            if (UnknownFlag == 1):        
                tmp = []
                tmp.append(image_id)
                tmp.append([Action_dic[action_name]])
                tmp.append(human_box)
                tmp.append(object_box)
                tmp.append(pose)
                trianval_GT.append(tmp)
            
            UnknownFlag = 1
            action_name = vcoco_all[i]['action_name'] + '_' + vcoco_all[i]['role_name'][2]
            human_box = vcoco_all[i]['bbox'][id]
            object_box = vcoco_all[i]['role_bbox'][id][8:]
            pose = getOpenPose(image_id, human_box)   
                        
            for index, element in enumerate(trianval_GT):
                    if ((element[0] == image_id) and (all(human_box == element[2])) and ((all(object_box == element[3])) or ((all(np.isnan(element[3]))) and (all(np.isnan(object_box)))))):
                        trianval_GT[index][1].append(Action_dic[action_name])
                        UnknownFlag = 0
                        break
            if (UnknownFlag == 1):        
                tmp = []
                tmp.append(image_id)
                tmp.append([Action_dic[action_name]])
                tmp.append(human_box)
                tmp.append(object_box)
                tmp.append(pose)
                trianval_GT.append(tmp)
            
pickle.dump( trianval_GT, open( "/home/chengao/Network/trainval_GT_multiLabel.pkl", "wb" ) )

with open('/home/chengao/TFFRCNN/data/coco/annotations/instances_train2014.json') as json_data:
    coco_train = json.load(json_data)

for index, instance in enumerate(trianval_GT):
    flag = 0
    if index % 100 == 0:
        print index
    if np.isnan(instance[3][0]):
        continue
    for i in coco_train['annotations']:
        if (i['image_id'] == instance[0]) and all((instance[3][:2] ==  i['bbox'][:2])):
            trianval_GT[index].append(i['category_id'])
            flag = 1
            break

pickle.dump( trianval_GT, open( "/home/chengao/Network/trainval_GT_multiLabel_objclass.pkl", "wb" ) )

                    