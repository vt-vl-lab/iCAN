import sys
sys.path.insert(0, '/home/chengao/Dataset/v-coco/')

import numpy as np
import json
import pickle
import copy
import ipdb

def Check_no_interaction(Obj_id):

    if Obj_id == 1:
        return 169
    if Obj_id == 2:
        return 23
    if Obj_id == 3:
        return 75
    if Obj_id == 4:
        return 159
    if Obj_id == 5:
        return 9
    if Obj_id == 6:
        return 64
    if Obj_id == 7:
        return 193
    if Obj_id == 8:
        return 575
    if Obj_id == 9:
        return 45
    if Obj_id == 10:
        return 566
    if Obj_id == 11:
        return 329
    if Obj_id == 12:
        return 505
    if Obj_id == 13:
        return 417
    if Obj_id == 14:
        return 246
    if Obj_id == 15:
        return 30
    if Obj_id == 16:
        return 85
    if Obj_id == 17:
        return 128
    if Obj_id == 18:
        return 145
    if Obj_id == 19:
        return 185
    if Obj_id == 20:
        return 106
    if Obj_id == 21:
        return 324
    if Obj_id == 22:
        return 238
    if Obj_id == 23:
        return 599
    if Obj_id == 24:
        return 347
    if Obj_id == 25:
        return 213
    if Obj_id == 26:
        return 583
    if Obj_id == 27:
        return 355
    if Obj_id == 28:
        return 545
    if Obj_id == 29:
        return 515
    if Obj_id == 30:
        return 341
    if Obj_id == 31:
        return 473
    if Obj_id == 32:
        return 482
    if Obj_id == 33:
        return 501
    if Obj_id == 34:
        return 375
    if Obj_id == 35:
        return 231
    if Obj_id == 36:
        return 234
    if Obj_id == 37:
        return 462
    if Obj_id == 38:
        return 527
    if Obj_id == 39:
        return 537
    if Obj_id == 40:
        return 53
    if Obj_id == 41:
        return 594
    if Obj_id == 42:
        return 304
    if Obj_id == 43:
        return 335
    if Obj_id == 44:
        return 382
    if Obj_id == 45:
        return 487
    if Obj_id == 46:
        return 256
    if Obj_id == 47:
        return 223
    if Obj_id == 48:
        return 207
    if Obj_id == 49:
        return 444
    if Obj_id == 50:
        return 406
    if Obj_id == 51:
        return 263
    if Obj_id == 52:
        return 282
    if Obj_id == 53:
        return 362
    if Obj_id == 54:
        return 428
    if Obj_id == 55:
        return 312
    if Obj_id == 56:
        return 272
    if Obj_id == 57:
        return 91
    if Obj_id == 58:
        return 95
    if Obj_id == 59:
        return 173
    if Obj_id == 60:
        return 242
    if Obj_id == 61:
        return 110
    if Obj_id == 62:
        return 557
    if Obj_id == 63:
        return 197
    if Obj_id == 64:
        return 388
    if Obj_id == 65:
        return 396
    if Obj_id == 66:
        return 437
    if Obj_id == 67:
        return 367
    if Obj_id == 68:
        return 289
    if Obj_id == 69:
        return 392
    if Obj_id == 70:
        return 413
    if Obj_id == 71:
        return 549
    if Obj_id == 72:
        return 452
    if Obj_id == 73:
        return 433
    if Obj_id == 74:
        return 251
    if Obj_id == 75:
        return 294
    if Obj_id == 76:
        return 587
    if Obj_id == 77:
        return 448
    if Obj_id == 78:
        return 532
    if Obj_id == 79:
        return 351
    if Obj_id == 80:
        return 561
        
        
def Add_mask(classid):
    if classid == 1 : begin = 161; finish =  170 # 1 person
    if classid == 2 : begin = 11; finish =   24  # 2 bicycle
    if classid == 3 : begin = 66; finish =   76  # 3 car
    if classid == 4 : begin = 147; finish =  160 # 4 motorcycle
    if classid == 5 : begin = 1; finish =    10  # 5 airplane
    if classid == 6 : begin = 55; finish =   65  # 6 bus
    if classid == 7 : begin = 187; finish =  194 # 7 train
    if classid == 8 : begin = 568; finish =  576 # 8 truck
    if classid == 9 : begin = 32; finish =   46  # 9 boat
    if classid == 10: begin = 563; finish =  567 # 10 traffic light
    if classid == 11: begin = 326; finish = 330 # 11 fire_hydrant
    if classid == 12: begin = 503; finish = 506 # 12 stop_sign
    if classid == 13: begin = 415; finish = 418 # 13 parking_meter
    if classid == 14: begin = 244; finish = 247 # 14 bench
    if classid == 15: begin = 25; finish =   31 # 15 bird
    if classid == 16: begin = 77; finish =   86 # 16 cat
    if classid == 17: begin = 112; finish = 129 # 17 dog
    if classid == 18: begin = 130; finish = 146 # 18 horse
    if classid == 19: begin = 175; finish = 186 # 19 sheep
    if classid == 20: begin = 97; finish = 107  # 20 cow
    if classid == 21: begin = 314; finish = 325 # 21 elephant
    if classid == 22: begin = 236; finish = 239 # 22 bear
    if classid == 23: begin = 596; finish = 600 # 23 zebra
    if classid == 24: begin = 343; finish = 348 # 24 giraffe
    if classid == 25: begin = 209; finish = 214 # 25 backpack
    if classid == 26: begin = 577; finish = 584 # 26 umbrella
    if classid == 27: begin = 353; finish = 356 # 27 handbag
    if classid == 28: begin = 539; finish = 546 # 28 tie
    if classid == 29: begin = 507; finish = 516 # 29 suitcase
    if classid == 30: begin = 337; finish = 342 # 30 Frisbee
    if classid == 31: begin = 464; finish = 474 # 31 skis
    if classid == 32: begin = 475; finish = 483 # 32 snowboard
    if classid == 33: begin = 489; finish = 502 # 33 sports_ball
    if classid == 34: begin = 369; finish = 376 # 34 kite
    if classid == 35: begin = 225; finish = 232 # 35 baseball_bat
    if classid == 36: begin = 233; finish = 235 # 36 baseball_glove
    if classid == 37: begin = 454; finish = 463 # 37 skateboard
    if classid == 38: begin = 517; finish = 528 # 38 surfboard
    if classid == 39: begin = 534; finish = 538 # 39 tennis_racket
    if classid == 40: begin = 47; finish = 54   # 40 bottle
    if classid == 41: begin = 589; finish = 595 # 41 wine_glass
    if classid == 42: begin = 296; finish = 305 # 42 cup
    if classid == 43: begin = 331; finish = 336 # 43 fork
    if classid == 44: begin = 377; finish = 383 # 44 knife
    if classid == 45: begin = 484; finish = 488 # 45 spoon
    if classid == 46: begin = 253; finish = 257 # 46 bowl
    if classid == 47: begin = 215; finish = 224 # 47 banana
    if classid == 48: begin = 199; finish = 208 # 48 apple
    if classid == 49: begin = 439; finish = 445 # 49 sandwich
    if classid == 50: begin = 398; finish = 407 # 50 orange
    if classid == 51: begin = 258; finish = 264 # 51 broccoli
    if classid == 52: begin = 274; finish = 283 # 52 carrot
    if classid == 53: begin = 357; finish = 363 # 53 hot_dog
    if classid == 54: begin = 419; finish = 429 # 54 pizza
    if classid == 55: begin = 306; finish = 313 # 55 donut
    if classid == 56: begin = 265; finish = 273 # 56 cake
    if classid == 57: begin = 87; finish = 92   # 57 chair
    if classid == 58: begin = 93; finish = 96   # 58 couch
    if classid == 59: begin = 171; finish = 174 # 59 potted_plant
    if classid == 60: begin = 240; finish = 243 #60 bed
    if classid == 61: begin = 108; finish = 111 #61 dining_table
    if classid == 62: begin = 551; finish = 558 #62 toilet
    if classid == 63: begin = 195; finish = 198 #63 TV
    if classid == 64: begin = 384; finish = 389 #64 laptop
    if classid == 65: begin = 394; finish = 397 #65 mouse
    if classid == 66: begin = 435; finish = 438 #66 remote
    if classid == 67: begin = 364; finish = 368 #67 keyboard
    if classid == 68: begin = 284; finish = 290 #68 cell_phone
    if classid == 69: begin = 390; finish = 393 #69 microwave
    if classid == 70: begin = 408; finish = 414 #70 oven
    if classid == 71: begin = 547; finish = 550 #71 toaster
    if classid == 72: begin = 450; finish = 453 #72 sink
    if classid == 73: begin = 430; finish = 434 #73 refrigerator
    if classid == 74: begin = 248; finish = 252 #74 book
    if classid == 75: begin = 291; finish = 295 #75 clock
    if classid == 76: begin = 585; finish = 588 #76 vase
    if classid == 77: begin = 446; finish = 449 #77 scissors
    if classid == 78: begin = 529; finish = 533 #78 teddy_bear
    if classid == 79: begin = 349; finish = 352 #79 hair_drier
    if classid == 80: begin = 559; finish = 562 #80 toothbrush
    mask_list = []
    for i in range(begin-1, finish):
        mask_list.append(i)
    return mask_list
 
        
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

trainval_GT    = pickle.load( open( "/home/chengao/Project/Network/data/trainval_GT_HO_HICO.pkl", "rb" ) )
COCO_dic       = pickle.load( open( "/home/chengao/Project/Network/data/Train_Faster_RCNN_R-50-PFN_2x_HICO.pkl", "rb" ) )


Trainval_COCO_N = {}
for COCO_imgID, COCO_box in COCO_dic.iteritems():
    if COCO_imgID%100 == 0:
        print(COCO_imgID)
    for idx_out, HO_out in enumerate(COCO_box): 

        id = HO_out[0]
        if (int(HO_out[4]) == 1) and HO_out[5] > 0.8: # loop every human box
            COCO_H = HO_out[2]
            gt_list_w_obj  = []
            
            # for every COCO human, try to find the matched GT human
            for ele_idx, ele in enumerate(trainval_GT):
                if ele[0] == id: # find the same image
                    if bb_IOU(ele[2], COCO_H) > 0.8: # find the human GT
                            gt_list_w_obj.append(ele[3]) # append the object that is related to this GT
    
            if gt_list_w_obj: # gt_list is not empty. This COCO_H has match

                for idx_in, HO_in in enumerate(COCO_box):
                    COCO_O = HO_in[2]
                    
                    if (not np.all(COCO_O == COCO_H)) and (not (any(bb_IOU(COCO_O, x) > 0.8 for x in gt_list_w_obj))) and (HO_in[5] > 0.4): # COCO_H COCO_O is Neg
                        tmp = []
                        tmp.append(id)            # id
                        tmp.append(int(Check_no_interaction(int(HO_in[4])))) # action class
                        tmp.append(COCO_H)
                        tmp.append(COCO_O) 
                        tmp.append(Add_mask(int(HO_in[4])))   # mask
                        tmp.append(int(HO_in[4])) # object class
                        tmp.append(HO_in[5])      # Object class score
                        if COCO_imgID not in Trainval_COCO_N:
                            Trainval_COCO_N[COCO_imgID] = [tmp]
                        else:
                            Trainval_COCO_N[COCO_imgID].append(tmp)

            
            else: # This COCO_H has no match
                
                for idx_in, HO_in in enumerate(COCO_box):
                    COCO_O = HO_in[2]
                    
                    if (not np.all(COCO_O == COCO_H)) and (HO_in[5] > 0.4): # COCO_H COCO_O is Neg
                        tmp = []
                        tmp.append(id)            # id
                        tmp.append(int(Check_no_interaction(int(HO_in[4]))))  # action class
                        tmp.append(COCO_H)
                        tmp.append(COCO_O) 
                        tmp.append(Add_mask(int(HO_in[4])))   # mask
                        tmp.append(int(HO_in[4])) # object class
                        tmp.append(HO_in[5])   # Object class score
                        if COCO_imgID not in Trainval_COCO_N:
                            Trainval_COCO_N[COCO_imgID] = [tmp]
                        else:
                            Trainval_COCO_N[COCO_imgID].append(tmp)

                            
pickle.dump( Trainval_COCO_N, open( "/home/chengao/Project/Network/data/trainval_N_dic_HICO.pkl", "wb" ) )