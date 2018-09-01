# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao
# --------------------------------------------------------

def apply_prior(Object, prediction):
    
    
    if Object[4] != 32: # not a snowboard, then the action is impossible to be snowboard
        prediction[0][0][21] = 0

    if Object[4] != 74: # not a book, then the action is impossible to be read
        prediction[0][0][24] = 0

    if Object[4] != 33: # not a sports ball, then the action is impossible to be kick
        prediction[0][0][7] = 0   

    if (Object[4] != 41) and (Object[4] != 40) and (Object[4] != 42) and (Object[4] != 46): # not 'wine glass', 'bottle', 'cup', 'bowl', then the action is impossible to be drink
        prediction[0][0][13] = 0       

    if Object[4] != 37: # not a skateboard, then the action is impossible to be skateboard
        prediction[0][0][26] = 0    

    if Object[4] != 38: # not a surfboard, then the action is impossible to be surfboard
        prediction[0][0][0] = 0  
                            
    if Object[4] != 31: # not a ski, then the action is impossible to be ski
        prediction[0][0][1] = 0      
                             
    if Object[4] != 64: # not a laptop, then the action is impossible to be work on computer
        prediction[0][0][8] = 0
                        
    if (Object[4] != 77) and (Object[4] != 43) and (Object[4] != 44): # not 'scissors', 'fork', 'knife', then the action is impossible to be cur instr
        prediction[0][0][2] = 0
                        
    if (Object[4] != 33) and (Object[4] != 30): # not 'sports ball', 'frisbee', then the action is impossible to be throw and catch
        prediction[0][0][15] = 0
        prediction[0][0][28] = 0
                              
    if Object[4] != 68: # not a cellphone, then the action is impossible to be talk_on_phone
        prediction[0][0][6] = 0   
                            
    if (Object[4] != 14) and (Object[4] != 61) and (Object[4] != 62) and (Object[4] != 60) and (Object[4] != 58)  and (Object[4] != 57): # not 'bench', 'dining table', 'toilet', 'bed', 'couch', 'chair', then the action is impossible to be lay
        prediction[0][0][12] = 0
                            
    if (Object[4] != 32) and (Object[4] != 31) and (Object[4] != 37) and (Object[4] != 38): # not 'snowboard', 'skis', 'skateboard', 'surfboard', then the action is impossible to be jump
        prediction[0][0][11] = 0   
   
    if (Object[4] != 47) and (Object[4] != 48) and (Object[4] != 49) and (Object[4] != 50) and (Object[4] != 51) and (Object[4] != 52) and (Object[4] != 53) and (Object[4] != 54) and (Object[4] != 55) and (Object[4] != 56): # not ''banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 'hot dog', 'pizza', 'cake', 'donut', then the action is impossible to be eat_obj
        prediction[0][0][9] = 0 

    if (Object[4] != 43) and (Object[4] != 44) and (Object[4] != 45): # not 'fork', 'knife', 'spoon', then the action is impossible to be eat_instr
        prediction[0][0][16] = 0 
            
    if (Object[4] != 39) and (Object[4] != 35): # not 'tennis racket', 'baseball bat', then the action is impossible to be hit_instr
        prediction[0][0][19] = 0 

    if (Object[4] != 33): # not 'sports ball, then the action is impossible to be hit_obj
        prediction[0][0][20] = 0 
                            
                            
    if (Object[4] != 2) and (Object[4] != 4) and (Object[4] != 6) and (Object[4] != 8) and (Object[4] != 9) and (Object[4] != 7) and (Object[4] != 5) and (Object[4] != 3) and (Object[4] != 18) and (Object[4] != 21): # not 'bicycle', 'motorcycle', 'bus', 'truck', 'boat', 'train', 'airplane', 'car', 'horse', 'elephant', then the action is impossible to be ride
        prediction[0][0][5] = 0 
                            
    if (Object[4] != 2) and (Object[4] != 4) and (Object[4] != 18) and (Object[4] != 21) and (Object[4] != 14) and (Object[4] != 57) and (Object[4] != 58) and (Object[4] != 60) and (Object[4] != 62) and (Object[4] != 61) and (Object[4] != 29) and (Object[4] != 27) and (Object[4] != 25): # not 'bicycle', 'motorcycle', 'horse', 'elephant', 'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table', 'suitcase', 'handbag', 'backpack', then the action is impossible to be sit
        prediction[0][0][10] = 0 
        
    if (Object[4] == 1):
        prediction[0][0][4] = 0 
    
    return prediction
                            


