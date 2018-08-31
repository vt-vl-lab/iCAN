clear all

List = load('data/hico_20160224_det/anno.mat', 'list_test');
List = List.list_test;
for i = 1 : 9658
    list(i) = str2double(List{i}(end-9 : end-4));
end

mkdir 'output/ho_1_s/hico_det_test2015/rcnn_caffenet_pconv_ip_iter_150000'
parpool(4)
parfor obj_idx = 1 : 80
    disp(obj_idx)
    Detection = load(['../../Results/HICO_DET/1800000_iCAN_ResNet50_HICO/detections_' sprintf('%02d',obj_idx) '.mat']);
    Detection = Detection.all_boxes;
    
    if obj_idx == 1
        start = 161; finish = 170;
    end % 1 person
    if obj_idx == 2
        start = 11; finish = 24;
    end % 2 bicycle
    if obj_idx == 3
        start = 66; finish = 76;
    end % 3 car
    if obj_idx == 4
        start = 147; finish = 160;
    end % 4 motorcycle
    if obj_idx == 5
        start = 1; finish = 10;
    end % 5 airplane
    if obj_idx == 6
        start = 55; finish = 65;
    end % 6 bus
    if obj_idx == 7
        start = 187; finish = 194;
    end % 7 train
    if obj_idx == 8
        start = 568  ; finish = 576;
    end % 8 truck
    if obj_idx == 9
        start = 32  ; finish =  46;
    end % 9 boat
    if obj_idx == 10
        start = 563 ; finish =  567;
    end % 10 traffic light
    if obj_idx == 11
        start = 326; finish =  330;
    end % 11 fire_hydrant
    if obj_idx == 12
        start = 503 ; finish = 506;
    end % 12 stop_sign
    if obj_idx == 13
        start = 415 ; finish = 418;
    end % 13 parking_meter
    if obj_idx == 14
        start = 244 ; finish = 247;
    end % 14 bench
    if obj_idx == 15
        start = 25  ; finish =  31;
    end % 15 bird
    if obj_idx == 16
        start = 77  ; finish =  86;
    end % 16 cat
    if obj_idx == 17
        start = 112 ; finish = 129;
    end % 17 dog
    if obj_idx == 18
        start = 130; finish =  146;
    end % 18 horse
    if obj_idx == 19
        start = 175 ; finish = 186;
    end % 19 sheep
    if obj_idx == 20
        start = 97 ; finish = 107 ;
    end  % 20 cow
    if obj_idx == 21
        start = 314; finish =  325;
    end % 21 elephant
    if obj_idx == 22
        start = 236; finish =  239 ;
    end % 22 bear
    if obj_idx == 23
        start = 596; finish =  600;
    end % 23 zebra
    if obj_idx == 24
        start = 343; finish =  348;
    end % 24 giraffe
    if obj_idx == 25
        start = 209 ; finish = 214 ;
    end % 25 backpack
    if obj_idx == 26
        start = 577; finish =  584;
    end % 26 umbrella
    if obj_idx == 27
        start = 353; finish =  356 ;
    end % 27 handbag
    if obj_idx == 28
        start = 539; finish =  546 ;
    end % 28 tie
    if obj_idx == 29
        start = 507; finish =  516 ;
    end % 29 suitcase
    if obj_idx == 30
        start = 337; finish =  342;
    end % 30 Frisbee
    if obj_idx == 31
        start = 464; finish =  474;
    end % 31 skis
    if obj_idx == 32
        start = 475; finish =  483 ;
    end % 32 snowboard
    if obj_idx == 33
        start = 489 ; finish = 502;
    end % 33 sports_ball
    if obj_idx == 34
        start = 369; finish =  376 ;
    end % 34 kite
    if obj_idx == 35
        start = 225; finish =  232 ;
    end % 35 baseball_bat
    if obj_idx == 36
        start = 233 ; finish = 235 ;
    end % 36 baseball_glove
    if obj_idx == 37
        start = 454 ; finish = 463 ;
    end % 37 skateboard
    if obj_idx == 38
        start = 517; finish =  528 ;
    end % 38 surfboard
    if obj_idx == 39
        start = 534; finish =  538 ;
    end % 39 tennis_racket
    if obj_idx == 40
        start = 47 ; finish = 54 ;
    end   % 40 bottle
    if obj_idx == 41
        start = 589; finish =  595;
    end % 41 wine_glass
    if obj_idx == 42
        start = 296; finish =  305 ;
    end % 42 cup
    if obj_idx == 43
        start = 331; finish =  336 ;
    end % 43 fork
    if obj_idx == 44
        start = 377 ; finish = 383 ;
    end % 44 knife
    if obj_idx == 45
        start = 484 ; finish = 488;
    end % 45 spoon
    if obj_idx == 46
        start = 253 ; finish = 257;
    end % 46 bowl
    if obj_idx == 47
        start = 215; finish =  224 ;
    end % 47 banana
    if obj_idx == 48
        start = 199 ; finish = 208 ;
    end % 48 apple
    if obj_idx == 49
        start = 439 ; finish = 445 ;
    end % 49 sandwich
    if obj_idx == 50
        start = 398 ; finish = 407;
    end % 50 orange
    if obj_idx == 51
        start = 258 ; finish = 264;
    end % 51 broccoli
    if obj_idx == 52
        start = 274 ; finish = 283;
    end % 52 carrot
    if obj_idx == 53
        start = 357 ; finish = 363 ;
    end % 53 hot_dog
    if obj_idx == 54
        start = 419 ; finish = 429 ;
    end % 54 pizza
    if obj_idx == 55
        start = 306 ; finish = 313;
    end % 55 donut
    if obj_idx == 56
        start = 265 ; finish = 273 ;
    end % 56 cake
    if obj_idx == 57
        start = 87 ; finish = 92 ;
    end   % 57 chair
    if obj_idx == 58
        start = 93 ; finish = 96 ;
    end   % 58 couch
    if obj_idx == 59
        start = 171 ; finish = 174 ;
    end % 59 potted_plant
    if obj_idx == 60
        start = 240 ; finish = 243 ;
    end %60 bed
    if obj_idx == 61
        start = 108 ; finish = 111 ;
    end %61 dining_table
    if obj_idx == 62
        start = 551 ; finish = 558 ;
    end %62 toilet
    if obj_idx == 63
        start = 195 ; finish = 198;
    end %63 TV
    if obj_idx == 64
        start = 384 ; finish = 389;
    end %64 laptop
    if obj_idx == 65
        start = 394 ; finish = 397;
    end %65 mouse
    if obj_idx == 66
        start = 435 ; finish = 438 ;
    end %66 remote
    if obj_idx == 67
        start = 364 ; finish = 368 ;
    end %67 keyboard
    if obj_idx == 68
        start = 284 ; finish = 290 ;
    end %68 cell_phone
    if obj_idx == 69
        start = 390 ; finish = 393 ;
    end %69 microwave
    if obj_idx == 70
        start = 408 ; finish = 414 ;
    end %70 oven
    if obj_idx == 71
        start = 547 ; finish = 550 ;
    end %71 toaster
    if obj_idx == 72
        start = 450; finish =  453 ;
    end %72 sink
    if obj_idx == 73
        start = 430 ; finish = 434 ;
    end %73 refrigerator
    if obj_idx == 74
        start = 248 ; finish = 252 ;
    end %74 book
    if obj_idx == 75
        start = 291 ; finish = 295 ;
    end %75 clock
    if obj_idx == 76
        start = 585 ; finish = 588;
    end %76 vase
    if obj_idx == 77
        start = 446 ; finish = 449;
    end %77 scissors
    if obj_idx == 78
        start = 529 ; finish = 533 ;
    end %78 teddy_bear
    if obj_idx == 79
        start = 349 ; finish = 352 ;
    end %79 hair_drier
    if obj_idx == 80
        start = 559 ; finish = 562 ;
    end %80 toothbrush
    
    all_boxes = cell(finish - start + 1, 9658);
 
    for i = 1 : size(Detection,1)
        H = round(Detection{i,1});
        O = round(Detection{i,2});
        imageId = Detection{i,3};
        HOI     = Detection{i,4} + 1;
        score   = Detection{i,5};


        if isempty(all_boxes{HOI, find(list == imageId)})
            all_boxes{HOI, find(list == imageId)} = [H O score];
        else
            all_boxes{HOI, find(list == imageId)} = [all_boxes{HOI, find(list == imageId)}; [H O score]];
        end

    end
        
%     clear all_boxes
    save_mat(obj_idx, 'output/ho_1_s/hico_det_test2015/rcnn_caffenet_pconv_ip_iter_150000', all_boxes)
end

delete(gcp('nocreate'))

eval_run

rmdir('evaluation/result', 's')