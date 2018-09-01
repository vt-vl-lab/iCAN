# iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection 

Official TensorFlow implementation for [iCAN: Instance-Centric Attention Network 
for Human-Object Interaction Detection](https://arxiv.org/abs/1808.10437).

See the [project page](https://gaochen315.github.io/iCAN/) for more details. Please contact Chen Gao (chengao@vt.edu) if you have any questions.

<img src='misc/HOI.gif'>

## Prerequisites

This codebase was developed and tested with Python2.7, Tensorflow 1.1.0 or 1.2.0, CUDA 8.0 and Ubuntu 16.04.


## Installation
1. Clone the repository. 
    ```Shell
    git clone https://github.com/vt-vl-lab/iCAN.git
    ```
2. Download V-COCO and HICO-DET dataset. Setup V-COCO and COCO API. Setup HICO-DET evaluation code.
    ```Shell
    chmod +x ./misc/download_dataset.sh 
    ./misc/download_dataset.sh 
    
    # Assume you cloned the repository to `iCAN_DIR'.
    # If you have download V-COCO or HICO-DET dataset somewhere else, you can create a symlink
    # ln -s /path/to/your/v-coco/folder Data/
    # ln -s /path/to/your/hico-det/folder Data/
    ```
    
## Evaluate V-COCO and HICO-DET detection results
1. Download detection results
    ```Shell
    chmod +x ./misc/download_detection_results.sh 
    ./misc/download_detection_results.sh
    ```
2. Evaluate V-COCO detection results using iCAN
    ```Shell
    python tools/Diagnose_VCOCO.py eval Results/300000_iCAN_ResNet50_VCOCO.pkl
    ```
3. Evaluate V-COCO detection results using iCAN (Early fusion)
    ```Shell
    python tools/Diagnose_VCOCO.py eval Results/300000_iCAN_ResNet50_VCOCO_Early.pkl
    ```
3. Evaluate HICO-DET detection results using iCAN
    ```Shell
    cd Data/ho-rcnn
    matlab -r "Generate_detection; quit"
    cd ../../
    ```
    Here we evaluate our best detection results under ```Results/HICO_DET/1800000_iCAN_ResNet50_HICO```. If you want to evaluate a different detection result, please specify the filename in ```Data/ho-rcnn/Generate_detection.m``` accordingly.
   
## Error diagnose on V-COCO
1. Diagnose V-COCO detection results using iCAN
    ```Shell
    python tools/Diagnose_VCOCO.py diagnose Results/300000_iCAN_ResNet50_VCOCO.pkl
    ```
2. Diagnose V-COCO detection results using iCAN (Early fusion)
    ```Shell
    python tools/Diagnose_VCOCO.py diagnose Results/300000_iCAN_ResNet50_VCOCO_Early.pkl
    ```

## Training
1. Download COCO pre-trained weights and training data
    ```Shell
    chmod +x ./misc/download_training_data.sh 
    ./misc/download_training_data.sh
    ```
2. Train an iCAN on V-COCO
    ```Shell
    python tools/Train_ResNet_VCOCO.py --model iCAN_ResNet50_VCOCO --num_iteration 300000
    ```
3. Train an iCAN (Early fusion) on V-COCO
    ```Shell
    python tools/Train_ResNet_VCOCO.py --model iCAN_ResNet50_VCOCO_Early --num_iteration 300000
4. Train an iCAN on HICO-DET
    ```Shell
    python tools/Train_ResNet_HICO.py --num_iteration 1800000
    ```
    
## Testing
1. Test an iCAN on V-COCO
    ```Shell
     python tools/Test_ResNet_VCOCO.py --model iCAN_ResNet50_VCOCO --num_iteration 300000
    ```
2. Test an iCAN (Early fusion) on V-COCO
    ```Shell
     python tools/Test_ResNet_VCOCO.py --model iCAN_ResNet50_VCOCO_Early --num_iteration 300000
    ```
3. Test an iCAN on HICO-DET
    ```Shell
    python tools/Test_ResNet_HICO.py --num_iteration 1800000
    ```

## Visualizing V-COCO detections
Check ```tools/Visualization.ipynb``` to see how to visualize the detection results.

## Demo/Test on your own images
0. To get the best performance, we use [Detection](https://github.com/facebookresearch/Detectron) as our object detector. For a simple demo purpose, we use [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) in this section instead.
1. Clone and setup the tf-faster-rcnn repository.
    ```Shell
    cd $iCAN_DIR
    chmod +x ./misc/setup_demo.sh 
    ./misc/setup_demo.sh
    ```
2. Put your own images to ```demo/``` folder.
3. Detect all objects
    ```Shell
    # images are saved in $iCAN_DIR/demo/
    python ../tf-faster-rcnn/tools/Object_Detector.py --img_dir demo/ --img_format png --Demo_RCNN demo/Object_Detection.pkl
    ``` 
4. Detect all HOIs
    ```Shell
    python tools/Demo.py --img_dir demo/ --Demo_RCNN demo/Object_Detection.pkl --HOI_Detection demo/HOI_Detection.pkl
    ```
5. Check ```tools/Demo.ipynb``` to visualize the detection results.

## Citation
If you find this code useful for your research, please consider citing the following papers:

    @inproceedings{gao2018ican,
    author    = {Gao, Chen and Zou, Yuliang and Huang, Jia-Bin}, 
    title     = {iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection}, 
    booktitle = {British Machine Vision Conference},
    year      = {2018}
    }

## Acknowledgement
Codes are built upon [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn). We thank [Jinwoo Choi](https://github.com/jinwchoi) for the code review.
