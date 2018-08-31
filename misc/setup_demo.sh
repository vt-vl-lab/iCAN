#!/bin/bash

# Clone object detector tf-faster-rcnn
echo "Cloning object detector tf-faster-rcnn..."
git clone https://github.com/endernewton/tf-faster-rcnn ../tf-faster-rcnn/


cd ../tf-faster-rcnn/lib
make clean
make
cd ../../iCAN

# Download V-COCO Pre-trained weights
echo "Download V-COCO Pre-trained weights..."
mkdir Weights/iCAN_ResNet50_VCOCO/

python lib/ult/Download_data.py 1PCr0p2etxDsQcqpeiR5Pc9JBDm8z0yHZ Weights/iCAN_ResNet50_VCOCO/HOI_iter_300000.ckpt.data-00000-of-00001
python lib/ult/Download_data.py 1K3fPbaQ2pr-07q7wlKC55ihR9ZZTK-4Y Weights/iCAN_ResNet50_VCOCO/HOI_iter_300000.ckpt.index
python lib/ult/Download_data.py 1Jon4piZY3bD_l5iqhpxK7SxK_D2BXIoi Weights/iCAN_ResNet50_VCOCO/HOI_iter_300000.ckpt.meta

cp misc/Object_Detector.py ../tf-faster-rcnn/tools/
cp misc/resnet_v1.py ../tf-faster-rcnn/lib/nets/


