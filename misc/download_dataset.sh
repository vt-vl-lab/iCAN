#!/bin/bash

# Download COCO dataset
echo "Downloading V-COCO"

# Download V-COCO
mkdir Data
git clone --recursive https://github.com/s-gupta/v-coco.git Data/v-coco/
cd Data/v-coco/coco

URL_2014_Train_images=http://images.cocodataset.org/zips/train2014.zip
URL_2014_Val_images=http://images.cocodataset.org/zips/val2014.zip
URL_2014_Test_images=http://images.cocodataset.org/zips/test2014.zip
URL_2014_Trainval_annotation=http://images.cocodataset.org/annotations/annotations_trainval2014.zip

wget -N $URL_2014_Train_images
wget -N $URL_2014_Val_images
wget -N $URL_2014_Test_images
wget -N $URL_2014_Trainval_annotation

mkdir images

unzip train2014.zip -d images/
unzip val2014.zip -d images/
unzip test2014.zip -d images/
unzip annotations_trainval2014.zip


rm train2014.zip
rm val2014.zip
rm test2014.zip
rm annotations_trainval2014

# Pick out annotations from the COCO annotations to allow faster loading in V-COCO
echo "Picking out annotations from the COCO annotations to allow faster loading in V-COCO"

cd ../
python script_pick_annotations.py coco/annotations

# Build
echo "Building"
cd coco/PythonAPI/ && make install
cd ../../ && make
cd ../../

# Download HICO-DET dataset
echo "Downloading HICO-DET"

URL_HICO_DET=http://napoli18.eecs.umich.edu/public_html/data/hico_20160224_det.tar.gz

wget -N $URL_HICO_DET -P Data/
tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
rm Data/hico_20160224_det.tar.gz


# Download HICO-DET evaluation code
cd Data/
git clone https://github.com/ywchao/ho-rcnn.git
cd ../
cp misc/Generate_detection.m Data/ho-rcnn/
cp misc/save_mat.m Data/ho-rcnn/
cp misc/load_mat.m Data/ho-rcnn/

mkdir Data/ho-rcnn/data/hico_20160224_det/
python lib/ult/Download_data.py 1cE10X9rRzzqeSPi-BKgIcDgcPXzlEoXX Data/ho-rcnn/data/hico_20160224_det/anno_bbox.mat
python lib/ult/Download_data.py 1ds_qW9wv-J3ESHj_r_5tFSOZozGGHu1r Data/ho-rcnn/data/hico_20160224_det/anno.mat


# Download COCO Pre-trained weights
echo "Downloading COCO Pre-trained weights..."

mkdir Weights/
python lib/ult/Download_data.py 1IbR4kiWgLF8seaKjOMmwaHs0Bfwl5Dq1 Weights/res50_faster_rcnn_iter_1190000.ckpt.data-00000-of-00001
python lib/ult/Download_data.py 1-DbfEloN4c2JaCEMnexaWAsSc4MDlZJx Weights/res50_faster_rcnn_iter_1190000.ckpt.index
python lib/ult/Download_data.py 1vc5d3OwCtMtRgXq3Pj4_twpK4x3kjgT0 Weights/res50_faster_rcnn_iter_1190000.ckpt.meta


# 
mkdir Results/



