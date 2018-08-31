#!/bin/bash

# Download VCOCO detection results
echo "Downloading VCOCO detection results"
# Detection results
python lib/ult/Download_data.py 1KKonPU_kQ5ZU16MTTCB4ZH2kfNo26m18 Results/300000_iCAN_ResNet50_VCOCO_Early.pkl
python lib/ult/Download_data.py 1LS8KsDD96eTbbV9OMoZczbQGplTNI2p2 Results/300000_iCAN_ResNet50_VCOCO.pkl

# Download HICO-DET detection results"
echo "Downloading HICO-DET detection results"
mkdir Results/HICO_DET
python lib/ult/Download_data.py 1DxrRuXGjRWIvWjdMK-HBrWaTR3WScRIY Results/HICO_DET/1800000_iCAN_ResNet50_HICO.zip
unzip Results/HICO_DET/1800000_iCAN_ResNet50_HICO.zip -d Results/HICO_DET/
rm Results/HICO_DET/1800000_iCAN_ResNet50_HICO.zip

