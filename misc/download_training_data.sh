#!/bin/bash

# Download training data
echo "Downloading training data..."
python lib/ult/Download_data.py 1z5iZWJsMl4hafo0u7f1mVj7A2S5HQBOD Data/action_index.json
python lib/ult/Download_data.py 1QeCGE_0fuQsFa5IxIOUKoRFakOA87JqY Data/prior_mask.pkl
python lib/ult/Download_data.py 1JRMaE35EbJYxkXNSADEgTtvAFDdU20Ru Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl
python lib/ult/Download_data.py 1Y9yRTntfThrKMJbqyMzVasua25GUucf4 Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl
python lib/ult/Download_data.py 1le4aziSn_96cN3dIPCYyNsBXJVDD8-CZ Data/Trainval_GT_HICO.pkl
python lib/ult/Download_data.py 1YrsQUcBEF31cvqgCZYmX5j-ns2tgYXw7 Data/Trainval_GT_VCOCO.pkl
python lib/ult/Download_data.py 1PPPya4M2poWB_QCoAheStEYn3rPMKIgR Data/Trainval_Neg_HICO.pkl
python lib/ult/Download_data.py 1oGZfyhvArB2WHppgGVBXeYjPvgRk95N9 Data/Trainval_Neg_VCOCO.pkl
