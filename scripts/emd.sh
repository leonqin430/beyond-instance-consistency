#!/bin/bash
# EMD-based similarity of the positive pairs of a view configuration (grid- and sampling-based).
# Use the same DATA_PATH / ANN_FILE / VIEWS / SCALE as for pre-training.
# ENCODER: resnet50 (used for the MoCo v2 results) | vit_small (used for the DINO results) |
#          resnet101 | vit_base | dino_vits16 | dino_vitb16 | dinov2_vits14 | dinov2_vitb14
# FRACTION: fraction of the images to use; 0.001 already gives stable scores (Appendix B.7).
DATA_PATH=${DATA_PATH:-/path/to/coco/train2017}
ANN_FILE=${ANN_FILE-/path/to/coco/annotations/instances_train2017.json}
VIEWS=${VIEWS:-random}
SCALE=${SCALE:-"0.2 1.0"}
ENCODER=${ENCODER:-resnet50}
FRACTION=${FRACTION:-1.0}
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/emd_${ENCODER}_${VIEWS}_${SCALE// /_}}

torchrun --nproc_per_node=8 main_emd.py \
    --data_path ${DATA_PATH} ${ANN_FILE:+--ann_file ${ANN_FILE}} ${CLASS_LIST:+--class_list ${CLASS_LIST}} \
    --views ${VIEWS} --crop_scale ${SCALE} \
    --encoder ${ENCODER} --strategy grid sampling --data_fraction ${FRACTION} \
    --output_dir ${OUTPUT_DIR}
