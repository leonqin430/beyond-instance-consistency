#!/bin/bash
# MoCo v2 pre-training: ResNet-50, 100 epochs, batch size 256 (8 GPUs x 32).
# VIEWS: random | zero_overlap | instance_vs_bg | only_bg | lower_bound   (see views.py)
# SCALE: crop scale, "0.2 1.0" (default), "0.4 1.0" (Larger Crop),
#        "0.08 0.4" (Smaller Crop, COCO) or "0.18 0.9" (Smaller Crop, ImageNet-100)
# COCO:         DATA_PATH=/path/to/coco/train2017 ANN_FILE=/path/to/coco/annotations/instances_train2017.json
# ImageNet-100: DATA_PATH=/path/to/imagenet/train ANN_FILE=/path/to/imagenet100_maskcut.json
#               (views without boxes also work with ANN_FILE= CLASS_LIST=util/imagenet100.txt)
DATA_PATH=${DATA_PATH:-/path/to/coco/train2017}
ANN_FILE=${ANN_FILE-/path/to/coco/annotations/instances_train2017.json}
VIEWS=${VIEWS:-random}
SCALE=${SCALE:-"0.2 1.0"}
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/moco_${VIEWS}_${SCALE// /_}}

torchrun --nproc_per_node=8 main_moco.py \
    --data_path ${DATA_PATH} ${ANN_FILE:+--ann_file ${ANN_FILE}} ${CLASS_LIST:+--class_list ${CLASS_LIST}} \
    --views ${VIEWS} --crop_scale ${SCALE} \
    --epochs 100 --batch_size 256 --lr 0.03 \
    --output_dir ${OUTPUT_DIR}
