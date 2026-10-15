#!/bin/bash
# DINO pre-training: ViT-S/16, 100 epochs, batch size 256 (8 GPUs x 32), 2 global + 8 local crops.
# The view configuration is applied to the two global crops.
# VIEWS: random | zero_overlap | instance_vs_bg | only_bg | lower_bound   (see views.py)
# SCALE: scale of the global crops, "0.2 1.0" (default), "0.4 1.0" (Larger Crop),
#        "0.08 0.4" (Smaller Crop, COCO) or "0.18 0.9" (Smaller Crop, ImageNet-100)
DATA_PATH=${DATA_PATH:-/path/to/coco/train2017}
ANN_FILE=${ANN_FILE-/path/to/coco/annotations/instances_train2017.json}
VIEWS=${VIEWS:-random}
SCALE=${SCALE:-"0.2 1.0"}
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/dino_${VIEWS}_${SCALE// /_}}

torchrun --nproc_per_node=8 main_dino.py \
    --arch vit_small --patch_size 16 \
    --data_path ${DATA_PATH} ${ANN_FILE:+--ann_file ${ANN_FILE}} ${CLASS_LIST:+--class_list ${CLASS_LIST}} \
    --views ${VIEWS} --global_crops_scale ${SCALE} \
    --epochs 100 --batch_size_per_gpu 32 --lr 0.0005 \
    --output_dir ${OUTPUT_DIR}
