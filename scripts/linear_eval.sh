#!/bin/bash
# Linear evaluation of a frozen backbone: 100 epochs, batch size 512.
# DATASET: cifar10 | cifar100 | stl10 | dtd | pets (downloaded to DATA_PATH by torchvision)
# ARCH=resnet50 (MoCo v2): SGD, lr 10.0, x0.1 at epochs 60 and 80.
# ARCH=vit_small (DINO):   SGD, lr 0.001 (x batch size / 256), cosine schedule, [CLS] tokens of the
#                          last 4 blocks.
PRETRAINED=${PRETRAINED:-./output_dir/moco_random_0.2_1.0/checkpoint.pth}
ARCH=${ARCH:-resnet50}
DATASET=${DATASET:-cifar10}
DATA_PATH=${DATA_PATH:-./data}
OUTPUT_DIR=${OUTPUT_DIR:-$(dirname ${PRETRAINED})/linear_${DATASET}}

if [[ ${ARCH} == resnet* ]]; then
    PROTOCOL="--lr 10.0 --lr_schedule step --lr_steps 60 80"
else
    PROTOCOL="--lr 0.001 --lr_scaling true --lr_schedule cosine --n_last_blocks 4"
fi

torchrun --nproc_per_node=8 main_linear.py \
    --pretrained ${PRETRAINED} --arch ${ARCH} \
    --dataset ${DATASET} --data_path ${DATA_PATH} \
    --epochs 100 --batch_size 512 ${PROTOCOL} \
    --output_dir ${OUTPUT_DIR}
