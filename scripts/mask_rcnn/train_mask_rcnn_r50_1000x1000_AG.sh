#!/bin/sh

export CUDA_VISIBLE_DEVICES=3

CHECKPOINT_DIR=../checkpoints/mask_rcnn_r50_1000x1000_from_800x800_AG
CONFIG_PATH=../configs/mask-rcnn/pipeline_mask_rcnn_resnet50_coco_1000x1000_AG.config
CHECKPOINT_STEPS=1000

python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
