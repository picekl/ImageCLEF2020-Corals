#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

CHECKPOINT_DIR=../checkpoints/mask_rcnn_r50_800x800_augment_AG
CONFIG_PATH=../configs/pipeline_mask_rcnn_resnet50_coco_augment_AG.config
CHECKPOINT_STEPS=1000

python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
