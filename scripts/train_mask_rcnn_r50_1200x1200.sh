#!/bin/sh

export CUDA_VISIBLE_DEVICES=5

CHECKPOINT_DIR=../checkpoints/mask_rcnn_r50_1200x1200_from_800x800_2
CONFIG_PATH=../configs/pipeline_mask_rcnn_resnet50_coco_1200x1200.config
CHECKPOINT_STEPS=1000


python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
