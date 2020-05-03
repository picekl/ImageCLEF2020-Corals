#!/bin/sh

export CUDA_VISIBLE_DEVICES=5

CHECKPOINT_DIR=../checkpoints/mask_rcnn_r50_800x800_100boxes
CONFIG_PATH=../configs/pipeline_mask_rcnn_resnet50_coco_100boxes.config
CHECKPOINT_STEPS=1000

python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
