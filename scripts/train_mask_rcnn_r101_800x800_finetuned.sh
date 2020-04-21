#!/bin/sh

export CUDA_VISIBLE_DEVICES=7

CHECKPOINT_DIR=../checkpoints/mask_rcnn_r101_800x800_finetuned
CONFIG_PATH=../configs/pipeline_mask_rcnn_resnet101_coco_finetuned.config
CHECKPOINT_STEPS=1000


python ../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
