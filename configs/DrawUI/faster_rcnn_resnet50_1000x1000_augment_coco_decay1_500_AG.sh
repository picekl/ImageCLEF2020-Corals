#!/bin/sh

export CUDA_VISIBLE_DEVICES=6

CHECKPOINT_DIR=../../checkpoints/DRAWUI_faster_rcnn_resnet50_1000x1000_augment_coco_decay1_500_AG
CONFIG_PATH=../../configs/DrawUI/pipeline_faster_rcnn_resnet50_1000x1000_augment_coco_decay1_500_AG.config
CHECKPOINT_STEPS=1050

python ../../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
