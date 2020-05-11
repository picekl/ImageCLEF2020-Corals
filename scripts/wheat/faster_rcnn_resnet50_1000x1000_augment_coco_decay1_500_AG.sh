#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

CHECKPOINT_DIR=../../checkpoints/wheat_faster_rcnn_resnet50_1000x1000_augment_coco_decay1_500_AG_one_class
CONFIG_PATH=../../configs/wheat/pipeline_faster_rcnn_resnet50_1000x1000_augment_coco_decay1_500_AG.config
CHECKPOINT_STEPS=2000

python ../../object_detection/model_main.py \
    --model_dir "$CHECKPOINT_DIR" \
    --pipeline_config_path "$CONFIG_PATH" \
    --checkpoint_steps "$CHECKPOINT_STEPS"
