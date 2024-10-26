#!/bin/bash

GPU_ID=0
MODEL_NAME="..."
TRAIN_DATA="..."
DATA_DIR="...."
BATCH_SIZE=1
VIS_OVERLAY=True

python src/demo.py \
--gpu_id=$GPU_ID \
--model_name=$MODEL_NAME \
--train_dataset=$TRAIN_DATA \
--data_dir=$DATA_DIR \
--checkpoint_path=$CKPT_PATH \
--batch_size=$BATCH_SIZE \
--visualise_overlay=$VIS_OVERLAY # \
#--save_overlay=$SAVE_OVERLAY \
#--save_res=$SAVE_RES \
#--dest_dir=$DEST_DIR
