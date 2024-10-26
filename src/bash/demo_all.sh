#!/bin/bash

GPU_ID=0
BATCH_SIZE=1
VIS_OVERLAY=False
SAVE_RES=True
SAVE_OVERLAY=True
TRAIN_DATA="UMD"
DATA_DIR="..."
DEST_DIR_BASE="..."
CKPT_DIR_BASE="..."

for MODEL_NAME in "CNN" "DRNAtt" "AffNet" "Mask2Former"
do
  if [ "$MODEL_NAME" = "CNN" ]
  then
    CKPT_PATH="$CKPT_DIR_BASE/UMD_CNN/cnn_54.pth"
  elif [[ "$MODEL_NAME" == "DRNAtt" ]]
  then
    CKPT_PATH="$CKPT_DIR_BASE/UMD_DRNAtt/drnatt_44.pth"
  elif [[ "$MODEL_NAME" == "AffNet" ]]
  then
    CKPT_PATH="$CKPT_DIR_BASE/UMD_AffNet/affnet_40.pth"
  elif [[ "$MODEL_NAME" == "Mask2Former" ]]
  then
    CKPT_PATH="$CKPT_DIR_BASE/UMD_Mask2Former/mask2former_89.pth"
  fi

  DEST_DIR="$DEST_DIR_BASE$MODEL_NAME"

  python src/demo.py \
  --gpu_id=$GPU_ID \
  --model_name=$MODEL_NAME \
  --train_dataset=$TRAIN_DATA \
  --data_dir=$DATA_DIR \
  --checkpoint_path=$CKPT_PATH \
  --batch_size=$BATCH_SIZE \
  --save_overlay=$SAVE_OVERLAY \
  --save_res=$SAVE_RES \
  --dest_dir=$DEST_DIR
done
