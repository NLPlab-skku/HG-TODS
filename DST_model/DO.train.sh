#!/bin/bash

# Parameters ------------------------------------------------------

# Project paths etc. ----------------------------------------------

OUT_DIR=sequential_batch8_epoch5_lr6.25e-5_newdata
mkdir -p ${OUT_DIR}

python main_dst.py \
    --output_dir ${OUT_DIR} \
    --do_train \
    --do_dev \
    --do_eval \
    --dataset_path "./data_230222" \
    --eval_concept \
    --decoding "sequential" \
    --train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --n_epochs 5