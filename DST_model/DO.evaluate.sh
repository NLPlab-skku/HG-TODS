#!/bin/bash

# Parameters ------------------------------------------------------

# Project paths etc. ----------------------------------------------

OUT_DIR=ind_decoding_batch16_newdata
mkdir -p ${OUT_DIR}

CUDA_VISIBLE_DEVICES=1 python main_dst.py \
    --output_dir ${OUT_DIR} \
    --dataset_path "./data_230222" \
    --do_eval \
    --decoding "ind_decoding" \
    # --eval_concept \