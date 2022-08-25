#!/bin/bash

# This shell script run train & inference phase for genre-controllable generation.
# You need to specify train & eval data file, classfier model path, and output directory path.

TRAIN_CLS_FILEPATH="/write/your/data/path_train.tsv"
EVAL_CLS_FILEPATH="/write/your/data/path_dev.tsv"
OUTPUT_DIR_PATH="/write/your/dir/path"

python ../train_cls.py --train_data_file $TRAIN_CLS_FILEPATH
  --eval_data_file $EVAL_CLS_FILEPATH \
  --max_seq_length 512\
  --overwrite_cache\
  --num_labels 2\
  --output_dir $OUTPUT_DIR_PATH \
  --overwrite_output_dir\
  --evaluation_strategy epoch\
  --save_strategy epoch\
  --num_train_epochs 10\
  --weight_decay 0.01\
  --gradient_accumulation_steps 2\
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8\
  --per_device_eval_batch_size 4\
  --seed 42\
  --fp16 \
  --fp16_opt_level 01\
  --load_best_model_at_end\
  --metric_for_best_model f1

