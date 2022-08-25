#!/bin/bash

# This shell script run train & inference phase for genre-controllable generation.
# You need to specify train & eval data file, classfier model path, and output directory path.

TRAIN_GEN_FILEPATH="/write/your/data/path"
EVAL_GEN_FILEPATH="/write/your/data/path"
CLS_MODEL_PATH="/write/your/cls_model/path"
OUTPUT_DIR_PATH="/write/your/dir/path"

python ../train_gen.py --train_data_file $TRAIN_GEN_FILEPATH --eval_data_file $EVAL_GEN_FILEPATH\
    --output_dir $OUTPUT_DIR_PATH --overwrite_output_dir\ 
    --do_train --do_eval\
    --num_train_epochs 3 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --fp16\
    --scl_weight 0.7 --tau 0.05 --evaluation_strategy no --seed 2021\
    --model_name_or_path gpt2 --contrast_max_seq_length 64 --max_seq_length 256\
    --anchor_genre True --pos_genre True --neg_genre False\
    --learning_rate 5e-5 --loss_type cross_entropy

python ../eval_gen.py --train_data_file $TRAIN_GEN_FILEPATH --eval_data_file $EVAL_GEN_FILEPATH\
    --output_dir $OUTPUT_DIR_PATH --overwrite_output_dir\ 
    --do_train --do_eval\
    --num_train_epochs 3 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --fp16\
    --scl_weight 0.7 --tau 0.05 --evaluation_strategy no --seed 2021\
    --model_name_or_path gpt2 --contrast_max_seq_length 64 --max_seq_length 256\
    --anchor_genre True --pos_genre True --neg_genre False\
    --learning_rate 5e-5 --loss_type cross_entropy