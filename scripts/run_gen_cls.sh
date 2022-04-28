#!/bin/bash

train_file_path="/write/your/data/path"
eval_file_path="/write/your/data/path"
output_dir_path="/write/your/dir/path"

python ../train_gen_cls.py --train_data_file $train_file_path --eval_data_file $train_file_path\
    --output_dir $output_dir_path --overwrite_output_dir\ 
    --do_train --do_eval\
    --num_train_epochs 3 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --fp16\
    --scl_weight 0.7 --tau 0.05 --evaluation_strategy no --seed 2021\
    --model_name_or_path gpt2 --contrast_max_seq_length 64 --max_seq_length 256\
    --anchor_genre True --pos_genre True --neg_genre False\
    --learning_rate 5e-5 --loss_type cross_entropy