#!/bin/bash

# gpt2
accelerate launch ../train_gen_acc.py --train_data_file ../data/train_1991_4.tsv --eval_data_file ../data/valid_1991_4.tsv --output_dir ../outputs/gpt2_finetune --overwrite_output_dir \
 --model_name_or_path gpt2 --do_train --do_eval --num_train_epochs 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --fp16 --scl_weight 0.0 --tau 0.05 --evaluation_strategy epoch --evaluation_metric both --evaluation_first --no_genre

# cclm
accelerate launch ../train_gen_acc.py --train_data_file ../data/train_1991_4.tsv --eval_data_file ../data/valid_1991_4.tsv --output_dir ../outputs/scl0_tau5 --overwrite_output_dir \
 --model_name_or_path gpt2 --do_train --do_eval --num_train_epochs 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --fp16 --scl_weight 0.0 --tau 0.05 --evaluation_strategy epoch --evaluation_metric both --evaluation_first

# scl
accelerate launch ../train_gen_acc.py --train_data_file ../data/train_1991_4.tsv --eval_data_file ../data/valid_1991_4.tsv --output_dir ../outputs/scl50_tau5 --overwrite_output_dir \
--model_name_or_path gpt2 --do_train --do_eval --num_train_epochs 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --fp16 --scl_weight 0.5 --tau 0.05 --evaluation_strategy epoch --evaluation_metric both --evaluation_first