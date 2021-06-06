from transformers import RobertaForSequenceClassification, RobertaTokenizer


from datasets import Dataset
from transformers import (
HfArgumentParser,
Trainer,
DataCollatorWithPadding,
set_seed,
RobertaConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.model_selection import train_test_split

import logging
import multiprocessing as mp
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from control.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

def main():

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    set_seed(train_args.seed)

    name = "roberta-large"
    # name = 'distilroberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(name)
    config = RobertaConfig.from_pretrained(name)
    config.num_labels = model_args.num_labels
    model = RobertaForSequenceClassification.from_pretrained(name, config=config)


    label_to_int = {'romance': 0,
                'horror': 2,
                'thriller': 1,
               'fantasy':3,
               'western':4}
    int_to_label = {v: k for k, v in label_to_int.items()}

    df = pd.read_csv(filepath_or_buffer=data_args.train_data_file, sep='\t',
                            header=0, index_col=False).dropna()
    df['label'] = df['genre'].apply(lambda x: label_to_int[x])


    X_train, X_valid, y_train, y_valid = train_test_split(list(df['content']), list(df['label']),
     test_size=0.1, random_state=2021, stratify=list(df['label']))

    train_df = pd.DataFrame({'input': X_train,
                           'label': y_train})
    valid_df = pd.DataFrame({'input': X_valid,
                           'label': y_valid})

    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)

    padding = False
    preprocessing_num_workers = int(mp.cpu_count() / 2)


    def preprocess_function(examples):
        inputs = examples['input']
        label = examples['label']

        model_inputs = tokenizer(inputs, max_length=data_args.max_seq_length, padding=padding, truncation=True)
        model_inputs['labels'] = label
        return model_inputs

    # train_ds = train_ds.select(range(20))
    train_dataset = train_ds.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    columns_to_return = ['input_ids', 'attention_mask', 'labels']
    train_dataset.set_format(type='torch', columns=columns_to_return)


    # valid_ds = valid_ds.select(range(20))
    valid_dataset = valid_ds.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    columns_to_return = ['input_ids','attention_mask', 'labels']
    valid_dataset.set_format(type='torch', columns=columns_to_return)


    data_collator = DataCollatorWithPadding(
        tokenizer,
        padding="max_length",
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=None,
    )


    from datasets import load_metric
    metric = load_metric("f1")

    def compute_metrics(p):
        # print(p.predictions)
        # preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # preds = 각 데이터 샘플마다 (num_labels) 만큼의 array 나옴
        # label_ids = p.label_ids[0] if isinstance(p.label_ids, tuple) else p.label_ids
        # label_ids = [p[0] for p in label_ids] if isinstance(label_ids[0], list) else label_ids
        preds = np.argmax(p.predictions, axis=1).tolist()
        label_ids = list(p.label_ids)
        
        return metric.compute(predictions=preds, references=label_ids, average='macro')

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset if train_args.do_train else None,
        eval_dataset=valid_dataset if train_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,

    )

    # Detecting last checkpoint.
    last_checkpoint=None
    if (
            os.path.isdir(train_args.output_dir)
            and train_args.do_train
            and not train_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
    if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({train_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )
        # print("hello")

    if train_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if train_args.do_eval:
        logger.info("*** Evaluate ***")


        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        metrics["eval_samples"] = len(valid_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == '__main__':
    main()