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
import torch
import pandas as pd

from control.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# label_to_int = {'romance': 0,
#             'action': 1,
#             'horror': 2,
#             'western':3,}
label_to_int = {'action': 0,
            'romance': 1,
            'horror': 2,
            'crime': 3,}
int_to_label = {v: k for k, v in label_to_int.items()}


def label_to_binary(label, selected_genre):
    return int(label == selected_genre)

from sklearn.metrics import f1_score


def acc_and_f1(preds, labels):
    assert len(preds) == len(labels)
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    #setattr(train_args, 'output_dir', f"../outputs/roberta_genre_{model_args.selected_genre}")


    set_seed(train_args.seed)

    # name = "roberta-large"
    # name = 'distilroberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path)
    config = RobertaConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = model_args.num_labels
    model = RobertaForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)

    if train_args.do_train:
        train_df = pd.read_csv(filepath_or_buffer=data_args.train_data_file, sep='\t',
                                header=0, index_col=False)
        train_df['int'] = train_df['genre'].apply(lambda x: label_to_int[x])
        valid_df = pd.read_csv(filepath_or_buffer=data_args.eval_data_file, sep='\t',
                                header=0, index_col=False)
        valid_df['int'] = valid_df['genre'].apply(lambda x: label_to_int[x])

        print(f"***** label 982*6*3=17676 == {train_df['int'].sum() * 3}  *****")
        print(f"***** label 200*6*3=3600 == {valid_df['int'].sum() * 3}  *****")
        train_df = pd.DataFrame({'input': train_df['content'].append(train_df['content_aug_09']).append(train_df['content_aug_05']).tolist(),
                               'labels': train_df['int'].append(train_df['int']).append(train_df['int']).tolist()})
        valid_df = pd.DataFrame({'input': valid_df['content'].append(valid_df['content_aug_09']).append(valid_df['content_aug_05']).tolist(),
                               'labels': valid_df['int'].append(valid_df['int']).append(valid_df['int']).tolist()})

        train_ds = Dataset.from_pandas(train_df)
        valid_ds = Dataset.from_pandas(valid_df)

        padding = False
        preprocessing_num_workers = int(mp.cpu_count() / 2)

        def preprocess_function(examples):
            inputs = examples['input']

            model_inputs = tokenizer(inputs, max_length=data_args.max_seq_length, padding=padding, truncation=True)
            model_inputs['labels'] = examples['labels']
            return model_inputs

        # train_ds = train_ds.select(range(20))
        train_dataset = train_ds.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        columns_to_return = ['input_ids', 'attention_mask', 'labels']
        # format = {'type': 'torch', 'format_kwargs': {'dtype': torch.long}, 'columns':columns_to_return}

        train_dataset.set_format(type='torch', columns=columns_to_return)

        # valid_ds = valid_ds.select(range(20))
        valid_dataset = valid_ds.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        columns_to_return = ['input_ids', 'attention_mask', 'labels']
        valid_dataset.set_format(type='torch', columns=columns_to_return)


    elif train_args.do_eval or train_args.do_predict:
        df = pd.read_csv(filepath_or_buffer=data_args.eval_data_file, sep='\t',
                                header=0, index_col=False)
        df['label'] = df['genre'].apply(lambda x: label_to_int[x])
        df['input'] = df['content']
        valid_ds = Dataset.from_pandas(df)

        padding = False
        preprocessing_num_workers = int(mp.cpu_count() / 2)

        def preprocess_function(examples):
            inputs = examples['input']
            label = examples['label']

            model_inputs = tokenizer(inputs, max_length=data_args.max_seq_length, padding=padding, truncation=True)
            model_inputs['labels'] = label
            return model_inputs
        # valid_ds = valid_ds.select(range(20))
        valid_dataset = valid_ds.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        columns_to_return = ['input_ids', 'attention_mask', 'labels']
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
    if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0 and not train_args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({train_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

    if train_args.do_train:
        logger.info(f"***** Running training *****")
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

    elif train_args.do_eval:
        logger.info("*** Evaluate ***")


        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        metrics["eval_samples"] = len(valid_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    elif train_args.do_predict:
        output_predict_file = os.path.join(train_args.output_dir, "/pred_results.txt")
        logger.info("*** Predict ***")

        outputs = trainer.predict(test_dataset=valid_dataset)

        with open(output_predict_file, "w") as writer:
            logger.info("*** 4-class classification result ***")
            writer.write("*** 4-class classification result ***\n")
            for key, value in sorted(outputs.metrics.items()):
                writer.write(f"{key} = {value}\n")
                logger.info(f"  {key} = {value}")

            logger.info("*** binary classification result ***")
            writer.write("*** binary classification result ***\n")
            for genre in label_to_int.keys():
                binary_prediction = np.array(list(map(lambda x: label_to_binary(x, selected_genre=label_to_int[genre]), np.argmax(outputs.predictions, axis=1))))
                binary_label_ids = np.array(list(map(lambda x: label_to_binary(x, selected_genre=label_to_int[genre]), outputs.label_ids)))

                results = acc_and_f1(binary_prediction, binary_label_ids)
                logger.info(f"***** genre {genre} *****")
                writer.write(f"***** genre {genre} *****\n")
                for key, value in sorted(results.items()):
                    writer.write(f"{key} = {value}\n")
                    logger.info(f"  {key} = {value}")

if __name__ == '__main__':
    main()