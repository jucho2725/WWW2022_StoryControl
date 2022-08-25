from transformers import RobertaForSequenceClassification, RobertaTokenizer
from datasets import Dataset, load_metric
from transformers import (
HfArgumentParser,
Trainer,
DataCollatorWithPadding,
set_seed,
RobertaConfig,
)
from transformers.trainer_utils import get_last_checkpoint

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

def no_genre_bos_first(text):
    if text.find("<|startoftext|>") != -1:
        return text[text.find("<|startoftext|>") + len("<|startoftext|>") + 1:]
    else:
        return text

def evaluate_gen(output_dir, num_train_epochs):

    data_args = DataArguments(overwrite_cache=True,
                  max_seq_length=512,)
    model_args = ModelArguments(num_labels=4,
                                model_name_or_path="cls_models/hall_of_fame/roberta_210623_1991_4/")
    train_args = TrainingArguments(output_dir=output_dir,
                      do_train=False,
                      do_eval=False,
                      do_predict=True,
                      per_device_eval_batch_size=8,
                      overwrite_output_dir=True,
                      report_to = None)

    set_seed(train_args.seed)

    # name = "roberta-large"
    # name = 'distilroberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path)
    config = RobertaConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = model_args.num_labels
    model = RobertaForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)

    data_path = os.path.join(train_args.output_dir, f"epoch_last", f"result_last.tsv")
    # data_path = "outputs/0929_story_cocon_output.tsv"
    # data_path = "outputs/pplm_output_bow1000_4000.tsv"
    df = pd.read_csv(filepath_or_buffer=data_path, sep='\t',
                     header=0, index_col=False)
    df['label'] = df['genre'].apply(lambda x: label_to_int[x])
    # df['input'] = df['content']
    df['input'] = df['content'].apply(no_genre_bos_first)
    valid_ds = Dataset.from_pandas(df)
    # valid_ds = valid_ds.select(range(32))

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
    metric = load_metric("accuracy")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1).tolist()
        label_ids = list(p.label_ids)

        return metric.compute(predictions=preds, references=label_ids)


    trainer = Trainer(
        model=model,
        args=train_args,
        # train_dataset=train_dataset if train_args.do_train else None,
        eval_dataset=valid_dataset if train_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    outputs = trainer.predict(test_dataset=valid_dataset)
    result_dict = {'4-class': outputs.metrics["test_accuracy"]}
    logger.info("showing acccuracy")
    for genre in label_to_int.keys():
        binary_prediction = np.array(list(map(lambda x: label_to_binary(x, selected_genre=label_to_int[genre]),
                                              np.argmax(outputs.predictions, axis=1))))
        binary_label_ids = np.array(
            list(map(lambda x: label_to_binary(x, selected_genre=label_to_int[genre]), outputs.label_ids)))
        results = acc_and_f1(binary_prediction, binary_label_ids)
        result_dict[genre] = results['f1']
    return result_dict

if __name__ == '__main__':
    # output_dir = "./outputs/0926_cocon"
    # num_train_epochs = 3
    results = evaluate_gen(output_dir, num_train_epochs)

    output_eval_file = os.path.join(output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in sorted(results.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")