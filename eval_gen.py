import logging
import os
from dataclasses import asdict
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.backends.cudnn as cudnn

from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    HfArgumentParser,
    get_linear_schedule_with_warmup,
)

from control.models import (
    SupConGPT2,
    GPT2LMHeadModel,
)

from control.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments
)
from control.utils import (
    set_seed, clean_text, write_sent, write_df
)
from control.dataset import (
    load_and_cache_examples_train,
    load_and_cache_examples_eval
)
from control.data_collator import (
    DataCollatorForSCL,
    DataCollatorForLanguageModeling,
    DataCollatorForGeneration
)

from control.evaluation import (
    evaluate_ppl,
    evaluate_dist_scores
)

from apex import amp
from torch_optimizer import Lamb
from torch.optim import AdamW
# import wandb
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import List

""" KL loss """
import torch.nn.functional as F

SMALL_CONST = 1e-15
kl_scale = 0.001
""" """
os.environ['WANDB_MODE'] = 'offline'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate(model, gpt2model, tokenizer, gpt2tokenizer, eval_dataset_gen, eval_dataset_ppl, data_args, train_args,
             gen_args, epoch):
    # ppl
    results = {}

    _ = evaluate_ppl(train_args, model, tokenizer, eval_dataset_ppl)
    generated_sentences = generate_sentences(data_args, train_args, gen_args, model, tokenizer, eval_dataset_gen,
                                             epoch=epoch)
    result = evaluate_ppl_dist(generated_sentences, tokenizer, train_args, gpt2model)

    results.update(result)
    return results, generated_sentences


# evaluate perplexity
def evaluate_ppl(train_args, model, tokenizer, eval_dataset):
    eval_output_dir = train_args.output_dir

    if train_args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=train_args.eval_batch_size, collate_fn=data_collator
    )

    # Eval
    # logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", train_args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    losses = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        if not train_args.no_cuda:
            batch = {k: v.to(train_args.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model.lm_model(**batch)
            loss = outputs.loss
            loss = loss / train_args.gradient_accumulation_steps
            eval_loss += loss.mean().item()
            losses.append(loss.repeat(train_args.per_device_eval_batch_size))
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]

    perplexity1 = torch.exp(torch.tensor(eval_loss))
    perplexity2 = torch.exp(torch.mean(losses))

    result = {"perplexity1": perplexity1,
              "perplexity2": perplexity2}

    logger.info("***** PPL for valid set Eval results *****")
    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

    return result


def count_ngram(text_samples, n, tokenizer=None):
    """
    Count the number of unique n-grams
    :param text_samples: list, a list of samples
    :param n: int, n-gram
    :return: the number of unique n-grams in text_samples
    """
    if len(text_samples) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    ngram = set()
    for sample in text_samples:
        if len(sample) < n:
            continue

        sample = list(map(str, sample))
        for i in range(len(sample) - n + 1):
            ng = ' '.join(sample[i: i + n])

            ngram.add(' '.join(ng))
    return len(ngram)


class TempDataset(Dataset):
    def __init__(self, tokenizer, lines):
        self.tokenizer = tokenizer
        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True,
                                                    max_length=128)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        try:
            # 앞에 장르 토큰과  bos 토큰은 빼줌
            return torch.tensor(self.examples[i][self.examples[i].index(self.tokenizer.bos_token_id) + 1:],
                                dtype=torch.long)
        except ValueError:  # bos 가 포함되지 않는 경우
            return torch.tensor(self.examples[i], dtype=torch.long)


def generate_sentences(data_args, train_args, gen_args, model, tokenizer, eval_dataset, epoch="none"):
    eval_output_dir = train_args.output_dir
    if train_args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
    )
    # eval_dataset = eval_dataset.select(range(40))
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=train_args.eval_batch_size, collate_fn=data_collator
    )

    generated_sequences = []

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", train_args.eval_batch_size)
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Generating"):
        if not train_args.no_cuda:
            batch = {k: v.to(train_args.device) for k, v in batch.items()}

        with torch.no_grad():
            output_sequences = model.lm_model.generate(input_ids=batch['input_ids'],
                                                       attention_mask=batch['attention_mask'],
                                                       # max_length=data_args.max_seq_length,
                                                       max_length=128,
                                                       pad_token_id=tokenizer.pad_token_id,
                                                       **asdict(gen_args))
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

        generated_sequence_idx = 0
        for generated_sequence in output_sequences:
            generated_sequence = generated_sequence.tolist()
            # Decode text
            generated_sequences.append(clean_text(tokenizer.decode(generated_sequence,
                                                                   skip_special_tokens=False,
                                                                   clean_up_tokenization_spaces=True)))
            generated_sequence_idx += 1

    save_path = os.path.join(train_args.output_dir, f"epoch_{epoch}")
    os.makedirs(save_path, exist_ok=True)
    print(f"save path is {save_path}")

    write_sent(generated_sequences, os.path.join(save_path, f"result_{epoch}.txt"))
    write_df(generated_sequences, data_args, os.path.join(save_path, f"result_{epoch}.tsv"))
    print("write sent df done")
    return generated_sequences


# evaluate Dist-K scores
def evaluate_ppl_dist(generated_sequences, tokenizer, train_args, gpt2model, ):
    dist_eval_samples = []
    num_tokens = 0

    dist_eval_dataset = TempDataset(tokenizer, generated_sequences)

    # print("dist eval example", dist_eval_dataset[0])

    def collate(examples: List[torch.Tensor]):
        # if tokenizer._pad_token is None:
        #     return pad_sequence(examples, batch_first=True)
        # return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
        return pad_sequence(examples, batch_first=True)

    dist_eval_sampler = SequentialSampler(dist_eval_dataset)
    dist_eval_dataloader = DataLoader(
        dist_eval_dataset, sampler=dist_eval_sampler, batch_size=train_args.eval_batch_size, collate_fn=collate
    )

    losses = []
    logger.info("***** GPT2 PPL and Dist 123 scores for generated Eval results *****")

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dist_eval_dataset))
    logger.info("  Batch size = %d", train_args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    gpt2model.eval()

    for batch in tqdm(dist_eval_dataloader, desc="Evaluating"):
        # print(batch)
        sample_flattened = batch.reshape(-1)
        dist_eval_samples.append(sample_flattened.tolist())
        num_tokens += len(sample_flattened)

        inputs = {'input_ids': batch.to(train_args.device),
                  'labels': batch.to(train_args.device)}

        with torch.no_grad():
            outputs = gpt2model(**inputs)
            loss = outputs.loss
            loss = loss / train_args.gradient_accumulation_steps
            eval_loss += loss.mean().item()
            losses.append(loss.repeat(train_args.per_device_eval_batch_size))
        nb_eval_steps += 1

    dist1_score = count_ngram(dist_eval_samples, 1) / float(num_tokens)
    dist2_score = count_ngram(dist_eval_samples, 2) / float(num_tokens)
    dist3_score = count_ngram(dist_eval_samples, 3) / float(num_tokens)

    result = {"Dist-1": dist1_score, "Dist-2": dist2_score, "Dist-3": dist3_score}

    eval_loss = eval_loss / nb_eval_steps
    losses = torch.cat(losses)
    losses = losses[: len(dist_eval_dataset)]

    perplexity1 = torch.exp(torch.tensor(eval_loss))
    perplexity2 = torch.exp(torch.mean(losses))

    result.update({"perplexity1": perplexity1,
                   "perplexity2": perplexity2})

    logger.info("***** Dist-1,2,3 Eval results *****")
    logger.info("***** PPL Eval results *****")

    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

    return result



def generate_gen():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    model_args, data_args, train_args, gen_args = parser.parse_args_into_dataclasses()


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed
    set_seed(train_args.seed)
    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path)
    gpt2tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path)
    gpt2orgtokenizer = GPT2Tokenizer.from_pretrained('gpt2')


    special_tokens_dict = {
        "pad_token": "<|pad|>",
        "bos_token": "<|startoftext|>",
        # "additional_special_tokens": ['<|action|>', '<|romance|>', '<|horror|>', '<|crime|>'],
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    config = GPT2Config.from_pretrained(model_args.model_name_or_path)
    # set more attr #
    setattr(config, 'f_embd', 768)
    setattr(config, 'classifier_dropout', 0.1)
    setattr(config, 'temperature', model_args.tau)
    setattr(config, 'pad_token_id', tokenizer.pad_token_id)
    setattr(config, 'dropout_aug', data_args.dropout_aug)
    setattr(config, 'margin', model_args.margin)
    setattr(config, 'device', str(train_args.device))
    # setattr(config, 'margin_triplet', model_args.margin_triplet)
    setattr(config, 'loss_type', model_args.loss_type)
    setattr(config, 'in_batch_supervision', model_args.in_batch_supervision)
    setattr(config, 'vocab_size', len(tokenizer))

    # logger.info(config)

    model = SupConGPT2(
        config=config,
    )
    # using pretrained gpt2 model
    # model.lm_model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
    model = SupConGPT2.from_pretrained(model_args.model_name_or_path, config=config)
    model = model.to(train_args.device)
    # to evaluate ppl
    gpt2model = GPT2LMHeadModel.from_pretrained('models/gpt2_210919').to(train_args.device)

    model.lm_model.resize_token_embeddings(len(tokenizer))
    # issue https://github.com/huggingface/transformers/issues/8039
    unk_tok_emb = model.lm_model.transformer.wte.weight.data[tokenizer.unk_token_id, :]
    for i in range(num_added_toks):
        model.lm_model.transformer.wte.weight.data[-(i + 1), :] = unk_tok_emb

    # print(f"wte shape {model.lm_model.transformer.wte.weight.shape}")

    logger.info(f"SCL WEIGHT {model_args.scl_weight}")

    eval_dataset_gen, eval_dataset_ppl = load_and_cache_examples_eval(data_args, tokenizer)
    # logger.info("***** Running evaluation *****")

    if train_args.n_gpu > 1:  # case of dist training
        results, generated_sentences = evaluate(model.module, gpt2model, tokenizer, gpt2tokenizer, eval_dataset_gen,
                                                eval_dataset_ppl, data_args, train_args, gen_args,
                                                epoch='last')
    else:
        results, generated_sentences = evaluate(model, gpt2model, tokenizer, gpt2tokenizer, eval_dataset_gen,
                                                eval_dataset_ppl, data_args, train_args, gen_args,
                                                epoch='last')

    for key, value in sorted(results.items()):
        logger.info(f"  {key} = {value}")
            # wandb.log({f"{key}": value})

    return train_args.output_dir, train_args.num_train_epochs


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
                'crime': 3, }
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
                              max_seq_length=512, )
    model_args = ModelArguments(num_labels=4,
                                model_name_or_path="cls_models/hall_of_fame/roberta_210623_1991_4/")
    train_args = TrainingArguments(output_dir=output_dir,
                                   do_train=False,
                                   do_eval=False,
                                   do_predict=True,
                                   per_device_eval_batch_size=8,
                                   overwrite_output_dir=True,
                                   report_to=None)

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
        # print(p.predictions)
        # preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # preds = 각 데이터 샘플마다 (num_labels) 만큼의 array 나옴
        # label_ids = p.label_ids[0] if isinstance(p.label_ids, tuple) else p.label_ids
        # label_ids = [p[0] for p in label_ids] if isinstance(label_ids[0], list) else label_ids
        preds = np.argmax(p.predictions, axis=1).tolist()
        label_ids = list(p.label_ids)

        # return metric.compute(predictions=preds, references=label_ids, average='macro')
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


#
if __name__ == '__main__':
    # output_dir, num_train_epochs = train_gen()
    _, _ = generate_gen()
    output_dir = "./outputs/1105_scl_kl_evalset/"
    num_train_epochs = 3
    results = evaluate_gen(output_dir, num_train_epochs)

    output_eval_file = os.path.join(output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in sorted(results.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")