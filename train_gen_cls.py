import logging
import os
from dataclasses import asdict
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler
from control.data_collator import DataCollatorForLanguageModeling
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
from control.data_collator import DataCollatorForSCL

from control.evaluation import (
    evaluate_ppl,
    evaluate_dist_scores
)

from apex import amp
from torch_optimizer import Lamb
import wandb
import pandas as pd

os.environ['WANDB_MODE'] = 'offline'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate(model, tokenizer, eval_dataset, data_args, model_args, train_args, gen_args, epoch):
    # ppl
    results = {}

    result = evaluate_ppl(data_args, train_args, model, tokenizer, eval_dataset)
    results.update(result)

    result = evaluate_dist_scores(data_args, train_args, gen_args, model, tokenizer, eval_dataset, epoch=epoch)
    results.update(result)
    return results


# evaluate perplexity
def evaluate_ppl(data_args, train_args, model, tokenizer, eval_dataset):
    eval_output_dir = train_args.output_dir

    if train_args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=train_args.eval_batch_size, collate_fn=data_collator
    )

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", train_args.eval_batch_size)
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

    eval_output_filename = "eval_result.txt"
    output_eval_file = os.path.join(eval_output_dir, eval_output_filename)

    logger.info("***** PPL Eval results *****")
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


# evaluate Dist-K scores
def evaluate_dist_scores(data_args, train_args, gen_args, model, tokenizer, eval_dataset, epoch="none"):
    eval_output_dir = train_args.output_dir
    if train_args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=train_args.eval_batch_size, collate_fn=data_collator
    )
    generated_sequences = []

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", train_args.eval_batch_size)
    model.eval()

    dist_eval_samples = []
    num_tokens = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        if not train_args.no_cuda:
            batch = {k: v.to(train_args.device) for k, v in batch.items()}

        with torch.no_grad():
            output_sequences = model.lm_model.generate(input_ids=batch['input_ids'],
                                                        attention_mask=batch['attention_mask'],
                                                        max_length=data_args.max_seq_length,
                                                        **asdict(gen_args))
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            dist_eval_samples.extend(list(filter(lambda x: x != tokenizer.pad_token_id, output_sequences.tolist())))
            num_tokens += sum([len(output) for output in output_sequences.tolist()])

        generated_sequence_idx = 0
        for generated_sequence in output_sequences:
            generated_sequence = generated_sequence.tolist()
            # Decode text
            generated_sequences.append(clean_text(tokenizer.decode(generated_sequence,
                                                                   skip_special_tokens=False,
                                                                   clean_up_tokenization_spaces=True)))
            generated_sequence_idx += 1

    dist1_score = count_ngram(dist_eval_samples, 1) / float(num_tokens)
    dist2_score = count_ngram(dist_eval_samples, 2) / float(num_tokens)
    dist3_score = count_ngram(dist_eval_samples, 3) / float(num_tokens)

    result = {"Dist-1": dist1_score, "Dist-2": dist2_score, "Dist-3": dist3_score}

    logger.info("***** Dist-1,2,3 Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    save_path = os.path.join(train_args.output_dir, f"epoch_{epoch}")
    os.makedirs(save_path, exist_ok=True)
    print(f"save path is {save_path}")

    write_sent(generated_sequences, os.path.join(save_path, f"result_{epoch}.txt"))
    write_df(generated_sequences, data_args, os.path.join(save_path, f"result_{epoch}.tsv"))
    print("write sent df done")
    return result


def train(train_dataset, eval_dataset, tokenizer, model, optimizer, scheduler, data_args, model_args, train_args,
          gen_args):
    data_collator = DataCollatorForSCL(tokenizer)
    t_total = len(train_dataset) // train_args.gradient_accumulation_steps * train_args.num_train_epochs

    sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=sampler,
                                  batch_size=train_args.train_batch_size, collate_fn=data_collator)

    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", train_args.num_train_epochs)
    logger.info("  Total train batch size = %d", train_args.train_batch_size * train_args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", train_args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss = 0.0
    model.zero_grad()

    for now_epoch in tqdm(range(int(train_args.num_train_epochs)), desc="Epoch"):
        model.mode = 'train'
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            model.train()
            # cuda
            for task_key in batch:
                if isinstance(batch[task_key], dict):
                    for key in batch[task_key]:
                        batch[task_key][key] = batch[task_key][key].to(train_args.device)
                else:
                    batch[task_key] = batch[task_key].to(train_args.device)

            # forward
            generator_loss, encoder_loss = model(batch)

            # loss sum
            loss = model_args.scl_weight * encoder_loss + (1.0 - model_args.scl_weight) * generator_loss

            if train_args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if train_args.gradient_accumulation_steps > 1:
                loss = loss / train_args.gradient_accumulation_steps

            if train_args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % train_args.gradient_accumulation_steps == 0:
                if train_args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), train_args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if (step + 1) % 10 == 0:
                pass
                # wandb.log({"Train Sum Loss": loss.item()})
                # wandb.log({"Train NLL Loss": generator_loss.mean().item()})
                # wandb.log({"Train SCL Loss": encoder_loss.mean().item()})
                # wandb.log({'learning_rate': optimizer.param_groups[0]['lr']})

        save_path = os.path.join(train_args.output_dir, f"epoch_{now_epoch}/")
        os.makedirs(save_path, exist_ok=True)
        if train_args.n_gpu > 1:
            model.module.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        if train_args.do_eval and train_args.evaluation_strategy == 'epoch':
            results = {}
            if train_args.evaluation_metric == "ppl" or train_args.evaluation_metric == "both":
                logger.info(f"***** Running evaluation {train_args.evaluation_metric} *****")
                if train_args.n_gpu > 1:
                    result = evaluate_ppl(data_args, train_args, model.module, tokenizer, eval_dataset)
                else:
                    result = evaluate_ppl(data_args, train_args, model, tokenizer, eval_dataset)
                results.update(result)
            if train_args.evaluation_metric == "dist" or train_args.evaluation_metric == "both":
                logger.info(f"***** Running evaluation {train_args.evaluation_metric} *****")
                if train_args.n_gpu > 1:
                    result = evaluate_dist_scores(data_args, train_args, gen_args, model.module, tokenizer,
                                                  eval_dataset, epoch=str(int(now_epoch)))
                else:
                    result = evaluate_dist_scores(data_args, train_args, gen_args, model, tokenizer, eval_dataset,
                                                  epoch=str(int(now_epoch)))
                results.update(result)

            for key, value in sorted(results.items()):
                logger.info(f"  {key} = {value}")
                # wandb.log({f"{key}": value})

    # save the last model
    if train_args.n_gpu > 1:
        model.module.save_pretrained(train_args.output_dir)
    else:
        model.save_pretrained(train_args.output_dir)
    tokenizer.save_pretrained(train_args.output_dir)

    return global_step, tr_loss / global_step


def train_gen():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    model_args, data_args, train_args, gen_args = parser.parse_args_into_dataclasses()

    if data_args.hard_negative:
        df = pd.read_csv(filepath_or_buffer=data_args.train_data_file, sep='\t', index_col=False)
        assert 'content_neg' in df.columns, "You must include negatives in dataset"
        print("************************ HARD NEGATIVE data ************************")
    else:
        print("************************ NORMAL data ************************")
    #
    # if data_args.no_genre:
    #     setattr(train_args, 'output_dir', f"./outputs/gpt2_finetune")
    # else:
    #     setattr(train_args, 'output_dir',
    #             f"./outputs/scl{int(model_args.scl_weight * 100)}_tau{int(model_args.tau * 100)}")
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        train_args.local_rank,
        train_args.device,
        train_args.n_gpu,
        bool(train_args.local_rank != -1),
        train_args.fp16,
    )

    # Set seed
    set_seed(train_args.seed)
    tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path)

    special_tokens_dict = {
        "pad_token": "<|pad|>",
        "bos_token": "<|endoftext|>",
        # "additional_special_tokens": ['[MALE]', '[FEMALE]', '[NEUTRAL]'],
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    config = GPT2Config.from_pretrained(model_args.model_name_or_path)
    # set more attr #
    setattr(config, 'f_embd', 768)
    setattr(config, 'classifier_dropout', 0.0)
    setattr(config, 'temperature', model_args.tau)
    setattr(config, 'pad_token_id', tokenizer.pad_token_id)
    setattr(config, 'dropout_aug', model_args.dropout_aug)

    model = SupConGPT2(
        config=config,
    )
    # using pretrained gpt2 model
    model.lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(train_args.device)

    model.lm_model.resize_token_embeddings(len(tokenizer))
    # issue https://github.com/huggingface/transformers/issues/8039
    unk_tok_emb = model.lm_model.transformer.wte.weight.data[tokenizer.unk_token_id, :]
    for i in range(num_added_toks):
        model.lm_model.transformer.wte.weight.data[-(i + 1), :] = unk_tok_emb


    if train_args.do_train:
        logger.info("***** Load dataset *****")
        train_dataset, origin_dataset = load_and_cache_examples_train(data_args, tokenizer)
        if train_args.evaluation_first or train_args.do_eval or train_args.evaluation_metric:
            eval_dataset, origin_eval_dataset = load_and_cache_examples_eval(data_args, tokenizer)

        # train_dataset = train_dataset.select(range(20))
        # eval_dataset = eval_dataset.select(range(8))

        t_total = len(train_dataset) // train_args.gradient_accumulation_steps * train_args.num_train_epochs
        # print(tokenizer.decode(train_dataset[0]['origin']['input_ids'], skip_special_tokens=False,
        #                        clean_up_tokenization_spaces=True))
        logger.info("***** Load optimizer *****")
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": train_args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        optimizer = Lamb(optimizer_grouped_parameters, lr=train_args.learning_rate, eps=train_args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=train_args.warmup_steps, num_training_steps=t_total
        )

        logger.info("***** Prepare fp16 / multi-gpu setting *****")
        if train_args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=train_args.fp16_opt_level)
            torch.cuda.empty_cache()

        # multi-gpu training (should be after apex fp16 initialization)
        if train_args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if train_args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[train_args.local_rank], output_device=train_args.local_rank,
                find_unused_parameters=False,
            )
        # Load pretrained model and tokenizer
        if train_args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

        # weight and bias monitoring

        logger.info("***** Running training *****")
        logger.info(f"***** Genre training {not data_args.no_genre} *****")
        # wandb.init(project="aaai_storycontrol_scl_tau", name=f"scl_{model_args.scl_weight}_temp_{model_args.tau}")
        # wandb.init(project="aiide_storycontrol", name=f"0612_gpt2", resume=True)
        # wandb.watch(model, log_freq=20)
        if train_args.evaluation_first:
            logger.info("***** Running evaluation *****")
            if train_args.n_gpu > 1:  # case of dist training
                results = evaluate(model.module, tokenizer, eval_dataset, data_args, model_args, train_args, gen_args,
                                   epoch='none')
            else:
                results = evaluate(model, tokenizer, eval_dataset, data_args, model_args, train_args, gen_args,
                                   epoch='none')
            for key, value in sorted(results.items()):
                logger.info(f"  {key} = {value}")
                # wandb.log({f"{key}": value})

        global_step, tr_avg_loss = train(train_dataset, eval_dataset, tokenizer, model, optimizer, scheduler,
                                         data_args, model_args, train_args, gen_args, )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_avg_loss)

    if train_args.do_eval or train_args.do_predict:
        eval_dataset, origin_eval_dataset = load_and_cache_examples_eval(data_args, tokenizer)
        logger.info("***** Running evaluation *****")

        if train_args.n_gpu > 1:  # case of dist training
            results = evaluate(model.module, tokenizer, eval_dataset, data_args, model_args, train_args, gen_args,
                               epoch='last')
        else:
            results = evaluate(model, tokenizer, eval_dataset, data_args, model_args, train_args, gen_args,
                               epoch='last')

        # if train_args.do_eval:
        #     output_eval_file = os.path.join(train_args.output_dir, "eval_results.txt")
        # else:
        #     output_eval_file = os.path.join(train_args.output_dir, "scrap/pred_results.txt")

        logger.info("***** Eval results *****")
        for key, value in sorted(results.items()):
            logger.info(f"  {key} = {value}")


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
    df = pd.read_csv(filepath_or_buffer=data_path, sep='\t',
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
        # return metric.compute(predictions=preds, references=label_ids)


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
    result_dict = {'4-class': outputs.metrics["test_f1"]}
    logger.info("showing accuracy")
    for genre in label_to_int.keys():
        binary_prediction = np.array(list(map(lambda x: label_to_binary(x, selected_genre=label_to_int[genre]),
                                              np.argmax(outputs.predictions, axis=1))))
        binary_label_ids = np.array(
            list(map(lambda x: label_to_binary(x, selected_genre=label_to_int[genre]), outputs.label_ids)))
        results = acc_and_f1(binary_prediction, binary_label_ids)
        result_dict[genre] = results['f1']
    return result_dict

if __name__ == '__main__':
    # output_dir, num_train_epochs = train_gen()
    output_dir = "./outputs/ep3_neg_dropaug"
    num_train_epochs = 3
    results = evaluate_gen(output_dir, num_train_epochs)


    output_eval_file = os.path.join(output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in sorted(results.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")