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

os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_MODE'] = 'offline'
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate(model, gpt2model, tokenizer, gpt2tokenizer, eval_dataset_gen, eval_dataset_ppl, data_args, train_args, gen_args, epoch):
    # ppl
    results = {}

    _ = evaluate_ppl(train_args, model, tokenizer, eval_dataset_ppl)
    generated_sentences = generate_sentences(data_args, train_args, gen_args, model, tokenizer, eval_dataset_gen, epoch=epoch)
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
            return torch.tensor(self.examples[i][self.examples[i].index(self.tokenizer.bos_token_id) + 1:],
                                dtype=torch.long)
        except ValueError:
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
def evaluate_ppl_dist(generated_sequences, tokenizer, train_args, gpt2model,):

    dist_eval_samples = []
    num_tokens = 0

    dist_eval_dataset = TempDataset(tokenizer, generated_sequences)


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


def train(train_dataset, eval_dataset_gen, eval_dataset_ppl, tokenizer, gpt2tokenizer, model, gpt2model, optimizer, scheduler, data_args, model_args, train_args,
          gen_args):
    data_collator = DataCollatorForSCL(tokenizer)
    t_total = len(train_dataset) // train_args.gradient_accumulation_steps * train_args.num_train_epochs

    # fix generator object to make batch equal on different train seed.
    gen_obj = torch.Generator()
    gen_obj.manual_seed(2021)

    sampler = RandomSampler(train_dataset, generator=gen_obj)
    train_dataloader = DataLoader(train_dataset, sampler=sampler,
                                  batch_size=train_args.train_batch_size, collate_fn=data_collator)

    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", train_args.num_train_epochs)
    # logger.info("  Total train batch size = %d", train_args.train_batch_size * train_args.gradient_accumulation_steps)
    # logger.info("  Gradient Accumulation steps = %d", train_args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", t_total)

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
            outputs = model(batch)
            generator_loss, encoder_loss = outputs.nll_loss, outputs.scl_loss

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
                # logger.info(f"Train Sum Loss: {loss.item()}")
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
            if train_args.n_gpu > 1:  # case of dist training
                results = evaluate(model.module, gpt2model, tokenizer, gpt2tokenizer, eval_dataset_gen, eval_dataset_ppl, data_args, model_args, train_args, gen_args,
                                   epoch='none')
            else:
                results = evaluate(model, gpt2model, tokenizer, gpt2tokenizer, eval_dataset_gen, eval_dataset_ppl, data_args, model_args, train_args, gen_args,
                                   epoch='none')

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
    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #     train_args.local_rank,
    #     train_args.device,
    #     train_args.n_gpu,
    #     bool(train_args.local_rank != -1),
    #     train_args.fp16,
    # )

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
    setattr(config, 'loss_type', model_args.loss_type)
    setattr(config, 'in_batch_supervision', model_args.in_batch_supervision)
    setattr(config, 'vocab_size', len(tokenizer))

    # logger.info(config)

    model = SupConGPT2(
        config=config,
    )
    # using pretrained gpt2 model
    model.lm_model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
    # model = SupConGPT2.from_pretrained(model_args.model_name_or_path, config=config)
    model = model.to(train_args.device)
    # to evaluate ppl
    gpt2model = GPT2LMHeadModel.from_pretrained('models/gpt2_210919').to(train_args.device)
    

    model.lm_model.resize_token_embeddings(len(tokenizer))
    # issue https://github.com/huggingface/transformers/issues/8039
    unk_tok_emb = model.lm_model.transformer.wte.weight.data[tokenizer.unk_token_id, :]
    for i in range(num_added_toks):
        model.lm_model.transformer.wte.weight.data[-(i + 1), :] = unk_tok_emb

    logger.info(f"SCL WEIGHT {model_args.scl_weight}")
    if train_args.do_train:
        # logger.info("***** Load dataset *****")
        train_dataset, origin_dataset = load_and_cache_examples_train(data_args, tokenizer)
        # logger.info(f"train input example: { tokenizer.decode(train_dataset[0]['origin']['input_ids'])}")
        # print("train label example", tokenizer.decode(train_dataset[0]['origin']['labels']))
        if train_args.evaluation_first or train_args.do_eval or train_args.evaluation_metric:
            eval_dataset_gen, eval_dataset_ppl = load_and_cache_examples_eval(data_args, tokenizer)
            # logger.info(f"eval input example: {tokenizer.decode(eval_dataset_ppl[0]['input_ids'])}")
            # logger.info(f"eval label example: {tokenizer.decode(eval_dataset[0]['labels'])}")
        # train_dataset = train_dataset.select(range(20))
        # eval_dataset = eval_dataset.select(range(8))

        t_total = len(train_dataset) // train_args.gradient_accumulation_steps * train_args.num_train_epochs
        # print(tokenizer.decode(train_dataset[0]['origin']['input_ids'], skip_special_tokens=False,
        #                        clean_up_tokenization_spaces=True))
        # logger.info("***** Load optimizer *****")
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

        # optimizer = Lamb(optimizer_grouped_parameters, lr=train_args.learning_rate, eps=train_args.adam_epsilon)
        optimizer = AdamW(optimizer_grouped_parameters, lr=train_args.learning_rate, weight_decay=train_args.weight_decay,
                          betas=(train_args.adam_beta1, train_args.adam_beta2),
                          eps=train_args.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=train_args.warmup_steps, num_training_steps=t_total
        )

        # logger.info("***** Prepare fp16 / multi-gpu setting *****")
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

        # logger.info("***** Running training *****")
        # logger.info(f"***** Genre training {not data_args.no_genre} *****")
          # wandb.init(project="www_storycontrol", name=f"0913scl_lr{train_args.learning_rate}_weight_{train_args.weight_decay}",
        #            resume=True)
        # wandb.watch(model, log_freq=20)
        if train_args.evaluation_first:
            # logger.info("***** Running evaluation *****")
            if train_args.n_gpu > 1:  # case of dist training
                results = evaluate(model.module, gpt2model, tokenizer, gpt2tokenizer, eval_dataset_gen, eval_dataset_ppl, data_args, train_args, gen_args,
                                   epoch='none')
            else:
                results = evaluate(model, gpt2model, tokenizer, gpt2tokenizer, eval_dataset_gen, eval_dataset_ppl, data_args, train_args, gen_args,
                                   epoch='none')
            for key, value in sorted(results.items()):
                logger.info(f"  {key} = {value}")
                # wandb.log({f"{key}": value})

        global_step, tr_avg_loss = train(train_dataset, eval_dataset_gen, eval_dataset_ppl, tokenizer, gpt2tokenizer, model, gpt2model, optimizer, scheduler,
                                         data_args, model_args, train_args, gen_args, )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_avg_loss)

    if train_args.do_eval or train_args.do_predict:
        eval_dataset_gen, eval_dataset_ppl = load_and_cache_examples_eval(data_args, tokenizer)
        # logger.info("***** Running evaluation *****")

        if train_args.n_gpu > 1:  # case of dist training
            results, generated_sentences = evaluate(model.module, gpt2model, tokenizer, gpt2tokenizer, eval_dataset_gen, eval_dataset_ppl, data_args, train_args, gen_args,
                               epoch='last')
        else:
            results, generated_sentences = evaluate(model, gpt2model, tokenizer, gpt2tokenizer, eval_dataset_gen, eval_dataset_ppl, data_args, train_args, gen_args,
                               epoch='last')

        # if train_args.do_eval:
        #     output_eval_file = os.path.join(train_args.output_dir, "eval_results.txt")
        # else:
        #     output_eval_file = os.path.join(train_args.output_dir, "scrap/pred_results.txt")

        logger.info("***** Eval results *****")
        for key, value in sorted(results.items()):
            logger.info(f"  {key} = {value}")

    
    gpt2model = GPT2LMHeadModel.from_pretrained('gpt2').to(train_args.device)
    special_tokens_dict = {
        "pad_token": "<|pad|>",
        "bos_token": "<|startoftext|>",
        # "additional_special_tokens": ['<|action|>', '<|romance|>', '<|horror|>', '<|crime|>'],
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    gpt2model.resize_token_embeddings(len(tokenizer))
    # issue https://github.com/huggingface/transformers/issues/8039
    unk_tok_emb = gpt2model.transformer.wte.weight.data[tokenizer.unk_token_id, :]
    for i in range(num_added_toks):
        gpt2model.transformer.wte.weight.data[-(i + 1), :] = unk_tok_emb

    results = evaluate_ppl_dist(generated_sentences, tokenizer, train_args, gpt2model)
    logger.info("***** Eval results for origin gpt2 *****")
    for key, value in sorted(results.items()):
        logger.info(f"  {key} = {value}")

    return train_args.output_dir, train_args.num_train_epochs

if __name__ == '__main__':
    output_dir, num_train_epochs = train_gen()