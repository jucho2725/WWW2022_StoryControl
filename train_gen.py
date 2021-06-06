import logging
import os
import time
from typing import Optional, Tuple


from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.backends.cudnn as cudnn

from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    HfArgumentParser,
    get_linear_schedule_with_warmup
)

from control.models import (
    SupConGPT2,
)

from control.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments
)
from control.utils import (
    set_seed,
)
from control.dataset import load_and_cache_examples_train

from control.data_collator import DataCollatorForSCL

from control.evaluation import (
    evaluate_ppl,
    evaluate_dist_scores
)

from apex import amp
from torch_optimizer import Lamb
import wandb


logger = logging.getLogger(__name__)


def evaluate(model, tokenizer, data_args, model_args, train_args, gen_args):
    # ppl
    results = {}

    model = model.from_pretrained(model_args.model_name_or_path)
    model.to(train_args.device)

    result = evaluate_ppl(data_args, train_args, model, tokenizer)
    results.update(result)

    result = evaluate_dist_scores(data_args, train_args, gen_args, model, tokenizer)
    results.update(result)
    return results


def train(train_dataset, tokenizer, model, optimizer, scheduler, data_args, model_args, train_args, gen_args):
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
                wandb.log({"Train Sum Loss": loss.item()})
                wandb.log({"Train NLL Loss": generator_loss.mean().item()})
                wandb.log({"Train SCL Loss": encoder_loss.mean().item()})
                wandb.log({'learning_rate': optimizer.param_groups[0]['lr']})

        model.module.save_pretrained(train_args.output_dir)
        if train_args.evaluation_strategy == "epoch":
            results = {}
            if train_args.evaluation_metric == "ppl" or train_args.evaluation_metric == "both":
                result = evaluate_ppl(data_args, train_args, model, tokenizer)
                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)
            if train_args.evaluation_metric == "dist" or train_args.evaluation_metric == "both":
                result = evaluate_dist_scores(data_args, train_args, gen_args, model, tokenizer)
                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)

            for key, value in sorted(results.metrics.items()):
                logger.info(f"  {key} = {value}")
                wandb.log({f"{key}": value})

    # save the last model
    model.module.save_pretrained(train_args.output_dir)

    return global_step, tr_loss / global_step

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    model_args, data_args, train_args, gen_args = parser.parse_args_into_dataclasses()
    
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
    tokenizer.pad_token = tokenizer.eos_token # gpt2 does not have pad token at first.
    config = GPT2Config.from_pretrained(model_args.model_name_or_path)
    # set more attr #
    setattr(config, 'f_embd', 768)
    setattr(config, 'classifier_dropout', 0.0)
    setattr(config, 'temperature', 0.1)
    setattr(config, 'pad_token_id', tokenizer.pad_token_id)

    model = SupConGPT2.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )
    model = model.to(train_args.device)

    if train_args.do_train:
        logger.info("***** Load dataset *****")
        train_dataset, origin_dataset = load_and_cache_examples_train(data_args, tokenizer)
        t_total = len(train_dataset) // train_args.gradient_accumulation_steps * train_args.num_train_epochs

        logger.info("***** Load optimizer *****")
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": train_args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        
        optimizer = Lamb(optimizer_grouped_parameters,  lr=train_args.learning_rate, eps=train_args.adam_epsilon)
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
                model, device_ids=[train_args.local_rank], output_device=train_args.local_rank, find_unused_parameters=False,
            )
        # Load pretrained model and tokenizer
        if train_args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab
        
        # weight and bias monitoring
        wandb.init(project="aiide_storycontrol", name=f"scl_{model_args.scl_weight}")
        wandb.watch(model, log_freq=20)
        logger.info("***** Running training *****")
        global_step, tr_avg_loss = train(train_dataset, tokenizer, model, optimizer, scheduler,
                                        data_args, model_args, train_args, gen_args, )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_avg_loss)


    if train_args.do_eval:
        logger.info("***** Running evaluation *****")
        results = evaluate(model.generator, tokenizer, data_args, model_args, train_args, gen_args,)
        output_eval_file = os.path.join(train_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in sorted(results.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()