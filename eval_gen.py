import logging
import os
from typing import Optional, Tuple


from tqdm.auto import tqdm
import torch

from transformers import (
    HfArgumentParser,
)

from control.models import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer
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
from control.evaluation import (
    evaluate_ppl,
    evaluate_dist_scores
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}

# def train():
#     for step, batch in epoch_iterator:
#         # forward
#         losses = model(batch)
#         task1_loss, task2_loss, task3_loss = losses
#
#         # loss sum
#         loss = 0
#         for task_loss in losses:
#             loss += task_loss

def evaluate(data_args, model_args, train_args, gen_args, model, tokenizer):
    # ppl
    results = {}

    model = model.from_pretrained(model_args.model_name_or_path)
    global_step = 0
    model.to(train_args.device)

    result = evaluate_ppl(data_args, train_args, model, tokenizer)
    result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
    results.update(result)

    result = evaluate_dist_scores(data_args, train_args, gen_args, model, tokenizer)
    result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
    results.update(result)
    return results

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    model_args, data_args, train_args, gen_args = parser.parse_args_into_dataclasses()

    # Setup CUDA, GPU & distributed training
    if train_args.local_rank == -1 or train_args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not train_args.no_cuda else "cpu")
        # setattr(train_args, 'n_gpu', torch.cuda.device_count())
        # train_args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(train_args.local_rank)
        device = torch.device("cuda", train_args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        train_args.n_gpu = 1
    # train_args.device = device
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        train_args.local_rank,
        device,
        train_args.n_gpu,
        bool(train_args.local_rank != -1),
        train_args.fp16,
    )

    # Set seed
    set_seed(train_args.seed)


    # Load pretrained model and tokenizer
    if train_args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    
    config = config_class.from_pretrained(model_args.model_name_or_path)
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )
    model.to(train_args.device)
    tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)
    # wandb.watch(model)

    results = evaluate(data_args, model_args, train_args, gen_args, model, tokenizer)

    output_eval_file = os.path.join(train_args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(results.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

if __name__ == '__main__':
    main()