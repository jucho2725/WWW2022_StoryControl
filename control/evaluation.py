import os
from typing import Dict, List, Tuple
from typing import List
from torch.utils.data import SequentialSampler, DataLoader
from control.data_collator import DataCollatorForLanguageModeling

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .dataset import load_and_cache_examples_eval
from .utils import clean_text, write_sent
import logging

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}



# evaluate perplexity
def evaluate_ppl(data_args, train_args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict:
    eval_output_dir = train_args.output_dir
    eval_dataset, origin_eval_dataset = load_and_cache_examples_eval(data_args, tokenizer, evaluate=train_args.do_eval)

    if train_args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=train_args.eval_batch_size, collate_fn=data_collator
    )

    # Eval
    logger.info("***** Running evaluation {} *****".format(prefix))
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
            outputs = model(**batch)
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
def evaluate_dist_scores(data_args, train_args, gen_args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,) -> Dict:
    eval_output_dir = train_args.output_dir
    eval_dataset, origin_eval_dataset = load_and_cache_examples_eval(data_args, tokenizer, evaluate=train_args.do_eval)

    if train_args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=train_args.eval_batch_size, collate_fn=data_collator
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
            output_sequences = model.generate(input_ids=batch['input_ids'],
                                              attention_mask=batch['attention_mask'],
                                              max_length=data_args.max_seq_length,
                                              temperature=gen_args.temperature,
                                              top_k=gen_args.top_k,
                                              top_p=gen_args.top_p,
                                              pad_token_id=tokenizer.eos_token_id,
                                              eos_token_id=tokenizer.eos_token_id,
                                              repetition_penalty=1.1,
                                              do_sample=gen_args.do_sample,
                                              num_return_sequences=gen_args.num_return_sequences, )
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            dist_eval_samples.extend(output_sequences.tolist())
            num_tokens += sum([len(output) for output in output_sequences.tolist()])

        prompt_texts = [clean_text(inp) for inp in tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False,
                                                                          clean_up_tokenization_spaces=True)]

        generated_sequence_idx = 0
        for generated_sequence, prompt_text in zip(output_sequences, prompt_texts):
            print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            generated_sequences.append(clean_text(tokenizer.decode(generated_sequence,
                                                                   skip_special_tokens=False, clean_up_tokenization_spaces=True)))
            generated_sequence_idx += 1

    dist1_score = count_ngram(dist_eval_samples, 1) / float(num_tokens)
    dist2_score = count_ngram(dist_eval_samples, 2) / float(num_tokens)
    dist3_score = count_ngram(dist_eval_samples, 3) / float(num_tokens)

    result = {"Dist-1": dist1_score, "Dist-2": dist2_score, "Dist-3": dist3_score}

    logger.info("***** Dist-1,2,3 Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))


    write_sent(generated_sequences, os.path.join(train_args.output_dir, "result_beam.txt"))
    return result
