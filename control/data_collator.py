from transformers import PreTrainedTokenizerBase, BatchEncoding
from dataclasses import dataclass
from typing import Optional, List, Dict, Union
import torch

from torch.nn.utils.rnn import pad_sequence
from typing import List

@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.

    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.

        batch = {}
        batch["input_ids"] = self.pad([ex['input_ids'] for ex in examples], self.tokenizer.pad_token_id)
        batch["attention_mask"] = self.pad([ex['attention_mask'] for ex in examples])

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    def pad(self, examples: List[torch.Tensor], padding_value=0):
        if self.tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=padding_value)

@dataclass
class DataCollatorForGeneration:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        

        # 10tokens 로 truncation 한거라 pad 필요없겠지만 혹시모르니 해두기
        batch = {}
        batch["input_ids"] = self.pad([ex['input_ids'] for ex in batch], self.tokenizer.pad_token_id)
        batch["attention_mask"] = self.pad([ex['attention_mask'] for ex in batch])

    def pad(self, examples: List[torch.Tensor], padding_value=0):
        if self.tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=padding_value)

@dataclass
class DataCollatorForSCL:
    """
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, batch):

        org_id = [ex['origin']['input_ids'] for ex in batch]
        org_attn = [ex['origin']['attention_mask'] for ex in batch]
        aug_09_id = [ex['aug_09']['input_ids'] for ex in batch]
        aug_09_attn = [ex['aug_09']['attention_mask'] for ex in batch]
        aug_05_id = [ex['aug_05']['input_ids'] for ex in batch]
        aug_05_attn = [ex['aug_05']['attention_mask'] for ex in batch]

        if 'aug_neg' in batch[0].keys():
            aug_neg_id = [ex['aug_neg']['input_ids'] for ex in batch]
            aug_neg_attn = [ex['aug_neg']['attention_mask'] for ex in batch]
            collated_batch = {'origin': {},
                                'aug_09': {},
                                'aug_05': {},
                              'aug_neg': {}}
            collated_batch['aug_neg']['input_ids']      = self.pad(aug_neg_id, self.tokenizer.pad_token_id)
            collated_batch['aug_neg']['attention_mask'] = self.pad(aug_neg_attn)
            collated_batch['neg_labels'] = torch.stack([ex['neg_labels'] for ex in batch])
        else:
            collated_batch = {'origin': {},
                                'aug_09': {},
                                'aug_05': {},}

        collated_batch['origin']['input_ids']      = self.pad(org_id, self.tokenizer.pad_token_id)
        collated_batch['origin']['attention_mask'] = self.pad(org_attn)

        # for language modeling, you need to change pad_token_id to -100 to prevent from computing loss
        labels =  collated_batch['origin']['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        collated_batch['origin']["labels"] = labels

        collated_batch['aug_09']['input_ids']      = self.pad(aug_09_id, self.tokenizer.pad_token_id)
        collated_batch['aug_09']['attention_mask'] = self.pad(aug_09_attn)
        collated_batch['aug_05']['input_ids']      = self.pad(aug_05_id, self.tokenizer.pad_token_id)
        collated_batch['aug_05']['attention_mask'] = self.pad(aug_05_attn)

        collated_batch['labels'] = torch.stack([ex['labels'] for ex in batch])

        return collated_batch

    def pad(self, examples: List[torch.Tensor], padding_value=0):
        if self.tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=padding_value)

