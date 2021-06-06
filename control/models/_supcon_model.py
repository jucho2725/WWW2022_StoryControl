
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers import (
    GPT2PreTrainedModel
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

from ._supcon_loss import SupConLoss, ContrastiveLoss
        
class GPT2SupConModel(GPT2PreTrainedModel):
    def __init__(self, config, model):
        super(GPT2SupConModel, self).__init__(config)
        self.model = model
        self._encoder = GPT2ForSequenceEncoder(config, model)
        # self._criterion = SupConLoss(temperature=config.temperature)
        self._criterion = ContrastiveLoss()


    def forward(self, batch_input1, batch_input2, labels):
        f1 = self._encoder(**batch_input1)
        f2 = self._encoder(**batch_input2)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = self._criterion(features, labels)

        return SequenceOutput(
            loss=loss,
            feature1=f1,
            feature2=f2,
        )

@dataclass
class SequenceOutput(ModelOutput):
    """
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, 
        feature1 (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, )`):
        feature2 (:obj:`tuple(tupel(torch.FloatTensor))`, `optional`,
    """

    loss: Optional[torch.FloatTensor] = None
    feature1: torch.FloatTensor = None
    feature2: torch.FloatTensor = None

class GPT2ForSequenceEncoder(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config, model):
        super().__init__(config)
        self.config=config
        self.transformer = model
        self.encoder_head =GPT2EncoderHead(config.n_embd,
                                           config.n_embd,
                                           config.f_embd, # feature embedding 
                                           config.classifier_dropout)

        # self.init_weights()
        self.transformer._init_weights(self.encoder_head.dense)
        self.transformer._init_weights(self.encoder_head.out_proj)
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print("trans out", transformer_outputs.last_hidden_state.shape)
        hidden_state = transformer_outputs.last_hidden_state
        # print(hidden_state.shape)
        batch_size, sequence_length = input_ids.shape[:2]
        # print(input_ids.shape[:2])
        # print(self.config.pad_token_id)
        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        pooled_logits = hidden_state[range(batch_size), sequence_lengths]
        # print("hidden", hidden_state.shape)
        # print(pooled_logits.shape, range(batch_size), sequence_lengths)
        feature = self.encoder_head(pooled_logits)
        # print(feature.shape)
        feature = F.normalize(feature, dim=1)
        # print(feature.shape)
        # print(여기)
        return feature


class GPT2EncoderHead(nn.Module):
    """Head for sentence-level encoding."""
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        output_dim: int,
        pooler_dropout: float,
    ):
        super().__init__()
        # print("inp inner out")
        # print(input_dim, inner_dim, output_dim)
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor):

        hidden_states = self.dropout(hidden_states)
        try:
            hidden_states = self.dense(hidden_states)
        except Exception as e:
            print(e)
            print("="* 80)
            print(hidden_states.shape)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

