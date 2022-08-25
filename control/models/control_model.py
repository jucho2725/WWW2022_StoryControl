
from .lmhead_model import GPT2LMHeadModel
from dataclasses import dataclass
from typing import Optional
from transformers.file_utils import (
    ModelOutput,
)
from transformers import (
    # GPT2LMHeadModel,
    GPT2PreTrainedModel
)
from .losses import ContrastiveLoss, align_loss, uniform_loss, TripletLoss, CELoss 


import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SupConGPT2(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lm_model = GPT2LMHeadModel(config=config)
        self.encoder_head = GPT2EncoderHead(config.n_embd,
                                            config.n_embd,
                                            config.f_embd,  # feature embedding
                                            config.classifier_dropout)
        # self.init_weights()
        self.lm_model.transformer._init_weights(self.encoder_head.dense)
        self.lm_model.transformer._init_weights(self.encoder_head.out_proj)


        if config.loss_type == 'hadsell':
            logger.info(f"*** margin / in_batch_supervision {config.in_batch_supervision}")
            self.criterion = ContrastiveLoss(margin=config.margin, in_batch_supervision=config.in_batch_supervision)
        elif config.loss_type == 'triplet':
            logger.info(f"*** triplet / in_batch_supervision {config.in_batch_supervision}")
            self.criterion = TripletLoss(margin=config.margin, in_batch_supervision=config.in_batch_supervision)
        elif config.loss_type == 'cross_entropy':
            logger.info(f"*** CE / in_batch_supervision {config.in_batch_supervision}")
            self.criterion = CELoss(temp=0.05, device=config.device, in_batch_supervision= config.in_batch_supervision)


    def encode(self, input_ids=None, past_key_values=None, attention_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
            use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.lm_model.transformer(
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
        hidden_state = transformer_outputs.last_hidden_state
        batch_size, sequence_length = input_ids.shape[:2]

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
        pooled_logits = hidden_state[range(batch_size), sequence_lengths]  # 마지막 토큰 풀링
        feature = self.encoder_head(pooled_logits)
        feature = F.normalize(feature, dim=1)
        return feature


    def encoder_forward(self, batch_input1, batch_input2, labels, batch_input3=None, neg_labels=None):
        # None None of not None not None
        assert (batch_input3 is not None and neg_labels is not None) or (batch_input3 is None and neg_labels is None)

        f_pos1 = self.encode(**batch_input1)
        f_pos2 = self.encode(**batch_input2)

        if batch_input3 is None and neg_labels is None:
            features = torch.cat([f_pos1.unsqueeze(1),
                                  f_pos2.unsqueeze(1),],
                                 dim=1)
            loss = self.criterion(features, labels)

            return EncoderOutputs(
                scl_loss=loss,
                align_loss=align_loss(f_pos1, f_pos2),
                uniform_loss=uniform_loss(torch.cat((f_pos1, f_pos2), dim=0)),
                feature_pos1=f_pos1,
                feature_pos2=f_pos2,
            )
        else:
            f_neg = self.encode(**batch_input3)
            features = torch.cat([f_pos1.unsqueeze(1),
                                  f_pos2.unsqueeze(1),
                                  f_neg.unsqueeze(1)], dim=1)

            loss = self.criterion(features, labels, neg_labels)

            # loss = self.criterion(features)
            return EncoderOutputs(
                scl_loss=loss,
                align_loss=align_loss(f_pos1, f_pos2),
                uniform_loss=uniform_loss(torch.cat((f_pos1, f_pos2, f_neg), dim=0)),
                feature_pos1=f_pos1,
                feature_pos2=f_pos2,
                feature_neg = f_neg
            )

    def resize_token_embeddings(self, new_size):
        self.lm_model.resize_token_embeddings(new_size)
        # self.encoder._encoder.resize_token_embeddings(new_size)

    def forward(self, batch):        
        # nll loss from lm_model
        batch_org = batch['origin']

        gen_output = self.lm_model(**batch_org)
        nll_loss = gen_output.loss

        # scl loss from encoder


        if 'aug_neg' not in batch.keys():
            batch_09, batch_05 = batch['aug_09'], batch['aug_05']
            encoder_output = self.encoder_forward(batch_09, batch_05, labels=batch['labels'])
        else:
            batch_09, batch_05, batch_neg = batch['aug_09'], batch['aug_05'], batch['aug_neg']
            encoder_output = self.encoder_forward(batch_09, batch_05, labels=batch['labels'], batch_input3=batch_neg, neg_labels=batch['neg_labels'])


        return ControlModelOutputs(
                nll_loss=nll_loss,
                scl_loss=encoder_output.scl_loss,
                align_loss=encoder_output.align_loss,
                uniform_loss=encoder_output.uniform_loss,
                lm_logits=gen_output.logits
            )


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
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor):

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

@dataclass
class EncoderOutputs(ModelOutput):
    """
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`,
        feature_pos1 (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, )`):
        feature_pos2 (:obj:`tuple(tupel(torch.FloatTensor))`, `optional`,
        feature_neg
    """
    scl_loss: torch.FloatTensor = None
    align_loss: torch.FloatTensor = None
    uniform_loss: torch.FloatTensor = None
    feature_pos1: Optional[torch.FloatTensor] = None
    feature_pos2: Optional[torch.FloatTensor] = None
    feature_neg: Optional[torch.FloatTensor] = None

@dataclass
class ControlModelOutputs(ModelOutput):
    """
    Args:

    """
    nll_loss: torch.FloatTensor = None
    scl_loss: torch.FloatTensor = None
    align_loss: torch.FloatTensor = None
    uniform_loss: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None