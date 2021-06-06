import torch
import torch.nn as nn
from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from ._supcon_model import GPT2SupConModel
from ._lmhead_model import GPT2LMHeadModel

class SupConGPT2(GPT2PreTrainedModel):
    def __init__(self, config):
        super(SupConGPT2, self).__init__(config)
        # setattr(config, 'f_embd', 128)
        # setattr(config, 'classifier_dropout', 0.9)
        # setattr(config, 'temperature', 0.04)
        self.model = GPT2Model(config)
        self.generater = GPT2LMHeadModel(config, self.model)
        self.encoder = GPT2SupConModel(config, self.model)

        self.init_weights()
        # self.model._init_weights(self.encoder._encoder.encoder_head.dense)
        # self.model._init_weights(self.encoder._encoder.encoder_head.out_proj)
        self.mode = "train"

    def resize_token_embeddings(self, new_size):
        self.model.resize_token_embeddings(new_size)
        self.generater.resize_token_embeddings(new_size)
        self.encoder.resize_token_embeddings(new_size)

    def forward(self, batch):        
        if self.mode == "train":
            # nll loss from generater
            # batch_org = batch['origin']

            # gen_output = self.generater(**batch_org)
            # nll_loss = gen_output.loss
            nll_loss = 0

            # scl loss from encoder
            batch_de, batch_ru = batch['aug_de'], batch['aug_ru']
            encoder_output = self.encoder(batch_de, batch_ru, labels=batch['labels'])
            scl_loss = encoder_output.loss

            return nll_loss, scl_loss

        elif self.mode == "eval":

            # TASK 2
            task_batch = batch['task2']
            task2_output = self._bart_classification(task_batch['input_ids'], attention_mask=task_batch['attention_mask'], labels=task_batch['target'])
            task2_logits = task2_output.logits
            # 0번에 대해서만. 
            # pred_idx = torch.argmax(task2_logits[:, 1])
            pred_idx = torch.argmax(torch.sigmoid(task2_logits))
            # TASK 3
            task_batch = self.pick_goldsample(batch['task3'], retrieve_idx=[pred_idx])
            task3_output = self._bart_generation(task_batch['input_ids'], attention_mask=task_batch['attention_mask'], labels=task_batch['target'])
            
            return pred_idx, task3_output.logits
        else:
            raise NotImplementedError
        
        return


