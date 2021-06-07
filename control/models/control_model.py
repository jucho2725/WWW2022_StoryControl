import torch
import torch.nn as nn
from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from ._supcon_model import GPT2SupConModel
from ._lmhead_model import GPT2LMHeadModel

class SupConGPT2(GPT2PreTrainedModel):
    def __init__(self, path, config):
        super(SupConGPT2, self).__init__(config)

        # self.model = GPT2Model.from_pretrained(path, config=config)
        # self.generater = GPT2LMHeadModel(config, self.model)
        # self.encoder = GPT2SupConModel(config, self.model)

        self.generater = GPT2LMHeadModel.from_pretrained(path, config=config)
        self.encoder = GPT2SupConModel(config, self.generater.transformer)
        self.init_weights()
        # self.model._init_weights(self.encoder._encoder.encoder_head.dense)
        # self.model._init_weights(self.encoder._encoder.encoder_head.out_proj)


    def resize_token_embeddings(self, new_size):
        # self.model.resize_token_embeddings(new_size)
        self.generater.resize_token_embeddings(new_size)
        # self.encoder._encoder.resize_token_embeddings(new_size)

    def forward(self, batch):        
        # nll loss from generater
        batch_org = batch['origin']

        gen_output = self.generater(**batch_org)
        nll_loss = gen_output.loss

        # scl loss from encoder
        batch_de, batch_ru = batch['aug_de'], batch['aug_ru']
        encoder_output = self.encoder(batch_de, batch_ru, labels=batch['labels'])
        scl_loss = encoder_output.loss

        return nll_loss, scl_loss

        
        return


