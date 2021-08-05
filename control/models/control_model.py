from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from ._supcon_model import GPT2SupConModel
from ._lmhead_model import GPT2LMHeadModel

class SupConGPT2(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.generater = GPT2LMHeadModel(config)
        self.encoder = GPT2SupConModel(config, self.generater.transformer)

    def resize_token_embeddings(self, new_size):
        self.generater.resize_token_embeddings(new_size)
        # self.encoder._encoder.resize_token_embeddings(new_size)

    def forward(self, batch):        
        # nll loss from generater
        batch_org = batch['origin']

        gen_output = self.generater(**batch_org)
        nll_loss = gen_output.loss

        # scl loss from encoder
        if 'aug_neg' not in batch.keys():
            batch_09, batch_05 = batch['aug_09'], batch['aug_05']
            encoder_output = self.encoder(batch_09, batch_05, labels=batch['labels'])
        else:
            batch_09, batch_05, batch_neg = batch['aug_09'], batch['aug_05'], batch['aug_neg']
            encoder_output = self.encoder(batch_09, batch_05, labels=batch['labels'], batch_input3=batch_neg, neg_labels=batch['neg_labels'])

        # print(f"{batch_org['input_ids'].shape} {batch_09['input_ids'].shape} {batch_05['input_ids'].shape} {batch_neg['input_ids'].shape}")
        # scl loss from encoder
        # if 'aug_neg' not in batch.keys():
        #     batch_05 = batch['aug_05']
        #     encoder_output = self.encoder(batch_org, batch_05, labels=batch['labels'])
        # else:
        #     batch_05, batch_neg = batch['aug_05'], batch['aug_neg']
        #     encoder_output = self.encoder(batch_org, batch_05, labels=batch['labels'], batch_input3=batch_neg, neg_labels=batch['neg_labels'])
        #
        # # scl loss from encoder
        # if 'aug_neg' not in batch.keys():
        #     batch_org, batch_05 = batch['origin'], batch['aug_05']
        #     encoder_output = self.encoder(batch_org, batch_05, labels=batch['labels'])
        # else:
        #     batch_org, batch_05, batch_neg = batch['origin'], batch['aug_05'], batch['aug_neg']
        #     encoder_output = self.encoder(batch_org, batch_05, labels=batch['labels'], batch_input3=batch_neg, neg_labels=batch['neg_labels'])

        scl_loss = encoder_output.loss

        return nll_loss, scl_loss

