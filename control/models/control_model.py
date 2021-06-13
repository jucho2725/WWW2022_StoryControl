from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from ._supcon_model import GPT2SupConModel
from ._lmhead_model import GPT2LMHeadModel

class SupConGPT2(GPT2PreTrainedModel):
    def __init__(self, path, config):
        super(SupConGPT2, self).__init__(config)
        self.generater = GPT2LMHeadModel.from_pretrained(path, config=config)
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
        batch_09, batch_05 = batch['aug_09'], batch['aug_05']
        encoder_output = self.encoder(batch_09, batch_05, labels=batch['labels'])
        scl_loss = encoder_output.loss

        return nll_loss, scl_loss

