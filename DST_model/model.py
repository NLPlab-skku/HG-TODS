import os
from torch import nn
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from transformers import BertTokenizer, BertModel
from transformers import BartForConditionalGeneration


class KoBART(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_class = BartForConditionalGeneration
        self.model = self.model_class.from_pretrained(get_pytorch_kobart_model())

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, \
                past_key_values=None, labels=None, decoder_attention_mask=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, \
                            decoder_input_ids=decoder_input_ids, labels=labels, \
                            past_key_values=past_key_values, decoder_attention_mask=decoder_attention_mask)

        return output

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        
        


if __name__ == "__main__":
    model = KoBART()

    model.from_pretrained()
