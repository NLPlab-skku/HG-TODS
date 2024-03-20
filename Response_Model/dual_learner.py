import json, os
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torchmetrics.text.rouge import ROUGEScore
from model.bart import BartForConditionalGeneration
from model.graph_bart import GraphBartForConditionalGeneration
from model.dual_graph_bart import DualGraphBartForConditionalGeneration

class Learner(LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        if 'gnn' in cfg.method:
            model = GraphBartForConditionalGeneration(cfg, tokenizer)
        else:
            model = BartForConditionalGeneration(cfg, tokenizer)
        if 'dual' in cfg.method:
            model = DualGraphBartForConditionalGeneration(cfg, tokenizer)

        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        output = self.model(
            enc_ids = batch['enc_ids'],
            enc_mask = batch['enc_mask'],
            g = batch['g'],
            cg = batch['cg'],
            dec_ids = batch['dec_ids'],
            dec_mask = batch['dec_mask'],
            label_ids = batch['label_ids']
        )
        loss = output['lm_loss']

        self.log('train_loss', loss, prog_bar = True)

        return {
            'loss' : loss
        }

    def validation_step(self, batch, batch_idx):
        import pdb; pdb.set_trace()
        output = self.model(
            enc_ids = batch['enc_ids'],
            enc_mask = batch['enc_mask'],
            g = batch['g'],
            cg = batch['cg'],
            dec_ids = batch['dec_ids'],
            dec_mask = batch['dec_mask'],
            label_ids = batch['label_ids']
        )
        loss = output['lm_loss']

        self.validation_step_outputs.append(loss)

        return {
            'loss' : loss
        }

    def on_validation_epoch_end(self):
        val_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        self.log('val_loss', val_loss, prog_bar = True)

    def test_step(self, batch, batch_idx):
        output = self.model.generate(
            enc_ids = batch['enc_ids'],
            enc_mask = batch['enc_mask'],
            g = batch['g'],
            cg = batch['cg'],
        )

        ret_output = {
            'enc_ids' : batch['enc_ids'].tolist(),
            'golden_response' : batch['response'],
            'generated_response' : output['generated_outputs']
        }

        self.test_step_outputs.append(ret_output)
        return ret_output

    def on_test_epoch_end(self):
        rouge = ROUGEScore(rouge_keys = ('rouge1', 'rouge2', 'rougeL'), use_stemmer = True)

        golden_response_list, generated_response_list = [], []
        score = {}

        for output in self.test_step_outputs:
            golden_response_list += [r.replace(self.tokenizer.bos_token, "").replace(self.tokenizer.eos_token, "") for r in output['golden_response']]
            generated_response_list += output['generated_response']

        rouge_output = rouge(golden_response_list, generated_response_list)
        
        score['ROUGE-1'] = round(float(rouge_output['rouge1_fmeasure']) * 100, 2)
        score['ROUGE-2'] = round(float(rouge_output['rouge2_fmeasure']) * 100, 2)
        score['ROUGE-L'] = round(float(rouge_output['rougeL_fmeasure']) * 100, 2)

        for key, value in score.items():
            self.log(key, value, prog_bar = True)

        with open(os.path.join(self.cfg.experiment_dir, 'golden.json'), 'w') as f:
            f.write(json.dumps(golden_response_list))

        with open(os.path.join(self.cfg.experiment_dir, 'generated.json'), 'w') as f:
            f.write(json.dumps(generated_response_list, ensure_ascii=False))

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr = self.cfg.learning_rate)