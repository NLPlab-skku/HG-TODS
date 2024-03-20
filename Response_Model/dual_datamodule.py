import json, os
import torch
import dgl
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from transformers import PreTrainedTokenizerFast
from bartenc import bartenc_conceptnet
from util_song import *
from util_common import *

class NRFDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, i):
        return self.data[i]
    
    def __len__(self):
        return len(self.data)

class DataModule(LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.setup()

        if cfg.method == 'BART':
            self.collate_fn = self.collate_fn_vanilla
        elif cfg.method == 'BART-triple':
            self.collate_fn = self.collate_fn_triple
        if 'gnn' in cfg.method:
            self.collate_fn = self.collate_fn_bartenc
        if 'dual' in cfg.method:
            self.collate_fn = self.collate_fn_bartenc_conceptnet
            

    def _read(self):
        filename_list = ['train.json', 'valid.json', 'test.json', 'songs.json', 'personal.json', 'nodemap.json']

        def _load(filename):
            path = os.path.join(self.cfg.data_dir, filename)
            with open(path, 'r') as f:
                return json.load(f)

        output = {}

        for filename in filename_list:
            output[filename.split('.')[0]] = _load(filename)

        return output

    def setup(self, stage = None):
        output = self._read()
        self.data = {'train' : None, 'valid' : None, 'test' : None}
        self.title_map, self.person_map, _ = output['nodemap']
        self.title_list, self.person_list = list(self.title_map.keys()), list(self.person_map.keys())
        self.songs = output['songs']
        self.personal = output['personal']

        for split in ['train', 'valid', 'test']:
            split_data = output[split]

            if split != 'test':
                self.data[split] = self._dial_to_task(split_data)
            else:
                self.data[split] = self._dial_to_task_test(split_data)

        table = [
            {
                'train' : len(self.data['train']),
                'valid' : len(self.data['valid']),
                'test' : len(self.data['test'])
            }
        ]
        print(tabulate(table, headers = 'keys'))

        self.sg = sg_utils(self.title_map, self.person_map, self.title_list, self.person_list, self.personal, self.tokenizer)
        self.songs_graph = self.sg._songs_to_graph(self.songs)
        self.cg = cg_utils()
        self.cg._concept_to_cgraph()


    def _dial_to_task(self, data):
        dialogues = []

        for dial in data:
            if dial['cur_size'] == 0:
                continue

            dial_id = dial['id']
            user_id = dial['user_id']

            plain_history = []
            usr_utt = None
            sys_response = None

            for i, turn in enumerate(dial['turns']):
                usr_utt = f"사용자: {turn['user_message']}"
                sys_response = f"시스템: {turn['system_message']}"

                history = " ".join(plain_history[-self.cfg.max_history:]) + " " + usr_utt

                dialogues.append(
                    {
                        'dial_id' : dial_id,
                        'user_id' : user_id,
                        'history' : history,
                        'prev_slot' : turn['system_slots'],
                        'response' : self.tokenizer.bos_token + sys_response + self.tokenizer.eos_token,
                        'turn' : turn['turn_num']
                    }
                )
                plain_history.append(usr_utt)
                plain_history.append(sys_response)

        return dialogues
    

    def _dial_to_task_test(self, data):
        dialogues = []

        for dial in data:
            if dial['cur_size'] == 0:
                continue

            dial_id = dial['id']
            user_id = dial['user_id']

            plain_history = []
            usr_utt = None
            sys_response = None

            for i, turn in enumerate(dial['turns'][:-1]):
                usr_utt = f"사용자: {turn['user_message']}"
                sys_response = f"시스템: {turn['system_message']}"
                
                plain_history.append(usr_utt)
                plain_history.append(sys_response)

            last_turn = dial['turns'][-1]
            usr_utt = f"사용자: {last_turn['user_message']}"
            sys_response = f"시스템: {last_turn['system_message']}"

            history = " ".join(plain_history[-self.cfg.max_history:]) + " " + usr_utt

            dialogues.append(
                {
                    'dial_id' : dial_id,
                    'user_id' : user_id,
                    'history' : history,
                    'prev_slot' : last_turn['system_slots'],
                    'response' : self.tokenizer.bos_token + sys_response + self.tokenizer.eos_token,
                    'turn' : -1
                }
            )

        return dialogues



    def collate_fn_vanilla(self, batch):
        '''
            Collate_fn for vanilla (w/o graph) 
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [b[key] for b in batch]

        enc_output = self.tokenizer(
            batch_data['history'],
            padding = True,
            return_tensors = 'pt',
            truncation = False,
            add_special_tokens = False,
            return_attention_mask = True
        )

        dec_output = self.tokenizer(
            batch_data['response'],
            padding = True,
            return_tensors = 'pt',
            truncation = False,
            add_special_tokens = False,
            return_attention_mask = True
        )

        batch_data['enc_ids'], batch_data['enc_mask'] = enc_output['input_ids'][:, :512], enc_output['attention_mask'][:, :512]
        batch_data['dec_ids'], batch_data['dec_mask'] = dec_output['input_ids'], dec_output['attention_mask']
        label_ids = torch.cat((batch_data['dec_ids'][:, 1:], torch.full((batch_data['dec_ids'].shape[0], 1), self.tokenizer.pad_token_id)), dim = -1)
        label_ids = label_ids.masked_fill(label_ids == self.tokenizer.pad_token_id, -100)
        batch_data['label_ids'] = label_ids

        batch_data['g'] = None

        return batch_data

    def collate_fn_transe(self, batch):
        '''
            Collate_fn for graph (w/ transe)

            Inputs
                batch : [
                    {
                        'dial_id' : str
                        'user_id' : str
                        'history' : str
                        'response' : str
                    }
                ]
            Outputs
                batch : [
                    {
                        'dial_id' : str
                        'user_id' : str
                        'history' : str
                        'response' : str
                        'enc_ids' : torch.Tensor
                        'enc_mask' : torch.Tensor
                        'dec_ids' : torch.Tensor
                        'dec_mask' : torch.Tensor
                        'label_ids' : torch.Tensor
                    }
                ]
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [b[key] for b in batch]

        enc_output = self.tokenizer(
            batch_data['history'],
            padding = True,
            return_tensors = 'pt',
            truncation = False,
            add_special_tokens = False,
            return_attention_mask = True
        )

        dec_output = self.tokenizer(
            batch_data['response'],
            padding = True,
            return_tensors = 'pt',
            truncation = False,
            add_special_tokens = False,
            return_attention_mask = True
        )

        batch_data['enc_ids'], batch_data['enc_mask'] = enc_output['input_ids'], enc_output['attention_mask']
        batch_data['dec_ids'], batch_data['dec_mask'] = dec_output['input_ids'], dec_output['attention_mask']
        label_ids = torch.cat((batch_data['dec_ids'][:, 1:], torch.full((batch_data['dec_ids'].shape[0], 1), self.tokenizer.pad_token_id)), dim = -1)
        label_ids = label_ids.masked_fill(label_ids == self.tokenizer.pad_token_id, -100)
        batch_data['label_ids'] = label_ids
        
        sg_list = []
        for user_id, prev_slot in zip(batch_data['user_id'], batch_data['prev_slot']):
            sg = self.sg._slot_to_subgraph(user_id, prev_slot)
            sg = dgl.to_homogeneous(sg, ndata = ['emb', 'label'], edata = ['emb'])
            sg_list.append(sg)

        batched_sg = dgl.batch(sg_list)

        batch_data['g'] = batched_sg
            
        return batch_data


    def collate_fn_triple(self, batch):
        '''
            Collate_fn for graph (w/ usage of triple)
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [b[key] for b in batch]

        history_triple_list = []
        for history, user_id, prev_slot in zip(batch_data['history'], batch_data['user_id'], batch_data['prev_slot']):
            sg = self.sg._slot_to_subgraph(user_id, prev_slot)
            history_triple_list.append(history + self.tokenizer.sep_token + self.sg._sg2triples(sg)) # triple 처리

        enc_output = self.tokenizer(
            history_triple_list,
            padding = True,
            return_tensors = 'pt',
            truncation = False,
            add_special_tokens = False,
            return_attention_mask = True
        )

        dec_output = self.tokenizer(
            batch_data['response'],
            padding = True,
            return_tensors = 'pt',
            truncation = False,
            add_special_tokens = False,
            return_attention_mask = True
        )

        batch_data['enc_ids'], batch_data['enc_mask'] = enc_output['input_ids'][:, :512], enc_output['attention_mask'][:, :512]
        batch_data['dec_ids'], batch_data['dec_mask'] = dec_output['input_ids'], dec_output['attention_mask']
        label_ids = torch.cat((batch_data['dec_ids'][:, 1:], torch.full((batch_data['dec_ids'].shape[0], 1), self.tokenizer.pad_token_id)), dim = -1)
        label_ids = label_ids.masked_fill(label_ids == self.tokenizer.pad_token_id, -100)
        batch_data['label_ids'] = label_ids

        batch_data['g'] = None

        return batch_data

    def collate_fn_bartenc(self, batch):
        '''
            Collate_fn for graph (w/ bart encoder embedding)

            Inputs
                batch : [
                    {
                        'dial_id' : str
                        'user_id' : str
                        'history' : str
                        'response' : str
                    }
                ]
            Outputs
                batch : [
                    {
                        'dial_id' : str
                        'user_id' : str
                        'history' : str
                        'response' : str
                        'enc_ids' : torch.Tensor
                        'enc_mask' : torch.Tensor
                        'dec_ids' : torch.Tensor
                        'dec_mask' : torch.Tensor
                        'label_ids' : torch.Tensor
                    }
                ]
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [b[key] for b in batch]

        history_triple_list = []
        for history, user_id, prev_slot in zip(batch_data['history'], batch_data['user_id'], batch_data['prev_slot']):
            sg = self.sg._slot_to_subgraph(user_id, prev_slot)
            history_triple_list.append(history + self.tokenizer.sep_token + self.sg._sg2triples(sg)) # triple 처리

        enc_output = self.tokenizer(
            history_triple_list,
            max_length = 512,
            padding = 'max_length',
            return_tensors = 'pt',
            truncation = True,
            add_special_tokens = False,
            return_attention_mask = True
        )

        dec_output = self.tokenizer(
            batch_data['response'],
            padding = True,
            return_tensors = 'pt',
            truncation = False,
            add_special_tokens = False,
            return_attention_mask = True
        )

        batch_data['enc_ids'], batch_data['enc_mask'] = enc_output['input_ids'][:, :512], enc_output['attention_mask'][:, :512]
        batch_data['dec_ids'], batch_data['dec_mask'] = dec_output['input_ids'], dec_output['attention_mask']
        label_ids = torch.cat((batch_data['dec_ids'][:, 1:], torch.full((batch_data['dec_ids'].shape[0], 1), self.tokenizer.pad_token_id)), dim = -1)
        label_ids = label_ids.masked_fill(label_ids == self.tokenizer.pad_token_id, -100)
        batch_data['label_ids'] = label_ids
        
        sg_list = []
        for user_id, prev_slot in zip(batch_data['user_id'], batch_data['prev_slot']):
            sg = self.sg._slot_to_subgraph(user_id, prev_slot)
            sg = dgl.to_homogeneous(sg, ndata = ['emb', 'label'], edata = ['emb'])
            sg_list.append(sg)

        batched_sg = dgl.batch(sg_list)
        batch_data['g'] = batched_sg

        return batch_data


    def collate_fn_bartenc_conceptnet(self, batch):
        
        '''
            Collate_fn for graph (w/ bart encoder embedding)

            Inputs
                batch : [
                    {
                        'dial_id' : str
                        'user_id' : str
                        'history' : str
                        'response' : str
                        'turn' : int
                    }
                ]
            Outputs
                batch : [
                    {
                        'dial_id' : str
                        'user_id' : str
                        'history' : str
                        'response' : str
                        'enc_ids' : torch.Tensor
                        'enc_mask' : torch.Tensor
                        'dec_ids' : torch.Tensor
                        'dec_mask' : torch.Tensor
                        'label_ids' : torch.Tensor
                    }
                ]
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [b[key] for b in batch]
            
        sg_concept_list = []
        slots_list = []
        for dial_id, utter_num in zip(batch_data['dial_id'], batch_data['turn']):         
            try:
                sg_concept, slot_concept = self.cg._slot_to_subgraph_concept(dial_id, utter_num)
                sg_concept = dgl.to_homogeneous(sg_concept, ndata = ['emb', 'label'], edata = ['emb'])
                sg_concept_list.append(sg_concept)
                slots_list.append(slot_concept)
            except:
                print('error', dial_id, utter_num)
        batched_sg_concept = dgl.batch(sg_concept_list)
        batch_data['cg'] = batched_sg_concept

        history_triple_list = []
        for i, (history, user_id, prev_slot) in enumerate(zip(batch_data['history'], batch_data['user_id'], batch_data['prev_slot'])):
            sg = self.sg._slot_to_subgraph(user_id, prev_slot)
            history_triple_list.append(history + self.tokenizer.sep_token + self.sg._sg2triples(sg) + f"{self.tokenizer.sep_token}".join(slots_list[i])  ) # triple 처리

        enc_output = self.tokenizer(
            history_triple_list,
            padding = True,
            return_tensors = 'pt',
            truncation = False,
            add_special_tokens = False,
            return_attention_mask = True
        )

        dec_output = self.tokenizer(
            batch_data['response'],
            padding = True,
            return_tensors = 'pt',
            truncation = False,
            add_special_tokens = False,
            return_attention_mask = True
        )

        batch_data['enc_ids'], batch_data['enc_mask'] = enc_output['input_ids'][:, :512], enc_output['attention_mask'][:, :512]
        batch_data['dec_ids'], batch_data['dec_mask'] = dec_output['input_ids'], dec_output['attention_mask']
        label_ids = torch.cat((batch_data['dec_ids'][:, 1:], torch.full((batch_data['dec_ids'].shape[0], 1), self.tokenizer.pad_token_id)), dim = -1)
        label_ids = label_ids.masked_fill(label_ids == self.tokenizer.pad_token_id, -100)
        batch_data['label_ids'] = label_ids

        sg_list = []
        for user_id, prev_slot in zip(batch_data['user_id'], batch_data['prev_slot']):
            sg = self.sg._slot_to_subgraph(user_id, prev_slot)
            sg = dgl.to_homogeneous(sg, ndata = ['emb', 'label'], edata = ['emb'])
            sg_list.append(sg)

        batched_sg = dgl.batch(sg_list)
        batch_data['g'] = batched_sg
        return batch_data



    def train_dataloader(self):
        dataloader = DataLoader(
            NRFDataset(self.data['train']),
            batch_size = self.cfg.batch_size,
            shuffle = True,
            collate_fn = self.collate_fn,
            num_workers=0
        )

        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            NRFDataset(self.data['valid']),
            batch_size = self.cfg.batch_size,
            shuffle = True,
            collate_fn = self.collate_fn,
            num_workers=0
        )

        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            NRFDataset(self.data['test']),
            batch_size = self.cfg.batch_size,
            shuffle = False,
            collate_fn = self.collate_fn,
            num_workers=0
        )

        return dataloader

if __name__ == '__main__':
    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    cfg = {
        'data_dir' : '/NRF/data',
        'batch_size' : 32,
        'max_history' : 100,
        'generate_max_length' : 64,
        'd_model' : 768
    }

    cfg = dotdict(cfg)

    tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')

    datamodule = DataModule(cfg, tokenizer)