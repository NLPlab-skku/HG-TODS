import json, os
import dgl
import torch
from bartenc import bartenc_conceptnet
from tabulate import tabulate


class cg_utils():
    def __init__(self):
        self.head_map = None
        self.tail_map = None

    def _concept_to_cgraph(self):
        data, relations = bartenc_conceptnet.read_data()
        dialogue_w_slot, conceptnet = data['dialogue_w_slot'], data['conceptnet']
        slots = bartenc_conceptnet.get_slots(dialogue_w_slot)
        triples = bartenc_conceptnet.get_triple(slots, conceptnet)
        inputs, tokenizer, heads, tails = bartenc_conceptnet.make_input(triples)
        self.triples = triples

        '''
        new dialogue slots:
        {'song title': '더러운 세상아',
            'singer': '',
            'genre': '',
            'composer': '',
            'lyricist': '',
            'playlist type': '',
            'playlist title': '',
            'recommendation': '',
            'topic': '',
            'concept': ''}
        '''
        head_map, edge_map, edge_list, tail_map = {}, {}, {}, {}
        
        for ind, k in enumerate(relations.values()):
            edge_map[k] = ind * 2
            edge_map[str(k + '_inv')] = ind * 2 + 1
            edge_list[k] = [] 
            edge_list[str(k + '_inv')] = []

        for ind, k in enumerate(heads):
            head_map[k] = ind 
        for ind, k in enumerate(tails):
            tail_map[k] = ind 

        # edge : [[node1, node2], [node1, node2]] 형태로 구성
        # 각 relation마다 추가
        for inp in inputs:
            head, tail, rel = inp['triple']

            edge_list[rel].append([head_map[head], tail_map[tail]])
            edge_list[str(rel + '_inv')].append([tail_map[tail], head_map[head]])

        data_dict = {}
        for rel in edge_map.keys():
            if 'inv' in rel:
                data_dict[('tail', rel, 'head')] = edge_list[rel]
            else:
                data_dict[('head', rel, 'tail')] = edge_list[rel]
        data_dict[('dummy', 'dummy', 'dummy')] = [[0, 0]]
        
        num_nodes_dict = {
            'head' : len(head_map),
            'tail' : len(tail_map),
            'dummy' : 1
        }

        g = dgl.heterograph(data_dict, num_nodes_dict = num_nodes_dict)
        g = dgl.to_simple(g)

        table = [
            {
                '# Head Nodes' : g.num_nodes('head'),
                '# Tail Nodes' : g.num_nodes('tail'),
            }
        ]
        print(tabulate(table, headers = 'keys'))

        g.ndata['label'] = {
            'head' : torch.tensor([i for i in range(len(head_map))]),
            'tail' : torch.tensor([i for i in range(len(tail_map))]),
            'dummy' : torch.tensor([0])
        }

        self.head_map = head_map
        self.head_map_swap = {v: k for k, v in head_map.items()}
        self.tail_map = tail_map
        self.tail_map_swap = {v: k for k, v in tail_map.items()}

        # initialize node embedding
        bartenc_embedding = torch.load(f"/workspace/NRF/cache/bartenc_embedding_conceptnet")

        assert g.num_nodes('head') == bartenc_embedding['head_embedding'].shape[0]
        assert g.num_nodes('tail') == bartenc_embedding['tail_embedding'].shape[0]

        g.ndata['emb'] = {
            'head' : bartenc_embedding['head_embedding'],
            'tail' : bartenc_embedding['tail_embedding'],
            'dummy' : torch.zeros((1, bartenc_embedding['head_embedding'].shape[1]), dtype = bartenc_embedding['head_embedding'].dtype)
        }

        relation_embedding = {}
        edge_embedding = {}

        for ind, rel in enumerate(edge_map.keys()):
            relation_embedding[rel] = bartenc_embedding['relation_embedding'][int(ind//2), :]
            if 'inv' not in rel:
                edge_embedding[('head', rel, 'tail')] = relation_embedding[rel].unsqueeze(0).expand(g.num_edges(('head', rel, 'tail')), -1)
            else:
                edge_embedding[('tail', rel, 'head')] = relation_embedding[rel].unsqueeze(0).expand(g.num_edges(('tail', rel, 'head')), -1)
        edge_embedding[('dummy', 'dummy', 'dummy')] = torch.zeros((1, bartenc_embedding['head_embedding'].shape[1]), dtype = bartenc_embedding['head_embedding'].dtype)
        
        g.edata['emb'] = edge_embedding
        self.concept_graph = g

        return g


    ###
    def _slot_to_subgraph_concept(self, dial_id, utter_num):
        
        '''
            "user_slots": {
                    "song title": "Ransomware (Feat. Moldy)",
                    "singer": "",
                    "genre": "",
                    "composer": "",
                    "lyricist": "",
                    "playlist type": "",
                    "playlist title": "",
                    "recommendation": "",
                    "topic": "",
                    "concept": ""
                }
        '''
        head_map = self.head_map
        head_map_swap = self.head_map_swap
        tail_map_swap = self.tail_map_swap

        # 원래 데이터에서 dial id에 맞는 slot 저장해놓음
        triple = self.triples[dial_id]
        slot = []

        max_turn = utter_num + 1 if utter_num < len(triple['user_slots']['topic']) else len(triple['user_slots']['topic'])
        for i in range(max_turn):

            utter = triple['user_slots']['topic'][i]
            if utter != None: slot.append(utter[0][0])

            utter = triple['user_slots']['concept'][i]
            if utter != None: slot.append(utter[0][0])

            utter = triple['system_slots']['topic'][i]
            if utter != None: slot.append(utter[0][0])

            utter = triple['system_slots']['concept'][i]
            if utter != None: slot.append(utter[0][0])

        slot = list(set(slot))
        slot_key = [head_map[head] for head in slot if head in head_map]

        if slot_key == []:
            sg, _ = dgl.khop_out_subgraph(self.concept_graph, {'dummy' : [0]}, k = 1)
            # from IPython import embed; embed(); exit()
            return sg, slot
        
        else:
            sg, _ = dgl.khop_out_subgraph(self.concept_graph, {'head' : slot_key}, k = 1)
            # ret_slot += [head_map_swap[head] for head in sg.ndata['label']['head'].tolist() if head in head_map_swap] 
            # ret_slot += [tail_map_swap[tail] for tail in sg.ndata['label']['tail'].tolist() if tail in tail_map_swap]
            return sg, slot

            