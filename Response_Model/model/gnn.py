# -*- coding:utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GraphConv, GATConv, EGATConv

class MkHidden(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        
    def forward(self, g):
        self.device = g.device
        batch_size = len(dgl.unbatch(g))
                
        graph_hidden_states = torch.zeros((batch_size, self.cfg.node_len, self.d_model), dtype = torch.float, device = self.device)
        graph_attention_mask = torch.zeros((batch_size, self.cfg.node_len), dtype = torch.long, device = self.device)
        for i, u_g in enumerate(dgl.unbatch(g)):
            num_nodes = u_g.num_nodes()
            if num_nodes == 1: # dummy
                continue

            graph_hidden_states[i, :num_nodes, :] = u_g.ndata['emb'][:512, :]
            graph_attention_mask[i, :num_nodes] = 1

        return graph_hidden_states, graph_attention_mask

class GCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.GCNLayers = nn.ModuleList([GraphConv(cfg.d_model, cfg.d_model) for _ in range(cfg.num_gnn - 1)])
        self.last_gcnlayer = GraphConv(cfg.d_model, cfg.d_model)

    def forward(self, g):
        '''
            Inputs
                g : dgl.Graph
            Outputs
                graph_hidden_states.shape = (B, E_L, H)
                graph_attention_mask.shape = (B, E_L)
        '''
        h = g.ndata['emb']

        for gcnlayer in self.GCNLayers:
            prev_h = h
            h = gcnlayer(g, h)
            h = F.elu(h)
            h = prev_h + h

        h = h + self.last_gcnlayer(g, h)

        g.ndata['emb'] = h

        return g

class GAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        head_outdim = int(cfg.d_model / 8)
        self.GATLayers = nn.ModuleList([GATConv(cfg.d_model, head_outdim, 8, 0.1, 0.1) for _ in range(cfg.num_gnn - 1)])
        self.last_gatlayer = GATConv(cfg.d_model, cfg.d_model, 8, 0.1, 0.1)

    def forward(self, g):
        '''
            Inputs
                g : dgl.Graph
            Outputs
                graph_hidden_states.shape = (B, E_L, H)
                graph_attention_mask.shape = (B, E_L)
        '''
        h = g.ndata['emb']

        for gatlayer in self.GATLayers:
            prev_h = h
            temp = gatlayer(g, h)
            h = F.elu(torch.cat([temp[:, i, :] for i in range(temp.shape[1])], dim = -1))
            h = prev_h + h
        
        prev_h = h
        temp = self.last_gatlayer(g, h)
        h = torch.mean(temp, dim = 1)
        h = prev_h + h

        g.ndata['emb'] = h

        return g

class RGAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_outdim = cfg.d_model // 8
        self.RGATLayers = nn.ModuleList([EGATConv(cfg.d_model, cfg.d_model, head_outdim, head_outdim, 8) for _ in range(cfg.num_gnn - 1)])
        self.last_rgatlayer = EGATConv(cfg.d_model, cfg.d_model, cfg.d_model, cfg.d_model, 8)

    def forward(self, g):
        n = g.ndata['emb']
        f = g.edata['emb']

        for rgatlayer in self.RGATLayers:
            prev_n, prev_f = n, f
            temp_n, temp_f = rgatlayer(g, n, f)
            n = F.elu(torch.cat([temp_n[:, i, :] for i in range(temp_n.shape[1])], dim = -1))
            f = F.elu(torch.cat([temp_f[:, i, :] for i in range(temp_f.shape[1])], dim = -1))
            n, f = prev_n + n, prev_f + f

        prev_n, prev_f = n, f
        temp_n, temp_f = self.last_rgatlayer(g, n, f)
        n, f = torch.mean(temp_n, dim = 1), torch.mean(temp_f, dim = 1)
        n, f = prev_n + n, prev_f + f

        g.ndata['emb'] = n
        g.edata['emb'] = f

        return g

