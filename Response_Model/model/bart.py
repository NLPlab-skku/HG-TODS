# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from model.gnn import *

def _expand_mask(mask, tgt_len = None):
    """
        Inputs
            mask.shape = (B, S_L)
        Outputs
            output.shape = (B, 1, T_L, S_L)
    """
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(torch.float)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(torch.float).min)

def _make_causal_mask(dec_ids, past_key_values_length: int = 0):
    """
        Inputs
            dec_ids.shape = (B, D_L) or (B, 1)
    """
    batch_size, tgt_len = dec_ids.size()
    device = dec_ids.device

    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(torch.float).to(device)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=torch.float, device = device), mask], dim=-1)
    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len + past_key_values_length)

class PositionalEmbedding(nn.Embedding):
    def __init__(self, cfg):
        self.offset = 2
        super().__init__(cfg.max_position_embeddings + self.offset, cfg.d_model)

    def forward(self, input_ids_shape, past_key_values_length = 0):
        batch_size, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + int(seq_len), dtype = torch.long, device = self.weight.device
        )
        return super().forward(positions + self.offset)

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, is_decoder = False, is_cross_attention = False):
        super().__init__()
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.attention_dropout = cfg.attention_dropout
        self.d_head = self.d_model // self.num_heads
        self.scaling = self.d_head ** -0.5
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2).contiguous()

    def forward(self, query_states, key_value_states, past_key_value = None, attention_mask = None):
        batch_size, tgt_len, d_model = query_states.size()
        _, src_len, _ = key_value_states.size()

        query_states = self._shape(self.q_proj(query_states) * self.scaling, tgt_len, batch_size)
        if self.is_cross_attention and past_key_value is not None:
            # Encoder key, value
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif self.is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states), -1, batch_size)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states), -1, batch_size)
            key_states = torch.cat([past_key_value[0], key_states], dim = 2)
            value_states = torch.cat([past_key_value[1], value_states], dim = 2)
        else:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states), -1, batch_size)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (batch_size * self.num_heads, -1, self.d_head)
        query_states, key_states, value_states = query_states.view(*proj_shape), key_states.view(*proj_shape), value_states.view(*proj_shape)
        # query_states.shape = (B * num_heads, T_L, H // num_heads), key_states.shape = (B * num_heads, S_L, H // num_heads)

        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))
        # attn_weights.shape = (B * num_heads, T_L, S_L)
        
        if attention_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim = -1)

        attn_probs = F.dropout(attn_weights, p = self.attention_dropout, training = self.training)

        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.view(batch_size, self.num_heads, tgt_len, self.d_head)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, tgt_len, d_model)

        attn_output = self.out_proj(attn_output)

        return attn_output, past_key_value

class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model

        self.self_attn = MultiHeadAttention(cfg)
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.dropout = cfg.dropout
        self.activation_fn = nn.GELU()
        self.activation_dropout = cfg.activation_dropout

        self.fc1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.fc2 = nn.Linear(4 * self.d_model, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, hidden_states, enc_self_mask):
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            query_states = hidden_states,
            key_value_states = hidden_states,
            attention_mask = enc_self_mask
        )
        hidden_states = F.dropout(hidden_states, p = self.dropout, training = self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return (hidden_states, )

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model

        self.self_attn = MultiHeadAttention(cfg, is_decoder = True, is_cross_attention = False)
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.dropout = cfg.dropout
        self.activation_fn = nn.GELU()
        self.activation_dropout = cfg.activation_dropout

        self.encoder_attn = MultiHeadAttention(cfg, is_decoder = True, is_cross_attention = True)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.fc1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.fc2 = nn.Linear(4 * self.d_model, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(
        self, 
        hidden_states, 
        dec_self_mask = None, 
        enc_hidden_states = None, 
        enc_dec_mask = None, 
        past_key_value = None
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        enc_attn_past_key_value = past_key_value[2:] if past_key_value is not None else None

        residual = hidden_states
        hidden_states, self_attn_present_key_value = self.self_attn(
            query_states = hidden_states,
            key_value_states = hidden_states,
            past_key_value = self_attn_past_key_value,
            attention_mask = dec_self_mask
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states, enc_attn_present_key_value = self.encoder_attn(
            query_states = hidden_states,
            key_value_states = enc_hidden_states,
            past_key_value = enc_attn_past_key_value,
            attention_mask = enc_dec_mask,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p = self.activation_dropout, training = self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        present_key_value = self_attn_present_key_value + enc_attn_present_key_value

        return (hidden_states, present_key_value)

class Encoder(nn.Module):
    def __init__(self, cfg, embed_tokens):
        super().__init__()
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        self.embed_tokens = embed_tokens

        self.embed_positions = PositionalEmbedding(cfg)

        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(self.d_model)

    def forward(self, enc_ids, enc_mask):
        token_embedding = self.embed_tokens(enc_ids)
        pos_embedding = self.embed_positions(enc_ids.shape)

        hidden_states = token_embedding + pos_embedding
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p = self.dropout, training = self.training)

        enc_self_mask = _expand_mask(enc_mask)
        # enc_self_mask.shape = (B, 1, E_L, E_L)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states, enc_self_mask)
            hidden_states = layer_outputs[0]

        return {
            'enc_hidden_states' : hidden_states
        }

class Decoder(nn.Module):
    def __init__(self, cfg, embed_tokens):
        super().__init__()
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        self.embed_tokens = embed_tokens
        self.embed_positions = PositionalEmbedding(cfg)

        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(self.d_model)

    def forward(self, dec_ids, dec_mask = None, enc_hidden_states = None, enc_mask = None, past_key_values = None):
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        token_embedding = self.embed_tokens(dec_ids)
        pos_embedding = self.embed_positions(dec_ids.shape, past_key_values_length)

        hidden_states = token_embedding + pos_embedding
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p = self.dropout, training = self.training)

        if dec_ids.shape[-1] == 1:
            dec_self_mask = None
        else:
            temp1 = _make_causal_mask(dec_ids)
            temp2 = _expand_mask(dec_mask)
            dec_self_mask = temp1 + temp2
        enc_dec_mask = _expand_mask(enc_mask, dec_ids.shape[-1])

        cache = ()

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(hidden_states, dec_self_mask, enc_hidden_states, enc_dec_mask, past_key_value)

            hidden_states = layer_outputs[0]
            cache += (layer_outputs[1],)
        
        past_key_values = cache

        return {
            'dec_hidden_states' : hidden_states,
            'past_key_values' : past_key_values
        }

class BartModel(nn.Module):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.embed_tokens = nn.Embedding(cfg.plm_vocab_size, cfg.d_model)
        self.embed_tokens.load_state_dict(torch.load('/workspace/NRF/cache/embed'))

        self.new_embed_tokens = self.resize_token_embeddings()

        self.encoder = Encoder(cfg, self.new_embed_tokens)
        self.decoder = Decoder(cfg, self.new_embed_tokens)

        self.load_plm()

    def load_plm(self):
        state_dict = torch.load('/workspace/NRF/cache/kobart')
        state_dict.pop('encoder.embed_tokens.weight')
        state_dict.pop('decoder.embed_tokens.weight')
        self.load_state_dict(state_dict, strict = False)

    def resize_token_embeddings(self):
        old_embeddings = self.embed_tokens
        
        new_embeddings = nn.Embedding(len(self.tokenizer), old_embeddings.weight.shape[1])
        new_embeddings.to(old_embeddings.weight.device, dtype = old_embeddings.weight.dtype)

        n = min(old_embeddings.weight.shape[0], new_embeddings.weight.shape[0])

        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask = None, enc_hidden_states = None, past_key_values = None):
        if enc_hidden_states is None:
            enc_outputs = self.encoder(enc_ids, enc_mask)
            enc_hidden_states = enc_outputs['enc_hidden_states']

        dec_outputs = self.decoder(dec_ids, dec_mask, enc_hidden_states, enc_mask, past_key_values)

        return {
            'enc_hidden_states' : enc_hidden_states,
            'dec_hidden_states' : dec_outputs['dec_hidden_states'],
            'past_key_values' : dec_outputs['past_key_values']
        }

class BartForConditionalGeneration(nn.Module):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.model = BartModel(cfg, tokenizer)
        self.lm_head = nn.Linear(cfg.d_model, self.vocab_size, bias = False)

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask = None, enc_hidden_states = None, past_key_values = None, label_ids = None, g = None, graph_hidden_states = None, graph_mask = None):
        transformer_outputs = self.model(enc_ids, enc_mask, dec_ids, dec_mask, enc_hidden_states, past_key_values)
        lm_logits = self.lm_head(transformer_outputs['dec_hidden_states'])
        lm_loss = None
        if label_ids is not None:
            criterion = nn.CrossEntropyLoss()
            lm_loss = criterion(lm_logits.view(-1, self.vocab_size), label_ids.view(-1))

        return {
            'enc_hidden_states' : transformer_outputs['enc_hidden_states'],
            'dec_hidden_states' : transformer_outputs['dec_hidden_states'],
            'past_key_values' : transformer_outputs['past_key_values'],
            'lm_logits' : lm_logits,
            'lm_loss' : lm_loss
        }

    def generate(self, enc_ids, enc_mask, g = None):
        batch_size = enc_ids.shape[0]
        device = enc_ids.device
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        outputs = []
        has_eos = torch.zeros(batch_size, dtype = torch.bool).to(device)

        dec_ids = torch.tensor([[bos_token_id]] * batch_size, dtype = torch.long, device = device)
        enc_hidden_states = None
        past_key_values = None

        for _ in range(self.cfg.generate_max_length):
            model_outputs = self.forward(
                enc_ids, enc_mask, dec_ids, enc_hidden_states = enc_hidden_states, past_key_values = past_key_values
            )

            new_token_ids = torch.argmax(model_outputs['lm_logits'][:, -1, :], dim = -1)

            has_eos = has_eos | (new_token_ids == eos_token_id)
            new_token_ids = new_token_ids.masked_fill(has_eos, eos_token_id)
            outputs.append(new_token_ids)

            dec_ids = new_token_ids.unsqueeze(-1)
            enc_hidden_states = model_outputs['enc_hidden_states']
            past_key_values = model_outputs['past_key_values']

            if torch.all(has_eos):
                break

        outputs = torch.stack(outputs, dim = -1).tolist()
        generated_outputs = []

        for output in outputs:
            generated_outputs.append(self.tokenizer.decode(output, skip_special_tokens = True))
        
        return {
            'generated_outputs' : generated_outputs
        }