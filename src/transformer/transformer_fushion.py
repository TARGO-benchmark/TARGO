"""Modified from DETR's transformer.py

- Cross encoder layer is similar to the decoder layers in Transformer, but
  updates both source and target features
- Added argument to control whether value has position embedding or not for
  TransformerEncoderLayer and TransformerDecoderLayer
- Decoder layer now keeps track of attention weights
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from linear_attention_transformer.linear_attention_transformer import SelfAttention

## quesitons
## 2 cross attention layers, what is the third one?
## how many layers? linear attention vs dot product attention


class TransformerFusion(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod',):
                #  ):
        super().__init__()

        self.attention_type = attention_type
        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) ## d_model: 32, nhead: 2
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)    ## d_model: 32, nhead: 2
        elif attention_type == 'linear_attention':
            self.self_attn = SelfAttention(dim=d_model, heads=nhead, causal=False)
            self.multihead_attn = SelfAttention(dim=d_model, heads=nhead, causal=False)
        else:
            raise NotImplementedError

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis
        # self.occ_exists = occ_exists

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def process_self_attention_input_tensor(self, input_tensor, input_tensor_pos, input_tensor_mask, input_tensor_key_padding_mask):
        self_attention_time = 0
        input_tensor2 = self.norm1(input_tensor)     ## Layer norm
        input_tensor2_w_pos = self.with_pos_embed(input_tensor2, input_tensor_pos)  ## Add positional embeddings
        q = k = input_tensor2_w_pos     ## Query and key are same for self attention, Shape: (N, BS, N_MODEL)

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        if self.attention_type == 'dot_prod':
            input_tensor2, satt_weights_input_tensor = self.self_attn(      ## attn_mask is aimed to mask the padding tokens
                q, k,
                value=input_tensor2_w_pos if self.sa_val_has_pos_emb else input_tensor,
                attn_mask=input_tensor_mask,    
                key_padding_mask=input_tensor_key_padding_mask
            )
        elif self.attention_type == 'linear_attention':
            assert self.sa_val_has_pos_emb, 'input tensor should have pos embeddings'
            input_tensor2 = self.self_attn(x=input_tensor2_w_pos, input_mask=input_tensor_key_padding_mask.T)

        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            self_attention_time += start.elapsed_time(end)

        input_tensor = input_tensor + self.dropout1(input_tensor2)      ## Add and norm

        return input_tensor, input_tensor2, self_attention_time, (satt_weights_input_tensor if self.attention_type == 'dot_prod' else None)

    def position_wise_feedforward(self, input_tensor):
        # layer norm
        tensor_norm = self.norm3(input_tensor)

        # linear transformation
        tensor_transformed = self.linear2(self.dropout(self.activation(self.linear1(tensor_norm))))

        ## update tensor
        updated_tensor = input_tensor + self.dropout3(tensor_transformed)

        return updated_tensor

    def forward(self, src, targ,
                    src_mask: Optional[Tensor] = None,
                    targ_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    targ_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    targ_pos: Optional[Tensor] = None,
                    time_file=None):
        
        # total_weights = {}
        # weights = {}
        # for i in range(layer_num):
        cross_attention_time = 0
        src, src2, self_attention_time_src, satt_weights_src = self.process_self_attention_input_tensor(src, src_pos, src_mask, src_key_padding_mask)   ## (N,BS,32)
        cross_attention_time += self_attention_time_src
        targ, targ2, self_attention_time_targ, satt_weights_targ = self.process_self_attention_input_tensor(targ, targ_pos, targ_mask, targ_key_padding_mask)
        cross_attention_time += self_attention_time_targ
        src2, targ2 = self.norm2(src2), self.norm2(targ2)
        src2_w_pos, targ2_w_pos = self.with_pos_embed(src2, src_pos), self.with_pos_embed(targ2, targ_pos)

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        targ3, xatt_weights_targ = self.multihead_attn(
            query = self.with_pos_embed(targ2, targ_pos),       ## [N, BS, 32]
            key = src2_w_pos,
            value = src2_w_pos if self.ca_val_has_pos_emb else src2_w_pos,
            attn_mask = src_mask,
            key_padding_mask = src_key_padding_mask
        )
        # weights['xatt_weights_targ'] = xatt_weights_targ

        src3, xatt_weights_src = self.multihead_attn(
            query = self.with_pos_embed(src2, src_pos),
            key = targ2_w_pos,
            value = targ2_w_pos if self.ca_val_has_pos_emb else targ2,
            attn_mask = targ_mask,
            key_padding_mask = targ_key_padding_mask
        )
        # weights['xatt_weights_src'] = xatt_weights_src
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            cross_attention_time +=  start.elapsed_time(end)

        src = src + self.dropout2(src3)
        targ = targ + self.dropout2(targ3)

        src = self.position_wise_feedforward(src)
        targ = self.position_wise_feedforward(targ)
            # total_weights[i] = weights
        self.satt_weights = (satt_weights_src, satt_weights_targ)
        self.xatt_weights = (xatt_weights_src, xatt_weights_targ)
    
        return src, targ,cross_attention_time


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
