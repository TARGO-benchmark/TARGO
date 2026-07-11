import torch
import MinkowskiEngine as ME
from typing import List, Any, Dict
from torchtyping import TensorType
from src.transformer.me_transformer_utils import PositionEmbeddingLearned, create_quantized_sparse_tensor
from src.transformer.transformer_fushion import TransformerFusion
from torch import nn
import copy
import numpy as np
import time
import logging

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerFusionModel(torch.nn.Module):
    def __init__(self, attention_params: Dict[str, Any], num_layers, return_intermediate, cross_att_key, d_model):
        super().__init__()
        self.attention_params = attention_params
        self.return_intermediate = return_intermediate

        # Initialize components
        self.sparse_encoder_in = ME.MinkowskiConvolution(
            in_channels=32, 
            out_channels=32, 
            kernel_size=3, 
            stride=1,
            dimension=3
        )

        self.encoder_dense = nn.Conv3d(32, 32, 3, padding=1)
        self.me_bn = ME.MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.me_relu = ME.MinkowskiReLU(inplace=True)
        self.me_bn_pos_embed = PositionEmbeddingLearned(3, 32)

        self.encoder_points = nn.Conv1d(3, 32, kernel_size=1, stride=1)
        transformer_fusion = self._init_transformer_fusion(cross_att_key, d_model)
        self.layers = _get_clones(transformer_fusion, num_layers)
        self.attention_type = transformer_fusion.attention_type
        if attention_params[cross_att_key]['pre_norm']:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = None


        self.size = 0.30000000000000004


    def pad_sequence(self,sequences, require_padding_mask=False, require_lens=False,
                    batch_first=False):
        """List of sequences to padded sequences

        Args:
            sequences: List of sequences (N, D)
            require_padding_mask:

        Returns:
            (padded_sequence, padding_mask), where
            padded sequence has shape (N_max, B, D)
            padding_mask will be none if require_padding_mask is False
        """
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
        padding_mask = None
        padding_lens = None

        if require_padding_mask:
            B = len(sequences)
            seq_lens = list(map(len, sequences))
            padding_mask = torch.zeros((B, padded.shape[0]), dtype=torch.bool, device=padded.device)
            for i, l in enumerate(seq_lens):
                padding_mask[i, l:] = True

        if require_lens:
            padding_lens = [seq.shape[0] for seq in sequences]

        return padded, padding_mask, padding_lens


    def unpad_sequences(self,padded, seq_lens):
        """Reverse of pad_sequence"""
        sequences = [padded[..., :seq_lens[b], b, :] for b in range(len(seq_lens))]
        return sequences


    def _init_transformer_fusion(self, cross_att_key, d_model):
        # cross_att_key = 'pointnet_cross_attention'
        params = self.attention_params[cross_att_key]
        return TransformerFusion(
            d_model=d_model,
            nhead=params['nhead'],
            dim_feedforward=params['d_feedforward'],
            dropout=params['dropout'],
            activation=params['transformer_act'],
            normalize_before=params['pre_norm'],
            sa_val_has_pos_emb=params['sa_val_has_pos_emb'],
            ca_val_has_pos_emb=params['ca_val_has_pos_emb'],
            attention_type=params['attention_type'],
        )
    
    
    def batch_feat_size(self, x) -> List[int]:
        _, batch_feat_size = torch.unique(x[:,0], return_counts=True)
        return batch_feat_size.tolist()

    def batch_tolist(self, x:TensorType, seq:List[int]) -> List[TensorType]:
        x = list(torch.split(x, seq))
        return x

    def forward(self, scene_pc, targ_pc,) -> torch.Tensor:
        src_intermediate, tgt_intermediate = [], []
        other_type = 'linear_attention' if self.attention_type == 'dot_prod' else 'dot_prod'
        time_dict_global = {self.attention_type: 0, other_type: 0}

        assert targ_pc is not None and scene_pc is not None

        targ_pc_feat =self.encoder_points(targ_pc.transpose(1,2))
        scene_pc_feat = self.encoder_points(scene_pc.transpose(1,2))
        targ_pc_feat = targ_pc_feat.transpose(1,2)
        scene_pc_feat = scene_pc_feat.transpose(1,2)
        sinput_targ = create_quantized_sparse_tensor(targ_pc, targ_pc_feat)     ## [866,4]
        sinput_scene = create_quantized_sparse_tensor(scene_pc, scene_pc_feat)      ## [866,32]

        ## Process inputs and perform computations
        c_targ = self.sparse_encoder_in(sinput_targ)
        c_scene = self.sparse_encoder_in(sinput_scene)
        c_targ = self.me_bn(c_targ)
        c_scene = self.me_bn(c_scene)
        c_targ = self.me_relu(c_targ)
        c_scene = self.me_relu(c_scene)

        scene_batch_feat_size = self.batch_feat_size(sinput_scene.C)
        scene_pe = self.batch_tolist(self.me_bn_pos_embed(c_scene.C[:, 1:].to(torch.float)), scene_batch_feat_size)
        scene_feats_un = self.batch_tolist(c_scene.F, scene_batch_feat_size)
        scene_pe_padded, _, _ = self.pad_sequence(scene_pe)
        scene_feats_padded, scene_key_padding_mask, _ = self.pad_sequence(scene_feats_un,require_padding_mask=True)

        targ_batch_feat_size = self.batch_feat_size(sinput_targ.C)
        targ_pe = self.batch_tolist(self.me_bn_pos_embed(c_targ.C[:, 1:].to(torch.float)), targ_batch_feat_size)
        targ_feats_un = self.batch_tolist(c_targ.F, targ_batch_feat_size)
        targ_pe_padded, _, _ = self.pad_sequence(targ_pe)
        targ_feats_padded, targ_key_padding_mask, _ = self.pad_sequence(targ_feats_un,require_padding_mask=True)  

        src, target = targ_feats_padded, scene_feats_padded
        src_key_padding_mask, targ_key_padding_mask = targ_key_padding_mask, scene_key_padding_mask
        src_pos, targ_pos = targ_pe_padded, scene_pe_padded

        ## record time
        start_time = time.time()

        for layer in self.layers:
            src, target, cross_attention_time = layer(
                src = src, targ = target,
                src_key_padding_mask= src_key_padding_mask,  targ_key_padding_mask = targ_key_padding_mask,
                src_pos=src_pos, targ_pos= targ_pos
            )
            time_dict_global[self.attention_type] += cross_attention_time
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(target) if self.norm is not None else tgt)
        
        end_time = time.time()
        # logging.info(f"Time taken for {self.attention_type} attention: {time_dict_global[self.attention_type]}")
        total_time = end_time - start_time
        # print(f"cross attention time={total_time}")
        print(f"cross attention time={total_time:.3g}")
        if self.norm is not None:
            src = self.norm(src)
            tgt = self.norm(target)
            if self.return_intermediate:
                if len(self.layers) > 0:
                    src_intermediate.pop()
                    tgt_intermediate.pop()
                src_intermediate.append(src)
                tgt_intermediate.append(target)
            
        src_fused = src.unsqueeze(0)
        tgt_fused = target.unsqueeze(0)

        src_fused_cond = torch.squeeze(src_fused, dim=0)
        tgt_fused_cond = torch.squeeze(tgt_fused, dim=0)

        src_fused_list = self.unpad_sequences(src_fused_cond, targ_batch_feat_size)
        tgt_fused_list = self.unpad_sequences(tgt_fused_cond, scene_batch_feat_size)

        src_feat_fused = torch.vstack(src_fused_list)
        tgt_feat_fused = torch.vstack(tgt_fused_list)

        src_feat = ME.SparseTensor(coordinates=sinput_targ.C, features=src_feat_fused)
        tgt_feat = ME.SparseTensor(coordinates=sinput_scene.C, features=tgt_feat_fused)

        BS = targ_pc.shape[0]
        src_feat_dense = src_feat.dense(torch.Size((BS, 32,40,40,40)))[0]
        tgt_feat_dense = tgt_feat.dense(torch.Size((BS, 32,40,40,40)))[0]

        src_feat_dense_updated = self.encoder_dense(src_feat_dense)
        tgt_feat_dense_updated = self.encoder_dense(tgt_feat_dense)

        c_fused = torch.cat([src_feat_dense_updated, tgt_feat_dense_updated], dim=1)

        return c_fused