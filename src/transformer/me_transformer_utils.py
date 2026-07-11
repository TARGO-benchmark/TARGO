import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import MinkowskiEngine as ME
from torchtyping import TensorType
from torch_scatter import scatter_mean

def to_sparse_coo(data):
    # coords = torch.nonzero(data > 0.5, as_tuple=False)
    coords = torch.nonzero(data > 0, as_tuple=False)
    # feats = data[data > 0.5].unsqueeze(1)
    # feats = (data[data > 0.5]>0.5).float().unsqueeze(1)
    feats = data[data > 0].unsqueeze(1)
    return coords, feats

def create_quantized_masked_sparse_tensor(inputs, feat):
    all_coords = []
    all_values = []
    device = inputs.device

    for b in range(inputs.shape[0]):
        lower_bound = torch.tensor([0.02, 0.02, 0.055]).to(inputs.device)
        upper_bound = torch.tensor([0.28, 0.28, 0.30000000000000004]).to(inputs.device)

        mask = (inputs[b] >= lower_bound) & (inputs[b] <= upper_bound)
        mask = torch.all(mask,dim=-1)

        quant_points, unique_map, inverse_map  = ME.utils.sparse_quantize(inputs[b][mask], quantization_size=0.0075,return_index=True, return_inverse=True)
        feat_batch = feat[b][mask]
        quant_value = scatter_mean(feat_batch, inverse_map.to(device), dim=0)
        all_coords.append(quant_points)
        all_values.append(quant_value)
    
    coords, values = ME.utils.sparse_collate(all_coords, all_values)
    sinput = ME.SparseTensor(features=values, coordinates=coords, device=inputs.device)
    return sinput


def create_quantized_sparse_tensor(inputs, feat):
    all_coords = []
    all_values = []
    device = inputs.device
    inputs = inputs + 0.5

    for b in range(inputs.shape[0]):
        points = inputs[b]
        feat_batch = feat[b]
        finite_mask = torch.isfinite(points).all(dim=-1)
        points = points[finite_mask]
        feat_batch = feat_batch[finite_mask]
        if points.shape[0] == 0:
            points = torch.zeros((1, 3), dtype=inputs.dtype, device=device)
            feat_batch = torch.zeros((1, feat.shape[-1]), dtype=feat.dtype, device=device)
        points = points.clamp(min=0.0, max=1.0 - 1e-6)
        quant_points, unique_map, inverse_map  = ME.utils.sparse_quantize(points, quantization_size=0.025,return_index=True, return_inverse=True)
        quant_value = scatter_mean(feat_batch, inverse_map.to(device), dim=0)
        all_coords.append(quant_points)
        all_values.append(quant_value)
    
    coords, values = ME.utils.sparse_collate(all_coords, all_values)
    sinput = ME.SparseTensor(features=values, coordinates=coords, device=inputs.device)
    return sinput

def create_sparse_tensor(inputs):
    all_coords = []
    all_values = []

    # Get the device from the input data
    device = inputs.device

    for b in range(inputs.shape[0]):
        coords, values = to_sparse_coo(inputs[b])
        all_coords.append(coords)
        all_values.append(values)

    coords, values = ME.utils.sparse_collate(all_coords, all_values)
    sinput = ME.SparseTensor(features=values, coordinates=coords, device=device)

    return sinput


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, n_dim: int = 1, d_model: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.mlp(xyz)

class LocalSelfAttentionBase(nn.Module):
    def __init__(self, kernel_size, stride, dilation, dimension):
        super(LocalSelfAttentionBase, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension
        
        self.kernel_generator = KernelGenerator(kernel_size=kernel_size,
                                                stride=stride,
                                                dilation=dilation,
                                                dimension=dimension)
        self.kernel_volume = self.kernel_generator.kernel_volume

    def get_kernel_map_and_out_key(self, stensor):
        cm = stensor.coordinate_manager
        in_key = stensor.coordinate_key
        out_key = cm.stride(in_key, self.kernel_generator.kernel_stride)
        region_type, region_offset, _ = self.kernel_generator.get_kernel(
            stensor.tensor_stride, False)
        kernel_map = cm.kernel_map(in_key,
                                   out_key,
                                   self.kernel_generator.kernel_stride,
                                   self.kernel_generator.kernel_size,
                                   self.kernel_generator.kernel_dilation,
                                   region_type=region_type,
                                   region_offset=region_offset)
        return kernel_map, out_key

    def key_query_map_from_kernel_map(self, kernel_map):
        kq_map = []
        for kernel_idx, in_out in kernel_map.items():
            in_out[0] = in_out[0] * self.kernel_volume + kernel_idx
            kq_map.append(in_out)
        kq_map = torch.cat(kq_map, -1)
        return kq_map
    
    def key_query_indices_from_kernel_map(self, kernel_map):
        kq_indices = []
        for _, in_out in kernel_map.items():
            kq_indices.append(in_out)
            kq_indices = torch.cat(kq_indices, -1)
        return kq_indices

    def key_query_indices_from_key_query_map(self, kq_map):
        kq_indices = kq_map.clone()
        kq_indices[0] = kq_indices[0] // self.kernel_volume
        return kq_indices

class LightweightSelfAttentionLayer(LocalSelfAttentionBase):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        dilation=1,
        num_heads=8,
        linear_att=False
    ):
        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels % num_heads == 0
        assert kernel_size % 2 == 1
        assert stride == 1, "Currently, this layer only supports stride == 1"
        assert dilation == 1,"Currently, this layer only supports dilation == 1"
        super(LightweightSelfAttentionLayer, self).__init__(kernel_size, stride, dilation, dimension=3)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.num_heads = num_heads
        self.attn_channels = out_channels // num_heads

        self.to_query = nn.Sequential(
            ME.MinkowskiLinear(in_channels, out_channels),
            ME.MinkowskiToFeature()
        )
        self.to_value = nn.Sequential(
            ME.MinkowskiLinear(in_channels, out_channels),
            ME.MinkowskiToFeature()
        )
        self.to_out = nn.Linear(out_channels, out_channels)

        self.inter_pos_enc = nn.Parameter(torch.FloatTensor(self.kernel_volume, self.num_heads, self.attn_channels))
        self.intra_pos_mlp = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )
        nn.init.normal_(self.inter_pos_enc, 0, 1)

    def forward(self, stensor, norm_points):
        dtype = stensor._F.dtype
        device = stensor._F.device

        # query, key, value, and relative positional encoding
        intra_pos_enc = self.intra_pos_mlp(norm_points)
        stensor = stensor + intra_pos_enc
        q = self.to_query(stensor).view(-1, self.num_heads, self.attn_channels).contiguous()
        v = self.to_value(stensor).view(-1, self.num_heads, self.attn_channels).contiguous()

        # key-query map
        kernel_map, out_key = self.get_kernel_map_and_out_key(stensor)
        kq_map = self.key_query_map_from_kernel_map(kernel_map)

        # attention weights with cosine similarity
        attn = torch.zeros((kq_map.shape[1], self.num_heads), dtype=dtype, device=device)
        norm_q = F.normalize(q, p=2, dim=-1)
        norm_pos_enc = F.normalize(self.inter_pos_enc, p=2, dim=-1)
        attn = ops.dot_product_cuda(norm_q, norm_pos_enc, attn, kq_map)

        # aggregation & the output
        out_F = torch.zeros((len(q), self.num_heads, self.attn_channels),
                            dtype=dtype,
                            device=device)
        kq_indices = self.key_query_indices_from_key_query_map(kq_map)
        out_F = ops.scalar_attention_cuda(attn, v, out_F, kq_indices)
        out_F = self.to_out(out_F.view(-1, self.out_channels).contiguous())
        return ME.SparseTensor(out_F,
                               coordinate_map_key=out_key,
                               coordinate_manager=stensor.coordinate_manager)


class TransformerCrossEncoder(nn.Module):

    def __init__(self, cross_encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.attention_type = cross_encoder_layer.attention_type
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    ## src: x_feats_padded, tgt: y_feats_padded
    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                time_file=None):

        src_intermediate, tgt_intermediate = [], []
        
        other_type = 'linear_attention' if self.attention_type == 'dot_prod' else 'dot_prod'
        time_dict_global = {self.attention_type: 0, other_type: 0}
        for layer in self.layers:
            src, tgt, cross_attention_time = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                    src_key_padding_mask=src_key_padding_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    src_pos=src_pos, tgt_pos=tgt_pos, time_file=time_file)
            time_dict_global[self.attention_type] += cross_attention_time
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        # if self.norm is not None:
        #     src = self.norm(src)
        #     tgt = self.norm(tgt)
        #     if self.return_intermediate:
        #         if len(self.layers) > 0:
        #             src_intermediate.pop()
        #             tgt_intermediate.pop()
        #         src_intermediate.append(src)
        #         tgt_intermediate.append(tgt)

        # if self.return_intermediate:
        #     return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src.unsqueeze(0), tgt.unsqueeze(0), time_dict_global

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)

class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        self.attention_type = attention_type
        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,
                     time_file=None):   ## time_file is for analysis, not used in training

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        ## src: (seq_len, batch_size, d_model)
        ## tgt: (seq_len, batch_size, d_model)
        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos   ## x_feats becomes query and key
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if self.attention_type == 'dot_prod':
            src2, satt_weights_s = self.self_attn(q, k,
                                value=src_w_pos if self.sa_val_has_pos_emb else src,
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        elif self.attention_type == 'linear_attention':
            assert self.sa_val_has_pos_emb, 'source should have pos embeddings'
            src2 = self.self_attn(x=src_w_pos, input_mask=src_key_padding_mask.T)
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            cross_attention_time = start.elapsed_time(end)
        else:
            cross_attention_time = 0
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos ## y_feats becomes query and key 
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if self.attention_type == 'dot_prod':
            tgt2, satt_weights_t = self.self_attn(q, k,
                                                value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                                attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask)
        elif self.attention_type == 'linear_attention':
            assert self.sa_val_has_pos_emb, 'target should have pos embeddings'
            tgt2 = self.self_attn(x=tgt_w_pos, input_mask=tgt_key_padding_mask.T)
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            cross_attention_time +=  start.elapsed_time(end)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if self.attention_type == 'dot_prod':
            src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                    key=tgt_w_pos,
                                                    value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                    attn_mask=tgt_mask,
                                                    key_padding_mask=tgt_key_padding_mask)
            tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                    key=src_w_pos,
                                                    value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                    attn_mask=src_mask,
                                                    key_padding_mask=src_key_padding_mask)
        elif self.attention_type == 'linear_attention':
            src2 = self.multihead_attn(x=self.with_pos_embed(src, src_pos),
                                       input_mask=src_key_padding_mask.T,
                                       context=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                       context_mask=tgt_key_padding_mask.T)
            tgt2 = self.multihead_attn(x=self.with_pos_embed(tgt, tgt_pos),
                                       input_mask=tgt_key_padding_mask.T,
                                       context=src_w_pos if self.ca_val_has_pos_emb else src2,
                                       context_mask=src_key_padding_mask.T) 
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            cross_attention_time +=  start.elapsed_time(end)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Stores the attention weights for analysis, if required
        if self.attention_type == 'dot_prod':
            self.satt_weights = (satt_weights_s, satt_weights_t)
            self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt, cross_attention_time

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,
                    time_file=None):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        src2 = self.norm1(src)
        src2_w_pos = self.with_pos_embed(src2, src_pos)
        q = k = src2_w_pos
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if self.attention_type == 'dot_prod':
            src2, satt_weights_s = self.self_attn(q, k,
                                                value=src2_w_pos if self.sa_val_has_pos_emb else src2,
                                                attn_mask=src_mask,
                                                key_padding_mask=src_key_padding_mask)
        elif self.attention_type == 'linear_attention':
            assert self.sa_val_has_pos_emb, 'source should have pos embeddings'
            src2 = self.self_attn(x=src2_w_pos, input_mask=src_key_padding_mask.T)
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            cross_attention_time = start.elapsed_time(end)
        else:
            cross_attention_time = 0
        src = src + self.dropout1(src2)

        tgt2 = self.norm1(tgt)
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)
        q = k = tgt2_w_pos
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if self.attention_type == 'dot_prod':
            tgt2, satt_weights_t = self.self_attn(q, k,
                                                value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
                                                attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask)
        elif self.attention_type == 'linear_attention':
            assert self.sa_val_has_pos_emb, 'target should have pos embeddings'
            tgt2 = self.self_attn(x=tgt2_w_pos, input_mask=tgt_key_padding_mask.T)
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            cross_attention_time +=  start.elapsed_time(end)

        tgt = tgt + self.dropout1(tgt2)

        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if self.attention_type == 'dot_prod':
            src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
                                                    key=tgt_w_pos,
                                                    value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                                    attn_mask=tgt_mask,
                                                    key_padding_mask=tgt_key_padding_mask)
            tgt3, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                                    key=src_w_pos,
                                                    value=src_w_pos if self.ca_val_has_pos_emb else src2,
                                                    attn_mask=src_mask,
                                                    key_padding_mask=src_key_padding_mask)
        elif self.attention_type == 'linear_attention':
            src3 = self.multihead_attn(x=self.with_pos_embed(src2, src_pos),
                                       input_mask=src_key_padding_mask.T,
                                       context=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                       context_mask=tgt_key_padding_mask.T)
            tgt3 = self.multihead_attn(x=self.with_pos_embed(tgt2, tgt_pos),
                                       input_mask=tgt_key_padding_mask.T,
                                       context=src_w_pos if self.ca_val_has_pos_emb else src2,
                                       context_mask=src_key_padding_mask.T)
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            cross_attention_time += start.elapsed_time(end)

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        if self.attention_type == 'dot_prod':
            self.satt_weights = (satt_weights_s, satt_weights_t)
            self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt, cross_attention_time

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                time_file=None):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos, time_file=time_file)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos, time_file=time_file)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def batch_feat_size(x) -> List[int]:
    _, batch_feat_size = torch.unique(x[:,0], return_counts=True)
    return batch_feat_size.tolist()

def batch_tolist( x:TensorType, seq:List[int]) -> List[TensorType]:
    x = list(torch.split(x, seq))
    return x

def pad_sequence(sequences, require_padding_mask=False, require_lens=False,
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


def combine_cross_attention(pos_embed, transformer_layer,  x, y_c, y_f, time_file=None):
    x_batch_feat_size = batch_feat_size(x.C)
    y_batch_feat_size = batch_feat_size(y_c)

    ## Positional Embedding
    x_pe = batch_tolist(pos_embed(x.C[:, 1:].to(torch.float)), x_batch_feat_size)
    y_pe = batch_tolist(pos_embed(y_c[:, 1:].to(torch.float)), y_batch_feat_size)
    y_feats_un = batch_tolist(y_f, y_batch_feat_size)  ## y_feats_un: [batch, seq, feat_dim], un means un-padded
    x_feats_un = batch_tolist(x.F, x_batch_feat_size)  ## x_feats_un: [batch, seq, feat_dim]

    x_pe_padded, _, _ = pad_sequence(x_pe)
    y_pe_padded, _, _ = pad_sequence(y_pe)

    x_feats_padded, x_key_padding_mask, _ = pad_sequence(x_feats_un,
                                                            require_padding_mask=True)
    ## x_key_padding_mask: [batch, seq], 1 means padded, 0 means not padded
    y_feats_padded, y_key_padding_mask, _ = pad_sequence(y_feats_un,
                                                            require_padding_mask=True)

    x_feats_cond, y_feats_cond, cross_attention_time_dict = transformer_layer(
        x_feats_padded, y_feats_padded,
        src_key_padding_mask=x_key_padding_mask,
        tgt_key_padding_mask=y_key_padding_mask,
        src_pos=x_pe_padded,
        tgt_pos=y_pe_padded,
        time_file=time_file
    )

    x_feats_cond = torch.squeeze(x_feats_cond, dim=0)
    y_feats_cond = torch.squeeze(y_feats_cond, dim=0)
    x_feats_list = unpad_sequences(x_feats_cond, x_batch_feat_size)
    # y_feats_list = unpad_sequences(y_feats_cond, y_batch_feat_size)

    x_feats = torch.vstack(x_feats_list)
    # y_feats = torch.vstack(y_feats_list)

    x = ME.SparseTensor(coordinates=x.C, features=x_feats)
    
    return x, cross_attention_time_dict

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def unpad_sequences(padded, seq_lens):
    """Reverse of pad_sequence"""
    sequences = [padded[..., :seq_lens[b], b, :] for b in range(len(seq_lens))]
    return sequences
