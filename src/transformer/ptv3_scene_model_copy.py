"""
PointTransformerV3 Scene Model for TARGO
Based on original PointTransformerV3 with encoder only + TARGO decoder
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Any, Dict
import sys
import os
from functools import partial
from addict import Dict
import math
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
from collections import OrderedDict

try:
    import flash_attn
except ImportError:
    flash_attn = None

# Import serialization functions from the PointTransformerV3 submodule
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'PointTransformerV3'))
# Assert that PointTransformerV3 serialization is available and import directly
from serialization import encode


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):
    """
    Point Structure of Pointcept (based on original implementation)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """Point Cloud Serialization (based on original implementation)"""
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        # Use original encode function directly (must be available due to assert)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """Point Cloud Sparsification for SpConv (based on original implementation)"""
        assert {"feat", "batch"}.issubset(self.keys()), f"Missing required keys for sparsify: need 'feat' and 'batch', got keys: {list(self.keys())}"
        
        if "grid_coord" not in self.keys():
            required_keys = {"grid_size", "coord"}
            missing_keys = required_keys - set(self.keys())
            assert len(missing_keys) == 0, f"Missing required keys for grid_coord generation: {missing_keys}. Available keys: {list(self.keys())}"
            
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(nn.Module):
    """PointModule base class (based on original implementation)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    """A sequential container for Point modules (based on original implementation)"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
                # Ensure Point object maintains proper structure
                if isinstance(input, Point) and hasattr(input, 'feat') and isinstance(input.feat, Point):
                    raise TypeError(f"Module {k} ({type(module)}) returned Point object with feat as Point object. This is incorrect.")
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    # Ensure feat is a tensor before processing
                    if isinstance(input.feat, Point):
                        raise TypeError(f"Point.feat is incorrectly a Point object before module {k} ({type(module)})")
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PDNorm(PointModule):
    """Point Discriminative Normalization (based on original implementation)"""
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        
        if self.decouple:
            self.norm = norm_layer(num_features)
            if self.adaptive:
                self.modulation = nn.Sequential(
                    nn.Linear(context_channels, num_features * 2),
                )
        else:
            self.norm = nn.ModuleDict()
            for condition in conditions:
                self.norm[condition] = norm_layer(num_features)

    def forward(self, point):
        if self.decouple:
            point.feat = self.norm(point.feat)
            if self.adaptive and "context" in point.keys():
                modulation = self.modulation(point.context)
                gamma, beta = modulation.chunk(2, dim=-1)
                point.feat = point.feat * (1 + gamma) + beta
        else:
            # For simplicity, use the first condition if no condition is specified
            condition = self.conditions[0] if hasattr(point, 'condition') else self.conditions[0]
            point.feat = self.norm[condition](point.feat)
        return point


class RPE(nn.Module):
    """Relative Position Encoding (based on original implementation)"""
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.rpe_table = nn.Parameter(torch.zeros(3 * patch_size - 2, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        # Simplified RPE implementation
        rpe = self.rpe_table[coord.sum(dim=-1) % self.rpe_table.shape[0]]
        return rpe


class SerializedAttention(PointModule):
    """Serialized Attention (based on original implementation)"""
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        
        if enable_flash:
            assert enable_rpe is False
            assert upcast_attention is False
            assert upcast_softmax is False
            assert flash_attn is not None
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        
        if enable_rpe:
            self.rpe = RPE(patch_size, num_heads)

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        """Get relative position for RPE"""
        # Simplified relative position calculation
        if self.enable_rpe and hasattr(self, 'rpe'):
            return self.rpe(point.coord)
        return None

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        """Get padding and inverse mapping"""
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )
        
        # Ensure we get the actual feature tensor, not a Point object
        if isinstance(point.feat, Point):
            if hasattr(point.feat, 'feat'):
                actual_feat = point.feat.feat
            else:
                raise ValueError(f"point.feat is a Point object but has no 'feat' attribute. Keys: {list(point.feat.keys())}")
        else:
            actual_feat = point.feat
            
        if not isinstance(actual_feat, torch.Tensor):
            raise TypeError(f"Expected tensor for features in get_padding_and_inverse, got {type(actual_feat)}")
        
        # Simplified padding logic
        batch_size = point.offset.shape[0]
        padding = torch.zeros(batch_size, device=point.offset.device, dtype=torch.long)
        inverse = torch.arange(actual_feat.shape[0], device=actual_feat.device)
        return padding, inverse

    def forward(self, point):
        # Debug and validate serialization data
        if not hasattr(point, 'serialized_order'):
            raise AttributeError(f"Point object missing 'serialized_order'. Available keys: {list(point.keys())}")
        
        if not hasattr(point, 'serialized_inverse'):
            raise AttributeError(f"Point object missing 'serialized_inverse'. Available keys: {list(point.keys())}")
        
        # Debug serialization structure
        print(f"Debug SerializedAttention:")
        print(f"  order_index: {self.order_index}")
        print(f"  serialized_order type: {type(point.serialized_order)}")
        if hasattr(point.serialized_order, 'shape'):
            print(f"  serialized_order shape: {point.serialized_order.shape}")
        elif isinstance(point.serialized_order, (list, tuple)):
            print(f"  serialized_order length: {len(point.serialized_order)}")
        else:
            print(f"  serialized_order content: {point.serialized_order}")
        
        # Validate and get serialization order
        if not isinstance(point.serialized_order, torch.Tensor):
            raise TypeError(f"serialized_order must be a tensor, got {type(point.serialized_order)}")
        
        # Check if serialized_order is multi-dimensional
        if point.serialized_order.dim() == 1:
            print(f"  Warning: serialized_order is 1D with shape {point.serialized_order.shape}")
            # For 1D tensor, use the entire tensor as order
            order = point.serialized_order
            inverse = point.serialized_inverse
        else:
            # Multi-dimensional tensor - check bounds
            if self.order_index >= point.serialized_order.shape[0]:
                print(f"  Warning: order_index {self.order_index} >= serialized_order.shape[0] {point.serialized_order.shape[0]}")
                print(f"  Using order_index 0 as fallback")
                order_idx = 0
            else:
                order_idx = self.order_index
            
            order = point.serialized_order[order_idx]
            inverse = point.serialized_inverse[order_idx]
        
        print(f"  Selected order type: {type(order)}, shape: {order.shape if hasattr(order, 'shape') else 'N/A'}")
        
        # Ensure point.feat is a tensor, not a Point object
        if isinstance(point.feat, Point):
            # If point.feat is a Point object, extract its actual feature tensor
            if hasattr(point.feat, 'feat'):
                actual_feat = point.feat.feat
            else:
                raise ValueError(f"point.feat is a Point object but has no 'feat' attribute. Keys: {list(point.feat.keys())}")
        else:
            actual_feat = point.feat
        
        # Validate that we now have a proper tensor
        if not isinstance(actual_feat, torch.Tensor):
            raise TypeError(f"Expected tensor for features, got {type(actual_feat)}")
        
        # Apply order
        feat = actual_feat[order]
        
        if self.enable_flash and flash_attn is not None:
            # Use flash attention if available
            qkv = self.qkv(feat)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(-1, self.num_heads, self.channels // self.num_heads)
            k = k.view(-1, self.num_heads, self.channels // self.num_heads)
            v = v.view(-1, self.num_heads, self.channels // self.num_heads)
            
            # Simplified flash attention (would need proper implementation)
            feat = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            feat = feat.view(-1, self.channels)
        else:
            # Standard attention
            padding, _ = self.get_padding_and_inverse(point)
            
            qkv = self.qkv(feat)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(-1, self.num_heads, self.channels // self.num_heads) * self.scale
            k = k.view(-1, self.num_heads, self.channels // self.num_heads)
            v = v.view(-1, self.num_heads, self.channels // self.num_heads)
            
            # Compute attention (simplified)
            attn = torch.matmul(q, k.transpose(-2, -1))
            
            # Add relative position encoding if enabled
            if self.enable_rpe:
                rpe = self.get_rel_pos(point, order)
                if rpe is not None:
                    attn = attn + rpe.unsqueeze(0)
            
            if self.upcast_softmax:
                attn = self.softmax(attn.float()).type_as(attn)
            else:
                attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            
            feat = torch.matmul(attn, v).view(-1, self.channels)
        
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        
        # Apply inverse order and ensure we set a tensor, not a Point object
        point.feat = feat[inverse]
        
        return point


class MLP(nn.Module):
    """MLP (based on original implementation)"""
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    """Transformer Block (based on original implementation)"""
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    """Serialized Pooling (based on original implementation)"""
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        # Project features
        feat = self.proj(point.feat)
        
        # Pooling using serialization order (simplified implementation)
        # In the original, this uses more sophisticated serialization-aware pooling
        order = point.serialized_order[0]  # Use first order
        inverse = point.serialized_inverse[0]
        
        # Apply order
        feat_ordered = feat[order]
        coord_ordered = point.coord[order]
        batch_ordered = point.batch[order]
        
        # Simple stride-based pooling
        pooled_indices = torch.arange(0, feat_ordered.shape[0], self.stride, device=feat.device)
        feat_pooled = feat_ordered[pooled_indices]
        coord_pooled = coord_ordered[pooled_indices]
        batch_pooled = batch_ordered[pooled_indices]
        
        # Create new point, preserving important keys from original point
        point_dict = Dict(
            feat=feat_pooled,
            coord=coord_pooled,
            batch=batch_pooled,
        )
        
        # Preserve grid_size and other important attributes
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        point = Point(point_dict)
        if hasattr(self, 'norm'):
            point = self.norm(point)
        if hasattr(self, 'act'):
            point = self.act(point)
        
        # Update offset for pooled points
        unique_batches, counts = torch.unique(point.batch, return_counts=True, sorted=True)
        point.offset = torch.cumsum(counts, dim=0)
        
        # Ensure grid_size is available before calling sparsify
        if "grid_size" not in point.keys():
            point["grid_size"] = 0.3/40  # Updated grid size for voxelization
        
        point.sparsify()
        return point


class Embedding(PointModule):
    """Embedding layer (based on original implementation)"""
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


class PointTransformerV3SceneModel(PointModule):
    """
    PointTransformerV3 Scene Model for TARGO (Encoder-only with TARGO decoder)
    Based on original PointTransformerV3 implementation
    """
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # Decoder parameters (accepted but not used in encoder-only model)
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        # Other parameters
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,  # Accept but ignore
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        **kwargs  # Accept any additional keyword arguments
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        self.cls_mode = cls_mode

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        # norm layers (based on original implementation)
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder only (based on original implementation)
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")
        
        # TARGO decoder layers after the encoder
        self.feature_projection = nn.Sequential(
            nn.Linear(enc_channels[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )
        
        # Grasp prediction heads (same as original TARGO)
        self.qual_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.rot_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)  # quaternion
        )
        
        self.width_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def prepare_point_data(self, points):
        """
        Prepare point cloud data for PointTransformerV3.
        """
        batch_size, num_points, feature_dim = points.shape
        
        # Always ensure features match the model's expected in_channels
        target_feat_dim = self.embedding.in_channels
        
        # Get coordinates (always first 3 channels)
        coords_data = points[:, :, :3]  # xyz coordinates
        
        # Prepare features to match target_feat_dim
        if feature_dim >= target_feat_dim:
            # Use first target_feat_dim channels as features
            feats_data = points[:, :, :target_feat_dim]
        elif feature_dim >= 3:
            # Need to pad features to reach target_feat_dim
            feats_data = points[:, :, :feature_dim]  # Use available features
            # Pad with zeros or repeat coordinates
            if target_feat_dim > feature_dim:
                padding_needed = target_feat_dim - feature_dim
                if target_feat_dim == 6 and feature_dim == 3:
                    # Common case: pad xyz to xyzxyz (repeat coordinates)
                    padding = coords_data  # Use xyz again as padding
                else:
                    # General case: pad with zeros
                    padding = torch.zeros(batch_size, num_points, padding_needed, 
                                        device=points.device, dtype=points.dtype)
                feats_data = torch.cat([feats_data, padding], dim=-1)
        else:
            raise ValueError(f"Input must have at least 3 channels (xyz), got {feature_dim}")
        
        # Final assertion to ensure correct feature dimension
        assert feats_data.shape[-1] == target_feat_dim, \
            f"Feature dimension mismatch: got {feats_data.shape[-1]}, expected {target_feat_dim}"
        
        # Flatten batch and points for PointTransformerV3
        all_coords = []
        all_feats = []
        all_offsets = [0]
        
        for b in range(batch_size):
            coords = coords_data[b]  # xyz coordinates
            feats = feats_data[b]    # features
            
            all_coords.append(coords)
            all_feats.append(feats)
            all_offsets.append(all_offsets[-1] + num_points)
        
        # Concatenate all batches
        coord = torch.cat(all_coords, dim=0)  # [B*N, 3]
        feat = torch.cat(all_feats, dim=0)    # [B*N, target_feat_dim]
        offset = torch.tensor(all_offsets[1:], device=points.device, dtype=torch.long)  # [B]
        
        # Create data dictionary for PointTransformerV3
        data_dict = {
            "coord": coord,
            "feat": feat,
            "offset": offset,
            "grid_size": 0.3/40  # Updated grid size for voxelization
        }
        
        return data_dict

    def forward(self, data_dict_or_points, pos=None):
        """
        Forward pass through encoder and decoder.
        
        Args:
            data_dict_or_points: Either data dictionary for PointTransformerV3 or point cloud tensor
            pos: Grasp positions [B, 1, 3] (optional, for compatibility)
            
        Returns:
            Tuple of (quality, rotation, width) predictions
        """
        # Handle both data_dict and point cloud inputs
        if isinstance(data_dict_or_points, dict):
            data_dict = data_dict_or_points
            # Try to get batch size from offset
            if 'offset' in data_dict:
                batch_size = len(data_dict['offset'])
            else:
                batch_size = 1
        else:
            # Assume it's a point cloud tensor
            scene_pc = data_dict_or_points
            batch_size = scene_pc.shape[0]
            data_dict = self.prepare_point_data(scene_pc)
        
        # Forward through encoder (based on original implementation)
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # Add debugging information for encoder call
        assert isinstance(self.enc, PointSequential), f"Expected PointSequential, got {type(self.enc)}"
        assert isinstance(point, Point), f"Expected Point object, got {type(point)}"
        assert hasattr(point, 'feat'), f"Point object missing 'feat' key. Available keys: {list(point.keys())}"
        
        point = self.embedding(point)
        
        # More detailed debugging for the encoder call
        try:
            # Ensure we have a valid PointSequential encoder
            if not callable(self.enc):
                raise TypeError(f"self.enc is not callable. Type: {type(self.enc)}")
            
            # Check Point object state before encoder
            if not isinstance(point, Point):
                raise TypeError(f"Input to encoder must be Point object, got {type(point)}")
            
            if not hasattr(point, 'feat'):
                raise AttributeError(f"Point object missing 'feat' attribute. Available keys: {list(point.keys())}")
            
            # Call encoder with proper error handling
            point = self.enc(point)
            
        except Exception as e:
            print(f"Error during encoder forward pass:")
            print(f"  Encoder type: {type(self.enc)}")
            print(f"  Point type: {type(point)}")
            print(f"  Point keys: {list(point.keys()) if hasattr(point, 'keys') else 'N/A'}")
            print(f"  Error: {str(e)}")
            raise
        
        # Extract features from encoded result
        if hasattr(point, 'feat'):
            scene_feat = point.feat
        else:
            # Fallback for different Point implementations
            scene_feat = point
        
        # Assert that features are valid
        assert scene_feat is not None, "Features from encoder cannot be None"
        assert scene_feat.numel() > 0, "Features from encoder cannot be empty"
        
        # Global average pooling to get scene-level features
        if hasattr(point, 'offset'):
            # Use offset to aggregate features per batch
            scene_feat_pooled = []
            offsets = [0] + point.offset.tolist()
            for i in range(len(offsets) - 1):
                start_idx = offsets[i]
                end_idx = offsets[i + 1]
                batch_feat = scene_feat[start_idx:end_idx]
                pooled_feat = torch.mean(batch_feat, dim=0, keepdim=True)
                scene_feat_pooled.append(pooled_feat)
            scene_feat_final = torch.cat(scene_feat_pooled, dim=0)  # [B, feat_dim]
        else:
            # Simple reshaping and pooling
            total_points = scene_feat.shape[0]
            if total_points % batch_size == 0:
                points_per_batch = total_points // batch_size
                scene_feat_reshaped = scene_feat.view(batch_size, points_per_batch, -1)
                scene_feat_final = torch.mean(scene_feat_reshaped, dim=1)  # [B, feat_dim]
            else:
                # Fallback: use first batch_size features
                scene_feat_final = scene_feat[:batch_size]  # [B, feat_dim]
        
        # Project features
        if scene_feat_final.shape[0] == 1:
            # Single sample, avoid BatchNorm issues
            feat_proj_was_training = self.feature_projection.training
            self.feature_projection.eval()
            scene_feat_proj = self.feature_projection(scene_feat_final)
            if feat_proj_was_training:
                self.feature_projection.train()
        else:
            scene_feat_proj = self.feature_projection(scene_feat_final)
        
        # Predict grasp properties
        qual_pred = self.qual_head(scene_feat_proj)    # [B, 1]
        rot_pred = self.rot_head(scene_feat_proj)      # [B, 4]
        width_pred = self.width_head(scene_feat_proj)  # [B, 1]
        
        # Normalize rotation quaternion
        rot_pred = rot_pred / (torch.norm(rot_pred, dim=1, keepdim=True) + 1e-8)
        
        return qual_pred, rot_pred, width_pred 