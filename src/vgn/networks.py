from builtins import super

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.vgn.ConvONets.conv_onet.config import get_model, get_model_targo, get_model_targo_ptv3, get_model_ptv3_scene, get_model_ptv3_clip

def get_network(name):
    models = {
        "vgn": ConvNet,
        "giga_aff": GIGAAff,
        "giga": GIGA,
        "giga_geo": GIGAGeo,
        "giga_detach": GIGADetach,
        "targo":TARGONet,
        "targo_full_targ": TARGONet,
        "targo_hunyun2": TARGONet,
        "targo_partial": TARGONet,
        "targo_full_gt": TARGONet,
        "targo_ptv3": TARGOPtv3Net,
        "ptv3_scene": Ptv3SceneNet,
        "ptv3_clip": Ptv3ClipNet,
        "ptv3_clip_gt": Ptv3ClipNet,
        "ptv3_scene_gt": Ptv3SceneNet,
    }
    return models[name.lower()]()


def load_network(path, device, model_type=None):
    """Construct the neural network and load parameters from the specified file.

    Args:
        path: Path to the model parameters. The name must conform to `vgn_name_[_...]`.

    """
    if model_type is None:
        model_name = '_'.join(path.stem.split("_")[1:-1])
    else:
        model_name = model_type
    print(f'Loading [{model_type}] model from {path}')
    net = get_network(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net


def conv(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


def conv_stride(in_channels, out_channels, kernel_size):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2
    )


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(1, [16, 32, 64], [5, 3, 3])
        self.decoder = Decoder(64, [64, 32, 16], [3, 3, 5])
        self.conv_qual = conv(16, 1, 5)
        self.conv_rot = conv(16, 4, 5)
        self.conv_width = conv(16, 1, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        qual_out = torch.sigmoid(self.conv_qual(x))
        rot_out = F.normalize(self.conv_rot(x), dim=1)
        width_out = self.conv_width(x)
        return qual_out, rot_out, width_out

def GIGAAff():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': False,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32
    }
    return get_model(config)

def GIGA():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': True,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32
    }
    return get_model(config)


def GIGAGeo():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': True,
        'tsdf_only': True,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32
    }
    return get_model(config)

def GIGADetach():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': True,
        'detach_tsdf': True,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32
    }
    return get_model(config)

def TARGONet():
    attention_params  = {
            'self_attention': {
                'linear_att': False,
                'num_layers': 3,
                'kernel_size': 3,
                'stride': 1,
                'dilation': 1,
                'num_heads': 2  # 2
            },
            'pointnet_cross_attention': {
                'pnt2s': True,
                'nhead': 2,
                'd_feedforward': 64,
                'dropout': 0,
                'transformer_act': 'relu',
                'pre_norm': True,
                'attention_type': 'dot_prod',
                'sa_val_has_pos_emb': True,
                'ca_val_has_pos_emb': True,
                'num_encoder_layers': 2,
                'transformer_encoder_has_pos_emb': True
            }
        }
    config = {
        'model_type': 'targo',
        'd_model': 32,
        'cross_att_key': 'pointnet_cross_attention',
        'num_attention_layers': 2,  # 2,0
        'attention_params': attention_params,
         'return_intermediate': False,
        'encoder': 'voxel_simple_local_without_3d',

        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'grid_resolution': 40,
            'in_channels_scale': 2,
            'unet3d': False,
            'unet3d_kwargs':{
            'num_levels': 3,
            'f_maps': 64,
            'in_channels': 64,
            'out_channels': 64},
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32 
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': False,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 64,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 64
    }
    
    return get_model_targo(config)

def TARGOPtv3Net():
    """
    TARGO network using PointTransformerV3 as encoder instead of TransformerFusionModel
    """
    config = {
        'model_type': 'targo_ptv3',
        'd_model': 64,  # Increased to match PTv3 output
        'cross_att_key': 'pointnet_cross_attention',
        'num_attention_layers': 0,  # No transformer fusion layers needed
        'attention_params': {},  # Empty since we use PTv3
        'return_intermediate': False,
        'encoder': 'voxel_simple_local_without_3d',

        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'grid_resolution': 40,
            'in_channels_scale': 2,
            'unet3d': False,
            'unet3d_kwargs':{
                'num_levels': 3,
                'f_maps': 64,
                'in_channels': 64,
                'out_channels': 64
            },
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32 
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': False,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 64,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 64
    }
    
    return get_model_targo_ptv3(config)

def Ptv3SceneNet():
    """
    Network configuration for PTv3 Scene model - only processes scene point cloud.
    """
    cfg = {
        'model_type': 'ptv3_scene',
        'd_model': 64,
        'cross_att_key': 'pointnet_cross_attention',
        'num_attention_layers': 0,
        'encoder': 'voxel_simple_local_without_3d',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 4,
                'merge_mode': 'concat',
                'start_filts': 32
            },
        },
        'decoder': 'simple_local',
        'decoder_kwargs': {
            'sample_mode': 'bilinear',
            'hidden_size': 32,
        },
        'padding': 0,
        'c_dim': 64,
    }
    
    return get_model_ptv3_scene(cfg)

def Ptv3ClipNet():
    """
    Network configuration for PTv3 CLIP model - processes scene point cloud with CLIP features.
    """
    cfg = {
        'model_type': 'ptv3_clip',
        'd_model': 64,
        'cross_att_key': 'pointnet_cross_attention',
        'num_attention_layers': 0,
        'encoder': 'voxel_simple_local_without_3d',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 4,
                'merge_mode': 'concat',
                'start_filts': 32
            },
        },
        'decoder': 'simple_local',
        'decoder_kwargs': {
            'sample_mode': 'bilinear',
            'hidden_size': 32,
        },
        'padding': 0,
        'c_dim': 64,
    }
    
    return get_model_ptv3_clip(cfg)


class Encoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv_stride(in_channels, filters[0], kernels[0])
        self.conv2 = conv_stride(filters[0], filters[1], kernels[1])
        self.conv3 = conv_stride(filters[1], filters[2], kernels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv(in_channels, filters[0], kernels[0])
        self.conv2 = conv(filters[0], filters[1], kernels[1])
        self.conv3 = conv(filters[1], filters[2], kernels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = F.interpolate(x, 10)
        x = self.conv2(x)
        x = F.relu(x)

        x = F.interpolate(x, 20)
        x = self.conv3(x)
        x = F.relu(x)

        x = F.interpolate(x, 40)
        return x


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx 
    
def count_num_trainable_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)