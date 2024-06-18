from builtins import super
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from vgn.ConvONets.conv_onet.config import get_model, get_model_grid


def get_network_vgn(name):
    models = {
        "vgn": ConvNet,
    }
    return models[name.lower()]()

def get_network(name, shared_weights, add_single_supervision,fusion_type, feat_type, num_encoder_layers = None):
    models = {
        "vgn": ConvNet,
        "giga_aff": GIGAAff,
        "giga": GIGA,
        "giga_hr": GIGA,
        "giga_geo": GIGAGeo,
        "giga_detach": GIGADetach,
        "afford_scene_pc": AffordScenePC,

        "giga_aff_with_geo": GIGAAff_With_Geo,

        "giga_aff_plus_target_input":GIGAAff_With_Single_Input_Fusion,
        "giga_aff_plus_occluder_input": GIGAAff_With_Single_Input_Fusion,
        "giga_aff_plus_target_occluder_input": GIGAAff_With_Double_Input_Fusion,
 
        "giga_aff_plus_target_occluder_grid":GIGAAff_With_Double_Grids,
        "afford_scene_targ_pc":GIGAAff_With_Single_Grid,

        "giga_aff_decouple_target_occluder_input": GIGAAff_With_Double_Input_Fusion,

        "giga_aff_decouple_target_occluder_grid": GIGAAff_With_Double_Grids,

        # "giga_aff_plus_occluder_grid":GIGAAff_With_Single_Grid,
        # "giga_aff_plus_occluder_grid":GIGAAff_With_Single_Grid,
        # "giga_aff_plus_target_occluder_plane":GIGAAff_With_Double_Planes,
        # "giga_aff_plus_target_plane":GIGAAff_With_Single_Plane,
        #  "giga_aff_plus_occluder_plane":GIGAAff_With_Single_Plane,

    }
    if name in ("vgn", "giga_aff", "giga_hr", "giga", "giga_geo", "giga_detach", "giga_aff_plus_occluder_input"):
        return models[name.lower()]()
    if name == "afford_scene_pc":
        return models[name.lower()](shared_weights,add_single_supervision,fusion_type, feat_type,name, num_encoder_layers)
    if name in ("giga_aff_plus_target_occluder_plane", "giga_aff_plus_target_occluder_grid", 
                "afford_scene_targ_pc", "giga_aff_plus_target_plane"):
        # return models[name.lower()](shared_weights, add_single_supervision)
        return models[name.lower()](shared_weights,add_single_supervision,fusion_type, feat_type,name, num_encoder_layers)
    if name in ("giga_aff_plus_occluder_plane","giga_aff_plus_occluder_grid"):
        # return models[name.lower()](shared_weights)
        return models[name.lower()](shared_weights, name)
    if name in ("giga_aff_plus_target_input", "giga_aff_plus_target_occluder_input"):
        # return models[name.lower()](add_single_supervision)
        return models[name.lower()](add_single_supervision, name)


def load_network(path, device, model_type,shared_weights, add_single_supervision, fusion_type, feat_type, num_encoder_layers):
    """Construct the neural network and load parameters from the specified file.

    Args:
        path: Path to the model parameters. The name must conform to `vgn_name_[_...]`.

    """
    if model_type is None:
        model_name = '_'.join(path.stem.split("_")[1:-1])
    else:
        model_name = model_type
    print(f'Loading [{model_type}] model from {path}')
    net = get_network(model_name,shared_weights, add_single_supervision, fusion_type=fusion_type, feat_type=feat_type, num_encoder_layers= num_encoder_layers).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net

def load_network_vgn(path, device, model_type):
    """Construct the neural network and load parameters from the specified file.

    Args:
        path: Path to the model parameters. The name must conform to `vgn_name_[_...]`.

    """
    if model_type is None:
        model_name = '_'.join(path.stem.split("_")[1:-1])
    else:
        model_name = model_type
    print(f'Loading [{model_type}] model from {path}')
    net = get_network_vgn(model_name).to(device)
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
    
    
def GIGAAff_With_Single_Input_Fusion(add_single_supervision, model_name):
    config = {
        # 'model_type': 'GIGAAff_With_Single_Input_Fusion',
        'model_type': model_name,
        'encoder': 'voxel_simple_local_with_input_fushion',
        'add_single_supervision': add_single_supervision,
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'input_dim': 2,
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
    # if add_single_supervision:
    #     config['decoder_kwargs']['out_dim'] = 2
    # else:
    #     config['decoder_kwargs']['out_dim'] = 1
    return get_model(config)

def GIGAAff_With_Double_Input_Fusion(add_single_supervision,model_name):
    config = {
        'encoder': 'voxel_simple_local_with_input_fushion',
        'add_single_supervision': add_single_supervision,
        'model_type': model_name,
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'input_dim': 3,
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
    # if add_single_supervision:
    #     config['decoder_kwargs']['out_dim'] = 2
    # else:
    #     config['decoder_kwargs']['out_dim'] = 1
    return get_model(config)

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

def AffordScenePC(shared_weights, add_single_supervision, fusion_type, feat_type, model_name,num_encoder_layers):
    # if fusion_type in ('transformer_query_scene','transformer_query_target', 'transformer_concat'):
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
    
    if feat_type == 'Plane_feat':
        plane_type = ['xz', 'xy', 'yz']
    else:
        plane_type = ['grid']
    config = {
        'shared_weights': shared_weights,
        'add_single_supervision': add_single_supervision,
        'model_type': model_name,
        'fusion_type': fusion_type,
        'd_model': 32,
        'cross_att_key': 'pointnet_cross_attention',
        'num_attention_layers': num_encoder_layers,  # 2,0
        'attention_params': attention_params,
         'return_intermediate': False,
        'encoder': 'voxel_simple_local_without_3d',
        'encoder_kwargs': {
            # 'plane_type': ['xz', 'xy', 'yz'],
            'plane_type': plane_type,
            'plane_resolution': 40,
            'grid_resolution': 40,
            # 'stride_ds': 3,
            'in_channels_scale': 2,
            'unet3d': True if feat_type == 'Grid_feat' else False,
            # 'unet3d': True,
            'unet3d_kwargs':{
            'num_levels': 3,
            'f_maps': 64,
            'in_channels': 64,
            'out_channels': 64},
            'unet': True if feat_type == 'Plane_feat' else False,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32 
                # 'start_filts': 32
            }
        },
        # 'type_modal': 'GIGAff_Occluder',
        'decoder': 'simple_local',
        'decoder_tsdf': False,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            # 'concat_feat': True
            # 'concat_feat': False    ## 3d unet
            'concat_feat': False if feat_type == 'Grid_feat' else True
        },
        'padding': 0,
        'c_dim': 32
    }
    
    return get_model_grid(config)


def GIGAAff_With_Single_Grid(shared_weights, add_single_supervision, fusion_type, feat_type, model_name,num_encoder_layers ):
    # if fusion_type in ('transformer_fusion','transformer_concat') :
    if fusion_type in ('transformer_query_scene','transformer_query_target', 'transformer_concat'):
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
    else:
        attention_params = None
    if feat_type == 'Plane_feat':
        plane_type = ['xz', 'xy', 'yz']
    else:
        plane_type = ['grid']
    config = {
        'shared_weights': shared_weights,
        'add_single_supervision': add_single_supervision,
        'model_type': model_name,
        'fusion_type': fusion_type,
        'd_model': 32,
        'cross_att_key': 'pointnet_cross_attention',
        'num_attention_layers': num_encoder_layers,  # 2,0
        'attention_params': attention_params,
         'return_intermediate': False,
        'encoder': 'voxel_simple_local_without_3d',
        # 'reso'
        # 'encoder_kwargs': {
        #     'in_channels_scale' : 2,
        #     'plane_type': plane_type,
        #     'plane_resolution': 40,
        #     'grid_resolution': 40,
       
        #     'unet3d': True,
        #     'unet3d_kwargs':{
        #     'num_levels': 3,
        #     'f_maps': 32,
        #     'in_channels': 32,
        #     'out_channels': 32}
        # },

        # # 'type_modal': 'GIGAff_Occluder',
        # 'decoder': 'simple_local',
        # 'decoder_tsdf': False,
        # 'decoder_kwargs': {
        #     'dim': 3,
        #     'sample_mode': 'bilinear',
        #     'hidden_size': 32,
        #     'concat_feat': True
        #     # 'concat_feat': False    ## 3d unet
        # },
        # 'padding': 0,
        # 'c_dim': 32,

        'encoder_kwargs': {
            # 'plane_type': ['xz', 'xy', 'yz'],
            'plane_type': plane_type,
            'plane_resolution': 40,
            'grid_resolution': 40,
            # 'stride_ds': 3,
            'in_channels_scale': 2,
            'unet3d': True if feat_type == 'Grid_feat' else False,
            # 'unet3d': True,
            'unet3d_kwargs':{
            'num_levels': 3,
            'f_maps': 64,
            'in_channels': 64,
            'out_channels': 64},
            'unet': True if feat_type == 'Plane_feat' else False,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32 
                # 'start_filts': 32
            }
        },
        # 'type_modal': 'GIGAff_Occluder',
        'decoder': 'simple_local',
        'decoder_tsdf': False,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            # 'concat_feat': True
            # 'concat_feat': False    ## 3d unet
            'concat_feat': False if feat_type == 'Grid_feat' else True
        },
        'padding': 0,
        'c_dim': 32
    }
    
    return get_model_grid(config)

def GIGAAff_With_Double_Grids(shared_weights, add_single_supervision, fusion_type, model_name):
    # if fusion_type == 'transformer_fusion':
    if fusion_type in ('transformer_query_scene','transformer_query_target', 'transformer_concat'):
        attention_params  = {
                'self_attention': {
                    'linear_att': False,
                    'num_layers': 3,
                    'kernel_size': 3,
                    'stride': 1,
                    'dilation': 1,
                    'num_heads': 2
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
    else:
        attention_params = None
    
    # if feat_type == 'plane':
        # plane_type = ['xz', 'xy', 'yz']

    config = {
        'shared_weights': shared_weights,
        'add_single_supervision': add_single_supervision,
        # 'model_type': 'GIGAAff_With_Double_Grids',
        'fusion_type': fusion_type,
        'num_attention_layers': 2,  # 2,0
        'attention_params': attention_params,
        'd_model': 32,
        'cross_att_key': 'pointnet_cross_attention',
         'return_intermediate': False,
        'model_type': model_name,
        'encoder': 'voxel_simple_local_without_3d',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            # 'stride_ds': 3,
            'in_channels_scale': 3,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        # 'type_modal': 'GIGAff_Occluder',
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
    return get_model_grid(config)

def GIGAAff_With_Geo():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_tsdf': True,
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

def GIGA_HR():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 60,
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