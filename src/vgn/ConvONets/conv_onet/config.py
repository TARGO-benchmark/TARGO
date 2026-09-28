import torch
import torch.distributions as dist
from torch import nn
import os
import sys

from vgn.ConvONets.encoder import encoder_dict
from vgn.ConvONets.conv_onet import models, training
from vgn.ConvONets.common import decide_total_volume_range, update_reso

def get_model_targo(cfg, device=None, dataset=None, **kwargs):
    ''' Return the original TARGO model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['decoder']
    encoder = cfg['encoder']
    c_dim = cfg['c_dim']
    decoder_kwargs = cfg['decoder_kwargs']
    encoder_kwargs = cfg['encoder_kwargs']
    padding = cfg['padding']
    if padding is None:
        padding = 0.1
    
    # for pointcloud_crop
    try: 
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except:
        pass
    # local positional encoding
    if 'local_coord' in cfg.keys():
        encoder_kwargs['local_coord'] = cfg['local_coord']
        decoder_kwargs['local_coord'] = cfg['local_coord']
    if 'pos_encoding' in cfg:
        encoder_kwargs['pos_encoding'] = cfg['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['pos_encoding']

    tsdf_only = 'tsdf_only' in cfg.keys() and cfg['tsdf_only']

    ##----------------in different fusion, the decoder is same----------------## 
    detach_tsdf = 'detach_tsdf' in cfg.keys() and cfg['detach_tsdf']

    if tsdf_only:
        decoders = []
    else:
        out_dim_qual, out_dim_rot, out_dim_width = 1,4,1
        decoder_qual = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=out_dim_qual, 
            **decoder_kwargs
        )
        decoder_rot = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=out_dim_rot,
            **decoder_kwargs
        )
        decoder_width = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=out_dim_width,
            **decoder_kwargs
        )
        
        decoders = [decoder_qual, decoder_rot, decoder_width]

    # Import TransformerFusionModel only when needed (for original targo model)
    from transformer.fusion_model import TransformerFusionModel
    encoder_in = TransformerFusionModel(cfg['attention_params'], cfg['num_attention_layers'],\
            cfg['return_intermediate'], cfg['cross_att_key'], cfg['d_model'])
    
    encoder_aff_scene = encoder_dict[encoder](
        c_dim=c_dim, padding=padding,
        **encoder_kwargs
    )
    encoders_in = [encoder_in]
    
    model = models.ConvolutionalOccupancyNetwork_Grid(
        decoders, encoders_in, encoder_aff_scene,device=device, detach_tsdf=detach_tsdf, model_type='targo')
    return model

def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['decoder']
    encoder = cfg['encoder']
    c_dim = cfg['c_dim']
    decoder_kwargs = cfg['decoder_kwargs']
    encoder_kwargs = cfg['encoder_kwargs']
    padding = cfg['padding']
    if padding is None:
        padding = 0.1
    
    # for pointcloud_crop
    try: 
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except:
        pass
    # local positional encoding
    if 'local_coord' in cfg.keys():
        encoder_kwargs['local_coord'] = cfg['local_coord']
        decoder_kwargs['local_coord'] = cfg['local_coord']
    if 'pos_encoding' in cfg:
        encoder_kwargs['pos_encoding'] = cfg['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['pos_encoding']

    tsdf_only = 'tsdf_only' in cfg.keys() and cfg['tsdf_only']
    detach_tsdf = 'detach_tsdf' in cfg.keys() and cfg['detach_tsdf']

    if tsdf_only:
        decoders = []
    else:
        decoder_qual = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoder_rot = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=4,
            **decoder_kwargs
        )
        decoder_width = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoders = [decoder_qual, decoder_rot, decoder_width]
    if cfg['decoder_tsdf'] or tsdf_only:
        decoder_tsdf = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoders.append(decoder_tsdf)

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    if tsdf_only:
        model = models.ConvolutionalOccupancyNetworkGeometry(
            decoder_tsdf, encoder, device=device
        )
    else:
        model = models.ConvolutionalOccupancyNetwork(
            decoders, encoder, device=device, detach_tsdf=detach_tsdf
        )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
    )

    return trainer


