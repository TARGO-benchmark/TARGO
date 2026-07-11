import torch
import torch.distributions as dist
from torch import nn
import os
import sys
sys.path.append('.')

from src.vgn.ConvONets.encoder import encoder_dict
from src.vgn.ConvONets.conv_onet import models, training
from src.vgn.ConvONets.conv_onet import generation
# from src.vgn.ConvONets import data  # Commented out as data module doesn't exist
from src.vgn.ConvONets.common import decide_total_volume_range, update_reso
# Conditional import - only import TransformerFusionModel when needed (not for targo_ptv3/ptv3_scene)
# TransformerFusionModel will be imported inside get_model_targo function

def get_model_targo_ptv3(cfg, device=None, dataset=None, **kwargs):
    ''' Return the TARGO model with PointTransformerV3 encoder.

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

    # Import PointTransformerV3FusionModel only when needed (for targo_ptv3 model)
    from src.transformer.ptv3_fusion_model import PointTransformerV3FusionModel
    encoder_in = PointTransformerV3FusionModel()
    
    encoder_aff_scene = encoder_dict[encoder](
        c_dim=c_dim, padding=padding,
        **encoder_kwargs
    )
    encoders_in = [encoder_in]
    
    model = models.ConvolutionalOccupancyNetwork_Grid(
        decoders, encoders_in, encoder_aff_scene, device=device, 
        detach_tsdf=detach_tsdf, model_type='targo_ptv3')
    return model

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
    from src.transformer.fusion_model import TransformerFusionModel
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


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    
    if cfg['data']['input_type'] == 'pointcloud_crop':
        # calculate the volume boundary
        query_vol_metric = cfg['data']['padding'] + 1
        unit_size = cfg['data']['unit_size']
        recep_field = 2**(cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)
        if 'unet' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet_kwargs']['depth']
        elif 'unet3d' in cfg['model']['encoder_kwargs']:
            depth = cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels']
        
        vol_info = decide_total_volume_range(query_vol_metric, recep_field, unit_size, depth)
        
        grid_reso = cfg['data']['query_vol_size'] + recep_field - 1
        grid_reso = update_reso(grid_reso, depth)
        query_vol_size = cfg['data']['query_vol_size'] * unit_size
        input_vol_size = grid_reso * unit_size
        # only for the sliding window case
        vol_bound = None
        if cfg['generation']['sliding_window']:
            vol_bound = {'query_crop_size': query_vol_size,
                         'input_crop_size': input_vol_size,
                         'fea_type': cfg['model']['encoder_kwargs']['plane_type'],
                         'reso': grid_reso}

    else: 
        vol_bound = None
        vol_info = None

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info = vol_info,
        vol_bound = vol_bound,
    )
    return generator


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        if input_type != 'pointcloud_crop':
            fields['points'] = data.PointsField(
                cfg['data']['points_file'], points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )
        else:
            fields['points'] = data.PatchPointsField(
                cfg['data']['points_file'], 
                transform=points_transform,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
            )

    
    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            if input_type == 'pointcloud_crop':
                fields['points_iou'] = data.PatchPointsField(
                points_iou_file,
                unpackbits=cfg['data']['points_unpackbits'],
                multi_files=cfg['data']['multi_files']
                )
            else:
                fields['points_iou'] = data.PointsField(
                    points_iou_file,
                    unpackbits=cfg['data']['points_unpackbits'],
                    multi_files=cfg['data']['multi_files']
                )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields

def get_model_ptv3_scene(cfg, device=None, dataset=None):
    """
    Create PTv3 Scene model that only processes scene point cloud.
    
    Args:
        cfg: Configuration dictionary
        device: Device to place model on
        dataset: Dataset (optional)
    
    Returns:
        ConvolutionalOccupancyNetwork_Grid model with PTv3 scene encoder
    """
    decoder = cfg['decoder']
    encoder = cfg['encoder']
    c_dim = cfg['c_dim']
    decoder_kwargs = cfg['decoder_kwargs']
    encoder_kwargs = cfg['encoder_kwargs']
    padding = cfg.get('padding', 0.1)
    local_coord = cfg.get('local_coord', False)
    pos_encoding = cfg.get('pos_encoding', 'linear')

    # Handle optional parameters
    tsdf_only = cfg.get('tsdf_only', False)
    detach_tsdf = cfg.get('detach_tsdf', False)

    # Initialize decoders
    decoder_kwargs['c_dim'] = c_dim
    decoder_kwargs['padding'] = padding
    # Remove out_dim from decoder_kwargs to avoid conflict
    if tsdf_only:
        decoder_kwargs['tsdf_only'] = tsdf_only
    if detach_tsdf:
        decoder_kwargs['detach_tsdf'] = detach_tsdf

    # Create decoders with proper out_dim
    decoder_qual = models.decoder_dict[decoder](out_dim=1, **decoder_kwargs)
    decoder_rot = models.decoder_dict[decoder](out_dim=4, **decoder_kwargs)
    decoder_width = models.decoder_dict[decoder](out_dim=1, **decoder_kwargs)
    
    # Create decoders list
    decoders = [decoder_qual, decoder_rot, decoder_width]

    # Import PointTransformerV3SceneModel only when needed (for ptv3_scene model)
    from src.transformer.ptv3_scene_model import PointTransformerV3SceneModel
    encoder_in = PointTransformerV3SceneModel()
    
    # Create encoder_aff (scene encoder)
    encoder_aff_scene = encoder_dict[encoder](
        c_dim=c_dim, padding=padding,
        **encoder_kwargs
    )
    
    # Create encoders_in list
    encoders_in = [encoder_in]

    # Create the model with correct parameter format
    model = models.ConvolutionalOccupancyNetwork_Grid(
        decoders, encoders_in, encoder_aff_scene,
        device=device, detach_tsdf=detach_tsdf, model_type='ptv3_scene'
    )

    return model

def get_model_ptv3_clip(cfg, device=None, dataset=None):
    """
    Create PTv3 CLIP model that processes scene point cloud with CLIP features.
    
    Args:
        cfg: Configuration dictionary
        device: Device to place model on
        dataset: Dataset (optional)
    
    Returns:
        ConvolutionalOccupancyNetwork_Grid model with PTv3 CLIP encoder
    """
    decoder = cfg['decoder']
    encoder = cfg['encoder']
    c_dim = cfg['c_dim']
    decoder_kwargs = cfg['decoder_kwargs']
    encoder_kwargs = cfg['encoder_kwargs']
    padding = cfg.get('padding', 0.1)
    local_coord = cfg.get('local_coord', False)
    pos_encoding = cfg.get('pos_encoding', 'linear')

    # Handle optional parameters
    tsdf_only = cfg.get('tsdf_only', False)
    detach_tsdf = cfg.get('detach_tsdf', False)

    # Initialize decoders
    decoder_kwargs['c_dim'] = c_dim
    decoder_kwargs['padding'] = padding
    # Remove out_dim from decoder_kwargs to avoid conflict
    if tsdf_only:
        decoder_kwargs['tsdf_only'] = tsdf_only
    if detach_tsdf:
        decoder_kwargs['detach_tsdf'] = detach_tsdf

    # Create decoders with proper out_dim
    decoder_qual = models.decoder_dict[decoder](out_dim=1, **decoder_kwargs)
    decoder_rot = models.decoder_dict[decoder](out_dim=4, **decoder_kwargs)
    decoder_width = models.decoder_dict[decoder](out_dim=1, **decoder_kwargs)
    
    # Create decoders list
    decoders = [decoder_qual, decoder_rot, decoder_width]

    # Import PointTransformerV3CLIPModel only when needed (for ptv3_clip model)
    from sys import path
    path.append('.')
    from src.transformer.ptv3_clip_model import PointTransformerV3CLIPModel
    encoder_in = PointTransformerV3CLIPModel()
    
    # Create encoder_aff (scene encoder)
    encoder_aff_scene = encoder_dict[encoder](
        c_dim=c_dim, padding=padding,
        **encoder_kwargs
    )
    
    # Create encoders_in list
    encoders_in = [encoder_in]

    # Create the model with correct parameter format
    model = models.ConvolutionalOccupancyNetwork_Grid(
        decoders, encoders_in, encoder_aff_scene,
        device=device, detach_tsdf=detach_tsdf, model_type='ptv3_clip'
    )

    return model
