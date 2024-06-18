import torch
import torch.distributions as dist
from torch import nn
import os
from vgn.ConvONets.encoder import encoder_dict
from vgn.ConvONets.conv_onet import models, training
from vgn.ConvONets.conv_onet import generation
from vgn.ConvONets import data
from vgn.ConvONets import config
from vgn.ConvONets.common import decide_total_volume_range, update_reso
from torchvision import transforms
import numpy as np
import torch.nn as nn
from transformer.fusion_model import TransformerFusionModel, TransformerSceneModel

def get_model_grid(cfg, device=None, dataset=None, **kwargs):
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

    ##----------------in different fusion, the decoder is same----------------## 
    detach_tsdf = 'detach_tsdf' in cfg.keys() and cfg['detach_tsdf']

    if tsdf_only:
        decoders = []
    else:
        # if not cfg['add_single_supervision']:
        out_dim_qual, out_dim_rot, out_dim_width = 1,4,1
        # else:
        #     out_dim_qual, out_dim_rot, out_dim_width = 2,8,2

        decoder_qual = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=out_dim_qual,fusion_type=cfg['fusion_type'],
            **decoder_kwargs
        )
        decoder_rot = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=out_dim_rot,fusion_type=cfg['fusion_type'],
            **decoder_kwargs
        )
        decoder_width = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=out_dim_width,fusion_type=cfg['fusion_type'],
            **decoder_kwargs
        )
        if not cfg['add_single_supervision']:
            decoders = [decoder_qual, decoder_rot, decoder_width]
        else:
            decoder_qual_s = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=out_dim_qual,
            **decoder_kwargs
            )
            decoder_rot_s = models.decoder_dict[decoder](
                c_dim=c_dim, padding=padding, out_dim=out_dim_rot,
                **decoder_kwargs
            )
            decoder_width_s = models.decoder_dict[decoder](
                c_dim=c_dim, padding=padding, out_dim=out_dim_width,
                **decoder_kwargs
            )
            decoders = [decoder_qual, decoder_rot, decoder_width, decoder_qual_s, decoder_rot_s, decoder_width_s]
    
    if cfg['decoder_tsdf'] or tsdf_only:
        decoder_tsdf = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoders.append(decoder_tsdf)

    if cfg['model_type'] == "afford_scene_pc":
            encoder_in = TransformerSceneModel(cfg['attention_params'], cfg['num_attention_layers'],\
                 cfg['return_intermediate'], cfg['cross_att_key'], cfg['d_model'], fusion_type=cfg['fusion_type'])
            encoder_aff_scene = encoder_dict[encoder](
                c_dim=c_dim, padding=padding,
                fusion_type=cfg['fusion_type'],
                # num_layers=cfg['num_layers'],
                **encoder_kwargs
            )
            encoders_in = [encoder_in]

    if cfg['model_type'] == "giga_aff_plus_target_occluder_grid":
        # if cfg['fusion_type'] == "MLP_fusion":
        if cfg['fusion_type'] in ("MLP_fusion", "CNN_concat", "CNN_add"):
        # if cfg['fusion_type'] in ("MLP_fusion", "transformer_fusion", "transformer_concat"):
            if cfg['shared_weights'] == False:
                encoder_in_targ = nn.Conv3d(1, c_dim, 3, padding=1)
                encoder_in_scene = nn.Conv3d(1, c_dim, 3, padding=1)
                encoder_in_occ = nn.Conv3d(1, c_dim, 3, padding=1)
                encoders_in = [encoder_in_targ, encoder_in_scene, encoder_in_occ]
                encoder_aff_scene = encoder_dict[encoder](
                    c_dim=c_dim, padding=padding,
                    fusion_type=cfg['fusion_type'],
                    **encoder_kwargs
                )
            else:
                encoder_in = nn.Conv3d(1, c_dim, 3, padding=1)
                encoders_in = [encoder_in]
                encoder_aff_scene = encoder_dict[encoder](
                    c_dim=c_dim, padding=padding,
                    fusion_type=cfg['fusion_type'],
                    **encoder_kwargs
                )
        elif cfg['fusion_type'] in ('transformer_query_scene','transformer_query_target', 'transformer_concat'):
            # encoder_in = nn.Conv3d(1, c_dim, 3, padding=1)            
            encoder_in = TransformerFusionModel(cfg['attention_params'], cfg['num_attention_layers'],\
                 cfg['return_intermediate'], cfg['cross_att_key'], cfg['d_model'], fusion_type=cfg['fusion_type'])
            encoder_aff_scene = encoder_dict[encoder](
                c_dim=c_dim, padding=padding,
                fusion_type=cfg['fusion_type'],
                # num_layers=cfg['num_layers'],
                **encoder_kwargs
            )
            encoders_in = [encoder_in]

    if cfg['model_type'] in ("afford_scene_targ_pc", "giga_aff_plus_occluder_grid"):
        # if cfg['fusion_type'] == "MLP_fusion":
        # if cfg['fusion_type'] in ("MLP_fusion", "CNN_concat"):
        if cfg['fusion_type'] in ("MLP_fusion", "CNN_concat", "CNN_add"):
        # if cfg['fusion_type'] in ("MLP_fusion", "transformer_fusion", "transformer_concat"):
            if cfg['shared_weights'] == False:
                encoder_in_other = nn.Conv3d(1, c_dim, 3, padding=1)
                encoder_in_scene = nn.Conv3d(1, c_dim, 3, padding=1)
                encoders_in = [encoder_in_other, encoder_in_scene]
                encoder_aff_scene = encoder_dict[encoder](
                    c_dim=c_dim, padding=padding,
                    fusion_type=cfg['fusion_type'],
                    **encoder_kwargs
                )
            else:
                encoder_in = nn.Conv3d(1, c_dim, 3, padding=1)
                encoders_in = [encoder_in]
                encoder_aff_scene = encoder_dict[encoder](
                    c_dim=c_dim, padding=padding,
                    fusion_type=cfg['fusion_type'],
                    **encoder_kwargs
                )
                
        # elif cfg['fusion_type'] == "transformer_fusion":
        elif cfg['fusion_type'] in ('transformer_query_scene','transformer_query_target', 'transformer_concat'):
            assert cfg['model_type'] == "afford_scene_targ_pc"
            encoder_in = TransformerFusionModel(cfg['attention_params'], cfg['num_attention_layers'],\
                 cfg['return_intermediate'], cfg['cross_att_key'], cfg['d_model'], fusion_type=cfg['fusion_type'])
            encoder_aff_scene = encoder_dict[encoder](
                c_dim=c_dim, padding=padding,
                fusion_type=cfg['fusion_type'],
                **encoder_kwargs
            )
            encoders_in = [encoder_in]
    
        
    model = models.ConvolutionalOccupancyNetwork_Grid(
        decoders, encoders_in, encoder_aff_scene,device=device, detach_tsdf=detach_tsdf,
        shared_weights = cfg['shared_weights'], add_single_supervision= cfg['add_single_supervision'],
        model_type = cfg['model_type'], fusion_type = cfg['fusion_type'], attention_params = cfg['attention_params'])
    return model


    if 'encoder_tsdf' not in cfg:
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
    else:
        if cfg['encoder_tsdf'] == True:
            encoder_tsdf = encoder_dict[encoder](
                c_dim=c_dim, padding=padding,
                **encoder_kwargs
            )
            encoder_aff = encoder_dict[encoder](
                c_dim=c_dim, padding=padding,
                **encoder_kwargs
            )
            encoders = [encoder_tsdf, encoder_aff]
            model = models.ConvolutionalOccupancyNetwork_Sequential(
                decoders, encoders, device=device, detach_tsdf=detach_tsdf
            )

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
                # if not cfg['add_single_supervision']:
        out_dim_qual, out_dim_rot, out_dim_width = 1,4,1
        # else:
        #     out_dim_qual, out_dim_rot, out_dim_width = 2,8,2

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
        if 'add_single_supervision' not in cfg.keys():
            decoders = [decoder_qual, decoder_rot, decoder_width]
        else:
            if not cfg['add_single_supervision']:
                decoders = [decoder_qual, decoder_rot, decoder_width]
            else:
                decoder_qual_s = models.decoder_dict[decoder](
                c_dim=c_dim, padding=padding, out_dim=out_dim_qual,
                **decoder_kwargs
                )
                decoder_rot_s = models.decoder_dict[decoder](
                    c_dim=c_dim, padding=padding, out_dim=out_dim_rot,
                    **decoder_kwargs
                )
                decoder_width_s = models.decoder_dict[decoder](
                    c_dim=c_dim, padding=padding, out_dim=out_dim_width,
                    **decoder_kwargs
                )
            decoders = [decoder_qual, decoder_rot, decoder_width, decoder_qual_s, decoder_rot_s, decoder_width_s]
    if cfg['decoder_tsdf'] or tsdf_only:
        decoder_tsdf = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoders.append(decoder_tsdf)
    if 'type_modal' in cfg and cfg['type_modal'] == 'GIGAff_Occluder':
            encoder_targ = encoder_dict[encoder](
                c_dim=c_dim, padding=padding,
                **encoder_kwargs
            )
            encoder_scene = encoder_dict[encoder](
                c_dim=c_dim, padding=padding,
                **encoder_kwargs
            )
            encoder_occ = encoder_dict[encoder](
                c_dim=c_dim, padding=padding,
                **encoder_kwargs
            )
            encoders = [encoder_targ, encoder_scene, encoder_occ]
            model = models.ConvolutionalOccupancyNetwork_Occluder(
                decoders, encoders, device=device, detach_tsdf=detach_tsdf
            )
            return model


    if 'encoder_tsdf' not in cfg:
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
    else:
        if cfg['encoder_tsdf'] == True:
            encoder_tsdf = encoder_dict[encoder](
                c_dim=c_dim, padding=padding,
                **encoder_kwargs
            )
            encoder_aff = encoder_dict[encoder](
                c_dim=c_dim, padding=padding,
                **encoder_kwargs
            )
            encoders = [encoder_tsdf, encoder_aff]
            model = models.ConvolutionalOccupancyNetwork_Sequential(
                decoders, encoders, device=device, detach_tsdf=detach_tsdf
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
