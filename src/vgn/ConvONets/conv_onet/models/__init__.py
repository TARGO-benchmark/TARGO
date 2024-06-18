import torch
import torch.nn as nn
from torch import distributions as dist
from vgn.ConvONets.conv_onet.models import decoder
from utils_giga import visualize_and_save_tsdf, tsdf_to_ply, pad_sequence, unpad_sequences, pad_to_target, save_point_cloud_as_ply
        # import torch
# import MinkowskiEngine as ME
from typing import List
from transformer.me_transformer_utils import LightweightSelfAttentionLayer, PositionEmbeddingLearned, create_sparse_tensor, TransformerCrossEncoderLayer, TransformerCrossEncoder, combine_cross_attention
from transformer.transformer_fushion import TransformerFusion
from torchtyping import TensorType
import numpy as np
from utils_giga import transform_pc

# Decoder dictionary
decoder_dict = {
    'simple_fc': decoder.FCDecoder,
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder
}

attention_params = {
    'self_attention': {
        'linear_att': False,
        'num_layers': 3,
        'kernel_size': 3,
        'stride': 1,
        'dilation': 1,
        'num_heads': 8
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


class ConvolutionalOccupancyNetwork_Grid(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoders, encoders_in, encoder_aff, device=None, detach_tsdf=False, 
                 add_single_supervision = False, shared_weights = False, model_type=None, 
                 fusion_type=None, attention_params = None):
        super().__init__()
        
        self.decoder_qual = decoders[0].to(device)
        self.decoder_rot = decoders[1].to(device)
        self.decoder_width = decoders[2].to(device)

        if add_single_supervision:
            self.decoder_qual_single = decoders[3].to(device)
            self.decoder_rot_single = decoders[4].to(device)
            self.decoder_width_single = decoders[5].to(device)

        if model_type == "giga_aff_plus_target_occluder_grid":
            if shared_weights == False:
                self.encoder_in_targ = encoders_in[0].to(device)
                self.encoder_in_scene = encoders_in[1].to(device)
                self.encoder_in_occ = encoders_in[2].to(device)
            else:
                self.encoder_in = encoders_in[0].to(device)
        
        if model_type == "afford_scene_targ_pc":
            if shared_weights == False:
                self.encoder_in_targ = encoders_in[0].to(device)
                self.encoder_in_scene = encoders_in[1].to(device)
            else:
                self.encoder_in = encoders_in[0].to(device)
        
        if model_type == "afford_scene_pc":
            self.encoder_in = encoders_in[0].to(device)
        
        if model_type == "giga_aff_plus_occluder_grid":
            if shared_weights == False:
                self.encoder_in_occ = encoders_in[0].to(device)
                self.encoder_in_scene = encoders_in[1].to(device)
            else:
                self.encoder_in = encoders_in[0].to(device)

        self.encoder_aff = encoder_aff.to(device)
        self._device = device

        self.detach_tsdf = detach_tsdf
        self.add_single_supervision = add_single_supervision
        self.shared_weights = shared_weights
        self.model_type = model_type
        self.fusion_type = fusion_type

    def forward(self, inputs, p, p_tsdf=None, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        
        if self.model_type == "giga_aff_plus_target_occluder_grid":
            # if self.fusion_type == "MLP_fusion":
            # if self.fusion_type in ("MLP_fusion", "CNN_concat"):
            if self.fusion_type in ("MLP_fusion", "CNN_concat", "CNN_add"):
                if not self.shared_weights:
                    feat_3d_targ = self.encoder_in_targ(inputs[0].unsqueeze(1))
                    feat_3d_scene = self.encoder_in_scene(inputs[1].unsqueeze(1))
                    feat_3d_occ = self.encoder_in_occ(inputs[2].unsqueeze(1))
                else:
                    feat_3d_targ = self.encoder_in(inputs[0].unsqueeze(1))
                    feat_3d_scene = self.encoder_in(inputs[1].unsqueeze(1))
                    feat_3d_occ = self.encoder_in(inputs[2].unsqueeze(1))
                features_fused = torch.cat([feat_3d_targ, feat_3d_scene, feat_3d_occ], dim=1)
            elif self.fusion_type in ('transformer_query_scene','transformer_query_target', 'transformer_concat'):
                ## check model data parallel
                features_fused = self.encoder_in(inputs[0], inputs[1], inputs[2])
        if self.model_type == "afford_scene_pc":
            features_fused = self.encoder_in(inputs)

        if self.model_type == "afford_scene_targ_pc":
            # if self.fusion_type == "MLP_fusion":
            if self.fusion_type in ("MLP_fusion", "CNN_concat", "CNN_add"):
                if not self.shared_weights:
                    feat_3d_targ = self.encoder_in_targ(inputs[0].unsqueeze(1))
                    feat_3d_scene = self.encoder_in_scene(inputs[1].unsqueeze(1))
                else:
                    feat_3d_scene = self.encoder_in(inputs[0].unsqueeze(1))
                    feat_3d_targ = self.encoder_in(inputs[1].unsqueeze(1))
                if self.fusion_type == "CNN_add":
                    features_fused = feat_3d_targ + feat_3d_scene
                else:
                    features_fused = torch.cat([feat_3d_scene, feat_3d_targ], dim=1)
            # elif self.fusion_type == "transformer_fusion":
            elif self.fusion_type in ("transformer_query_scene","transformer_query_target", "transformer_concat"):
                features_fused =  self.encoder_in(inputs[0], inputs[1])

                # save_point_cloud_as_ply(inputs[0][0].cpu().numpy(), "58877.ply")
                # save_point_cloud_as_ply(inputs[1][0].cpu().numpy(), "58887.ply")
                # indices = np.where(inputs[3][0].cpu().numpy() > 0.5)
                # points = np.vstack(indices).T
                # save_point_cloud_as_ply(points, "58877.ply")

                # visualize_and_save_tsd
                
                # visualize_and_save_tsdf(inputs[1][0,:],'/usr/stud/dira/GraspInClutter/grasping/visual/demo/111.png')
                # visualize_and_save_tsdf(inputs[0][0,:],'/usr/stud/dira/GraspInClutter/grasping/visual/demo/222.png')
                
                # features_fused = self.encoder_in(inputs[0], inputs[1], inputs[2])
            
        
        if self.model_type == "giga_aff_plus_occluder_grid":
            assert self.fusion_type in ("MLP_fusion", "CNN_concat", "CNN_add")
            if not self.shared_weights:
                feat_3d_occ = self.encoder_in_occ(inputs[0].unsqueeze(1))
                feat_3d_scene = self.encoder_in_scene(inputs[1].unsqueeze(1))
            else:
                feat_3d_occ = self.encoder_in(inputs[0].unsqueeze(1))
                feat_3d_scene = self.encoder_in(inputs[1].unsqueeze(1))
            features_fused = torch.cat([feat_3d_occ, feat_3d_scene], dim=1)
        
        c = self.encoder_aff(features_fused)
        if self.add_single_supervision:
            qual, rot, width, qual_single, rot_single, width_single = self.decode_with_additional_single(p, c)
            return qual, rot, width, qual_single, rot_single, width_single
        else:
            qual, rot, width = self.decode(p, c)   
            return qual, rot, width

            
    def infer_geo(self, inputs, p_tsdf, **kwargs):
        # c = self.encode_inputs(inputs)
        c = self.encode_tsdf(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf


    def encode_inputs_set_theory(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)

    def decode_feature(self, p, feature):
        qual = self.decoder_qual.compute_out(p, feature)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot.compute_out(p, feature)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width.compute_out(p, feature)
        return qual, rot, width

    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot(p, c, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width(p, c, **kwargs)
        return qual, rot, width

    def decode_with_additional_single(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot(p, c, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width(p, c, **kwargs)

        qual_single = self.decoder_qual_single(p, c, **kwargs)
        qual_single = torch.sigmoid(qual_single)
        rot_single = self.decoder_rot_single(p, c, **kwargs)
        rot_single = nn.functional.normalize(rot_single, dim=2)
        width_single = self.decoder_width_single(p, c, **kwargs)

        return qual, rot, width, qual_single, rot_single, width_single

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            # print(qual_out)
            loss = - qual_out.sum()
            loss.backward()
            optimizer.step()
            # print(qual_out.mean().item())
        with torch.no_grad():
            #print(pos, pos_tmp)
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, width_out = self.forward(x, pos_tmp)
            # print(pos, pos_tmp, qual_out)
            # print(qual_out.mean().item())
        # import pdb; pdb.set_trace()
        # self.train()
        for p in self.parameters():
            p.requres_grad = True
        # import pdb; pdb.set_trace()
        return qual_out, pos_tmp, rot_out, width_out
    

class ConvolutionalOccupancyNetwork_Sequential(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoders, encoders=None, device=None, detach_tsdf=False):
        super().__init__()
        
        self.decoder_qual = decoders[0].to(device)
        self.decoder_rot = decoders[1].to(device)
        self.decoder_width = decoders[2].to(device)
        if len(decoders) == 4:
            self.decoder_tsdf = decoders[3].to(device)
        self.encoder_tsdf = encoders[0].to(device)
        self.encoder_aff = encoders[1].to(device)
        self._device = device

        self.detach_tsdf = detach_tsdf

    def forward(self, inputs, p, p_tsdf=None, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)

        c0 = self.encoder_tsdf(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c0, **kwargs)
        visualize_and_save_tsdf(tsdf[0].reshape(40,40,40).detach().cpu().numpy(), '/usr/stud/dira/GraspInClutter/grasping/visual/002tsdf_grid.jpg')
        inputs_aff = inputs

        c1 = self.encoder_aff(inputs_aff)
        qual, rot, width = self.decode(p, c1)   
        return qual, rot, width, tsdf

            
    def infer_geo(self, inputs, p_tsdf, **kwargs):
        # c = self.encode_inputs(inputs)
        c = self.encode_tsdf(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf


    def encode_inputs_set_theory(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)

    def decode_feature(self, p, feature):
        qual = self.decoder_qual.compute_out(p, feature)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot.compute_out(p, feature)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width.compute_out(p, feature)
        return qual, rot, width

    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot(p, c, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width(p, c, **kwargs)
        return qual, rot, width

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            # print(qual_out)
            loss = - qual_out.sum()
            loss.backward()
            optimizer.step()
            # print(qual_out.mean().item())
        with torch.no_grad():
            #print(pos, pos_tmp)
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, width_out = self.forward(x, pos_tmp)
            # print(pos, pos_tmp, qual_out)
            # print(qual_out.mean().item())
        # import pdb; pdb.set_trace()
        # self.train()
        for p in self.parameters():
            p.requres_grad = True
        # import pdb; pdb.set_trace()
        return qual_out, pos_tmp, rot_out, width_out
    

class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoders, encoder=None, device=None, detach_tsdf=False, add_single_supervision = False):
        super().__init__()
        
        self.decoder_qual = decoders[0].to(device)
        self.decoder_rot = decoders[1].to(device)
        self.decoder_width = decoders[2].to(device)
        if len(decoders) == 4:
            self.decoder_tsdf = decoders[3].to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

        self.detach_tsdf = detach_tsdf

    def forward(self, inputs, p, p_tsdf=None, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        
        # feature = self.query_feature(p, c)
        # qual, rot, width = self.decode_feature(p, feature)
        qual, rot, width = self.decode(p, c)
        if p_tsdf is not None:
            if self.detach_tsdf:
                for k, v in c.items():
                    c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width
            
    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf


    def encode_inputs_set_theory(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)

    def decode_feature(self, p, feature):
        qual = self.decoder_qual.compute_out(p, feature)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot.compute_out(p, feature)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width.compute_out(p, feature)
        return qual, rot, width

    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot(p, c, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width(p, c, **kwargs)
        return qual, rot, width

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            # print(qual_out)
            loss = - qual_out.sum()
            loss.backward()
            optimizer.step()
            # print(qual_out.mean().item())
        with torch.no_grad():
            #print(pos, pos_tmp)
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, width_out = self.forward(x, pos_tmp)
            # print(pos, pos_tmp, qual_out)
            # print(qual_out.mean().item())
        # import pdb; pdb.set_trace()
        # self.train()
        for p in self.parameters():
            p.requres_grad = True
        # import pdb; pdb.set_trace()
        return qual_out, pos_tmp, rot_out, width_out

class ConvolutionalOccupancyNetworkGeometry(nn.Module):
    def __init__(self, decoder, encoder=None, device=None, add_single_supervision = False):
        super().__init__()
        
        self.decoder_tsdf = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, inputs, p, p_tsdf, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf
    
    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c
        
    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r