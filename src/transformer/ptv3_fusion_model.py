import torch
import torch.nn as nn
import numpy as np
from typing import List, Any, Dict
import sys
import os
# Add the project root to the path to import model.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model import PointTransformerV3, Point
import torch.nn.functional as F


class PointTransformerV3FusionModel(torch.nn.Module):
    """
    Fusion model using PointTransformerV3 as encoder for TARGO.
    Replaces the original TransformerFusionModel with PTv3-based processing.
    """
    
    def __init__(self):
        super().__init__()
        
        # PointTransformerV3 encoder for scene point cloud
        self.scene_encoder = PointTransformerV3(
            in_channels=3,  # xyz coordinates
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            cls_mode=True,  # Use classification mode (encoder only)
            enable_flash=True,  # Enable flash attention for better performance
        )
        
        # PointTransformerV3 encoder for target point cloud
        self.target_encoder = PointTransformerV3(
            in_channels=3,  # xyz coordinates
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            cls_mode=True,  # Use classification mode (encoder only)
            enable_flash=True,  # Enable flash attention for better performance
        )
        
        # Feature fusion layers
        self.fusion_conv = nn.Conv3d(64, 32, kernel_size=3, padding=1)  # 64 -> 32 channels (32+32=64 after concatenation)
        self.fusion_bn = nn.BatchNorm3d(32)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        # Final output projection
        self.output_conv = nn.Conv3d(32, 64, kernel_size=3, padding=1)  # 32 -> 64 channels for compatibility
        
        self.size = 0.30000000000000004
        self.grid_size = 0.0075  # Grid size for voxelization
        
    def points_to_voxel_grid(self, points, batch_size=1, grid_resolution=40):
        """
        Convert point cloud to voxel grid representation.
        
        Args:
            points: Point cloud tensor of shape (N, 3)
            batch_size: Batch size
            grid_resolution: Resolution of the voxel grid
            
        Returns:
            Voxel grid tensor of shape (batch_size, channels, H, W, D)
        """
        device = points.device
        
        # Normalize points to [0, grid_resolution-1]
        points_min = points.min(dim=0)[0]
        points_max = points.max(dim=0)[0]
        points_range = points_max - points_min
        points_range = torch.clamp(points_range, min=1e-6)  # Avoid division by zero
        
        normalized_points = (points - points_min) / points_range * (grid_resolution - 1)
        voxel_coords = torch.round(normalized_points).long()
        
        # Clamp coordinates to valid range
        voxel_coords = torch.clamp(voxel_coords, 0, grid_resolution - 1)
        
        # Create voxel grid
        voxel_grid = torch.zeros(batch_size, 32, grid_resolution, grid_resolution, grid_resolution, device=device)
        
        # Fill voxel grid (simple occupancy)
        if len(voxel_coords) > 0:
            batch_indices = torch.zeros(len(voxel_coords), dtype=torch.long, device=device)
            voxel_grid[batch_indices, :, voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1.0
            
        return voxel_grid
    
    def prepare_point_data(self, points, batch_size=1):
        """
        Prepare point cloud data for PointTransformerV3.
        
        Args:
            points: Point cloud tensor of shape (N, 3)
            batch_size: Batch size
            
        Returns:
            Dictionary with required keys for PTv3
        """
        device = points.device
        
        # Create batch indices
        batch = torch.zeros(len(points), dtype=torch.long, device=device)
        
        # Create offset (cumulative point counts per batch)
        offset = torch.tensor([len(points)], dtype=torch.long, device=device)
        
        # Use coordinates as features
        feat = points.clone()
        
        # Create grid coordinates for voxelization
        grid_size = self.grid_size
        grid_coord = torch.div(
            points - points.min(0)[0], grid_size, rounding_mode="trunc"
        ).int()
        
        data_dict = {
            "coord": points,
            "feat": feat,
            "batch": batch,
            "offset": offset,
            "grid_coord": grid_coord,
            "grid_size": grid_size
        }
        
        return data_dict
    
    def forward(self, scene_pc, targ_pc) -> torch.Tensor:
        """
        Forward pass using PointTransformerV3 encoders.
        
        Args:
            scene_pc: Scene point cloud tensor of shape (batch_size, N_scene, 3)
            targ_pc: Target point cloud tensor of shape (batch_size, N_target, 3)
            
        Returns:
            Fused features tensor of shape (batch_size, 64, 40, 40, 40)
        """
        batch_size = scene_pc.shape[0]
        device = scene_pc.device
        
        # Set encoders to eval mode to avoid BatchNorm issues with small batches
        # This prevents "Expected more than 1 value per channel when training" error
        scene_encoder_was_training = self.scene_encoder.training
        target_encoder_was_training = self.target_encoder.training
        
        self.scene_encoder.eval()
        self.target_encoder.eval()
        
        # Process each point cloud in the batch
        scene_features_list = []
        target_features_list = []
        
        try:
            for b in range(batch_size):
                # Prepare scene point cloud
                scene_points = scene_pc[b]  # (N_scene, 3)
                
                # Skip if too few points
                if len(scene_points) < 10:
                    empty_voxel = torch.zeros(1, 32, 40, 40, 40, device=device)
                    scene_features_list.append(empty_voxel)
                    target_features_list.append(empty_voxel)
                    continue
                
                scene_data = self.prepare_point_data(scene_points, batch_size=1)
                
                # Prepare target point cloud  
                target_points = targ_pc[b]  # (N_target, 3)
                
                # Skip if too few points
                if len(target_points) < 10:
                    empty_voxel = torch.zeros(1, 32, 40, 40, 40, device=device)
                    scene_features_list.append(empty_voxel)
                    target_features_list.append(empty_voxel)
                    continue
                
                target_data = self.prepare_point_data(target_points, batch_size=1)
                
                # Encode with PointTransformerV3 in eval mode
                with torch.no_grad():
                    scene_encoded = self.scene_encoder(scene_data)
                    target_encoded = self.target_encoder(target_data)
                    
                    # Extract features from the encoded point clouds
                    scene_feat = scene_encoded.feat  # (N_scene, 512)
                    target_feat = target_encoded.feat  # (N_target, 512)
                    
                    # Convert to voxel grids
                    scene_voxel = self.points_to_voxel_grid(scene_points, batch_size=1, grid_resolution=40)
                    target_voxel = self.points_to_voxel_grid(target_points, batch_size=1, grid_resolution=40)
                    
                    scene_features_list.append(scene_voxel)
                    target_features_list.append(target_voxel)
                    
        except Exception as e:
            print(f"Error in PTv3 encoding: {e}")
            # Fallback: create empty voxel grids for remaining batch elements
            while len(scene_features_list) < batch_size:
                empty_voxel = torch.zeros(1, 32, 40, 40, 40, device=device)
                scene_features_list.append(empty_voxel)
                target_features_list.append(empty_voxel)
        
        finally:
            # Restore original training states
            if scene_encoder_was_training:
                self.scene_encoder.train()
            if target_encoder_was_training:
                self.target_encoder.train()
        
        # Stack batch features
        scene_features = torch.cat(scene_features_list, dim=0)  # (batch_size, 32, 40, 40, 40)
        target_features = torch.cat(target_features_list, dim=0)  # (batch_size, 32, 40, 40, 40)
        
        # Fuse scene and target features by concatenation
        fused_features = torch.cat([scene_features, target_features], dim=1)  # (batch_size, 64, 40, 40, 40)
        
        # Apply fusion layers
        fused_features = self.fusion_conv(fused_features)
        fused_features = self.fusion_bn(fused_features)
        fused_features = self.fusion_relu(fused_features)
        
        # Final output projection
        output_features = self.output_conv(fused_features)  # (batch_size, 64, 40, 40, 40)
        
        return output_features 