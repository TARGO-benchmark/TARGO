from vgn.ConvONets.encoder import (
    pointnet, voxels, pointnetpp
)


encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    'voxel_simple_local': voxels.LocalVoxelEncoder,
    'voxel_simple_local_with_input_fushion': voxels.LocalVoxelEncoder_With_InputFushion,
    'voxel_simple_local_without_3d': voxels.LocalVoxelEncoder_without_3d,
}
