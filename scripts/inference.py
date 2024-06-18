import argparse
import numpy as np
import json
from pathlib import Path
import os
import glob
from vgn.detection import VGN
from datetime import datetime
from vgn.detection_implicit import VGNImplicit
from vgn.experiments import target_sample, clutter_removal, target_sample_offline
from vgn.utils.misc import set_random_seed

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def create_and_write_args_to_result_path(args):
    # Determine the model name based on the condition
    if args.type == "afford_scene_targ_pc":
        model_name = f'{args.type}/{args.fusion_type}/sc_{args.shape_completion}/gt_shape_{args.complete_shape}_num_layer_{args.num_encoder_layers}'
    elif args.type in ("giga", "giga_aff", "vgn", "giga_hr"):
        model_name = f'{args.type}/gt_shape_{args.complete_shape}/data_aug_{args.data_aug}'
    elif args.type == "afford_scene_pc":
        model_name = f'{args.type}/sc_{args.shape_completion}/gt_shape_{args.complete_shape}'
    else:
        print("Unsupported type.")
        return  # Early exit if the type is not supported

    # Construct the path to the result directory
    result_directory = f'{args.result_root}/{model_name}'

    # Ensure the directory exists
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # Generate a timestamp for the result file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_filename = f"{timestamp}"

    # Path for the result file
    result_file_path = os.path.join(result_directory, result_filename)

    args.result_path = result_file_path 

    # Serialize args to a string
    args_dict = vars(args)
    args_content = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())

    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    if args.vis == True:
        args.logdir = Path(result_file_path)

    result_initial_path = f'{result_file_path}/initial.txt'
    # Write the args content to the file
    with open(result_initial_path, 'w') as result_file:
        result_file.write(args_content)

    print(f"Args saved to {result_initial_path}")


def find_and_assign_first_checkpoint(args):
    # Determine the model name based on the condition
    if args.type == "afford_scene_targ_pc":
        model_name = f'{args.type}/{args.fusion_type}/sc_{args.shape_completion}_gt_shape_{args.complete_shape}/num_layer_{args.num_encoder_layers}'
    elif args.type in ("giga", "giga_aff", "vgn", "giga_hr"):
        model_name = f'{args.type}/gt_shape_{args.complete_shape}/data_aug_{args.data_aug}'
    elif args.type == "afford_scene_pc":  # Fixed typo here from 'tyep' to 'type'
        model_name = f'{args.type}/sc_{args.shape_completion}_gt_shape_{args.complete_shape}'
    else:
        print("Unsupported type.")
        return  # Early exit if the type is not supported

    print(model_name)

    # Construct the path to the model directory
    model_directory = os.path.join(args.model_root, model_name)

    # Find all checkpoint files in the model directory
    checkpoint_pattern = os.path.join(model_directory, '*.pt')
    checkpoint_files = glob.glob(checkpoint_pattern)

    # Assign the first checkpoint file to args.model if available
    if len(checkpoint_files) > 0:
        args.model = checkpoint_files[0]
        print("Assigned model:", args.model)
    else:
        print("No checkpoint files found.")

def main(args):
    if args.type == 'vgn':
        grasp_planner = VGN(args.model,
                            args.type,
                            best=args.best,
                            qual_th=args.qual_th,
                            force_detection=args.force,
                            # out_th=0.1,
                            out_th= 0.1,
                            visualize=args.vis,)
    ## TOO [] output threshold
    elif args.type in ['giga', 'giga_aff', 'giga_hr', 'afford_scene_pc','giga_aff_plus_occluder_input',\
                        'giga_aff_plus_target_occluder_grid', 'afford_scene_targ_pc']:
        grasp_planner = VGNImplicit(args.model,
                                args.type,
                                best=args.best,
                                qual_th=args.qual_th,
                                force_detection=args.force,
                                # out_th=0.1,
                                out_th=args.out_thre,
                                select_top=False,
                                visualize=args.vis,
                                shared_weights = args.shared_weights,
                                add_single_supervision = args.add_single_supervision,
                                fusion_type=args.fusion_type,
                                feat_type= args.feat_type,
                                shape_completion=args.shape_completion,
                                num_encoder_layers=args.num_encoder_layers,
                                )
    else:
        raise NotImplementedError(f'model type {args.type} not implemented!')

    gsr = []
    dr = []
    if args.task_eval == 'target_driven_offline':
        occ_level_sr = target_sample_offline.run(
            grasp_plan_fn=grasp_planner,
            logdir=args.logdir,
            description=args.description,
            scene=args.scene,            
            object_set=args.object_set,
            num_objects=args.num_objects,
            n=args.num_view,
            num_rounds=args.num_rounds,
            # seed=seed,
            sim_gui=args.sim_gui,
            result_path= args.result_path,
            add_noise=args.add_noise,
            sideview=args.sideview,
            silence=args.silence,
            visualize=args.vis,
            task_eval= args.task_eval ,
            complete_shape=args.complete_shape,
            type = args.type,
            fusion_type = args.fusion_type,
            test_root = args.test_root,
            input_points=args.input_points,
            shape_completion=args.shape_completion,
            )
        ## write to file
        with open('occ_level_sr.json', 'w') as f:
            json.dump(occ_level_sr, f)
    elif args.task_eval in ('clutter_removal','target_driven_online'):
        for seed in args.seeds:
            set_random_seed(seed)
        # if existing_tst

        if args.task_eval == 'target_driven_online':
            success_rate, declutter_rate = target_sample.run(
                grasp_plan_fn=grasp_planner,
                logdir=args.logdir,
                description=args.description,
                scene=args.scene,
                object_set=args.object_set,
                num_objects=args.num_objects,
                n=args.num_view,
                num_rounds=args.num_rounds,
                seed=seed,
                sim_gui=args.sim_gui,
                result_path=None,
                add_noise=args.add_noise,
                sideview=args.sideview,
                silence=args.silence,
                visualize=args.vis,
                task_eval= args.task_eval ,
                complete_shape=args.complete_shape,
                type = args.type,
                fusion_type = args.fusion_type,
                # test_root = args.test_root,
                )
        elif args.task_eval == 'clutter_removal':
            success_rate, declutter_rate = clutter_removal.run(
                grasp_plan_fn=grasp_planner,
                logdir=args.logdir,
                description=args.description,
                scene=args.scene,
                object_set=args.object_set,
                num_objects=args.num_objects,
                n=args.num_view,
                num_rounds=args.num_rounds,
                seed=seed,
                sim_gui=args.sim_gui,
                result_path=None,
                add_noise=args.add_noise,
                sideview=args.sideview,
                silence=args.silence,
                visualize=args.vis,
                task_eval  = args.task_eval ,
                complete_shape=args.complete_shape,
                type = args.type,
                )
        gsr.append(success_rate)
        dr.append(declutter_rate)
        results = {
            'gsr': {
                'mean': np.mean(gsr),
                'std': np.std(gsr),
                'val': gsr
            },
            'dr': {
                'mean': np.mean(dr),
                'std': np.std(dr),
                'val': dr
            }
        }
        print('Average results:')
        print(f'Grasp sucess rate: {np.mean(gsr):.2f} ± {np.std(gsr):.2f} %')
        print(f'Declutter rate: {np.mean(dr):.2f} ± {np.std(dr):.2f} %')
        with open(args.result_path, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="afford_scene_targ_pc", help = "giga_hr | giga_aff | giga | vgn | afford_scene_targ_pc| afford_scene_pc")
    parser.add_argument("--fusion_type", default='transformer_concat', type=str, 
                    choices=['transformer_query_target', 'transformer_query_scene', 'CNN_concat', 'CNN_add', 'MLP_fusion', 'transformer_concat'])
    parser.add_argument("--feat_type", default='Plane_feat', type=str, help = "plane | grid")
    parser.add_argument("--model_root", type=Path, default='/usr/stud/dira/GraspInClutter/grasping/checkpoints_noisy')
    parser.add_argument("--result_root", type=Path, default='/usr/stud/dira/GraspInClutter/grasping/eval_results_gaussian')
    parser.add_argument("--model", type=Path, default='/usr/stud/dira/GraspInClutter/grasping/train_logs_gaussian_noise/args.nets=afford_scene_targ_pc_shared_weights=True_add_single_supervision=False_fusion_type=transformer_concat/shape_completion_True/encoder_layer_2/2024-06-10_11-07-29/24-06-10-11-07_dataset=combined,augment=False,net=afford_scene_targ_pc,batch_size=32,lr=2e-04/best_vgn_afford_scene_targ_pc_val_acc=0.9264.pt')
    parser.add_argument("--result-path", default = '', type=str)
    parser.add_argument("--data-aug", default = True, type=str2bool)
    parser.add_argument("--data_contain", type=str, default="pc",\
         help = "pc | pc and targ_grid | pc and occ_grid | pc and targ_grid and occ_grid  ")
    parser.add_argument("--decouple", type=str2bool, default=False, help = "")
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/test")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-view", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs='+', default=[0])
    parser.add_argument("--sim-gui", type=str2bool, default=False)
    parser.add_argument("--task_eval", type=str, default='target_driven_offline', help= "clutter_removal | target_driven_online | target_driven_offline")
    parser.add_argument("--shared_weights", default=True, type=str2bool, help = "")
    parser.add_argument("--add_single_supervision", default=False, type=str2bool, help = "")
    parser.add_argument("--test_root", type=str, default='/storage/user/dira/nips_data_version6/combined/test_set_gaussian_0.002/', help= "clutter_removal | target_driven_online | target_driven_offline")
    parser.add_argument("--qual-th", type=float, default=0.9)
    parser.add_argument("--eval-geo", action="store_true", help='whether evaluate geometry prediction')
    parser.add_argument("--use_complete_targ", type=str2bool, default=False, help="If true, use the target grasp mode, else use the clutter removal mode")
    parser.add_argument("--best", action="store_true", default = True, help="Whether to use best valid grasp (or random valid grasp)")
    parser.add_argument("--force", action="store_true",default = True, help="When all grasps are under threshold, force the detector to select the best grasp")
    parser.add_argument("--add-noise", type=str, default="dex", help="Whether add noise to depth observation, trans | dex | norm | ''")
    parser.add_argument("--sideview", action="store_true", default = True,help="Whether to look from one side")
    parser.add_argument("--silence", action="store_true", help="Whether to disable tqdm bar")
    parser.add_argument("--vis", action="store_true", default=False, help="visualize and save affordance")
    parser.add_argument("--input_points", type=str, default='tsdf_points', help="depth_bp | tsdf_points")
    parser.add_argument("--complete_shape", type=str2bool, default=False, help="use the complete the TSDF for grasp planning")
    parser.add_argument("--shape_completion", type=str2bool, default=True, help="use the complete the TSDF for grasp planning")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help= 1 | 2 | 3 |4)
    parser.add_argument("--out_thre", type=float, default=0.15)

    args = parser.parse_args()

    ## automatically find the checkpoint, according to the model type
    if str(args.model) == ".":
        find_and_assign_first_checkpoint(args)
    if str(args.result_path) == "":
        create_and_write_args_to_result_path(args)


    if args.type in ('giga_aff', 'giga', 'vgn', 'giga_hr'):
        args.shape_completion = False

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    main(args)