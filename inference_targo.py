#!/usr/bin/env python3
"""Run TARGO inference on the VGN evaluation set."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "shape_completion" / "chamfer_dist"))
sys.path.insert(0, str(ROOT / "src" / "shape_completion" / "pointnet2_ops"))

for build_dir in (
    ROOT / "src" / "shape_completion" / "chamfer_dist" / "build",
    ROOT / "src" / "shape_completion" / "pointnet2_ops" / "build",
):
    if build_dir.exists():
        for lib_dir in build_dir.iterdir():
            if lib_dir.is_dir():
                sys.path.insert(0, str(lib_dir))

def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "y", "1"}:
        return True
    if lowered in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def default_paths():
    root = "data/nips_data_version6/test_set_gaussian_0.002"
    return {
        "test_root": root,
        "result_root": "targo_eval_results/vgn/eval_results/targo",
        "target_file": "example_targets/target_list.txt",
    }


def make_result_path(result_root, model_type):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_path = Path(result_root) / model_type / timestamp
    result_path.mkdir(parents=True, exist_ok=True)
    return result_path


def write_args(result_path, args):
    with (result_path / "initial.txt").open("w", encoding="utf-8") as handle:
        for key, value in vars(args).items():
            handle.write(f"{key}: {value}\n")


def run_inference(args):
    defaults = default_paths()
    args.test_root = args.test_root or defaults["test_root"]
    args.occ_level_dict = args.occ_level_dict or str(Path(args.test_root) / "occ_level_dict.json")
    args.result_root = args.result_root or defaults["result_root"]
    args.target_file = args.target_file or defaults["target_file"]
    args.logdir = args.logdir or args.result_root

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    test_root = Path(args.test_root)
    required_dataset_paths = [
        test_root,
        test_root / "scenes",
        test_root / "mesh_pose_dict",
        Path(args.occ_level_dict),
    ]
    missing_dataset_paths = [path for path in required_dataset_paths if not path.exists()]
    if missing_dataset_paths:
        missing = "\n".join(f"  - {path}" for path in missing_dataset_paths)
        raise FileNotFoundError(
            "VGN inference dataset is missing. The clean code is runnable, but it needs "
            "the processed VGN test scenes.\n"
            f"Missing paths:\n{missing}\n"
            "Pass a valid VGN dataset with --test_root, or extract/copy the processed "
            "dataset so this default exists: data/nips_data_version6/test_set_gaussian_0.002\n"
            "Do not pass a literal placeholder path like /path/to/test_set_gaussian_0.002.\n"
            "Dataset references:\n"
            "  - TARGO: https://targo-benchmark.github.io/\n"
            "  - GIGA: https://github.com/UT-Austin-RPL/GIGA"
        )

    if args.shape_completion and not Path(args.sc_model_path).exists():
        raise FileNotFoundError(f"Shape completion checkpoint not found: {args.sc_model_path}")

    from src.vgn.detection_implicit_vgn import VGNImplicit
    from src.vgn.experiments import target_sample_offline_vgn

    result_path = make_result_path(args.result_root, args.model_type)
    args.result_path = str(result_path)
    write_args(result_path, args)

    planner = VGNImplicit(
        args.model,
        args.model_type,
        best=args.best,
        qual_th=args.qual_th,
        force_detection=args.force,
        out_th=args.out_th,
        select_top=False,
        visualize=args.vis,
        sc_model_path=args.sc_model_path if args.shape_completion else None,
        cd_iou_measure=True,
    )

    success_rate = target_sample_offline_vgn.run(
        grasp_plan_fn=planner,
        logdir=Path(args.logdir),
        description=args.description,
        scene=args.scene,
        object_set=args.object_set,
        sim_gui=args.sim_gui,
        result_path=str(result_path),
        add_noise=None,
        sideview=args.sideview,
        visualize=args.vis,
        type=args.model_type,
        test_root=args.test_root,
        occ_level_dict_path=args.occ_level_dict,
        model_type=args.model_type,
        video_recording=args.video_recording,
        target_file_path=args.target_file,
        max_scenes=args.max_scenes,
        sc_net=getattr(planner, "sc_net", None),
        vis_failure_only=args.vis_failure_only,
        vis_rgb=args.vis_rgb,
        vis_gt_target=args.vis_gt_target,
        vis_grasps=args.vis_grasps,
        vis_target_mesh_pred=args.vis_target_mesh_pred,
    )

    with (result_path / "occ_level_sr.json").open("w", encoding="utf-8") as handle:
        json.dump({"success_rate": success_rate}, handle, indent=2)

    print("=" * 60)
    print("TARGO inference completed")
    print(f"Dataset: {args.dataset}")
    print(f"Model type: {args.model_type}")
    print(f"Result path: {result_path}")
    print(f"Success rate: {success_rate:.4f}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["vgn"], default="vgn")
    parser.add_argument("--model", type=Path, default=Path("checkpoints/targonet.pt"))
    parser.add_argument("--model_type", choices=["targo"], default="targo")
    parser.add_argument("--shape_completion", type=str2bool, default=True)
    parser.add_argument("--sc_model_path", type=Path, default=Path("checkpoints/adapointr.pth"))
    parser.add_argument("--test_root", default="")
    parser.add_argument("--occ_level_dict", default="")
    parser.add_argument("--result_root", default="")
    parser.add_argument("--logdir", default="")
    parser.add_argument("--description", default="targo_inference")
    parser.add_argument("--scene", choices=["pile", "packed"], default="packed")
    parser.add_argument("--object-set", dest="object_set", default="packed/train")
    parser.add_argument("--target-file", dest="target_file", default="")
    parser.add_argument("--out_th", type=float, default=0.2)
    parser.add_argument("--qual-th", dest="qual_th", type=float, default=0.9)
    parser.add_argument("--best", type=str2bool, default=True)
    parser.add_argument("--force", type=str2bool, default=True)
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=0,
        help="Maximum number of scenes to evaluate. Use 0, the default, to traverse all scenes.",
    )
    parser.add_argument("--sim-gui", dest="sim_gui", type=str2bool, default=False)
    parser.add_argument("--sideview", type=str2bool, default=True)
    parser.add_argument("--vis", type=str2bool, default=False)
    parser.add_argument("--video-recording", dest="video_recording", type=str2bool, default=False)
    parser.add_argument("--vis_failure_only", type=str2bool, default=False)
    parser.add_argument("--vis_rgb", type=str2bool, default=False)
    parser.add_argument("--vis_gt_target", type=str2bool, default=True)
    parser.add_argument("--vis_grasps", type=str2bool, default=True)
    parser.add_argument("--vis_target_mesh_pred", type=str2bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(ROOT)
    try:
        run_inference(parse_args())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
