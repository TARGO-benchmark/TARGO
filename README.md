# TARGO: Benchmarking Target-driven Object Grasping under Occlusions

[Yan Xia](https://yan-xia.github.io)\*, [Ran Ding](https://randing2000.github.io)\*, [Ziyuan Qin](https://scholar.google.com/citations?user=pcCR2lUAAAAJ)\*, [Guanqi Zhan](https://www.robots.ox.ac.uk/~guanqi), [Kaichen Zhou](https://www.cs.ox.ac.uk/people/kaichen.zhou), [Long Yang](https://scholar.google.com/citations?user=KOTg0mQAAAAJ&hl), [Hao Dong](https://zsdonghao.github.io), [Daniel Cremers](https://cvg.cit.tum.de/members/cremers)

*International Journal of Computer Vision (IJCV), 2026*

[**Project page**](https://targo-benchmark.github.io/) · [**Paper**](https://doi.org/10.1007/s11263-025-02716-9) · [**arXiv**](https://arxiv.org/abs/2407.06168) · [**Dataset**](https://huggingface.co/datasets/randing2000/TARGO) · [**Checkpoints**](https://huggingface.co/randing2000/TARGO-Net)

TARGO is a benchmark for **TAR**get-driven **G**rasping under **O**cclusions. It measures how 6-DoF grasp
prediction degrades as the target object becomes more occluded in cluttered scenes, and provides a large
synthetic training set, synthetic and real-world evaluation scenes, and **TARGO-Net**, a transformer-based
grasping model that completes the shape of the target before predicting grasps.

This repository contains the training and evaluation code of TARGO-Net:

| Script | Purpose |
| --- | --- |
| `train_targo.py` / `run_train.sh` | Train TARGO-Net on the TARGO synthetic training set. |
| `inference_targo.py` / `run_inference.sh` | Evaluate a checkpoint on the synthetic test set in PyBullet and report grasp success rates per occlusion level. |

## Repository layout

```text
inference_targo.py, run_inference.sh   evaluation entry point and launcher
train_targo.py, run_train.sh           training entry point and launcher
src/vgn/                               simulation, data loading and grasp detection (based on VGN / GIGA)
src/vgn/ConvONets/                     convolutional occupancy network components (based on ConvONets)
src/transformer/                       TARGO-Net fusion transformer
src/shape_completion/                  AdaPoinTr shape completion (based on PoinTr) with the chamfer and pointnet2 CUDA ops
scripts/convonet_setup.py              builds the Cython mesh-intersection extension
setup/                                 small assets used by the simulation and the data pipeline
generated_min_test_set/                a one-scene test set for a quick check of the installation
example_targets/                       example list of target names
```

## Installation

Tested with Python 3.10 and PyTorch 1.13.0 + CUDA 11.7 on Linux. A CUDA-capable GPU is required; the
CUDA extensions are built from source.

### 1. Python environment

```bash
git clone https://github.com/TARGO-benchmark/TARGO.git
cd TARGO

conda create -n targo python=3.10 -y
conda activate targo

# PyTorch build matching your CUDA driver (example: CUDA 11.7)
pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
pip install -e .
```

### 2. Dependencies with CUDA-specific builds

```bash
# torch-scatter: use the wheel index matching your torch / CUDA version
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# MinkowskiEngine 0.5.4 (sparse convolutions used by the fusion transformer)
conda install -c anaconda openblas-devel -y
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps \
  --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" \
  --config-settings="--blas=openblas"
```

See the [MinkowskiEngine README](https://github.com/NVIDIA/MinkowskiEngine) if its build fails on your system.

### 3. Native extensions (built in place)

```bash
export CUDA_HOME=/usr/local/cuda        # must match the CUDA version of your PyTorch build
# export TORCH_CUDA_ARCH_LIST="8.6"     # optional: the compute capability of your GPU

python scripts/convonet_setup.py build_ext --inplace
(cd src/shape_completion/chamfer_dist && python setup.py build_ext --inplace)
(cd src/shape_completion/pointnet2_ops && python setup.py build_ext --inplace)
```

The entry points add these directories to `sys.path`, so nothing else needs to be installed.

## Download the assets

All assets live below the repository root; `checkpoints/` and `data/` are ignored by git.

### Checkpoints

```bash
huggingface-cli download randing2000/TARGO-Net --local-dir . \
  --include "checkpoints/targonet.pt" "checkpoints/adapointr.pth"
```

`checkpoints/targonet.pt` is the TARGO-Net grasp model and `checkpoints/adapointr.pth` the AdaPoinTr
shape-completion model it uses.

### Object models (URDFs and meshes)

The PyBullet simulation loads the VGN / GIGA object set from `data/urdfs`
(`data/urdfs/packed/{train,test}`, `data/urdfs/setup` and `data/urdfs/panda`). Download
[GIGA's data archive](https://utexas.box.com/s/h3ferwjhuzy6ja8bzcm3nu9xq1wkn94s) (see the
[GIGA README](https://github.com/UT-Austin-RPL/GIGA)) and copy its `data/urdfs` folder to `data/urdfs`.

### Synthetic test set

```bash
huggingface-cli download randing2000/TARGO --repo-type dataset --local-dir data \
  --include "test_set_gaussian_0.005/**"
```

Layout: `data/test_set_gaussian_0.005/{scenes/, mesh_pose_dict/, occ_level_dict.json, setup.json, ...}`.

### Synthetic training set (large)

```bash
huggingface-cli download randing2000/TARGO --repo-type dataset --local-dir data \
  --include "syn_train/**"
```

Layout: `data/syn_train/{grasps.csv, scenes/, mesh_pose_dict/}`. On Hugging Face the `scenes/` and
`mesh_pose_dict/` folders are split into numbered sub-folders; the data loader handles both the sharded
and the flat layout. If `setup.json` is missing from the training root, the default TARGO / VGN setup
in `setup/setup.json` is copied there automatically.

## Quick check

Evaluate the released checkpoints on the bundled one-scene test set (needs the checkpoints and `data/urdfs`):

```bash
./run_inference.sh --test_root generated_min_test_set --result_root eval_results/smoke
```

The run should end with `Success rate: 1.0000`; the result files are written to
`eval_results/smoke/targo/<timestamp>/`.

## Evaluation

```bash
./run_inference.sh --test_root data/test_set_gaussian_0.005
```

Useful options (`python inference_targo.py --help` lists all of them):

| Option | Meaning |
| --- | --- |
| `--model`, `--sc_model_path` | grasp and shape-completion checkpoints (default: `checkpoints/...`) |
| `--max_scenes N` | evaluate only the first `N` scenes (`0` = all) |
| `--result_root DIR` | where results are written (default: `eval_results`) |
| `--vis true` | export scene, grasp and completion visualisations for every scene |
| `--sim-gui true` | show the PyBullet GUI |

Each run writes to `<result_root>/targo/<timestamp>/`: `sr_rate.txt` (overall success rate and success
rate per occlusion level), `occ_level_sr.json`, `meta_evaluations.txt`, `intermediate_result.txt`
(one line per attempted grasp) and per-target statistics as JSON.

The launchers run from the repository root and use `python3` from your `PATH`; set
`TARGO_PYTHON=/path/to/python` to choose another interpreter.

## Training

```bash
./run_train.sh --dataset data/syn_train --dataset_raw data/syn_train \
  --epochs 50 --batch-size 2 --lr 2e-4 --use_wandb
```

Checkpoints and TensorBoard logs are written to `train_logs_targo/<run>/`. After every epoch the current
model is evaluated in simulation on `--val_max_scenes` scenes (default 50) of `--val_root`
(default `data/test_set_gaussian_0.005`); this success rate selects the `best_vgn_*.pt` checkpoint.
`--use_wandb` additionally logs the metrics to a Weights & Biases project called `targo`.

## Acknowledgements

This code base builds on [VGN](https://github.com/ethz-asl/vgn) and [GIGA](https://github.com/UT-Austin-RPL/GIGA)
(simulation, data format and grasp detection),
[ConvONets](https://github.com/autonomousvision/convolutional_occupancy_networks) (occupancy network components),
[PoinTr](https://github.com/yuxumin/PoinTr) (AdaPoinTr shape completion and the chamfer distance extension),
[Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) (`pointnet2_ops`),
[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and
[linear-attention-transformer](https://github.com/lucidrains/linear-attention-transformer).
Third-party components keep their original licenses (see the `LICENSE` files in the respective folders).

## License

This repository is released under the MIT License (see [LICENSE](LICENSE)).

## Citation

```bibtex
@article{xia2026targo,
  title={TARGO and TARGO-Net: Benchmarking Target-Driven Object Grasping Under Occlusions},
  author={Xia, Yan and Ding, Ran and Qin, Ziyuan and Zhan, Guanqi and Zhou, Kaichen and Yang, Long and Dong, Hao and Cremers, Daniel},
  journal={International Journal of Computer Vision},
  year={2026},
  doi={10.1007/s11263-025-02716-9}
}
```
