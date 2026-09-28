#!/usr/bin/env sh
# Build the chamfer distance and pointnet2_ops CUDA extensions in place.
# CUDA_HOME must point to a CUDA toolkit matching your PyTorch build.
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT_DIR/chamfer_dist"
python setup.py build_ext --inplace

cd "$ROOT_DIR/pointnet2_ops"
python setup.py build_ext --inplace
