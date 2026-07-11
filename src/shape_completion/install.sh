#!/usr/bin/env sh
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT_DIR/chamfer_dist"
python setup.py install

cd "$ROOT_DIR/pointnet2_ops"
python setup.py install
