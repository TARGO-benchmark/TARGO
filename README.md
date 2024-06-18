# TARGO: Benchmarking Target-driven Object Grasping under Occlusions

[Yan Xia](https://yan-xia.github.io), [Ran Ding](https://randing2000.github.io), [Ziyuan Qin](https://scholar.google.com/citations?user=pcCR2lUAAAAJ), [Guanqi Zhan](https://www.robots.ox.ac.uk/~guanqi), [Kaichen Zhou](https://www.cs.ox.ac.uk/people/kaichen.zhou), [Long Yang](https://scholar.google.com/citations?user=KOTg0mQAAAAJ&hl), [Hao Dong](https://zsdonghao.github.io), [Daniel Cremers](https://cvg.cit.tum.de/members/cremers)

## Introduction

Recent advances in predicting 6D grasp poses from a single depth image have resulted in a promising performance in robotic grasping. However, previous grasping models face challenges in cluttered environments where nearby objects impact the target object’s grasp. In this paper, we first set up a new benchmark dataset for TARget-driven Grasping under Occlusions, named TARGO. We make the following contributions: 1) We are the first to study the occlusion level of grasping. 2) We set up an evaluation benchmark consisting of large-scale synthetic data and part of real-world data, and we evaluated five grasp models and found that even the current SOTA model suffers when the occlusion level increases, which makes grasping under occlusion still a challenge. 3) We also generate a large-scale training dataset via a scalable pipeline, which can be used to boost the performance of grasping under occlusion and zero-shot generate to the real world. 4) We further propose a transformer-based model involving a shape completion module, termed TARGO-Net, which performs most robustly as occlusion increases.

<!-- ## Installation -->

## Dataset Generation

### TARGO Dataset

TARGO dataset adds single scenes in addition to cluttered scenes in VGN dataset.

`python generate_targo_dataset.py --root <output dataset directory>`

### Shape Completion Dataset

Training shape completion requires no additional dataset: simply read in VGN dataset, and generate the corresponding dataset for shape completion.

`python generate_sc_dataset.py --root_mpl <VGN dataset's mesh pose list directory> --dest <h5 destination directory>`

## Inference

TARGO-Net predicts parallel-jaw gripper grasps.

`python inference.py`

<!-- insert img, should fix paths -->

## Acknowledgement

Our code is inspired by [VGN](https://github.com/ethz-asl/vgn) and [GIGA](https://github.com/UT-Austin-RPL/GIGA).

<!-- ## Citation -->
