# TARGO: Benchmarking Target-driven Object Grasping under Occlusions

[Yan Xia](https://yan-xia.github.io)\*, [Ran Ding](https://randing2000.github.io)\*, [Ziyuan Qin](https://scholar.google.com/citations?user=pcCR2lUAAAAJ)\*, [Guanqi Zhan](https://www.robots.ox.ac.uk/~guanqi), [Kaichen Zhou](https://www.cs.ox.ac.uk/people/kaichen.zhou), [Long Yang](https://scholar.google.com/citations?user=KOTg0mQAAAAJ&hl), [Hao Dong](https://zsdonghao.github.io), [Daniel Cremers](https://cvg.cit.tum.de/members/cremers)

## Introduction

Recent advances in predicting 6D grasp poses from a single depth image have resulted in a promising performance in robotic grasping. However, previous grasping models face challenges in cluttered environments where nearby objects impact the target object’s grasp. In this paper, we first set up a new benchmark dataset for TARget-driven Grasping under Occlusions, named TARGO. We make the following contributions: 1) We are the first to study the occlusion level of grasping. 2) We set up an evaluation benchmark consisting of large-scale synthetic data and part of real-world data, and we evaluated five grasp models and found that even the current SOTA model suffers when the occlusion level increases, which makes grasping under occlusion still a challenge. 3) We also generate a large-scale training dataset via a scalable pipeline, which can be used to boost the performance of grasping under occlusion and zero-shot generate to the real world. 4) We further propose a transformer-based model involving a shape completion module, termed TARGO-Net, which performs most robustly as occlusion increases.

<!-- ## Installation -->
## Getting started
### Environment Setup
<details>
<summary>Pytorch 1.12.1, Cuda 11.3</summary>

The environment `targo` is set up with the following key configurations:

- **Python version**: 3.9.20  
- **PyTorch version**: 1.12.1+cu113  
- **Torchvision version**: 0.13.1+cu113  
- **Torchaudio version**: 0.12.1+cu113  
- **CUDA version**: 11.3  

Other notable packages include:
- **NumPy**: 2.0.2  
- **Matplotlib**: 3.9.2  
- **Open3D**: 0.18.0  
- **Scikit-learn**: 1.5.2  
- **Scipy**: 1.9.0  
</details>


### Dataset
You can either **download our dataset** or **generate it yourself**. Please place the dataset in the `datasets` directory. A download link for the dataset will be provided later. Demo data is already available in `datasets/demo`.

### Checkpoints
Checkpoints can either be **downloaded directly** or **generated using our provided scripts**.


## Dataset Generation

### TARGO Dataset
The TARGO dataset extends the VGN dataset by adding both single-object scenes and cluttered scenes.

To generate the TARGO dataset:
```bash
python scripts/generate_targo_dataset.py --root <output dataset directory>
```

### Shape Completion Dataset
For shape completion, no external dataset is required. Simply use the VGN dataset and generate the shape completion dataset.

To generate the shape completion dataset:
```bash
python scripts/generate_sc_dataset.py --root_mpl <VGN dataset's mesh pose list directory> --dest <h5 destination directory>
```


## Training
Train the TARGO-Net with the following command:
```bash
python scripts/train_targo.py --dataset_raw
```


## Benchmark
All models are evaluated on the `targo_synthetic` test set for target-driven object grasping. A grasp is considered successful only if it picks up the target object and remains collision-free throughout planning.

To run inference for different models:
```bash
python scripts/inference.py --net TARGO | giga | giga_aff | vgn | icgnet | edgegraspnet | VN-edgegraspnet
```

<!-- Insert relevant images and ensure paths are fixed -->


## Acknowledgements
This project builds upon ideas and codebases from [VGN](https://github.com/ethz-asl/vgn) and [GIGA](https://github.com/UT-Austin-RPL/GIGA). We express our gratitude to the authors for laying the groundwork and inspiring this work.

<!-- ## Citation -->
### Citation
If you find this work useful, please cite:
```bibtex
@article{xia2024targo,
  title={TARGO: Benchmarking Target-driven Object Grasping under Occlusions},
  author={Xia, Yan and Ding, Ran and Qin, Ziyuan and Zhan, Guanqi and Zhou, Kaichen and Yang, Long and Dong, Hao and Cremers, Daniel},
  journal={arXiv preprint arXiv:2407.06168},
  year={2024}
}
```

