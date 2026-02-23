# Fore-Mamba3D: Mamba-based Foreground-Enhanced Encoding for 3D Object Detection

> **ICLR 2026** | Accepted

## Overview

Fore-Mamba3D is a novel backbone architecture for 3D object detection that leverages Mamba-based linear modeling with foreground-enhanced encoding. Existing Mamba-based methods encode the entire scene including abundant background information, which is inefficient. Our method achieves superior performance by focusing on foreground voxels with enhanced contextual representation.

## Key Contributions

1. **Foreground-Enhanced Linear Encoding**: Selects top-k foreground voxels via predicted scores, significantly reducing computational cost while maintaining accuracy

2. **Regional-to-Global Sliding Window (RGSW)**: Aggregates information from local patches to the entire sequence, addressing response attenuation in sparse foreground voxels

3. **SASFMamba Module**: Enhances contextual representation through:
   - **Semantic-Assisted Fusion (SAF)**: Enables long-range interactions between semantically similar voxels
   - **State Spatial Fusion (SSF)**: Recovers geometric information lost in linearization

## Method Overview

![Framework](imgs/framework.pdf)

The backbone consists of four stages, each containing an instance selection block and a downsampling block. The instance selection block includes foreground voxel sampling, RGSW strategy, and SASFMamba encoder for effective foreground-focused 3D object detection.

## Installation

```bash
git clone https://github.com/yourusername/Fore-Mamba3D.git
cd Fore-Mamba3D

# Create conda environment
conda create -n fore-mamba3d python=3.10
conda activate fore-mamba3d

# Install dependencies
pip install -r requirements.txt

# Install torch with CUDA support (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dataset Preparation

### KITTI Dataset

```bash
# Download KITTI dataset
# Place it in the following structure:
data/kitti/
├── training/
│   ├── calib/
│   ├── image_2/
│   ├── label_2/
│   └── velodyne/
└── testing/
    ├── calib/
    ├── image_2/
    └── velodyne/
```

### nuScenes Dataset

```bash
# Download nuScenes dataset
# Place it in the following structure:
data/nuscenes/
├── samples/
├── sweeps/
├── maps/
└── v1.0-trainval/
```

## Training

```bash
# Single GPU training
python train.py --cfg configs/kitti_train.yaml

# Multi-GPU training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py --cfg configs/kitti_train.yaml
```

## Evaluation

```bash
# Evaluate on KITTI validation set
python eval.py --cfg configs/kitti_train.yaml \
    --ckpt path/to/checkpoint.pth

# Evaluate on nuScenes
python eval.py --cfg configs/nuscenes_train.yaml \
    --ckpt path/to/checkpoint.pth
```

## Results

### KITTI Dataset (Car Class)

| Method | Backbone | AP₃D (Easy) | AP₃D (Moderate) | AP₃D (Hard) |
|--------|----------|-----------|-----------------|-----------|
| Fore-Mamba3D | Mamba | **XX.XX** | **XX.XX** | **XX.XX** |

### nuScenes Dataset

| Method | Backbone | mAP | NDS |
|--------|----------|-----|-----|
| Fore-Mamba3D | Mamba | **XX.XX** | **XX.XX** |


## Citation

If you find this project useful in your research, please cite:

```bibtex
@article{ning2026foremamba3d,
  title={Fore-Mamba3D: Mamba-based Foreground-Enhanced Encoding for 3D Object Detection},
  author={Ning, Zhiwei and Gao, Xuanang and Cao, Jiaxi and Yang, Runze and Yang, Jie and Liu, Wei and Xu, Huiying and Zhu, Xinzhong},
  journal={arXiv preprint arXiv:2601.xxxxx},
  year={2026}
}
```

## Acknowledgments

This work is partially supported by:
- National Natural Science Foundation of China (Grant No. 62376153, 62402318, 24Z990200676, 62376252)
- Zhejiang Province Leading Geese Plan (2025C02025, 2025C01056)
- Zhejiang Province Province-Land Synergy Program (2025SDXT004-3)


## License

This project is released under the [MIT License](LICENSE).

## Contributing

We welcome contributions from the community. Please feel free to submit pull requests or open issues.

## Related Work

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.08956)
- [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337)
- [PointNet++: Deep Hierarchical Feature Learning on Point Sets](https://arxiv.org/abs/1706.02413)

