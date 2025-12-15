# Soft Task-Aware Routing of Experts for Equivariant Representation Learning [NeurIPS 2025 Poster]

[![Conference](https://img.shields.io/badge/NeurIPS-2025%20Poster-0b5fff.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/117730)
[![Paper](https://img.shields.io/badge/Paper-arXiv-4b9e5d.svg)](https://arxiv.org/abs/2510.27222)

This repository provides an official implementation of our NeurIPS 2025 Poster paper:
> [**Soft Task-Aware Routing of Experts for Equivariant Representation Learning**](https://arxiv.org/abs/2510.27222)  
> **Jaebyeong Jeon**, **Hyeonseo Jang**, **Jy-yong Sohn**, and **Kibok Lee**

## Citation
If you use this code or data, please consider citing our paper:
```BibTeX
@inproceedings{jeon2025soft,
  title={Soft Task-Aware Routing of Experts for Equivariant Representation Learning},
  author={Jeon, Jaebyeong and Jang, Hyeonseo and Sohn, Jy-yong and Lee, Kibok},
  booktitle={NeurIPS},
  year={2025}
}
```

## Requirements

- Python 3.8
- [Conda](https://docs.conda.io/en/latest/)
- `kornia=0.4.1`

## Installation

### 1. Create and activate the conda environment

We recommend using Python 3.8:

```bash
conda create -n E-SSL python=3.8
conda activate E-SSL
```

Please install a compatible version of torch and torchvision separately, using the PyTorch installation guide.
Make sure to select the latest versions compatible with Python 3.8 and your system:

```bash
# Example
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 2. Install additional Python dependencies

Install the remaining packages:

```bash
conda install ignite -y
pip install -r requirements.txt
```

## Checkpoints
Below are the checkpoints for the models trained on STL10 and ImageNet100 datasets.

- [STL10](https://drive.google.com/file/d/1m3kLKPr2BP8APHC4O-fZoWxe-mTOKBBc/view?usp=sharing)
- [ImageNet100](https://drive.google.com/file/d/1UNNiVqfDKIV7bcwEvQ0kqAWH9EtV9o8_/view?usp=sharing)

## Pretraining

```bash
# STL10
bash scripts/stl10.sh

# ImageNet100
bash scripts/imagenet100.sh
```

## Evaluation

We provide scripts for linear evaluation on out-of-domain classification dataset (Table 1) and for few-shot benchmarks (Table 4)

### Linear Evaluation

```bash
bash scripts/linear_eval.sh
```

### Few-Shot Classification

```bash
bash scripts/few_shot.sh
```

## Notes
- This work has been tested with Python 3.8.
- This work builds upon the following open-source projects:
    * [AugSelf](https://github.com/hankook/AugSelf)
    * [LightlySSL](https://github.com/lightly-ai/lightly)
