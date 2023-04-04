# DreamFusionAcc

## Overview

This is a minimal PyTorch implementation of [DreamFusion](https://arxiv.org/abs/2209.14988) and its variant [Magic3D](https://arxiv.org/abs/2211.10440), where we utilize [NerfAcc](https://github.com/KAIR-BAIR/nerfacc) to handle the neural rendering part.

- A highly detailed stone bust of Theodoros Kolokotronis
- A metal sculpture of a lion's head, highly detailed
- A highly detailed sandcastle
- A lemur taking notes in a journal
- A DSLR photo of a peacock on a surfboard

It takes around ~30mins to train a NeRF model given a text prompt on a single 3090.

## Installation
```
git clone https://github.com/chinhsuanwu/dreamfusionacc.git
cd dreamfusionacc
pip install -r requirements.txt
```
<details>
  <summary> Dependencies (click to expand) </summary>

  ## Dependencies
  - torch
  - nerfacc
  - numpy
  - imageio
  - einops
  - diffusers
  - trainsformers

  [NerfAcc](https://github.com/KAIR-BAIR/nerfacc) provides pre-built wheels covering major combinations of Pytorch + CUDA. This repo is built upon torch 1.13.0 + cu117.
</details>


## Usage

To train Magic3D:
```bash
python train.py --config config/magic3d.yaml
```
To train DreamFusion:
```bash
python train.py --config config/dreamfusion.yaml
```
You can find all controllable settings in those *.yaml files.


## Citation

```bibtex
@article{poole2022dreamfusion,
  author = {Poole, Ben and Jain, Ajay and Barron, Jonathan T. and Mildenhall, Ben},
  title = {DreamFusion: Text-to-3D using 2D Diffusion},
  journal = {arXiv},
  year = {2022},
}
@inproceedings{lin2023magic3d,
  title={Magic3D: High-Resolution Text-to-3D Content Creation},
  author={Lin, Chen-Hsuan and Gao, Jun and Tang, Luming and Takikawa, Towaki and Zeng, Xiaohui and Huang, Xun and Kreis, Karsten and Fidler, Sanja and Liu, Ming-Yu and Lin, Tsung-Yi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2023}
}
```

## Acknowledgments

This implementation is heavily based on [NerfAcc](https://github.com/KAIR-BAIR/nerfacc) and [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion).