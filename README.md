# DreamFusionAcc

## Overview

This is a minimal PyTorch implementation of [DreamFusion](https://arxiv.org/abs/2209.14988) and its variant [Magic3D](https://arxiv.org/abs/2211.10440), where we utilize [NerfAcc](https://github.com/KAIR-BAIR/nerfacc) to handle the neural rendering part.

![](https://github.com/chinhsuanwu/dreamfusionacc/assets/67839539/4e274e05-c9c8-464e-abfb-29c0049f2833)

It takes ~30mins to train on a single 3090.

⚠️ Please use [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) or [threestudio](https://github.com/threestudio-project/threestudio) for higher quality 3D generation, this repo is not optimized but a light-weight re-implementation.


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

## Howtos
To train
```bash
python train.py --config config/peacock.yaml
```
You can find all controllable settings in those *.yaml files.
After the training is done, run
```bash
python test.py --config config/peacock.yaml
```
to get 360 visualizations.



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