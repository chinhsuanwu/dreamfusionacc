# DreamFusionAcc

## Overview

This is a minimal PyTorch implementation of [DreamFusion](https://arxiv.org/abs/2209.14988) and its variant [Magic3D](https://arxiv.org/abs/2211.10440), where we utilize [NerfAcc](https://github.com/KAIR-BAIR/nerfacc) as the neural renderer and [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion)/[DeepFloyd IF](https://github.com/deep-floyd/IF) as the guidance.

![](https://github.com/chinhsuanwu/dreamfusionacc/assets/67839539/3793259d-b4a1-4c9e-83e9-fc0416eadf4c)

It takes ~30mins to train on a single 3090.

⚠️ Please use [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) or [threestudio](https://github.com/threestudio-project/threestudio) for higher quality 3D generation. This repo is not well-optimized, but it is a lightweight implementation that might be a good starting point if you are interested.


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
You can find all controllable settings in the yaml file.
After the training is done, run
```bash
python test.py --config config/peacock.yaml
```
to render 360 visualizations.



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

<a href="https://github.com/KAIR-BAIR/nerfacc">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/3310961/199083722-881a2372-62c1-4255-8521-31a95a721851.png" />
    <img alt="tyro logo" src="https://user-images.githubusercontent.com/3310961/199084143-0d63eb40-3f35-48d2-a9d5-78d1d60b7d66.png" width="250px" />
</picture>
</a>