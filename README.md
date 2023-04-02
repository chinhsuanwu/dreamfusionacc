# DreamFusionAcc

## Overview

This is a minimal PyTorch implementation of [DreamFusion](https://arxiv.org/abs/2209.14988) and its variant [Magic3D](https://arxiv.org/abs/2211.10440), where we utilize [NerfAcc](https://github.com/KAIR-BAIR/nerfacc) to handle the neural rendering part.

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