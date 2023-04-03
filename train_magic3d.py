import argparse
import os
import shutil
import time
import yaml

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from attrdict import AttrDict
from einops import rearrange
from nerfacc import ContractionType, OccupancyGrid
from tqdm import tqdm, trange

from dataset.dreamfusion import DreamFusionLoader
from model.ngp import NGPradianceField
from prior.diffusion import StableDiffusion
from utils import render_image, set_random_seed


import pdb


def read_config(fn: str = "config/magic3d.yaml"):
    with open(fn, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config = AttrDict(config)
    str2list = lambda s: [float(item) for item in s.split(",")]
    config.aabb = str2list(config.aabb)
    config.shading_sample_prob = str2list(config.shading_sample_prob)
    return config


if __name__ == "__main__":
    config = read_config()

    pdb.set_trace()
