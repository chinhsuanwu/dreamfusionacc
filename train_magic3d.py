import argparse
import os
import shutil
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from nerfacc import ContractionType, OccupancyGrid
from tqdm import tqdm, trange

from dataset.dreamfusion import DreamFusionLoader
from model.ngp import NGPradianceField
from prior.diffusion import StableDiffusion
from utils import render_image, set_random_seed
