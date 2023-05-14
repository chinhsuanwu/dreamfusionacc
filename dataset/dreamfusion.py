import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import Rays, rand_poses, circle_poses


class DreamFusionLoader(Dataset):
    """Generates rays and direction texts in random views.

    Returns:
        - rays: [num_rays, 3]
        - direction: str
        - color_bkgd: [3,]
        - shading: str
    """

    WIDTH, HEIGHT = 128, 128
    NEAR, FAR = 0.1, 1.0e10
    OPENGL_CAMERA = True

    def __init__(
        self,
        size: int = 100,
        training: int = True,
        color_bkgd_aug: str = "white",
        width: int = None,
        height: int = None,
        near: float = None,
        far: float = None,
        shading_sample_prob: list = [1, 0, 0],
        device=None,
    ):
        super().__init__()
        self.size = size
        self.training = training
        self.color_bkgd_aug = color_bkgd_aug
        self.width = self.WIDTH if width is None else width
        self.height = self.HEIGHT if height is None else height
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.num_rays = self.width * self.height
        self.shading_sample_prob = shading_sample_prob
        self.device = device

        self.focal = 0.7 * self.width
        self.K = torch.tensor(
            [
                [self.focal, 0, self.width / 2.0],
                [0, self.focal, self.height / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )  # (3, 3)

        x, y = torch.meshgrid(
            torch.arange(self.width), torch.arange(self.height), indexing="xy"
        )
        self.x = x.flatten().to(self.device)
        self.y = y.flatten().to(self.device)

    def __len__(self):
        return self.size

    @torch.no_grad()
    def __getitem__(self, index):
        return self.fetch_novel_view(None if self.training else index)

    def compute_rays(self, c2w):
        # generate rays
        camera_dirs = F.pad(
            torch.stack(
                [
                    (self.x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (self.y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

        return Rays(origins=origins, viewdirs=viewdirs)

    def fetch_novel_view(self, index=None):
        # random camera focal
        if self.training:
            focal = np.random.uniform(0.7, 1.35) * self.width
        else:
            focal = self.focal
        self.K[0][0] = self.K[1][1] = focal

        if self.training:
            pose, direction = rand_poses(1, device=self.device, return_dirs=True)
        else:
            phi = (index / self.size) * 360
            pose, direction = circle_poses(
                radius=1.25, phi=phi, device=self.device, return_dirs=True
            )

        c2w = pose[0, :3, :].expand(self.num_rays, 3, 4)  # (num_rays, 3, 4)
        rays = self.compute_rays(c2w)

        direction = {
            0: "front",
            1: "side",
            2: "back",
            3: "side",
            4: "top",
            5: "bottom",
        }[direction.item()]

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.device)
            shading = np.random.choice(
                ["albedo", "textureless", "lambertian"], 1, p=self.shading_sample_prob
            ).item()
        else:
            # only use white during inference
            color_bkgd = torch.ones(3, device=self.device)
            shading = "albedo"

        return {
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "direction": direction,  # str
            "color_bkgd": color_bkgd,  # [3,]
            "shading": shading,  # str
        }
