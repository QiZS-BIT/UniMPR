import math
import torch
import torch.nn as nn
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class BEVCamera:
    def __init__(self, x_range=(0, 360), y_range=(0, 80), image_size_w=57, image_size_h=13):
        # Orthographic projection parameters
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.image_width = image_size_w
        self.image_height = image_size_h

        # Set up FoV to cover the range [-50, 50] for both X and Y
        self.FoVx = (self.x_max - self.x_min)  # Width of the scene in world coordinates
        self.FoVy = (self.y_max - self.y_min)  # Height of the scene in world coordinates

        # Camera position: placed above the scene, looking down along Z-axis
        # It has been disabled actually
        self.camera_center = torch.tensor([0, 0, 0], dtype=torch.float32)  # High above Z-axis

        # Orthographic projection matrix for BEV
        self.set_transform()

    def set_transform(self, h=200, w=200, h_meters=80, w_meters=360):
        """ Set up an orthographic projection matrix for BEV. """
        # Create an orthographic projection matrix
        sh = h / h_meters
        sw = w / w_meters
        self.world_view_transform = torch.tensor([
            [0., sh, 0., 0.],
            [sw, 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ], dtype=torch.float32)

        self.full_proj_transform = torch.tensor([
            [0., -sh, 0., h / 2.],
            [-sw, 0., 0., w / 2.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
        ], dtype=torch.float32)
        self.full_proj_transform = torch.tensor([
            [0., -sh, 0., h],
            [-sw, 0., 0., w],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
        ], dtype=torch.float32)

    def set_size(self, h, w):
        self.image_height = h
        self.image_width = w


class GaussianRenderer(nn.Module):
    def __init__(self, embed_dims, threshold=0.05):
        super().__init__()
        self.viewpoint_camera = BEVCamera()
        self.rasterizer = GaussianRasterizer()
        self.embed_dims = embed_dims
        self.threshold = threshold

    def forward(self, features, means3D, cov3D, opacities):
        """
        features: b G d
        means3D: b G 3
        uncertainty: b G 6
        opacities: b G 1
        """
        b = features.shape[0]
        device = means3D.device

        bev_out = []
        mask = (opacities > self.threshold)
        mask = mask.squeeze(-1)
        self.set_render_scale(13, 57)
        self.set_Rasterizer(device)
        for i in range(b):
            rendered_bev, _ = self.rasterizer(
                means3D=means3D[i][mask[i]],
                means2D=None,
                shs=None,  # No SHs used
                colors_precomp=features[i][mask[i]],
                opacities=opacities[i][mask[i]],
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D[i][mask[i]]
            )
            bev_out.append(rendered_bev)

        x = torch.stack(bev_out, dim=0)  # b d h w
        num_gaussians = (mask.detach().float().sum(1)).mean().cpu()

        return x, num_gaussians

    @torch.no_grad()
    def set_Rasterizer(self, device):
        tanfovx = math.tan(self.viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(self.viewpoint_camera.FoVy * 0.5)

        bg_color = torch.zeros((self.embed_dims)).to(device)  # self.embed_dims
        # bg_color[-1] = -4
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.viewpoint_camera.image_height),
            image_width=int(self.viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1,
            viewmatrix=self.viewpoint_camera.world_view_transform.to(device),
            projmatrix=self.viewpoint_camera.full_proj_transform.to(device),
            sh_degree=0,  # No SHs used
            campos=self.viewpoint_camera.camera_center.to(device),
            prefiltered=False,
            debug=False
        )
        self.rasterizer.set_raster_settings(raster_settings)

    @torch.no_grad()
    def set_render_scale(self, h, w):
        self.viewpoint_camera.set_size(h, w)
        self.viewpoint_camera.set_transform(h, w)


def polar_quantizer(xyz):
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    theta = 180. + torch.atan2(y, x) * 180. / torch.pi
    theta = torch.where(theta == 360., theta - 0.0001, theta)
    dist = torch.sqrt(x ** 2 + y ** 2)
    coord = torch.stack([theta, dist, z], dim=-1)
    return coord
