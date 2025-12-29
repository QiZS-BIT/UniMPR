import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.netvlad import NetVLADLoupe
from torchvision.models.resnet import resnet18
from modules.gaussian_utils import GaussianRenderer, polar_quantizer
from modules.moe_module import MoETransformerEncoderLayer


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)), dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


class LCRFusionGCNExp(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.l_bev_bank = nn.Parameter(torch.randn(512, 7, 29))
        self.c_bev_bank = nn.Parameter(torch.randn(128, 7, 29))
        self.r_bev_bank = nn.Parameter(torch.randn(128, 7, 29))

        self.norm_layer = True
        self.num_channels = 384
        self.encoder = torch.hub.load(
            '/home/octane17/dinov3',
            'dinov3_vits16',
            source='local',
            pretrained=False
        )

        self.embed_dims = 64
        self.opacity_filter = 0.01
        self.gs_render = GaussianRenderer(self.embed_dims, self.opacity_filter)

        self.linear_dino = nn.Linear(self.num_channels, self.embed_dims)
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(True),
            nn.Linear(self.embed_dims, 1),
            nn.Tanh()
        )

        self.conv1_l = pretrained.conv1
        self.bn1_l = pretrained.bn1
        self.maxpool1_l = pretrained.maxpool
        self.layer1_l = pretrained.layer1
        self.layer2_l = pretrained.layer2
        self.layer3_l = pretrained.layer3
        self.layer4_l = pretrained.layer4

        self.conv1_r = copy.deepcopy(pretrained.conv1)
        self.bn1_r = copy.deepcopy(pretrained.bn1)
        self.maxpool1_r = copy.deepcopy(pretrained.maxpool)
        self.layer1_r = copy.deepcopy(pretrained.layer1)
        self.layer2_r = copy.deepcopy(pretrained.layer2)

        self.layer_c = copy.deepcopy(pretrained.layer2)

        self.pos_emb_l = LearnedPositionalEncoding(256, 7, 29)
        self.pos_emb_r = LearnedPositionalEncoding(64, 7, 29)
        self.pos_emb_c = LearnedPositionalEncoding(64, 7, 29)

        self.mlp_align_l = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        self.mlp_align_c = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )
        self.mlp_align_r = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )
        self.mm_encoder_layer = MoETransformerEncoderLayer(d_model=384, nhead=4, dim_feedforward=1024, dropout=0.1, batch_first=True, num_expert=8, top_k=2)
        self.mm_encoder = nn.TransformerEncoder(self.mm_encoder_layer, num_layers=1)

        self.net_vlad = NetVLADLoupe(
            feature_size=384,
            cluster_size=64,
            output_dim=256,
            gating=True,
            add_batch_norm=False
        )
        self.net_vlad_l = NetVLADLoupe(
            feature_size=512,
            cluster_size=64,
            output_dim=256,
            gating=True,
            add_batch_norm=False
        )
        self.net_vlad_r = NetVLADLoupe(
            feature_size=128,
            cluster_size=32,
            output_dim=256,
            gating=True,
            add_batch_norm=False
        )
        self.net_vlad_c = NetVLADLoupe(
            feature_size=128,
            cluster_size=32,
            output_dim=256,
            gating=True,
            add_batch_norm=False
        )

        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(128)
        self.norm4 = nn.LayerNorm(128)

    def forward(self, img, anchor_points, meta, l_bev, r_bev):
        if img.ndim != 1:
            img = img.view(img.shape[0] * img.shape[1], img.shape[2], 3, img.shape[4], img.shape[5])
            anchor_points = anchor_points.view(anchor_points.shape[0] * anchor_points.shape[1], anchor_points.shape[-2], 3)

            token, feat = self.encoder_forward(img)
            anchor_points_pixel, cam_mask = self.point_sampling(anchor_points, meta, feat.shape[-1])
            features = self.sample_features(feat, anchor_points_pixel, cam_mask)

            features_embed = self.linear_dino(features)
            xyz_polar = polar_quantizer(anchor_points)
            opacities = self.mlp_opacity(features_embed)
            cov_precomp = torch.tensor(([1.0000e-04, 0.0000e+00, 0.0000e+00, 1.0000e-04, 0.0000e+00, 1.0000e-04]),
                                       device=features_embed.device, dtype=features_embed.dtype)
            cov_precomp = cov_precomp.unsqueeze(0).unsqueeze(0).repeat(img.shape[0], anchor_points.shape[-2], 1)
            feat_c_bev, num_gaussians = self.gs_render(features_embed, xyz_polar, cov_precomp, opacities)
            feat_c_bev = feat_c_bev.flip(dims=[-2])
            feat_c_bev = self.layer_c(feat_c_bev)
        else:
            feat_c_bev = self.c_bev_bank.expand(1, -1, -1, -1)

        if r_bev.ndim != 1:
            r_bev = r_bev.view(r_bev.shape[0] * r_bev.shape[1], 3, 50, 225)
            feat_r_bev = self.conv1_r(r_bev)
            feat_r_bev = self.bn1_r(feat_r_bev)
            feat_r_bev = self.maxpool1_r(feat_r_bev)
            feat_r_bev = self.layer1_r(feat_r_bev)
            feat_r_bev = self.layer2_r(feat_r_bev)
        else:
            feat_r_bev = self.r_bev_bank.expand(1, -1, -1, -1)

        if l_bev.ndim != 1:
            l_bev = l_bev.view(l_bev.shape[0] * l_bev.shape[1], 3, 200, 900)
            feat_l_bev = self.conv1_l(l_bev)
            feat_l_bev = self.bn1_l(feat_l_bev)
            feat_l_bev = self.relu(feat_l_bev)
            feat_l_bev = self.maxpool1_l(feat_l_bev)
            feat_l_bev = self.layer1_l(feat_l_bev)
            feat_l_bev = self.layer2_l(feat_l_bev)
            feat_l_bev = self.layer3_l(feat_l_bev)
            feat_l_bev = self.layer4_l(feat_l_bev)
        else:
            feat_l_bev = self.l_bev_bank.expand(1, -1, -1, -1)

        # positional embedding
        feat_l_bev_shape = feat_l_bev.shape
        bev_mask_l = torch.zeros((feat_l_bev_shape[0], feat_l_bev_shape[2], feat_l_bev_shape[3]), device=feat_l_bev.device, dtype=feat_l_bev.dtype)
        if img.ndim != 1:
            pos_c = self.pos_emb_c(bev_mask_l)
            feat_c_bev = feat_c_bev + pos_c
        if r_bev.ndim != 1:
            pos_r = self.pos_emb_r(bev_mask_l)
            feat_r_bev = feat_r_bev + pos_r
        if l_bev.ndim != 1:
            pos_l = self.pos_emb_l(bev_mask_l)
            feat_l_bev = feat_l_bev + pos_l

        feat_l_bev = feat_l_bev.flatten(2).permute(0, 2, 1)
        feat_r_bev = feat_r_bev.flatten(2).permute(0, 2, 1)
        feat_c_bev = feat_c_bev.flatten(2).permute(0, 2, 1)

        feat_l_bev_align = self.mlp_align_l(feat_l_bev)
        feat_c_bev_align = self.mlp_align_c(feat_c_bev)
        feat_r_bev_align = self.mlp_align_r(feat_r_bev)
        fuse_l_bev = torch.cat([feat_l_bev_align, feat_c_bev_align, feat_r_bev_align], dim=-1)
        fuse_l_bev = self.mm_encoder(fuse_l_bev)

        # aggregation
        descriptor = self.net_vlad(fuse_l_bev)
        descriptor = F.normalize(descriptor, dim=1)
        descriptor_l = self.net_vlad_l(feat_l_bev)
        descriptor_l = F.normalize(descriptor_l, dim=1)
        if r_bev.ndim != 1:
            descriptor_r = self.net_vlad_r(feat_r_bev)
            descriptor_r = F.normalize(descriptor_r, dim=1)
        else:
            descriptor_r = torch.zeros_like(descriptor_l)
        if img.ndim != 1:
            descriptor_c = self.net_vlad_c(feat_c_bev)
            descriptor_c = F.normalize(descriptor_c, dim=1)
        else:
            descriptor_c = torch.zeros_like(descriptor_l)

        descriptor_concat = torch.cat([descriptor, descriptor_l, descriptor_r, descriptor_c], dim=-1)

        return descriptor_l, descriptor_r, descriptor_c, descriptor_concat

    @classmethod
    def create(cls, weights=None):
        if weights is not None:
            pretrained = resnet18(weights=weights)
        else:
            pretrained = resnet18()
        model = cls(pretrained)
        return model

    def encoder_forward(self, img):
        B, N, C, H, W = img.shape
        img = img.view(B * N, C, H, W)

        t2_x, hw_tuple = self.encoder.prepare_tokens_with_masks(img, None)
        x = [t2_x]
        rope = [hw_tuple]

        with torch.no_grad():
            for _, blk in enumerate(self.encoder.blocks[:-4]):
                if self.encoder.rope_embed is not None:
                    rope_sincos = [self.encoder.rope_embed(H=H, W=W) for H, W in rope]
                else:
                    rope_sincos = [None for r in rope]
                x = blk(x, rope_sincos)
        x[0] = x[0].detach()
        for _, blk in enumerate(self.encoder.blocks[-4:]):
            if self.encoder.rope_embed is not None:
                rope_sincos = [self.encoder.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)

        x_processed = x[0]
        x_norm = self.encoder.norm(x_processed)
        x_norm_cls_reg = x_norm[:, : self.encoder.n_storage_tokens + 1]
        x_norm_patch = x_norm[:, self.encoder.n_storage_tokens + 1:]

        t = x_norm_cls_reg
        f = x_norm_patch.reshape((B, N, H // 16, W // 16, self.num_channels)).permute(0, 1, 4, 2, 3)
        return t, f

    @staticmethod
    def point_sampling(reference_points, img_metas, fmap_size):
        img_metas['c_l'] = img_metas['c_l'].reshape(-1, img_metas['intr'].shape[1], 4, 4)
        img_metas['intr'] = img_metas['intr'][0:1, :, :, :].repeat(img_metas['c_l'].shape[0], 1, 1, 1)
        img_metas['intr'][..., 0, 0] *= (fmap_size / img_metas['img_size'][0][0])
        img_metas['intr'][..., 0, 2] *= (fmap_size / img_metas['img_size'][0][0])
        img_metas['intr'][..., 1, 1] *= (fmap_size / img_metas['img_size'][0][1])
        img_metas['intr'][..., 1, 2] *= (fmap_size / img_metas['img_size'][0][1])

        lidar2img_extr = reference_points.new_tensor(img_metas['c_l'])  # (B, N, 4, 4)
        lidar2img_intr = reference_points.new_tensor(img_metas['intr'])
        reference_points = reference_points.clone()

        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

        B, num_query = reference_points.size()[:2]
        num_cam = lidar2img_extr.size(1)

        # (bs, cam, num_query, 4, 1)
        reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img_extr = lidar2img_extr.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
        lidar2img_intr = lidar2img_intr.view(B, num_cam, 1, 3, 3).repeat(1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img_extr.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)[..., 0:3]
        eps = 1e-5
        cam_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_pixel = torch.matmul(lidar2img_intr.to(torch.float32),
                                              reference_points_cam.to(torch.float32).unsqueeze(-1)).squeeze(-1)
        reference_points_pixel = reference_points_pixel[..., 0:2] / torch.maximum(
            reference_points_pixel[..., 2:3], torch.ones_like(reference_points_pixel[..., 2:3]) * eps)

        reference_points_pixel[..., 0] /= fmap_size
        reference_points_pixel[..., 1] /= fmap_size

        cam_mask = (cam_mask & (reference_points_pixel[..., 1:2] > 0.0)
                    & (reference_points_pixel[..., 1:2] < 1.0)
                    & (reference_points_pixel[..., 0:1] < 1.0)
                    & (reference_points_pixel[..., 0:1] > 0.0))
        cam_mask = cam_mask.new_tensor(np.nan_to_num(cam_mask.cpu().numpy()))

        cam_mask = cam_mask.squeeze(-1)

        return reference_points_pixel, cam_mask

    @staticmethod
    def sample_features(feature_maps, pixel_coords, valid_masks):
        # B: batch size, C: camera num, D: feature channel, N: anchor points num
        B, C, D, H, W = feature_maps.shape
        N = pixel_coords.shape[2]

        grid = pixel_coords * 2.0 - 1.0
        invalid_mask = ~valid_masks.unsqueeze(-1)  # [B, C, N, 1]
        grid = grid.masked_fill(invalid_mask, 2.0)
        grid = grid.view(B * C, N, 1, 2)

        # B*C, N, 1, 2 -> B*C, D, N, 1 -> B*C, D, N -> B*C, N, D
        sampled = F.grid_sample(
            feature_maps.view(B * C, D, H, W),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(-1).permute(0, 2, 1).view(B, C, N, D)

        num_valid = valid_masks.sum(dim=1, keepdim=True).float().clamp(min=1).permute(0, 2, 1)  # [B, N, 1]

        sum_features = sampled.sum(dim=1)  # [B, N, D]
        final_features = sum_features / num_valid  # [B, N, D]

        return final_features
