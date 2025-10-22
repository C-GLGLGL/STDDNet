import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from lib.mamba.models_mamba import VisionMamba
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout(drop)

        self.conv1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)

        self.apply(self._init_weights)
    
    def dwconv(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x

class DDFE(nn.Module):
    def __init__(self, img_size=(64, 128), channels=32, embed_dim=256, depth=4, f_num=1, **kwargs):
        super(DDFE, self).__init__()

        in_channels = channels * f_num
        self.group = True
        if not self.group:
            f_num = 1
            in_channels = channels * 2

        patch_size = 2
        self.downsample_ratio_1 = 1 / patch_size
        self.patch_size_1 = patch_size
        # middle_embeds = embed_dim // 2
        self.mamba_1 = VisionMamba(
            patch_size=patch_size, img_size=img_size, channels=in_channels, d_state=1, d_conv=3, 
            conv_bias=False, mask_guide=True, f_num=f_num,
            stride=patch_size, embed_dim=in_channels, depth=1, drop_path_rate=0., rms_norm=False,
            residual_in_fp32=True, fused_add_norm=True, final_pool_type='all', 
            if_abs_pos_embed=False, bimamba_type="v2", if_cls_token=False, 
            if_devide_out=True, use_middle_cls_token=False, **kwargs
        )
        middle_size = tuple(elem // patch_size for elem in img_size)
        self.patch_size_2 = 2
        self.downsample_ratio_2 = 1 / self.patch_size_2
        self.mamba_2 = VisionMamba(
            patch_size=self.patch_size_2, img_size=middle_size, channels=in_channels, d_state=1, d_conv=3,
            conv_bias=False, mask_guide=True, f_num=f_num,
            stride=self.patch_size_2, embed_dim=embed_dim, depth=1, drop_path_rate=0., rms_norm=False,
            residual_in_fp32=True, fused_add_norm=True, final_pool_type='all', 
            if_abs_pos_embed=False, bimamba_type="v2", if_cls_token=False, 
            if_devide_out=True, use_middle_cls_token=False, **kwargs
        )
        self.mamba_1.default_cfg = _cfg()
        self.mamba_2.default_cfg = _cfg()

        self.filter = Mlp(in_features=embed_dim, hidden_features=channels*2, out_features=channels)
        self.channels = channels

        self.s_fea_mem = None

        ### GRU ###
        self.gru = nn.Sequential(
            nn.Conv2d(channels*2, channels*3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels*3),
            # nn.PReLU(),
            nn.Conv2d(channels*3, channels*3, kernel_size=3, stride=1, padding=1),
        )

        self.lf_temp = None
        self.temp = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(1, (44, 44))
        )


    def forward(self, x, masks):
        B, T, C, H, W = x.shape
        if self.group:
            f = x[:, 0]
            for ti in range(1, T):
                f = torch.cat([f, x[:, ti]], dim=1)
            m = F.interpolate(1-masks.reshape(B, T, H, W), scale_factor=self.downsample_ratio_1, mode='bilinear', align_corners=False)
            structure_f = self.mamba_1(f, (m, m))
            m1 = F.interpolate(m, scale_factor=self.downsample_ratio_2, mode='bilinear', align_corners=False)
            sf_H = H // self.patch_size_1
            sf_W = W // self.patch_size_1
            structure_f = self.mamba_2(structure_f.reshape(B, structure_f.shape[-1], sf_H, sf_W), (m1, m1))
            sf_H = sf_H // self.patch_size_2
            sf_W = sf_W // self.patch_size_2
            sf_C = structure_f.shape[-1]
            structure_f = F.interpolate(structure_f.reshape(B, structure_f.shape[-1], sf_H, sf_W), size=(H, W), mode='bilinear', align_corners=False)
            structure_f = self.filter(structure_f.reshape(B, H*W, sf_C), H, W).reshape(B, self.channels, H, W)
            self.s_fea_mem = structure_f
        else:
            f0 = x[:, 0]
            self.s_fea_mem = f0
            for ti in range(1, T):
                f1 = self.s_fea_mem
                f2 = x[:, ti]
                # channel-wise concat
                f = torch.cat([f1, f2], dim=1)
                # mask-embeddings
                m1 = F.interpolate(1-masks[:, ti], scale_factor=self.downsample_ratio_1, mode='bilinear', align_corners=False)
                structure_f = self.mamba_1(f, (m1, m1))
                m1 = F.interpolate(m1, scale_factor=self.downsample_ratio_2, mode='bilinear', align_corners=False)
                sf_H = H // self.patch_size_1
                sf_W = W // self.patch_size_1
                structure_f = self.mamba_2(structure_f.reshape(B, structure_f.shape[-1], sf_H, sf_W), (m1, m1))
                sf_H = sf_H // self.patch_size_2
                sf_W = sf_W // self.patch_size_2
                sf_C = structure_f.shape[-1]
                structure_f = F.interpolate(structure_f.reshape(B, structure_f.shape[-1], sf_H, sf_W), size=(H, W), mode='bilinear', align_corners=False)
                structure_f = self.filter(structure_f.reshape(B, H*W, sf_C), H, W).reshape(B, self.channels, H, W)
                self.s_fea_mem = structure_f
        

        self.lf_temp = masks[:, 0]
        for ti in range(0, T):
            d_f = torch.cat([self.s_fea_mem, x[:, ti]], dim=1)
            values = self.gru(d_f)
            static_v = torch.sigmoid(values[:, :self.channels])
            varying_v = torch.sigmoid(values[:, self.channels:self.channels*2])
            dynamic_v = torch.tanh(values[:, self.channels*2:])
            dynamic_f = static_v * self.s_fea_mem * (1-varying_v) + varying_v * dynamic_v

            ratio = self.temp(torch.cat([self.lf_temp, masks[:, ti]], dim=1))
            x[:, ti] = x[:, ti] + dynamic_f * ratio
            self.lf_temp = masks[:, ti]
            
        return x