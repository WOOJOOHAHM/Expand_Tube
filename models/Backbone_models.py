from collections import OrderedDict
from typing import Tuple, Union
from download import load

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from Positional_embeding import get_3d_sincos_pos_embed

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class expand_tubevit(nn.Module):
    def __init__(self, scale, width):
        super(expand_tubevit, self).__init__()
        self.spatial_start_point = [45, 48, 87, 90]
        self.patch_size = [3, 5, 7, 9]
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

    def get_patch_index(self, x, spatial_point, patch_size):
        if patch_size == 3:
            sp = spatial_point
        elif patch_size == 5:
            sp = spatial_point - 15
        elif patch_size == 7:
            sp = spatial_point - (15 * 2)
        elif patch_size == 9:
            sp = spatial_point - (15 * 3)

        patch_gap = (patch_size + 1) // 2
        additional_pathes = [sp, sp+patch_gap, sp+(patch_gap*2),
                                sp+(14 * patch_gap), sp+(14*patch_gap)+((patch_gap*2)),
                                sp+(14 * patch_gap*2), sp+(14 * patch_gap*2)+patch_gap, sp+(14 * patch_gap*2)+(patch_gap*2)]
        center_patch = [(14 * i) + sp + j + 1 for j in range(patch_size) for i in range(0, patch_size)] # j: 각 patch를 grid하게 탐색, i: 각 patch의 시작 point를 찾음
        selected_patch = additional_pathes + center_patch
        selected_patch.sort()
        selected_patch = torch.index_select(x.cpu(), 0, torch.tensor(selected_patch)).cuda()
        return selected_patch
    
    def forward(self, x):
        expand_tube = torch.empty(0, x.size(1), x.size(2), x.size(3)).cuda()

        for batch in range(x.size(0)):
            tube = torch.empty(0, x.size(2), x.size(3)).cuda()
            for frame_idx in range(x.size(1)):
                # 현재 프레임 선택
                distribute_frame = frame_idx % 4
                if distribute_frame == 0:
                    for spatial_point in self.spatial_start_point:
                        st_tube = torch.empty(0,768).cuda()
                        for i, idx in enumerate(range(frame_idx, frame_idx+4 , 1)):
                            current_frame = x[batch, idx, :, :]
                            selected_frame = self.get_patch_index(current_frame, spatial_point, self.patch_size[i])
                            st_tube = torch.cat([st_tube, selected_frame], dim=0)
                        st_tube = st_tube.unsqueeze(0)
                        tube = torch.cat([tube, st_tube], dim=0)
                else:
                    pass
            tube = tube.unsqueeze(0)
            expand_tube = torch.cat([expand_tube, tube], dim=0)
        return expand_tube
    
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, num_classes: int, classifier: str, num_frames: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.spatial_positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2, width))
        self.temporal_positional_embedding = nn.Parameter(scale * torch.randn(num_frames, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.dropout = nn.Dropout(0.5)
        self.classifier = classifier
        if self.classifier == 'mean' or self.classifier == 'difference':
            self.expand_fc = nn.Linear(width, num_classes)
            self.ln_expand_post = LayerNorm(width)
        elif self.classifier == 'span2':
            self.expand_fc = nn.Linear(width * 2, num_classes)
            self.ln_expand_post = LayerNorm(width * 2)
        nn.init.normal_(self.expand_fc.weight, std=0.02)
        nn.init.constant_(self.expand_fc.bias, 0.)
        self.expand_tubevit = expand_tubevit(scale, width)

    def _generate_position_embedding(self) -> torch.nn.Parameter:
        position_embedding = [torch.zeros(1, self.hidden_dim)]

        for i in range(len(self.kernel_sizes)):
            tube_shape = self._calc_conv_shape(self.kernel_sizes[i], self.strides[i], self.offsets[i])
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                tube_shape=tube_shape,
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                offset=self.offsets[i],
            )
            position_embedding.append(pos_embed)

        position_embedding = torch.cat(position_embedding, dim=0).contiguous()
        return position_embedding
    

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1) # Permute: (B,T,C,H,W), Flatten: (B*T, C, H, W)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        spatial_size = tuple(x.size()[2:])
        x = x.flatten(-2).permute(0, 2, 1) # shape = []
        x = x + self.spatial_positional_embedding.to(x.dtype)
        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1], x.size(-1))

        # Positional Embedding
        S_patch, embedded_patch = x.shape[2], x.shape[3]
        x = x.permute(0, 2, 1, 3).flatten(0, 1) + self.temporal_positional_embedding
        x = x.contiguous().view(B, T, S_patch, embedded_patch)

        # Expand tube vit
        x = self.expand_tubevit(x)
        x = x.flatten(0, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))
        
        if self.classifier == 'mean':
            x = x[:, :, 0, :].mean(dim=1)
        elif self.classifier == 'span2':
            x = torch.stack((x[:, 0:T//2, 0, :].mean(dim=1), x[:, T//2:, 0, :].mean(dim=1))).view(B, x.size()[3] * 2)
        elif self.classifier == 'difference':
            x = x[:, 0:T//2, 0, :].mean(dim=1) - x[:, T//2:, 0, :].mean(dim=1)
        
        x = self.ln_expand_post(x)
        x = self.dropout(x)
        x = self.expand_fc(x)
        return x
    
def build_model(model_name, download_root, num_classes, classifier, num_frames):
    state_dict = load(model_name, download_root)
    if 'ViT' in model_name:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        vision_heads = vision_width // 64
        model = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            num_classes=num_classes,
            classifier = classifier,
            num_frames=num_frames
        )
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
        vision_heads = vision_width // 64
        embed_dim = state_dict["text_projection"].shape[1]
        model = ModifiedResNet(
                layers=vision_layers,
                num_classes=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                classifier = classifier
        )
    if 'ViT' in model_name:
        model_name = model_name.replace('/' ,'-')
    checkpoint = torch.jit.load(f'{download_root}/{model_name}.pt', map_location='cpu')
    print(model.load_state_dict(checkpoint.visual.state_dict(), strict=False))
    return model