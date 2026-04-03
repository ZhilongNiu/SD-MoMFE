import json
import math
import os
from abc import *
from collections import OrderedDict
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import util.bricks_seg as bricks
# import utils

torch.cuda.device_count()
device = torch.device("cuda:1")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class StemConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, 
                 norm_layer: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            bricks.DownSampling(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                norm_layer=norm_layer
            ),
            bricks.DownSampling(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                norm_layer=norm_layer
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class MSCA(nn.Module):
    """多尺度通道注意力模块(Multi-Scale Channel Attention)
    
    该模块通过不同尺度的深度可分离卷积来捕获多尺度特征，并实现通道注意力机制。
    
    Args:
        in_channels (int): 输入通道数
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        
        self.conv5 = bricks.DepthwiseConv(
            in_channels=in_channels,
            kernel_size=(5, 5),
            padding=(2, 2),
            bias=True
        )
        self.conv7 = nn.Sequential(
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                bias=True
            ),
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                bias=True
            )
        )

        self.conv11 = nn.Sequential(
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 11),
                padding=(0, 5),
                bias=True
            ),
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(11, 1),
                padding=(5, 0),
                bias=True
            )
        )
        
        self.conv21 = nn.Sequential(
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 21),
                padding=(0, 10),
                bias=True
            ),
            bricks.DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(21, 1),
                padding=(10, 0),
                bias=True
            )
        )

        self.fusion = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x
        
        out = self.conv5(x)
        
        out7 = self.conv7(out)
        out11 = self.conv11(out)
        out21 = self.conv21(out)
        
        out = self.fusion(out + out7 + out11 + out21)
        
        return out * identity


class Attention(nn.Module):

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )
        self.msca = MSCA(in_channels=in_channels)
        self.fc2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = F.gelu(self.fc1(x))
        out = self.msca(out)
        return self.fc2(out)


class FFN(nn.Module):

    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, drop_prob: float = 0.) -> None:
        super().__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=(1, 1)
        )
        self.dw = bricks.DepthwiseConv(
            in_channels=hidden_features,
            kernel_size=(3, 3),
            bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=(1, 1)
        )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.fc1(x)
        out = F.gelu(self.dw(out))
        out = self.fc2(out)
        return self.dropout(out)

class Block(nn.Module):

    def __init__(self, in_channels: int, expand_ratio: float, 
                 drop_prob: float = 0., drop_path_prob: float = 0.) -> None:
        super().__init__()

        # 标准化层和注意力模块
        self.norm1 = nn.BatchNorm2d(num_features=in_channels)
        self.attention = Attention(in_channels=in_channels)
        self.drop_path = bricks.DropPath(
            drop_prob=drop_path_prob if drop_path_prob >= 0 else nn.Identity
        )
        
        # FFN部分
        self.norm2 = nn.BatchNorm2d(num_features=in_channels)
        self.ffn = FFN(
            in_features=in_channels,
            hidden_features=int(expand_ratio * in_channels),
            out_features=in_channels,
            drop_prob=drop_prob
        )

        # Layer Scale参数
        layer_scale_init_value = 1e-2
        self.layer_scale1 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )
        self.layer_scale2 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

       
        out = self.norm1(x)
        out = self.attention(out)
        out = x + self.drop_path(
            self.layer_scale1.unsqueeze(-1).unsqueeze(-1) * out
        )
        
        
        identity = out
        out = self.norm2(out)
        out = self.ffn(out)
        out = identity + self.drop_path(
            self.layer_scale2.unsqueeze(-1).unsqueeze(-1) * out
        )

        return out


class Stage(nn.Module):

    def __init__(
            self,
            stage_id: int,
            in_channels: int,
            out_channels: int,
            expand_ratio: float,
            blocks_num: int,
            drop_prob: float = 0.,
            drop_path_prob: List[float] = [0.]
    ) -> None:
        super().__init__()
        assert blocks_num == len(drop_path_prob), \
            f"blocks_num ({blocks_num}) must match length of drop_path_prob ({len(drop_path_prob)})"
        
        # 下采样层
        if stage_id == 0:
            self.down_sampling = StemConv(
                in_channels=in_channels,
                out_channels=out_channels
            )
        else:
            self.down_sampling = bricks.DownSampling(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2)
            )

        # Block模块序列
        self.blocks = nn.Sequential(*[
            Block(
                in_channels=out_channels,
                expand_ratio=expand_ratio,
                drop_prob=drop_prob,
                drop_path_prob=drop_path_prob[i]
            ) for i in range(blocks_num)
        ])

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_sampling(x)
        out = self.blocks(out)
        
        batch_size, channels, height, width = out.shape
        out = out.view(batch_size, channels, -1)
        out = torch.transpose(out, -2, -1).contiguous()
        out = self.norm(out)
        out = torch.transpose(out, -2, -1).contiguous()
        out = out.view(batch_size, -1, height, width)

        return out


class MSCAN(nn.Module):
    def __init__(
            self,
            embed_dims: List[int] = [3, 32, 64, 160, 256],
            expand_ratios: List[float] = [8, 8, 4, 4],
            depths: List[int] = [3, 3, 5, 2],
            drop_prob: float = 0.1,
            drop_path_prob: float = 0.1
    ) -> None:
        super().__init__()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, sum(depths))]
        
        self.stages = nn.Sequential(*[
            Stage(
                stage_id=stage_id,
                in_channels=embed_dims[stage_id],
                out_channels=embed_dims[stage_id + 1],
                expand_ratio=expand_ratios[stage_id],
                blocks_num=depths[stage_id],
                drop_prob=drop_prob,
                drop_path_prob=dpr[sum(depths[: stage_id]): sum(depths[: stage_id + 1])]
            ) for stage_id in range(len(depths))
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = x
        outputs = []
        
        for idx, stage in enumerate(self.stages):
            out = stage(out)
            if idx != 0:  # 跳过第一个stage的输出
                outputs.append(out)

        return [x, *outputs]

class Hamburger(nn.Module):

    def __init__(
            self,
            hamburger_channels: int = 256,
            nmf2d_config: str = json.dumps({
                "SPATIAL": True,
                "MD_S": 1,
                "MD_D": 512,
                "MD_R": 64,
                "TRAIN_STEPS": 6,
                "EVAL_STEPS": 7,
                "INV_T": 1,
                "ETA": 0.9,
                "RAND_INIT": True,
                "return_bases": False,
                "device": "cuda"
            })
    ) -> None:
        super().__init__()
        
        # 输入转换
        self.ham_in = nn.Sequential(
            nn.Conv2d(
                in_channels=hamburger_channels,
                out_channels=hamburger_channels,
                kernel_size=(1, 1)
            )
        )

        # NMF2D模块
        self.ham = bricks.NMF2D(args=nmf2d_config)

        # 输出转换
        self.ham_out = nn.Sequential(
            nn.Conv2d(
                in_channels=hamburger_channels,
                out_channels=hamburger_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=hamburger_channels
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.ham_in(x)
        out = self.ham(out)
        out = self.ham_out(out)
        return F.relu(identity + out)


class LightHamHead(nn.Module):

    def __init__(
            self,
            in_channels_list: List[int] = [64, 160, 256],
            hidden_channels: int = 256,
            out_channels: int = 256,
            classes_num: int = 9,
            drop_prob: float = 0.1,
            nmf2d_config: str = json.dumps({
                "SPATIAL": True,
                "MD_S": 1,
                "MD_D": 512,
                "MD_R": 64,
                "TRAIN_STEPS": 6,
                "EVAL_STEPS": 7,
                "INV_T": 1,
                "ETA": 0.9,
                "RAND_INIT": True,
                "return_bases": False,
                "device": "cuda"
            })
    ) -> None:
        super().__init__()

        # 分类头
        self.cls_seg = nn.Sequential(
            nn.Dropout2d(drop_prob),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=classes_num,
                kernel_size=(1, 1)
            )
        )

        # 特征压缩
        self.squeeze = nn.Sequential(
            nn.Conv2d(
                in_channels=sum(in_channels_list),
                out_channels=hidden_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=hidden_channels,
            ),
            nn.ReLU()
        )

        # Hamburger注意力
        self.hamburger = Hamburger(
            hamburger_channels=hidden_channels,
            nmf2d_config=nmf2d_config
        )

        # 特征对齐
        self.align = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=out_channels
            ),
            nn.ReLU()
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) >= 2, "需要至少2个输入特征"
        
        # 获取原始输入和标准尺寸
        original = inputs[0]
        batch_size, _, standard_height, standard_width = inputs[1].shape
        standard_shape = (standard_height, standard_width)

        # 将所有特征调整到标准尺寸
        resized_features = [
            F.interpolate(
                input=x,
                size=standard_shape,
                mode="bilinear",
                align_corners=False
            )
            for x in inputs[1:]
        ]

        x = torch.cat(resized_features, dim=1)
        out = self.squeeze(x)
        out = self.hamburger(out)
        out = self.align(out)
        out = self.cls_seg(out)

        # 调整输出尺寸
        original_height, original_width = original.shape[2:]
        out = F.interpolate(
            input=out,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False
        )

        # 重排输出维度
        return torch.transpose(
            out.view(batch_size, -1, original_height * original_width), 
            -2, -1
        ).contiguous()


class SegNeXt(nn.Module):
    def __init__(
            self,
            embed_dims: List[int] = [3, 32, 64, 160, 256],
            expand_rations: List[float] = [8, 8, 4, 4],
            depths: List[int] = [3, 3, 5, 2],
            drop_prob_of_encoder: float = 0.1,
            drop_path_prob: float = 0.1,
            hidden_channels: int = 256,
            out_channels: int = 256,
            classes_num: int = 9,
            drop_prob_of_decoder: float = 0.1,
            nmf2d_config: str = json.dumps({
                "SPATIAL": True,
                "MD_S": 1,
                "MD_D": 512,
                "MD_R": 64,
                "TRAIN_STEPS": 6,
                "EVAL_STEPS": 7,
                "INV_T": 1,
                "ETA": 0.9,
                "RAND_INIT": False,
                "return_bases": False,
                "device": "cuda"
            })
    ) -> None:
        super().__init__()

        # 编码器
        self.encoder = MSCAN(
            embed_dims=embed_dims,
            expand_ratios=expand_rations,
            depths=depths,
            drop_prob=drop_prob_of_encoder,
            drop_path_prob=drop_path_prob
        )

        # 解码器
        self.decoder = LightHamHead(
            in_channels_list=embed_dims[-3:],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            classes_num=classes_num,
            drop_prob=drop_prob_of_decoder,
            nmf2d_config=nmf2d_config
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)

def load_seg_pth(weight_path: str, model: nn.Module, local_rank: torch.device) -> None:

    ckpt = torch.load(weight_path)
    net = model.to(local_rank)
    module_lst = [i for i in net.state_dict()]
    weights = OrderedDict()
    
    for idx, (k, v) in enumerate(ckpt['state_dict'].items()):
        if net.state_dict()[module_lst[idx]].size() == v.size():
            weights[module_lst[idx]] = v
        else:
            print(f"参数大小不匹配: {k} - {v.size()} vs {module_lst[idx]} - {net.state_dict()[module_lst[idx]].size()}")
    
    net.load_state_dict(weights, strict=False)


class ConvBNLReLu(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, ks: int = 1, 
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        return F.leaky_relu(x)

    def init_weight(self) -> None:
        """初始化网络权重。"""
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class ChannelAttention(nn.Module):

    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):

    def __init__(self, size: Tuple[int, int] = (480, 640), 
                 kernel_size: int = 7) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class SegAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        channel = [64, 128, 160, 256]
        
        # 特征转换层
        self.conv1 = ConvBNLReLu(in_chan=channel[2], out_chan=channel[1])
        self.conv2 = ConvBNLReLu(in_chan=channel[3], out_chan=channel[1])
        self.conv3 = ConvBNLReLu(in_chan=channel[1], out_chan=channel[0])
        
        # 上采样层
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        # 注意力模块
        self.CA = ChannelAttention(channel[0])
        self.SA = SpatialAttention()

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 特征融合
        out = self.conv2(x[3])
        out = self.conv1(x[2]) + self.up1(out)
        out = self.conv3(out)
        out = x[1] + self.up1(out)
        
        # 计算注意力权重
        ca_mask = self.CA(out)
        out = self.up2(out)
        sa_mask = self.SA(out)
        
        return ca_mask, sa_mask


if __name__ == "__main__":
    device = torch.device("cuda:3")
    net = SegNeXt().to(device)
    net.eval()
    
    batch_size = 1
    vis = torch.randn(batch_size, 3, 480, 640).to(device)
    x = net(vis)
    
    seg = SegAttention().to(device)
    ca_mask, sa_mask = seg(x)
    print(f"通道注意力形状: {ca_mask.shape}, 空间注意力形状: {sa_mask.shape}")