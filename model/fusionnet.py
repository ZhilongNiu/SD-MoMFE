#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import math
import json
import re
from collections import OrderedDict
from typing import Optional, Tuple, Union
from thop import profile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torchvision import transforms
import cv2

import yaml

from model.DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from model.segnet import SegNeXt, SegAttention
from util.img_process import gray2rgb




class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, is_last: bool = False):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.is_last = is_last
        
        self.init_weight()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        
        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Conv3Layer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, is_last: bool = False):
        super(Conv3Layer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1,padding = 1)
        self.is_last = is_last
        self.init_weight()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2d(x)
        
        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)




class Conv1Layer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Conv1Layer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,padding=0)
        self.init_weight()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2d(x)
        out = F.leaky_relu(out, inplace=True)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class Conv_F(nn.Module):
    def __init__(self, in_chan: int = 1, out_chan: int = 64, ks: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.mid_chan = round(math.sqrt(in_chan * out_chan))
        self.conv1 = nn.Conv2d(in_chan, self.mid_chan, kernel_size=ks,
                              stride=stride, padding=padding, bias=True)
        self.bn1 = nn.GroupNorm(num_groups=int(self.mid_chan / 2), 
                               num_channels=self.mid_chan)
        self.conv2 = nn.Conv2d(self.mid_chan, out_chan, kernel_size=ks,
                              stride=stride, padding=padding, bias=True)
        self.bn2 = nn.GroupNorm(num_groups=int(out_chan / 2), 
                               num_channels=out_chan)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

class DenseConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, is_last: bool = False):
        super().__init__()
        self.is_last = is_last
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size,
                                  stride, is_last=self.is_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_conv(x)


class DenseFuse_net_change(nn.Module):
    def __init__(self, input_nc: int = 1):
        super().__init__()
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.denseblock1 = DenseConv2d(nb_filter[0], nb_filter[3], kernel_size, stride)
        self.denseblock2 = DenseConv2d(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.denseblock3 = DenseConv2d(nb_filter[2], nb_filter[3], kernel_size, stride)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, 
                                                   torch.Tensor, torch.Tensor]:
        x = self.conv1(input)
        x1 = self.denseblock1(x)
        x2 = torch.cat([x, x1], 1)
        x3 = self.denseblock2(x2)
        x4 = torch.cat([x1, x3], 1)
        out_1 = self.denseblock3(x4)
        out_2 = torch.cat([x3, out_1], 1)
        return out_1, out_2, x1, x4



class Sobelxy(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1,
                 stride: int = 1, dilation: int = 1, groups: int = 1):
        super().__init__()

        sobel_filter = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
        
  
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                              padding=padding, stride=stride, dilation=dilation,
                              groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        

        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                              padding=padding, stride=stride, dilation=dilation,
                              groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sobelx = self.convx(x)
        sobely = self.convy(x)
        return torch.abs(sobelx) + torch.abs(sobely)



    
class Expert(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, ks: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=ks,
                              stride=stride, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=ks,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        return x


class grad_Expert(nn.Module):

    def __init__(self, in_chan: int, out_chan: int, ks: int = 1, 
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.grad_conv = Sobelxy(channels=in_chan)
        self.bn = nn.GroupNorm(num_groups=int(out_chan / 2), num_channels=out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grad_conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x)
        return x


class MoMFE(nn.Module):
    def __init__(self, vis_chan: int, ir_chan: int, out_chan: int,
                 W: int, H: int, n_expert: int = 6, K: int = 4):
        super().__init__()
        self.n_expert = n_expert
        self.k = K
        

        self.w_gate = nn.Parameter(
            torch.Tensor(int(vis_chan*2*W*H/16/16), self.n_expert)
        )
        self.w_noise = nn.Parameter(
            torch.Tensor(int(vis_chan*2*W*H/16/16), self.n_expert)
        )

        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=3)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.gap = nn.AdaptiveAvgPool2d((int(H/16), int(W/16)))
        self.gmp = nn.AdaptiveMaxPool2d((int(H/16), int(W/16)))


        self.expert_layers_vis_h = Expert(vis_chan, out_chan)
        self.expert_layers_vis_l = Expert(vis_chan, out_chan)
        self.expert_layers_ir_h = Expert(vis_chan, out_chan)
        self.expert_layers_ir_l = Expert(vis_chan, out_chan)
        self.sobel_vis = Sobelxy(channels=vis_chan)
        self.sobel_ir = Sobelxy(channels=vis_chan)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.init_weight()

    def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
        return (gates > 0).sum(0)

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _prob_in_top_k(self, clean_values: torch.Tensor, 
                       noisy_values: torch.Tensor,
                       noise_stddev: torch.Tensor, 
                       noisy_top_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def forward(self, vis_h: torch.Tensor, vis_l: torch.Tensor,
                ir_h: torch.Tensor, ir_l: torch.Tensor,
                vis: torch.Tensor, ir: torch.Tensor,
                loss_coef: float = 1e-2,
                if_train: bool = True) -> torch.Tensor:

        self.if_train = if_train


        x_local = torch.cat((vis, ir), dim=1)
        x_local = self.gap(x_local) + self.gmp(x_local)
        s_local = self.flatten_layer(x_local)
        s_local = F.leaky_relu(s_local, inplace=True)


        clean_logits = s_local @ self.w_gate
        raw_noise_stddev = s_local @ self.w_noise
        noise_stddev = ((self.softplus(raw_noise_stddev) + loss_coef))
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits

        if self.if_train:
            top_logits, top_indices = logits.topk(
                min(self.n_expert + 1, self.n_expert), dim=1
            )
            top_k_logits = top_logits[:, :self.n_expert]
            top_k_indices = top_indices[:, :self.n_expert]
        else:
            top_logits, top_indices = logits.topk(
                min(self.k + 1, self.n_expert), dim=1
            )
            top_k_logits = top_logits[:, :self.k]
            top_k_indices = top_indices[:, :self.k]

        G_local = self.softmax(top_k_logits)

        E_net = []
        
        if self.if_train:
            E_net = E_net + [torch.unsqueeze(self.expert_layers_vis_h(vis_h), 0)]
            E_net = E_net + [torch.unsqueeze(self.expert_layers_vis_l(vis_l), 0)]
            E_net = E_net + [torch.unsqueeze(self.expert_layers_ir_h(ir_h), 0)]
            E_net = E_net + [torch.unsqueeze(self.expert_layers_ir_l(ir_l), 0)]
            E_net = E_net + [torch.unsqueeze(self.sobel_vis(vis), 0)]
            E_net = E_net + [torch.unsqueeze(self.sobel_ir(ir), 0)]
            E_net = torch.cat(E_net, dim=0)
            E_net = torch.transpose(E_net, 0, 1)

            y = []
            for bs in range(G_local.size(0)):
                g = G_local[bs]
                y_local = torch.zeros_like(E_net[0][0])
                for i in range(len(top_k_indices.tolist()[bs])):
                    y_local += g[i] * E_net[bs][top_k_indices.tolist()[bs][i]]
                y = y + [torch.unsqueeze(y_local, 0)]
        else:

            expert_layers = [
                self.expert_layers_vis_h,
                self.expert_layers_vis_l,
                self.expert_layers_ir_h,
                self.expert_layers_ir_l,
                self.sobel_vis,
                self.sobel_ir
            ]
            expert_inputs = [vis_h, vis_l, ir_h, ir_l, vis, ir]
            
            y = []
            for bs in range(G_local.size(0)):
                g = G_local[bs]
                y_local = None
                for i, expert_idx in enumerate(top_k_indices[bs]):
                    expert_output = expert_layers[expert_idx](expert_inputs[expert_idx][bs].unsqueeze(0))
                    if y_local is None:
                        y_local = g[i] * expert_output
                    else:
                        y_local += g[i] * expert_output
                y = y + [y_local]


                
        y= torch.cat(y, dim=0)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, G_local)
        importance = gates.sum(0)
        loss = self.cv_squared(importance)
        loss *= loss_coef
        return y, loss

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

class Decoder_Unet_att(nn.Module):
    def __init__(self, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        nb_filter = [80, 64, 32, 16, 8]
        
        self.conv1_1 = Conv1Layer(nb_filter[0], nb_filter[1])
        self.conv3_1 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv1_2 = Conv1Layer(nb_filter[1], nb_filter[1])
        self.conv3_2 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv3_3 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv3_4 = ConvLayer(nb_filter[3], nb_filter[4], kernel_size, stride)
        self.conv3_5 = ConvLayer(nb_filter[4], 1, kernel_size, stride, is_last=True)
        
        self.init_weight()

    def forward(self, 
                x: torch.Tensor, 
                vis_1: torch.Tensor, 
                vis_3: torch.Tensor,
                ir_1: torch.Tensor, 
                ir_3: torch.Tensor, 
                ca: Optional[torch.Tensor] = None,
                sa: Optional[torch.Tensor] = None, 
                if_seg: bool = True
                ) -> torch.Tensor:
        x = self.conv1_1(x)
        x1 = self.conv3_1(x)
        
        if if_seg:
            x = ca * x1
            x = sa * x
            x = self.conv1_2(x)
            x = x + x1
        else:
            x = x1
            for param in self.conv1_2.parameters():
                param.requires_grad = False
  
        x = self.conv3_2(x)
        x = self.conv3_3(x + vis_3 + ir_3)
        x = self.conv3_4(x + vis_1 + ir_1)
        x = self.conv3_5(x)
        return x

    def init_weight(self):

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)







class SMFMNet(nn.Module):
    def __init__(self, num_classes: int, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
      
        self.dense_vis = DenseFuse_net_change(input_nc=1)
        self.dense_ir = DenseFuse_net_change(input_nc=1)
        

        self.dwtb_vis = DWT_2D('haar')
        self.dwtb_ir = DWT_2D('haar')
        self.conv_vis_h = Conv_F(in_chan=1, out_chan=16)
        self.conv_vis_l = Conv_F(in_chan=1, out_chan=16)
        self.conv_ir_h = Conv_F(in_chan=1, out_chan=16)
        self.conv_ir_l = Conv_F(in_chan=1, out_chan=16)
        
        self.moe = MoMFE(vis_chan=16, ir_chan=16, out_chan=16,
                                      W=640, H=480)

        self.segatt = SegAttention()
        self.decoder_fusion = Decoder_Unet_att()
        self.up_f = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
   
    def forward(self, 
                vis: torch.Tensor, 
                ir: torch.Tensor,
                mid_feature: torch.Tensor, 
                if_train: bool = True,
                if_seg: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = vis.size()[2:]
        vis = vis[:, 0:1, :, :]

        
        vis_LL, vis_LH, vis_HL, vis_HH = self.dwtb_vis(vis)  # [B,1,320,240]
        ir_LL, ir_LH, ir_HL, ir_HH = self.dwtb_ir(ir)  # [B,1,320,240]
        
        vis_H = F.interpolate(vis_LH + vis_HL + vis_HH, (H, W),
                            mode='bilinear', align_corners=True)  # [B,1,640,480]
        vis_L = F.interpolate(vis_LL, (H, W),
                            mode='bilinear', align_corners=True)  # [B,1,640,480]
        ir_H = F.interpolate(ir_LH + ir_HL + ir_HH, (H, W),
                           mode='bilinear', align_corners=True)  # [B,1,640,480]
        ir_L = F.interpolate(ir_LL, (H, W),
                           mode='bilinear', align_corners=True)  # [B,1,640,480]
        

        vis_H = self.conv_vis_h(vis_H)  # [B,16,640,480]
        vis_L = self.conv_vis_l(vis_L)  # [B,16,640,480]
        ir_H = self.conv_ir_h(ir_H)     # [B,16,640,480]
        ir_L = self.conv_ir_l(ir_L)     # [B,16,640,480]
        
        vis, vis_dense, vis_1, vis_3 = self.dense_vis(vis)  
        # vis[B,16,640,480], vis_dense[B,32,640,480], vis_1[B,16,640,480], vis_3[B,32,640,480]
        ir, ir_dense, ir_1, ir_3 = self.dense_ir(ir)
        # ir[B,16,640,480], ir_dense[B,32,640,480], ir_1[B,16,640,480], ir_3[B,32,640,480]
        
        if if_seg:
            ca_mask, sa_mask = self.segatt(mid_feature)  # ca_mask[B,64,1,1], sa_mask[B,1,640,480]
        else:
            for param in self.segatt.parameters():
                param.requires_grad = False
            ca_mask = sa_mask = None
        
        
        f, loss = self.moe(vis_H, vis_L, ir_H, ir_L, vis, ir, if_train=if_train)  # f[B,16,640,480]


        f = torch.cat((vis_dense, f, ir_dense), dim=1)  # [B,80,640,480]
        
        f = self.decoder_fusion(f, vis_1, vis_3, ir_1, ir_3,
                              ca=ca_mask, sa=sa_mask, if_seg=if_seg)  # [B,1,640,480]
        
        f = torch.tanh(f) / 2 + 0.5  # [B,1,640,480]
        
        return f, loss


