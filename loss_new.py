#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.img_process import *
from math import exp
import numpy as np
import kornia


from DWT_IDWT.DWT_IDWT_layer import DWT_2D,IDWT_2D

torch.cuda.device_count()
device = torch.device("cuda:3")

class SSIMloss(nn.Module):
    def __init__(self):
        super(SSIMloss, self).__init__()
    
    def forward(self,image_vis,image_ir,image_f,logits_fusion,label):
        f_y = RGB2YCrCb(image_f)[:, 0:1, :, :]


        vis = kornia.losses.ssim_loss(image_vis ,image_f ,3,reduction='mean')
        ir = kornia.losses.ssim_loss(logits_fusion ,f_y ,1,reduction='mean')

        
        return (vis + ir) / 2

class Gradloss(nn.Module):
    def __init__(self):
        super(Gradloss, self).__init__()
        self.sobelconv=Sobelxy()
    
    def forward(self,image_vis,image_ir,image_f):
        N, C, H, W = image_vis.size()
        vis_grad=self.sobelconv(image_vis)
        ir_grad=self.sobelconv(image_ir)
        f_grad=self.sobelconv(image_f)
        x_grad=torch.max(vis_grad,ir_grad)
        loss_grad = F.l1_loss(x_grad,f_grad)
        return loss_grad

class Pixelloss(nn.Module):
    def __init__(self):
        super(Pixelloss, self).__init__()
    def forward(self,image_f,image_vis,image_ir,labels):
        image_f = RGB2YCrCb(image_f)[:, 0:1, :, :]
        image_vis = RGB2YCrCb(image_vis)[:, 0:1, :, :]
        x_max = torch.max(image_vis,image_ir)
        concatenated_tensor = torch.cat([image_vis, image_ir])
        x_mean = torch.mean(concatenated_tensor)
        # reversed_labels = 1-labels
        loss = F.l1_loss(image_f,x_max)



        return loss

        
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        self.kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        self.kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        
    def forward(self,x):
        device = x.device
        self.weightx = nn.Parameter(data=self.kernelx, requires_grad=False).cuda(device)
        self.weighty = nn.Parameter(data=self.kernely, requires_grad=False).cuda(device)
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)