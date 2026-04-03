#!/usr/bin/python
# -*- encoding: utf-8 -*-
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

from model.fusionnet import SMFMNet
from model.segnet import SegNeXt
from dataset.eval_dataset import EvalDataset as FusionDataset
from util.img_process import RGB2YCrCb, YCrCb2RGB, save_img_single

class Config:
    
    def __init__(self):
        self.seed = 3074
        self.dataset_type = 'MSRS'
        self.num_classes = 9
        self.model_name = 'SDMoMFE'
        self.batch_size = 4
        self.num_workers = 8
        
        self.save_path = 'results/'
        
        self.segmodel_path = 'fusion_model.pth'
        self.fusion_model_path = 'seg_model.pth'
        

def setup_environment(config: Config) -> None:

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(config.seed)

def load_seg_model(weight_path: Path, model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    ckpt = torch.load(weight_path, map_location='cpu')
    net = model.to(local_rank)
    module_lst = [i for i in net.state_dict()]
    weights = OrderedDict()
    
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    for idx, (k, v) in enumerate(state_dict.items()):
        if net.state_dict()[module_lst[idx]].size() == v.size():
            weights[module_lst[idx]] = v
        elif local_rank == 0:
            print(f"Size mismatch at {module_lst[idx]}: Expected {net.state_dict()[module_lst[idx]].size()}, got {v.size()}")
            
    net.load_state_dict(weights, strict=False)
    return net

def test_fusion(config: Config, local_rank: int, test_mode: str = 'train', if_seg: bool = True) -> None:

    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    segmodel = load_seg_model(config.segmodel_path, SegNeXt(), local_rank)
    segmodel.eval()
    
    fusionmodel = SMFMNet(config.num_classes).to(local_rank)
    fusionmodel.load_state_dict(torch.load(config.fusion_model_path, map_location='cpu'))
    fusionmodel = DDP(fusionmodel, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    fusionmodel.eval()
    
    test_dataset = FusionDataset(split=test_mode, dataset_type=config.dataset_type)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=test_sampler
    )
    
    with torch.no_grad():
        for img_vis, img_ir, name in tqdm(test_loader):
            img_vis = img_vis.to(local_rank)
            img_ir = img_ir.to(local_rank)
            vi_YCrCb = RGB2YCrCb(img_vis).to(local_rank)
            
            fake_ir = img_ir.repeat(1, 3, 1, 1)
            fusion = fake_ir + img_vis
            mid_feature = segmodel(fusion)
            
            vi_Cb = vi_YCrCb[:, 1:2, :, :]
            vi_Cr = vi_YCrCb[:, 2:, :, :]
            
            logits_fusion,_ = fusionmodel(vi_YCrCb, img_ir, mid_feature, if_train=False, if_seg=True)
            fusion_ycrcb = torch.cat((logits_fusion, vi_Cb, vi_Cr), dim=1)
            
            fusion_rgb = YCrCb2RGB(fusion_ycrcb)
            fusion_result = torch.clamp(fusion_rgb, 0, 1)
            
            for k, img_name in enumerate(name):
                save_path = save_dir / img_name
                save_img_single(fusion_result[k], save_path)
                
    del fusionmodel, segmodel
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--local_rank", default=0, type=int, help='rank')
    args = parser.parse_args()
    
    config = Config()
    setup_environment(config)
    
    local_rank = args.local_rank
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    
    
    test_fusion(config, args.local_rank, test_mode='test', if_seg=True)