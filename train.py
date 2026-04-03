#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import torch
import numpy as np
import argparse
import datetime
import time
import logging
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.fusionnet import SMFMNet as SDMoMFE
from model.segnet import SegNeXt
from loss_new import *

from dataset.eval_dataset import EvalDataset as Fusion_dataset
from util.img_process import RGB2YCrCb, YCrCb2RGB, save_img_single
from natsort import natsorted
from tqdm import tqdm
from collections import OrderedDict

seed = 3074
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

dataset_type = 'MSRS'
exp_root = 'exp'
num_classes = 9
logpath = os.path.join('./logs/', dataset_type, exp_root)
lb_ignore = [255]
rootpth = os.path.join('./model', dataset_type, exp_root)
train_batchsize = 6
segmodel_pth = './seg_model.pth'
max_epoch = 100
alpha = 10
beta = 1
gamma = 1
lr_start = 5e-4
lr_decay = 0.7
momentum = 0.9
weight_decay = 1e-4
use_warmup = True
use_dynamic_hyperparameter = True
use_lr_decay = False
val_per_epoch = 10
warmup_epoch = 10
warmup_start_lr = 1e-5
co_num = 7


def load_seg_pth(weight_path, model, local_rank):
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

def train_fusion(batch_size, num, if_seg=True, num_classes=num_classes, logger=None, local_rank=-1):
    global alpha
    global beta

    fusionmodel = SDMoMFE(num_classes).to(local_rank)
    modelpth = rootpth
    fusionmodel.train()
            
    segmodel = SegNeXt().to(local_rank)
    path = segmodel_pth
    load_seg_pth(path, segmodel, local_rank)
    segmodel.eval()
    
    if num > 0:
        fusion_model_path = os.path.join(modelpth, 'final_model.pth')
        fusionmodel.load_state_dict(torch.load(fusion_model_path, map_location='cpu')['model_state_dict'])
    
    fusionmodel = DDP(fusionmodel, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train_dataset = Fusion_dataset(split='train', dataset_type=dataset_type, if_seg=if_seg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    if local_rank == 0:
        print(f"the training dataset is length:{train_dataset.length}")
        print(dataset_type)
        
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )
    train_loader.n_iter = len(train_loader)
        
    optimizer = torch.optim.AdamW(fusionmodel.parameters(), lr=lr_start, weight_decay=weight_decay, amsgrad=True)
    
    if num > 0:
        optimizer.load_state_dict(torch.load(fusion_model_path, map_location='cpu')['optimizer_state_dict'])

    iter_per_epoch = len(train_loader)
    
    pixelloss = Pixelloss()
    gradloss = Gradloss()
    ssimloss = SSIMloss()
    epoch = val_per_epoch
    st = glob_st = time.time()
    
    if local_rank == 0:
        logger.info('Training Fusion Model start~')

    for epo in range(0, epoch):
        if local_rank == 0:
            print(alpha)
            print(beta)
        
        if use_lr_decay:
            lr_this_epo = (lr_start * lr_decay ** (((epo) % 10) - 1))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_epo
        else:
            lr_this_epo = lr_start
        
        for it, (image_vis, image_ir, label, fusion, name) in enumerate(train_loader):
            if use_warmup and num == 0 and epo < warmup_epoch:
                warmup_factor = (lr_start / warmup_start_lr) ** (1. / (warmup_epoch * iter_per_epoch))
                lr_this_epo = warmup_start_lr * (warmup_factor ** (epo * iter_per_epoch + it))
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_epo
            
            train_loader.sampler.set_epoch(epoch)
            optimizer.zero_grad()
            
            image_vis = Variable(image_vis).to(local_rank)
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).to(local_rank)
            label = Variable(label).to(local_rank)
            fusion = Variable(fusion).to(local_rank)
            
            result_tensor = torch.clamp(label, 0, 1).to(local_rank)
            
            if if_seg:
                mid_feature = segmodel(fusion)
            else:
                fake_ir = image_ir.repeat(1, 3, 1, 1)
                fusion = fake_ir + image_vis
                mid_feature = segmodel(fusion)

            logits_fusion, loss_load = fusionmodel(image_vis, image_ir, mid_feature, if_train=True, if_seg=if_seg)
            
            fusion_ycrcb = torch.cat(
                (logits_fusion, image_vis_ycrcb[:, 1:2, :, :],
                image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )

            fusion_rgb = YCrCb2RGB(fusion_ycrcb).to(local_rank)
            
            ones = torch.ones_like(fusion_rgb)
            zeros = torch.zeros_like(fusion_rgb)
            fusion_rgb = torch.where(fusion_rgb > ones, ones, fusion_rgb)
            fusion_rgb = torch.where(fusion_rgb < zeros, zeros, fusion_rgb)
            
            vis_y = image_vis_ycrcb[:, 0:1, :, :]
            logits_fusion = RGB2YCrCb(fusion_rgb)[:, 0:1, :, :]
            
            loss_pixel_fus = pixelloss(fusion_rgb, image_vis, image_ir, result_tensor)
            loss_grad = gradloss(vis_y, image_ir, logits_fusion)
            loss_ssim = ssimloss(image_vis, image_ir, fusion_rgb, logits_fusion, result_tensor)

            loss_total = beta * loss_pixel_fus + alpha * loss_grad + loss_ssim + loss_load
            loss_total.backward()

            optimizer.step()
            
            if local_rank == 0:
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = it + 1

                eta = int((train_loader.n_iter - now_it + train_loader.n_iter * (max_epoch - num * epo - 1)) * t_intv / 10)
                eta = str(datetime.timedelta(seconds=eta))
                
                if now_it % 10 == 0:
                    msg = ', '.join([
                        'epoch: {epoch}',
                        'step: {it}/{max_it}',
                        'lr: {lr_this_epo:.6f}',
                        'loss_total: {loss_total:.4f}',
                        'loss_pixel: {loss_pixel:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_load: {loss_load:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]).format(
                        epoch=epo+1,
                        it=now_it,
                        lr_this_epo=lr_this_epo,
                        max_it=train_loader.n_iter,
                        loss_total=loss_total.item(),
                        loss_pixel=loss_pixel_fus.item(),
                        loss_grad=loss_grad.item(),
                        loss_load=loss_load,
                        loss_ssim=loss_ssim.item(),
                        time=t_intv,
                        eta=eta,
                    )
                    logger.info(msg)
                    st = ed
        
        if use_dynamic_hyperparameter:
            all_loss = loss_pixel_fus + loss_grad + loss_ssim + loss_load
            alpha = max(all_loss.item() / 3. / loss_grad.item() * gamma, 1.)
            beta = max(all_loss.item() / 3. / loss_pixel_fus.item(), 1.)
        
    if local_rank == 0:
        final_model_file = os.path.join(modelpth, 'epoch' + str((num + 1) * 10) + '.pth')
        last_model_file = os.path.join(modelpth, 'final_model.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': fusionmodel.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_total.item(),
        }, last_model_file)
        
        torch.save(fusionmodel.module.state_dict(), final_model_file)
        logger.info(f"Fusion Model Save to: {final_model_file}")
        logger.info('\n')

    del fusionmodel
    del segmodel
    torch.cuda.empty_cache()

def test_fusion(local_rank=-1, test_mode='train', if_seg=True, fusion_model_path=os.path.join(rootpth, 'final_model.pth')):
    path = segmodel_pth
    segmodel = SegNeXt().to(local_rank)
    load_seg_pth(path, segmodel, local_rank)
    segmodel.eval()
    
    fusionmodel = SDMoMFE(num_classes)
    fusionmodel.to(local_rank)
    fusionmodel.eval()
    
    fusionmodel.load_state_dict(torch.load(fusion_model_path, map_location='cpu')['model_state_dict'])
    fusionmodel = fusionmodel.to(local_rank)

    fusionmodel = DDP(fusionmodel, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    if local_rank == 0:
        print('fusionmodel load done!')
    
    if test_mode == 'train':
        save_dir = '/data/nzl/dataset/multimodel/MSRS/train/fusion'
    else:
        save_dir = '/data/nzl/dataset/multimodel/MSRS/test/DWT_test'
        
    test_dataset = Fusion_dataset(split=test_mode, dataset_type=dataset_type, if_seg=if_seg)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        sampler=test_sampler
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)

    with torch.no_grad():
        for it, (img_vis, img_ir, label, fusion, name) in enumerate(test_bar):
            img_vis = img_vis.to(local_rank)
            img_ir = img_ir.to(local_rank)
            vi_YCrCb = RGB2YCrCb(img_vis).to(local_rank)
            fusion = fusion.to(local_rank)

            fake_ir = img_ir.repeat(1, 3, 1, 1)
            fusion = fake_ir + img_vis
            mid_feature = segmodel(fusion)

            vi_Cb = vi_YCrCb[:, 1:2, :, :].to(local_rank)
            vi_Cr = vi_YCrCb[:, 2:, :, :].to(local_rank)
            
            logits_fusion = fusionmodel(vi_YCrCb, img_ir, mid_feature, if_train=False, if_seg=if_seg)
            
            fusion_ycrcb = torch.cat(
                (logits_fusion, vi_Cb, vi_Cr),
                dim=1,
            )

            fusion_rgb = YCrCb2RGB(fusion_ycrcb).to(local_rank)
            ones = torch.ones_like(fusion_rgb)
            zeros = torch.zeros_like(fusion_rgb)
            fusion_rgb = torch.where(fusion_rgb > ones, ones, fusion_rgb)
            fusion_rgb = torch.where(fusion_rgb < zeros, zeros, fusion_rgb)

            fusion_result = fusion_rgb

            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                save_img_single(fusion_result[k, ::], save_path)
                if local_rank == 0:
                    test_bar.set_description(f'Fusion {name[k]} Sucessfully!')

    del fusionmodel
    del segmodel
    torch.cuda.empty_cache()

def EN_function(image_array):
    histogram, bins = np.histogram(image_array, bins=256, range=(0, 255))
    histogram = histogram / float(np.sum(histogram))
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy

def AG_function(image):
    width = image.shape[1] - 1
    height = image.shape[0] - 1
    grady, gradx = np.gradient(image)
    s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
    AG = np.sum(np.sum(s)) / (width * height)
    return AG

def eval_image(logger):
    global gamma

    f_dir = os.path.join('/data/nzl/dataset/multimodel/MSRS/train/fusion')
    filelist = natsorted(os.listdir(f_dir))
    eval_bar = tqdm(filelist)
    
    EN_list = []
    AG_list = []
    
    for _, item in enumerate(eval_bar):
        f_name = os.path.join(f_dir, item)
        f_img = Image.open(f_name).convert('L')

        f_img_int = np.array(f_img).astype(np.int32)
        f_img_double = np.array(f_img).astype(np.float32)

        EN = EN_function(f_img_int)
        AG = AG_function(f_img_double)
        
        EN_list.append(EN)
        AG_list.append(AG)
    
    mean_en = np.mean(EN_list)
    mean_ag = np.mean(AG_list)
    
    logger.info("the mean of EN on training datasets is:")
    logger.info(mean_en)
    logger.info("the mean of ag on training datasets is:")
    logger.info(mean_ag)
    
    gamma = mean_en / mean_ag * co_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SD-MoMFE')
    parser.add_argument('--batch_size', '-B', type=int, default=16)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    
    local_rank = args.local_rank

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
    if not os.path.exists(logpath):
        os.makedirs(logpath, exist_ok=True)
    
    if not os.path.exists(rootpth):
        os.makedirs(rootpth, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(logpath, 'train.log'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    for i in range(0, round(max_epoch / 10)):
        if i == 0:
            if_seg = False
        else:
            if_seg = True
        
        train_fusion(batch_size=train_batchsize, num=i, if_seg=if_seg, logger=logger, local_rank=local_rank)
        dist.barrier()
        test_fusion(test_mode='train', if_seg=False, local_rank=local_rank)
        dist.barrier()
        
        if local_rank == 0:
            eval_image(logger=logger)
        dist.barrier()
