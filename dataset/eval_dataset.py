# coding:utf-8
from typing import Tuple, List, Dict, Union
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from PIL import Image
from natsort import natsorted

class EvalDataset(Dataset):
    
    VALID_SPLITS = {'train', 'test'}
    VALID_DATASET_TYPES = {'MSRS', 'FMB', 'MFNet'}
    
    def __init__(
        self, 
        split: str, 
        dataset_type: str, 
        ir_path: Union[str, Path, None] = None, 
        vi_path: Union[str, Path, None] = None
    ) -> None:

        super().__init__()
        
        assert split in self.VALID_SPLITS, \
            f'split必须是以下值之一: {", ".join(self.VALID_SPLITS)}'
            
        assert dataset_type in self.VALID_DATASET_TYPES, \
            f'dataset_type必须是以下值之一: {", ".join(self.VALID_DATASET_TYPES)}'
            
        self.split = split
        self._init_dataset_paths(dataset_type, split)
        self.filelist = self._get_sorted_filelist()
        self.length = len(self.filelist)

    def _init_dataset_paths(self, dataset_type: str, split: str) -> None:

        base_path = Path('/data/nzl/dataset/multimodel')
        
        dataset_configs: Dict = {
            'MSRS': {
                'train': {
                    'vis_dir': base_path / 'MSRS/train/vi',
                    'ir_dir': base_path / 'MSRS/train/ir',
                },
                'test': {
                    'vis_dir': base_path / 'MSRS/test/vi',
                    'ir_dir': base_path / 'MSRS/test/ir',
                }
            }
        }
        
        
        config = dataset_configs[dataset_type][split]
        
        self.vis_dir = config['vis_dir']
        self.ir_dir = config['ir_dir']

    def _get_sorted_filelist(self) -> List[str]:
        return natsorted(os.listdir(self.vis_dir))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:

        if not 0 <= index < self.length:
            raise IndexError(f"索引{index}超出范围[0, {self.length})")
            
        img_name = self.filelist[index]
        
        vis_path = self.vis_dir / img_name
        ir_path = self.ir_dir / img_name
        
        img_vis = self.imread(path=vis_path)
        img_ir = self.imread(path=ir_path, vis_flag=False)
        
        img_vis = TF.resize(img_vis, [480, 640])
        img_ir = TF.resize(img_ir, [480, 640])
        
        return img_vis, img_ir, img_name

    def __len__(self) -> int:
        return self.length
    
    @staticmethod
    def imread(path: Path, label: bool = False, vis_flag: bool = True) -> torch.Tensor:
        try:
            if not path.exists():
                raise FileNotFoundError(f"not found: {path}")
                
            if label:
                img = Image.open(path)
                return TF.to_tensor(img) * 255
            
            img = Image.open(path).convert('L' if not vis_flag else 'RGB')
            return TF.to_tensor(img)
            
        except Exception as e:
            raise RuntimeError(f"加载图像 {path} 时发生错误: {str(e)}")