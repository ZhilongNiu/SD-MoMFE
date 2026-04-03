#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import argparse
import os
from collections import OrderedDict
from typing import Any, Union
import pprint

def modify_img_path(d: Union[dict, Any], new_path: str) -> Union[dict, Any]:
    """
    递归修改字典中的img_path值
    
    Args:
        d: 要修改的对象
        new_path: 新的路径
    """
    if isinstance(d, (dict, OrderedDict)):
        for key, value in d.items():
            if key == 'train_dataloader' and isinstance(value, dict):
                if 'img_path' in value:
                    value['img_path'] = new_path
            elif isinstance(value, (dict, OrderedDict)):
                d[key] = modify_img_path(value, new_path)
    return d

def remove_runtime_info(checkpoint_path: str, save_dir: str) -> None:
    """
    删除checkpoint中的runtime_info并保存到指定目录
    
    Args:
        checkpoint_path: 输入的checkpoint文件路径
        save_dir: 保存输出文件的目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取原文件名
    filename = os.path.basename(checkpoint_path)
    save_path = os.path.join(save_dir, filename)
    
    print(f"正在处理文件: {checkpoint_path}")
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 删除runtime_info
        if isinstance(checkpoint, dict) and 'runtime_info' in checkpoint:
            del checkpoint['runtime_info']
            print("已删除runtime_info")
        else:
            print("未找到runtime_info")
        
        # 保存修改后的文件
        torch.save(checkpoint, save_path)
        print(f"已保存到: {save_path}")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")

def print_dict_structure(d: Union[dict, Any], level: int = 0, max_level: int = None) -> None:
    """
    递归打印字典结构
    
    Args:
        d: 要打印的对象
        level: 当前递归层级
        max_level: 最大递归层级
    """
    indent = '  ' * level
    
    if max_level is not None and level >= max_level:
        print(f"{indent}...")
        return
        
    if isinstance(d, (dict, OrderedDict)):
        for key, value in d.items():
            if isinstance(value, (dict, OrderedDict)):
                print(f"{indent}{key}: dict(")
                print_dict_structure(value, level + 1, max_level)
                print(f"{indent})")
            elif isinstance(value, (list, tuple)):
                print(f"{indent}{key}: {type(value).__name__}[{len(value)}] (")
                if len(value) > 0:
                    print(f"{indent}  示例元素类型: {type(value[0]).__name__}")
                print(f"{indent})")
            elif isinstance(value, torch.Tensor):
                print(f"{indent}{key}: Tensor(shape={list(value.shape)}, dtype={value.dtype})")
            else:
                print(f"{indent}{key}: {type(value).__name__}({value})")
    else:
        print(f"{indent}非字典对象: {type(d).__name__}")
        if hasattr(d, '__dict__'):
            print(f"{indent}对象属性:")
            print_dict_structure(d.__dict__, level + 1, max_level)

def modify_pth_content(pth_path: str, save_path: str, new_img_path: str) -> None:
    """
    修改pth文件中的img_path并保存
    
    Args:
        pth_path: 输入pth文件的路径
        save_path: 保存修改后文件的路径
        new_img_path: 新的img_path值
    """
    print(f"\n正在加载文件: {pth_path}")
    
    try:
        # 加载文件
        content = torch.load(pth_path, map_location='cpu')
        
        print("\n原始文件内容结构:")
        print("=" * 80)
        print_dict_structure(content)
        print("=" * 80)
        
        # 修改img_path
        modified_content = modify_img_path(content, new_img_path)
        
        print("\n修改后的文件内容结构:")
        print("=" * 80)
        print_dict_structure(modified_content)
        print("=" * 80)
        
        # 保存修改后的文件
        torch.save(modified_content, save_path)
        print(f"\n已保存修改后的文件到: {save_path}")
            
    except Exception as e:
        print(f"操作失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='删除checkpoint中的runtime_info')
    parser.add_argument('checkpoint_path', type=str, help='输入的checkpoint文件路径')
    parser.add_argument('save_dir', type=str, help='保存输出文件的目录')
    
    args = parser.parse_args()
    remove_runtime_info(args.checkpoint_path, args.save_dir)

if __name__ == '__main__':
    main() 