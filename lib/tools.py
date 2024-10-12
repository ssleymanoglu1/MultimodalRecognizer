"""This is a module providing mandatory tools for training and data analysis
"""
import numpy as np
import operator
import torch

from torch import nn
from functools import reduce
from typing import Tuple, Union, Dict, Any, List


def get_num_patches(inp_dim: Tuple[int, ...], patch_size: int) -> int:
    """Computes total number of patches

    Args:
        inp_dim (Tuple[int, ...]): Size of input tensor
        patch_size (int): Size of each patch

    Returns:
        int: Total number of patches
    """
    return reduce(operator.mul, tuple(map(lambda x: x//patch_size, inp_dim)), 1)

def init_weights(self):
    """Initialize the weights using appropriate initialization methods."""
    for m in self.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
def compute_class_weights(x: Any) -> torch.Tensor:
    """Compute class weights in an imbalanced dataset

    Args:
        x (Any): Array-like object including classes 

    Returns:
        torch.Tensor: Class weight tensor
    """    
    _, class_count = np.unique(x, return_counts=True)
    weights = 1. / class_count
    return torch.tensor(weights, dtype=torch.float)

def to_index(class_labels: List) -> Dict:
    """Make a dict with keys: labels to value: index

    Args:
        class_labels (np.ndarray): list of class labels

    Returns:
        Dict: key, value : label, index
    """    
    class_labels = sorted(class_labels)
    label_index_dict = {label: index for index, label in enumerate(class_labels)}
    return label_index_dict

def fetch_list_of_backup_files(model_arch: str) -> tuple:
    """Defines a list of files to backup based on model architecture.

    Args:
        model_arch (str): Type of model architecture like ViT3D, ViT2D,
        GenomicEnc, and GenMRIFusion.

    Returns:
        tuple: Includes list of files to backup.
    """    
    if model_arch == 'ViT3D':
        file_names_ = ('ViT3D.yaml', 'ViT3D_recognition_e1.py')
    elif model_arch == 'ViT2D':
        file_names_ = ('-', '-')
    elif model_arch == 'GenomicEnc':
        file_names_ = ('-', '-')        
    else:
        file_names_ = ('-', '-')
    
    return file_names_