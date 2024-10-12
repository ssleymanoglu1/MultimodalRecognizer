"""
Dataset class for brain scans - sMRI data for classification of brain disease.
"""

import torch
import torch.nn.functional as F
import pandas as pd

from nilearn.image import load_img
from typing import Tuple, Union 
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from lib.tools import *


def img_transform(has_inp_ch: bool = True):
    """Implementation of callable transform function.

    Returns:
        callable: transform the input volume by adding channel dim.
    """
    transformation_queue = []
    if has_inp_ch:
        transformation_queue.append(transforms.Lambda(lambda x: x.unsqueeze(0)))
    return transforms.Compose(transformation_queue)

class GenMRIFusionDataset(Dataset):
   
    def __init__(self,
                 genMRIDataset: str,
                 mask_file: str,
                 imbalanced_flag = False,
                 transform = True,
                 has_input_channel = True,
                 task = 'Classification'):
        self.info_dataframe = pd.read_pickle(genMRIDataset)
        self.mask_img = mask_file
        self.imbalanced_weights = compute_class_weights(
            self.info_dataframe.Diagnosis) if imbalanced_flag else None
        self.transform = img_transform(
            has_inp_ch=has_input_channel) if transform else None
        self.task = task
        self.class_dict = to_index(list(self.info_dataframe.Diagnosis.unique()))

    def _load_img(self, sample_idx: int) -> torch.Tensor:  
        """Load and preprocess an fMRI image.

        Args:
            sample_idx (int): Index of a subject in dataset.

        Returns:
            torch.Tensor: 4D tensor of image data, (#channel_size, 3D space).
        """
        img_dir = self.info_dataframe.iloc[sample_idx].sMRIpath
        img = load_img(img_dir).get_fdata()
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)        
        return img.float()

    def _load_fnc(self, sample_idx: int) -> torch.Tensor:
        """Load FNC of each sample and convert to a tensor.

        Args:
            sample_idx (int): Index of a subject in dataset.

        Returns:
            torch.Tensor: FNC of relevent class.
        """
        fnc_data = self.info_dataframe.iloc[sample_idx].FNC
        if isinstance(fnc_data, np.ndarray):
            fnc_data = torch.from_numpy(fnc_data)
        return fnc_data.float().unsqueeze(0) # type: ignore
    
    def _load_snp(self, sample_idx: int) -> torch.Tensor:
        """Load SNP of each sample and convert to a tensor.

        Args:
            sample_idx (int): Index of a subject in dataset.

        Returns:
            torch.Tensor: SNP of relevent class.
        """
        snp_data = self.info_dataframe.iloc[sample_idx].SNP
        if isinstance(snp_data, np.ndarray):
            snp_data = torch.from_numpy(snp_data)
        return snp_data.float().unsqueeze(0) # type: ignore

    def __len__(self):
        return len(self.info_dataframe)

    def __getitem__(self, idx: int):
        sample = {'SubjectID': self.info_dataframe.iloc[idx].SubjectID,
                  'Age': self.info_dataframe.iloc[idx].Age,
                  'Gender': self.info_dataframe.iloc[idx].Gender,
                  'Diagnosis': float(self.class_dict[self.info_dataframe.iloc[idx].Diagnosis]),
                  'sMRI': self._load_img(idx) if self.task in ["ViT3D", "GenMRIFusion"] else torch.full((1,), float('nan')),
                  'FNC': self._load_fnc(idx) if self.task in ["ViT2D", "GenMRIFusion"] else torch.full((1,), float('nan')),
                  'SNP': self._load_snp(idx)  if self.task in ["GenomicEnc, GenMRIFusion"] else torch.full((1,), float('nan')),
                  }
        return sample
    