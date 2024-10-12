"""
This module contains VisionTransformer3D class for unimodal classification of
sMRI data.
"""

import torch

from lib.tools import get_num_patches
from torch import nn
from typing import Tuple, Union, Dict, Any, List

class PatchEmbedding3D(nn.Module):
    """
    Creates 3D patch embeddings from input volumetric data using 3D convolution.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3
        for RGB).
        patch_size (int): Size of the patches to divide the input image into.
        embed_dim (int): Dimensionality of the embedding space.
        img_size (int or tuple): Size of the input image (height, width, depth).
    """
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, 
                 img_size: Tuple[int, ...]):
        super(PatchEmbedding3D, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = get_num_patches(inp_dim = img_size,
                                         patch_size = patch_size)
        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, 
                              stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class MultiHeadSelfAttentionBlock(nn.Module):
    """
    Implements multi-head self-attention.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output
    
class MLPBlock(nn.Module):
    """
    Implements a simple Multi-Layer Perceptron (MLP) with two fully connected
    layers.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        hidden_dim (int): Dimensionality of the hidden layer.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, p: float = 0.):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return self.activation2(self.fc2(self.activation1(self.fc1(x))))
    
class TransformerEncoderBlock(nn.Module):
    """
    Implements a single transformer encoder block consisting of a self-attention
    block followed by an MLP block.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimensionality of the hidden layer in the MLP.
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, p_ratio: float = 0.):
        super(TransformerEncoderBlock, self).__init__()
        self.attn_block = MultiHeadSelfAttentionBlock(embed_dim, num_heads)
        self.mlp_block = MLPBlock(embed_dim, hidden_dim, p_ratio)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn_block(self.layer_norm1(x))
        x = x + self.mlp_block(self.layer_norm2(x))
        return x
    
class VisionTransformer3D(nn.Module):
    """
    Implements a 3D Vision Transformer model for processing volumetric data.

    Args:
        num_classes (int): Number of output classes for classification.
        image_size (int or tuple): Size of the input 3D image (depth, height,
        width).
        in_channels (int, optional): Number of input channels. Defaults to 1
        (grayscale).
        patch_size (int, optional): Size of the patches for patch embedding.
        Defaults to 8.
        embed_dim (int, optional): Dimensionality of the embeddings. Defaults to
        1024.
        num_heads (int, optional): Number of attention heads. Defaults to 4.
        num_layers (int, optional): Number of transformer layers. Defaults to 4.
        hidden_dim (int, optional): Dimensionality of the hidden layer in the
        MLP. Defaults to 128.
    """
    def __init__(self, num_classes: int, image_size: Tuple[int, ...], 
                 in_channels: int = 1, patch_size: int = 8,embed_dim: int = 1024, 
                 num_heads: int = 4, num_layers: int = 4, hidden_dim: int = 128,
                 p_ratio: float = 0.):
        super(VisionTransformer3D, self).__init__()
        self.patch_embedding = PatchEmbedding3D(in_channels, patch_size, 
                                                embed_dim, image_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, hidden_dim, p_ratio)
            for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embedding.n_patches, embed_dim)
            )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x