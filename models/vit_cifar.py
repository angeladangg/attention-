# models/vit_cifar.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    """
    Splits a 32×32 image into (patch_size × patch_size) patches,
    flattens each patch, and projects to a vector of size embed_dim.
    For CIFAR-10, img_size=32, patch_size=4 → num_patches = (32/4)**2 = 64.
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # A single conv layer with kernel=patch_size, stride=patch_size, no padding,
        # so it “chunks” the image into non-overlapping patches and projects them.
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        # After proj: [B, embed_dim, 8, 8] if patch_size=4 → 8×8 = 64 patches
        x = self.proj(x)  
        # flatten: [B, embed_dim, 8, 8] → [B, embed_dim, 64]
        x = x.flatten(2)
        # transpose: [B, embed_dim, 64] → [B, 64, embed_dim]
        x = x.transpose(1, 2)
        return x  # shape [B, num_patches, embed_dim]


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention:
    - Takes input of shape [B, N, dim]
    - Projects to Q, K, V (each [B, N, dim])
    - Reshapes to [B, num_heads, N, head_dim]
    - Computes attention per head, concatenates, and linearly projects back.
    """
    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # One linear that outputs 3× dim (for Q, K, V)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        # x: [B, N, dim]
        B, N, C = x.shape

        # Compute Q, K, V in one go
        # out: [B, N, 3*dim] → split into 3 tensors each [B, N, dim]
        qkv = self.qkv(x)  
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  
        # Now qkv: [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  
        # Each of q,k,v: [B, num_heads, N, head_dim]

        # Compute scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  
        # attn_scores: [B, num_heads, N, N]
        attn_probs = attn_scores.softmax(dim=-1)  
        attn_probs = self.attn_drop(attn_probs)

        out = attn_probs @ v  
        # out: [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)  
        # out: [B, N, dim] after concatenating all heads

        out = self.proj(out)
        out = self.proj_drop(out)
        return out,(q, k) # optionally return q and k (queries and keys)


class MLP(nn.Module):
    """
    A simple MLP with one hidden layer: (dim → hidden_dim → dim)
    Uses GELU nonlinearity and dropout.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    One Transformer encoder block for ViT:
    ┌────────────────────────────────────┐
    │  x → LayerNorm → MSA → + (residual) │
    │   → LayerNorm → MLP → + (residual)  │
    └────────────────────────────────────┘
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, 
                 attn_dropout=0.0, proj_dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim, 
            num_heads=num_heads, 
            attn_dropout=attn_dropout, 
            proj_dropout=proj_dropout
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, 
                       hidden_features=hidden_dim, 
                       dropout=proj_dropout)

    def forward(self, x):
        # Pre‑attn LayerNorm
        x_res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x) + x_res

        # Pre‑MLP LayerNorm
        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x) + x_res
        return x


class VisionTransformer(nn.Module):
    """
    A small ViT for CIFAR‑10. 
    - img_size: 32
    - patch_size: 4 → num_patches = 8×8 = 64
    - embed_dim: 128 (the embedding dimension of each patch)
    - depth: number of Transformer blocks
    - num_heads: number of attention heads per block
    - mlp_ratio: hidden dim in MLP is embed_dim * mlp_ratio
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=128,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
        drop_path_rate=0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=proj_dropout)

        # Build Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth rates
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize patch_embed.proj to truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Linear layers in qkv and proj: use xavier
        def _init_module(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
        self.apply(_init_module)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        B = x.size(0)
        x = self.patch_embed(x)  
        # x: [B, num_patches, embed_dim] = [B, 64, 128]

        # Expand and prepend the CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)           # [B, 65, embed_dim]
        x = x + self.pos_embed                          # add positional embeddings
        x = self.pos_drop(x)

        # Pass through Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)        # [B, 65, embed_dim]
        cls_final = x[:, 0]     # take only the CLS token → [B, embed_dim]
        logits = self.head(cls_final)  # [B, num_classes]
        return logits


# A small helper for Stochastic Depth (DropPath)
# Source: timm (https://github.com/rwightman/pytorch-image-models)
class DropPath(nn.Module):
    """
    DropPath implements stochastic depth:
    During training, each sample in the batch randomly drops
    its residual branch with probability drop_prob.
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample when training."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape = [batch_size, 1, 1, ..., 1] to broadcast over all dims except batch
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize to {0,1}
    output = x.div(keep_prob) * random_tensor
    return output
