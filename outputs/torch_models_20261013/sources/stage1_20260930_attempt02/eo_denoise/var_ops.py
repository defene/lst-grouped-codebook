"""Deterministic bicubic resize for VAR's small latent grids (align_corners=False).

Separable linear maps implement PyTorch's cubic kernel a=-0.75 and edge clamp.
Forward and backward equivalence to the reference interpolate are tested. Area
selection only sees detached residuals and retains the original implementation.
"""
from functools import lru_cache
import torch
import torch.nn.functional as F


@lru_cache(maxsize=64)
def _weights(source, target):
    u = (torch.arange(target,dtype=torch.float64)+.5)*(source/target)-.5
    base = u.floor().long()
    indices = base[:,None]+torch.arange(-1,3)[None]
    d = (u[:,None]-indices).abs()
    a=-.75
    w=torch.where(d<=1,((a+2)*d-(a+3))*d*d+1,
                  torch.where(d<2,((a*d-5*a)*d+8*a)*d-4*a,0))
    matrix=torch.zeros(target,source,dtype=torch.float64)
    matrix.scatter_add_(1,indices.clamp(0,source-1),w)
    return matrix.float()


def interpolate(x, size, mode):
    if mode != 'bicubic':
        return F.interpolate(x,size=size,mode=mode)
    h,w=size
    if x.shape[-2:] == (h,w):return x
    wy=_weights(x.shape[-2],h).to(device=x.device,dtype=x.dtype)
    wx=_weights(x.shape[-1],w).to(device=x.device,dtype=x.dtype)
    return torch.matmul(torch.matmul(wy,x),wx.T)
