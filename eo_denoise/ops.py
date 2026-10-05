"""Integer-grid operations with deterministic CUDA backward paths."""
import torch.nn.functional as F


def pool_grid(x, size):
    height, width = (size, size) if isinstance(size, int) else size
    h, w = x.shape[-2:]
    if h % height or w % width or height > h or width > w:
        raise ValueError('Grid must evenly divide the source dimensions')
    return F.avg_pool2d(x, (h//height, w//width), stride=(h//height, w//width))


def resize_grid(x, size):
    h, w = x.shape[-2:]
    if (h, w) == (size, size):
        return x
    if size < h and size < w:
        return pool_grid(x, size)
    if size % h or size % w:
        raise ValueError('Only exact integer grid scaling is supported')
    return x.repeat_interleave(size//h, -2).repeat_interleave(size//w, -1)
