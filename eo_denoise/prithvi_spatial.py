"""Spatial-only, configurable-size adapter of the official Prithvi-EO-2.0 MAE.

Uses the pinned IBM reference encoder/decoder and timm Blocks. Changes: Conv2d,
2D positions, no temporal/location metadata, and a common VQ bottleneck instead
of MAE token dropping. The encoder CLS state cannot bypass the discrete codes.
Reference revision: 9eb1b1102806593963daa333bcc491b1c6f8562f (Apache-2.0).
"""
import numpy as np
import torch
from torch import nn
from .prithvi_reference import PrithviViT, MAEDecoder, get_1d_sincos_pos_embed_from_grid


def spatial_position(dim, grid=16):
    x = get_1d_sincos_pos_embed_from_grid(dim // 2, np.arange(grid))
    y = get_1d_sincos_pos_embed_from_grid(dim // 2, np.arange(grid))
    positions = np.concatenate((np.tile(x, (grid, 1)), np.repeat(y, grid, axis=0)), axis=1)
    return torch.from_numpy(np.concatenate((np.zeros((1, dim)), positions), axis=0)).float()[None]


class PrithviSpatialBackbone(nn.Module):
    def __init__(self, channels, c):
        super().__init__()
        self.channels = channels
        self.encoder = PrithviViT(img_size=256, patch_size=(1, 16, 16), num_frames=1,
            in_chans=channels+1, embed_dim=c['eo_dim'], depth=c['eo_depth'],
            num_heads=c['eo_heads'], mlp_ratio=4, coords_encoding=[], drop_path=0)
        original = self.encoder.patch_embed.proj
        spatial_patch = nn.Conv2d(channels+1, c['eo_dim'], 16, stride=16)
        with torch.no_grad():
            spatial_patch.weight.copy_(original.weight[:, :, 0])
            spatial_patch.bias.copy_(original.bias)
        self.encoder.patch_embed.proj = spatial_patch
        self.encoder.pos_embed = spatial_position(c['eo_dim'])
        self.to_latent = nn.Linear(c['eo_dim'], c['latent_dim'])
        self.decoder = MAEDecoder(patch_size=(1, 16, 16), grid_size=(1, 16, 16),
            in_chans=channels, encoder_embed_dim=c['latent_dim'],
            decoder_embed_dim=c['eo_decoder_dim'], depth=c['eo_decoder_depth'],
            num_heads=c['eo_decoder_heads'], mlp_ratio=4, coords_encoding=[])
        self.decoder.decoder_pos_embed = spatial_position(c['eo_decoder_dim'])
        nn.init.xavier_uniform_(self.to_latent.weight)
        nn.init.zeros_(self.to_latent.bias)

    def encode(self, x):
        # Deliberately never call the reference's 5D/multitemporal forward path.
        tokens = self.encoder.patch_embed.proj(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.encoder.pos_embed[:, 1:].to(tokens.dtype)
        cls = self.encoder.cls_token.expand(x.shape[0], -1, -1).to(tokens.dtype)
        tokens = torch.cat((cls, tokens), dim=1)
        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)[:, 1:]
        return self.to_latent(tokens).transpose(1, 2).reshape(x.shape[0], -1, 16, 16)

    def decode(self, z):
        tokens = self.decoder.decoder_embed(z.flatten(2).transpose(1, 2))
        # Reference mask token becomes a learned constant decoder CLS token.
        # No sample-dependent continuous feature is passed around quantization.
        cls = self.decoder.mask_token.expand(z.shape[0], -1, -1).to(tokens.dtype)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.decoder.decoder_pos_embed.to(tokens.dtype)
        for block in self.decoder.decoder_blocks:
            tokens = block(tokens)
        pixels = self.decoder.decoder_pred(self.decoder.decoder_norm(tokens))[:, 1:]
        pixels = pixels.reshape(z.shape[0], 16, 16, 16, 16, self.channels)
        # Official patchify order is patch-row, patch-column, channels.
        return pixels.permute(0, 5, 1, 3, 2, 4).reshape(z.shape[0], self.channels, 256, 256)
