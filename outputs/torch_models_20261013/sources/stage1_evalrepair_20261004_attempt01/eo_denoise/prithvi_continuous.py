"""Spatial Prithvi EO 2.0 autoencoder: native continuous ViT features, no VQ.

Uses the pinned reference ViT and MAE decoder blocks. Images are 2D, all 256
patches remain present, and the encoder CLS travels with the patch features.
There is no codebook, latent projection to 32 dimensions, VQ loss, or KL loss.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .prithvi_reference import PrithviViT, MAEDecoder
from .prithvi_spatial import spatial_position


class PrithviContinuousCodec(nn.Module):
    def __init__(self, channels, c):
        super().__init__()
        if c['quantizer'] != 'ae' or c['latent_dim'] != c['eo_dim'] or c['grid'] != 16:
            raise ValueError('Continuous Prithvi keeps all native ViT patch features without quantization')
        self.channels, self.config = channels, c
        self.encoder = PrithviViT(img_size=256, patch_size=(1,16,16), num_frames=1,
            in_chans=channels+1, embed_dim=c['eo_dim'], depth=c['eo_depth'],
            num_heads=c['eo_heads'], mlp_ratio=4, coords_encoding=[], drop_path=0)
        original = self.encoder.patch_embed.proj
        spatial_patch = nn.Conv2d(channels+1, c['eo_dim'], 16, stride=16)
        with torch.no_grad():
            spatial_patch.weight.copy_(original.weight[:,:,0])
            spatial_patch.bias.copy_(original.bias)
        self.encoder.patch_embed.proj = spatial_patch
        self.encoder.pos_embed = spatial_position(c['eo_dim'])
        self.decoder = MAEDecoder(patch_size=(1,16,16), grid_size=(1,16,16),
            in_chans=channels, encoder_embed_dim=c['eo_dim'],
            decoder_embed_dim=c['eo_decoder_dim'], depth=c['eo_decoder_depth'],
            num_heads=c['eo_decoder_heads'], mlp_ratio=4, coords_encoding=[])
        self.decoder.decoder_pos_embed = spatial_position(c['eo_decoder_dim'])

    def _block(self, block, x):
        if self.training and self.config['activation_checkpointing']:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def encode(self, image, input_mask):
        if image.shape[-2:] != (256,256):
            raise ValueError('Frozen 256x256 inputs required')
        x = torch.cat((torch.where(input_mask.bool(), image, 0), input_mask.to(image.dtype)), 1)
        patches = self.encoder.patch_embed.proj(x).flatten(2).transpose(1,2)
        patches = patches + self.encoder.pos_embed[:,1:].to(patches.dtype)
        cls = self.encoder.cls_token.expand(x.shape[0],-1,-1).to(patches.dtype)
        tokens = torch.cat((cls, patches), 1)
        for block in self.encoder.blocks:
            tokens = self._block(block, tokens)
        return self.encoder.norm(tokens)

    def decode_latent(self, latent):
        if latent.shape[1:] != (257, self.config['eo_dim']):
            raise ValueError('Expected CLS plus 256 continuous patch features')
        tokens = self.decoder.decoder_embed(latent)
        tokens = tokens + self.decoder.decoder_pos_embed.to(tokens.dtype)
        for block in self.decoder.decoder_blocks:
            tokens = self._block(block, tokens)
        pixels = self.decoder.decoder_pred(self.decoder.decoder_norm(tokens))[:,1:]
        return pixels.reshape(latent.shape[0],16,16,16,16,self.channels).permute(0,5,1,3,2,4).reshape(latent.shape[0],self.channels,256,256)

    def forward(self, image, input_mask):
        latent = self.encode(image, input_mask)
        patch_grid = latent[:,1:].transpose(1,2).reshape(image.shape[0],-1,16,16)
        zero = latent.new_zeros(())
        return {'reconstruction':self.decode_latent(latent), 'continuous_latent':latent,
            # Compatibility field for common reconstruction/stability bookkeeping;
            # this is the unchanged continuous feature map, never a quantized value.
            'quantized':patch_grid, 'structure':patch_grid, 'indices':None,
            'bottleneck_loss':zero, 'kl':zero, 'log_probs':None}

    def decode_indices(self, indices):
        raise ValueError('Continuous Prithvi has no discrete indices or codebook')

    def rate(self):
        elements = 257*self.config['eo_dim']
        return {'kind':'continuous_float32_latent_payload', 'patch_tokens':256,
            'cls_tokens':1, 'feature_dim':self.config['eo_dim'], 'latent_elements':elements,
            'bits_per_image':elements*32, 'not_a_discrete_codec_rate':True,
            'rate_matched_to_var':False}
