"""VAR's actual VQVAE with EO channel/mask adaptation and aligned Prithvi option."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .var_reference import VQVAE
from .prithvi_spatial import PrithviSpatialBackbone


class SpatialEncoder(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.encoder,self.to_latent=backbone.encoder,backbone.to_latent

    def forward(self,x):
        tokens=self.encoder.patch_embed.proj(x).flatten(2).transpose(1,2)
        tokens=tokens+self.encoder.pos_embed[:,1:].to(tokens.dtype)
        cls=self.encoder.cls_token.expand(x.shape[0],-1,-1).to(tokens.dtype)
        tokens=torch.cat((cls,tokens),1)
        for block in self.encoder.blocks:tokens=block(tokens)
        return self.to_latent(self.encoder.norm(tokens)[:,1:]).transpose(1,2).reshape(x.shape[0],-1,16,16)


class SpatialDecoder(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.decoder,self.channels=backbone.decoder,backbone.channels

    def forward(self,z):
        tokens=self.decoder.decoder_embed(z.flatten(2).transpose(1,2))
        cls=self.decoder.mask_token.expand(z.shape[0],-1,-1).to(tokens.dtype)
        tokens=torch.cat((cls,tokens),1)+self.decoder.decoder_pos_embed.to(tokens.dtype)
        for block in self.decoder.decoder_blocks:tokens=block(tokens)
        pixels=self.decoder.decoder_pred(self.decoder.decoder_norm(tokens))[:,1:]
        return pixels.reshape(z.shape[0],16,16,16,16,self.channels).permute(0,5,1,3,2,4).reshape(z.shape[0],self.channels,256,256)


class VarCodec(nn.Module):
    def __init__(self,channels,c):
        super().__init__()
        self.channels,self.config=channels,c
        self.scales=tuple(c['var_scales'])
        self.network=VQVAE(vocab_size=c['codebook_size'],z_channels=c['latent_dim'],ch=c['var_ch'],
            beta=c['commitment'],quant_conv_ks=3,quant_resi=c['var_quant_resi'],
            share_quant_resi=c['var_share_quant_resi'],v_patch_nums=self.scales,test_mode=False)
        if c['backbone']=='var_conv':
            self.network.encoder.conv_in=nn.Conv2d(channels+1,c['var_ch'],3,padding=1)
            self.network.encoder.in_channels=channels+1
            self.network.decoder.conv_out=nn.Conv2d(c['var_ch'],channels,3,padding=1)
            self.network.decoder.in_channels=channels
        elif c['backbone']=='var_prithvi':
            spatial=PrithviSpatialBackbone(channels,c)
            self.network.encoder=SpatialEncoder(spatial)
            self.network.decoder=SpatialDecoder(spatial)
        else:raise ValueError('Unsupported VAR backbone')
        # Both architectures retain exactly the VAR quantizer and its pre/post conv.
        self.network.quantize.size=self.network.quantize.vocab_size
        self.projection=nn.Sequential(nn.Conv2d(c['latent_dim'],32,1),nn.SiLU(),nn.Conv2d(32,16,1))

    @property
    def quantizer(self):return self.network.quantize

    def _block(self,module,x):
        if self.training and self.config['activation_checkpointing']:
            return checkpoint(module,x,use_reentrant=False)
        return module(x)

    def forward(self,image,input_mask):
        if image.shape[-2:] != (256,256):raise ValueError('Frozen 256x256 inputs required')
        x=torch.cat((torch.where(input_mask.bool(),image,0),input_mask.to(image.dtype)),1)
        z=self.network.quant_conv(self._block(self.network.encoder,x))
        # Preserve reference float32 residual quantization under BF16 training.
        with torch.autocast(device_type=z.device.type,enabled=False):
            ids_by_scale=self.quantizer.f_to_idxBl_or_fhat(z.float(),False)
            if self.training:
                quantized,_,vq_loss=self.quantizer(z.float(),False)
            else:
                quantized=self._decode_tokens(ids_by_scale)
                vq_loss=z.float().new_zeros(())
        prediction=self._block(self.network.decoder,self.network.post_quant_conv(quantized))
        return {'reconstruction':prediction,'quantized':quantized,'bottleneck_loss':vq_loss,
                'kl':vq_loss.new_zeros(()),'log_probs':None,
                'indices':torch.cat(ids_by_scale,1)[:,None],
                'indices_by_scale':[a.reshape(image.shape[0],s,s) for a,s in zip(ids_by_scale,self.scales)],
                'structure':self.projection(quantized)}

    def _decode_tokens(self,ids_by_scale):
        embeddings=[self.quantizer.embedding(a.long()).transpose(1,2).reshape(a.shape[0],self.config['latent_dim'],s,s)
                    for a,s in zip(ids_by_scale,self.scales)]
        return self.quantizer.embed_to_fhat(embeddings,all_to_max_scale=True,last_one=True)

    def decode_indices(self,indices):
        flat=indices.reshape(indices.shape[0],-1)
        if flat.shape[1] != sum(s*s for s in self.scales) or flat.min()<0 or flat.max()>=self.quantizer.size:
            raise ValueError('Invalid multiscale token sequence')
        parts=torch.split(flat,[s*s for s in self.scales],dim=1)
        z=self._decode_tokens(parts)
        return self.network.decoder(self.network.post_quant_conv(z))

    def rate(self):
        tokens=sum(s*s for s in self.scales)
        bits=tokens*math.ceil(math.log2(self.quantizer.size))
        return {'kind':'fixed_length_multiscale_token_payload','scales':list(self.scales),
                'tokens':tokens,'vocabulary':self.quantizer.size,'bits_per_image':bits,
                'bpp_spatial':bits/65536,'bpp_band_pixel':bits/(65536*self.channels),
                'includes_mask_header_or_model':False}
