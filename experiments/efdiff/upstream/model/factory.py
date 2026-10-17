"""Network factory and weight initialization for LST super-resolution.

Builds the appropriate diffusion network from config `which_model_G`. Supports:
    sr3, sr3_prithvi, resshift, resshift_prithvi
"""
import functools
import logging

import torch.nn as nn
from torch.nn import init

logger = logging.getLogger('base')


# ============================================================================
# Weight initialization
# ============================================================================

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'Linear' in classname:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm2d' in classname:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if 'Conv2d' in classname:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'Linear' in classname:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm2d' in classname:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'Linear' in classname:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif 'BatchNorm2d' in classname:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


# ============================================================================
# Network factory (define_G)
# ============================================================================

def _unet_kwargs(model_opt):
    """Extract common UNet constructor kwargs from config."""
    return dict(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size'],
    )


def _build_prithvi_unet(model_opt):
    """Build UNet with cross-attention + PrithviEncoder. Returns (unet, encoder)."""
    from .unet import UNet
    from .prithvi_encoder import PrithviEncoder

    prithvi_opt = model_opt.get('prithvi', {})
    context_dim = prithvi_opt.get('embed_dim', 1024)

    unet = UNet(
        **_unet_kwargs(model_opt),
        context_dim=context_dim,
        cross_attn_heads=prithvi_opt.get('cross_attn_heads', 8),
        cross_attn_res=model_opt['unet'].get('cross_attn_res', None),
    )

    encoder = PrithviEncoder(
        model_name=prithvi_opt.get(
            'model_name', 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M'),
        embed_dim=context_dim,
        num_frames=prithvi_opt.get('num_frames', 1),
        freeze=prithvi_opt.get('freeze', True),
        input_is_reflectance=prithvi_opt.get('input_is_reflectance', True),
    )

    return unet, encoder


def define_G(opt):
    """Build the diffusion generator from config."""
    model_opt = opt['model']
    which = model_opt['which_model_G']

    has_prithvi = which in ('sr3_prithvi', 'resshift_prithvi')

    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups'] = 32

    # Build UNet (+ encoder if Prithvi)
    if has_prithvi:
        unet_model, enc = _build_prithvi_unet(model_opt)
    else:
        from .unet import UNet
        unet_model = UNet(**_unet_kwargs(model_opt))
        enc = None

    # Build diffusion wrapper
    if which in ('sr3', 'sr3_prithvi'):
        from .diffusion import GaussianDiffusion
        netG = GaussianDiffusion(
            denoise_fn=unet_model,
            image_size=model_opt['diffusion']['image_size'],
            channels=model_opt['diffusion']['channels'],
            loss_type='l1',
            conditional=model_opt['diffusion']['conditional'],
            prithvi_encoder=enc,
            concat_hls=False,
        )
    elif which in ('resshift', 'resshift_prithvi'):
        from .diffusion import ResShiftDiffusion
        resshift_opt = model_opt.get('resshift', {})
        netG = ResShiftDiffusion(
            denoise_fn=unet_model,
            image_size=model_opt['diffusion']['image_size'],
            channels=model_opt['diffusion']['channels'],
            loss_type=resshift_opt.get('loss_type', 'l2'),
            conditional=model_opt['diffusion']['conditional'],
            schedule_opt=model_opt['beta_schedule']['train'],
            kappa=resshift_opt.get('kappa', 1.0),
            prithvi_encoder=enc,
            concat_hls=False,
        )
    else:
        raise ValueError(
            f"Unknown model type: {which}. Supported: sr3, sr3_prithvi, "
            f"resshift, resshift_prithvi")

    # Initialize UNet weights (not Prithvi encoder)
    if opt['phase'] == 'train':
        init_weights(unet_model, init_type='orthogonal')

    return netG
