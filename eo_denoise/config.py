from copy import deepcopy
from pathlib import Path
import hashlib
import json

DEFAULT = {
    'version': 1, 'name': 'prithvi_vq11', 'seed': 17,
    'data': {'dataset': 'prithvi', 'root': '.', 'modalities': ['hls', 'lst'],
             'train_limit': None, 'val_limit': 256, 'subset_seed': 20260910,
             'alignment': 'same_tile_same_time_metadata', 'train_views': 2},
    'noise': {'kind': 'bank_missing', 'coverages': [0.05, 0.10, 0.20],
              'clean_probability': 0.25, 'sigma': 0.1,
              'bank_root': 'experiments/paper1/mask_banks', 'seed': 20260910},
    'model': {'backbone': 'conv', 'quantizer': 'vq', 'base_width': 64,
              'blocks': 2, 'latent_dim': 32, 'grid': 16, 'codebook_size': 512,
              'commitment': 0.25, 'levels': [8, 8, 8], 'temperature': 1.0,
              'vit_dim': 192, 'vit_depth': 4, 'vit_heads': 6,
              'eo_dim': 320, 'eo_depth': 6, 'eo_heads': 8,
              'eo_decoder_dim': 192, 'eo_decoder_depth': 3, 'eo_decoder_heads': 6,
              'var_ch':160, 'var_scales':[1,2,3,4,5,6,8,10,13,16],
              'var_quant_resi':0.5, 'var_share_quant_resi':4, 'activation_checkpointing':False},
    'loss': {'gradient': 0.1, 'stability': 0.1, 'cross_view': 0.1,
             'kl': 0.0001, 'min_token_valid': 0.75, 'relation_grid': 4},
    'train': {'steps': 20000, 'batch_size': 2, 'accumulation': 32,
              'workers': 2, 'lr': 0.0001, 'weight_decay': 0.01,
              'warmup': 1000, 'eval_every': 2000, 'save_every': 2000,
              'log_every': 20, 'clip': 1.0, 'device': 'auto', 'amp': True,
              'memory_limit_gib':26.0},
    'eval': {'coverage': 0.1, 'realizations': 2, 'batch_size': 4,
             'bootstrap_replicates': 500},
}


def merge(base, extra, prefix=''):
    out = deepcopy(base)
    for key, value in extra.items():
        if key not in base:
            raise ValueError(f'Unknown config key: {prefix}{key}')
        out[key] = merge(base[key], value, prefix + key + '.') if isinstance(base[key], dict) else value
    return out


def load_config(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        supplied = json.loads(path.read_text(encoding='utf-8'))
    else:
        import yaml
        supplied = yaml.safe_load(path.read_text(encoding='utf-8'))
    config = merge(DEFAULT, supplied)
    validate(config)
    return config


def validate(c):
    if c['version'] != 1 or c['data']['dataset'] != 'prithvi':
        raise ValueError('Paper 1 v1 uses independent one-to-one prithvi pairs only')
    mods = c['data']['modalities']
    if not mods or len(set(mods)) != len(mods) or any(m not in ('hls', 'lst') for m in mods):
        raise ValueError('modalities must be a nonempty subset of hls,lst')
    if c['loss']['cross_view'] and set(mods) != {'hls', 'lst'}:
        raise ValueError('Cross-view loss requires both modalities')
    if c['data']['alignment'] not in ('same_tile_same_time_metadata', 'all_pairs'):
        raise ValueError('Unknown alignment policy')
    if c['model']['backbone'] not in ('conv', 'prithvi_style', 'prithvi_eo2_spatial','var_conv','var_prithvi','prithvi_eo2_continuous'):
        raise ValueError('Unknown backbone')
    if c['data']['train_views'] not in (1, 2):
        raise ValueError('train_views must be 1 or 2')
    if c['loss']['stability'] and c['data']['train_views'] < 2:
        raise ValueError('Stability loss requires two training views')
    for prefix in ('eo_', 'eo_decoder_'):
        dim, heads, depth = (c['model'][prefix+k] for k in ('dim', 'heads', 'depth'))
        if dim < 16 or dim % 16 or heads < 1 or dim % heads or depth < 1:
            raise ValueError('Prithvi dimensions must divide heads and 16; depth must be positive')
    if c['model']['quantizer'] not in ('ae', 'vae', 'vq', 'fsq','var_vq'):
        raise ValueError('Unknown bottleneck')
    if c['model']['backbone'].startswith('var_') != (c['model']['quantizer']=='var_vq'):
        raise ValueError('VAR backbones require their original multiscale VAR quantizer')
    if c['model']['backbone']=='prithvi_eo2_continuous':
        if c['model']['quantizer']!='ae' or c['model']['latent_dim']!=c['model']['eo_dim'] or c['model']['grid']!=16:
            raise ValueError('Continuous Prithvi requires AE and native unquantized ViT feature dimensions')
    scales=c['model']['var_scales']
    if scales != sorted(set(scales)) or len(scales)<2 or scales[-1]!=16 or any(s<1 for s in scales):
        raise ValueError('VAR scales must increase and end at 16')
    if c['model']['var_ch']%32 or c['model']['var_ch']<32:
        raise ValueError('VAR width must be a multiple of 32')
    if c['model']['grid'] not in (8, 16, 32):
        raise ValueError('Supported rate grids: 8,16,32')
    if any(int(l) != l or l < 2 for l in c['model']['levels']):
        raise ValueError('FSQ levels must be integers >=2')
    if c['model']['codebook_size'] < 2 or c['model']['temperature'] <= 0:
        raise ValueError('Invalid quantizer settings')
    if c['model']['base_width'] < 4 or c['model']['latent_dim'] < 1 or c['model']['blocks'] < 1:
        raise ValueError('Invalid model dimensions')
    if c['model']['vit_dim'] % 6 or c['model']['vit_dim'] % c['model']['vit_heads']:
        raise ValueError('ViT dimension must be divisible by 6 and by heads')
    if c['noise']['kind'] not in ('none', 'bank_missing', 'procedural_missing', 'gaussian', 'stripes'):
        raise ValueError('Unknown noise kind')
    if not c['noise']['coverages'] or any(not 0 <= x < 1 for x in c['noise']['coverages']):
        raise ValueError('Noise coverage must be in [0,1)')
    if not 0 <= c['noise']['clean_probability'] <= 1 or c['noise']['sigma'] < 0:
        raise ValueError('Invalid noise settings')
    if not 0 <= c['eval']['coverage'] < 1:
        raise ValueError('Invalid evaluation coverage')
    for key in ('steps', 'batch_size', 'accumulation', 'eval_every', 'save_every', 'log_every'):
        if c['train'][key] < 1:
            raise ValueError(f'train.{key} must be positive')
    if c['train']['workers'] < 0 or c['eval']['realizations'] < 2:
        raise ValueError('workers must be >=0; at least two evaluation views are required')
    if c['train']['memory_limit_gib'] <= 0:
        raise ValueError('train.memory_limit_gib must be positive')
    for key in ('train_limit', 'val_limit'):
        if c['data'][key] is not None and c['data'][key] < 1:
            raise ValueError('Subset limits must be positive or null')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
