"""Eight scratch runs: actual VAR VQVAE versus continuous spatial Prithvi AE."""
from pathlib import Path
import copy
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from eo_denoise.config import DEFAULT,validate
from eo_data.core import json_write

def config(m,backbone,noise,phase):
    c=copy.deepcopy(DEFAULT)
    c['name']=f'{phase}_{m}_{backbone}_{noise}_seed17'
    c['data'].update(modalities=[m],train_views=1,train_limit=64 if phase=='pilot' else None,
        val_limit=64 if phase=='pilot' else 256)
    c['noise'].update(kind='bank_missing',coverages=[0 if noise=='clean' else .1],clean_probability=0)
    c['loss'].update(stability=0,cross_view=0,kl=0)
    c['model'].update(backbone='var_conv' if backbone=='conv' else 'prithvi_eo2_continuous',
        quantizer='var_vq' if backbone=='conv' else 'ae',codebook_size=4096,
        latent_dim=32 if backbone=='conv' else 768,
        eo_dim=768,eo_depth=11,eo_heads=12,eo_decoder_dim=512,eo_decoder_depth=9,eo_decoder_heads=8,
        activation_checkpointing=False)
    micro=8
    effective=16 if phase=='pilot' else 64
    c['train'].update(steps=128 if phase=='pilot' else 10813,
        batch_size=micro,accumulation=effective//micro,workers=2,device='cuda',
        warmup=16 if phase=='pilot' else 500,
        eval_every=128 if phase=='pilot' else 1802,save_every=128 if phase=='pilot' else 300,
        log_every=16 if phase=='pilot' else 20,memory_limit_gib=20 if backbone=='conv' else 6)
    c['eval'].update(batch_size=micro,bootstrap_replicates=100 if phase=='pilot' else 500)
    validate(c)
    return c

if __name__=='__main__':
    for phase in ('pilot','full'):
        (HERE/'configs'/phase).mkdir(parents=True,exist_ok=True)
        for m in ('hls','lst'):
            for backbone in ('conv','prithvi'):
                for noise in ('clean','noisy'):
                    c=config(m,backbone,noise,phase)
                    json_write(HERE/'configs'/phase/(c['name']+'.json'),c)
