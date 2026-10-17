"""Freeze A07 configuration, pretrained identity and explicit HLP v2 input fingerprints."""
from pathlib import Path
import copy
import json
import importlib.metadata
from eo_data.core import json_write, sha256_file
from .data import DatasetV2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT/'runs/paper1/lst_sd_color_20260921'
ARMS = ('linear_gray','asinh_gray','asinh_rgb')
REVISION = '31f26fdeee1355a5c34592e401dd41e45d25a493'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    assert not (OUTPUT/'manifest.json').exists(), 'Experiment already frozen'
    base = read(next((ROOT/'experiments/paper1/lst_residual_scale/configs/full').glob('*.json')))
    weights = ROOT/'models/sd-vae-ft-mse-31f26fd'
    base['model'] = dict(backbone='sd_kl_f8',quantizer='vae',grid=32,latent_dim=4)
    base.pop('residual_experiment');base.pop('residual_scaling')
    base['pretrained'] = dict(repo='stabilityai/sd-vae-ft-mse',revision=REVISION,directory=str(weights),
        sha256={n:sha256_file(weights/n) for n in ('config.json','diffusion_pytorch_model.safetensors')})
    base['representation'] = dict(kind=None,std_celsius=16.29419157454727,
        residual_std_celsius=2.6950881812941048,bound_celsius=120.,
        center='visible_input_mean_only',source_valid_range_celsius=[-50,70],
        scale_source='runs/paper1/lst_residual_scale_20260918/residual_scale.json',
        fit_split='train',output_precision='float32',input_quantization='none')
    base['loss'] = dict(gradient=.1,spatial_extra=3.,representation=.01,kl=1e-6,
        units='frozen_v2_normalized_temperature',kl_reduction='mean_over_latent_elements')
    base['train'].update(batch_size=2,accumulation=32,lr=1e-5,memory_limit_gib=8.5,
        workers=2,log_every=10,save_every=100)
    base['eval'].update(batch_size=2,amp=False)
    base['experiment'].update(id='A07',output_root=str(OUTPUT),pretrained_weights=True,
        trainable='all_encoder_and_decoder_parameters',primary_checkpoint='last_after_exact_epochs')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for phase in ('smoke','full'):
        (HERE/'configs'/phase).mkdir(parents=True,exist_ok=True)
        for arm in ARMS:
            c=copy.deepcopy(base)
            c['name']=phase+'_'+arm
            c['representation']['kind']=arm
            c['experiment']['phase']=phase
            if phase=='smoke':
                c['data'].update(train_limit=67,val_limit=8,test_limit=8)
                c['experiment']['epochs']=2
                c['train'].update(steps=4,warmup=1,save_every=1,log_every=1)
                c['eval']['bootstrap_replicates']=0
            json_write(HERE/'configs'/phase/(arm+'.json'),c)
    preflight={}
    c=read(HERE/'configs/full/linear_gray.json')
    for split in ('train','val','test'):
        ds=DatasetV2(c,split)
        assert abs(ds.sources['lst'].stats['std'][0]/100-base['representation']['std_celsius'])<1e-12
        preflight[split]=ds.fingerprint
        ds.close()
    json_write(OUTPUT/'data_preflight.json',dict(status='passed',splits=preflight))
    json_write(OUTPUT/'packages.json',{n:importlib.metadata.version(n) for n in
        ('torch','torchvision','diffusers','huggingface_hub','safetensors','numpy')})
    json_write(OUTPUT/'pretrained.json',base['pretrained'])
    print('Prepared',OUTPUT)


if __name__=='__main__':main()
