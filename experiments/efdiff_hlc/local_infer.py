"""Local end-to-end smoke evaluation: CPU frozen encoder + batch-one GPU diffusion."""
import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch

from pilot_data import HLCPairs, HERE, UPSTREAM
from model.unet import UNet
from model.factory import _unet_kwargs
from model.diffusion import GaussianDiffusion, ResShiftDiffusion

ROOT = HERE.parents[1]
OUTPUT = ROOT/'output/efdiff_hlc_pilot_20260915'


class FixedContext(torch.nn.Module):
    def __init__(self,context):
        super().__init__()
        self.register_buffer('context',context)

    def forward(self,hls):
        return self.context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=16)
    parser.add_argument('--variant',choices=['both','x0','epsilon'],default='both')
    args = parser.parse_args()
    assert 1 <= args.samples <= 512
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(.20)
    stats = json.loads((OUTPUT/'normalization.json').read_text())
    ds = HLCPairs(ROOT,split='val',stats=stats)
    positions = np.linspace(0,len(ds)-1,args.samples,dtype=int)
    models = OUTPUT/'models'
    manifest = json.loads((models/'manifest.json').read_text())
    variants = ('x0','epsilon') if args.variant=='both' else (args.variant,)
    for name,expected in manifest['files'].items():
        if name not in ['prithvi_encoder_batch1.pt']+[f'{v}_epoch5_ema.pt' for v in variants]:
            continue
        with (models/name).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==expected['sha256']
    print('All local weight hashes match cloud exports',flush=True)
    started = time.time()
    encoder = torch.jit.load(str(models/'prithvi_encoder_batch1.pt'),map_location='cpu').eval()
    samples,contexts = [],[]
    cloud_examples = np.load(OUTPUT/'x0/predictions_epoch5.npz')
    cloud_positions = {int(index):i for i,index in enumerate(cloud_examples['index'])}
    matched = 0
    condition_max_difference = 0.0
    with torch.inference_mode():
        for i,position in enumerate(positions):
            sample = ds[int(position)]
            if sample['index'] in cloud_positions:
                j = cloud_positions[sample['index']]
                for key in ('target_c','mask','record_id','month'):
                    assert np.array_equal(sample[key],cloud_examples[key][j]), f'Local/cloud data mismatch: {key}'
                difference = float(np.abs(sample['condition_c']-cloud_examples['condition_c'][j]).max())
                condition_max_difference = max(condition_max_difference,difference)
                np.testing.assert_allclose(sample['condition_c'],cloud_examples['condition_c'][j],rtol=0,atol=2e-5)
                matched += 1
            context = encoder(torch.from_numpy(sample['HLS']).unsqueeze(0))
            assert context.shape==(1,196,1024) and torch.isfinite(context).all()
            contexts.append(context)
            samples.append(sample)
            print(f'Local HLC read + encoder {i+1}/{len(positions)}',flush=True)
    del encoder
    gc.collect()
    metrics = {}
    dest = OUTPUT/'local_test'
    dest.mkdir(exist_ok=True)
    common = {k:np.asarray([s[k] for s in samples]) for k in ('target_c','condition_c','mask','index','record_id','month')}
    for variant in variants:
        state = torch.load(models/f'{variant}_epoch5_ema.pt',map_location='cpu',weights_only=False,mmap=True)
        assert state['variant']==variant and state['epoch']==5 and state['step']==1280
        assert state['normalization']==stats
        for relative,expected in state['training_code_sha256'].items():
            if relative.startswith('efdiff/upstream/'):
                assert hashlib.sha256((HERE.parent/relative).read_bytes()).hexdigest()==expected
        opt = state['config']['model']
        unet = UNet(**_unet_kwargs(opt),context_dim=1024,cross_attn_heads=8,cross_attn_res=opt['unet']['cross_attn_res'])
        unet.load_state_dict(state['denoiser'],strict=True)
        kwargs = dict(denoise_fn=unet,image_size=224,channels=1,conditional=True,
                      prithvi_encoder=FixedContext(contexts[0]),concat_hls=False)
        if variant=='x0':
            net = ResShiftDiffusion(**kwargs,loss_type='l2',kappa=1.0)
        else:
            net = GaussianDiffusion(**kwargs,loss_type='l1')
        net.set_new_noise_schedule(opt['beta_schedule']['train'],device)
        net = net.to(device).eval()
        del state,unet
        sse=sae=bsse=bsae=pixels=0
        predictions=[]
        variant_started=time.time()
        with torch.inference_mode():
            for i,(sample,context) in enumerate(zip(samples,contexts)):
                net.prithvi_encoder = FixedContext(context.to(device))
                batch = {k:torch.from_numpy(sample[k]).unsqueeze(0).to(device) for k in ('SR','HLS')}
                torch.manual_seed(20260916+int(sample['index']))
                if device.type=='cuda':
                    torch.cuda.manual_seed_all(20260916+int(sample['index']))
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    residual = net.super_resolution(batch,verbose=False,ddim_steps=50 if variant=='epsilon' else None)
                prediction = sample['condition_c'] + residual[0].float().cpu().numpy()*(3*stats['residual']['std'])+stats['residual']['mean']
                assert np.isfinite(prediction).all()
                mask = sample['mask']
                error=(prediction-sample['target_c'])[mask].astype(np.float64)
                base=(sample['condition_c']-sample['target_c'])[mask].astype(np.float64)
                sse+=float(np.square(error).sum()); sae+=float(np.abs(error).sum())
                bsse+=float(np.square(base).sum()); bsae+=float(np.abs(base).sum()); pixels+=len(error)
                predictions.append(prediction)
                print(f'{variant} local inference {i+1}/{len(samples)}',flush=True)
        np.savez_compressed(dest/f'{variant}_predictions.npz',**common,prediction_c=np.asarray(predictions))
        metrics[variant] = {'records':len(samples),'valid_pixels':pixels,'rmse_c':math.sqrt(sse/pixels),'mae_c':sae/pixels,
                            'bicubic_rmse_c':math.sqrt(bsse/pixels),'bicubic_mae_c':bsae/pixels,
                            'seconds':time.time()-variant_started,'all_predictions_finite':True}
        if device.type=='cuda':
            metrics[variant]['peak_gpu_gb']=torch.cuda.max_memory_allocated()/1e9
        del net,batch,residual
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache()
        print(json.dumps(metrics[variant]),flush=True)
    result = {'status':'passed','scope':'local inference check on fixed validation examples, not test-set evaluation',
              'checkpoint':'fixed_epoch5_ema','device':str(device),'encoder_device':'cpu','micro_batch':1,
              'encoder_precision':'float32','diffusion_precision':'bf16 autocast' if device.type=='cuda' else 'float32',
              'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,'torch':torch.__version__,
              'local_cloud_input_examples_matched':matched,
              'local_cloud_bicubic_max_difference_c':condition_max_difference,
              'source_indices':[int(s['index']) for s in samples],'metrics':metrics,
              'cloud_comparison_note':'Different sample count, per-sample noise seed, batch size and encoder precision; not a bitwise replay of cloud validation.',
              'elapsed_seconds':time.time()-started}
    filename = 'metrics.json' if args.variant=='both' else f'{args.variant}_metrics.json'
    (dest/filename).write_text(json.dumps(result,indent=2)+'\n')
    if args.variant!='both' and all((dest/f'{v}_metrics.json').exists() for v in ('x0','epsilon')):
        others = [json.loads((dest/f'{v}_metrics.json').read_text()) for v in ('x0','epsilon')]
        assert others[0]['source_indices']==others[1]['source_indices']
        result['metrics'] = {v:r['metrics'][v] for v,r in zip(('x0','epsilon'),others)}
        result['elapsed_seconds'] = sum(r['elapsed_seconds'] for r in others)
        (dest/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
