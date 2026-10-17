"""Fixed epoch-one EMA evaluation on every HLC test record, with resumable chunks."""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data import HLCPairs, HERE, COUNTS, check_stats
from pilot_data import HLCPairs as ValidationPairs
from model.unet import UNet
from model.factory import _unet_kwargs
from model.diffusion import GaussianDiffusion, ResShiftDiffusion

ROOT = HERE.parents[1]
OUT = ROOT/'output/efdiff_hlc_epoch1_deadline_io16_20260916'
MODELS = OUT/'models'
VARIANTS = ('x0','epsilon')


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


class FixedContext(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.context = None

    def forward(self, hls):
        assert self.context is not None and self.context.shape[0] == hls.shape[0]
        return self.context


def build_net(variant, stats):
    state = torch.load(MODELS/f'{variant}_epoch1_ema.pt',map_location='cpu',weights_only=False,mmap=True)
    assert state['variant']==variant and state['epoch']==1 and state['step']==40763
    assert state['presented']==2608772 and state['normalization']==stats
    for name, expected in state['training_code_sha256'].items():
        if name.startswith('efdiff/upstream/'):
            assert sha(HERE.parent/name)==expected, name
    opt = state['config']['model']
    unet = UNet(**_unet_kwargs(opt),context_dim=1024,cross_attn_heads=8,
                cross_attn_res=opt['unet']['cross_attn_res'])
    unet.load_state_dict(state['denoiser'],strict=True)
    kwargs = dict(denoise_fn=unet,image_size=224,channels=1,conditional=True,
                  prithvi_encoder=FixedContext(),concat_hls=False)
    net = ResShiftDiffusion(**kwargs,loss_type='l2',kappa=1.0) if variant=='x0' else GaussianDiffusion(**kwargs,loss_type='l1')
    net.set_new_noise_schedule(opt['beta_schedule']['train'],'cuda')
    return net.cuda().eval()


def error_sums(prediction, target, mask):
    """Float64 accumulation, without squaring rounded float32 errors."""
    result = []
    for p,t,m in zip(prediction,target,mask):
        delta = p[m].astype(np.float64)-t[m].astype(np.float64)
        assert len(delta)>0 and np.isfinite(delta).all()
        result.append((int(len(delta)),float(np.square(delta).sum()),float(np.abs(delta).sum())))
    return result


def aggregate(parts):
    result = {}
    for name in (*VARIANTS,'bicubic'):
        pixels = sum(p['sums'][name]['pixels'] for p in parts)
        sse = sum(p['sums'][name]['sse'] for p in parts)
        sae = sum(p['sums'][name]['sae'] for p in parts)
        result[name]={'records':sum(p['records'] for p in parts),'valid_pixels':pixels,
                      'rmse_c':math.sqrt(sse/pixels),'mae_c':sae/pixels,
                      'squared_error_sum':sse,'absolute_error_sum':sae}
    return result


def report(dest, protocol, parts, metrics):
    """Generate the final report only after complete ordered coverage checks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=['# HLC full test-set evaluation — fixed epoch-one EMA','',
        f"Split: {protocol['split']}; {metrics['x0']['records']:,} valid fine-LST patch-months. "
        'Synthetic Wald ×32 coarse LST + associated HLS; center 224×224 crop; valid reference mask.',
        '', 'Fixed epoch-one EMA weights; train-only normalization; no selection or tuning on test. '
        'x₀ native 15-step sampling; epsilon DDIM 50 steps; local RTX 5090 BF16 autocast. '
        'Noise uses a fixed seed per ordered batch. This is not a bitwise replay of cloud validation.',
        '', '| Model | RMSE (°C) | MAE (°C) |', '|---|---:|---:|']
    for v in ('bicubic',*VARIANTS):
        lines.append(f"| {v} | {metrics[v]['rmse_c']:.6f} | {metrics[v]['mae_c']:.6f} |")
    captures=[]
    for p in sorted((dest/'chunks').glob('*_examples.npz')):
        with np.load(p) as z:
            for i in range(len(z['index'])):
                captures.append({k:z[k][i] for k in z.files})
    captures.sort(key=lambda x:int(x['index']))
    assert len(captures)==16 and len({int(c['index']) for c in captures})==16
    for page in range(4):
        fig,axes=plt.subplots(4,6,figsize=(18,12),layout='constrained')
        for row,c in enumerate(captures[page*4:(page+1)*4]):
            mask=c['mask'][0]
            target,base,x0,eps=(c[k][0] for k in ('target_c','condition_c','x0','epsilon'))
            low,high=np.percentile(target[mask],[1,99])
            lim=max(float(np.percentile(np.abs(x0-target)[mask],98)),float(np.percentile(np.abs(eps-target)[mask],98)),.1)
            for col,(value,title) in enumerate(zip((target,base,x0,eps,x0-target,eps-target),
                ('Fine LST reference','Bicubic','EFDiff-x0','EFDiff-epsilon','x0 error','epsilon error'))):
                im=axes[row,col].imshow(np.where(mask,value,np.nan),cmap='coolwarm' if col>=4 else 'turbo',
                    vmin=-lim if col>=4 else low,vmax=lim if col>=4 else high)
                axes[row,col].set_title(title);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                fig.colorbar(im,ax=axes[row,col],fraction=.046,pad=.02,label='°C')
            axes[row,0].set_ylabel(f"record {int(c['record_id'])}\nmonth {c['month']}")
        name=f'comparison_{page+1}.png';fig.savefig(dest/name,dpi=120);plt.close(fig)
        lines += ['',f'![Fixed test examples {page+1}]({name})']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # One row for every evaluated test record, not a sampled diagnostic table.
    with (dest/'per_record_metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        for i,p in enumerate(parts):
            with (dest/'chunks'/f"{p['start']:07d}_records.csv").open(encoding='utf-8') as source:
                header=next(source)
                if i==0: stream.write(header)
                for line in source: stream.write(line)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true',help='32 validation records; no test metrics')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-chunks',type=int,default=0,help='Technical bounded run; default all remaining chunks')
    args=parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.85)
    torch.backends.cudnn.benchmark=True
    torch.manual_seed(20260921)
    stats=json.loads((OUT/'normalization.json').read_text())
    check_stats(stats,HLCPairs(ROOT,load_hls=False))
    manifest=json.loads((MODELS/'manifest.json').read_text())
    assert manifest['checkpoint']=='fixed_epoch1_ema'
    for name,info in manifest['files'].items():
        assert (MODELS/name).stat().st_size==info['bytes'] and sha(MODELS/name)==info['sha256'], name
    ds=ValidationPairs(ROOT,split='val',stats=stats) if args.smoke else HLCPairs(ROOT,split='test',stats=stats)
    count=32 if args.smoke else len(ds)
    assert args.smoke or count==COUNTS['test']==141524
    if not args.smoke:
        db=ds.lst._connect()
        tiles={s:{r[0] for r in db.execute('SELECT DISTINCT tile FROM records WHERE split=?',(s,))} for s in ('train','val','test')}
        assert len(tiles['test'])==123 and not tiles['test']&tiles['train'] and not tiles['test']&tiles['val']
        assert len(ds.hls)==13665
    dest=OUT/('local_epoch1_smoke' if args.smoke else 'full_test_20260921')
    dest.mkdir(parents=True,exist_ok=True);(dest/'chunks').mkdir(exist_ok=True)
    batch_size,chunk_size=16,512
    protocol={'version':'hlc-epoch1-local-fulltest-v1','split':'val' if args.smoke else 'test',
        'expected_records':count,'crop':224,'degradation':'synthetic Wald32; same frozen data.py',
        'checkpoint':'fixed_epoch1_ema','batch_size':batch_size,'chunk_size':chunk_size,
        'seed_rule':'20260921 + first source index of each ordered batch; reset separately for each variant',
        'encoder_precision':'GPU BF16 autocast','denoiser_precision':'GPU BF16 autocast',
        'sampling':{'x0':15,'epsilon':50},'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'data_fingerprint':ds.fingerprint,
        'stats_sha256':hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest(),
        'model_manifest':manifest,'code_sha256':{p.name:sha(p) for p in (Path(__file__),HERE/'data.py')}}
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    existing=dest/'protocol.json'
    if existing.exists():
        assert json.loads(existing.read_text())==protocol,'Evaluation protocol changed; use a new output version'
    else: write_json(existing,protocol)
    parts=[]
    for start in range(0,count,chunk_size):
        p=dest/'chunks'/f'{start:07d}.json'
        if not p.exists(): break
        item=json.loads(p.read_text())
        assert item['protocol_sha256']==fingerprint and item['start']==start and item['end']==min(start+chunk_size,count)
        assert item['records']==item['end']-start
        parts.append(item)
    offset=parts[-1]['end'] if parts else 0
    if offset==count:
        metrics=aggregate(parts)
        report(dest,protocol,parts,metrics)
        write_json(dest/'metrics.json',{'status':'complete','protocol_sha256':fingerprint,'split':protocol['split'],
            'checkpoint':'fixed_epoch1_ema','records':count,'metrics':metrics,
            'total_chunk_seconds':sum(p['seconds'] for p in parts),'test_used_for_selection':False})
        write_json(dest/'status.json',{'phase':'complete','records_done':count,'records_total':count,
            'fraction':1.0,'protocol_sha256':fingerprint,'resumed_after_final_chunk':True})
        return
    encoder=torch.jit.load(str(MODELS/'prithvi_encoder.pt'),map_location='cuda').eval()
    nets={v:build_net(v,stats) for v in VARIANTS}
    capture_positions=set(np.linspace(0,count-1,16,dtype=int).tolist())
    loader=DataLoader(ds,sampler=range(offset,count),batch_size=batch_size,num_workers=args.workers,
        pin_memory=True,persistent_workers=args.workers>0)
    start=offset;processed=offset;done_chunks=0
    run_start=time.time();chunk_started=run_start
    sums={v:{'pixels':0,'sse':0.,'sae':0.} for v in (*VARIANTS,'bicubic')}
    rows=[];captures=[];timings={v:0. for v in VARIANTS}
    def status(phase):
        seconds=time.time()-run_start
        throughput=(processed-offset)/seconds if processed>offset else None
        write_json(dest/'status.json',{'phase':phase,'pid':os.getpid(),'records_done':processed,'records_total':count,
            'fraction':processed/count,'elapsed_seconds_this_process':seconds,'samples_per_second_this_process':throughput,
            'estimated_remaining_hours':(count-processed)/throughput/3600 if throughput else None,
            'peak_gpu_GB':torch.cuda.max_memory_allocated()/1e9,'protocol_sha256':fingerprint})
    status('running')
    with torch.inference_mode():
        for raw in loader:
            n=len(raw['index']);positions=np.arange(processed,processed+n)
            if not args.smoke: np.testing.assert_array_equal(raw['index'].numpy(),positions)
            b={k:raw[k].cuda(non_blocking=True) for k in ('SR','HLS')}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                context=encoder(b['HLS'])
            assert context.shape==(n,196,1024) and torch.isfinite(context).all()
            target=raw['target_c'].numpy();condition=raw['condition_c'].numpy();mask=raw['mask'].numpy()
            per_record={'bicubic':error_sums(condition,target,mask)}
            predictions={}
            for v,net in nets.items():
                net.prithvi_encoder.context=context
                torch.manual_seed(20260921+int(raw['index'][0]));torch.cuda.manual_seed_all(20260921+int(raw['index'][0]))
                torch.cuda.synchronize();tick=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    residual=net.super_resolution(b,verbose=False,ddim_steps=50 if v=='epsilon' else None)
                prediction=condition+residual.float().cpu().numpy()*(3*stats['residual']['std'])+stats['residual']['mean']
                timings[v]+=time.perf_counter()-tick
                assert np.isfinite(prediction).all()
                predictions[v]=prediction;per_record[v]=error_sums(prediction,target,mask)
            for i,position in enumerate(positions):
                month=raw['month'][i].item() if torch.is_tensor(raw['month']) else raw['month'][i]
                row={'index':int(raw['index'][i]),'record_id':int(raw['record_id'][i]),'month':str(month)}
                for v in (*VARIANTS,'bicubic'):
                    pixels,sse,sae=per_record[v][i]
                    sums[v]['pixels']+=pixels;sums[v]['sse']+=sse;sums[v]['sae']+=sae
                    row.update({f'{v}_sse':sse,f'{v}_sae':sae,f'{v}_rmse_c':math.sqrt(sse/pixels),f'{v}_mae_c':sae/pixels})
                row['valid_pixels']=pixels;rows.append(row)
                if int(position) in capture_positions:
                    capture={k:raw[k][i].numpy() if torch.is_tensor(raw[k]) else raw[k][i] for k in ('index','record_id','month','target_c','condition_c','mask')}
                    capture.update({v:predictions[v][i] for v in VARIANTS});captures.append(capture)
            processed+=n
            if processed%64==0 or processed==count:
                status('running');print(f'Both models: {processed}/{count} records',flush=True)
            if processed-start==chunk_size or processed==count:
                assert len(rows)==processed-start
                if captures:
                    temp=dest/'chunks'/f'{start:07d}_examples.tmp'
                    with temp.open('wb') as stream: np.savez_compressed(stream,**{k:np.asarray([c[k] for c in captures]) for k in captures[0]})
                    temp.replace(dest/'chunks'/f'{start:07d}_examples.npz')
                temp=dest/'chunks'/f'{start:07d}_records.tmp'
                with temp.open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                temp.replace(dest/'chunks'/f'{start:07d}_records.csv')
                part={'start':start,'end':processed,'records':processed-start,'sums':sums,
                    'protocol_sha256':fingerprint,'seconds':time.time()-chunk_started,'diffusion_seconds':timings}
                write_json(dest/'chunks'/f'{start:07d}.json',part);parts.append(part)
                done_chunks+=1
                if args.max_chunks and done_chunks>=args.max_chunks and processed<count:
                    status('checkpointed');return
                start=processed;chunk_started=time.time();rows=[];captures=[]
                sums={v:{'pixels':0,'sse':0.,'sae':0.} for v in (*VARIANTS,'bicubic')};timings={v:0. for v in VARIANTS}
    assert processed==count and sum(p['records'] for p in parts)==count
    assert parts[0]['start']==0 and parts[-1]['end']==count
    assert all(a['end']==b['start'] for a,b in zip(parts,parts[1:]))
    metrics=aggregate(parts);report(dest,protocol,parts,metrics)
    write_json(dest/'metrics.json',{'status':'complete','protocol_sha256':fingerprint,'split':protocol['split'],
        'checkpoint':'fixed_epoch1_ema','records':count,'metrics':metrics,
        'total_chunk_seconds':sum(p['seconds'] for p in parts),'test_used_for_selection':False})
    status('complete');print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.CreateMutexW.restype=ctypes.c_void_p
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        mutex=kernel.CreateMutexW(None,False,'Local\\EFDiff_HLC_Epoch1_FullTest_20260921')
        if not mutex or kernel.GetLastError()==183:
            raise RuntimeError('Another local full-test process already holds the evaluation lock')
        kernel.SetThreadExecutionState(0x80000001)  # Keep system awake only while this process runs.
        try:
            main()
        finally:
            kernel.SetThreadExecutionState(0x80000000)
            kernel.CloseHandle(mutex)
    else:
        main()
