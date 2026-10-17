"""Local, equal-budget technical pilot. Never submits jobs or opens test data."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, gc, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from experiments.paper1.lst_restoration_lossfix_20261014 import objective
from experiments.paper1.lst_restoration_lossfix_20261014.data import LSTDataset, EpochSampler, stable_seed
from experiments.paper1.lst_restoration_lossfix_20261014.models import build_model
from experiments.paper1.lst_restoration_lossfix_20261014.representations import Representation
from experiments.paper1.lst_restoration_lossfix_20261014.grouped_adapter import learning_rate, grouped_statistics
from .quantizers import EMAVQ, EMAGrouped, controllers, initialize_from_latents

ROOT=Path(__file__).resolve().parents[3]

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def batch(samples):
    b=default_collate(samples)
    return {k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in b.items()}

def weights_hash(model,exclude_quantizer=False):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        if exclude_quantizer and k.startswith('quantizer.'):continue
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def evaluate(model,c,rep,validation,micro):
    model.eval();result={}
    for condition,samples in validation.items():
        ss=ae=n=missing_ss=missing_n=image_mse=0.;indices=[]
        for start in range(0,len(samples),micro):
            b=batch(samples[start:start+micro])
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**objective.model_inputs(b,rep))
            error=(objective.readout(out,b,rep,c)-b['temperature']).double()
            valid=b['reference_valid'];missing=valid & ~b['observed_valid']
            ss+=float(error.square()[valid].sum());ae+=float(error.abs()[valid].sum());n+=int(valid.sum())
            missing_ss+=float(error.square()[missing].sum());missing_n+=int(missing.sum())
            image_mse+=float(objective.masked_per_image(error.square(),valid)[0].sum())
            indices.append(out['indices'].cpu().numpy())
        ids=np.concatenate(indices)
        if ids.ndim==3 and c['model']['kind']=='vq':
            counts=np.bincount(ids.ravel(),minlength=c['model']['codebook_size']);p=counts/counts.sum()
            usage={'used_codes':int((counts>0).sum()),'perplexity':float(np.exp(-(p*np.log(p.clip(1e-30))).sum())),
                   'top1_fraction':float(p.max()),'index_counts':counts.tolist()}
        else:usage=grouped_statistics(ids,c['model']['scales'],c['model']['codebook_size'])
        result[condition]=dict(rmse_c=math.sqrt(ss/n),mae_c=ae/n,valid_pixels=int(n),
            image_mean_mse_c2=image_mse/len(samples),missing_rmse_c=math.sqrt(missing_ss/missing_n) if missing_n else None,
            record_views=len(samples),code_usage=usage)
    model.train();return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--updates',type=int,default=128)
    ap.add_argument('--effective-batch',type=int,default=16);ap.add_argument('--micro',type=int,default=2)
    ap.add_argument('--val-records',type=int,default=64);ap.add_argument('--arms',default='vq:baseline,vq:ema,vq:ema_maint,grouped_g8k256:baseline,grouped_g8k256:ema,grouped_g8k256:ema_maint')
    args=ap.parse_args();outdir=ROOT/args.output;outdir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(6);torch.manual_seed(17);np.random.seed(17);random.seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    assert args.effective_batch%args.micro==0 and args.updates>0
    configs={}
    for family in ('vq','grouped_g8k256'):
        c=json.loads((ROOT/f'outputs/torch_models_20261013/models/{family}_G/config.original.json').read_text())
        c['data']['root']=str(ROOT);c['data']['historical_test_exposure']=False
        c['tracking']={'enabled':False};c['execution']={'enabled':False,'note':'Separate explicitly authorized local technical pilot runner'}
        c['train'].update(updates=args.updates,effective_batch=args.effective_batch,micro_batch=args.micro,
            warmup_updates=min(10,args.updates),workers=0,selection='fixed_validation_subset_pixel10_rmse',epochs=None,
            record_presentations=args.updates*args.effective_batch,memory_limit_gib=None)
        configs[family]=c
    c=configs['vq'];train=LSTDataset(c,'train')
    selected=list(EpochSampler(len(train),17,0))[:args.updates*args.effective_batch]
    train_samples=[train[i] for i in selected]
    val_indices=np.random.default_rng(stable_seed('ema_pilot_val',17)).permutation(11684)[:args.val_records].tolist()
    validation={};fingerprints={'train':train.fingerprint}
    for kind,views in [('clean',1),('pixel10',2)]:
        samples=[]
        for view in range(views):
            ds=LSTDataset(c,'val',evaluation_profile={'kind':kind,'coverage':.1,'seed':20260925},view=view)
            samples.extend(ds[i] for i in val_indices);fingerprints[f'{kind}_{view}']=ds.fingerprint;ds.close()
        validation[kind]=samples
    train.close()
    assert not ({s['tile_id'] for s in train_samples}&{s['tile_id'] for s in validation['clean']})
    manifest=dict(purpose='technical pilot, not five full epochs',arguments=vars(args),seed=17,representation='G',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,data=fingerprints,
        train_members=[dict(index=i[1],sample_id=s['sample_id'],tile_id=s['tile_id']) for i,s in zip(selected,train_samples)],
        validation_members=[dict(index=i,sample_id=s['sample_id'],tile_id=s['tile_id']) for i,s in zip(val_indices,validation['clean'])],
        notes=['Scratch models; no old trained weights. Train-only normalization. No test data.',
        'EMA decay .99, update after full effective batch. No RMS normalization.',
        'Maintenance combines train K-means (10 iterations) and dead-code replacement; cannot isolate their individual effects.',
        'Maintenance threshold: bias-corrected EMA frequency < .01/K every 16 updates after 16-update grace.',
        'Grouped shared dictionaries aggregate residual assignments across all scales; Phi keeps native fitting loss.',
        'Total losses differ by codebook update method; compare physical validation error and quantization MSE.',
        'Matched initialization within each family; families differ in architecture and nominal bitrate.',
        'K-means uses first effective training batch latents, additional encoder-only calibration passes.',
        'Validation step 0/mid/final, raw pixel-weighted Celsius RMSE; selection uses subset only.'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in
            ['experiments/paper1/lst_ema_pilot_20261014','experiments/paper1/lst_restoration_lossfix_20261014','eo_data','eo_denoise']
            for p in (ROOT/folder).rglob('*.py')})
    save(outdir/'manifest.json',manifest);results=[]
    print(json.dumps({'event':'data_ready','train':len(train_samples),'val':args.val_records}),flush=True)
    for arm in args.arms.split(','):
        family,method=arm.split(':');folder=outdir/(family+'_'+method);folder.mkdir()
        c=copy.deepcopy(configs[family]);c['pilot_method']=method;save(folder/'config.json',c)
        torch.manual_seed(17);torch.cuda.manual_seed_all(17);np.random.seed(17);random.seed(17)
        model=build_model(c);initial_backbone=weights_hash(model,True)
        if method!='baseline':model.quantizer=(EMAVQ if family=='vq' else EMAGrouped)(model.quantizer,method=='ema_maint')
        model=model.cuda();rep=Representation(c['representation']);optimizer=torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],lr=c['train']['lr'],betas=(.9,.95),weight_decay=.01)
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        if method=='ema_maint':
            model.eval();latents=[]
            with torch.no_grad():
                for pos in range(0,args.effective_batch,args.micro):
                    b=batch(train_samples[pos:pos+args.micro]);inputs=objective.model_inputs(b,rep);inputs.pop('patch_missing')
                    with torch.autocast('cuda',dtype=torch.bfloat16):latents.append(model._encode(model.input_adapter(**inputs)).float())
                initialize_from_latents(model.quantizer,torch.cat(latents))
            del latents
        evaluations=[];best=float('inf');beststep=None
        def validate(step):
            nonlocal best,beststep
            metrics=evaluate(model,c,rep,validation,args.micro);evaluations.append(dict(step=step,metrics=metrics))
            score=metrics['pixel10']['rmse_c'];state={'model':model.state_dict(),'step':step,'config':c}
            torch.save(state,folder/f'step_{step:04d}.pt')
            if score<best:best=score;beststep=step
            save(folder/'evaluations.json',evaluations)
            print(json.dumps(dict(event='validation',arm=arm,step=step,rmse_c=score,clean=metrics['clean']['rmse_c'])),flush=True)
        validate(0);model.train()
        with (folder/'training.jsonl').open('w',encoding='utf-8') as log:
            for step in range(args.updates):
                tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(c,step)
                for controller in controllers(model):controller.begin()
                stats={};examples=train_samples[step*args.effective_batch:(step+1)*args.effective_batch]
                for pos in range(0,len(examples),args.micro):
                    b=batch(examples[pos:pos+args.micro]);fraction=len(b['temperature'])/len(examples)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        prediction=model(**objective.model_inputs(b,rep));loss,logged=objective.loss(prediction,b,rep,c)
                    (loss*fraction).backward()
                    for k,v in logged.items():stats[k]=stats.get(k,0.)+v*fraction
                    del prediction,loss,b
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                for controller in controllers(model):controller.commit()
                stats.update(step=step+1,lr=optimizer.param_groups[0]['lr'],gradient_norm=float(norm),
                    seconds=time.perf_counter()-tick,reset_counts=[int(e.reset_count) for e in controllers(model)])
                log.write(json.dumps(stats,allow_nan=False)+'\n');log.flush()
                if (step+1)%8==0 or step==0:print(json.dumps(dict(event='training',arm=arm,**stats)),flush=True)
                if step+1 in {max(1,args.updates//2),args.updates}:validate(step+1)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=args.updates,config=c,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),folder/'last_resume.pt')
        result=dict(arm=arm,best_step=beststep,best_pixel10_rmse_c=best,final=evaluations[-1],
            seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,initial_backbone_sha256=initial_backbone,
            parameters=sum(p.numel() for p in model.parameters()),reset_counts=[int(e.reset_count) for e in controllers(model)])
        save(folder/'result.json',result);results.append(result);save(outdir/'results.json',results)
        print(json.dumps({'event':'arm_complete','arm':arm,'seconds':result['seconds'],'peak_gib':result['peak_allocated_gib']}),flush=True)
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    save(outdir/'complete.json',{'completed':True,'arms':len(results)})

if __name__=='__main__':main()
