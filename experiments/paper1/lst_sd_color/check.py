"""Round trip, hidden-input isolation, finite gradient, exact epochs and stochastic resume checks."""
import argparse
import gc
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from eo_data.core import json_write
from eo_denoise import engine
from eo_denoise.data import EpochSampler
from .prepare import HERE,OUTPUT,ARMS,read
from .data import DatasetV2
from . import runtime


def numerical():
    engine.seed_everything(17);torch.set_num_threads(4)
    result={}
    for arm in ARMS:
        c=read(HERE/'configs/smoke'/f'{arm}.json')
        device=engine.device_for(c)
        model=runtime.ExperimentModel(c).to(device)
        codec=model.codecs['lst'];rep=codec.representation
        residual=torch.linspace(-120,120,65537,device=device).reshape(1,1,1,-1)/rep.std_c
        decoded,_,_=rep.decode(rep.encode(residual))
        error=float(((decoded-residual)*rep.std_c).abs().max())
        assert error<.0002, error
        ds=DatasetV2(c,'train')
        batch=next(iter(DataLoader(ds,batch_size=2)))
        moved=engine.move_batch(batch,device,model.codecs)
        model.eval()
        with torch.no_grad():
            x=moved['lst']['image'][:,0];mask=moved['lst']['input_mask'][:,0]
            p=codec(x,mask)
            altered=codec(torch.where(mask.bool(),x,12345),mask)
            assert torch.equal(p['reconstruction'],altered['reconstruction'])
            assert p['quantized'].shape==(2,4,32,32)
            assert torch.isfinite(p['reconstruction']).all()
        del p,altered
        model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        with engine.amp_context(c,device):
            outputs=model(engine.inputs_only(moved,model.codecs))
        loss,terms=runtime.objective(outputs,moved,c)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        assert torch.isfinite(loss)
        result[arm]=dict(roundtrip_max_error_celsius=error,invalid_values_ignored=True,
            latent_shape=[4,32,32],parameters=sum(p.numel() for p in model.parameters()),
            finite_loss=float(loss.detach()),gradient_norm=float(grad),memory=engine.memory_stats(device))
        ds.close()
        del optimizer,outputs,loss,terms,model,codec,rep,moved,batch
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    json_write(OUTPUT/'numerical_checks.json',dict(status='passed',arms=result))


def same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


def gate(phase):
    result={}
    for arm in ARMS:
        p=OUTPUT/phase/arm
        c=read(p/'config.json')
        rows=[__import__('json').loads(s) for s in (p/'train.jsonl').read_text().splitlines()]
        ids=np.load(p/'train_record_ids.npy')
        n=len(ids)
        assert len(rows)==c['train']['steps']
        assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
        for epoch in range(c['experiment']['epochs']):
            actual=[int(s.rsplit(':',1)[1]) for r in rows if r['epoch_index']==epoch for s in r['sample_ids']]
            expected=[int(ids[i]) for _,i in EpochSampler(n,c['seed'],epoch)]
            assert actual==expected and len(set(actual))==n
            assert [r for r in rows if r['epoch_index']==epoch][-1]['batch_samples']==n%64
        assert all(math.isfinite(r['total']) for r in rows)
        assert rows[-1]['records_presented']==n*c['experiment']['epochs']
        assert read(p/'status.json')['validated_epoch']==c['experiment']['epochs']
        if phase=='smoke':
            a=torch.load(p/'last.pt',weights_only=True,map_location='cpu')
            b=torch.load(OUTPUT/'smoke_uninterrupted'/arm/'last.pt',weights_only=True,map_location='cpu')
            for key in ('model','optimizer','rng_cpu','rng_cuda','records_presented','step','epoch'):same(a[key],b[key])
            del a,b
        result[arm]=dict(records=n*c['experiment']['epochs'],updates=len(rows),tail=n%64)
    json_write(OUTPUT/phase/'gate.json',dict(status='passed',exact_epochs=True,
        stochastic_resume_bitwise_equal=phase=='smoke',arms=result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['numerical','smoke','full'])
    a=p.parse_args();numerical() if a.mode=='numerical' else gate(a.mode)
    print('PASS',a.mode,flush=True)
