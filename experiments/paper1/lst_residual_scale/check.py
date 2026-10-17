"""Numerical codec checks and exact-epoch / resume gates, isolated from full run."""
import argparse
import gc
import json
import math
import numpy as np
import torch
from eo_data.core import json_write
from eo_denoise import engine
from . import runtime
from .prepare import HERE, OUTPUT, NAME, BASE_RUN, read
from .data import DatasetV2


def numerical():
    c = read(HERE/'configs/smoke'/f'{NAME.replace("full_", "smoke_", 1)}.json')
    engine.seed_everything(c['seed'])
    torch.set_num_threads(4)
    device = engine.device_for(c)
    model = runtime.ExperimentModel(c).to(device).eval()
    codec = model.codecs['lst']
    ds = DatasetV2(c, 'train')
    a = ds[0]['lst']
    x = torch.from_numpy(a['image'][0:1]).to(device)
    mask = torch.from_numpy(a['input_mask'][0:1]).to(device)
    scale = codec.residual_scale.clone()
    with torch.inference_mode():
        codec.residual_scale.fill_(1)
        old = runtime.base.ResidualCodec(codec.inner)
        with engine.amp_context(c,device):
            original = old(x, mask)
            identity = codec(x, mask)
        assert torch.equal(original['reconstruction'], identity['reconstruction'])
        assert torch.equal(original['indices'], identity['indices'])
        codec.residual_scale.copy_(scale)
        mu = runtime.base.visible_mean(x, mask)
        residual = (x-mu)/scale
        torch.testing.assert_close((mu+scale*residual)[mask.bool()], x[mask.bool()], rtol=1e-6, atol=1e-6)
        with engine.amp_context(c,device):
            scaled = codec(x,mask)
            decoded = codec.decode_indices(scaled['indices'], scaled['temperature_baseline'])
            altered = codec(torch.where(mask.bool(),x,torch.full_like(x,12345)),mask)
        torch.testing.assert_close(decoded,scaled['reconstruction'],rtol=0,atol=0)
        assert torch.equal(scaled['reconstruction'],altered['reconstruction'])
        assert codec.rate()['bits_per_image'] == 6832
        # An externally supplied target is never an argument to the codec.
    ds.close()
    json_write(OUTPUT/'numerical_checks.json',dict(status='passed', scale_one_matches_existing_codec=True,
        inverse_scale_roundtrip=True, token_decode_matches_forward=True, invalid_values_ignored=True,
        per_image_bits=6832, memory=engine.memory_stats(device)))
    del model, codec, old
    gc.collect(); torch.cuda.empty_cache()


def gate(phase):
    name = NAME.replace('full_',phase+'_',1)
    p=OUTPUT/phase/name
    c=read(p/'config.json')
    rows=[json.loads(s) for s in (p/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    n=c['data']['train_limit'] or 218975
    assert len(rows)==c['train']['steps']
    assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
    assert all(math.isfinite(r['total']) for r in rows)
    members=set(np.load(p/'train_record_ids.npy').tolist())
    assert len(members)==n
    for e in range(c['experiment']['epochs']):
        er=[r for r in rows if r['epoch_index']==e]
        ids=[int(s.rsplit(':',1)[1]) for r in er for s in r['sample_ids']]
        assert len(ids)==n and set(ids)==members
        assert er[-1]['batch_samples']==(n%64 or 64)
    assert rows[-1]['records_presented']==n*c['experiment']['epochs']
    assert read(p/'status.json')['validated_epoch']==c['experiment']['epochs']
    if phase=='smoke':
        resumed=torch.load(p/'last.pt',map_location='cpu',weights_only=True)
        uninterrupted=torch.load(OUTPUT/'smoke_uninterrupted'/name/'last.pt',map_location='cpu',weights_only=True)
        def same(a,b):
            if torch.is_tensor(a):
                assert torch.equal(a,b)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a:same(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b)
                for x,y in zip(a,b):same(x,y)
            else:assert a==b
        for k in ('model','optimizer','rng_cpu','rng_cuda','records_presented','step','epoch'):
            same(resumed[k],uninterrupted[k])
    else:
        old=[json.loads(s) for s in (BASE_RUN/'train.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['sample_ids'] for r in rows]==[r['sample_ids'] for r in old]
        assert read(p/'data_fingerprint.json')==read(BASE_RUN/'data_fingerprint.json')
    json_write(OUTPUT/phase/'gate.json',dict(status='passed', exact_epochs=True,
        records=n*c['experiment']['epochs'], updates=len(rows), tail=n%64,
        check='bitwise resume equivalence' if phase=='smoke' else 'same complete sample stream as historical control'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['numerical','smoke','full'])
    a=parser.parse_args()
    numerical() if a.mode=='numerical' else gate(a.mode)
    print('PASS',a.mode)
