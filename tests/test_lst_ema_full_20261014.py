import copy,json,math
import numpy as np
import pytest
import torch
from experiments.paper1.lst_ema_full_20261014 import config as cfg,data,models,runtime,epoch_validation as ev
from experiments.paper1.lst_restoration_lossfix_20261014.config import default_config
from experiments.paper1.lst_ema_pilot_20261014.quantizers import EMAVQ,controllers
from experiments.paper1.lst_restoration_lossfix_20261014.quantizers import VectorQuantizer

class ToyDataset(torch.utils.data.Dataset):
    opened=[]
    def __init__(self,c,split,evaluation_profile=None,view=0):
        self.split=split;self.record_ids=np.arange(cfg.COUNTS[split]);self.fingerprint={'split':split,'profile':evaluation_profile}
        self.opened.append(split)
    def __len__(self):return len(self.record_ids)
    def __getitem__(self,item):
        epoch,i=item if isinstance(item,tuple) else (0,item)
        field=torch.full((1,2,2),.1*i);valid=torch.ones_like(field,dtype=torch.bool)
        if i==0:valid[0,0,0]=False
        return dict(sample_id=f'{self.split}-{i}',record_id=i,tile_id=str(i),temperature=torch.zeros_like(field),input_field=field,
            target_field=torch.zeros_like(field),reference_valid=valid,observed_valid=valid,mean_celsius=0.,center_celsius=0.,
            patch_missing=torch.zeros(1,1,dtype=torch.bool),profile='pixel10')
    def close(self):pass

class ToyModel(torch.nn.Module):
    def __init__(self,ema=False):
        super().__init__();self.bias=torch.nn.Parameter(torch.tensor(1.));self.sub=torch.nn.Dropout(.2)
        self.quantizer=EMAVQ(VectorQuantizer(1,4)) if ema else None
    def forward(self,encoded_input,**kwargs):
        z=encoded_input+self.bias
        if self.quantizer is not None:
            q=self.quantizer(z)
            return dict(encoded_prediction=q['quantized'],indices=q['indices'],latent=q['quantized'],prequantized_latent=z,bottleneck_loss=q['loss'])
        noise=torch.randn_like(z)*.01 if self.training else 0.
        return dict(encoded_prediction=z+noise)

@pytest.fixture
def tiny(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(cfg,'COUNTS',{'train':5,'val':3,'test':2})
    monkeypatch.setattr(data,'LSTDataset',ToyDataset)
    monkeypatch.setattr(models,'build_model',lambda c:ToyModel(c['ema_method']=='ema'))
    monkeypatch.setattr(runtime,'execution_guard',lambda c:None)
    monkeypatch.setattr(runtime,'source_hashes',lambda:{'test':'fixed'})
    ToyDataset.opened=[]
    c=default_config('vq','R');c['ema_method']='baseline';c['epoch_validation']=ev.default_protocol()
    c['train'].update(device='cpu',workers=0,effective_batch=4,micro_batch=2,epochs=5,updates=10,
        record_presentations=25,lr=.01,checkpoint_every=1,amp=False)
    c['evaluation']['batch_size']=2;c['task']['residual_std_celsius']=2.;c['representation']['A_celsius']=1.
    c['model']['codebook_size']=4
    return c

def test_full_val_weighting_and_two_views(tiny,tmp_path):
    m=ToyModel();m.sub.eval();states=[x.training for x in m.modules()];rng=torch.get_rng_state()
    cells=ev.measure_validation(m,tiny,torch.device('cpu'),tmp_path)
    for cell in cells:
        assert cell['raw']['pixel_weighted_rmse']==pytest.approx(math.sqrt((3+4*1.21+4*1.44)/11))
        assert cell['loss_pixel']==pytest.approx((1+1.21+1.44)/12)
        assert cell['row_count']==6 and cell['valid_pixels']==22
    assert states==[x.training for x in m.modules()] and torch.equal(rng,torch.get_rng_state())
    assert ToyDataset.opened==['val']*4

@pytest.mark.parametrize('method',['baseline','ema'])
@pytest.mark.parametrize('failure',['none','mid_epoch','validation'])
def test_exact_tail_budget_and_resume(tiny,tmp_path,monkeypatch,method,failure):
    tiny['ema_method']=method
    if method=='ema':
        # Tiny test latent is 2x2; production retains 16x16.
        from experiments.paper1.lst_ema_full_20261014.usage import Usage
        usage_init=Usage.__init__
        def tiny_usage(self,c):usage_init(self,c);self.scales=[2]
        monkeypatch.setattr(Usage,'__init__',tiny_usage)
    full=tmp_path/'full';resumed=tmp_path/'resume';runtime.train(tiny,full)
    if failure!='none':
        module,name=(runtime,'train_update') if failure=='mid_epoch' else (ev,'measure_validation')
        original=getattr(module,name);calls=0
        def fail(*a,**kw):
            nonlocal calls
            calls+=1
            if calls==4:raise RuntimeError('injected interruption')
            return original(*a,**kw)
        monkeypatch.setattr(module,name,fail)
        with pytest.raises(RuntimeError,match='injected'):runtime.train(tiny,resumed)
        monkeypatch.setattr(module,name,original)
    runtime.train(tiny,resumed)
    a=torch.load(full/'last.pt',weights_only=True);b=torch.load(resumed/'last.pt',weights_only=True)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
    assert torch.equal(a['rng_cpu'],b['rng_cpu'])
    assert (b['epoch'],b['records'],b['step'])==(5,25,10)
    assert len(list((resumed/'epochs').glob('*.pt')))==5
    assert len(list((resumed/'epoch_validation').glob('*.json')))==5
    logs=[json.loads(l) for p in full.glob('train_*.jsonl') for l in p.read_text().splitlines()]
    assert [x['actual_batch'] for x in logs]==[4,1]*5
    if method=='ema':assert int(b['model']['quantizer.ema.steps'])==10
    assert 'test' not in ToyDataset.opened
    selection=json.loads((resumed/'selection.json').read_text())
    assert cfg.file_digest(resumed/'best.pt')==selection['best']['checkpoint_sha256']
    for cell in json.loads((resumed/'epoch_validation/epoch_05.json').read_text())['cells']:
        assert cell['row_count']==6

def test_failed_validation_cannot_advance(tiny,tmp_path,monkeypatch):
    monkeypatch.setattr(ev,'measure_validation',lambda *a:(_ for _ in ()).throw(RuntimeError('val failed')))
    for _ in range(2):
        with pytest.raises(RuntimeError,match='val failed'):runtime.train(tiny,tmp_path)
        state=torch.load(tmp_path/'last.pt',weights_only=True)
        assert state['epoch']==1 and state['step']==2
        assert not (tmp_path/'epoch_validation/epoch_01.json').exists()

def test_hlp_sampler_complete_unique_tail():
    n=218975
    for epoch in range(5):
        order=list(data.EpochSampler(n,17,epoch))
        assert len(order)==n and len(set(i for _,i in order))==n
        assert all(e==epoch for e,_ in order)
        assert len(order[-(n%64):])==31 and math.ceil(n/64)==3422
        assert list(data.EpochSampler(n,17,epoch,64))==order[64:]

def test_tail_gradient_is_sample_weighted(tiny):
    from torch.utils.data import default_collate
    class Linear(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(.5))
        def forward(self,encoded_input,**kw):return {'encoded_prediction':encoded_input*self.w}
    samples=[ToyDataset(tiny,'train')[i] for i in range(5)]
    a=Linear();b=copy.deepcopy(a);oa=torch.optim.SGD(a.parameters(),lr=.01);ob=torch.optim.SGD(b.parameters(),lr=.01)
    from experiments.paper1.lst_ema_full_20261014.representations import Representation
    rep=Representation(tiny['representation'])
    runtime.train_update(a,[default_collate(samples[:2]),default_collate(samples[2:4]),default_collate(samples[4:])],oa,rep,tiny,torch.device('cpu'))
    runtime.train_update(b,[default_collate(samples)],ob,rep,tiny,torch.device('cpu'))
    torch.testing.assert_close(a.w,b.w)
