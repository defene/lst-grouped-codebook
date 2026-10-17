"""Local checks: budget, spatial mask, cross-modal gradients and VP compatibility."""
import json
import sys
from pathlib import Path
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dit_timegeo_model import TimeGeoDiT
from raw_mmdit_model import RawHLSMMDiT
from raw_mmdit_objective import RawMMDiTObjective
from dit_timegeo_objectives import TimeGeoObjective


def tiny(checkpointing=False):
    return RawHLSMMDiT(image_size=16, patch_size=4, hidden_dim=32, heads=4,
                      encoder_depth=1, depth=2, ffn_hidden=64,
                      gradient_checkpointing=checkpointing)


def batch():
    return dict(SR=torch.randn(2,1,16,16), HR=torch.rand(2,1,16,16)*1.5-.75,
                mask=torch.ones(2,1,16,16,dtype=torch.bool),
                HLS=torch.rand(2,6,16,16), HLS_valid=torch.ones(2,1,16,16,dtype=torch.bool),
                month=torch.tensor([1,7]), geo_latlon=torch.tensor([[30.,110.],[-35.,-70.]]))


def open_gates(model):
    # Zero gates/head intentionally block upstream gradients at initialization.
    # Open them to check the full trained computation graph without a long fit.
    with torch.no_grad():
        model.head.weight.normal_(std=.02)
        for block in model.blocks:
            for stream in (block.condition, block.temperature):
                stream.modulation[-1].weight.normal_(std=.02)
        model.final_modulation[-1].weight.normal_(std=.02)


def test_real_parameter_budget():
    cfg=json.loads((HERE/'dit_timegeo_v_plan.json').read_text())['architecture']
    with torch.device('meta'):
        old, new=TimeGeoDiT(**cfg), RawHLSMMDiT()
    a=sum(p.numel() for p in old.parameters() if p.requires_grad)
    b=sum(p.numel() for p in new.parameters() if p.requires_grad)
    assert a == 216994944
    assert abs(b/a-1) < .001
    assert all(p.requires_grad for p in new.parameters())


def test_full_mask_geometry_and_invalid_fill():
    torch.manual_seed(0)
    model=tiny().eval()
    b=batch(); hls=torch.zeros_like(b['HLS'])
    left=b['HLS_valid'].clone(); left[:,:,:,::2]=False
    right=~left
    c1=model.encode_condition(b['SR'],hls,left)
    c2=model.encode_condition(b['SR'],hls,right)
    assert left.sum()==right.sum() and not torch.allclose(c1,c2)
    garbage=hls.masked_fill(~left.expand_as(hls),float('nan'))
    torch.testing.assert_close(c1,model.encode_condition(b['SR'],garbage,left))
    with pytest.raises(ValueError,match='boolean'):
        model.encode_condition(b['SR'],hls,left.float())
    with pytest.raises(ValueError,match='Nonfinite'):
        model.encode_condition(b['SR'],garbage,right)


@pytest.mark.parametrize('checkpointing',[False,True])
def test_full_graph_gradients_and_v_target(checkpointing):
    torch.manual_seed(1)
    model=tiny(checkpointing); open_gates(model)
    net=RawMMDiTObjective(model)
    b=batch(); b['HLS'].requires_grad_(); b['SR'].requires_grad_(); b['geo_latlon'].requires_grad_()
    t=torch.tensor([100,800]); eps=torch.randn_like(b['HR'])
    xt,target,a=net.training_pair(b['HR'],eps,t)
    s=(1-net.alpha_bar[t]).sqrt().reshape(-1,1,1,1)
    torch.testing.assert_close(target,a*eps-s*b['HR'])
    loss=net(b,t,eps); loss.backward()
    assert torch.isfinite(loss)
    missing=[name for name,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0]
    assert not missing, missing
    for key in ('HLS','SR','geo_latlon'):
        assert b[key].grad.abs().sum()>0
    assert net.prithvi_encoder is None


def test_cache_equivalence_and_no_cross_step_mutation():
    torch.manual_seed(2)
    model=tiny().eval(); open_gates(model)
    b=batch(); x=torch.cat((b['SR'],torch.randn_like(b['HR'])),1)
    a=torch.tensor([.3,.8]); kw=dict(month=b['month'],geo_latlon=b['geo_latlon'])
    with torch.no_grad():
        c=model.encode_condition(b['SR'],b['HLS'],b['HLS_valid']); saved=c.clone()
        direct=model(x,a,hls=b['HLS'],hls_valid=b['HLS_valid'],**kw)
        cached=model(x,a,context=c,**kw)
        torch.testing.assert_close(c,saved)
        torch.testing.assert_close(direct,cached)
        changed=model(x,a,context=c,month=torch.tensor([6,12]),geo_latlon=b['geo_latlon'])
        assert not torch.allclose(cached,changed)
        # Reordering the batch must not mix examples.
        reversed_output=model(x.flip(0),a.flip(0),context=c.flip(0),month=b['month'].flip(0),geo_latlon=b['geo_latlon'].flip(0))
        torch.testing.assert_close(cached,reversed_output.flip(0),atol=1e-6,rtol=1e-5)


def test_sampler_caches_encoder_once_and_preserves_v_protocol():
    model=tiny().eval(); open_gates(model); net=RawMMDiTObjective(model)
    b=batch(); noise=torch.randn_like(b['HR']); calls=[]
    hook=model.condition_embed.register_forward_hook(lambda *args: calls.append(1))
    result=net.sample(b,5,noise)
    hook.remove()
    assert len(calls)==1 and result.shape==b['HR'].shape and torch.isfinite(result).all()
    class Uncached(TimeGeoObjective):
        def encode_batch(self,batch): return None
        def denoise_conditioned(self,xt,batch,level,context):
            return self.denoise_fn(torch.cat((batch['SR'],xt),1),level,
                                   hls=batch['HLS'],hls_valid=batch['HLS_valid'],
                                   month=batch['month'],geo_latlon=batch['geo_latlon'])
    torch.testing.assert_close(result,Uncached(model,None,'v').sample(b,5,noise))


def test_production_resolution_forward():
    # Full default model, actual 224-pixel grid. CPU only; no remote GPU job.
    model=RawHLSMMDiT(gradient_checkpointing=False).eval()
    with torch.inference_mode():
        y=model(torch.zeros(1,2,224,224),torch.tensor([.5]),
                hls=torch.zeros(1,6,224,224),hls_valid=torch.ones(1,1,224,224,dtype=torch.bool),
                month=torch.tensor([1]),geo_latlon=torch.zeros(1,2))
    assert y.shape==(1,1,224,224) and torch.isfinite(y).all()
    assert torch.count_nonzero(y)==0  # adaLN-Zero output initialization


def test_zero_initialization_can_start_learning():
    torch.manual_seed(3)
    model=tiny(); net=RawMMDiTObjective(model)
    b=batch(); t=torch.tensor([200,700]); noise=torch.randn_like(b['HR'])
    original=model.condition_embed.weight.detach().clone()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        loss=net(b,t,noise)
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
    assert not torch.equal(original,model.condition_embed.weight)
    assert model.metadata.mlp[0].weight.grad.abs().sum()>0
