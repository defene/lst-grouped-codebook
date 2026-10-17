"""Behavior checks: periodicity, geometry, condition influence, and sampler parity."""
import copy
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dit_model import PrithviCrossDiT
from dit_objectives import DiTObjective
from dit_timegeo_model import CalendarGeoCondition, TimeGeoDiT
from dit_timegeo_objectives import TimeGeoObjective
from timegeo_metadata import crop_center


def tiny(checkpointing=False):
    return TimeGeoDiT(image_size=16, patch_size=4, hidden_dim=32, depth=2,
                     heads=4, context_dim=24, condition_dim=8, gradient_checkpointing=checkpointing)


def activate(net):
    # adaLN-Zero intentionally masks metadata influence at initialization.
    with torch.no_grad():
        net.head.weight.normal_(std=.05)
        for block in net.blocks:
            block.modulation[-1].weight.normal_(std=.05)
        net.final_modulation[-1].weight.normal_(std=.05)


def test_calendar_wrap_and_geo_seams():
    c = CalendarGeoCondition(32,8)
    m = c.month_features(torch.tensor([12,1,2]))
    torch.testing.assert_close((m[0]-m[1]).square().sum(), (m[1]-m[2]).square().sum())
    geo = c.geo_features(torch.tensor([[30.,-180.],[30.,180.],[90.,-30.],[90.,120.]]))
    torch.testing.assert_close(geo[0],geo[1],atol=3e-6,rtol=1e-5)
    torch.testing.assert_close(geo[2],geo[3],atol=3e-6,rtol=1e-5)
    assert not torch.allclose(c.geo_features(torch.tensor([[30.,20.]])),c.geo_features(torch.tensor([[-30.,20.]])))


@pytest.mark.parametrize('month',[torch.tensor([0]),torch.tensor([13]),torch.tensor([2.5]),torch.tensor([float('nan')])])
def test_invalid_month_rejected(month):
    with pytest.raises(ValueError): CalendarGeoCondition(32,8).month_features(month)


@pytest.mark.parametrize('geo',[[91.,0.],[0.,181.],[float('nan'),0.]])
def test_invalid_geo_rejected(geo):
    with pytest.raises(ValueError): CalendarGeoCondition(32,8).geo_features(torch.tensor([geo]))


def test_center_uses_full_affine_and_crop_extent():
    assert crop_center([30,0,300000,0,-30,7400040],0,0) == (303840,7396200)
    assert crop_center([2,3,10,4,5,20],256,512) == (2698,4756)


def test_zero_init_missing_metadata_and_old_weights_rejected():
    net = tiny()
    x,a,ctx = torch.randn(2,2,16,16),torch.tensor([.2,.8]),torch.randn(2,4,24)
    with pytest.raises(ValueError,match='requires'): net(x,a,ctx)
    assert net(x,a,ctx,month=torch.tensor([1,7]),geo_latlon=torch.tensor([[30.,20.],[-30.,20.]])).count_nonzero()==0
    old = PrithviCrossDiT(image_size=16,patch_size=4,hidden_dim=32,depth=2,heads=4,context_dim=24)
    with pytest.raises(RuntimeError,match='Missing key'): net.load_state_dict(old.state_dict())


def test_each_condition_changes_output_and_receives_gradient():
    torch.manual_seed(41)
    net = tiny(); activate(net)
    x,a,ctx = torch.randn(2,2,16,16),torch.tensor([.2,.8]),torch.randn(2,4,24)
    month,geo = torch.tensor([1,7]),torch.tensor([[30.,20.],[-30.,20.]])
    y = net(x,a,ctx,month=month,geo_latlon=geo)
    assert not torch.allclose(y,net(x,a,ctx,month=month+1,geo_latlon=geo))
    assert not torch.allclose(y,net(x,a,ctx,month=month,geo_latlon=geo+10))
    assert not torch.allclose(y,net(x,1-a,ctx,month=month,geo_latlon=geo))
    y.square().mean().backward()
    for group in (net.time_mlp,net.calendar_geo.month_encoder,net.calendar_geo.geo_encoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters())
        assert sum(p.grad.abs().sum() for p in group.parameters()) > 0
    for block in net.blocks:
        assert block.modulation[-1].weight.grad.abs().sum() > 0


def test_checkpointing_preserves_metadata_gradients():
    torch.manual_seed(2)
    plain = tiny(); activate(plain)
    checked = copy.deepcopy(plain); checked.gradient_checkpointing = True
    args=(torch.randn(2,2,16,16),torch.tensor([.2,.8]),torch.randn(2,4,24))
    kw=dict(month=torch.tensor([1,7]),geo_latlon=torch.tensor([[30.,20.],[-30.,20.]]))
    y,z=plain(*args,**kw),checked(*args,**kw)
    torch.testing.assert_close(y,z)
    y.square().mean().backward(); z.square().mean().backward()
    for p,q in zip(plain.parameters(),checked.parameters()): torch.testing.assert_close(p.grad,q.grad)


@pytest.mark.parametrize('objective',['flow','epsilon'])
def test_objective_and_sampler_equal_legacy_when_predictor_equal(objective):
    class Predictor(torch.nn.Module):
        def __init__(self): super().__init__(); self.seen=[]
        def forward(self,x,a,context=None,**metadata):
            self.seen.append(metadata)
            return x[:,1:]*.1+a.reshape(-1,1,1,1)*.03
    p,q=Predictor(),Predictor()
    old,new=DiTObjective(p,None,objective),TimeGeoObjective(q,None,objective)
    batch=dict(HR=torch.rand(2,1,8,8),SR=torch.rand(2,1,8,8),mask=torch.ones(2,1,8,8).bool(),
               month=torch.tensor([1,7]),geo_latlon=torch.tensor([[30.,20.],[-30.,20.]]))
    noise=torch.randn_like(batch['HR'])
    t=torch.tensor([.2,.8]) if objective=='flow' else torch.tensor([199,799])
    torch.testing.assert_close(old(batch,t,noise),new(batch,t,noise))
    for steps in (1,5,10): torch.testing.assert_close(old.sample(batch,steps,noise),new.sample(batch,steps,noise))
    assert len(q.seen)==17
    for kw in q.seen:
        assert torch.equal(kw['month'],batch['month']) and torch.equal(kw['geo_latlon'],batch['geo_latlon'])


@pytest.mark.parametrize('objective',['flow','epsilon'])
def test_train_updates_and_sample_with_metadata(objective):
    class Encoder(torch.nn.Module):
        def forward(self,hls): return hls
    torch.manual_seed(3)
    net=TimeGeoObjective(tiny(),Encoder(),objective)
    opt=torch.optim.Adam(net.denoise_fn.parameters(),lr=.001)
    b=dict(HR=torch.rand(2,1,16,16),SR=torch.rand(2,1,16,16),HLS=torch.randn(2,4,24),
           mask=torch.ones(2,1,16,16).bool(),month=torch.tensor([1,7]),geo_latlon=torch.tensor([[30.,20.],[-30.,20.]]))
    for _ in range(4):
        opt.zero_grad(); loss=net(b); loss.backward(); opt.step(); assert torch.isfinite(loss)
    for encoder in (net.denoise_fn.calendar_geo.month_encoder,net.denoise_fn.calendar_geo.geo_encoder):
        assert sum(p.grad.abs().sum() for p in encoder.parameters()) > 0
    net.eval()
    pred=net.sample(b,5,torch.randn_like(b['HR']))
    assert pred.shape==b['HR'].shape and torch.isfinite(pred).all()
