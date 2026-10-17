import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dit_timegeo_objectives import TimeGeoObjective
from vscore_diffusion import VScoreDiffusion, predictions
from prithvi_input_encoder import normalize_hls
from test_dit_timegeo import tiny


def test_v_training_target_and_oracle_sampling():
    x0=torch.rand(2,1,8,8)*1.5-.75
    class Oracle(torch.nn.Module):
        calls=0
        def forward(self,x,a,context=None,**kw):
            self.calls+=1
            assert kw['month'].tolist()==[1,7]
            assert kw['geo_latlon'].shape==(2,2)
            a=a.reshape(-1,1,1,1);s=(1-a*a).sqrt()
            noise=(x[:,1:]-a*x0)/s
            return a*noise-s*x0
    net=TimeGeoObjective(Oracle(),None,'v')
    noise=torch.randn_like(x0);t=torch.tensor([199,799])
    xt,v,a=net.training_pair(x0,noise,t);s=(1-net.alpha_bar[t]).sqrt().reshape(-1,1,1,1)
    torch.testing.assert_close(v,a*noise-s*x0)
    xx,ee=predictions(v,xt,a,s,'v')
    torch.testing.assert_close(xx,x0);torch.testing.assert_close(ee,noise)
    b=dict(HR=x0,SR=torch.zeros_like(x0),mask=torch.ones_like(x0).bool(),month=torch.tensor([1,7]),geo_latlon=torch.zeros(2,2))
    assert net(b,t,noise)<1e-10
    for steps in (1,5,20,50):
        net.denoise_fn.calls=0
        torch.testing.assert_close(net.sample(b,steps,noise),x0,atol=2e-4,rtol=2e-4)
        assert net.denoise_fn.calls==steps


def test_v_matches_frozen_v_implementation():
    class Predictor(torch.nn.Module):
        def forward(self,x,a,context=None,**kw): return x[:,1:]*.1+a.reshape(-1,1,1,1)*.03
    a,b=TimeGeoObjective(Predictor(),None,'v'),VScoreDiffusion(Predictor(),None,'v')
    batch=dict(HR=torch.rand(2,1,8,8),SR=torch.rand(2,1,8,8),mask=torch.ones(2,1,8,8).bool(),month=torch.tensor([1,7]),geo_latlon=torch.zeros(2,2))
    noise=torch.randn_like(batch['HR']);t=torch.tensor([10,900])
    torch.testing.assert_close(a(batch,t,noise),b(batch,t,noise))
    for steps in (5,20,50): torch.testing.assert_close(a.sample(batch,steps,noise),b.sample(batch,steps,noise))


def test_hls_unclipped_normalization_and_explicit_mask():
    mean=torch.tensor([1087,1342,1433,2734,1958,1363]).reshape(1,6,1,1)
    std=torch.tensor([2248,2179,2178,1850,1242,1049]).reshape(1,6,1,1)
    hls=torch.tensor([-.1,1.2,0.,float('nan')]).reshape(1,1,2,2).expand(1,6,2,2)
    valid=torch.tensor([True,True,True,False]).reshape(1,1,2,2)
    out=normalize_hls(hls,valid,mean,std)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:,:,:,0],((hls*10000-mean)/std)[:,:,:,0])
    torch.testing.assert_close(out[:,:,0,1],((hls*10000-mean)/std)[:,:,0,1])
    assert (out[:,:,1,1]==.0001).all()
    # Real zero reflectance remains a valid observation, not a nodata sentinel.
    torch.testing.assert_close(out[:,:,1,0],(-mean/std).reshape(1,6))


def test_v_corrected_encoder_receives_hls_mask_during_training_and_sampling():
    class Encoder(torch.nn.Module):
        requires_hls_valid=True
        calls=0
        def forward(self,x,valid):
            assert valid.dtype==torch.bool and valid.shape==(2,1,16,16)
            self.calls+=1
            return x
    encoder=Encoder();net=TimeGeoObjective(tiny(),encoder,'v')
    b=dict(HR=torch.rand(2,1,16,16),SR=torch.rand(2,1,16,16),mask=torch.ones(2,1,16,16).bool(),HLS=torch.randn(2,4,24),
           HLS_valid=torch.ones(2,1,16,16).bool(),month=torch.tensor([1,7]),geo_latlon=torch.zeros(2,2))
    opt=torch.optim.Adam(net.denoise_fn.parameters(),lr=.001)
    for _ in range(4):
        opt.zero_grad();loss=net(b);loss.backward();opt.step();assert torch.isfinite(loss)
    for enc in (net.denoise_fn.calendar_geo.month_encoder,net.denoise_fn.calendar_geo.geo_encoder):
        assert sum(p.grad.abs().sum() for p in enc.parameters())>0
    result=net.sample(b,5,torch.randn_like(b['HR']))
    assert torch.isfinite(result).all() and encoder.calls==5
