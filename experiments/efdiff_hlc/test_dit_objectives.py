import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dit_objectives import DiTObjective, masked_mse
from dit_model import PrithviCrossDiT


@pytest.mark.parametrize('objective',['flow','epsilon'])
def test_exact_targets_oracle_loss_and_sampler(objective):
    x0=torch.rand(2,1,8,8)*1.8-.9
    noise=torch.randn_like(x0)
    class Oracle(torch.nn.Module):
        calls=0
        target=None
        def forward(self,x,level,context=None):
            self.calls+=1
            if self.target is not None:return self.target
            if objective=='flow':return x0-noise
            a=level.reshape(-1,1,1,1)
            return (x[:,1:]-a*x0)/(1-a.square()).sqrt()
    oracle=Oracle();net=DiTObjective(oracle,None,objective)
    t=torch.tensor([.2,.8]) if objective=='flow' else torch.tensor([199,799])
    xt,target,level=net.training_pair(x0,noise,t)
    if objective=='flow':
        torch.testing.assert_close(target,x0-noise)
        for end,expected in [(0.,noise),(1.,x0)]:
            torch.testing.assert_close(net.training_pair(x0,noise,torch.full((2,),end))[0],expected)
        assert torch.isfinite(net.flow_level(torch.tensor([0.,1.]))).all()
    else: torch.testing.assert_close(target,noise)
    batch={'HR':x0,'SR':torch.zeros_like(x0),'mask':torch.ones_like(x0).bool()}
    oracle.target=target
    assert net(batch,t=t,eps=noise)==0
    oracle.target=None
    for steps in [1,5,10]:
        oracle.calls=0
        result=net.sample(batch,steps,noise)
        torch.testing.assert_close(result,x0,atol=1e-4,rtol=1e-4)
        assert oracle.calls==steps


def test_l2_equal_record_weight_and_mask():
    pred=torch.zeros(2,1,2,2,requires_grad=True)
    target=torch.tensor([[[[2.,20.],[20.,20.]]],[[[3.,3.],[3.,3.]]]])
    mask=torch.ones_like(target).bool();mask[0]=False;mask[0,0,0,0]=True
    loss=masked_mse(pred,target,mask)
    assert loss==6.5
    loss.backward();assert pred.grad[~mask].count_nonzero()==0


@pytest.mark.parametrize('objective',['flow','epsilon'])
def test_real_dit_gradients_and_sampling(objective):
    class Encoder(torch.nn.Module):
        def forward(self,hls): return hls
    torch.manual_seed(30)
    denoiser=PrithviCrossDiT(image_size=16,patch_size=4,hidden_dim=32,depth=2,heads=4,context_dim=24)
    net=DiTObjective(denoiser,Encoder(),objective)
    opt=torch.optim.Adam(denoiser.parameters(),lr=.001)
    b={'HR':torch.rand(2,1,16,16),'SR':torch.rand(2,1,16,16),'HLS':torch.randn(2,4,24),'mask':torch.ones(2,1,16,16).bool()}
    for _ in range(3):
        opt.zero_grad();loss=net(b);loss.backward();opt.step();assert torch.isfinite(loss)
    assert denoiser.blocks[0].cross_attention.kv.weight.grad.abs().sum()>0
    net.eval();result=net.sample(b,5,torch.randn_like(b['HR']))
    assert result.shape==b['HR'].shape and torch.isfinite(result).all()
