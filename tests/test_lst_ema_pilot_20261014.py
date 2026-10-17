import copy
import pytest
import torch
from experiments.paper1.lst_ema_pilot_20261014.quantizers import (
    DeferredEMA,EMAVQ,EMAGrouped,controllers,initialize_from_latents,kmeans)
from experiments.paper1.lst_restoration_lossfix_20261014.quantizers import VectorQuantizer
from experiments.paper1.lst_restoration_lossfix_20261014.grouped_adapter import GroupedProductQuantizer


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(17);torch.set_num_threads(1)


def test_ema_update_is_centroid_and_only_at_boundary():
    e=torch.nn.Embedding(2,2);q=DeferredEMA(e)
    old=e.weight.detach().clone();q.begin()
    q.accumulate(torch.tensor([[2.,4.],[4.,6.]]),torch.tensor([0,0]))
    assert torch.equal(e.weight,old)
    q.accumulate(torch.tensor([[8.,10.]]),torch.tensor([1]));q.commit()
    torch.testing.assert_close(e.weight,torch.tensor([[3.,5.],[8.,10.]]))
    assert not e.weight.requires_grad


def test_microbatch_partition_invariance():
    a=DeferredEMA(torch.nn.Embedding(4,3));b=DeferredEMA(torch.nn.Embedding(4,3))
    b.embedding.weight.data.copy_(a.embedding.weight)
    x=torch.randn(24,3);ids=torch.arange(24)%4
    a.begin();b.begin();a.accumulate(x,ids)
    for i in range(0,24,4):b.accumulate(x[i:i+4],ids[i:i+4])
    a.commit();b.commit()
    torch.testing.assert_close(a.embedding.weight,b.embedding.weight)


def test_kmeans_and_eval_do_not_change_usage_or_model():
    q=EMAVQ(VectorQuantizer(4,8),True);x=torch.randn(2,4,4,4)
    initialize_from_latents(q,x)
    assert q.ema.initialized and q.ema.steps==0
    q.eval();state=copy.deepcopy(q.state_dict());a=q(x)
    assert torch.equal(a['quantized'],q.decode_indices(a['indices']))
    assert all(torch.equal(v,q.state_dict()[k]) for k,v in state.items())


def test_vq_reconstruction_grad_and_optimizer_exclusion():
    q=EMAVQ(VectorQuantizer(4,8));x=torch.randn(2,4,4,4,requires_grad=True)
    q.ema.begin();r=q(x);(r['quantized'].square().mean()+r['loss']).backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum()>0
    assert q.embedding.weight.grad is None
    q.ema.commit()


def test_grouped_matches_native_indices_forward_loss_and_phi_gradient_before_update():
    c=dict(scales=[1,2,4],groups=2,codebook_size=8,latent_dim=4,commitment=.25,quant_resi=.5,share_quant_resi=4)
    original=GroupedProductQuantizer(c);ema=EMAGrouped(copy.deepcopy(original))
    x=torch.randn(2,4,4,4,requires_grad=True);y=x.detach().clone().requires_grad_()
    for e in ema.emas:e.begin()
    before=[e.weight.detach().clone() for e in ema.grouped.embeddings]
    a=original(x);b=ema(y)
    assert torch.equal(a['indices'],b['indices'])
    torch.testing.assert_close(a['quantized'],b['quantized'],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(a['loss'],b['loss'])
    (b['quantized'].square().mean()+b['loss']).backward()
    assert y.grad.isfinite().all() and y.grad.abs().sum()>0
    (a['quantized'].square().mean()+a['loss']).backward()
    torch.testing.assert_close(x.grad,y.grad)
    for p,q in zip(original.grouped.quant_resi.parameters(),ema.grouped.quant_resi.parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad)
    assert all(torch.equal(w,e.weight) for w,e in zip(before,ema.grouped.embeddings))
    assert all(int(e.pending_mass.sum())==2*(1+4+16) for e in ema.emas)
    for e in ema.emas:e.commit()
    ema.eval();out=ema(y)
    assert torch.equal(out['quantized'],ema.decode_indices(out['indices']))


def test_dead_reset_updates_ema_state_and_roundtrips_checkpoint():
    q=EMAVQ(VectorQuantizer(2,4),True)
    q.ema.initialize(torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]]))
    for _ in range(16):
        q.ema.begin();q.ema.accumulate(torch.tensor([[9.,10.]]),torch.tensor([0]));q.ema.commit()
    assert q.ema.reset_count==3
    assert torch.isfinite(q.embedding.weight).all()
    for i in range(1,4):torch.testing.assert_close(q.ema.sums[i]/q.ema.mass[i],q.embedding.weight[i])
    new=EMAVQ(VectorQuantizer(2,4),True);new.load_state_dict(q.state_dict())
    for obj in (q,new):
        obj.ema.begin();obj.ema.accumulate(torch.tensor([[8.,9.]]),torch.tensor([1]));obj.ema.commit()
    for k,v in q.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k])


def test_kmeans_does_not_mutate_training_rng():
    x=torch.randn(64,4);before=torch.get_rng_state()
    centers=kmeans(x,8)
    assert torch.equal(before,torch.get_rng_state()) and centers.shape==(8,4)
