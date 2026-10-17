import copy
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from eo_denoise.config import DEFAULT
from eo_denoise.models import CrossViewModel
from eo_denoise.var_codec import VarCodec
from eo_denoise.var_ops import interpolate
from eo_denoise.tokens import pack,unpack

torch.set_num_threads(2)


@pytest.mark.parametrize('source',[1,2,3,4,5,6,8,10,13])
def test_deterministic_cubic_matches_reference_forward_and_gradient(source):
    x=torch.randn(2,3,source,source,requires_grad=True)
    a=interpolate(x,(16,16),'bicubic')
    b=F.interpolate(x,size=(16,16),mode='bicubic',align_corners=False)
    torch.testing.assert_close(a,b,atol=4e-6,rtol=2e-5)
    upstream=torch.randn_like(a)
    ga=torch.autograd.grad(a,x,upstream,retain_graph=True)[0]
    gb=torch.autograd.grad(b,x,upstream)[0]
    torch.testing.assert_close(ga,gb,atol=2e-5,rtol=2e-5)


def small(backbone='var_conv'):
    c=copy.deepcopy(DEFAULT['model'])
    c.update(backbone=backbone,quantizer='var_vq',var_ch=32,latent_dim=4,codebook_size=32,
             eo_dim=48,eo_depth=1,eo_heads=4,eo_decoder_dim=32,eo_decoder_depth=1,eo_decoder_heads=4)
    return c


def upstream_class():
    root=Path(__file__).resolve().parents[1]/'experiments/paper1/ablation_vq_prithvi/sources/var'
    # Load the unchanged pinned source as an isolated package, not production code.
    from types import ModuleType
    package=ModuleType('upstream_var_test');package.__path__=[str(root/'models')]
    sys.modules['upstream_var_test']=package
    dist=ModuleType('dist');dist.initialized=lambda:False
    with patch.dict(sys.modules,{'dist':dist}):
        from upstream_var_test.vqvae import VQVAE
    return VQVAE


def test_actual_var_modules_match_unmodified_reference():
    reference=upstream_class()(vocab_size=32,z_channels=4,ch=32,test_mode=False)
    model=VarCodec(3,small())
    reference.encoder.conv_in=torch.nn.Conv2d(4,32,3,padding=1)
    reference.load_state_dict(model.network.state_dict())
    x=torch.randn(1,3,256,256);mask=torch.ones(1,1,256,256,dtype=torch.bool)
    mask[:,:,60:120,40:90]=False
    raw=torch.cat((torch.where(mask,x,0),mask.float()),1)
    with patch('torch.distributed.get_world_size',return_value=1):
        expected,_,loss=reference(raw)
    actual=model(x,mask)
    torch.testing.assert_close(actual['bottleneck_loss'],loss,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(actual['reconstruction'],expected,atol=2e-5,rtol=2e-5)
    assert actual['indices'].shape==(1,1,680)


@pytest.mark.parametrize('backbone',['var_conv','var_prithvi'])
def test_var_eo_mask_gradients_multiscale_roundtrip(backbone):
    model=VarCodec(1,small(backbone))
    image=torch.randn(1,1,256,256);mask=torch.ones_like(image,dtype=torch.bool)
    mask[:,:,30:60,40:100]=False
    out=model(image,mask)
    (out['reconstruction'].square().mean()+out['bottleneck_loss']).backward()
    assert model.quantizer.embedding.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        a=model(image,mask);image[~mask]=1e8;b=model(image,mask)
        torch.testing.assert_close(a['reconstruction'],b['reconstruction'],rtol=0,atol=0)
        blob=pack(a['indices'][0].numpy(),32,mask[0].numpy(),{'scales':list(model.scales)})
        ids,recovered,header=unpack(blob)
        torch.testing.assert_close(model.decode_indices(torch.from_numpy(ids[None])),a['reconstruction'],rtol=0,atol=0)
        assert np.array_equal(recovered,mask[0].numpy())
    assert model.rate()['tokens']==680
