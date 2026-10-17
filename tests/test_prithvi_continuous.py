import copy
import pytest
import torch
from eo_denoise.config import DEFAULT, validate
from eo_denoise.models import CrossViewModel


def settings():
    c=copy.deepcopy(DEFAULT)
    c['data'].update(modalities=['hls'],train_views=1)
    c['loss'].update(stability=0,cross_view=0)
    c['model'].update(backbone='prithvi_eo2_continuous',quantizer='ae',latent_dim=48,
        eo_dim=48,eo_depth=1,eo_heads=4,eo_decoder_dim=32,eo_decoder_depth=1,eo_decoder_heads=4)
    return c


def test_continuous_prithvi_has_no_codebook_and_preserves_reference_decoder():
    c=settings();validate(c)
    model=CrossViewModel(c).codecs['hls']
    assert not hasattr(model,'quantizer')
    assert not any(isinstance(m,(torch.nn.Embedding,torch.nn.Conv3d)) for m in model.modules())
    image=torch.randn(2,6,256,256)
    mask=torch.ones(2,1,256,256,dtype=torch.bool);mask[:,:,20:50,30:60]=False
    out=model(image,mask)
    assert out['indices'] is None and out['bottleneck_loss'].item()==out['kl'].item()==0
    assert out['continuous_latent'].shape==(2,257,48)
    out['reconstruction'].square().mean().backward()
    assert model.encoder.patch_embed.proj.weight.grad.abs().sum()>0
    assert model.decoder.decoder_embed.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        a=model(image,mask);changed=torch.where(mask,image,torch.full_like(image,1e8))
        b=model(changed,mask)
        torch.testing.assert_close(a['reconstruction'],b['reconstruction'],rtol=0,atol=0)
        native=model.decoder(a['continuous_latent'],torch.arange(256).expand(2,-1),input_size=[2,6,1,256,256])
        expected=native.reshape(2,16,16,16,16,6).permute(0,5,1,3,2,4).reshape(2,6,256,256)
        torch.testing.assert_close(a['reconstruction'],expected,rtol=0,atol=0)
        altered=a['continuous_latent'].clone();altered[:,1,0]+=.01
        assert (model.decode_latent(altered)-a['reconstruction']).abs().max()>0
    with pytest.raises(ValueError,match='no discrete'):
        model.decode_indices(torch.zeros(1,16,16,dtype=torch.long))
    c['model']['quantizer']='vq'
    with pytest.raises(ValueError,match='Continuous Prithvi'):
        validate(c)
