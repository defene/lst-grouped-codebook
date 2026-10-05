import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from eo_denoise.var_quant import VectorQuantizer2
from eo_denoise.config import DEFAULT
from eo_denoise.var_codec import VarCodec
from experiments.paper1.lst_grouped_vq.quantizer import GroupedQuantizer, GroupedCodec
from experiments.paper1.lst_grouped_vq.metrics import EvaluationMetrics
from experiments.paper1.lst_residual.runtime import ResidualCodec

torch.set_num_threads(2)


def native(c=32, k=32):
    return VectorQuantizer2(k,c,False,v_patch_nums=(1,2,4,8,16),quant_resi=.5,share_quant_resi=4)


def test_single_group_matches_native_values_losses_and_gradients():
    torch.manual_seed(17)
    a = native()
    b = GroupedQuantizer(copy.deepcopy(a),1,32)
    x = torch.randn(2,32,16,16,requires_grad=True)
    y = x.detach().clone().requires_grad_()
    qa,_,la = a(x)
    qb,lb,ids = b(y)
    torch.testing.assert_close(qa,qb)
    torch.testing.assert_close(la,lb)
    (qa.square().mean()+la).backward()
    (qb.square().mean()+lb).backward()
    torch.testing.assert_close(x.grad,y.grad)
    torch.testing.assert_close(a.embedding.weight.grad,b.embeddings[0].weight.grad)
    for pa,pb in zip(a.quant_resi.parameters(),b.quant_resi.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad)
    expected = a.f_to_idxBl_or_fhat(x.detach(),False)
    assert all(torch.equal(i,j[:,0]) for i,j in zip(expected,ids))


@pytest.mark.parametrize('groups,codes',[(8,16),(8,256)])
def test_grouped_quantization_decode_gradients_and_serialization(groups,codes):
    q = GroupedQuantizer(native(),groups,codes)
    z = torch.randn(2,32,16,16,requires_grad=True)
    out,loss,ids = q(z)
    (out.square().mean()+loss).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    for e in q.embeddings:
        assert torch.isfinite(e.weight.grad).all() and e.weight.grad.abs().sum()>0
    q.eval()
    with torch.no_grad():
        out,_,ids = q(z)
        torch.testing.assert_close(q.decode(torch.cat(ids,2)),out,rtol=0,atol=0)
        restored = GroupedQuantizer(native(),groups,codes).eval()
        restored.load_state_dict(q.state_dict())
        torch.testing.assert_close(restored(z)[0],out,rtol=0,atol=0)
    with pytest.raises(ValueError):
        q.decode(torch.full((1,groups,341),codes,dtype=torch.long))


def config(groups,codes):
    c = copy.deepcopy(DEFAULT['model'])
    c.update(backbone='var_conv',quantizer='var_vq' if groups==1 else 'grouped_var_vq',var_ch=32,
             latent_dim=32,codebook_size=1024,groups=groups,codes_per_group=codes)
    return c


@pytest.mark.parametrize('groups,codes,bits',[(8,16,21792),(8,256,43552)])
def test_codec_mask_invariance_roundtrip_and_rate(groups,codes,bits):
    codec = ResidualCodec((VarCodec if groups==1 else GroupedCodec)(1,config(groups,codes))).eval()
    image = torch.randn(1,1,256,256)
    mask = torch.ones_like(image,dtype=torch.bool)
    mask[:,:,40:90,60:150] = False
    with torch.no_grad():
        a = codec(image,mask)
        image[~mask] = 1e8
        b = codec(image,mask)
        torch.testing.assert_close(a['reconstruction'],b['reconstruction'],rtol=0,atol=0)
        torch.testing.assert_close(codec.decode_indices(a['indices'],a['temperature_baseline']),a['reconstruction'],rtol=0,atol=0)
    assert codec.rate()['bits_per_image']==bits
    assert a['indices'].shape==(1,groups,680)


def test_shared_modules_start_identically():
    torch.manual_seed(17)
    base = VarCodec(1,config(1,1024))
    for g,k in [(8,16),(8,256)]:
        torch.manual_seed(17)
        grouped = GroupedCodec(1,config(g,k))
        d = grouped.state_dict()
        for name,value in base.state_dict().items():
            if not name.startswith('network.quantize.') or name.startswith('network.quantize.quant_resi.'):
                torch.testing.assert_close(value,d[name],rtol=0,atol=0)


def test_group_metrics_do_not_mix_dictionaries_and_measure_within_image():
    c = config(2,32)
    c['var_scales'] = [1,2]
    fake = SimpleNamespace(codecs={'lst':SimpleNamespace(config=c)})
    metrics = EvaluationMetrics({'lst':SimpleNamespace(stats={'std':[100.],'mean':[0.]})},fake)
    mask = torch.ones(2,1,2,2,dtype=torch.bool)
    # Both images spatially constant but have different codes; pooled entropy > within-image entropy.
    grid = torch.stack([torch.zeros(2,2,2,dtype=torch.long),torch.ones(2,2,2,dtype=torch.long)])
    output = {'reconstruction':torch.zeros(2,1,2,2),'indices_by_scale':[grid[:,:,:1,:1],grid]}
    data = dict(target=torch.zeros(2,1,2,2),reference_mask=mask,
                input_mask=mask[:,None],corruption_mask=torch.zeros_like(mask[:,None]),actual_coverage=torch.zeros(2,1))
    metrics.add('lst',[output],data,['a','b'],['t1','t2'])
    summary = metrics.summary(10)['tokens']['lst']
    assert len(summary['per_group'])==2
    for group in summary['per_group']:
        assert group['used_codes']==2 and group['perplexity']==2
        assert group['scales'][-1]['within_image_mean_entropy_bits']==0
        assert group['scales'][-1]['within_image_mean_unique_codes']==1
        assert group['scales'][-1]['within_image_adjacent_change_fraction']==0
    assert metrics.counts.sum()==20
