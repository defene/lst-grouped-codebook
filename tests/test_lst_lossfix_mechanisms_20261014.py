"""CPU mechanism regressions on synthetic tensors; no datasets or GPU jobs."""
import copy
import json
from pathlib import Path
import pytest
import torch
from experiments.paper1.lst_restoration_lossfix_20261014.representations import Representation
from experiments.paper1.lst_restoration_lossfix_20261014.stable_vq import SphericalVectorQuantizer
from experiments.paper1.lst_restoration_lossfix_20261014.stage2_models import DriftingVectorQuantizer
from experiments.paper1.lst_restoration_lossfix_20261014.stage2_audit import energy_checks


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(71014)


@pytest.mark.parametrize('kind', ['C', 'C_power', 'C_asinh'])
def test_continuous_inverse_roundtrip_and_local_gradient(kind):
    r = Representation(dict(kind=kind, colour_inverse='trilinear_v1'))
    fields = torch.linspace(-120., 120., 20001).reshape(1,1,1,-1)
    assert (r.decode(r.encode(fields))-fields).abs().max() < .05
    x = torch.tensor([-.7, -.2, .1, .8, 9.]).reshape(1,1,1,-1).requires_grad_()
    grad = torch.autograd.grad(r.decode(r.encode(x)).sum(), x)[0]
    torch.testing.assert_close(grad, torch.ones_like(grad), atol=5e-4, rtol=5e-4)


def test_old_discontinuity_counterexample_is_removed():
    r = Representation(dict(kind='C', colour_inverse='trilinear_v1'))
    old = Representation(dict(kind='C'))
    a = torch.zeros(1,3,1,1)
    b = torch.tensor([.0000609989,.0001076061,.00004200344]).reshape_as(a)
    assert abs(float(old.decode(b)-old.decode(a))) > 200
    assert abs(float(r.decode(b)-r.decode(a))) < .02


@pytest.mark.parametrize('kind', ['C', 'C_power', 'C_asinh'])
def test_cube_boundaries_finite_bounded_and_escape_gradient(kind):
    r = Representation(dict(kind=kind, colour_inverse='trilinear_v1', inverse_gradient_policy='saturated_softpath_v1'))
    x = torch.tensor([[3.,-3.,-3.],[-3.,3.,3.],[3.,3.,3.],[-3.,-3.,-3.]]).reshape(4,3,1,1).requires_grad_()
    y = r.decode(x)
    assert y.isfinite().all() and y.abs().max() <= 120.01
    grad = torch.autograd.grad(y.square().sum(), x)[0]
    assert grad.isfinite().all() and (grad.abs().flatten(1).sum(1)>0).all()
    assert torch.equal(y.detach(), r.decode(x.detach()))


def test_trilinear_gradient_matches_finite_difference_inside_cube():
    r = Representation(dict(kind='C', colour_inverse='trilinear_v1'))
    x = torch.tensor([.15,-.27,.31]).reshape(1,3,1,1).requires_grad_()
    g = torch.autograd.grad(r.decode(x).sum(),x)[0]
    for i in range(3):
        step = torch.zeros_like(x);step[:,i]=1e-3
        fd = (r.decode(x.detach()+step)-r.decode(x.detach()-step))/(2e-3)
        torch.testing.assert_close(g[:,i:i+1],fd,atol=.01,rtol=.002)


def test_spherical_vq_train_init_replay_scale_gradients_and_resume():
    q = SphericalVectorQuantizer(4,16)
    x = (100*torch.randn(2,4,4,4)).requires_grad_()
    q.eval(); before=copy.deepcopy(q.state_dict()); q(x)
    assert all(torch.equal(v,q.state_dict()[k]) for k,v in before.items())
    q.train();o=q(x)
    assert q.initialized_from_train and (q.usage_counts > 0).sum() == 16
    assert torch.equal(o['quantized'],q.decode_indices(o['indices']))
    assert o['loss'] <= 5.0001  # unit-RMS squared distance <= 4; beta=.25
    g=torch.autograd.grad((o['quantized']*torch.randn_like(x)).mean(),x,retain_graph=True)[0]
    assert g.isfinite().all() and g.abs().sum()>0
    o['loss'].backward()
    assert q.embedding.weight.grad.isfinite().all()
    restored=SphericalVectorQuantizer(4,16);restored.load_state_dict(q.state_dict())
    q.eval();restored.eval()
    assert torch.equal(q(x)['quantized'],restored(x)['quantized'])
    weight=restored.embedding.weight.detach().clone()
    restored.train();restored(x*10)
    assert torch.equal(weight,restored.embedding.weight)


def test_spherical_vq_zero_tokens_and_small_batch():
    q=SphericalVectorQuantizer(4,16)
    x=torch.zeros(1,4,1,1,requires_grad=True)
    o=q(x);(o['loss']+o['quantized'].sum()).backward()
    assert x.grad.isfinite().all() and o['loss'].isfinite()


@pytest.mark.parametrize('policy',['codes_only','encoder_and_codes_v1'])
def test_drift_gradient_policies_keep_exact_codes_and_energy(policy):
    assert energy_checks(policy)['reconstruction_gradient']==policy


def test_drift_policy_changes_only_gradient_not_forward_or_energy():
    conf=dict(latent_dim=4,codebook_size=16,drift_tau=.2,drift_beta=.25,drift_tokens=8)
    old=DriftingVectorQuantizer(conf)
    new=DriftingVectorQuantizer({**conf,'drift_reconstruction_gradient':'encoder_and_codes_v1'})
    new.load_state_dict(old.state_dict())
    x=torch.randn(2,4,4,4,requires_grad=True)
    state=torch.get_rng_state();a=old(x);torch.set_rng_state(state);b=new(x)
    for k in ('indices','quantized','loss'): assert torch.equal(a[k],b[k])


def test_weighted_loss_gradient_probe_reports_disconnection_without_changing_backward():
    from experiments.paper1.lst_restoration_lossfix_20261014.objective import loss
    r=Representation(dict(kind='R',A_celsius=1.))
    config={'loss':dict(pixel=1.,gradient=.1,representation=0.,bottleneck=1.,kl=1e-6,region='all_valid'),
            'task':{'residual_std_celsius':1.},'model':{'kind':'vq_drifting'}}
    latent=torch.randn(1,1,2,2,requires_grad=True)
    decoded=torch.randn_like(latent,requires_grad=True)
    batch=dict(reference_valid=torch.ones_like(latent,dtype=torch.bool),observed_valid=torch.ones_like(latent,dtype=torch.bool),
               temperature=torch.zeros_like(latent),target_field=torch.zeros_like(latent),center_celsius=torch.zeros(1))
    output=dict(encoded_prediction=decoded,prequantized_latent=latent,bottleneck_loss=latent.square().mean())
    total,plain=loss(output,batch,r,config)
    again,logged=loss(output,batch,r,config,diagnose_gradients=True)
    assert logged['gradient_probe/pixel_connected']==0
    assert logged['gradient_probe/bottleneck_connected']==1
    assert logged['total']==plain['total']
    g1=torch.autograd.grad(total,(latent,decoded),retain_graph=True)
    g2=torch.autograd.grad(again,(latent,decoded))
    for a,b in zip(g1,g2): assert torch.equal(a,b)


def test_generated_recipes_preserve_pretrained_and_formal_protocol():
    root=Path(__file__).resolve().parents[1]
    configs=root/'runs/paper1/lst_restoration_lossfix_20261014/configs'
    for p in configs.glob('*.json'):
        if p.name=='manifest.json':continue
        c=json.loads(p.read_text())
        assert c['loss']['representation']==0
        assert c['train']['epochs']==5 and c['train']['effective_batch']==64
        assert c['execution']['enabled'] is False
        if c['model']['kind']=='vqgan_pretrained': assert 'vq_geometry' not in c['model']
        if c['model']['kind']=='vq': assert c['model']['vq_geometry']=='unit_rms_train_init_v1'
        if c['representation']['kind'] in ('C','C_power','C_asinh'):
            assert c['representation']['colour_inverse']=='trilinear_v1'


def test_paired_controls_change_only_registered_mechanism():
    from experiments.paper1.lst_restoration_lossfix_20261014.config import validate
    root=Path(__file__).resolve().parents[1]/'runs/paper1/lst_restoration_lossfix_20261014'
    plan=json.loads((root/'CONTROLLED_CHECK_PLAN.json').read_text())
    for pair in plan['pairs']:
        candidate=json.loads((root/pair['candidate']).read_text())
        control=json.loads((root/pair['control']).read_text())
        assert validate(control)['ok'] and not control['execution']['enabled']
        assert candidate['tracking']['run_id'] != control['tracking']['run_id']
        section,key=pair['changed_field'].split('.')
        candidate[section][key]=control[section][key]
        for value in (candidate,control):
            value.pop('name');value.pop('tracking')
        assert candidate==control


def test_mechanism_gate_blocks_incomplete_launch_evidence(tmp_path,monkeypatch):
    from experiments.paper1.lst_restoration_lossfix_20261014 import runtime
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'runs/paper1/lst_restoration_lossfix_20261014/configs/sd_C.json').read_text())
    c['execution']['enabled']=True
    c['execution']['preflight_report']=str(tmp_path/'fake.json')
    monkeypatch.setattr(runtime,'source_hashes',lambda:{'synthetic-test':'fixed'})
    gates={k:True for k in ('components','model_contract','backward_finite','parameter_rate_audit',
                            'budget_resume','resource_profile','data_contract','epoch_validation')}
    (tmp_path/'fake.json').write_text(json.dumps(dict(protocol_config_sha256=runtime.protocol_digest(c),
        source_hashes=runtime.source_hashes(),gates=gates)))
    with pytest.raises(RuntimeError,match='Incomplete runtime preflight'):
        runtime.execution_guard(c)
