"""Exercise the new book width with real quantizers and scheduler release conditions."""
import copy
import pytest
import torch
from experiments.paper1.lst_ema_pilot_20261014.quantizers import EMAVQ, EMAGrouped, controllers, initialize_from_latents
from experiments.paper1.lst_restoration_lossfix_20261014.quantizers import VectorQuantizer
from experiments.paper1.lst_restoration_lossfix_20261014.grouped_adapter import GroupedProductQuantizer
from experiments.paper1.lst_ema_k512_20261015.run import parent_ready, write_json


@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('method', ['baseline', 'ema', 'ema_maint'])
def test_k512_training_and_decode_roundtrip(grouped, method):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    c = dict(groups=8, codebook_size=512, latent_dim=32, scales=[1,2,3,4,5,6,8,10,13,16],
             commitment=.25, quant_resi=.5, share_quant_resi=4)
    q = GroupedProductQuantizer(c) if grouped else VectorQuantizer(32,512)
    if method != 'baseline':
        q = (EMAGrouped if grouped else EMAVQ)(q, method == 'ema_maint')
    x = torch.randn(2,32,16,16,requires_grad=True)
    if method == 'ema_maint':
        initialize_from_latents(q, x.detach())
    books = list(q.grouped.embeddings) if grouped else [q.embedding]
    assert len(books) == (8 if grouped else 1)
    assert all(b.weight.shape == (512, 4 if grouped else 32) for b in books)
    for e in controllers(q):
        e.begin()
    r = q(x)
    assert r['indices'].min() >= 0 and r['indices'].max() < 512
    (r['quantized'].square().mean() + r['loss']).backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
    assert all((b.weight.grad is None) == (method != 'baseline') for b in books)
    for e in controllers(q):
        e.commit()
        assert int(e.steps) == 1
    q.eval()
    before = copy.deepcopy(q.state_dict())
    r = q(x.detach())
    assert torch.equal(r['quantized'], q.decode_indices(r['indices']))
    assert all(torch.equal(v, q.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('phase,completed,ready', [
    ('running',['a'],False), ('failed',['a','b'],False), ('stopped',[],False),
    ('completed',['a'],False), ('completed',['a','b'],True)])
def test_scheduler_requires_all_parent_jobs(tmp_path, phase, completed, ready):
    write_json(tmp_path/'queue.json', dict(jobs=[dict(name='a'),dict(name='b')]))
    write_json(tmp_path/'queue_status.json', dict(status=phase,completed=completed))
    assert parent_ready(tmp_path) == ready
