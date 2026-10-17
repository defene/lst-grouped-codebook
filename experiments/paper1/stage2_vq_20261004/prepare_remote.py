"""Apply once to a newly copied frozen source snapshot; never edits old runs."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import uuid

root = Path(sys.argv[1]).resolve()
package = root / 'source/experiments/paper1/lst_restoration'

def edit(name, old, new, count=1):
    path = package / name
    text = path.read_text()
    if text.count(old) != count:
        raise ValueError((name, old[:80], text.count(old), count))
    path.write_text(text.replace(old, new))

for name in ('stage2_models.py', 'stage2_audit.py'):
    shutil.copy(Path(__file__).parent / name, package / name)
shutil.copy(Path(__file__).parent / 'run_stage2.py', root / 'source/run_stage2.py')
shutil.copy(root / 'assets/taming_model.py', package / 'taming_model.py')
edit('config.py', ') + GEO\n', ') + GEO + ("vqgan_pretrained", "vq_drifting")\n')
edit('config.py', 'return m["kind"] in ("sd", "sdxl", "flux1")',
     'return m["kind"] in ("sd", "sdxl", "flux1", "vqgan_pretrained")')
edit('config.py', '    if m["kind"] == "satmae_pp":\n', '''    if m["kind"] in ("vqgan_pretrained", "vq_drifting"):
        check(m['grid'] == 16 and m['codebook_size'] == 1024 and m['bits_per_token'] == 10,
              'Stage2 single-scale f16 K1024 rate is frozen')
        check(t['effective_batch'] == 64, 'Stage2 effective batch64 required')
        check(c['loss']['representation'] == 0, 'Stage2 physical-only reconstruction required')
        check(c['evaluation']['splits'] == ['val'], 'Stage2 selection uses validation only')
        if m['kind'] == 'vqgan_pretrained':
            check(m['latent_dim'] == 256 and m['num_res_blocks'] == 2 and t['micro_batch'] == 8,
                  'Official VQGAN architecture and profiled microbatch8 required')
            check(bool(m['pretrained_sha256']) and bool(m['pretrained_path']), 'Verified pretrained assets required')
        else:
            check(m['latent_dim'] == 32 and t['micro_batch'] == 16 and not m['pretrained_path'],
                  'Drifting scratch D32/micro16 required')
            check(m['drift_tau'] == .2 and m['drift_beta'] == .25 and m['drift_tokens'] == 512,
                  'Frozen drifting recipe required')
    if m["kind"] == "satmae_pp":
''')
edit('models.py', '    if kind in ("vq", "fsq", "lfq", "bsq", "ae", "var_multiscale"):\n',
'''    if kind == 'vqgan_pretrained':
        from .stage2_models import PretrainedVQGAN
        return PretrainedVQGAN(config)
    if kind in ("vq", "fsq", "lfq", "bsq", "ae", "var_multiscale", "vq_drifting"):
''')
edit('quantizers.py', '    if kind == "ae":\n', '''    if kind == 'vq_drifting':
        from .stage2_models import DriftingVectorQuantizer
        return DriftingVectorQuantizer(model_config)
    if kind == "ae":
''')
edit('objective.py', '    return total, {key: float(value.detach()) for key, value in terms.items()}\n',
'''    logged = {key: float(value.detach()) for key, value in terms.items()}
    for key, value in output.get('quantizer_diagnostics', {}).items():
        logged['quantizer/' + key] = float(value.detach())
    return total, logged
''')
for name in ('runtime.py', '__main__.py'):
    edit(name, '"budget_resume", "resource_profile", "data_contract")',
         '"budget_resume", "resource_profile", "data_contract", "stage2_mechanism")')
edit('runtime.py', 'files = list(Path(__file__).parent.glob("*.py"))',
     'files = list(Path(__file__).parent.glob("*.py")) + [root / "run_stage2.py"]')
# Integrate the verified codec repair, with normal snapshot provenance.
repair = (root.parent / 'stage1_evalrepair_20261004_attempt01/evaluate_repair.py').read_text()
start = repair.index('\ndef source_hashes():')
end = repair.index('\ndef condition_id(', start)
repair = repair[:start] + '\n' + repair[end:]
repair = repair.replace('source_hashes as training_source_hashes', 'source_hashes')
(package / 'evaluate.py').write_text(repair)

sys.path.insert(0, str(root / 'source'))
from experiments.paper1.lst_restoration.config import validate, write_json
from experiments.paper1.lst_restoration.runtime import source_hashes, utc, protocol_digest
assets = json.loads((root / 'assets/manifest.json').read_text())
reps = ['R', 'G', 'C', 'G_power', 'C_power', 'G_asinh', 'C_asinh', 'C_affine']
jobs = []
for kind in ('vqgan_pretrained', 'vq_drifting'):
    for rep in reps:
        base = root.parent / 'stage1_20260930_attempt02/runs' / ('vq_' + rep) / 'training/config.json'
        c = json.loads(base.read_text())
        name = kind + '_' + rep
        c['name'] = name + '_pixel10_seed17_20261004'
        c['experiment_revision'] = 'stage2-single-scale-pretrained-and-drifting-20261004-v1'
        c['model'].update(kind=kind, initialization='scratch', pretrained_path=None, pretrained_sha256={})
        c['train']['micro_batch_protocol'] = 'stage2-vq-20261004-v1'
        if kind == 'vqgan_pretrained':
            c['model'].update(initialization='official_imagenet_f16_1024', latent_dim=256,
                num_res_blocks=2, pretrained_path=str(root / 'assets'),
                pretrained_sha256={k: assets['sha256'][k] for k in ('autoencoder_state.pt', 'model.yaml')},
                source_repo='CompVis/taming-transformers', source_revision=assets['revision'],
                quantizer_loss='canonical_codebook_plus_0.25_commitment',
                pretrained_objective='VQGAN', finetuning_objective='physical_LST_no_GAN_no_LPIPS')
            c['train'].update(micro_batch=8, lr=1e-5)
        else:
            c['model'].update(drift_tau=.2, drift_beta=.25, drift_tokens=512,
                drift_distance='rms_euclidean', drift_potential='U_div_tau_squared',
                drift_code_initialization='uniform_minus_0.5_plus_0.5',
                drift_reconstruction_gradient='hard_embedding_no_STE',
                drift_source='https://kindxiaoming.github.io/blog/2026/representation-1/',
                drift_source_notebook_sha256=assets['sha256']['drifting_vqvae.ipynb'])
            c['train'].update(micro_batch=16, lr=1e-4)
        c['execution'] = dict(enabled=False, preflight_report=str(root/'runs'/name/'preflight.json'), alignment_report=None)
        c['tracking'].update(run_id=uuid.uuid4().hex[:12], name=name,
                             group='stage2-vq-pretrained-drifting-20261004')
        c['tracking']['tags'] = ['stage2', 'HLP-v2', 'single-scale', kind, rep, 'five-exact-epochs']
        result = validate(c)
        assert result['ok'], result
        path = root/'configs'/(name+'.json')
        write_json(path,c)
        jobs.append(dict(name=name, model=kind, representation=rep, config=str(path),
                         run_id=c['tracking']['run_id'], protocol_config_sha256=protocol_digest(c)))
write_json(root/'manifest.json', dict(created_utc=utc(), jobs=jobs, source_hashes=source_hashes(),
    assets=assets, baseline='stage1_20260930_attempt02', formal_epochs=5, test_used=False,
    note='Two experiment families, not a same-architecture initialization-only ablation.'))
print(json.dumps({'jobs':len(jobs),'manifest':str(root/'manifest.json')}))
