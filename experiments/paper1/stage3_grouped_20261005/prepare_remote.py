"""Prepare only the new grouped snapshot, with pinned source and new run IDs."""
from pathlib import Path
import json
import shutil
import sys
import uuid

r=Path(sys.argv[1]).resolve()
pkg=r/'source/experiments/paper1/lst_restoration'
old=r.parent/'stage1_20260930_attempt02'
stage2=r.parent/'stage2_vq_20261005_attempt02/source'

def edit(name, before, after):
    p=pkg/name
    s=p.read_text()
    assert s.count(before)==1,(name,before[:90],s.count(before))
    p.write_text(s.replace(before,after))

shutil.copy(Path(__file__).parent/'grouped_adapter.py',pkg/'grouped_adapter.py')
edit('config.py',') + GEO\n',') + GEO + ("grouped_g8k256",)\n')
edit('config.py','    if m["kind"] == "satmae_pp":\n', '''    if m['kind'] == 'grouped_g8k256':
        check((m['groups'],m['codebook_size'],m['latent_dim'])==(8,256,32), 'Frozen g8k256 required')
        check(m['scales']==[1,2,3,4,5,6,8,10,13,16], 'Pinned multiscale recipe required')
        check(m['base_channels']==160 and m['num_res_blocks']==2, 'Pinned backbone architecture required')
        check(t['effective_batch']==64 and t['micro_batch']==8, 'Grouped profiled batch contract required')
        check(t['warmup_updates']==500 and t['lr']==1e-4, 'Pinned learning rate recipe required')
        check(not m['pretrained_path'] and c['evaluation']['splits']==['val'], 'Scratch / validation only required')
    if m["kind"] == "satmae_pp":
''')
edit('models.py','    if kind in ("vq", "fsq", "lfq", "bsq", "ae", "var_multiscale"):\n',
'''    if kind == 'grouped_g8k256':
        from .grouped_adapter import GroupedRestorationCodec
        return GroupedRestorationCodec(config)
    if kind in ("vq", "fsq", "lfq", "bsq", "ae", "var_multiscale"):
''')
edit('quantizers.py','    if kind == "ae":\n', '''    if kind == 'grouped_g8k256':
        from .grouped_adapter import GroupedProductQuantizer
        return GroupedProductQuantizer(model_config)
    if kind == "ae":
''')
edit('objective.py','    return total, {key: float(value.detach()) for key, value in terms.items()}\n',
'''    logged = {key: float(value.detach()) for key, value in terms.items()}
    for key, value in output.get('quantizer_diagnostics', {}).items():
        logged['quantizer/' + key] = float(value.detach())
    return total, logged
''')
edit('runtime.py','files = list(Path(__file__).parent.glob("*.py"))',
     'files = list(Path(__file__).parent.glob("*.py")) + [root / "run_stage3.py"]')
for name in ('runtime.py','__main__.py'):
    edit(name,'"budget_resume", "resource_profile", "data_contract")',
         '"budget_resume", "resource_profile", "data_contract", "grouped_mechanism")')
edit('runtime.py','lr = config["train"]["lr"] * .5 * (1 + math.cos(math.pi * step / config["train"]["updates"]))',
     'from .grouped_adapter import learning_rate\n                lr = learning_rate(config, step)')
edit('audit_data_budget.py','lr = toy["train"]["lr"] * .5 * (1 + math.cos(math.pi * step / toy["train"]["updates"]))',
     'from .grouped_adapter import learning_rate\n                    lr = learning_rate(toy, step)')
edit('audit_data_budget.py','views=1, workers=0, updates=4, record_presentations=134)',
     'views=1, workers=0, updates=4, record_presentations=134, warmup_updates=1)')

# The previously verified same-batch codec repair, now part of this frozen source.
s=(stage2/'experiments/paper1/lst_restoration/evaluate.py').read_text()
s=s.replace('discrete_options = dict(indices=index_array,','discrete_options = dict(indices=None,')
s=s.replace('codebook_size=config["model"]["codebook_size"] if indices is not None else None)',
            'codebook_size=None)')
needle='        _write_rows(output / "rows_raw.jsonl", rows)'
assert s.count(needle)==1
s=s.replace(needle,'''        if all_indices:
            from .grouped_adapter import grouped_statistics
            result['grouped_token_statistics'] = grouped_statistics(
                np.concatenate(all_indices), config['model']['scales'], config['model']['codebook_size'])
'''+needle)
(pkg/'evaluate.py').write_text(s)

s=(stage2/'experiments/paper1/lst_restoration/stage2_audit.py').read_text()
s=s.replace('from .stage2_models import drifting_energy, DriftingVectorQuantizer',
            'from .grouped_adapter import mechanism_checks')
start,end=s.index('\ndef energy_checks():'),s.index('\ndef run(')
s=s[:start]+'\n'+s[end:]
start=s.index("            if config['model']['kind'] == 'vqgan_pretrained':")
end=s.index('            del out, replay',start)
s=s[:start]+s[end:]
s=s.replace('energy_checks()', 'mechanism_checks()').replace("'energy_checks'","'grouped_mechanism_checks'")
s=s.replace('stage2_mechanism','grouped_mechanism')
(pkg/'grouped_audit.py').write_text(s)
s=(stage2/'run_stage2.py').read_text().replace('stage2_audit','grouped_audit').replace("'stage2'","'grouped'").replace("'stage2.json'","'grouped.json'")
(r/'source/run_stage3.py').write_text(s)
s=(old/'launch.sbatch').read_text().replace('p1-stage1','p1-grouped').replace('run_stage1_launch.py','run_stage3.py')
(r/'launch.sbatch').write_text(s)

sys.path.insert(0,str(r/'source'))
from experiments.paper1.lst_restoration.config import validate,write_json
from experiments.paper1.lst_restoration.runtime import source_hashes,protocol_digest,utc
reps=['R','G','C','G_power','C_power','G_asinh','C_asinh','C_affine']
jobs=[]
for rep in reps:
    c=json.loads((old/'runs'/('vq_'+rep)/'training/config.json').read_text())
    name='grouped_g8k256_'+rep
    c['name']=name+'_pixel10_seed17_20261005'
    c['experiment_revision']='grouped-9465a23-eight-representations-20261005-v1'
    c['model'].update(kind='grouped_g8k256',base_channels=160,num_res_blocks=2,
        groups=8,codebook_size=256,bits_per_token=8,latent_dim=32,
        scales=[1,2,3,4,5,6,8,10,13,16],quant_resi=.5,share_quant_resi=4,
        initialization='scratch',pretrained_path=None,pretrained_sha256={},
        source_repo='defene/lst-grouped-codebook',source_revision='9465a23adbdc5688093be33c62cadf8860301f2e',
        source_configuration='A07 g8k256; lowest published val clean and noisy10 RMSE')
    c['train'].update(micro_batch=8,lr=1e-4,warmup_updates=500,
                      micro_batch_protocol='grouped-micro8-effective64-20261005')
    c['execution']=dict(enabled=False,preflight_report=str(r/'runs'/name/'preflight.json'),alignment_report=None)
    c['tracking'].update(run_id=uuid.uuid4().hex[:12],name=name,group='stage3-grouped-g8k256-20261005',
                         tags=['stage3','grouped-g8k256','multiscale','HLP-v2',rep,'five-exact-epochs'])
    assert validate(c)['ok'],validate(c)
    path=r/'configs'/(name+'.json');write_json(path,c)
    jobs.append(dict(name=name,model='grouped_g8k256',representation=rep,config=str(path),
                     run_id=c['tracking']['run_id'],protocol_config_sha256=protocol_digest(c)))
write_json(r/'manifest.json',dict(created_utc=utc(),jobs=jobs,source_hashes=source_hashes(),
    repository_revision='9465a23adbdc5688093be33c62cadf8860301f2e',
    selection=dict(metric='published validation RMSE; clean and noisy10 both lowest',
                   configuration='g8k256',val_clean=.644670,val_noisy10=.721231,test_for_selection=False),
    inherited_architecture=dict(groups=8,codes_per_group=256,group_dimension=4,latent_dim=32,
        ch=160,residual_blocks=2,scales=[1,2,3,4,5,6,8,10,13,16],quant_resi=.5,share_quant_resi=4),
    protocol_adaptations=['8 representations instead of source normalized LST only',
        'current pixel10 instead of unavailable source QA mask bank',
        'current oracle mean and physical pixel+gradient loss; no source extra spatial3 term',
        'current observation adapter and zero mean branch / output initialization',
        'micro8 accumulation8 instead of micro4 accumulation16; effective64 unchanged',
        'full validation only; no test selection or new test evaluation'],
    historical_results_not_directly_comparable=True))
print('prepared',len(jobs),'configs')
