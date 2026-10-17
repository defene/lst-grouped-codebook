"""Export a declarative research design; not runnable trainer configurations."""
import argparse
import itertools
import json
from pathlib import Path


def design():
    jobs = []
    def add(stage, models, reps, trains, inits=('scratch',), means=('oracle',), seeds=(17,), bits=(10,)):
        for m, r, t, init, mean, seed, bit in itertools.product(models, reps, trains, inits, means, seeds, bits):
            row = dict(stage=stage, model=m, representation=r, training=t, initialization=init,
                       mean_condition=mean, seed=seed, core_bits_per_token=bit if m in ('vq','fsq','lfq','bsq') else None)
            row['id'] = '__'.join(str(row[k]) for k in ('model','representation','training','initialization','mean_condition','seed','core_bits_per_token'))
            row.update(execution_enabled=False, state='design_not_executable', epochs=5, presentations=1094875, updates=17110)
            jobs.append(row)
    add('P1_representation', ('vq',), ('R','G','C','G_power','C_power','G_asinh','C_asinh','C_affine'), ('pixel10',))
    add('P1_representation', ('sd',), ('G','C','G_power','C_power','G_asinh','C_asinh','C_affine'), ('pixel10',), inits=('finetune',))
    add('P2_mean', ('vq',), ('C_STAR',), ('pixel10',), means=('visible','none'))
    add('P2_mean', ('sd',), ('C_STAR',), ('pixel10',), inits=('finetune',), means=('visible','none'))
    add('P3_initialization', ('sd',), ('G_MATCH','C_STAR'), ('pixel10',))
    add('P3_initialization', ('cosmos_di8x8',), ('G_MATCH','C_STAR'), ('pixel10',), inits=('scratch','finetune'))
    add('P4_core', ('vq',), ('R','C_STAR'), ('clean',))
    add('P4_core', ('fsq','lfq','bsq'), ('R','C_STAR'), ('clean','pixel10'))
    add('P4_core', ('sd',), ('G_MATCH','C_STAR'), ('clean',), inits=('finetune',))
    for model, rep, init in (('vq','G_MATCH','scratch'),('vq','C_STAR','scratch'),
                             ('fsq','R','scratch'),('fsq','C_STAR','scratch'),
                             ('sd','C_STAR','scratch'),('sd','C_STAR','finetune'),
                             ('cosmos_di8x8','C_STAR','scratch'),('cosmos_di8x8','C_STAR','finetune')):
        add('P5_confirm_seeds', (model,), (rep,), ('pixel10',), inits=(init,), seeds=(23,42))
    add('P6_mixed', ('vq','fsq'), ('R','C_STAR'), ('mixed',))
    add('P6_mixed', ('sd',), ('G_MATCH','C_STAR'), ('mixed',), inits=('finetune',))
    add('P7_geo', ('eo_vae',), ('R','C_STAR'), ('clean','pixel10'))
    add('P7_geo', ('prithvi_mae','satmae_pp'), ('R',), ('clean','patch16'))
    add('P7_geo', ('vq',), ('R',), ('patch16',))
    add('P8_rate', ('vq','fsq'), ('C_STAR',), ('pixel10',), bits=(8,12))
    add('P9_external', ('sdxl','flux1'), ('G_MATCH','C_STAR'), ('clean','pixel10'), inits=('finetune',))
    add('P9_external', ('terramind_fsq',), ('R','C_STAR'), ('clean','pixel10'))
    assert len(jobs)==88 and len({x['id'] for x in jobs})==88
    # Check symbolic resolution creates no duplicate NEW trainings for any choice.
    for color, gray in (('C','G'),('C_power','G_power'),('C_asinh','G_asinh')):
        keys=[]
        for j in jobs:
            rep={'C_STAR':color,'G_MATCH':gray}.get(j['representation'],j['representation'])
            keys.append((j['model'],rep,j['training'],j['initialization'],j['mean_condition'],j['seed'],j['core_bits_per_token']))
        assert len(set(keys))==88
    return {'version':'lst-final-design-v2-20260927', 'status':'design_only_not_training_configs',
            'maximum_formal_training_jobs':88, 'first_decisive_block_jobs':25, 'core_without_P9_jobs':76,
            'record_presentations_if_all_run':88*1094875, 'training_started':False,
            'symbols':{'C_STAR':'one color among C/C_power/C_asinh chosen once on VQ seed17 validation',
                       'G_MATCH':'G/G_power/G_asinh using exactly the same scalar transform as C_STAR'},
            'stages':{s:sum(j['stage']==s for j in jobs) for s in dict.fromkeys(j['stage'] for j in jobs)},
            'selection':{'only_split':'val','model':'vq','seed':17,
                'score':'0.5*equal_image_MSE(clean)+0.5*equal_image_MSE(pixel10)',
                'tie_within_relative_best':0.01,'tie_preference':['C','C_asinh','C_power'],
                'screening_gate':'at least 2% lower score than G_MATCH and C_affine; pixel10 gradient_RMSE no more than 2% worse than G_MATCH',
                'gate_interpretation':'predeclared practical screening, not significance or final superiority',
                'failure':'report negative results; suspend dependent color expansions and publish a new smaller allocation before test'},
            'new_protocol_changes':['RGB auxiliary coefficient 0 for all main comparisons; keep common physical MSE/gradient and declared native regularizers',
                'stage evaluation val only; freeze representation/comparisons before final test',
                'no-mean interface removes per-image mean in encoder, decoder, inverse centering and payload'],
            'model_reference':{'cosmos_di8x8':{'hub':'nvidia/Cosmos-0.1-Tokenizer-DI8x8',
                'role':'official pretrained discrete image FSQ comparator; retain native model',
                'source':'https://github.com/NVIDIA/Cosmos-Tokenizer', 'adapter_implemented':False}},
            'implementation_gaps':['G_power/G_asinh','no-mean end-to-end interface',
                'same-architecture SD scratch','Cosmos trainable adapter and strict official-weight loading',
                'val-only stage runner and selection freeze','table fill and cross-seed statistics'],
            'zero_shot':'evaluation only, reuse by exact weight/representation/inference/data fingerprints; no training count',
            'jobs':jobs}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists(): raise FileExistsError('Do not overwrite an existing design')
    obj=design()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:obj[k] for k in ('version','maximum_formal_training_jobs','first_decisive_block_jobs','stages','training_started')},ensure_ascii=False,indent=2))
