"""Create and verify a new six-arm local experiment before execution."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,gc,itertools,json,math,sqlite3,time,uuid,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,default_collate,Subset
from .config import COUNTS,PROJECT,write_json,file_digest,validate
from .data import LSTDataset,EpochSampler
from .epoch_validation import default_protocol,isolated_inference
from .models import build_model,initialize,controllers
from .representations import Representation
from . import runtime

def generate(output,micro=8,workers=4):
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True)
    jobs=[]
    for family in ('vq','grouped_g8k256'):
        original=PROJECT/f'outputs/torch_models_20261013/models/{family}_G/config.original.json'
        for method in ('baseline','ema','ema_maint'):
            c=json.loads(original.read_text());name=f'{family}_G_{method}'
            c.update(version='lst-ema-full-20261015',name=name,ema_method=method)
            c['data'].update(root=str(PROJECT),historical_test_exposure=False)
            c['task']['scale_evidence']=str(PROJECT/'processed_data/hlp_split_v2_20260914/prithvi/residual_scale.json')
            c['train'].update(epochs=5,effective_batch=64,micro_batch=micro,workers=workers,selection='best_validation_rmse',
                updates=5*math.ceil(COUNTS['train']/64),record_presentations=5*COUNTS['train'],device='cuda:0',
                checkpoint_every=100,memory_limit_gib=24.,micro_batch_protocol='local-micro8-effective64-full-v2')
            c['train']['warmup_updates']=500 if family=='grouped_g8k256' else 0
            c['epoch_validation']=default_protocol()
            c['evaluation'].update(stage='selection',splits=['val'],views=2,batch_size=micro)
            c['tracking']={'enabled':False,'run_id':uuid.uuid4().hex,'every_updates':10}
            c['execution']={'enabled':False,'preflight_report':str(output/'preflight'/f'{name}.json')}
            c['diagnostics']={'gradient_every_updates':100}
            result=validate(c)
            if not result['ok']:raise ValueError(result)
            path=output/'configs'/f'{name}.json';write_json(path,c)
            jobs.append(dict(name=name,config=str(path),historical_config_sha256=file_digest(original)))
    write_json(output/'queue.json',dict(jobs=jobs,scope='G, six arms, five full HLP v2 train epochs',
        status='prepared_not_launched',total_record_presentations=6*5*COUNTS['train'],
        note='No pilot weights reused. Family-specific original LR schedule retained. No test data.'))

def preflight(output,test_xml):
    output=Path(output);test_xml=Path(test_xml)
    suite=ET.parse(test_xml).getroot();cases=list(suite.iter('testcase'))
    if len(cases)<17 or any(list(suite.iter(tag)) for tag in ('failure','error','skipped')):raise ValueError('CPU tests incomplete or failed')
    queue=json.loads((output/'queue.json').read_text());frozen=runtime.source_hashes();data_evidence=None
    for job in queue['jobs']:
        c=json.loads(Path(job['config']).read_text());device=runtime.setup(c)
        dataset=LSTDataset(c,'train');validation=LSTDataset(c,'val',default_protocol()['profiles'][1],0)
        if data_evidence is None:
            if np.intersect1d(dataset.record_ids,validation.record_ids).size:raise ValueError('Train/val record overlap')
            conn=dataset.source._connect();tiles=dict(conn.execute('SELECT id,tile FROM records'))
            tt={tiles[int(x)] for x in dataset.record_ids};vt={tiles[int(x)] for x in validation.record_ids}
            if tt&vt or len(tt)!=2796 or len(vt)!=147:raise ValueError('Tile split mismatch')
            data_evidence=dict(train=dataset.fingerprint,val=validation.fingerprint,train_tiles=len(tt),val_tiles=len(vt),
                record_and_tile_disjoint=True,normalization_fit_split=dataset.source.stats['fit_split'])
            write_json(output/'preflight/data.json',data_evidence)
        if dataset.fingerprint!=data_evidence['train'] or validation.fingerprint!=data_evidence['val']:raise ValueError('Data fingerprint changed')
        model=build_model(c).to(device);rep=Representation(c['representation'])
        initialize(model,dataset,c,rep,device);model.train()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['train']['lr'],betas=(.9,.95),weight_decay=.01)
        order=list(itertools.islice(EpochSampler(len(dataset),17,0),95))
        # Exercise Windows worker spawning and the exact full/tail weighting path.
        loader=DataLoader(Subset(dataset,order),batch_size=c['train']['micro_batch'],num_workers=c['train']['workers'],
            drop_last=False,generator=torch.Generator().manual_seed(17))
        batches=list(loader);del loader
        reports=[];cursor=0
        for actual_target in (64,31):
            count=math.ceil(actual_target/c['train']['micro_batch']);group=batches[cursor:cursor+count];cursor+=count
            start=time.perf_counter()
            actual,terms,_=runtime.train_update(model,group,opt,rep,c,device,update_step=len(reports)+1)
            assert actual==actual_target and all(math.isfinite(v) for v in terms.values())
            reports.append(dict(actual_batch=actual,seconds=time.perf_counter()-start,loss=terms))
        assert all(int(e.steps)==2 for e in controllers(model))
        # Real validation forwards, without pretending this is full validation.
        before=[e.steps.clone() for e in controllers(model)]
        with isolated_inference(model,device,20260925):
            b=runtime.move_batch(default_collate([validation[i] for i in range(8)]),device)
            from .objective import model_inputs,readout
            with runtime.amp(c,device):result=model(**model_inputs(b,rep))
            assert torch.isfinite(readout(result,b,rep,c)[b['reference_valid']]).all()
        assert all(torch.equal(x,e.steps) for x,e in zip(before,controllers(model)))
        report=dict(protocol_config_sha256=runtime.protocol_digest(c),source_hashes=frozen,
            gates=dict(cpu_contracts=True,cuda_full_and_tail_update=True,data_contract=True,resource_profile=True),
            cpu_tests=dict(count=len(cases),report=str(test_xml),sha256=file_digest(test_xml)),
            updates=reports,memory=runtime.memory(device),data_evidence_sha256=file_digest(output/'preflight/data.json'),
            validation_smoke_records=8,full_validation_not_yet_run=True,parameters=sum(p.numel() for p in model.parameters()))
        write_json(Path(c['execution']['preflight_report']),report)
        print(json.dumps(dict(event='preflight_passed',name=job['name'],seconds=[r['seconds'] for r in reports],memory=report['memory'])),flush=True)
        dataset.close();validation.close();del model,opt,batches,b,result;gc.collect();torch.cuda.empty_cache()
    if frozen!=runtime.source_hashes():raise ValueError('Source changed during preflight')
    # Enable only after all six measured checks have passed.
    for job in queue['jobs']:
        path=Path(job['config']);c=json.loads(path.read_text());c['execution']['enabled']=True;write_json(path,c);runtime.execution_guard(c)
    queue.update(status='ready',source_hashes=frozen,cpu_test_sha256=file_digest(test_xml));write_json(output/'queue.json',queue)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['generate','preflight']);ap.add_argument('--output',required=True)
    ap.add_argument('--tests');args=ap.parse_args()
    if args.action=='generate':generate(args.output)
    else:preflight(args.output,args.tests)
