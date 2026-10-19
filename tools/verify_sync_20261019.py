"""Verify the 2026-10-19 source/evidence increment without data, model loading or network."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import subprocess

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--staged',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    locations=['docs/sync_20261019/source_manifest.json','reports/sync_20261019/source_manifest.json','nyu_snapshot_20261017/sync_20261019/source_manifest.json']
    manifests=[p for p in locations if (root/p).is_file()]
    assert len(manifests)==1,manifests
    def read(name):
        if args.staged:
            return subprocess.run(['git','-C',str(root),'show',':'+name],capture_output=True,check=True).stdout
        return (root/name).read_bytes()
    m=json.loads(read(manifests[0])); paths={}
    syntax=0
    for item in m['files']:
        rel=item['path'];assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        data=read(rel)
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],rel
        if rel in paths:assert paths[rel]==item['sha256']
        paths[rel]=item['sha256']
        if rel.endswith('.py'):ast.parse(data,filename=rel);syntax+=1
        if rel.endswith('.json'):json.loads(data)
    forbidden={'.pt','.pth','.ckpt','.safetensors','.onnx','.bin','.zip','.tar','.gz','.7z','.npz','.npy','.sqlite','.sqlite3','.nc','.eoz','.pyc'}
    bad=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    total=0;new_paths=[]
    if args.staged:
        raw=subprocess.run(['git','-C',str(root),'diff','--cached','--name-only','--diff-filter=ACMR','-z'],capture_output=True,check=True).stdout
        new_paths=[p for p in raw.decode('utf-8').split('\0') if p]
        assert new_paths,'Empty staged increment'
        for rel in new_paths:
            assert Path(rel).suffix.lower() not in forbidden,rel
            data=read(rel);total+=len(data);assert len(data)<50_000_000,rel
            if Path(rel).suffix.lower() in {'.py','.json','.jsonl','.md','.txt','.log','.ps1','.sh','.yaml','.yml','.toml'}:
                assert not bad.search(data),rel
            if rel.endswith('.py'):ast.parse(data,filename=rel)
            if rel.endswith('.json'):json.loads(data)
    if manifests[0].startswith('docs/'):
        docs=['README.md','docs/RESULTS.md','docs/sync_20261019/README.md','docs/sync_20261019/hlc/README.md','docs/delivery_20261019/README.md']
        for kind in ['models','openimages']:
            base='docs/delivery_20261019/records/'
            receipt=json.loads(read(base+kind+'_complete.verification.json'))
            status=json.loads(read(base+kind+'_status.json'))
            assert receipt['stage']==status['stage']=='complete'
            assert receipt['sha256']==status['sha256']==read(base+kind+'_complete.zip.sha256').decode().split()[0]
            entries=json.loads(read(base+kind+'_package_manifest.json'))['files']
            assert len(entries)+1==receipt['files']
        images=json.loads(read('docs/delivery_20261019/records/openimages_complete.verification.json'))
        assert images['images_sha256_verified']==219175
    elif manifests[0].startswith('reports/'):
        docs=['README.md','reports/sync_20261019/README.md']
        models=json.loads(read('outputs/torch_models_20261013/MODEL_MANIFEST.json'))['models']
        assert len(models)==len({x['name'] for x in models})==39
    else:
        docs=['README.md','nyu_snapshot_20261017/sync_20261019/README.md']
    links=0
    for name in docs:
        text=read(name).decode('utf-8')
        # Check newly written docs and the materialized conclusion links; ignore unrelated historical root links.
        if name=='README.md':text='\n'.join(s for s in text.splitlines() if '20261019' in s)
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#')[0]
            assert (root/name).parent.joinpath(target).exists(),(name,target)
            links+=1
    print(json.dumps({'status':'passed','copied_source_records':len(m['files']),'unique_source_files':len(paths),
        'source_python_syntax_checked':syntax,'checked_local_links':links,'staged_files':len(new_paths),
        'staged_bytes':total,'large_archives_datasets_weights_staged':False,'high_confidence_secret_hits':0,
        'training_or_model_evaluation_started':False},indent=2))

if __name__=='__main__':main()
