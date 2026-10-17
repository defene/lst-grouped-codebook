"""Finish the matched-size report only after download and census audits complete."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
import numpy as np
from color_analyze import ROOT,OUT,NAT,COUNT,dump

def main():
    while True:
        p=OUT/'progress.json'
        if p.exists() and json.loads(p.read_text(encoding='utf-8'))['status']=='complete':break
        time.sleep(5)
    dump(OUT/'finish_status.json',{'status':'running','phase':'audit_examples'})
    here=Path(__file__).parent
    for script in ('audit_examples.py','summarize.py'):
        dump(OUT/'finish_status.json',{'status':'running','phase':script})
        with (OUT/(script+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(here/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=OUT/'results';s=json.loads((result/'summary.json').read_text(encoding='utf-8'))
    manifest=[json.loads(t) for t in (NAT/'download_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    ids=[r['image_id'] for r in manifest]
    assert len(ids)==len(set(ids))==COUNT
    members={p.stem for p in (NAT/'images').glob('*/*.jpg')}
    assert members==set(ids)
    downloaded=json.loads((NAT/'verification.json').read_text(encoding='utf-8'))
    assert downloaded['status']=='complete' and downloaded['images']==COUNT and downloaded['all_sha256_audited']
    old=ROOT/'runs/openimages_hlp_color_20260917/results'
    prefix_errors={}
    with np.load(result/'natural_per_image_moments.npz') as a,np.load(old/'natural_per_image_moments.npz') as b:
        assert a['ids'].tolist()==ids
        for k in b.files:
            v=a[k][:100000];w=b[k]
            if v.dtype.kind in 'SUb':assert np.array_equal(v,w)
            else:
                e=float(np.max(np.abs(v-w)));prefix_errors[k]=e
                assert np.allclose(v,w,rtol=1e-10,atol=1e-12),(k,e)
    with np.load(result/'hlp_per_image_moments.npz') as a,np.load(old/'hlp_per_image_moments.npz') as b:
        for k in b.files:assert np.array_equal(a[k],b[k]),k
        assert len(a['ids'])==COUNT and len(np.unique(a['tiles']))==2796
        assert int(a['hls_pixels'].sum())==14119203144
        assert int(a['lst_pixels'].sum())==14347866752
    for k,v in s['statistics'].items():
        t=np.array(v['covariance_total']);w=np.array(v['covariance_within']);b=np.array(v['covariance_between'])
        assert np.allclose(t,w+b,atol=1e-12)
        assert abs(v['within_variance_share']+v['between_variance_share']-1)<1e-12
        assert np.linalg.eigvalsh(t).min()>-1e-12 and np.isfinite(t).all()
        if k!='hlp_lst':assert min(v['min'])>=0 and max(v['max'])<=1
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(OUT/'color_analyze_executed.py')==protocol['script_sha256']
    assert sha(OUT/'statistics_implementation.py')==protocol['statistic_implementation_sha256']
    protocol['executed_source_snapshot']='color_analyze_executed.py'
    protocol['statistics_source_snapshot']='statistics_implementation.py'
    dump(OUT/'protocol.json',protocol)
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    checks=dict(status='passed',natural_members_exact_match=COUNT,hlp_members_exact_match=COUNT,hlp_tiles=2796,
        old_natural_100000_moments_prefix_max_abs_errors=prefix_errors,hlp_all_moments_exactly_unchanged=True,
        full_normalization_crosscheck=s['normalization_crosscheck'],download_verification=downloaded,
        icc_audit_n=audit['natural_audit_n'],icc_conversion_fallbacks=len(audit['profile_conversion_errors']),
        visual_inspection='pending')
    dump(OUT/'verification.json',checks)
    subprocess.run([sys.executable,str(here/'write_report.py')],cwd=ROOT,check=True)
    dump(OUT/'finish_status.json',{'status':'complete','visual_inspection':'pending','report':str(OUT/'REPORT_ZH.md')})
    print(json.dumps({'status':'complete','coverage':s['coverage'],'report':str(OUT/'REPORT_ZH.md')}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(OUT/'finish_status.json',{'status':'failed','error':str(e)})
        raise
