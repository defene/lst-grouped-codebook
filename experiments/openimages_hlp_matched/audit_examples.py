"""Deterministic visual examples and independent preprocessing checks."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import io,json,hashlib,sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageCms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from color_analyze import NAT,OUT,PREP,natural,hlp,moments,linearize,dump
from eo_data import open_dataset

plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})

def main():
    OUT.mkdir(exist_ok=True)
    rows=[json.loads(s) for s in (NAT/'download_manifest.jsonl').read_text().splitlines()]
    rank=lambda s:hashlib.sha256(('color-audit-20260917:'+s).encode()).digest()
    selected=sorted(rows,key=lambda r:rank(r['image_id']))[:1024]
    profiles=Counter();modes=Counter();names={};errors=[];natural_check=[];converted_check=[]
    dst=ImageCms.createProfile('sRGB')
    for r in selected:
        x,_=natural(NAT/r['path'])
        # Independent moments, not using the census covariance implementation.
        z=x.reshape(3,-1).astype(float);natural_check.append((z.mean(1),z.var(1)))
        with Image.open(NAT/r['path']) as im:
            modes[im.mode]+=1; profile=im.info.get('icc_profile')
            if profile:
                h=hashlib.sha256(profile).hexdigest();profiles[h]+=1
                try:
                    src=ImageCms.ImageCmsProfile(io.BytesIO(profile))
                    names[h]=ImageCms.getProfileDescription(src).strip()
                    im=ImageCms.profileToProfile(im,src,dst,outputMode='RGB')
                except Exception as e:
                    errors.append(dict(image_id=r['image_id'],error=str(e)));im=im.convert('RGB')
            else:profiles['untagged']+=1;im=im.convert('RGB')
            w,h=im.size;size=(288,int(288*h/w)) if w<=h else (int(288*w/h),288)
            im=im.resize(size,Image.Resampling.LANCZOS)
            left=round((size[0]-256)/2);top=round((size[1]-256)/2)
            a=np.asarray(im.crop((left,top,left+256,top+256)),dtype=float).reshape(-1,3)/255
            converted_check.append((a.mean(0),a.var(0)))
    def share(rows):
        a=np.asarray(rows);w=a[:,1].mean(0).sum();b=a[:,0].var(0).sum();return float(w/(w+b))
    ds=open_dataset('prithvi','hls','train',prepared=PREP)
    conn=ds._connect();mapping=dict(conn.execute('select id,key from records where split="train"'))
    indices=sorted(range(len(ds)),key=lambda i:rank(mapping[int(ds.indices[i,0])]))[:12]
    checks=[]
    for i in indices:
        x,m,y,n,key,tile=hlp(i,True)
        s=moments(x,m,np.random.default_rng(1));z=x[:,m].astype(float)
        assert np.allclose(s['cov'],np.cov(z,bias=True),atol=1e-12)
        checks.append(dict(index=i,key=key,tile=tile,covariance_error=float(np.max(np.abs(s['cov']-np.cov(z,bias=True))))))
    # Explicit pixel-for-pixel check against the earlier torchvision preprocessing.
    import torch
    torch.set_num_threads(1)
    from torchvision import transforms
    tr=transforms.Compose([transforms.Resize(288,interpolation=transforms.InterpolationMode.LANCZOS),transforms.CenterCrop(256),transforms.ToTensor()])
    transform_errors=[]
    for r in selected[:20]:
        x,_=natural(NAT/r['path'])
        with Image.open(NAT/r['path']) as im:t=tr(im.convert('RGB')).numpy()
        transform_errors.append(float(np.max(np.abs(x-t))))
    assert max(transform_errors)==0
    report=dict(natural_audit_n=1024,selection='smallest SHA256(color-audit-20260917:ImageID) from full 218975',
        selected_ids=[r['image_id'] for r in selected],profile_counts=[dict(sha256=k,count=v,description=names.get(k,k)) for k,v in profiles.most_common()],
        original_modes=dict(modes),profile_conversion_errors=errors,
        encoded_within_share_before_icc=share(natural_check),encoded_within_share_after_icc=share(converted_check),
        torchvision_transform_exact_match_n=20,hlp_independent_checks=checks)
    dump(OUT/'audit.json',report)
    # Display is fixed across all HLP examples, never per-image min/max.
    def encode(a):return np.where(a<=.0031308,12.92*a,1.055*np.maximum(a,0)**(1/2.4)-.055)
    fig,axs=plt.subplots(3,6,figsize=(15,8.5))
    ex=[]
    for j,r in enumerate(selected[:6]):
        x,_=natural(NAT/r['path']);axs[0,j].imshow(x.transpose(1,2,0));axs[0,j].set_title(r['image_id'],fontsize=8)
        x,m,y,n,key,tile=hlp(indices[j]);rgb=x[[2,1,0]].transpose(1,2,0)
        raw=rgb.copy();raw[~m]=.5;display=encode(np.clip(rgb/.3,0,1));display[~m]=.5
        axs[1,j].imshow(raw);axs[1,j].set_title(key.split('/')[-3]+'/'+key.split('/')[-1],fontsize=8)
        axs[2,j].imshow(display)
        ex.append(dict(column=j,natural_id=r['image_id'],hlp_key=key,hlp_index=indices[j]))
    for i,label in enumerate(['自然图像\n编码 RGB','HLP RGB\n直接显示反射率','同一 HLP\n固定增益 + sRGB 曲线']):
        axs[i,0].set_ylabel(label,fontsize=10)
    for ax in axs.ravel():ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('固定哈希抽取的示例：显示方式会改变视觉印象，统计必须用原数值',fontsize=16,x=.03,ha='left')
    fig.text(.03,.025,'第 3 行仅供观看：sRGB 编码(clip(反射率 / 0.3, 0, 1))，会裁掉高亮；全部 HLP 共用同一规则。灰色为无效区域。\n自然与遥感示例不配对；未按颜色或对比度挑图。所有定量结果均不使用第 3 行显示增强。',fontsize=10)
    fig.subplots_adjust(left=.10,right=.99,top=.91,bottom=.11,hspace=.20,wspace=.06)
    fig.savefig(OUT/'fixed_display_examples.png');plt.close(fig);dump(OUT/'example_members.json',ex)
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected_ids','hlp_independent_checks')},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
