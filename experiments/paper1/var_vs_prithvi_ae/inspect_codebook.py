"""Read-only, fixed-subset per-scale code usage audit of the four final VQ-VAEs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import gc
import json
import math
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from eo_data.core import json_write,sha256_file
from eo_denoise.engine import read_checkpoint,seed_everything,device_for,amp_context,source_hashes
from eo_denoise.data import PairedDataset
from eo_denoise.models import CrossViewModel


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def describe(counts):
    counts=np.asarray(counts,dtype=np.int64)
    total=int(counts.sum())
    if not total:return dict(tokens=0,used_codes=0,entropy_bits=None,perplexity=None)
    p=counts/total
    ordered=np.argsort(-counts,kind='stable')
    nonzero=p[p>0]
    entropy=float(-(nonzero*np.log2(nonzero)).sum())
    return dict(tokens=total,used_codes=int((counts>0).sum()),unused_codes=int((counts==0).sum()),
        usage_fraction=float((counts>0).mean()),entropy_bits=entropy,perplexity=2**entropy,
        top1_fraction=float(p[ordered[0]]),top1_code=int(ordered[0]),
        top10_fraction=float(p[ordered[:10]].sum()),
        codes_covering_90pct=int(np.searchsorted(np.cumsum(p[ordered]),.9)+1),
        codes_covering_99pct=int(np.searchsorted(np.cumsum(p[ordered]),.99)+1),
        top10_codes=[dict(code=int(i),count=int(counts[i])) for i in ordered[:10]])


@torch.inference_mode()
def main(root=None):
    root=(root or ROOT/'runs/paper1/var_vs_prithvi_ae_full').resolve()
    destination=root/'report/codebook_audit'
    destination.mkdir(exist_ok=True)
    assert read(root/'status.json')['status']=='training_and_evaluation_complete'
    manifest=read(root/'manifest.json')
    assert source_hashes()==manifest['source_hashes']
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),subset_records=256,views_per_record=2,
        subset_definition='same fixed val subset as all training diagnostics; shared across models',
        counts_scope='all transmitted positions; separately reference support >=75%',
        no_training=True,source_hashes=source_hashes(),script_sha256=sha256_file(__file__),runs={})
    baseline_ids=None
    for run in sorted(root.glob('full_*_conv_*')):
        if not run.is_dir():continue
        started=time.perf_counter()
        checkpoint=read_checkpoint(run/'last.pt')
        config=checkpoint['config']
        seed_everything(config['seed']);torch.set_num_threads(4)
        device=device_for(config)
        model=CrossViewModel(config).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        m=config['data']['modalities'][0];codec=model.codecs[m];q=codec.quantizer
        weight=q.embedding.weight.detach().cpu().numpy()
        record=dict(checkpoint_sha256=sha256_file(run/'last.pt'),
            embedding_shape=list(weight.shape),exact_distinct_embedding_vectors=int(len(np.unique(weight,axis=0))),
            ema_hit_total=float(q.ema_vocab_hit_SV.sum()),
            ema_note='EMA hit buffer is updated only under distributed initialization; unavailable in these single-process runs. Zero means unrecorded, not collapse.',
            evaluations={},full_validation_metrics={})
        del checkpoint
        for condition,cov in [('clean',0),('noisy10',.1)]:
            ds=PairedDataset(config,'val',coverage=cov)
            assert len(ds)==256
            assert ds.fingerprint==read(run/'val_step_0010813/metrics.json')['data_fingerprint']
            if baseline_ids is None:baseline_ids=ds.record_ids.copy()
            else:assert np.array_equal(baseline_ids,ds.record_ids)
            loader=DataLoader(ds,batch_size=config['eval']['batch_size'],shuffle=False,num_workers=2,
                pin_memory=True,generator=torch.Generator().manual_seed(20260910))
            counts=np.zeros((len(codec.scales),q.size),np.int64)
            valid_counts=np.zeros_like(counts)
            residual_mse=np.zeros(len(codec.scales),np.float64)
            encoded_mse=0.;examples=0
            for batch in loader:
                reference=batch[m]['reference_mask'].to(device)
                for view in range(2):
                    image=batch[m]['image'][:,view].to(device)
                    mask=batch[m]['input_mask'][:,view].to(device)
                    x=torch.cat((torch.where(mask.bool(),image,0),mask.to(image.dtype)),1)
                    with amp_context(config,device):
                        z=codec.network.quant_conv(codec.network.encoder(x))
                    z=z.float()
                    # Exact float32 quantization path used by the saved final evaluations.
                    ids=q.f_to_idxBl_or_fhat(z,False)
                    embeddings=[q.embedding(v).transpose(1,2).reshape(len(v),config['model']['latent_dim'],s,s)
                                for v,s in zip(ids,codec.scales)]
                    decoded=q.embed_to_fhat(embeddings,all_to_max_scale=True,last_one=False)
                    encoded_mse+=float(z.square().flatten(1).mean(1).sum())
                    examples+=len(z)
                    for i,(grid,scale,fhat) in enumerate(zip(ids,codec.scales,decoded)):
                        values=grid.cpu().numpy().ravel()
                        counts[i]+=np.bincount(values,minlength=q.size)
                        support=(F.adaptive_avg_pool2d(reference.float(),(scale,scale))[:,0]>=.75).flatten(1)
                        kept=grid[support].cpu().numpy()
                        valid_counts[i]+=np.bincount(kept,minlength=q.size)
                        residual_mse[i]+=float((z-fhat).square().flatten(1).mean(1).sum())
            ds.close();del loader
            assert examples==512
            assert np.array_equal(counts.sum(1),np.array(codec.scales)**2*512)
            summary=describe(counts.sum(0))
            if condition=='noisy10':
                saved=read(run/'val_step_0010813/metrics.json')['tokens'][m]
                assert summary['used_codes']==saved['used_codes']
                assert math.isclose(summary['perplexity'],saved['perplexity'],rel_tol=1e-10)
            per_scale=[dict(scale=s,all_positions=describe(counts[i]),reference_valid=describe(valid_counts[i]),
                remaining_latent_mse=float(residual_mse[i]/examples),
                remaining_fraction_of_latent_energy=float(residual_mse[i]/encoded_mse)) for i,s in enumerate(codec.scales)]
            record['evaluations'][condition]=dict(pooled=summary,per_scale=per_scale,
                encoded_latent_mean_square=encoded_mse/examples,
                pooled_reference_valid=describe(valid_counts.sum(0)),
                matches_saved_diagnostic_counts=True if condition=='noisy10' else None)
            full=read(run/('eval_'+condition)/'metrics.json')
            record['full_validation_metrics'][condition]=dict(samples=full['evaluated_pairs'],**full['tokens'][m])
            np.savez_compressed(destination/f'{run.name}_{condition}_counts.npz',counts=counts,
                reference_valid_counts=valid_counts,scales=codec.scales,record_ids=ds.record_ids)
            print(json.dumps(dict(run=run.name,condition=condition,pooled=summary,
                scales=[dict(scale=r['scale'],used=r['all_positions']['used_codes'],pp=round(r['all_positions']['perplexity'],3),
                             top1=round(r['all_positions']['top1_fraction'],5),residual=round(r['remaining_fraction_of_latent_energy'],6)) for r in per_scale])),flush=True)
        record['elapsed_seconds']=time.perf_counter()-started
        output['runs'][run.name]=record
        json_write(destination/'results.json',output)
        del model,codec,q,ds,config,weight
        gc.collect();torch.cuda.empty_cache()
    assert len(output['runs'])==4
    output.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    json_write(destination/'results.json',output)
    print('CODEBOOK AUDIT COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    main(parser.parse_args().output)
