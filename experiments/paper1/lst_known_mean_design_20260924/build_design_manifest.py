"""Build a design inventory and read-only evidence audit; never train models."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
moments = PROJECT / 'runs/openimages_hlp_color_matched_20260917/results/hlp_per_image_moments.npz'
scale_path = PROJECT / 'runs/paper1/lst_residual_scale_20260918/residual_scale.json'
scale = json.loads(scale_path.read_text(encoding='utf-8'))
with np.load(moments) as a:
    mu, lo, hi = (a[k].ravel() for k in ('lst_mean','lst_minimum','lst_maximum'))
    max_abs = np.maximum(mu-lo, hi-mu)
    stats = dict(samples=len(mu), temperature_min=float(lo.min()), temperature_max=float(hi.max()),
                 residual_std_celsius=float(np.sqrt(a['lst_cov'].mean())),
                 per_image_max_abs_residual_quantiles={str(q):float(np.quantile(max_abs,q)) for q in (.5,.9,.95,.99,.999,1.)},
                 fraction_images_entirely_in_central_segment_at_A120=float(np.mean(max_abs < 120/7)))
assert stats['samples'] == 218975
assert abs(stats['residual_std_celsius']-scale['residual_std_celsius']) < 1e-10

jobs = []
def add(model, rep, mask, track, init, objective):
    jobs.append(dict(id=f'lst_mean_{model}_{rep}_{mask}_s17', model=model, representation=rep,
                     train_mask=mask, track=track, initialization=init, objective=objective,
                     status='design_only_not_launched', seed=17, epochs=5,
                     record_presentations=1094875, updates=17110, effective_batch=64))
for rep in ('R','G','C'):
    for mask in ('clean','pixel10'):
        add('var_vq',rep,mask,'pixel_main','random','physical_plus_vq')
for model in ('sd_vae','sdxl_vae','flux1_vae'):
    for rep in ('G','C'):
        for mask in ('clean','pixel10'):
            add(model,rep,mask,'pixel_main','official_pretrained','physical_plus_kl')
for model in ('prithvi_mae','satmae_pp'):
    for mask in ('clean','patch10'):
        add(model,'R',mask,'mae_patch','random',
            'full_image_ae_control' if mask=='clean' else 'masked_reconstruction_native_mechanism')
for rep in ('R','C'):
    add('var_vq',rep,'patch10','mae_patch','random','physical_plus_vq')

pending=[dict(model=m,representation='R_or_validated_native_adapter',train_mask=mask,
              status='lst_adapter_unresolved_not_scheduled')
         for m in ('eo_vae','terramind_tokenizer') for mask in ('clean','pixel10')]
zero=[dict(model=m,representation=r,status='planned_not_evaluated')
      for m in ('sd_vae','sdxl_vae','flux1_vae') for r in ('G','C')]
cells=set()
for job in jobs:
    conditions=('clean','pixel10') if job['track']=='pixel_main' else ('clean','patch10')
    for split in ('val','test'):
        for condition in conditions:
            cells.add((job['id'],split,condition))
for rep in ('R','C'):
    for split in ('val','test'):
        cells.add((f'lst_mean_var_vq_{rep}_clean_s17',split,'patch10'))
for z in zero:
    for split in ('val','test'):
        for condition in ('clean','pixel10'):
            cells.add((f"zero_{z['model']}_{z['representation']}",split,condition))
assert len(jobs)==24 and len({x['id'] for x in jobs})==24
assert sum(j['track']=='pixel_main' for j in jobs)==18
assert len(cells)==124

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
moment_hash=sha(moments)
assert moment_hash == scale['source_moments_sha256']
manifest=dict(schema_version=1,status='design_only_requires_implementation_preflight',training_launched=False,
              dataset=dict(prepared='processed_data/hlp_split_v2_20260914',modality='lst',train=218975,val=11684,test=11757),
              confirmed=dict(mean='complete_valid_reference_mean_given_to_all',main_mask='random10_pixels',mae='separate_patch_completion'),
              mapping=dict(kind='linear_residual_normalization_then_piecewise_linear_rgb',A_celsius=120,
                           A_status='conservative_proposal_pending_reader_range_preflight',clipping_input=False),
              statistics=stats,
              source_hashes={str(moments.relative_to(PROJECT)):moment_hash,str(scale_path.relative_to(PROJECT)):sha(scale_path)},
              parameter_probe=dict(method='torch_meta_instantiation_no_forward_no_training',
                  var_rgb_128_blocks2=69771619,var_rgb_128_blocks3=87225699,var_rgb_160_blocks2=108850051,
                  sd_local_parameters=83653863,selected_var_relative_difference=87225699/83653863-1,
                  other_models_alignment_measured=False),
              jobs=jobs,pending_adapter_jobs=pending,zero_finetuning_states=zero,
              evaluation_cells=[dict(state=s,split=p,condition=c) for s,p,c in sorted(cells)],
              counts=dict(main_training=18,additional_mae_track_training=6,total_designed_training=24,
                          pending_adapter_training=4,zero_finetuning_states=6,unique_full_eval_cells=124),
              freeze_gates=['reader_valid_range_and_data_hash','color_segment_occupancy_and_roundtrip',
                            'actual_parameter_latent_counts','pinned_official_weights_and_posterior_api',
                            'pixel_mask_and_native_mae_mask_no_double_corruption',
                            'tail_gradients_and_resume','memory_and_throughput'])
(ROOT/'design_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(counts=manifest['counts'], statistics=stats,parameter_delta=manifest['parameter_probe']['selected_var_relative_difference']),ensure_ascii=False))
