"""Generate and statically validate a research plan. Never import ML or run jobs.

Only standard-library JSON/text I/O and arithmetic are used. Existing datasets,
model checkpoints and experiment outputs are never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SCHEMA = 'lst-discrete-design-20260925-v2'
QUANTIZERS = ('vq', 'fsq', 'lfq', 'bsq')
SEEDS = (17, 23, 42)
N = 218975
BATCH = 64
EPOCHS = 5
UPDATES = EPOCHS * math.ceil(N / BATCH)
SPLITS = {'train': N, 'val': 11684, 'test': 11757}
LEVELS = {8: [8, 8, 4], 10: [8, 8, 16], 12: [8, 8, 8, 8]}


def write_json(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bottleneck(model, bits=10):
    if model in QUANTIZERS or model in ('vq_ema', 'fsq1000'):
        q = {'vq_ema': 'vq', 'fsq1000': 'fsq'}.get(model, model)
        levels = [8, 5, 5, 5] if model == 'fsq1000' else LEVELS[bits] if q == 'fsq' else None
        states = math.prod(levels) if levels else 2 ** bits
        internal_dim = len(levels) if levels else 32 if q == 'vq' else bits
        return dict(kind='discrete', quantizer=q, layout='single_scale', grid=[16, 16],
                    token_count=256, bits_per_token=bits, states=states, levels=levels,
                    internal_dim=internal_dim, shared_decoder_interface_dim=32,
                    token_bits=256 * bits, same_rate_group=f'grid16_fixed{bits}',
                    codebook_update='ema_restart' if model == 'vq_ema' else 'gradient' if q == 'vq' else 'not_applicable',
                    fixed_bits_are_actual_file_size=False)
    if model == 'var_multiscale':
        scales = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
        count = sum(s * s for s in scales)
        return dict(kind='discrete', quantizer='var_multiscale_vq', layout='multiscale',
                    scales=scales, token_count=count, bits_per_token=10, states=1024,
                    internal_dim=32, token_bits=count * 10,
                    same_rate_group='var680_fixed10', fixed_bits_are_actual_file_size=False)
    if model == 'ae':
        return dict(kind='continuous_deterministic', latent_shape=[32, 16, 16],
                    nominal_bits=None, same_rate_group=None)
    if model in ('sd', 'sdxl', 'flux1'):
        return dict(kind='continuous_kl', latent_shape=[16 if model == 'flux1' else 4, 32, 32],
                    nominal_bits=None, same_rate_group=None, evaluation_posterior='mode')
    return dict(kind='masked_autoencoder', patch_size=16, spatial_patch_count=256,
                nominal_bits=None, same_rate_group=None)


def make_jobs():
    jobs = []

    def add(module, model, rep, mask, seed=17, bits=10, mean='oracle_mean', track='pixel'):
        job_id = f'{module}__{model}__{rep}__{mask}__{mean}__b{bits}__s{seed}'
        bn = bottleneck(model, bits)
        side_bits = 32  # Absolute controls receive the SAME mean, visible_mean sends its computed mean.
        native_mae = model in ('prithvi_mae', 'satmae_pp')
        jobs.append(dict(
            id=job_id, module=module, model=model, representation=rep, train_mask=mask,
            seed=seed, mean_condition=mean, track=track, status='design_only_not_launched',
            execution_enabled=False, training_launched=False, dataset='hlp_v2_lst',
            prepared='processed_data/hlp_split_v2_20260914', split_sizes=SPLITS,
            epochs=EPOCHS, effective_batch=BATCH, updates_per_epoch=math.ceil(N / BATCH),
            final_update_samples=N % BATCH, updates=UPDATES,
            record_presentations=N * EPOCHS, training_views_per_record=1,
            initialization='official_pretrained_unseen_hlp' if model in ('sd', 'sdxl', 'flux1') else 'scratch',
            budget_interpretation='equal_data_budget_not_convergence_or_equal_compute',
            condition_contract='values_valid_observed_mean_adapter_v1',
            mask_rng_includes_model=False, mask_rng_includes_training_seed=False,
            checkpoint_rule='last_epoch_5', test_for_selection=False,
            main_output='complete_reconstruction_no_observed_pixel_copy',
            output_range_contract='common_physical_projection_and_unprojected_diagnostics',
            objective=('native_masked_reconstruction' if mask == 'patch10' else 'full_image_ae_control')
                if native_mae else 'physical_plus_algorithm_specific_regularization',
            representation_aux_weight=0.0 if module == 'inverse_loss_ablation' or native_mae else 0.01,
            field_contract='absolute_global_center_no_mu_addback' if mean == 'absolute_with_same_mean' else 'residual_condition_center_and_addback',
            bottleneck=bn, mean_side_bits=side_bits,
            nominal_token_and_mean_bits=bn.get('token_bits', 0) + side_bits if bn.get('token_bits') is not None else None,
            nominal_payload_excludes=['valid_support', 'mask_if_transmitted', 'container_headers', 'shared_model'],
            counted_as_training=True,
            prerequisites=['user_requests_execution', 'protocol_freeze', 'data_audit', 'mapping_gradients',
                'quantizer_adapters', 'parameter_latent_audit', 'budget_resume', 'resource_profile'],
        ))

    for q, r, m, s in product(QUANTIZERS, ('R', 'G', 'C'), ('clean', 'pixel10'), SEEDS):
        add('core', q, r, m, s)
    for r, m in product(('R', 'G', 'C'), ('clean', 'pixel10')):
        add('continuous_ablation', 'ae', r, m)
        add('multiscale', 'var_multiscale', r, m)
    for model, r, m in product(('sd', 'sdxl', 'flux1'), ('G', 'C'), ('clean', 'pixel10')):
        add('pretrained', model, r, m)
    for q, r, mean in product(('vq', 'fsq'), ('R', 'C'), ('absolute_with_same_mean', 'visible_mean')):
        add('mean_ablation', q, r, 'pixel10', mean=mean)
    for q in QUANTIZERS:
        add('gain_control', q, 'C_affine', 'pixel10')
    for q, r in product(('fsq1000', 'vq_ema'), ('R', 'C')):
        add('quantizer_recipe', q, r, 'pixel10')
    for q, r in product(('vq', 'fsq'), ('R', 'C')):
        add('inverse_loss_ablation', q, r, 'pixel10')
    for model, m in product(('prithvi_mae', 'satmae_pp'), ('clean', 'patch10')):
        add('mae_patch', model, 'R', m, track='patch')
    for q, r in product(('vq', 'fsq'), ('R', 'C')):
        add('mae_patch', q, r, 'patch10', track='patch')
    for q, r, b in product(QUANTIZERS, ('R', 'C'), (8, 12)):
        add('rate_sweep', q, r, 'pixel10', bits=b)
    return jobs


def make_comparisons(jobs):
    def ids(module, **filters):
        return [j['id'] for j in jobs if j['module'] == module and all(j[k] == v for k, v in filters.items())]
    groups = []
    for rep, mask, seed in product(('R', 'G', 'C'), ('clean', 'pixel10'), SEEDS):
        groups.append(dict(id=f'quantizer_{rep}_{mask}_s{seed}', kind='quantizer_mechanism',
            job_ids=ids('core', representation=rep, train_mask=mask, seed=seed),
            primary_contrast='fsq_minus_vq', same_nominal_rate=True, same_backbone=True,
            same_algorithm_specific_loss=False))
    for q, mask, seed in product(QUANTIZERS, ('clean', 'pixel10'), SEEDS):
        groups.append(dict(id=f'representation_{q}_{mask}_s{seed}', kind='representation_mechanism',
            job_ids=ids('core', model=q, train_mask=mask, seed=seed),
            primary_contrast='C_minus_R', supporting_contrast='C_minus_G',
            same_nominal_rate=True, same_backbone=True))
    for q, rep in product(QUANTIZERS, ('R', 'C')):
        groups.append(dict(id=f'rate_{q}_{rep}', kind='rate_distortion',
            job_ids=ids('rate_sweep', model=q, representation=rep)
                + ids('core', model=q, representation=rep, train_mask='pixel10', seed=17),
            same_nominal_rate=False, same_grid=True))
    for module in ('continuous_ablation', 'pretrained', 'multiscale', 'mean_ablation', 'gain_control', 'quantizer_recipe', 'inverse_loss_ablation', 'mae_patch'):
        groups.append(dict(id=module, kind='supporting_or_system_comparison', job_ids=ids(module),
            same_nominal_rate=False,
            interpretation='Pair with matching core controls; this list is not one universal leaderboard.'))
    return groups


def main():
    jobs = make_jobs()
    cells = set()
    for j in jobs:
        for split, cond in product(('val', 'test'), ('clean', 'patch10') if j['track'] == 'patch' else ('clean', 'pixel10')):
            cells.add((j['id'], split, cond))
    # Reuse existing clean core states, not extra training.
    for j in jobs:
        if j['module'] == 'core' and j['model'] in ('vq', 'fsq') and j['representation'] in ('R', 'C') and j['train_mask'] == 'clean' and j['seed'] == 17:
            for split in ('val', 'test'):
                cells.add((j['id'], split, 'patch10'))
    zero = [dict(id=f'zero__{m}__{r}', model=m, representation=r, initialization='official_pretrained_identity_adapter',
                 counted_as_training=False, execution_enabled=False, status='design_only_not_evaluated')
            for m, r in product(('sd', 'sdxl', 'flux1'), ('G', 'C'))]
    for z in zero:
        for split, cond in product(('val', 'test'), ('clean', 'pixel10')):
            cells.add((z['id'], split, cond))

    gates = [
        ('user_requests_execution', '用户目前明确暂不运行', 'blocked_by_user_instruction'),
        ('protocol_freeze', '表示域、损失、官方revision、条件接口、搜索预算全部冻结', 'not_done'),
        ('data_audit', 'v2成员、train统计、全范围、mask、footprint和重复证据', 'not_done'),
        ('mapping_gradients', '往返、物理初值、并列分支、端点死区与局部增益', 'not_done'),
        ('quantizer_adapters', 'VQ/FSQ/LFQ/BSQ原生机制与同格/bit接口', 'not_done'),
        ('parameter_latent_audit', '实测有效参数、latent shape与无旁路', 'not_done'),
        ('budget_resume', '尾批样本加权、epoch边界、断点RNG与不中断等价', 'not_done'),
        ('bitstream_roundtrip', '索引、均温、支持与容器的实际码流', 'not_done'),
        ('resource_profile', '在未来获准后测显存、吞吐与GPU成本', 'not_done'),
        ('external_confirmation', '未用于设计的独立确认集资格和精确成员', 'not_done'),
    ]
    registries = dict(
        dataset=dict(id='hlp_v2_lst', name='HLS-LST-Pairs', prepared='processed_data/hlp_split_v2_20260914',
                     split_sizes=SPLITS, tile_counts={'train': 2796, 'val': 147, 'test': 151},
                     unit='degree_Celsius', decode='ds.denormalize(image)/100.0',
                     support='reference_mask', split_policy='frozen_tile_disjoint_not_verified_footprint_disjoint',
                     historical_test_exposure=True, external_confirmation_verified=False),
        representations={
            'R': dict(channels=1, forward='field/A', inverse='A*y', role='native_scalar'),
            'G': dict(channels=3, forward='repeat(field/A,3)', inverse='A*mean(y)', role='channel_control'),
            'C': dict(channels=3, forward='2*g((field/A+1)/2)-1', inverse='nearest_segment_inverse', role='piecewise_linear_color_hypothesis'),
            'C_affine': dict(channels=3, forward='[-1,1,7*field/A]', inverse='A*y[2]/7', role='affine_gain_control', bounded_rgb=False),
            'common': dict(A_celsius=120.0, status='conservative_reference_not_freeze_approved',
                input_clipping=False, residual_loss_scale_celsius=2.6950881812941048,
                source_fraction_images_entirely_central_segment=0.907372987784,
                natural_image_distribution_match_claim=False,
                cube_vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1],[1,1,1]],
                mean_storage='float32_both_centering_and_reconstruction',
                scratch_physical_initial_prediction='zero_residual_after_representation_inverse',
                color_zero_residual_head_bias=[-1,1,0], gradients_verified=False),
        },
        field_contracts={
            'residual_condition_center_and_addback': dict(field='T-mean_f32', missing_fill=0,
                readout='mean_f32+inverse_representation(prediction)', scratch_initial_temperature='mean_f32'),
            'absolute_global_center_no_mu_addback': dict(field='T-c_train', c_train='train_only_pixel_weighted_global_mean_pending_fit',
                same_A_as_residual=True, missing_fill=0, readout='c_train+inverse_representation(prediction)',
                scratch_initial_temperature='c_train', oracle_mean_available_to_encoder_and_decoder=True,
                claim='output_parameterization_including_shortcut_and_initialization_not_pure_information_change'),
        },
        readout_order=['representation_prediction','raw_inverse_with_C_intrinsic_segment_projection',
            'training_physical_and_representation_loss','evaluation_common_field_domain_projection',
            'evaluation_common_temperature_projection'],
        representation_auxiliary=dict(default_weight=0.01, status='proposal_requires_gradient_preflight',
            formula='mean_valid(sum_channels((prediction-F(field))**2)*A**2/(k*s**2))',
            derivative_norm_factors={'R':1,'G':3,'C':49,'C_affine':49},
            global_equivalence_to_physical_loss=False),
        quantizers={q:bottleneck(q) for q in QUANTIZERS},
        fsq_exact_levels_by_bits={str(k): v for k,v in LEVELS.items()},
        model_sources={
            'shared_conv': 'https://github.com/FoundationVision/VAR',
            'fsq': 'https://raw.githubusercontent.com/google-research/google-research/master/fsq/fsq.ipynb',
            'lfq': 'https://arxiv.org/html/2310.05737v3',
            'bsq': 'https://github.com/zhaoyue-zephyrus/npq-vit',
            'prithvi_mae': 'https://github.com/NASA-IMPACT/Prithvi-EO-2.0',
            'satmae_pp': 'https://github.com/techmn/satmae_pp',
            'sd': 'https://huggingface.co/stabilityai/sd-vae-ft-mse',
            'sdxl': 'https://huggingface.co/stabilityai/sdxl-vae',
            'flux1': 'https://huggingface.co/black-forest-labs/FLUX.1-schnell',
        },
        new_model_implementations_ready=False, official_revisions_frozen=False,
        parameters=dict(target_context='roughly_84_to_87M_not_a_hard_requirement',
            shared_backbone='var_conv_ch128_resblocks3_chmult_1_1_2_2_4',
            previous_multiscale_rgb_probe=87225699, previous_sd_probe=83653863,
            current_adapted_models_measured=False, forbid_unused_parameter_padding=True),
        regularizers={
            'vq':'codebook_gradient_plus_commitment_candidate_beta0.25',
            'fsq':'none_native_round_STE',
            'lfq':'native_commitment_and_entropy_adapter_coefficients_unfrozen',
            'bsq':'native_spherical_and_entropy_no_automatic_vq_commitment',
            'kl':'element_mean_KL_candidate_1e-6_pending_api_audit',
        },
        masks=dict(pixel10='uniform_without_replacement_round_0.1_Nvalid',
            patch10=dict(patch_size=16, selected=26, total=256, nominal_fraction=26/256),
            input='values_R_O_mu_1x1_adapter_identity_values_zero_other_weights',
            decoder='common_zero_initialized_mean_condition_affine_branch_no_observed_value_skip',
            rng_keys=['split','record_id','epoch','view','mask_seed'], mask_seed=20260925,
            evaluation_views=2, clean_views_independent=False),
    )

    pending = [
        dict(model='eo_vae', reason='temperature_product_modality_adapter_not_verified', planned_track='EO_system'),
        dict(model='terramind_tokenizer', reason='LST_modality_and_native_tokenizer_adapter_not_verified', planned_track='EO_system'),
        dict(model='rq_vae', reason='residual_depth_vs_spatial_grid_rate_contract_needed', planned_track='architecture_extension'),
        dict(model='leech_spherical_quantization', reason='recent_candidate_needs_primary_implementation_rate_audit', planned_track='quantizer_extension'),
    ]
    eval_extensions = [
        dict(id='mask_stress', type='evaluation_only', conditions=['pixel05','pixel20','pixel40'], selection='predeclared_core_R_C_seed17', membership_frozen=False),
        dict(id='mean_stress', type='evaluation_only', offsets_celsius=[-2,-1,-0.5,0,0.5,1,2], rule='same_offset_for_centering_and_add_back', membership_frozen=False),
        dict(id='external_confirmation', type='evaluation_only', dataset=None, independence_verified=False),
        dict(id='frozen_token_prediction', type='new_downstream_training_not_in_140', config_frozen=False),
        dict(id='hls_six_band_extension', type='future_protocol', config_frozen=False),
        dict(id='color_range_redesign', type='separate_representation_version', config_frozen=False),
    ]

    groups = make_comparisons(jobs)
    source_files = ['AGENTS.md', 'experiments/paper1/lst_known_mean_design_20260924/design_manifest.json',
        'runs/paper1/lst_residual_scale_20260918/residual_scale.json',
        'data_inspection/hlp_split_20260914/split_summary.json']
    hashes = {name: sha(PROJECT/name) for name in source_files}
    counts = Counter(j['module'] for j in jobs)
    manifest = dict(schema=SCHEMA, status='design_only_execution_disabled', execution_enabled=False,
        training_launched=False, no_training_entrypoint=True, registries=registries,
        jobs=jobs, comparisons=groups, zero_finetuning_states=zero,
        full_evaluation_cells=[dict(state_id=s, split=p, condition=c, expected_unique_records=SPLITS[p],
            views=2, views_are_not_independent_records=True) for s,p,c in sorted(cells)],
        pending_models=pending, separately_scoped_extensions=eval_extensions,
        freeze_gates=[dict(id=i, requirement=r, status=s) for i,r,s in gates], source_hashes=hashes,
        counts=dict(training_jobs=len(jobs), core_training=counts['core'], core_seed17=24,
            core_replication_additional=48, supporting_training=len(jobs)-counts['core'],
            by_module=dict(counts), zero_states=len(zero), full_evaluation_cells=len(cells),
            full_budget_record_presentations=len(jobs)*N*EPOCHS, full_budget_updates=len(jobs)*UPDATES),
        statistics=dict(primary_metrics=['clean_pooled_RMSE_C','pixel10_missing_pooled_RMSE_C'],
            required_companions=['spatial_RMSE_C','gradient_RMSE_C_per_pixel','observed_RMSE_C','variance_ratio'],
            bootstrap_unit='tile', bootstrap_draws=2000, seed_uncertainty_reported_separately=True,
            primary_contrasts=['C_minus_R_within_VQ_or_FSQ','FSQ_minus_VQ_within_representation'],
            multiple_comparison_policy='preregistered_family_Holm_if_significance_claimed',
            no_test_tuning=True, blind_new_test_claim=False),
    )

    assert len(jobs) == 140 and len({j['id'] for j in jobs}) == len(jobs)
    assert counts['core'] == 72 and UPDATES == 17110 and N % BATCH == 31
    assert len(cells) == 592
    by_id = {j['id']:j for j in jobs}
    for j in jobs:
        assert not j['execution_enabled'] and not j['training_launched'] and not j['test_for_selection']
        assert set(j['prerequisites']) <= {g[0] for g in gates}
        assert j['field_contract'] in registries['field_contracts']
        assert j['record_presentations'] == 1094875 and j['updates'] == 17110
        bn = j['bottleneck']
        if bn['kind'] == 'discrete':
            assert math.ceil(math.log2(bn['states'])) == bn['bits_per_token']
            assert bn['token_bits'] == bn['token_count'] * bn['bits_per_token']
            assert j['nominal_token_and_mean_bits'] == bn['token_bits'] + 32
            if bn.get('levels'):
                assert math.prod(bn['levels']) == bn['states']
        else:
            assert j['nominal_token_and_mean_bits'] is None
    for group in groups:
        assert all(i in by_id for i in group['job_ids'])
        if group.get('same_nominal_rate'):
            assert len({by_id[i]['nominal_token_and_mean_bits'] for i in group['job_ids']}) == 1
            assert len({by_id[i]['bottleneck']['token_count'] for i in group['job_ids']}) == 1
    eval_ids = set(by_id) | {z['id'] for z in zero}
    assert all(s in eval_ids for s,_,_ in cells)

    write_json('design_manifest.json', manifest)
    write_json('registries.json', registries)
    write_json('comparison_groups.json', groups)
    (HERE/'experiments.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs),encoding='utf-8')
    validation = dict(status='static_design_checks_passed_not_model_preflight',
        schema=SCHEMA, training_started_by_this_script=False, gpu_used_by_this_script=False,
        counts=manifest['counts'], passed_checks=['unique_job_ids','complete_factorial_core',
            'epoch_tail_and_update_arithmetic','fixed_width_payload_and_FSQ_state_products',
            'comparison_group_rate_contract','evaluation_dedup_and_record_counts','execution_disabled_everywhere',
            'prerequisite_references_resolve','absolute_and_residual_field_contracts_resolve'],
        not_checked=[g[0] for g in gates], output_hashes={f:sha(HERE/f) for f in
            ('design_manifest.json','registries.json','comparison_groups.json','experiments.jsonl')})
    write_json('validation.json', validation)
    print(json.dumps(dict(status=validation['status'], counts=manifest['counts'], execution_enabled=False),ensure_ascii=False))


if __name__ == '__main__':
    main()
