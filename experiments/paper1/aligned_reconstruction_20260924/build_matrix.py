"""Materialize the proposed experiment inventory; never launches training."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
jobs = []
for modality in ("lst", "hls"):
    for representation, models in (
        ("native_absolute", ("prithvi_ae", "var_vq")),
        ("native_visible_residual", ("prithvi_ae", "var_vq")),
        ("linear_gray", ("var_vq", "sd_kl", "sdxl_kl")),
        ("asinh_gray", ("var_vq", "sd_kl", "sdxl_kl")),
        ("asinh_rgb", ("var_vq", "sd_kl", "sdxl_kl")),
    ):
        for model in models:
            for train_mask in ("clean", "random10_pixels"):
                jobs.append(dict(
                    id=f"{modality}_{representation}_{model}_{train_mask}_s17",
                    category="primary", modality=modality,
                    representation=representation, model=model,
                    train_mask=train_mask,
                    initialization="official_pretrained" if model in ("sd_kl", "sdxl_kl") else "random",
                    mean_source="visible" if representation != "native_absolute" else "no_decoder_mean_side_information",
                ))
    for model in ("prithvi_ae", "var_vq"):
        jobs.append(dict(
            id=f"{modality}_oracle_residual_{model}_random10_pixels_s17",
            category="oracle_mean", modality=modality,
            representation="native_reference_mean_residual", model=model,
            train_mask="random10_pixels", initialization="random",
            mean_source="complete_reference_extra_information",
        ))
    for train_mask in ("clean", "random10_pixels"):
        jobs.append(dict(
            id=f"{modality}_asinh_gray_sd_kl_random_init_{train_mask}_s17",
            category="initialization_control", modality=modality,
            representation="asinh_gray", model="sd_kl",
            train_mask=train_mask, initialization="random", mean_source="visible",
        ))

for job in jobs:
    job.update(status="superseded_by_model_selection_not_launched", seed=17, epochs=5,
               train_records=218975, record_presentations=1094875,
               effective_batch=64, updates=17110, tail_update_records=31,
               evaluation_splits=["val", "test"],
               evaluation_conditions=["clean", "random10_pixels"],
               evaluation_views=2)
    job["hls_spectral_interface"] = (
        "six_shared_per_band_rgb_encodings" if job["modality"] == "hls" and job["representation"] in ("linear_gray", "asinh_gray", "asinh_rgb")
        else "native_joint_six_band" if job["modality"] == "hls" else "single_band"
    )

zero_states = [dict(modality=m, representation=r, model=a, status="planned_not_evaluated")
               for m in ("lst", "hls")
               for r in ("linear_gray", "asinh_gray", "asinh_rgb")
               for a in ("sd_kl", "sdxl_kl")]
assert len(jobs) == 60 and len({x["id"] for x in jobs}) == 60
assert sum(x["category"] == "primary" for x in jobs) == 52
assert all(sum(x["modality"] == m for x in jobs) == 30 for m in ("lst", "hls"))
assert len(zero_states) == 12
result = dict(
    schema_version=1, design_date="2026-09-24",
    status="superseded_by_MODEL_SELECTION_ZH_requires_new_matrix",
    training_launched=False,
    confirmed_mean_policy="visible_main_reference_oracle",
    prepared="processed_data/hlp_split_v2_20260914",
    parameter_alignment=dict(target_approx=83653863, tolerance_fraction=0.05, measured=False),
    codebook=dict(size=1024, dimension=32, scales=[1,2,3,4,5,6,8,10,13,16], tokens_per_encoding=680),
    jobs=jobs, zero_finetuning_states=zero_states,
    counts=dict(primary_training=52, oracle_mean_training=4, random_init_training=4,
                total_training=60, zero_finetuning_states=12,
                full_evaluation_cells_including_supplements=288),
    unresolved_before_freeze=["actual_parameter_counts_and_model_configs", "sdxl_weight_revision_and_shapes",
                              "hls_band_validity_bounds_and_train_only_mapping_statistics",
                              "representation_gradient_checks", "memory_and_resume_preflight"],
    excluded_optional_jobs=["hls_native_per_band_bridge_2", "additional_training_seeds", "codebook_sensitivity"],
)
(ROOT / "experiment_matrix.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(result["counts"], ensure_ascii=False))
