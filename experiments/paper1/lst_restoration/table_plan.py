"""Table-oriented study registry. Standard library only; never launches jobs."""
from collections import Counter
from pathlib import Path

from .config import CORE, default_config, digest, profile, validate, write_json

PLAN_VERSION = "lst-tables-v1-20260927"
STAGES = {
    "P1": "共同离散基线",
    "P2": "代表模型的表示消融与 SD 迁移",
    "P3": "混合污染训练",
    "P4": "地理与自然预训练模型扩展",
    "P5": "均值侧信息对照",
    "P6": "率失真",
    "P7": "独立 MAE 整块补全",
    "P8": "预先指定的多种子复验",
}
TABLES = {
    "T1": ("总体复原", ["clean_rmse", "pixel10_rmse", "pixel10_missing_rmse", "gaussian_rmse", "cloud_rmse", "joint_spatial_rmse", "joint_gradient_rmse"]),
    "T2": ("训练污染消融", ["clean_rmse", "pixel10_rmse", "pixel10_missing_rmse", "gaussian_rmse", "cloud_missing_rmse", "joint_rmse"]),
    "T3": ("表示消融", ["clean_rmse", "pixel10_rmse", "pixel10_spatial_rmse", "pixel10_gradient_rmse", "tail_rmse"]),
    "T4": ("预训练迁移", ["zero_shot_clean_rmse", "zero_shot_pixel10_rmse", "final_clean_rmse", "final_pixel10_rmse", "training_job_hours"]),
    "T5": ("率失真与效率", ["actual_bpp", "pixel10_rmse", "joint_rmse", "encode_ms", "decode_ms", "peak_memory_gib", "training_job_hours"]),
    "T6": ("均值与残差", ["pixel10_rmse", "pixel10_mean_rmse", "pixel10_spatial_rmse", "joint_rmse", "mean_side_bits"]),
    "S1": ("配置与公平性", ["measured_parameters", "index_grid", "nominal_states", "measured_payload_bits", "alignment_passed"]),
    "S2": ("离散状态空间诊断", ["utilization", "joint_entropy_bits", "within_image_entropy_bits", "neighbor_equal_fraction", "cross_condition_agreement", "between_within_decomposition"]),
    "S3": ("MAE 整块补全", ["clean_rmse", "patch16_rmse", "patch16_missing_rmse", "patch16_spatial_rmse", "patch16_gradient_rmse"]),
    "S4": ("多种子与更强污染", ["pixel10_rmse", "pixel10_paired_tile_ci", "strong_gaussian_rmse", "cloud30_rmse", "joint30_rmse"]),
}


def table_study_plan():
    jobs = []

    def add(stage, models, reps, trainings, seeds=(17,), mean="oracle", bits=10):
        for model in models:
            for rep in reps:
                for training in trainings:
                    for seed in seeds:
                        c = default_config(model, rep, training, seed)
                        c["task"]["mean_source"] = mean
                        if mean != "oracle":
                            c["name"] += "_" + mean
                        if bits != 10:
                            c["name"] += f"_rate{bits}"
                            c["model"].update(bits_per_token=bits, codebook_size=2**bits,
                                levels={8: [8, 8, 4], 12: [8, 8, 8, 8]}[bits])
                        c["tier"] = stage
                        if stage == "P7":
                            # Discrete comparator sees the identical missing patch footprint.
                            c["evaluation"]["profiles"] = [profile("clean"), profile("patch16")]
                        else:
                            c["evaluation"]["profiles"] += [profile("sensor_gaussian", sigma_celsius=1.),
                                profile("cloud_missing", coverage=.30),
                                profile("joint", coverage=.30, sigma_celsius=1.)]
                            if stage == "P1" and model in ("vq", "fsq") and training == "clean":
                                c["evaluation"]["profiles"].append(profile("patch16"))
                        c["study"] = {"plan_version": PLAN_VERSION, "stage": stage,
                            "runtime_verified": False, "table_ids": []}
                        jobs.append(c)

    add("P1", CORE, ("R", "C"), ("clean", "pixel10"))
    add("P2", ("vq", "fsq"), ("G", "C_power", "C_asinh"), ("clean", "pixel10"))
    add("P2", ("sd",), ("G", "C", "C_power", "C_asinh"), ("clean", "pixel10"))
    add("P3", CORE, ("R", "C"), ("mixed",))
    add("P3", ("sd",), ("G", "C"), ("mixed",))
    add("P4", ("eo_vae", "terramind_fsq"), ("R", "C"), ("clean", "pixel10", "mixed"))
    add("P4", ("sdxl", "flux1"), ("G", "C"), ("clean", "pixel10", "mixed"))
    add("P5", ("vq", "fsq"), ("R", "C"), ("clean", "pixel10", "mixed"), mean="visible")
    for bits in (8, 12):
        add("P6", CORE, ("R", "C"), ("mixed",), bits=bits)
    add("P7", ("prithvi_mae", "satmae_pp"), ("R", "C"), ("clean", "patch16"))
    add("P7", ("vq", "fsq"), ("R", "C"), ("patch16",))
    add("P8", CORE, ("R", "C"), ("clean", "pixel10", "mixed"), seeds=(23, 42))
    add("P8", ("vq", "fsq"), ("C_power", "C_asinh"), ("pixel10",), seeds=(23, 42))
    # Table membership is a many-to-many mapping; identical training is never duplicated.
    for c in jobs:
        m, r, t = c["model"]["kind"], c["representation"]["kind"], c["corruption"]["training_name"]
        primary = c["seed"] == 17 and c["task"]["mean_source"] == "oracle" and c["model"]["bits_per_token"] == 10
        baseline = primary and r in ("R", "G", "C") and c["tier"] != "P7"
        tables = ["S1"]
        if baseline:
            tables += ["T2"]
            if t == "mixed" and (r != "G" or m in ("sd", "sdxl", "flux1")):
                tables += ["T1"]
        if primary and m in ("vq", "fsq", "sd") and t in ("clean", "pixel10") and c["tier"] != "P7":
            tables += ["T3"]
        if m in ("sd", "sdxl", "flux1"):
            tables += ["T4"]
        if m in CORE and t == "mixed" and r in ("R", "C") and c["seed"] == 17 and c["task"]["mean_source"] == "oracle":
            tables += ["T5"]
        if m in ("vq", "fsq") and r in ("R", "C") and c["seed"] == 17 and c["model"]["bits_per_token"] == 10 and c["tier"] != "P7":
            tables += ["T6"]
        if baseline and m in CORE:
            tables += ["S2"]
        if c["tier"] == "P7" or (primary and m in ("vq", "fsq") and r in ("R", "C") and t == "clean"):
            tables += ["S3"]
        if c["tier"] == "P8" or (primary and c["tier"] != "P7" and m in CORE and (r in ("R", "C") or (r in ("C_power", "C_asinh") and m in ("vq", "fsq") and t == "pixel10"))):
            tables += ["S4"]
        c["study"]["table_ids"] = sorted(set(tables))
    return jobs


def unresolved_cells():
    return [
        {"id": "absolute_without_mean", "tables": ["T6"], "candidate_training_jobs": 12,
         "status": "blocked_implementation", "reason": "Current absolute-field runtime still supplies per-image mean to encoder/decoder; requires a no-mean interface and payload audit.",
         "design": "VQ/FSQ x R/C x clean/pixel10/mixed; no per-image oracle or visible mean side input"},
        {"id": "same_architecture_natural_kl_scratch", "tables": ["T4"], "candidate_training_jobs": None,
         "status": "blocked_implementation", "reason": "Current SD/SDXL/FLUX builder requires pretrained assets. Random-init same-architecture controls are not implemented; do not substitute a different backbone."},
        {"id": "natural_pretrained_discrete", "tables": ["T1", "T3", "T4", "T5"], "candidate_training_jobs": None,
         "status": "blocked_asset_and_adapter", "reason": "Need identified official VQ/FSQ/LFQ/BSQ checkpoint, architecture and license audits. Native pretrained K/grid cannot be resized and still called zero-shot."},
        {"id": "statistics_extensions", "tables": ["T3", "T4", "S2", "S4"], "candidate_training_jobs": 0,
         "status": "blocked_reporting", "reason": "Train-only residual-tail thresholds, between/within decomposition, cross-seed aggregation and convergence-threshold reporting still need implementation; null is not zero."},
    ]


def export_table_plan(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use a new empty directory; preserve existing configs and filled tables")
    jobs = table_study_plan()
    if len({c["name"] for c in jobs}) != len(jobs):
        raise ValueError("Duplicate training job identity")
    for c in jobs:
        result = validate(c)
        if not result["ok"] or c["execution"]["enabled"]:
            raise ValueError(result)
    entries = []
    for c in jobs:
        path = "configs/" + c["name"] + ".json"
        write_json(directory / path, c)
        entries.append({"job_id": c["name"], "stage": c["tier"], "tables": c["study"]["table_ids"],
            "config": path, "config_sha256": digest(c), "state": "planned_disabled",
            "result_directory": None, "source_evidence": None})
    # Evaluation-only reuse proposal: weight hashes and physical input digests must
    # match at execution time; the current runner does NOT implement a shared cache.
    zero_groups = {}
    for c in jobs:
        if c["model"]["kind"] in ("sd", "sdxl", "flux1"):
            key = c["model"]["kind"] + "_" + c["representation"]["kind"]
            zero_groups.setdefault(key, []).append(c["name"])
    manifest = {"plan_version": PLAN_VERSION, "status": "static_plan_only",
        "formal_training_jobs_configured": len(jobs), "training_jobs_started": 0,
        "record_presentations_if_all_configured_jobs_run": sum(c["train"]["record_presentations"] for c in jobs),
        "stages": [{"id": stage, "title": title, "jobs": sum(c["tier"] == stage for c in jobs)} for stage, title in STAGES.items()],
        "jobs": entries, "unresolved_cells": unresolved_cells(),
        "proposed_zero_shot_reuse_groups": zero_groups,
        "shared_zero_shot_cache_implemented": False,
        "global_gates": ["data/component/model preflight", "measured parameter and payload alignment",
            "source/weight assets bound", "nonlinear loss gradient audit", "STOP and free memory checked"],
        "schedule": "P0 preflight -> P1 -> P2 -> P3 -> P4 -> P5/P6/P7 -> P8 -> final table audit; no test-driven pruning"}
    write_json(directory / "manifest.json", manifest)
    table_registry = {}
    for table, (title, columns) in TABLES.items():
        selected = [c for c in jobs if table in c["study"]["table_ids"]]
        rows = []
        lines = [f"# {table} {title}", "", "全部为待填占位；— 不代表零。主结果使用完整 test，val 单独保存。结果必须关联配置、权重与评估证据。", "",
                 "|job_id|模型|表示|训练污染|均值|seed|" + "|".join(columns) + "|状态|",
                 "|" + "---|" * (7 + len(columns))]
        for c in selected:
            row = {"job_id": c["name"], "model": c["model"]["kind"], "representation": c["representation"]["kind"],
                "training": c["corruption"]["training_name"], "mean": c["task"]["mean_source"], "seed": c["seed"],
                "nominal_bits_per_token": c["model"]["bits_per_token"] if c["model"]["kind"] in CORE else None,
                "split": "test", "state": "not_run", "values": dict.fromkeys(columns),
                "config_sha256": digest(c), "evidence": []}
            rows.append(row)
            lines.append("|" + "|".join([row[k] if isinstance(row[k], str) else str(row[k]) for k in
                ("job_id", "model", "representation", "training", "mean", "seed")] + ["—"] * len(columns) + ["未运行"]) + "|")
        pending = [x for x in unresolved_cells() if table in x["tables"]]
        for item in pending:
            lines += ["", f"待补：`{item['id']}` — {item['reason']}"]
        write_json(directory / "tables" / (table + ".json"), {"title": title, "rows": rows, "unresolved": pending})
        (directory / "tables" / (table + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        table_registry[table] = {"title": title, "planned_rows": len(rows), "columns": columns, "unresolved": [x["id"] for x in pending]}
    write_json(directory / "table_registry.json", table_registry)
    return {"plan_version": PLAN_VERSION, "configured_jobs": len(jobs), "stages": dict(Counter(c["tier"] for c in jobs)),
            "tables": len(TABLES), "zero_shot_reuse_groups_proposed": len(zero_groups), "training_jobs_started": 0}
