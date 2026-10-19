"""Source-backed result tables, paired audits, and interrupted-job costs."""
from datetime import datetime
import json
from pathlib import Path

from .config import file_digest, write_json, is_pretrained


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def union_seconds(intervals):
    if not intervals:
        return 0.
    intervals = sorted(intervals)
    start, end = intervals[0]
    total = 0.
    for left, right in intervals[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def cost_audit(job_directories):
    """Preserve raw logs; count only evidenced session intervals, never downtime."""
    jobs, intervals, gpu_ids = [], [], set()
    for directory in job_directories:
        directory = Path(directory)
        sources, sessions, damaged_tails = {}, {}, []
        for path in sorted(directory.rglob("sessions*.jsonl")):
            sources[str(path)] = file_digest(path)
            lines = path.read_text(encoding="utf-8").splitlines()
            events = []
            for line_number, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    if line_number != len(lines) - 1:
                        raise
                    damaged_tails.append({"path": str(path), "line": line_number + 1})
            for event in events:
                sessions.setdefault(event["session"], []).append(event)
        training, cpu_training, evaluation, incomplete = 0., 0., 0., []
        for sid, events in sessions.items():
            starts = [e for e in events if e["event"] == "start"]
            if len(starts) != 1:
                raise ValueError("Missing/duplicate session start: " + sid)
            start = starts[0]
            timed = [e for e in events if "elapsed_seconds" in e]
            elapsed = max((e["elapsed_seconds"] for e in timed), default=0.)
            ended = any(e["event"] in ("end", "interrupted") for e in events)
            if not ended:
                incomplete.append(sid)
            if start["stage"] == "training":
                if str(start.get("device", "")).startswith("cuda"):
                    training += elapsed
                    gpu_ids.add(start.get("gpu_uuid") or "unknown")
                    timestamp = datetime.fromisoformat(start["utc"]).timestamp()
                    intervals.append((timestamp, timestamp + elapsed))
                else:
                    cpu_training += elapsed
            else:
                evaluation += elapsed
        jobs.append({"directory": str(directory), "training_shared_gpu_job_hours_lower_bound": training / 3600,
                     "evaluation_job_hours_lower_bound": evaluation / 3600,
                     "cpu_training_job_hours_lower_bound": cpu_training / 3600,
                     "unknown_exit_sessions": incomplete, "damaged_log_tails": damaged_tails, "source_sha256": sources})
    single_gpu_verified = len(gpu_ids) == 1 and "unknown" not in gpu_ids
    return {"jobs": jobs, "training_interval_union_hours_lower_bound": union_seconds(intervals) / 3600,
            "single_gpu_identity_verified": single_gpu_verified, "gpu_uuids": sorted(gpu_ids),
            "single_card_training_gpu_hours_lower_bound": union_seconds(intervals) / 3600 if single_gpu_verified else None,
            "assumption": "Union is a single-card GPU-hour bound only if all jobs share the same physical GPU and exclude CPU work",
            "note": "Includes checkpoint/loader/validation if inside session, GPU contention and recomputation; not pure compute. "
                    "Unknown exits use last evidenced elapsed time; downtime excluded. Per-job hours are not summed as physical GPU-hours."}


def compare_cells(cell_a, cell_b, output):
    from .metrics import paired_comparison
    a, b = _load(Path(cell_a) / "metrics.json"), _load(Path(cell_b) / "metrics.json")
    for key in ("split", "condition", "physical_input_sha256", "row_count", "unique_records"):
        if a[key] != b[key]:
            raise ValueError(f"Unpaired input {key}; refuse metric-only comparison")
    for path, cell in ((cell_a, a), (cell_b, b)):
        if file_digest(Path(path) / "rows_raw.jsonl") != cell["artifact_sha256"]["rows_raw.jsonl"]:
            raise ValueError("Rows changed since evaluation")
    bootstrap_a, bootstrap_b = a["raw"]["tile_cluster_bootstrap"], b["raw"]["tile_cluster_bootstrap"]
    if (bootstrap_a["seed"], bootstrap_a["replicates"]) != (bootstrap_b["seed"], bootstrap_b["replicates"]):
        raise ValueError("Different bootstrap protocols")
    result = paired_comparison(_rows(Path(cell_a) / "rows_raw.jsonl"), _rows(Path(cell_b) / "rows_raw.jsonl"),
                               bootstrap_seed=bootstrap_a["seed"], bootstrap_replicates=bootstrap_a["replicates"])
    result.update(cell_a=str(cell_a), cell_b=str(cell_b), same_mean=a["mean_sha256"] == b["mean_sha256"],
                  same_encoded_input=a["encoded_input_sha256"] == b["encoded_input_sha256"],
                  causal_fairness="Input pairing only; model initialization, parameter and rate track must also match")
    write_json(output, result)
    return result


def build_report(job_directory):
    directory = Path(job_directory)
    cells = sorted(directory.rglob("metrics.json"))
    if not cells:
        raise ValueError("No completed evaluation cells; no experimental results to report")
    config = _load(directory / "config.json")
    expected_phases = ("pretrained", "final") if is_pretrained(config) else ("final",)
    phase_checks = {phase: (_load(directory / phase / "completion_checks.json")
                   if (directory / phase / "completion_checks.json").exists() else None) for phase in expected_phases}
    from .evaluate import condition_id
    expected_relative = {f"{split}/{condition_id(profile)}" for split in config["evaluation"]["splits"]
                         for profile in config["evaluation"]["profiles"]}
    final_complete = all(value and value.get("passed") for value in phase_checks.values())
    for phase, check in phase_checks.items():
        found = {str(p.parent.relative_to(directory / phase)).replace("\\", "/")
                 for p in (directory / phase).rglob("metrics.json")}
        if found != expected_relative or not check:
            final_complete = False
            continue
        registered = {f"{x['split']}/{x['condition']}" for x in check["cells"]}
        if registered != found:
            final_complete = False
        for cell in check["cells"]:
            metric_path = directory / phase / cell["split"] / cell["condition"] / "metrics.json"
            if not metric_path.exists() or _load(metric_path)["checkpoint_sha256"] != cell["checkpoint_sha256"]:
                final_complete = False
    title = "完整数值结果（视觉复核待完成）" if final_complete else "部分结果（不能视为最终结论）"
    if final_complete and config["evaluation"].get("stage") == "selection":
        title = "验证阶段数值结果（不含测试集；视觉复核待完成）"
    lines = ["# LST 图像复原：" + title, "", f"配置：`{config['name']}`。单位为摄氏度。",
             "下游任务不在本轮范围；以下合成污染结果不自动代表真实传感器噪声或云雾恢复能力。",
             "连续 AE/KL-VAE 与离散模型分轨，不宣称相同码率；地球空间 MAE 列为结构对照。", "",
             f"已发现 {len(cells)} 个完成评估单元；预期阶段：{', '.join(expected_phases)}。仅预训练结果不代表微调结果。", "",
             "|评估阶段/划分/污染|全图 RMSE|空间 RMSE|梯度 RMSE|缺失区 RMSE|均温 RMSE|记录数|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for path in cells:
        value = _load(path)
        for name, expected in value["artifact_sha256"].items():
            if file_digest(path.parent / name) != expected:
                raise ValueError("Report evidence changed: " + str(path.parent / name))
        m = value["raw"]
        missing = m["missing_region"]["pixel_weighted_rmse"]
        fmt = lambda x: "—" if x is None else f"{x:.5f}"
        label = str(path.parent.relative_to(directory)).replace("\\", "/")
        lines.append(f"|{label}|{fmt(m['pixel_weighted_rmse'])}|{fmt(m['spatial_rmse'])}|{fmt(m['gradient_rmse'])}|"
                     f"{fmt(missing)}|{fmt(m['mean_temperature_rmse'])}|{value['unique_records']}|")
    audit = cost_audit([directory])
    write_json(directory / "cost_audit.json", audit)
    lines += ["", "主表使用 raw 温度读出；共同物理范围投影的结果另存每个 metrics.json/projected。",
              "宏平均、像素加权和 tile 配对区间各有明确字段；两视图不算两倍独立记录。",
              "码本困惑度只对离散模型报告。无训练的观测保留＋均值填空基线也在每个评估单元中。", "",
              "## 成本口径", "", "训练作业时间包含加载、存盘与共享设备竞争，异常会话取已记录下界。",
              "各作业小时不能相加后称为物理单卡 GPU-hours；成本证据见 cost_audit.json。", "",
              "## 视觉复核", "", "尚未人工复核。gallery.npz 或生成 PNG 的存在不代表已经查看或验证。",
              "不能据此写入‘细节更好’等视觉结论。"]
    destination = directory / "REPORT_ZH.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def render_gallery(cell_directory):
    """Only renders stored predictions; does not run inference or mark review done."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(cell_directory)
    with np.load(directory / "gallery.npz", allow_pickle=False) as z:
        examples = min(8, len(z["sample_id"]))
        figure, axes = plt.subplots(examples, 4, figsize=(14, 3 * examples), squeeze=False)
        for i in range(examples):
            target, pred = z["temperature"][i, 0], z["prediction_raw"][i, 0]
            valid, observed = z["reference_valid"][i, 0], z["observed_valid"][i, 0]
            lo, hi = float(target[valid].min()), float(target[valid].max())
            if hi == lo:
                hi = lo + .01
            error = pred - target
            limit = max(.01, float(np.nanmax(np.abs(error[valid]))))
            fields = (("Reference (C)", np.where(valid, target, np.nan), "viridis", lo, hi),
                      ("Observation", np.where(observed, z["observation"][i, 0], np.nan), "viridis", lo, hi),
                      ("Restored", np.where(valid, pred, np.nan), "viridis", lo, hi),
                      ("Error (C)", np.where(valid, error, np.nan), "coolwarm", -limit, limit))
            for ax, (title, array, cmap, lower, upper) in zip(axes[i], fields):
                im = ax.imshow(array, cmap=cmap, vmin=lower, vmax=upper)
                ax.set_title(title)
                ax.axis("off")
                figure.colorbar(im, ax=ax, shrink=.65)
            axes[i, 0].set_ylabel(str(z["sample_id"][i]))
        figure.tight_layout()
        figure.savefig(directory / "gallery.png", dpi=140)
        plt.close(figure)
    return directory / "gallery.png"


def token_stability(cell_a, cell_b, output):
    """Same weights only; agreement alone cannot distinguish robustness/collapse."""
    import numpy as np
    a, b = _load(Path(cell_a) / "metrics.json"), _load(Path(cell_b) / "metrics.json")
    if a["checkpoint_sha256"] != b["checkpoint_sha256"] or a["split"] != b["split"]:
        raise ValueError("Token IDs can only be compared within one frozen model and split")
    for field in ("source", "stats"):
        if a["data_fingerprint"][field] != b["data_fingerprint"][field]:
            raise ValueError("Token comparison reference data changed")
    for directory, cell in ((cell_a, a), (cell_b, b)):
        if file_digest(Path(directory) / "indices.npz") != cell["artifact_sha256"]["indices.npz"]:
            raise ValueError("Index archive changed")
    with np.load(Path(cell_a) / "indices.npz", allow_pickle=False) as x, np.load(Path(cell_b) / "indices.npz", allow_pickle=False) as y:
        if not np.array_equal(x["sample_id"], y["sample_id"]) or not np.array_equal(x["view"], y["view"]):
            raise ValueError("Unpaired index archive")
        if x["indices"].shape != y["indices"].shape:
            raise ValueError("Different index layout")
        agreement = (x["indices"] == y["indices"]).reshape(len(x["indices"]), -1).mean(1)
        result = {"paired_rows": len(agreement), "mean_token_agreement": float(agreement.mean()),
                  "same_mean": a["mean_sha256"] == b["mean_sha256"],
                  "condition_a": a["condition"], "condition_b": b["condition"],
                  "interpretation": "Must read jointly with restoration error, joint code usage and within-image entropy; constant tokens also agree"}
    write_json(output, result)
    return result


def alignment_audit(paths, output):
    """Measured core group audit. Does not certify equal bitrate of continuous VAEs."""
    from .runtime import source_hashes
    values = [_load(path) for path in paths]
    current = source_hashes()
    if any(v.get("source_hashes") != current or not v.get("gates", {}).get("parameter_rate_audit")
           or v.get("status") != "synthetic_model_checks_complete_other_gates_pending" for v in values):
        raise ValueError("Use complete current model audits")
    kinds = [v["model_metadata"]["kind"] for v in values]
    if sorted(kinds) != ["bsq", "fsq", "lfq", "vq"]:
        raise ValueError("Supply one measured audit for each of VQ/FSQ/LFQ/BSQ")
    reference = values[kinds.index("vq")]
    count = reference["actual_parameter_count"]
    fields = ("latent_grid", "indices_per_image", "codebook_size", "fixed_bits_per_index", "nominal_index_plus_mean_bits")
    rows = []
    for value in values:
        params = value["actual_parameter_count"]
        same_rate = all(value["model_metadata"][key] == reference["model_metadata"][key] for key in fields)
        same_channels = value["synthetic_batch_shape"] == reference["synthetic_batch_shape"]
        relative = abs(params - count) / count
        rows.append({"kind": value["model_metadata"]["kind"], "parameters": params,
                     "relative_parameter_difference_from_vq": relative, "same_rate": same_rate,
                     "same_input_channels": same_channels, "passed": same_rate and same_channels and relative <= .05})
    result = {"passed": all(row["passed"] for row in rows), "rows": rows,
              "parameter_tolerance_fraction": .05, "source_hashes": current,
              "protocol_config_sha256": [v["protocol_config_sha256"] for v in values],
              "limits": "Parameter counts and index payload only; shared architecture/config and physical observations still require protocol audit"}
    write_json(output, result)
    return result
