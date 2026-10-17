"""Complete, restartable val/test cells with separate raw/constrained readouts."""
import hashlib
import json
from pathlib import Path
import time
import uuid

from .config import COUNTS, digest, file_digest, write_json, is_pretrained, pretrained_identity
from .runtime import (amp, append_event, check_stop, execution_guard, load_checkpoint,
                      memory, move_batch, protocol_digest, setup, source_hashes, utc)


def condition_id(profile):
    return profile["kind"] + "_" + digest(profile)[:12]


def _record_digest(hasher, ids, arrays):
    # Record-wise updates do not depend on the selected evaluation batch size.
    for i, sid in enumerate(ids):
        encoded = sid.encode("utf-8")
        hasher.update(len(encoded).to_bytes(4, "little") + encoded)
        for name, array in arrays.items():
            value = array[i]
            hasher.update(name.encode() + str(value.shape).encode() + value.tobytes())


def _write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def evaluate(config, checkpoint, output):
    execution_guard(config)
    import torch
    from eo_data.locking import build_lock
    from .models import build_model
    output = Path(output)
    check_stop(output)
    with build_lock(output):
        if (output / "config.json").exists():
            stored = json.loads((output / "config.json").read_text(encoding="utf-8"))
            if protocol_digest(stored) != protocol_digest(config):
                raise ValueError("Evaluation directory already belongs to a different job")
        else:
            write_json(output / "config.json", config)
        device = setup(config)
        model = build_model(config).to(device)
        pretrained = str(checkpoint) == "pretrained"
        if pretrained:
            if not is_pretrained(config):
                raise ValueError("Zero fine-tune evaluation requires a registered pretrained model")
            weight_hash = digest(pretrained_identity(config))
        else:
            state = load_checkpoint(checkpoint, config)
            if state["step"] != config["train"]["updates"] or state["records"] != config["train"]["record_presentations"]:
                raise ValueError("Only the registered final-epoch checkpoint enters formal evaluation")
            model.load_state_dict(state["model"])
            del state
            weight_hash = file_digest(checkpoint)
        model.eval()
        cells = []
        for split in config["evaluation"]["splits"]:
            for profile in config["evaluation"]["profiles"]:
                path = output / split / condition_id(profile)
                check_stop(output)
                if (path / "metrics.json").exists():
                    result = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
                    if result["checkpoint_sha256"] != weight_hash or result["protocol_config_sha256"] != protocol_digest(config):
                        raise ValueError("An existing evaluation cell belongs to another frozen job")
                    if result.get("source_hashes") != source_hashes():
                        raise ValueError("Evaluation code differs; do not silently reuse old cells")
                    from .data import LSTDataset
                    current_data = LSTDataset(config, split, profile, view=0)
                    try:
                        if current_data.fingerprint != result["data_fingerprint"]:
                            raise ValueError("Evaluation data identity changed")
                    finally:
                        current_data.close()
                    for file, expected in result["artifact_sha256"].items():
                        if file_digest(path / file) != expected:
                            raise ValueError("Existing evaluation rows/gallery changed")
                else:
                    with torch.inference_mode():
                        result = evaluate_cell(model, config, device, split, profile, path, output, weight_hash)
                cells.append({"split": split, "condition": condition_id(profile), "path": str(path),
                              "records": result["unique_records"], "rows": result["row_count"],
                              "checkpoint_sha256": weight_hash})
        expected_cells = len(config["evaluation"]["profiles"]) * 2
        passed = len(cells) == expected_cells and all(c["records"] == COUNTS[c["split"]] and
                  c["rows"] == 2 * COUNTS[c["split"]] for c in cells)
        write_json(output / "completion_checks.json", {"passed": passed, "cells": cells,
            "expected_cells": expected_cells, "phase": "pretrained" if pretrained else "finetuned_or_scratch",
            "visual_review": "not_performed", "cross_model_digest_audit": "pending_separate_comparison"})
        if not passed:
            raise RuntimeError("Incomplete full evaluation")
        return cells


def evaluate_cell(model, config, device, split, profile, output, stop_root, weight_hash):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .data import LSTDataset
    from .representations import Representation
    from .objective import model_inputs, readout
    from .metrics import restoration_rows, summarize_rows
    from .bitstream import encode_indices, decode_indices, rate_audit
    output.mkdir(parents=True, exist_ok=True)
    representation = Representation(config["representation"])
    rows, projected, baseline_rows, rates, previews = [], [], [], [], []
    all_indices = []
    physical_digest, mean_digest, encoded_digest = (hashlib.sha256() for _ in range(3))
    condition = condition_id(profile)
    session = uuid.uuid4().hex
    session_log = output / f"sessions_{session}.jsonl"
    started, model_seconds, decode_seconds = time.perf_counter(), 0., 0.
    append_event(session_log, {"event": "start", "stage": "evaluation", "session": session,
                                          "condition": condition, "split": split, "device": str(device)})
    data_fingerprint = None
    try:
        for view in range(config["evaluation"]["views"]):
            dataset = LSTDataset(config, split, profile, view)
            data_fingerprint = dataset.fingerprint
            seen = []
            try:
                loader = DataLoader(dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False,
                    num_workers=config["train"]["workers"], drop_last=False, pin_memory=device.type == "cuda",
                    generator=torch.Generator().manual_seed(20260925))
                for cpu_batch in loader:
                    check_stop(stop_root)
                    ids = cpu_batch["sample_id"]
                    seen.extend(cpu_batch["record_id"].tolist())
                    _record_digest(physical_digest, ids, {k: cpu_batch[k].numpy() for k in
                        ("temperature", "observation", "reference_valid", "observed_valid", "affected_mask")})
                    _record_digest(mean_digest, ids, {"mean_celsius": cpu_batch["mean_celsius"].numpy()})
                    batch = move_batch(cpu_batch, device)
                    inputs = model_inputs(batch, representation)
                    _record_digest(encoded_digest, ids, {"encoded_input": inputs["encoded_input"].cpu().numpy()})
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    with amp(config, device):
                        predicted = model(**inputs)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    model_seconds += time.perf_counter() - t0
                    raw = readout(predicted, batch, representation, config)
                    constrained = readout(predicted, batch, representation, config, constrained=True)
                    for name, values in (("raw", raw), ("projected", constrained)):
                        if not torch.isfinite(values[batch["reference_valid"]]).all():
                            raise FloatingPointError(f"Non-finite {name} prediction")
                    indices = predicted.get("indices")
                    index_array = indices.detach().cpu().numpy() if indices is not None else None
                    if index_array is not None:
                        # Retain token locations for same-model cross-condition stability.
                        dtype = np.uint16 if config["model"]["codebook_size"] <= 65536 else np.uint32
                        all_indices.append(index_array.astype(dtype))
                    target = cpu_batch["temperature"].numpy()
                    valid, observed = cpu_batch["reference_valid"].numpy(), cpu_batch["observed_valid"].numpy()
                    options = dict(sample_ids=ids, tile_ids=cpu_batch["tile_id"], affected_mask=cpu_batch["affected_mask"].numpy())
                    discrete_options = dict(indices=index_array, codebook_size=config["model"]["codebook_size"] if indices is not None else None)
                    new_rows = restoration_rows(raw.cpu().numpy(), target, valid, observed, **options, **discrete_options)
                    new_projected = restoration_rows(constrained.cpu().numpy(), target, valid, observed, **options)
                    baseline = np.where(observed, cpu_batch["observation"].numpy(), cpu_batch["mean_celsius"].numpy()[:, None, None, None])
                    new_baseline = restoration_rows(baseline, target, valid, observed, **options)
                    for group in (new_rows, new_projected, new_baseline):
                        for row in group:
                            row.update(view=view, condition=condition)
                    rows.extend(new_rows)
                    projected.extend(new_projected)
                    baseline_rows.extend(new_baseline)
                    for i, sid in enumerate(ids):
                        if view == 0 and len(previews) < config["evaluation"]["save_gallery_samples"]:
                            previews.append({"sample_id": sid, "temperature": target[i], "observation": cpu_batch["observation"][i].numpy(),
                                "reference_valid": valid[i], "observed_valid": observed[i], "prediction_raw": raw[i].cpu().numpy(),
                                "prediction_projected": constrained[i].cpu().numpy()})
                        if indices is not None and view == 0 and len(rates) < config["evaluation"]["codec_samples"]:
                            array = index_array[i]
                            mean = float(cpu_batch["mean_celsius"][i])
                            packed = encode_indices(array, array.shape, config["model"]["codebook_size"], mean_celsius=mean)
                            decoded = decode_indices(packed)
                            restored_indices = torch.tensor(decoded["indices"], dtype=torch.long, device=device).reshape(1, *decoded["shape"])
                            restored_mean = torch.tensor([decoded["mean_celsius"]], device=device)
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            t1 = time.perf_counter()
                            with amp(config, device):
                                reconstruction = model.decode_indices(restored_indices, restored_mean)
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            decode_seconds += time.perf_counter() - t1
                            if not torch.allclose(reconstruction.float(), predicted["encoded_prediction"][i:i+1].float(), atol=2e-3, rtol=2e-3):
                                raise ValueError("Index-only reconstruction differs; possible hidden continuous bypass")
                            (output / f"codec_{len(rates):02d}.bin").write_bytes(packed)
                            full = encode_indices(array, array.shape, config["model"]["codebook_size"], mean_celsius=mean,
                                                  reference_mask=valid[i], observed_mask=observed[i])
                            rates.append({"sample_id": sid, "without_support_masks": rate_audit(packed, 256 * 256),
                                          "with_support_masks": rate_audit(full, 256 * 256), "index_decode_checked": True,
                                          "with_support_storage": "serialized in memory; bytes measured, file not persisted"})
                    append_event(session_log, {"event": "progress", "stage": "evaluation", "session": session,
                        "elapsed_seconds": time.perf_counter() - started, "processed_rows": len(rows)})
                if seen != dataset.record_ids.tolist():
                    raise ValueError("Incomplete or reordered evaluation membership")
            finally:
                dataset.close()
        if len(rows) != COUNTS[split] * 2 or len({(r["sample_id"], r["view"]) for r in rows}) != len(rows):
            raise ValueError("Evaluation record/view count mismatch")
        boot = {"bootstrap_seed": config["evaluation"]["bootstrap_seed"],
                "bootstrap_replicates": config["evaluation"]["bootstrap_replicates"]}
        result = {"split": split, "condition": condition, "profile": profile,
            "row_count": len(rows), "unique_records": COUNTS[split], "views": 2,
            "raw": summarize_rows(rows, **boot), "projected": summarize_rows(projected, **boot),
            "observation_mean_fill_baseline": summarize_rows(baseline_rows, **boot),
            "physical_input_sha256": physical_digest.hexdigest(), "mean_sha256": mean_digest.hexdigest(),
            "encoded_input_sha256": encoded_digest.hexdigest(), "data_fingerprint": data_fingerprint,
            "checkpoint_sha256": weight_hash, "protocol_config_sha256": protocol_digest(config),
            "source_hashes": source_hashes(), "rate": model.rate(), "bitstream_samples": rates,
            "timing": {"full_cell_job_seconds": time.perf_counter() - started,
                       "synchronized_model_forward_seconds": model_seconds,
                       "forward_images_per_second": len(rows) / model_seconds,
                       "codec_sample_decode_seconds": decode_seconds,
                       "note": "Shared-device wall timings include contention; not dedicated GPU compute"},
            "memory": memory(device), "visual_review": "not_performed", "utc": utc()}
        _write_rows(output / "rows_raw.jsonl", rows)
        _write_rows(output / "rows_projected.jsonl", projected)
        _write_rows(output / "rows_baseline.jsonl", baseline_rows)
        if previews:
            np.savez_compressed(output / "gallery.npz", **{key: np.stack([p[key] for p in previews]) for key in previews[0]})
        if all_indices:
            np.savez_compressed(output / "indices.npz", indices=np.concatenate(all_indices),
                                sample_id=np.asarray([r["sample_id"] for r in rows]), view=np.asarray([r["view"] for r in rows]))
        files = ["rows_raw.jsonl", "rows_projected.jsonl", "rows_baseline.jsonl"] + (["gallery.npz"] if previews else [])
        files += [f"codec_{i:02d}.bin" for i in range(len(rates))]
        files += ["indices.npz"] if all_indices else []
        result["artifact_sha256"] = {file: file_digest(output / file) for file in files}
        result["timing"]["full_cell_job_seconds"] = time.perf_counter() - started
        append_event(session_log, {"event": "end", "stage": "evaluation", "session": session,
            "elapsed_seconds": time.perf_counter() - started, "memory": memory(device)})
        write_json(output / "metrics.json", result)
        return result
    except BaseException as exc:
        append_event(session_log, {"event": "interrupted", "stage": "evaluation", "session": session,
            "elapsed_seconds": time.perf_counter() - started, "error": repr(exc)})
        raise
