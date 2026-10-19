"""Explicit execution guard, provenance, resumable five-epoch training.

Nothing runs on import. A generated plan cannot launch until its runtime
preflight evidence is present and execution has explicitly been enabled.
"""
from contextlib import nullcontext
from datetime import datetime, timezone
import copy
import json
import math
import os
from pathlib import Path
import time
import uuid

from .config import digest, file_digest, validate, write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def protocol_digest(config):
    value = copy.deepcopy(config)
    value.pop("execution", None)
    return digest(value)


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    files = list(Path(__file__).parent.glob("*.py"))
    files += [p for p in (Path(__file__).parent / "geo_sources").iterdir() if p.is_file()]
    files += list((root / "eo_data").glob("*.py"))
    files += list((root / "eo_denoise").glob("*.py"))
    return {str(p.relative_to(root)).replace("\\", "/"): file_digest(p) for p in sorted(set(files))}


def execution_guard(config):
    result = validate(config)
    if not result["ok"]:
        raise ValueError(result["errors"])
    if not config["execution"]["enabled"]:
        raise RuntimeError("Execution disabled: this is an unlaunched research plan")
    path = config["execution"]["preflight_report"]
    if not path:
        raise RuntimeError("Runtime preflight evidence is required before formal execution")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("components", "model_contract", "backward_finite", "parameter_rate_audit",
                "budget_resume", "resource_profile", "data_contract")
    if report.get("protocol_config_sha256") != protocol_digest(config):
        raise RuntimeError("Preflight does not bind this protocol")
    if report.get("source_hashes") != source_hashes():
        raise RuntimeError("Code changed after preflight; repeat it before freezing")
    if any(report.get("gates", {}).get(name) is not True for name in required):
        raise RuntimeError("Incomplete runtime preflight; static syntax checks are not execution approval")
    if config["model"]["kind"] in ("vq", "fsq", "lfq", "bsq"):
        path = config["execution"].get("alignment_report")
        if not path:
            raise RuntimeError("Core comparisons require measured cross-model parameter/rate alignment")
        alignment = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not alignment.get("passed") or alignment.get("source_hashes") != source_hashes() or
                protocol_digest(config) not in alignment.get("protocol_config_sha256", [])):
            raise RuntimeError("Missing or stale cross-model alignment evidence")
    return report


def append_event(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": utc(), **event}, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def check_stop(output):
    for path in (Path(output) / "STOP", Path(output).parent / "STOP"):
        if path.exists():
            raise InterruptedError(f"User STOP present: {path}: {path.read_text(encoding='utf-8', errors='replace')}")


def setup(config):
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(config["seed"])
    device = torch.device(config["train"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested, unavailable")
        torch.cuda.manual_seed_all(config["seed"])
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., config["train"]["memory_limit_gib"] * 2**30 / total), device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return device


def amp(config, device):
    import torch
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and config["train"]["amp"] else nullcontext()


def memory(device):
    import torch
    if device.type != "cuda":
        return {"device": str(device)}
    return {"device": str(device), "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unknown")),
            "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
            "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def move_batch(batch, device):
    import torch
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def atomic_checkpoint(path, state):
    import torch
    path = Path(path)
    temp = path.with_suffix(".partial")
    torch.save(state, temp)
    temp.replace(path)


def load_checkpoint(path, config, expected_data=None):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["protocol_config_sha256"] != protocol_digest(config) or state["source_hashes"] != source_hashes():
        raise ValueError("Resume requires the frozen configuration and exact implementation")
    if expected_data is not None and state["data_fingerprint"] != expected_data:
        raise ValueError("Data identity changed")
    return state


def train_update(model, group, optimizer, representation, config, device):
    """One optimizer boundary, reused by the future tail/resume integration audit."""
    import torch
    from .objective import forward_training, loss
    actual = sum(len(batch["sample_id"]) for batch in group)
    optimizer.zero_grad(set_to_none=True)
    totals, profile_counts = {}, {}
    for cpu_batch in group:
        for name in cpu_batch["profile"]:
            profile_counts[name] = profile_counts.get(name, 0) + 1
        batch = move_batch(cpu_batch, device)
        fraction = len(cpu_batch["sample_id"]) / actual
        with amp(config, device):
            predicted = forward_training(model, batch, representation)
            value, terms = loss(predicted, batch, representation, config)
        (value * fraction).backward()
        for key, number in terms.items():
            totals[key] = totals.get(key, 0.) + fraction * number
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    return actual, totals, profile_counts


def train(config, output):
    execution_guard(config)
    from eo_data.locking import build_lock
    output = Path(output)
    check_stop(output)
    with build_lock(output):
        try:
            return _train(config, output)
        except BaseException as exc:
            write_json(output / "status.json", {"status": "stopped" if isinstance(exc, InterruptedError) else "failed",
                                               "error": repr(exc), "utc": utc(), "pid": os.getpid()})
            raise


def _train(config, output):
    import torch
    from . import tracking
    tracking.phase("training")
    from torch.utils.data import DataLoader
    from .data import LSTDataset, EpochSampler, update_batches
    from .models import build_model
    from .representations import Representation
    device = setup(config)
    dataset = LSTDataset(config, "train")
    model = build_model(config).to(device)
    representation = Representation(config["representation"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"], betas=(.9, .95),
                                 weight_decay=config["train"]["weight_decay"])
    epoch, consumed, step, records = 0, 0, 0, 0
    frozen_sources = source_hashes()
    if (output / "last.pt").exists():
        state = load_checkpoint(output / "last.pt", config, dataset.fingerprint)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, consumed, step, records = (state[x] for x in ("epoch", "consumed", "step", "records"))
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        del state
    else:
        if any(output.glob("train*.jsonl")):
            raise RuntimeError("Log without checkpoint; retain evidence and use a fresh output")
        write_json(output / "config.json", config)
        write_json(output / "data_fingerprint.json", dataset.fingerprint)
        write_json(output / "source_hashes.json", frozen_sources)
        metadata = model.model_metadata() if hasattr(model, "model_metadata") else {}
        write_json(output / "model.json", {**metadata, "parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    n, effective = len(dataset), config["train"]["effective_batch"]
    if not 0 <= consumed < n or records != epoch * n + consumed or (consumed and consumed % effective):
        raise ValueError("Checkpoint not at a valid optimizer/epoch boundary")
    if step != epoch * math.ceil(n / effective) + consumed // effective:
        raise ValueError("Checkpoint update count inconsistent")
    if epoch == config["train"]["epochs"]:
        if records != config["train"]["record_presentations"] or step != config["train"]["updates"]:
            raise RuntimeError("Completed checkpoint has an inconsistent budget")
        write_json(output / "training_complete.json", {"epochs": epoch, "records": records, "updates": step,
            "checkpoint_sha256": file_digest(output / "last.pt"), "budget_passed": True,
            "completion_marker_recovered": True, "utc": utc()})
        write_json(output / "status.json", {"status": "training_complete_pending_evaluation", "utc": utc()})
        dataset.close()
        return output / "last.pt"

    def save():
        atomic_checkpoint(output / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "consumed": consumed, "step": step, "records": records,
            "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "protocol_config_sha256": protocol_digest(config), "source_hashes": frozen_sources,
            "data_fingerprint": dataset.fingerprint})

    save()
    session = uuid.uuid4().hex
    session_log = output / f"sessions_{session}.jsonl"
    training_log = output / f"train_{session}.jsonl"
    started = time.perf_counter()
    append_event(session_log, {"event": "start", "stage": "training", "session": session,
                 "step": step, "records": records, "pid": os.getpid(), "device": str(device),
                 "gpu_uuid": memory(device).get("gpu_uuid")})
    model.train()
    try:
        while epoch < config["train"]["epochs"]:
            check_stop(output)
            sampler = EpochSampler(n, config["train"]["sample_order_seed"], epoch, consumed)
            loader = DataLoader(dataset, batch_size=config["train"]["micro_batch"], sampler=sampler,
                                num_workers=config["train"]["workers"], drop_last=False,
                                pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(config["seed"] + epoch))
            for group in update_batches(iter(loader), effective // config["train"]["micro_batch"]):
                check_stop(output)
                # LR is a pure function of the committed update number.
                lr = config["train"]["lr"] * .5 * (1 + math.cos(math.pi * step / config["train"]["updates"]))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                actual, totals, profile_counts = train_update(model, group, optimizer, representation, config, device)
                consumed += actual
                records += actual
                step += 1
                if consumed == n:
                    epoch += 1
                    consumed = 0
                append_event(training_log, {"session": session, "step": step, "records": records,
                             "epoch_completed": epoch, "consumed": consumed, "actual_batch": actual, "lr": lr,
                             "loss": totals, "profile_counts": profile_counts,
                             "elapsed_session_seconds": time.perf_counter() - started})
                append_event(session_log, {"event": "progress", "session": session, "stage": "training",
                             "step": step, "elapsed_seconds": time.perf_counter() - started})
                if not consumed or step % config["train"]["checkpoint_every"] == 0:
                    save()
                write_json(output / "status.json", {"status": "training", "epoch_completed": epoch,
                    "consumed": consumed, "step": step, "records": records, "pid": os.getpid(), "utc": utc(), "memory": memory(device)})
                if step == 1 or not consumed or step % config.get("tracking", {}).get("every_updates", 10) == 0:
                    tracking.log({"train/step": step, "train/records": records,
                        "train/epoch": epoch + consumed / n, "train/actual_batch": actual,
                        "train/lr": lr, "train/budget_fraction": records / config["train"]["record_presentations"],
                        **{"train/loss_" + k: v for k, v in totals.items()},
                        **{"memory/" + k: v for k, v in memory(device).items() if isinstance(v, (int, float))}})
                if not consumed:
                    break
        if records != config["train"]["record_presentations"] or step != config["train"]["updates"]:
            raise RuntimeError("Final exact-epoch budget audit failed")
        save()
        append_event(session_log, {"event": "end", "stage": "training", "session": session,
            "elapsed_seconds": time.perf_counter() - started, "step": step, "memory": memory(device)})
        write_json(output / "training_complete.json", {"epochs": epoch, "records": records, "updates": step,
            "checkpoint_sha256": file_digest(output / "last.pt"), "budget_passed": True, "utc": utc()})
        write_json(output / "status.json", {"status": "training_complete_pending_evaluation", "utc": utc()})
        return output / "last.pt"
    except BaseException as exc:
        # Only earlier fully committed updates can be resumed. Logs are append-only;
        # work after last.pt remains visible as recomputation cost, not new budget.
        append_event(session_log, {"event": "interrupted", "stage": "training", "session": session,
            "elapsed_seconds": time.perf_counter() - started, "step": step, "error": repr(exc), "memory": memory(device)})
        raise
    finally:
        dataset.close()
