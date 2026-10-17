"""Explicit future data and CPU optimizer/resume audits; nothing runs on import.

The data gate examines every indexed membership, but reads only declared sample
positions from image storage. The budget gate performs a small synthetic CPU
optimizer exercise through the production ``train_update`` helper. It is not a
formal scientific experiment, a full-model audit, or execution authorization.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _data_contract(config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from .config import COUNTS, file_digest, profile
    from .data import LSTDataset

    datasets = {}
    opened = []
    try:
        for split in ("train", "val", "test"):
            datasets[split] = LSTDataset(config, split, evaluation_profile=profile("clean"))
            opened.append(datasets[split])
        expected_tiles = {"train": 2796, "val": 147, "test": 151}
        id_sets, tile_sets, summaries, tile_maps = {}, {}, {}, {}
        for split, dataset in datasets.items():
            source = dataset.source
            database = (source.directory / "index.sqlite").resolve()
            if database not in tile_maps:
                # This connection cannot modify the frozen metadata/index.
                connection = sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
                try:
                    pairs = connection.execute("SELECT id,tile FROM records").fetchall()
                    _require(len({int(rid) for rid, _ in pairs}) == len(pairs), "Duplicate IDs in source records table")
                    tile_maps[database] = {int(rid): str(tile) for rid, tile in pairs}
                finally:
                    connection.close()
            tile_map = tile_maps[database]
            ids = [int(value) for value in dataset.record_ids]
            _require(len(ids) == COUNTS[split] and len(set(ids)) == len(ids), f"{split}: incomplete/duplicate indexed records")
            _require(all(rid in tile_map for rid in ids), f"{split}: indexed record absent from source table")
            tiles = {tile_map[rid] for rid in ids}
            _require(len(tiles) == expected_tiles[split], f"{split}: unexpected tile count")
            _require(source.stats["fit_split"] == "train" and source.stats["samples"] == COUNTS["train"], "Normalization is not HLP v2 train-only")
            id_sets[split], tile_sets[split] = set(ids), tiles
            summaries[split] = {"records": len(ids), "tiles": len(tiles), "fingerprint": dataset.fingerprint,
                                "sampled_positions": [], "sampled_temperature_checks": []}
            # Three fixed positions per split are inspected, not the full arrays.
            positions = sorted({0, len(dataset) // 2, len(dataset) - 1})
            for index in positions:
                raw = source[index]
                sample = dataset[index]
                valid = np.asarray(raw["reference_mask"], dtype=bool)
                # Current AGENTS contract: undo stored normalization FIRST,
                # then Celsius x100 -> Celsius. No Landsat DN re-conversion.
                celsius = np.asarray(source.denormalize(raw["image"]) / 100.0, dtype=np.float32)
                expected = np.where(valid, celsius, np.float32(0)).astype(np.float32)
                _require(np.array_equal(sample["temperature"], expected), "Reader Celsius conversion differs from denormalize(image)/100")
                _require(np.array_equal(sample["reference_valid"], valid), "Reference validity changed")
                _require(np.array_equal(sample["observation"], expected), "Clean observation differs from reference Celsius")
                _require(np.array_equal(sample["observed_valid"], valid), "Clean observed-valid differs from reference-valid")
                _require(not sample["missing_mask"].any() and not sample["affected_mask"].any(), "Clean sample declares corruption")
                _require(int(sample["record_id"]) == ids[index] and sample["tile_id"] == tile_map[ids[index]], "Sample/index metadata differs")
                expected_mean = np.float32(expected[valid].mean(dtype=np.float64))
                _require(sample["mean_celsius"] == expected_mean, "Shared float32 mean is not computed from valid Celsius")
                center = expected_mean if config["task"]["field"] == "residual" else np.float32(config["task"]["absolute_center_celsius"])
                expected_field = np.where(valid, expected - center, np.float32(0)).astype(np.float32)
                _require(np.array_equal(sample["input_field"], expected_field), "Clean residual/absolute input field differs")
                _require(np.array_equal(sample["target_field"], expected_field), "Clean target field differs")
                _require(np.isfinite(expected[valid]).all(), "Non-finite reference-valid sampled temperatures")
                summaries[split]["sampled_positions"].append(index)
                summaries[split]["sampled_temperature_checks"].append({
                    "record_id": ids[index], "valid_pixels": int(valid.sum()),
                    "minimum_celsius": float(expected[valid].min()), "maximum_celsius": float(expected[valid].max()),
                    "mean_celsius_float32": float(expected_mean),
                })

        overlaps = {}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            record_overlap = len(id_sets[left] & id_sets[right])
            tile_overlap = len(tile_sets[left] & tile_sets[right])
            _require(record_overlap == 0 and tile_overlap == 0, f"{left}/{right} split membership overlaps")
            overlaps[f"{left}:{right}"] = {"record_ids": record_overlap, "tile_ids": tile_overlap}

        evidence_path = Path(config["task"]["scale_evidence"])
        if not evidence_path.is_absolute():
            evidence_path = Path(config["data"]["root"]) / evidence_path
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        _require(evidence["fit_split"] == "train" and evidence["samples"] == COUNTS["train"], "Scale fit has wrong split/count")
        _require(evidence.get("per_image_scale") is False, "Scale evidence permits per-image scale")
        _require(evidence.get("ordered_member_match") is True, "Scale evidence lacks ordered train-member confirmation")
        for key in ("residual_std_celsius",):
            _require(math.isclose(float(config["task"][key]), float(evidence[key]), rel_tol=1e-12, abs_tol=1e-12), f"Frozen {key} differs from train evidence")
        _require(math.isclose(float(config["task"]["mean_scale_celsius"]), float(evidence["global_std_celsius"]), rel_tol=1e-12, abs_tol=1e-12), "Mean normalization scale differs from train evidence")
        source_fingerprint = datasets["train"].source.fingerprint
        compared_keys = ("dataset", "modality", "split", "protocol", "sample_index", "normalization", "source_index", "record_index")
        for key in compared_keys:
            _require(evidence["data_fingerprint"][key] == source_fingerprint[key], f"Scale source fingerprint differs: {key}")
        _require(evidence["normalization"] == datasets["train"].source.stats, "Scale evidence carries different normalization statistics")
        moments_path = Path(evidence["source_moments"])
        _require(moments_path.is_file(), "Scale source moments unavailable for fingerprint verification")
        _require(file_digest(moments_path) == evidence["source_moments_sha256"], "Scale source moments hash changed")

        # Verify the visible-mean branch on an explicitly blind corruption.
        # Only the scalar differs between these variants; observation generation
        # must remain independent of the mean/source representation choice.
        mean_checks = {}
        for split in ("train", "val", "test"):
            variants = {}
            for mean_source in ("oracle", "visible"):
                c = copy.deepcopy(config)
                c["task"]["mean_source"] = mean_source
                instance = LSTDataset(c, split, evaluation_profile=profile("sensor_gaussian"))
                opened.append(instance)
                variants[mean_source] = instance[0]
            oracle, visible = variants["oracle"], variants["visible"]
            _require(np.array_equal(oracle["observation"], visible["observation"]), "Mean source changes shared noisy observation")
            expected_visible = np.float32(visible["observation"][visible["observed_valid"]].mean(dtype=np.float64))
            expected_oracle = np.float32(oracle["temperature"][oracle["reference_valid"]].mean(dtype=np.float64))
            _require(visible["mean_celsius"] == expected_visible and oracle["mean_celsius"] == expected_oracle, "Oracle/visible scalar source contract differs")
            mean_checks[split] = {"sample_position": 0, "oracle_mean": float(expected_oracle), "visible_noisy_mean": float(expected_visible)}
        return {
            "splits": summaries, "overlaps": overlaps, "mean_branch_checks": mean_checks,
            "scale_evidence": {"path": str(evidence_path), "sha256": file_digest(evidence_path),
                "source_moments_sha256": evidence["source_moments_sha256"], "compared_fingerprint_keys": list(compared_keys)},
            "coverage": "All indexed record/tile memberships; three clean image positions plus one noisy mean example per split.",
            "not_proven": ["Geographic footprint disjointness", "Full image-value range/integrity beyond frozen reader checks", "Real sensor/cloud calibration"],
            "unit_evidence": "Current project convention: source is Celsius x100; denormalize(image)/100; historical physical_temperature_unit_confirmed flag is preserved, not rewritten.",
        }
    finally:
        for dataset in opened:
            dataset.close()


def _budget_resume(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from .data import EpochSampler, update_batches
    from .representations import Representation
    from .runtime import atomic_checkpoint, load_checkpoint, protocol_digest, source_hashes, train_update

    cpu = torch.device("cpu")
    toy = copy.deepcopy(config)
    toy["representation"] = {"kind": "R", "A_celsius": 120.0}
    toy["task"].update(mean_source="oracle", field="residual")
    toy["loss"].update(pixel=1.0, gradient=0.1, representation=0.01, bottleneck=0.0, kl=0.0, region="all_valid")
    toy["train"].update(device="cpu", amp=False, epochs=2, effective_batch=64, micro_batch=4,
                        views=1, workers=0, updates=4, record_presentations=134)
    representation = Representation(toy["representation"])
    fingerprint = {"dataset": "synthetic_budget_audit", "records": 67, "epochs": 2, "shape": [1, 4, 4]}
    code_hashes = source_hashes()

    class ToyDataset(Dataset):
        def __len__(self):
            return 67

        def __getitem__(self, item):
            epoch, index = item
            valid = torch.ones(1, 4, 4, dtype=torch.bool)
            if index % 3 == 0:
                valid[:, 0, 0] = False
            pattern = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 8.0
            raw = pattern + 20.0 + index * 0.05
            mean = raw[valid].mean()
            target_field = torch.where(valid, raw - mean, torch.zeros_like(raw))
            observed = valid.clone()
            input_field = torch.where(observed, target_field + (index % 5 - 2) * .03, torch.zeros_like(raw))
            return {"sample_id": f"toy:{epoch}:{index}", "record_id": index, "profile": "clean",
                    "temperature": torch.where(valid, raw, torch.zeros_like(raw)),
                    "target_field": target_field, "input_field": input_field,
                    "reference_valid": valid, "observed_valid": observed,
                    "mean_celsius": mean, "center_celsius": mean,
                    "patch_missing": torch.zeros(1, 1, dtype=torch.bool)}

    class TinyRestorer(nn.Module):
        def __init__(self, stochastic):
            super().__init__()
            self.value = nn.Conv2d(1, 1, 1)
            self.stochastic = stochastic
            with torch.no_grad():
                self.value.weight.fill_(.8)
                self.value.bias.fill_(.005)

        def forward(self, encoded_input, reference_valid, observed_valid, mean_celsius, patch_missing=None):
            value = self.value(encoded_input)
            if self.stochastic:
                # A tiny CPU random term makes RNG restoration meaningful.
                value = value + (torch.rand_like(value) - .5) * .001
            return {"encoded_prediction": value, "bottleneck_loss": value.new_zeros(()), "kl": value.new_zeros(())}

    def optimizer_for(model):
        return torch.optim.AdamW(model.parameters(), lr=toy["train"]["lr"], betas=(.9, .95),
                                 weight_decay=toy["train"]["weight_decay"])

    def loader(epoch, consumed=0):
        return DataLoader(ToyDataset(), batch_size=4, sampler=EpochSampler(67, toy["train"]["sample_order_seed"], epoch, consumed),
                          drop_last=False, num_workers=0,
                          generator=torch.Generator().manual_seed(toy["seed"] + epoch))

    def equal_nested(left, right, path="state"):
        if isinstance(left, torch.Tensor):
            _require(torch.equal(left, right), f"Exact tensor replay failed: {path}")
        elif isinstance(left, dict):
            _require(left.keys() == right.keys(), f"State keys differ: {path}")
            for key in left:
                equal_nested(left[key], right[key], path + "/" + str(key))
        elif isinstance(left, (tuple, list)):
            _require(type(left) is type(right) and len(left) == len(right), f"State sequence differs: {path}")
            for index, (a, b) in enumerate(zip(left, right)):
                equal_nested(a, b, path + "/" + str(index))
        else:
            _require(left == right, f"Exact scalar replay failed: {path}")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260925)
        initial = TinyRestorer(stochastic=True)
        initial_state = copy.deepcopy(initial.state_dict())
        initial_rng = torch.get_rng_state().clone()

        def run(resume=None, stop_after=None):
            model = TinyRestorer(stochastic=True).to(cpu)
            optimizer = optimizer_for(model)
            if resume is None:
                model.load_state_dict(initial_state)
                torch.set_rng_state(initial_rng)
                epoch = consumed = step = records = 0
                traces, sequence = [], []
            else:
                model.load_state_dict(resume["model"])
                optimizer.load_state_dict(resume["optimizer"])
                torch.set_rng_state(resume["rng_cpu"])
                epoch, consumed, step, records = (resume[key] for key in ("epoch", "consumed", "step", "records"))
                traces, sequence = copy.deepcopy(resume["traces"]), list(resume["sequence"])
            model.train()
            while epoch < 2:
                batches = loader(epoch, consumed)
                for group in update_batches(iter(batches), 16):
                    lr = toy["train"]["lr"] * .5 * (1 + math.cos(math.pi * step / toy["train"]["updates"]))
                    for parameters in optimizer.param_groups:
                        parameters["lr"] = lr
                    for batch in group:
                        sequence.extend(batch["sample_id"])
                    actual, totals, counts = train_update(model, group, optimizer, representation, toy, cpu)
                    _require(counts == {"clean": actual}, "Production helper miscounts profile presentations")
                    consumed += actual
                    records += actual
                    step += 1
                    if consumed == 67:
                        epoch += 1
                        consumed = 0
                    traces.append({"step": step, "actual": actual, "lr": lr, "loss": totals})
                    if stop_after is not None and step == stop_after:
                        break
                if stop_after is not None and step == stop_after:
                    break
            return {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "rng_cpu": torch.get_rng_state().clone(), "rng_cuda": [],
                    "epoch": epoch, "consumed": consumed, "step": step, "records": records,
                    "traces": traces, "sequence": sequence, "protocol_config_sha256": protocol_digest(toy),
                    "source_hashes": code_hashes, "data_fingerprint": fingerprint}

        uninterrupted = run()
        interrupted = run(stop_after=1)
        _require((interrupted["epoch"], interrupted["consumed"], interrupted["records"]) == (0, 64, 64), "Checkpoint not after first committed full update")
        with tempfile.TemporaryDirectory(prefix="lst_cpu_budget_") as directory:
            checkpoint = Path(directory) / "last.pt"
            atomic_checkpoint(checkpoint, interrupted)
            reloaded = load_checkpoint(checkpoint, toy, expected_data=fingerprint)
            resumed = run(resume=reloaded)
        equal_nested(uninterrupted, resumed)
        _require(uninterrupted["records"] == 134 and uninterrupted["step"] == 4, "Toy budget not exactly 67*2 records / 4 updates")
        _require([event["actual"] for event in uninterrupted["traces"]] == [64, 3, 64, 3], "Tail is padded, lost or crosses epochs")
        _require(len(set(uninterrupted["sequence"])) == 134, "Toy epoch/sample IDs repeated or missing")

        # Test the actual accumulated update against one true 64-record batch
        # with no stochastic/entropy dependence on microbatch partitioning.
        group = next(update_batches(iter(loader(0)), 16))
        whole = {}
        for key in group[0]:
            values = [batch[key] for batch in group]
            whole[key] = torch.cat(values, 0) if isinstance(values[0], torch.Tensor) else sum(values, [])
        accumulated_model, full_model = TinyRestorer(False), TinyRestorer(False)
        accumulated_model.load_state_dict(initial_state)
        full_model.load_state_dict(initial_state)
        accumulated_optimizer, full_optimizer = optimizer_for(accumulated_model), optimizer_for(full_model)
        a_count, a_loss, _ = train_update(accumulated_model, group, accumulated_optimizer, representation, toy, cpu)
        b_count, b_loss, _ = train_update(full_model, [whole], full_optimizer, representation, toy, cpu)
        _require(a_count == b_count == 64, "Accumulation comparison did not see exactly 64 records")
        max_difference = 0.0
        for key, tensor in accumulated_model.state_dict().items():
            other = full_model.state_dict()[key]
            torch.testing.assert_close(tensor, other, rtol=2e-5, atol=2e-6)
            max_difference = max(max_difference, float((tensor - other).abs().max()))
        for key in a_loss:
            _require(math.isclose(a_loss[key], b_loss[key], rel_tol=2e-5, abs_tol=2e-6), f"Equal-record loss weighting differs for {key}")

    return {
        "device": "cpu", "uses_production_train_update": True,
        "optimizer": "AdamW betas=(0.9,0.95); production update-number cosine schedule",
        "toy_records_per_epoch": 67, "toy_epochs": 2, "toy_record_presentations": 134,
        "toy_updates": 4, "toy_update_sizes": [64, 3, 64, 3],
        "checkpoint_after_update": 1, "exact_resume_parameters_optimizer_rng_losses_and_order": True,
        "full_batch_vs_accumulation_max_parameter_difference": max_difference,
        "full_batch_vs_accumulation_tolerance": {"rtol": 2e-5, "atol": 2e-6},
        "formal_budget_derivation": {"train_records": 218975, "epochs": config["train"]["epochs"],
            "effective_batch": config["train"]["effective_batch"],
            "tail": 218975 % config["train"]["effective_batch"] or config["train"]["effective_batch"],
            "record_presentations": 218975 * config["train"]["epochs"],
            "updates": config["train"]["epochs"] * math.ceil(218975 / config["train"]["effective_batch"])},
        "scope": "Toy CPU update/cursor replay only; no full model, GPU dropout/AMP, entropy microbatch equivalence or formal training performed.",
    }


def run_data_budget_checks(config: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Run only after explicit authorization; save failures instead of hiding them."""
    from .config import write_json
    from .runtime import protocol_digest, source_hashes, utc
    before = copy.deepcopy(config)
    frozen_sources = source_hashes()
    started = time.perf_counter()
    checks, gates = [], {}
    for name, function in (("data_contract", _data_contract), ("budget_resume", _budget_resume)):
        start = time.perf_counter()
        try:
            details = function(config)
            gates[name] = True
            checks.append({"id": name, "status": "passed", "elapsed_seconds": time.perf_counter() - start, "details": details})
        except Exception as error:
            gates[name] = False
            checks.append({"id": name, "status": "failed", "elapsed_seconds": time.perf_counter() - start,
                           "error_type": type(error).__name__, "error": str(error)})
    unchanged = before == config
    if not unchanged:
        gates = {name: False for name in gates}
        checks.append({"id": "config_not_mutated", "status": "failed", "error": "Audit mutated caller configuration"})
    if source_hashes() != frozen_sources:
        gates = {name: False for name in gates}
        checks.append({"id": "source_code_not_changed", "status": "failed", "error": "Source code changed during audit"})
    report = {"status": "passed" if all(gates.values()) else "failed", "gates": gates,
              "created_utc": utc(), "elapsed_seconds": time.perf_counter() - started,
              "protocol_config_sha256": protocol_digest(before), "source_hashes": frozen_sources,
              "checks": checks, "caller_config_unchanged": unchanged, "authorizes_training": False,
              "scope": "Frozen membership + sampled reader/mean/scale contracts and tiny synthetic CPU optimizer resume; not a formal run."}
    write_json(output_path, report)
    return report
