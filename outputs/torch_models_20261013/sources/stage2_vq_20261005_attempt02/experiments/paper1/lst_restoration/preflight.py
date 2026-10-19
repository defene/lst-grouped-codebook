"""Future CPU component checks; importing this module executes no checks.

This source was prepared without running numerical components. An explicit
``run_component_checks`` invocation in a later authorized session is required.
Passing these checks does not certify a full model's gradients, memory use,
throughput, data integrity or training correctness and never enables execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import time
from datetime import datetime, timezone
from typing import Any, Callable


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _raises(kind: type[Exception], call: Callable[[], Any]) -> None:
    try:
        call()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_component_checks(config: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Explicit CPU-only tests on tiny synthetic arrays; no dataset/model job.

    Autograd checks operate on representation-output tensors, not an encoder or
    decoder. A tiny FSQ projection is instantiated to exercise index decoding;
    it is not trained. ``execution.enabled`` is intentionally irrelevant here.
    Every failure is recorded; no unavailable check is labeled as passed.
    """
    from .config import validate, write_json
    from .runtime import source_hashes

    started = time.perf_counter()
    frozen = copy.deepcopy(config)
    protocol_config = copy.deepcopy(config)
    protocol_config.pop("execution", None)
    frozen_source_hashes = source_hashes()
    checks: list[dict[str, Any]] = []
    runtime_versions: dict[str, str] = {"python": platform.python_version()}

    def check(name: str, function: Callable[[], dict[str, Any] | None]) -> None:
        before = time.perf_counter()
        try:
            details = function() or {}
            checks.append({"id": name, "status": "passed", "seconds": time.perf_counter() - before, "details": details})
        except Exception as error:
            checks.append({"id": name, "status": "failed", "seconds": time.perf_counter() - before,
                           "error_type": type(error).__name__, "error": str(error)})

    def configuration():
        result = validate(config)
        _require(result["ok"], "; ".join(result["errors"]))
        return {"static_validation": result, "execution_flag_unchanged": config["execution"]["enabled"]}

    check("configuration", configuration)

    def numerical_dependencies():
        import numpy as np
        import torch
        runtime_versions.update(numpy=np.__version__, torch=torch.__version__)
        # No CUDA probe or context is needed to check CPU tensors.
        return {"device": "cpu", "cuda_context_requested": False}

    check("cpu_dependencies", numerical_dependencies)

    def quantizer_exact_forward():
        import torch
        from .quantizers import straight_through, VectorQuantizer, FiniteScalarQuantizer, BinaryQuantizer
        soft = torch.tensor([1e8, -1e8, .13], requires_grad=True)
        hard = torch.tensor([1e-4, -1e-4, .25], requires_grad=True)
        result = straight_through(soft, hard)
        _require(torch.equal(result, hard), "STE must preserve exact hard values")
        result.sum().backward()
        _require(torch.equal(soft.grad, torch.ones_like(soft)), "STE identity gradient changed")
        _require(hard.grad is None, "Reconstruction STE must not update codebook")
        details = {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            quantizers = {"vq": VectorQuantizer(4, 16), "fsq": FiniteScalarQuantizer(4, [4, 4]),
                "lfq": BinaryQuantizer(4, 4, False, .25, .01, .1),
                "bsq": BinaryQuantizer(4, 4, True, .25, .01, .1)}
            for kind, quantizer in quantizers.items():
                latent = torch.randn(2, 4, 3, 3, requires_grad=True)
                output = quantizer(latent)
                replay = quantizer.decode_indices(output["indices"])
                _require(torch.equal(output["quantized"], replay), kind + " exact index replay failed")
                (output["quantized"].square().mean() + output["loss"]).backward()
                _require(bool(torch.isfinite(latent.grad).all()) and bool((latent.grad != 0).any()),
                         kind + " must retain nonzero finite encoder gradients")
                if kind == "vq":
                    gradient = quantizer.embedding.weight.grad
                    _require(bool(torch.isfinite(gradient).all()) and bool((gradient != 0).any()),
                             "VQ native codebook loss must retain embedding gradients")
                details[kind] = {"exact_index_replay": True, "finite_nonzero_gradients": True}
        return details

    check("quantizer_exact_forward_and_ste_gradients", quantizer_exact_forward)

    def representations_roundtrip():
        import torch
        from .representations import Representation
        scale = float(config["representation"]["A_celsius"])
        values = torch.linspace(-scale, scale, 15, dtype=torch.float32).reshape(1, 1, 3, 5)
        result = {}
        for kind in ("R", "G", "C", "C_affine"):
            rep = Representation({"kind": kind, "A_celsius": scale})
            encoded = rep.encode(values)
            decoded = rep.decode(encoded)
            _require(encoded.shape[1] == rep.channels, "Channel contract differs")
            torch.testing.assert_close(decoded, values, rtol=2e-6, atol=scale * 2e-6)
            zero = rep.encode(torch.zeros(1, 1, 1, 1))
            torch.testing.assert_close(zero.flatten(), torch.tensor(rep.zero_encoded), rtol=0, atol=1e-7)
            outside = torch.tensor([[[[-1.1 * scale, 1.1 * scale]]]])
            if kind == "C":
                _raises(ValueError, lambda: rep.encode(outside))
            else:
                torch.testing.assert_close(rep.decode(rep.encode(outside)), outside, rtol=2e-6, atol=2e-5)
            torch.testing.assert_close(rep.project_field(outside), outside.clamp(-scale, scale))
            result[kind] = {"max_roundtrip_abs_C": float((decoded - values).abs().max()), "channels": rep.channels}
        c = Representation({"kind": "C", "A_celsius": scale})
        ambiguous = torch.zeros(1, 3, 1, 1)
        first = c.decode(ambiguous)
        torch.testing.assert_close(first, torch.full_like(first, -6.0 * scale / 7.0), rtol=1e-6, atol=2e-5)
        _require(torch.equal(first, c.decode(ambiguous)), "C inverse tie handling is not deterministic")
        return result

    check("representation_roundtrip_boundaries_and_ties", representations_roundtrip)

    def nonlinear_representations():
        import torch
        from .representations import Representation
        result = {}
        scale = float(config["representation"]["A_celsius"])
        for kind in ("G_power", "G_asinh", "C_power", "C_asinh"):
            rep = Representation({**config["representation"], "kind": kind})
            field = torch.linspace(-scale, scale, 1001).reshape(1, 1, 1, -1)
            error = (rep.decode(rep.encode(field)) - field).abs()
            _require(float(error.max()) <= .05, "Nonlinear FP32 inverse exceeds 0.05 C technical tolerance")
            x = torch.tensor([-.05, -.01, 0., .01, .05]).reshape(1, 1, 1, -1).mul(scale).requires_grad_()
            encoded = rep.encode(x)
            measured = sum(torch.autograd.grad(encoded[:, j].sum(), x, retain_graph=True)[0].square()
                           for j in range(rep.channels))
            torch.testing.assert_close(measured, rep.local_jacobian_squared(x), rtol=2e-5, atol=1e-7)
            grad = torch.autograd.grad(rep.decode(encoded).sum(), x)[0]
            torch.testing.assert_close(grad, torch.ones_like(grad), rtol=5e-4, atol=5e-4)
            _raises(ValueError, lambda: rep.encode(torch.full((1, 1, 1, 1), 1.1 * scale)))
            result[kind] = {"max_roundtrip_abs_C": float(error.max()), "local_jacobian_checked": True}
        return result

    check("nonlinear_colour_roundtrip_and_local_jacobian", nonlinear_representations)

    def matched_gray_colour_warps():
        import torch
        from .representations import Representation
        field = torch.linspace(-120., 120., 1001).reshape(1, 1, 1, -1)
        for suffix in ("power", "asinh"):
            gray = Representation({**config["representation"], "kind": "G_"+suffix})
            colour = Representation({**config["representation"], "kind": "C_"+suffix})
            torch.testing.assert_close(gray._warp(field), colour._warp(field), rtol=0, atol=0)
            position = torch.linspace(-1., 1., 1001).reshape(1, 1, 1, -1)
            torch.testing.assert_close(gray._unwarp(position), colour._unwarp(position), rtol=0, atol=0)
            torch.testing.assert_close(gray.encode(torch.zeros_like(field)), torch.zeros(1,3,1,1001))
        return {"matched_scalar_warp_and_inverse": True, "gray_zero_initialization": True}

    check("matched_nonlinear_gray_colour_scalar_maps", matched_gray_colour_warps)

    def physical_only_saturation_gradients():
        import torch
        from .representations import Representation
        kind = config["representation"]["kind"]
        if (config["train"].get("micro_batch_protocol") != "core-micro16-effective64-20260930"
                or config["loss"]["representation"] != 0
                or kind not in ("C", "C_power", "C_asinh", "G_power")):
            return {"applicable": False, "scope": "new pure-physical bounded-inverse protocols"}
        rep = Representation(config["representation"])
        legacy = Representation({**config["representation"], "inverse_gradient_policy": "native"})
        failures = []
        for values in ((3.,-3.,-3.), (-3.,3.,3.), (3.,3.,3.), (-3.,-3.,-3.)):
            output = torch.tensor(values).reshape(1,3,1,1).requires_grad_()
            field = rep.decode(output)
            _require(torch.equal(field, legacy.decode(output)), "Gradient repair changed hard forward predictions")
            objective = field.square().mean()
            gradient = torch.autograd.grad(objective, output)[0]
            if not torch.isfinite(gradient).all() or (objective > 0 and not (gradient != 0).any()):
                failures.append({"encoded": values, "prediction_C": float(field.detach()),
                                 "physical_mse": float(objective.detach()),
                                 "gradient_l1": float(gradient.abs().sum())})
        _require(not failures, f"Pure physical inverse has saturated zero/invalid gradients: {failures}. "
                 "Revise and freeze a common inverse policy; do not restore an undeclared auxiliary loss.")
        return {"applicable": True, "tested_saturation_points": 4, "all_gradients_usable": True}

    check("physical_only_saturation_gradients", physical_only_saturation_gradients)

    def local_metric_equivalence():
        import torch
        from .representations import Representation
        scale = float(config["representation"]["A_celsius"])
        a = torch.tensor([[[[-0.04 * scale, 0.0, 0.02 * scale]]]])
        b = torch.tensor([[[[-0.02 * scale, 0.03 * scale, 0.06 * scale]]]])
        physical_squared = (a - b).square()
        errors = {}
        for kind in ("R", "G", "C", "C_affine"):
            rep = Representation({"kind": kind, "A_celsius": scale})
            equivalent = (rep.encode(a) - rep.encode(b)).square().sum(1, keepdim=True) * scale**2 / rep.jacobian_squared
            torch.testing.assert_close(equivalent, physical_squared, rtol=1e-5, atol=1e-4)
            errors[kind] = float((equivalent - physical_squared).abs().max())
        c = Representation({"kind": "C", "A_celsius": scale})
        affine = Representation({"kind": "C_affine", "A_celsius": scale})
        torch.testing.assert_close(c.encode(a), affine.encode(a), rtol=1e-6, atol=1e-6)
        return {"max_physical_squared_difference": errors,
                "scope": "central one-segment interval only; no global RGB/temperature MSE equivalence"}

    check("local_jacobian_and_affine_equivalence", local_metric_equivalence)

    def colour_wrong_branch_gradients():
        import torch
        from .objective import loss
        from .representations import Representation
        cfg = copy.deepcopy(config)
        cfg["representation"]["kind"] = "C"
        cfg["loss"].update(pixel=1.0, gradient=0.1, representation=0.01, bottleneck=0.0, kl=0.0, region="all_valid")
        rep = Representation(cfg["representation"])
        valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)
        field = torch.zeros(1, 1, 1, 1)
        batch = {"temperature": field, "target_field": field, "center_celsius": torch.zeros(1),
                 "reference_valid": valid, "observed_valid": valid}
        # This point projects onto a wrong saturated endpoint. A raw physical
        # inverse can have zero gradient here; representation supervision must
        # keep the combined objective connected and finite.
        encoded = torch.tensor([[[[3.0]], [[-3.0]], [[-3.0]]]], requires_grad=True)
        output = {"encoded_prediction": encoded}
        physical_only = rep.decode(encoded).square().mean()
        physical_gradient = torch.autograd.grad(physical_only, encoded, retain_graph=True)[0]
        total, terms = loss(output, batch, rep, cfg)
        gradient = torch.autograd.grad(total, encoded)[0]
        _require(bool(torch.isfinite(gradient).all()), "Combined wrong-branch gradient is not finite")
        _require(float(gradient.abs().sum()) > 0, "Representation auxiliary term cannot escape this saturated endpoint")
        return {"physical_inverse_gradient_l1": float(physical_gradient.abs().sum()),
                "combined_gradient_l1": float(gradient.abs().sum()), "loss_terms": terms,
                "scope": "output-tensor autograd only, not full-network trainability"}

    check("colour_wrong_branch_auxiliary_gradient", colour_wrong_branch_gradients)

    def corruption_contracts():
        import numpy as np
        from .config import PROFILE_DEFAULTS
        from .corruptions import corrupt
        base = (20.0 + (np.arange(256 * 256, dtype=np.float32).reshape(1, 256, 256) % 31) / 10.0)
        valid = np.ones_like(base, dtype=bool)
        valid[:, :5, :7] = False
        base[~valid] = np.nan
        original, original_valid = base.copy(), valid.copy()
        findings = {}
        for kind, settings in PROFILE_DEFAULTS.items():
            p = dict(settings, seed=20260925)
            a = corrupt(base, valid, p, sample_id="synthetic:preflight", split="val", epoch=0, view=0)
            b = corrupt(base, valid, p, sample_id="synthetic:preflight", split="val", epoch=91, view=0)
            for name in ("observation", "observed_valid", "missing_mask", "affected_mask", "patch_missing"):
                _require(np.array_equal(a[name], b[name]), f"{kind}/{name} changed with evaluation epoch")
            _require(a["seed"] == b["seed"], "Evaluation seed uses epoch")
            train_a = corrupt(base, valid, p, sample_id="synthetic:preflight", split="train", epoch=0, view=0)
            train_b = corrupt(base, valid, p, sample_id="synthetic:preflight", split="train", epoch=1, view=0)
            _require(train_a["seed"] != train_b["seed"], "Training seed does not change by epoch")
            _require(a["observation"].dtype == np.float32, "Observation dtype is not float32")
            _require(np.all(a["observed_valid"] <= valid), "Observed mask exceeds reference support")
            _require(np.array_equal(a["missing_mask"], valid & ~a["observed_valid"]), "Missing mask semantics differ")
            _require(np.all(a["affected_mask"] <= valid), "Affected support includes invalid reference pixels")
            _require(np.isfinite(a["observation"]).all(), "Invalid observations were not finite-filled")
            _require(np.all(a["observation"][~a["observed_valid"]] == 0), "Unobserved temperatures not zero-filled")
            if kind in ("clean", "sensor_gaussian", "sensor_stripe", "cloud_bias"):
                _require(np.array_equal(a["observed_valid"], valid), f"Blind profile {kind} exposes/removes error support")
            if kind in ("pixel10", "cloud_missing", "joint"):
                fraction = .1 if kind == "pixel10" else p.get("coverage", .1)
                _require(int(a["missing_mask"].sum()) == round(float(fraction) * int(valid.sum())), "Exact valid-pixel coverage differs")
            if kind == "patch16":
                _require(a["patch_missing"].shape == (16, 16) and int(a["patch_missing"].sum()) == 26, "Patch benchmark is not 26/256")
                expanded = np.repeat(np.repeat(a["patch_missing"], 16, 0), 16, 1)[None]
                _require(np.array_equal(a["missing_mask"], expanded & valid), "Patch mask does not expand to missing pixels")
            else:
                _require(not a["patch_missing"].any(), "Pixel/region corruption falsely describes MAE token removal")
            if kind == "clean":
                _require(np.array_equal(a["observation"], np.where(valid, base, 0)), "Clean profile changed valid data")
            _require(np.array_equal(base, original, equal_nan=True) and np.array_equal(valid, original_valid), "Corruption mutated its clean inputs")
            findings[kind] = {"missing": int(a["missing_mask"].sum()), "affected": int(a["affected_mask"].sum()), "seed": a["seed"]}
        return findings

    check("deterministic_corruption_masks_and_no_mutation", corruption_contracts)

    def fsq_index_bijection():
        import torch
        from .quantizers import FiniteScalarQuantizer
        schedules = [[8, 8, 16], [8, 5, 5, 5]]
        if config["model"].get("kind") == "fsq" and config["model"]["levels"] not in schedules:
            schedules.append(list(config["model"]["levels"]))
        findings = {}
        # Restore the caller's RNG after small projection initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260925)
            for levels in schedules:
                count, dims = math.prod(levels), len(levels)
                _require(count <= 65536, "Component enumeration deliberately limited to 65536 states")
                quantizer = FiniteScalarQuantizer(dims, levels).cpu()
                with torch.no_grad():
                    quantizer.project_out.weight.zero_()
                    quantizer.project_out.bias.zero_()
                    for channel in range(dims):
                        quantizer.project_out.weight[channel, channel, 0, 0] = 1.0
                    indices = torch.arange(count, dtype=torch.long).reshape(1, 1, count)
                    decoded = quantizer.decode_indices(indices)
                    distinct = torch.unique(decoded.permute(0, 2, 3, 1).reshape(count, dims), dim=0).shape[0]
                    _require(distinct == count, "FSQ scalar decode aliases multiple joint indices")
                    halves = (quantizer.levels // 2)[None, :, None, None].float()
                    digits = (decoded * halves + halves).round().long()
                    recovered = (digits * quantizer.bases[None, :, None, None]).sum(1)
                    _require(torch.equal(indices, recovered), "Mixed-radix recomposition differs")
                    _raises(ValueError, lambda: quantizer.decode_indices(torch.tensor([[[-1]]])))
                    _raises(ValueError, lambda: quantizer.decode_indices(torch.tensor([[[count]]])))
                findings[str(levels)] = {"states": count, "unique_native_codes": distinct,
                                        "nominal_bits": (count - 1).bit_length()}
        return findings

    check("fsq_all_indices_bijection", fsq_index_bijection)

    def bitstream_roundtrip():
        from .bitstream import decode_indices, encode_indices, rate_audit
        findings = {}
        for capacity in (1000, 1024):
            indices = [0, capacity - 1, 17]
            reference = [[True, True, False], [True, True, True]]
            observed = [[True, False, False], [True, False, True]]
            mean = 23.123456789
            payload = encode_indices(indices, [1, 3], capacity, mean, reference, observed)
            restored = decode_indices(payload)
            _require(restored["indices"] == indices and restored["shape"] == [1, 3], "Packed tokens differ")
            _require(restored["mean_celsius"] == struct.unpack("<f", struct.pack("<f", mean))[0], "Mean side channel not float32 exact")
            _require(restored["reference_mask"]["values"] == [x for row in reference for x in row], "Reference mask differs")
            _require(restored["observed_mask"]["values"] == [x for row in observed for x in row], "Observed mask differs")
            audit = rate_audit(payload, pixels=6)
            _require(audit["nominal_token_bits"] == 30 and audit["packed_index_bytes"] == 4, "10-bit non-byte-aligned accounting differs")
            _require(audit["index_padding_bits"] == 2 and audit["mean_bytes"] == 4, "Padding / mean accounting differs")
            _require(audit["actual_bytes"] == len(payload), "Actual size is not serialized byte count")
            changed = bytearray(payload)
            changed[-1] ^= 1
            _raises(ValueError, lambda: decode_indices(changed))
            _raises(ValueError, lambda: decode_indices(payload[:-1]))
            findings[str(capacity)] = audit
        no_side = decode_indices(encode_indices([0, 1], [2], 1024))
        _require(no_side["mean_celsius"] is None and no_side["reference_mask"] is None and no_side["observed_mask"] is None, "Optional side channel silently added")
        _raises(ValueError, lambda: encode_indices([1024], [1], 1024))
        _raises(ValueError, lambda: encode_indices([0], [1], 1024, reference_mask=[[False]], observed_mask=[[True]]))
        return findings

    check("fixed_width_stream_roundtrip_and_cost", bitstream_roundtrip)

    def exact_budget_and_resume():
        from .config import COUNTS
        from .data import EpochSampler, update_batches
        records = COUNTS["train"]
        epochs = config["train"]["epochs"]
        effective = config["train"]["effective_batch"]
        micro = config["train"]["micro_batch"]
        sizes = [min(micro, records - start) for start in range(0, records, micro)]
        update_sizes = [sum(group) for group in update_batches(iter(sizes), effective // micro)]
        _require(sum(update_sizes) == records, "Tail records were dropped or repeated")
        _require(len(update_sizes) * epochs == config["train"]["updates"], "Exact epoch update count differs")
        _require(sum(update_sizes) * epochs == config["train"]["record_presentations"], "Record budget differs")
        _require(all(size == effective for size in update_sizes[:-1]), "Unexpected non-tail effective batch")
        _require(update_sizes[-1] == (records % effective or effective), "Tail is padded or incorrectly counted")
        # Small synthetic IDs test the real sampler without opening HLP data.
        length, epoch, seed = 137, 2, 17
        whole = list(EpochSampler(length, seed, epoch))
        consumed = 64
        suffix = list(EpochSampler(length, seed, epoch, consumed=consumed))
        _require(whole[consumed:] == suffix, "Resume changed the sample order")
        _require(len(set(index for _, index in whole)) == length, "An epoch duplicates sample IDs")
        _require({index for _, index in whole} == set(range(length)), "An epoch omits sample IDs")
        _require(all(e == epoch for e, _ in whole), "Sampler lost epoch for per-record noise")
        _require(list(EpochSampler(length, seed, epoch, consumed=length)) == [], "Completed epoch is not empty")
        return {"records_per_epoch": records, "updates_per_epoch": len(update_sizes),
                "tail_records": update_sizes[-1], "total_updates": len(update_sizes) * epochs,
                "record_presentations": records * epochs, "resume_comparison_ids": length}

    check("exact_tail_budget_and_sampler_resume", exact_budget_and_resume)

    check("caller_config_not_mutated", lambda: (_require(config == frozen, "Preflight mutated the caller configuration") or {}))
    check("source_code_not_changed", lambda: (_require(source_hashes() == frozen_source_hashes, "Source code changed during component checks") or {}))
    passed = all(item["status"] == "passed" for item in checks)
    report = {
        "status": "passed" if passed else "failed",
        "gates": {"components": passed},
        "scope": "cpu_component_checks_only",
        "authorizes_training": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config_sha256": _digest(frozen),
        "protocol_config_sha256": _digest(protocol_config),
        "source_hashes": frozen_source_hashes,
        "runtime_versions": runtime_versions,
        "device": "cpu",
        "checks": checks,
        "pending_checks": [
            "Frozen HLP v2 member/normalization/source audit using actual data.",
            "Selected full model local-weight provenance, construction and measured parameter count.",
            "Selected full model physical/representation gradients and end-to-end discrete-only decode equivalence.",
            "Selected full model one-update equivalence across microbatches and checkpoint/resume state replay.",
            "Measured GPU memory, throughput and resource-budget enforcement under the declared hardware policy.",
            "Full evaluation row counts, identities, input digests, reference-valid masks and source-weight digests.",
        ],
    }
    write_json(output_path, report)
    return report
