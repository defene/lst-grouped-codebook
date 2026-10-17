"""Future explicit synthetic model preflight, NEVER run on import.

Calling run_model_checks performs real model forward/backward and FOUR B=1
synthetic optimizer updates, plus one formal-microbatch update on CUDA. This is a technical experiment, not static
validation. It must only be called when runtime checks have been authorized.
It does not open HLP, use validation/test records, download weights, persist
trained weights, or enable formal experiment execution.

This source was authored without executing its runtime checks.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any


def run_model_checks(config: dict[str, Any], output_path: str | Path,
                     device: str = "cpu") -> dict[str, Any]:
    """Run explicitly requested B=1, 256x256 synthetic checks and write evidence.

    GPU resource_gate passes only after an additional update at the configured
    formal microbatch, actually profiled device, AMP mode and optimizer. A B=1
    result never certifies a larger batch; CPU results never satisfy that gate. The separate
    components, data_contract and budget_resume gates remain false.
    """
    # Imports stay inside the function: CLI static validation can import this
    # module without importing torch, opening weights or initializing devices.
    import torch
    from .config import validate, write_json, pretrained_identity
    from .runtime import protocol_digest, source_hashes, utc, setup, amp, memory
    from .models import build_model
    from .representations import Representation
    from .objective import model_inputs, forward_training, loss

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Preserve earlier preflight evidence; choose a new output path")
    static = validate(config)
    bound_sources = source_hashes()
    report: dict[str, Any] = {
        "kind": "explicit_synthetic_model_runtime_preflight", "started_utc": utc(),
        "protocol_config_sha256": protocol_digest(config), "source_hashes": bound_sources,
        "configuration_execution": copy.deepcopy(config.get("execution", {})),
        "requested_device": str(device), "synthetic_batch_shape": [1, config["model"]["channels"], 256, 256],
        "training_dataset_used": False, "validation_or_test_used": False,
        "weight_download_allowed": False, "new_weights_persisted": False,
        "parameter_updates_planned": 5 if str(device).startswith("cuda") else 4,
        "parameter_updates_completed": 0,
        "parameter_updates_are_synthetic_technical_training": True,
        "not_formal_budget_presentations": True, "static_configuration": static,
        "gates": {"components": False, "model_contract": False, "backward_finite": False,
                  "parameter_rate_audit": False, "budget_resume": False,
                  "resource_profile": False, "data_contract": False},
        "component_checks": {}, "conditions": [], "status": "running",
        "unperformed": ["full data membership/unit audit", "tail-batch and optimizer-resume audit",
                        "complete representation/corruption/quantizer component suite",
                        "formal multi-record reconstruction evaluation"],
    }
    begin = time.perf_counter()
    model = optimizer = None
    dev = None

    def finite_tensor(value):
        return bool(torch.isfinite(value.detach()).all())

    def synchronize():
        if dev is not None and dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def gradient_audit():
        # Diagnose groups instead of accepting a finite gradient on only the
        # reconstruction head. First-step zero encoder gradients are legal.
        groups = {name: {"parameters_with_grad": 0, "nonzero_grad_tensors": 0,
                         "finite": True, "squared_l2": 0.0}
                  for name in ("encoder", "decoder", "input_adapter", "decoder_mean", "quantizer")}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            group = "encoder" if name.startswith(("encoder.", "vae.encoder.", "quant_conv.", "tokenizer.encoder.", "tokenizer.quant_proj.")) else None
            if name.startswith(("decoder.", "vae.decoder.", "post_quant_conv.", "tokenizer.decoder.")):
                group = "decoder"
            if name.startswith(("backbone.patch_embed.", "backbone.blocks.", "backbone.norm.", "backbone.cls_token")):
                group = "encoder"
            if name.startswith(("backbone.decoder_", "backbone.mask_token", "backbone.proj_up_", "backbone.up_block.")):
                group = "decoder"
            for other in ("input_adapter", "decoder_mean", "quantizer"):
                if name.startswith(other + "."):
                    group = other
            if group is None:
                continue
            stats = groups[group]
            gradient = parameter.grad.detach().float()
            stats["parameters_with_grad"] += 1
            ok = bool(torch.isfinite(gradient).all())
            stats["finite"] = stats["finite"] and ok
            if ok:
                stats["nonzero_grad_tensors"] += int(bool((gradient != 0).any()))
                stats["squared_l2"] += float(gradient.double().square().sum())
        return groups

    def make_synthetic_batch(condition):
        # An analytic, structured temperature field: no real image/record is
        # read. It spans the central colour segment and portions of its tails.
        coordinates = torch.linspace(-1.0, 1.0, 256, device=dev)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        field = 12 * torch.sin(3.141592653589793 * xx) + 7 * torch.cos(2 * 3.141592653589793 * yy)
        field = field + 5 * torch.exp(-((xx - .35).square() + (yy + .2).square()) / .04)
        temperature = (25.0 + field)[None, None].float()
        reference = torch.ones_like(temperature, dtype=torch.bool)
        patch = torch.zeros((1, 256), dtype=torch.bool, device=dev)
        if condition == "patch16":
            # Fixed 26/256 whole patches, deterministic and shared by methods.
            patch[:, torch.arange(26, device=dev) * 7 % 256] = True
        missing = patch.reshape(1, 1, 16, 16).repeat_interleave(16, 2).repeat_interleave(16, 3)
        observed = reference & ~missing
        source = temperature[reference] if config["task"]["mean_source"] == "oracle" else temperature[observed]
        mean = source.double().mean().float().reshape(1)
        center = mean if config["task"]["field"] == "residual" else mean.new_tensor([config["task"]["absolute_center_celsius"]])
        target = temperature - center.reshape(1, 1, 1, 1)
        return {"temperature": temperature, "reference_valid": reference, "observed_valid": observed,
                "input_field": torch.where(observed, target, 0.0), "target_field": target,
                "mean_celsius": mean, "center_celsius": center, "patch_missing": patch}

    try:
        if not static["ok"]:
            raise ValueError(f"Invalid protocol config: {static['errors']}")
        runtime_config = copy.deepcopy(config)
        runtime_config["train"]["device"] = str(device)
        dev = setup(runtime_config)
        if dev.type == "cuda":
            # Normalize an unspecified CUDA index to the device actually used;
            # do not assume that the process's current CUDA device is zero.
            dev = torch.device("cuda", torch.cuda.current_device() if dev.index is None else dev.index)
        report["runtime_device"] = str(dev)
        report["torch_version"] = str(torch.__version__)
        report["cuda_version"] = torch.version.cuda
        if dev.type == "cuda":
            report["gpu_name"] = torch.cuda.get_device_name(dev)
        model = build_model(config).to(dev)
        representation = Representation(config["representation"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"], betas=(.9, .95),
                                      weight_decay=config["train"]["weight_decay"])
        report["model_metadata"] = model.model_metadata()
        report["pretrained_source"] = {
            "path": config["model"].get("official_weights") or config["model"].get("pretrained_path"),
            "file_sha256_checked_by_builder": dict(pretrained_identity(config)),
            "official_identity_authenticated_by_this_audit": False,
            "note": "A file hash binds bytes, not official identity; retain repository/revision/acquisition provenance.",
        }
        contract_ok = backward_ok = rate_ok = True
        if hasattr(model, "vae") and hasattr(model, "strict_loading_info"):
            # Compare the fresh adapter to the official deterministic VAE path,
            # before any synthetic optimizer update can alter official weights.
            model.eval()
            probe = make_synthetic_batch("clean")
            with torch.no_grad():
                inputs = model_inputs(probe, representation)
                adapter_input = model.input_adapter(inputs["encoded_input"], inputs["reference_valid"],
                                                    inputs["observed_valid"], inputs["mean_celsius"])
                input_exact = torch.equal(inputs["encoded_input"], adapter_input)
                layout_matched = inputs["encoded_input"].is_contiguous() and adapter_input.is_contiguous()
                reference_z = model.vae.encode(inputs["encoded_input"]).latent_dist.mode()
                official = model.vae.decode(reference_z).sample.float()
                adapted = model(**inputs)["encoded_prediction"]
                reference_ok = input_exact and layout_matched and bool(torch.allclose(official, adapted, atol=1e-5, rtol=1e-5))
                report["pretrained_initialization_contract"] = {
                    "passed": reference_ok, "strict_loading_info": model.strict_loading_info,
                    "adapter_input_exact": input_exact, "canonical_nchw_layout_matched": layout_matched,
                    "official_vs_adapter_max_abs_difference": float((official-adapted).abs().max()),
                    "posterior": "mode", "before_any_optimizer_updates": True,
                    "input_adapter_identity_and_mean_affine_identity": reference_ok}
                contract_ok = contract_ok and reference_ok
            del probe, inputs, adapter_input, reference_z, official, adapted
        all_finite_gradients = True
        latent_evidence = []
        is_mae = config["model"]["kind"] in ("prithvi_mae", "satmae_pp")
        for condition in ("clean", "patch16"):
            batch = make_synthetic_batch(condition)
            condition_config = copy.deepcopy(config)
            # MAE clean is explicitly the same-architecture AE control; a
            # missing-only loss on a clean input must otherwise raise.
            if condition == "clean":
                condition_config["loss"]["region"] = "all_valid"
            elif is_mae:
                condition_config["loss"]["region"] = "missing_only"
            condition_report = {"condition": condition, "loss_region": condition_config["loss"]["region"],
                                "loss_region_override_for_technical_control": condition_config["loss"]["region"] != config["loss"]["region"],
                                "steps": []}
            model.train()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                synchronize()
                started = time.perf_counter()
                with amp(runtime_config, dev):
                    output = forward_training(model, batch, representation)
                    objective, terms = loss(output, batch, representation, condition_config)
                required = {"encoded_prediction", "indices", "bottleneck_loss", "kl", "latent"}
                shapes_ok = required.issubset(output) and tuple(output["encoded_prediction"].shape) == (1, representation.channels, 256, 256)
                values_ok = all(finite_tensor(output[key]) for key in ("encoded_prediction", "bottleneck_loss", "kl", "latent"))
                if config["model"]["kind"] == "satmae_pp":
                    shapes_ok = shapes_ok and tuple(output["encoded_prediction_low"].shape) == (1, representation.channels, 128, 128)
                    values_ok = values_ok and finite_tensor(output["encoded_prediction_low"])
                contract_ok = contract_ok and shapes_ok and values_ok
                objective.backward()
                gradients = gradient_audit()
                finite = all(v["finite"] for v in gradients.values())
                # All parameter gradients, including ungrouped library names,
                # are checked by the same formal trainer's clip operation.
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                report["parameter_updates_completed"] += 1
                synchronize()
                seconds = time.perf_counter() - started
                condition_report["steps"].append({"step_within_condition": step + 1, "objective": float(objective.detach()),
                    "terms": terms, "seconds_forward_backward_optimizer": seconds, "gradient_norm_before_clip": float(norm),
                    "gradients": gradients, "all_finite": finite and shapes_ok and values_ok})
                all_finite_gradients = all_finite_gradients and finite
                if step == 1:
                    meaningful = all(gradients[key]["nonzero_grad_tensors"] > 0 for key in
                                     ("encoder", "decoder", "input_adapter", "decoder_mean"))
                    condition_report["second_step_shared_component_gradients_nonzero"] = meaningful
                    backward_ok = backward_ok and meaningful and finite
                # Do not retain each graph across the next full-size forward.
                del output, objective
            model.eval()
            synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                # Deterministic FP32 inference contracts, independent of train AMP.
                output = model(**model_inputs(batch, representation))
                latent_evidence.append({"condition": condition, "shape": list(output["latent"].shape),
                                        "dtype": str(output["latent"].dtype)})
                if output["indices"] is not None:
                    indices = output["indices"]
                    replay = model.decode_indices(indices, batch["mean_celsius"])
                    difference = float((replay - output["encoded_prediction"]).abs().max())
                    equivalent = bool(torch.allclose(replay, output["encoded_prediction"], atol=1e-5, rtol=1e-5))
                    condition_report["index_only_decode"] = {"passed": equivalent, "maximum_absolute_difference": difference,
                                                               "indices_shape": list(indices.shape), "dtype": str(indices.dtype),
                                                               "decoder_inputs": ["indices", "mean_celsius"]}
                    contract_ok = contract_ok and equivalent
                    rate = model.rate()
                    rate_ok = rate_ok and indices.numel() == rate["indices_per_image"]
                    rate_ok = rate_ok and rate["nominal_index_plus_mean_bits"] == indices.numel() * rate["fixed_bits_per_index"] + 32
                    if hasattr(model, "quantizer") and model.quantizer is not None:
                        decoded_latent = model.quantizer.decode_indices(indices)
                        latent_match = bool(torch.allclose(decoded_latent, output["latent"], atol=1e-5, rtol=1e-5))
                        condition_report["index_latent_roundtrip"] = latent_match
                        contract_ok = contract_ok and latent_match
                else:
                    condition_report["index_only_decode"] = {"applicable": False, "reason": "continuous latent"}
                    rate = model.rate()
                    if "latent_elements" in rate:
                        rate_ok = rate_ok and output["latent"].numel() == rate["latent_elements"]
                    if is_mae:
                        expected_visible = 256 - (26 if condition == "patch16" else 0)
                        visible_ok = output["latent"].shape[1] == expected_visible + 1
                        visible_ok = visible_ok and output["mae_visible_patch_indices"].shape[1] == expected_visible
                        condition_report["native_visible_token_count"] = {"passed": visible_ok, "expected_patches": expected_visible}
                        contract_ok = contract_ok and visible_ok
            synchronize()
            condition_report["seconds_inference_and_optional_decode"] = time.perf_counter() - started
            report["conditions"].append(condition_report)
            del output
        report["latent_layouts"] = latent_evidence
        actual_parameters = sum(p.numel() for p in model.parameters())
        report["actual_parameter_count"] = actual_parameters
        report["cross_model_parameter_alignment"] = {
            "comparison_pending": True,
            "note": "Actual count is measured here; parameter matching requires a separate cross-model audit. No arbitrary 83-87M pass rule is imposed.",
        }
        rate_ok = rate_ok and actual_parameters == report["model_metadata"]["parameters_total"]
        report["gates"]["model_contract"] = bool(contract_ok)
        report["gates"]["backward_finite"] = bool(backward_ok and all_finite_gradients)
        report["gates"]["parameter_rate_audit"] = bool(rate_ok)
        expected_device = torch.device(config["train"]["device"])
        if dev.type == "cuda":
            expected_index = torch.cuda.current_device() if expected_device.index is None else expected_device.index
            same_gpu = expected_device.type == "cuda" and dev.index == expected_index
            micro_batch = int(config["train"]["micro_batch"])
            condition = "patch16" if is_mae else "clean"
            single_batch = make_synthetic_batch(condition)
            # Expand to the real formal microbatch, with materialized tensors:
            # an expand-only stride-zero view must not substitute for its input
            # storage footprint. All samples remain synthetic.
            profile_batch = {
                key: value.repeat((micro_batch,) + (1,) * (value.ndim - 1))
                for key, value in single_batch.items()
            }
            profile_config = copy.deepcopy(config)
            if is_mae:
                profile_config["loss"]["region"] = "missing_only"
            model.train()
            optimizer.zero_grad(set_to_none=True)
            synchronize()
            # AdamW states from the preceding four technical updates remain
            # allocated. Reset counters after zeroing gradients; reset does not
            # remove parameter/optimizer/input storage from current usage.
            torch.cuda.reset_peak_memory_stats(dev)
            baseline_memory = memory(dev)
            started = time.perf_counter()
            accumulation = int(config["train"]["effective_batch"]) // micro_batch
            profile_objective_value = 0.0
            profile_terms = {}
            # Retained gradient buffers affect later accumulation forwards.
            # Match the formal accumulation count rather than profiling only
            # the first microbatch with all .grad fields still empty.
            for _ in range(accumulation):
                with amp(runtime_config, dev):
                    profile_output = forward_training(model, profile_batch, representation)
                    profile_objective, terms = loss(profile_output, profile_batch, representation, profile_config)
                (profile_objective / accumulation).backward()
                profile_objective_value += float(profile_objective.detach()) / accumulation
                for key, value in terms.items():
                    profile_terms[key] = profile_terms.get(key, 0.0) + value / accumulation
                del profile_output, profile_objective
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            report["parameter_updates_completed"] += 1
            synchronize()
            profile_seconds = time.perf_counter() - started
            profile_memory = memory(dev)
            fits_allocated = profile_memory["peak_allocated_gib"] <= config["train"]["memory_limit_gib"]
            fits_reserved = profile_memory["peak_reserved_gib"] <= config["train"]["memory_limit_gib"]
            optimizer_tensors = [value for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)]
            optimizer_bytes = sum(value.numel() * value.element_size() for value in optimizer_tensors)
            report["resource_profile"] = {
                "profiled_micro_batch": micro_batch, "formal_micro_batch": micro_batch,
                "profiled_accumulation_microbatches": accumulation,
                "profiled_effective_batch": micro_batch * accumulation,
                "condition": condition, "loss_region": profile_config["loss"]["region"],
                "profiled_optimizer": "AdamW", "optimizer_state_bytes_present": optimizer_bytes,
                "optimizer_betas": [.9, .95], "training_amp": bool(config["train"]["amp"]),
                "autocast_dtype": "bfloat16" if config["train"]["amp"] else "float32",
                "parameter_dtype": str(next(model.parameters()).dtype),
                "requested_profile_device": str(device), "actual_device": str(dev),
                "configured_formal_device": str(expected_device), "matches_formal_device": same_gpu,
                "memory_fraction_cap_applied_before_model_allocation": True,
                "memory_limit_gib": config["train"]["memory_limit_gib"],
                "baseline_including_parameters_optimizer_inputs": baseline_memory,
                "memory": profile_memory, "within_allocated_memory_limit": fits_allocated,
                "within_reserved_memory_limit": fits_reserved,
                "seconds_forward_backward_clip_optimizer_synchronized": profile_seconds,
                "synthetic_records_per_second": micro_batch * accumulation / profile_seconds,
                "objective": profile_objective_value, "terms": profile_terms,
                "gradient_norm_before_clip": float(gradient_norm),
                "includes_synthetic_forward_backward_optimizer": True,
                "is_formal_throughput_estimate": False,
                "note": "One synthetic update at the formal microbatch and accumulation count; excludes data-loader/I/O and is not a full throughput benchmark.",
            }
            report["gates"]["resource_profile"] = bool(same_gpu and fits_allocated and fits_reserved and optimizer_bytes > 0)
            del profile_batch, single_batch
        else:
            report["resource_profile"] = {
                "profiled_micro_batch": 1, "formal_micro_batch": config["train"]["micro_batch"],
                "passed": False, "reason": "CPU evidence cannot satisfy the explicit CUDA resource gate",
                "is_formal_throughput_estimate": False,
            }
            report["unperformed"].append("explicit CUDA profile at the formal microbatch and AMP/optimizer configuration")
        report["memory"] = memory(dev)
        report["status"] = "synthetic_model_checks_complete_other_gates_pending"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if dev is not None:
            try:
                report["memory"] = memory(dev)
            except Exception:
                pass
    finally:
        report["finished_utc"] = utc()
        report["elapsed_seconds"] = time.perf_counter() - begin
        report["source_hashes_unchanged_during_audit"] = source_hashes() == bound_sources
        if not report["source_hashes_unchanged_during_audit"]:
            report["gates"] = {name: False for name in report["gates"]}
            report["status"] = "invalidated_source_changed_during_audit"
        # These must remain independent and unpassed even when all model
        # operations succeed. Aggregation requires their own bound evidence.
        for name in ("components", "budget_resume", "data_contract"):
            report["gates"][name] = False
        report["does_not_authorize_formal_execution"] = True
        write_json(output_path, report)
        del optimizer, model
    return report
