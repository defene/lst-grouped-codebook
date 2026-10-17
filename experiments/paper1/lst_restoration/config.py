"""Standard-library configuration and static checks; never imports torch."""
import copy
import hashlib
import json
import math
from pathlib import Path

from . import VERSION

PROJECT = Path(__file__).resolve().parents[3]
PREPARED = "processed_data/hlp_split_v2_20260914"
COUNTS = {"train": 218975, "val": 11684, "test": 11757}
CORE = ("vq", "fsq", "lfq", "bsq")
GEO = ("eo_vae", "terramind_fsq")
MODELS = CORE + ("ae", "var_multiscale", "sd", "sdxl", "flux1", "prithvi_mae", "satmae_pp") + GEO


def is_pretrained(config):
    m = config["model"]
    return m["kind"] in ("sd", "sdxl", "flux1")


def pretrained_identity(config):
    m = config["model"]
    return m.get("asset_sha256", {}) if m["kind"] in GEO else m.get("pretrained_sha256", {})
PROFILE_DEFAULTS = {
    "clean": {"kind": "clean"},
    "pixel10": {"kind": "pixel10", "coverage": .10},
    "sensor_gaussian": {"kind": "sensor_gaussian", "sigma_celsius": .5},
    "sensor_stripe": {"kind": "sensor_stripe", "stripe_sigma_celsius": .25},
    "cloud_missing": {"kind": "cloud_missing", "coverage": .10, "field_grid": 8},
    "cloud_bias": {"kind": "cloud_bias", "coverage": .30, "bias_celsius": 2., "field_grid": 8},
    "joint": {"kind": "joint", "sigma_celsius": .5, "stripe_sigma_celsius": .25,
              "coverage": .10, "field_grid": 8},
    "patch16": {"kind": "patch16", "coverage": .10, "patch_size": 16},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def profile(name, **overrides):
    value = copy.deepcopy(PROFILE_DEFAULTS[name])
    value.update(seed=20260925, **overrides)
    return value


def default_config(model="vq", representation="R", training="mixed", seed=17):
    if training == "mixed":
        # Equal-probability deterministic assignment; realized counts can differ.
        conditions = [profile(x) for x in ("clean", "pixel10", "sensor_gaussian",
                                          "sensor_stripe", "cloud_missing", "joint")]
    else:
        conditions = [profile(training)]
    channels = 1 if representation == "R" else 3
    config = {
        "version": VERSION, "seed": seed,
        "name": f"{model}_{representation}_{training}_seed{seed}",
        "execution": {"enabled": False, "preflight_report": None, "alignment_report": None},
        "data": {"root": str(PROJECT), "prepared": PREPARED, "dataset": "prithvi",
                 "display_name": "HLS-LST-Pairs", "modality": "lst", "counts": COUNTS.copy(),
                 "shape": [1, 256, 256], "historical_test_exposure": True},
        "task": {"mean_source": "oracle", "field": "residual", "mean_side_bits": 32,
                 "mean_scale_celsius": 16.29419157454727, "mean_center_celsius": 0.,
                 "absolute_center_celsius": None, "temperature_bounds_celsius": [-50., 70.],
                 "residual_std_celsius": 2.6950881812941048,
                 "scale_evidence": "runs/paper1/lst_residual_scale_20260918/residual_scale.json"},
        "representation": {"kind": representation, "A_celsius": 120.},
        "model": {"kind": model, "channels": channels, "base_channels": 128, "num_res_blocks": 3,
                  "latent_dim": 32, "grid": 16, "codebook_size": 1024, "levels": [8, 8, 16],
                  "bits_per_token": 10, "commitment": .25, "entropy_weight": .01,
                  "entropy_temperature": .1, "checkpointing": False,
                  "pretrained_path": None, "pretrained_sha256": {}},
        "corruption": {"training_name": training, "train_profiles": conditions,
                       "profile_selection_seed": 20260925, "synthetic_not_sensor_calibrated": True},
        "loss": {"pixel": 1., "gradient": .1, "representation": .01,
                 "bottleneck": 1., "kl": 1e-6, "region": "all_valid"},
        "train": {"epochs": 5, "effective_batch": 64, "micro_batch": 4, "workers": 0,
                  "lr": 1e-4, "weight_decay": .01, "grad_clip": 1., "amp": True,
                  "memory_limit_gib": 8.5, "device": "cuda:0", "checkpoint_every": 100,
                  "sample_order_seed": seed, "views": 1, "selection": "final_epoch",
                  "record_presentations": 1094875, "updates": 17110},
        "evaluation": {"profiles": [profile(x) for x in PROFILE_DEFAULTS if x != "patch16"],
                       "splits": ["val", "test"], "views": 2, "batch_size": 4,
                       "bootstrap_replicates": 1000, "bootstrap_seed": 20260925,
                       "save_gallery_samples": 8, "codec_samples": 8},
        "scope": {"downstream": False, "generation": False, "hls_training": False,
                  "cross_view_training": False, "real_cloud_claim": False},
    }
    if model in GEO:
        config["model"].update(initialization="scratch", source_root=None, source_sha256={},
                               architecture_adaptation="LST fixed-channel restoration")
        if model == "eo_vae":
            config["model"].update(ch_mult=[1, 2, 4, 4], num_res_blocks=2,
                                   grid=32, use_dynamic_ops=False,
                                   source_repo="nilsleh/eo-vae",
                                   source_revision="8e65a27546d6ad5c7b0b349024a27702d49f9333")
        else:
            config["model"].update(levels=[8, 8, 16], codebook_size=1024,
                bits_per_token=10, latent_dim=3, decoder_steps=50, decoder_seed=20260927,
                source_repo="terrastackai/terratorch",
                source_revision="0f3873f08058d991b638db8c5b71478777beb2ac")
    if representation in ("C_power", "C_asinh"):
        config["representation"].update(tau_celsius=config["task"]["residual_std_celsius"],
            scale_evidence=config["task"]["scale_evidence"],
            mapping_origin="signed bounded LST adaptation; not the exact nonnegative-depth formula")
    if model in ("prithvi_mae", "satmae_pp"):
        config["loss"]["region"] = "missing_only" if training == "patch16" else "all_valid"
        config["evaluation"]["profiles"] = [profile("clean"), profile("patch16")]
    if model == "satmae_pp":
        config["model"].update(initialization="scratch", encoder_dim=768, encoder_depth=9,
            encoder_heads=12, decoder_dim=512, decoder_depth=7, decoder_heads=8,
            encoder_resolution=128, encoder_patch_size=8, upsample_projection_ratio=4,
            source_repo="techmn/satmae_pp", source_revision="34f46564e4de1e4108fc12dc191fac600d62441c")
        config["loss"].update(pixel=0., gradient=0., representation=0., bottleneck=0., kl=0.,
                              satmae_low_mse=1., satmae_high_l1=1.)
    return config


def validate(config):
    errors = []
    def check(condition, description):
        if not condition:
            errors.append(description)
    c, d, t, m = config, config["data"], config["train"], config["model"]
    check(c["version"] == VERSION, "Unsupported protocol version")
    check(d["prepared"] == PREPARED and d["counts"] == COUNTS, "Frozen HLP v2 membership required")
    check(d["dataset"] == "prithvi" and d["modality"] == "lst", "This runtime is LST/HLP only")
    check(d["shape"] == [1, 256, 256], "Expected native LST 256x256")
    check(m["kind"] in MODELS, "Unknown model")
    rep = c["representation"]["kind"]
    check(rep in ("R", "G", "C", "C_affine", "C_power", "C_asinh"), "Unknown representation")
    check(m["channels"] == (1 if rep == "R" else 3), "Representation/model channels differ")
    check(c["representation"]["A_celsius"] > 0, "Representation scale must be positive")
    if rep in ("C_power", "C_asinh"):
        check(c["representation"].get("tau_celsius", 2.6950881812941048) > 0,
              "Nonlinear mapping needs a positive, train-only fixed scale")
    check(c["task"]["residual_std_celsius"] > 0 and c["task"]["mean_scale_celsius"] > 0, "Positive physical scales required")
    check(c["task"]["mean_source"] in ("oracle", "visible"), "Unknown mean condition")
    check(c["task"]["field"] in ("residual", "absolute"), "Unknown predicted field")
    if c["task"]["field"] == "absolute":
        check(isinstance(c["task"]["absolute_center_celsius"], (int, float)), "Absolute control needs frozen train-only center")
    check(t["epochs"] == 5 and t["views"] == 1, "Formal budget is five exact epochs / one view per presentation")
    check(t["micro_batch"] > 0 and t["effective_batch"] % t["micro_batch"] == 0, "Invalid gradient accumulation")
    check(t["updates"] == t["epochs"] * math.ceil(COUNTS["train"] / t["effective_batch"]), "Update budget mismatch")
    check(t["record_presentations"] == t["epochs"] * COUNTS["train"], "Record budget mismatch")
    check(t["selection"] == "final_epoch", "This frozen runner uses final epoch; no test selection")
    check(not any(c["scope"].values()), "Out-of-scope downstream/HLS/generation/cross-view/real-cloud claim")
    check(c["evaluation"]["views"] == 2, "Two registered evaluation views required")
    check(c["evaluation"]["splits"] == ["val", "test"], "Full validation and test required")
    if m["kind"] in CORE:
        check(m["grid"] == 16, "Core token grid must be 16x16")
        check(m["codebook_size"] == 2 ** m["bits_per_token"], "Core nominal state count / bit width mismatch")
        if m["kind"] == "fsq":
            check(math.prod(m["levels"]) == m["codebook_size"], "FSQ product must equal nominal K")
        check(t["micro_batch"] == 4, "Core comparisons freeze microbatch=4; LFQ/BSQ batch entropy changes with microbatch")
    if m["kind"] in ("sd", "sdxl", "flux1"):
        check(rep != "R", "Natural RGB VAEs require G or C / affine representation")
    if m["kind"] == "satmae_pp":
        check(m.get("initialization") == "scratch" and not m.get("pretrained_path"), "SatMAE++ adaptation must train from scratch")
        check(m.get("encoder_resolution") == 128 and m.get("encoder_patch_size") == 8,
              "SatMAE++ uses 128->256 with patch8 matching native patch16 footprint")
        check(all(p["kind"] in ("clean", "patch16") for p in c["corruption"]["train_profiles"] + c["evaluation"]["profiles"]),
              "SatMAE++ is registered only for the separate clean/patch16 MAE comparison")
        for prefix in ("encoder", "decoder"):
            check(m[prefix + "_dim"] % m[prefix + "_heads"] == 0 and m[prefix + "_dim"] % 4 == 0,
                  "SatMAE++ dimensions must support heads and 2D positions")
    if m["kind"] in GEO:
        check(m.get("initialization") == "scratch", "Geospatial architecture comparisons train from scratch on LST")
        check(not m.get("pretrained_path") and not m.get("official_weights"), "Do not load unrelated modality checkpoints")
        if m["kind"] == "eo_vae":
            check(m.get("use_dynamic_ops") is False, "Single-field LST uses explicit fixed-channel EO-VAE heads")
            check(m["grid"] == 256 // (2 ** (len(m["ch_mult"]) - 1)), "EO-VAE latent grid must match its downsampling")
        else:
            check(m["levels"] == [8, 8, 16] and m["codebook_size"] == 1024 and
                  m["bits_per_token"] == 10 and m["latent_dim"] == 3 and m["grid"] == 16,
                  "LST TerraMind FSQ uses aligned 1024 states and 16x16 grid")
            check(m.get("decoder_steps") == 50 and m.get("decoder_seed") == 20260927,
                  "Frozen TerraMind sampler requires 50 steps and public seed 20260927")
    for p in c["corruption"]["train_profiles"] + c["evaluation"]["profiles"]:
        check(p["kind"] in PROFILE_DEFAULTS, "Unknown corruption")
        check(0 <= p.get("coverage", 0) < 1, "Coverage must preserve observations")
        if p["kind"] in ("pixel10", "patch16"):
            check(p.get("coverage", .1) == .1, "pixel10/patch16 are fixed at ten percent sampling")
        grid = p.get("field_grid", 8)
        check(isinstance(grid, int) and 2 <= grid <= 256, "Cloud field grid must be an integer in [2,256]")
        for k in ("sigma_celsius", "stripe_sigma_celsius", "bias_celsius"):
            check(p.get(k, 0) >= 0, "Noise magnitude must be nonnegative")
    return {"ok": not errors, "errors": errors, "config_sha256": digest(config),
            "execution_enabled": c["execution"]["enabled"], "runtime_verified": False}


def study_plan():
    jobs = []
    def add(tier, models, reps, trainings, seeds=(17,), mean="oracle"):
        for model in models:
            for rep in reps:
                for training in trainings:
                    for seed in seeds:
                        c = default_config(model, rep, training, seed)
                        c["task"]["mean_source"] = mean
                        if mean != "oracle":
                            c["name"] += "_" + mean
                        c["tier"] = tier
                        jobs.append(c)
    add("core_seed17", CORE, ("R", "C"), ("clean", "pixel10", "mixed"))
    add("core_replication", CORE, ("R", "C"), ("clean", "pixel10", "mixed"), (23, 42))
    add("gray_control", CORE, ("G",), ("clean", "pixel10", "mixed"))
    add("natural_pretrained", ("sd", "sdxl", "flux1"), ("G", "C"), ("clean", "pixel10", "mixed"))
    add("geo_continuous_scratch", ("eo_vae",), ("R", "C"), ("clean", "pixel10", "mixed"))
    add("geo_discrete_diffusion_scratch", ("terramind_fsq",), ("R", "C"), ("clean", "pixel10", "mixed"))
    add("visible_mean_control", ("vq", "fsq"), ("R", "C"), ("pixel10", "mixed"), mean="visible")
    add("geo_mae_separate", ("prithvi_mae", "satmae_pp"), ("R", "C"), ("clean", "patch16"))
    add("affine_gain_control", ("vq", "fsq"), ("C_affine",), ("clean", "mixed"))
    # Direct temperature control uses the already frozen train-only equal-image
    # mean, not a newly fitted pixel-weighted statistic or per-image centering.
    absolute = []
    for model in CORE:
        for rep in ("R", "C"):
            for training in ("clean", "pixel10", "mixed"):
                c = default_config(model, rep, training, 17)
                c["name"] += "_absolute"
                c["tier"] = "absolute_temperature_control"
                c["task"].update(field="absolute", absolute_center_celsius=25.366523070738997,
                    absolute_center_source="frozen v2 normalization; equal-image valid-pixel mean /100")
                absolute.append(c)
    jobs.extend(absolute)
    for bits, levels in ((8, [8, 8, 4]), (12, [8, 8, 8, 8])):
        for model in CORE:
            for rep in ("R", "C"):
                c = default_config(model, rep, "mixed", 17)
                c["name"] += f"_rate{bits}"
                c["tier"] = "rate_distortion"
                c["model"].update(bits_per_token=bits, codebook_size=2**bits, levels=levels)
                jobs.append(c)
    # Severe conditions are predeclared evaluation stresses, never tuned on test.
    for c in jobs:
        if c["tier"] != "geo_mae_separate":
            c["evaluation"]["profiles"] += [profile("sensor_gaussian", sigma_celsius=1.),
                profile("cloud_missing", coverage=.30), profile("joint", coverage=.30, sigma_celsius=1.)]
    return jobs


def export_plan(directory):
    directory = Path(directory)
    jobs = study_plan()
    errors = {c["name"]: validate(c)["errors"] for c in jobs if not validate(c)["ok"]}
    if errors:
        raise ValueError(errors)
    for c in jobs:
        write_json(directory / "configs" / (c["name"] + ".json"), c)
    tiers = {tier: sum(c["tier"] == tier for c in jobs) for tier in sorted({c["tier"] for c in jobs})}
    summary = {"version": VERSION, "status": "implementation_static_only", "training_jobs_started": 0,
               "planned_jobs": len(jobs), "tiers": tiers, "runtime_verified": False,
               "downstream_excluded": True, "jobs": [{"name": c["name"], "tier": c["tier"],
                   "sha256": digest(c), "file": "configs/" + c["name"] + ".json"} for c in jobs]}
    write_json(directory / "manifest.json", summary)
    return summary
