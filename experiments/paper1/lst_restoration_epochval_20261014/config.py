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
MODELS = CORE + ("ae", "var_multiscale", "sd", "sdxl", "flux1", "prithvi_mae", "satmae_pp") + GEO + ("vqgan_pretrained", "vq_drifting", "grouped_g8k256")


def is_pretrained(config):
    m = config["model"]
    return m["kind"] in ("sd", "sdxl", "flux1", "vqgan_pretrained")


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


def default_config(model="vq", representation="R", training="pixel10", seed=17):
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
        "data": {"root": ".", "prepared": PREPARED, "dataset": "prithvi",
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
                  "sample_order_seed": seed, "views": 1, "selection": "best_validation_rmse",
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
    if representation in ("G_power", "G_asinh", "C_power", "C_asinh"):
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
    from .epoch_validation import default_protocol
    config['epoch_validation'] = default_protocol()
    config['train']['selection'] = 'best_validation_rmse'
    config['evaluation'].update(stage='selection', splits=['val'])
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
    check(c["representation"].get("inverse_gradient_policy", "native") in
          ("native", "saturated_softpath_v1"), "Unknown inverse gradient policy")
    tracking = c.get("tracking", {})
    if tracking.get("enabled", False):
        check(all(isinstance(tracking.get(k), str) and tracking[k] for k in
                  ("entity", "project", "run_id", "name", "group")), "Missing W&B run identity")
        check(isinstance(tracking.get("every_updates"), int) and tracking["every_updates"] > 0,
              "W&B logging interval must be a positive number of updates")
    check(rep in ("R", "G", "G_power", "G_asinh", "C", "C_affine", "C_power", "C_asinh"), "Unknown representation")
    check(m["channels"] == (1 if rep == "R" else 3), "Representation/model channels differ")
    check(c["representation"]["A_celsius"] > 0, "Representation scale must be positive")
    if rep in ("G_power", "G_asinh", "C_power", "C_asinh"):
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
    check(t["selection"] == "best_validation_rmse", "Use the preregistered validation selector")
    from .epoch_validation import protocol_errors
    errors.extend(protocol_errors(c))
    check(not any(c["scope"].values()), "Out-of-scope downstream/HLS/generation/cross-view/real-cloud claim")
    check(c["evaluation"]["views"] == 2, "Two registered evaluation views required")
    stage = c["evaluation"].get("stage", "final")
    check(stage in ("selection", "final"), "Unknown evaluation stage")
    expected_splits = ["val"] if stage == "selection" else ["val", "test"]
    check(c["evaluation"]["splits"] == expected_splits,
          "Selection must use validation only; final evaluation requires validation and test")
    profiles = c["evaluation"]["profiles"]
    check(bool(profiles) and len({digest(p) for p in profiles}) == len(profiles),
          "Evaluation profiles must be nonempty and unique")
    if m["kind"] in CORE:
        check(m["grid"] == 16, "Core token grid must be 16x16")
        check(m["codebook_size"] == 2 ** m["bits_per_token"], "Core nominal state count / bit width mismatch")
        if m["kind"] == "fsq":
            check(math.prod(m["levels"]) == m["codebook_size"], "FSQ product must equal nominal K")
        micro_protocol = t.get("micro_batch_protocol", "legacy-micro4")
        check(micro_protocol in ("legacy-micro4", "core-micro16-effective64-20260930"),
              "Unknown core microbatch protocol")
        if micro_protocol == "core-micro16-effective64-20260930":
            check(t["micro_batch"] == 16 and t["effective_batch"] == 64,
                  "New core protocol freezes microbatch16 and effective batch64")
        else:
            check(t["micro_batch"] == 4,
                  "Legacy core protocol freezes microbatch4; changing it requires a new bound protocol")
    if m["kind"] in ("sd", "sdxl", "flux1"):
        check(rep != "R", "Natural RGB VAEs require G or C / affine representation")
    if m['kind'] == 'grouped_g8k256':
        check((m['groups'], m['codebook_size'], m['latent_dim']) == (8, 256, 32), 'Frozen g8k256 required')
        check(m['scales'] == [1,2,3,4,5,6,8,10,13,16], 'Pinned multiscale recipe required')
        check(m['base_channels'] == 160 and m['num_res_blocks'] == 2, 'Pinned grouped backbone required')
        check(t['effective_batch'] == 64 and t['micro_batch'] == 8, 'Grouped batch contract required')
        check(t['warmup_updates'] == 500 and t['lr'] == 1e-4, 'Pinned grouped LR required')
        check(not m['pretrained_path'], 'Grouped initialization must be scratch')
    if m["kind"] in ("vqgan_pretrained", "vq_drifting"):
        check(m['grid'] == 16 and m['codebook_size'] == 1024 and m['bits_per_token'] == 10,
              'Stage2 single-scale f16 K1024 rate is frozen')
        check(t['effective_batch'] == 64, 'Stage2 effective batch64 required')
        check(c['loss']['representation'] == 0, 'Stage2 physical-only reconstruction required')
        check(c['evaluation']['splits'] == ['val'], 'Stage2 selection uses validation only')
        if m['kind'] == 'vqgan_pretrained':
            check(m['latent_dim'] == 256 and m['num_res_blocks'] == 2 and t['micro_batch'] == 8,
                  'Official VQGAN architecture and profiled microbatch8 required')
            check(bool(m['pretrained_sha256']) and bool(m['pretrained_path']), 'Verified pretrained assets required')
        else:
            check(m['latent_dim'] == 32 and t['micro_batch'] == 16 and not m['pretrained_path'],
                  'Drifting scratch D32/micro16 required')
            check(m['drift_tau'] == .2 and m['drift_beta'] == .25 and m['drift_tokens'] == 512,
                  'Frozen drifting recipe required')
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
