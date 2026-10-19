"""Equal-image physical supervision, independent of representation channel count."""
import torch
from .representations import Representation


def masked_per_image(value, mask):
    mask = mask.to(value.dtype).expand_as(value)
    numerator = (value * mask).flatten(1).sum(1)
    count = mask.flatten(1).sum(1)
    return numerator / count.clamp_min(1), count


def model_inputs(batch, representation):
    # Explicit whitelist: no target, synthetic noise support or clean mean outside
    # the registered oracle/visible scalar condition is handed to the encoder.
    # Canonical NCHW prevents colour gather's channels-last strides from
    # selecting a different TF32 convolution path in official VAE comparisons.
    # Values and the adapter's actual NCHW training path are unchanged.
    return {"encoded_input": representation.encode(batch["input_field"]).contiguous(),
            "reference_valid": batch["reference_valid"], "observed_valid": batch["observed_valid"],
            "mean_celsius": batch["mean_celsius"], "patch_missing": batch["patch_missing"]}


def forward_training(model, batch, representation):
    inputs = model_inputs(batch, representation)
    if hasattr(model, "training_prediction"):
        # Diffusion supervision belongs only to the decoder's training path.
        # Evaluation always uses model_inputs, which excludes this target.
        return model.training_prediction(inputs, representation.encode(batch["target_field"]))
    return model(**inputs)


def readout(output, batch, representation, config, constrained=False):
    field = representation.decode(output["encoded_prediction"].float())
    if constrained:
        field = representation.project_field(field)
    temperature = field + batch["center_celsius"].float().reshape(-1, 1, 1, 1)
    if constrained:
        temperature = temperature.clamp(*config["task"]["temperature_bounds_celsius"])
    return temperature


def loss(output, batch, representation, config):
    prediction = readout(output, batch, representation, config)
    valid = batch["reference_valid"].bool()
    region = valid if config["loss"]["region"] == "all_valid" else valid & ~batch["observed_valid"].bool()
    if not region.flatten(1).any(1).all():
        raise ValueError("Loss region empty; do not silently replace missing-only MAE objective")
    scale = config["task"]["residual_std_celsius"]
    error = (prediction - batch["temperature"].float()) / scale
    pixel = masked_per_image(error.square(), region)[0].mean()
    # Adjacent gradients are valid only where BOTH reference pixels exist.
    # For missing-only MAE include edges touching a missing reference pixel.
    edge_sums = torch.zeros(error.shape[0], device=error.device)
    edge_counts = torch.zeros_like(edge_sums)
    for axis in (-1, -2):
        a, b = [slice(None)] * 4, [slice(None)] * 4
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        edges = valid[a] & valid[b] & (region[a] | region[b])
        edge_sums += ((error[a] - error[b]).square() * edges).flatten(1).sum(1)
        edge_counts += edges.flatten(1).sum(1)
    gradient = (edge_sums / edge_counts.clamp_min(1)).mean()
    target = representation.encode(batch["target_field"])
    # Sum channels, THEN average pixels; k already accounts for channel Jacobian.
    encoded_error = (output["encoded_prediction"].float() - target.float()).square().sum(1, keepdim=True)
    encoded_error = encoded_error / (representation.local_jacobian_squared(batch["target_field"]) * scale ** 2)
    representation_loss = masked_per_image(encoded_error, region)[0].mean()
    native = output.get("bottleneck_loss", prediction.new_zeros(()))
    kl = output.get("kl", prediction.new_zeros(()))
    terms = {"pixel": pixel, "gradient": gradient, "representation": representation_loss,
             "bottleneck": native, "kl": kl}
    if config["model"]["kind"] == "satmae_pp":
        from torch.nn import functional as F
        low = representation.decode(output["encoded_prediction_low"].float())
        support = F.avg_pool2d(valid.float(), 2)
        target_low = F.avg_pool2d(torch.where(valid, batch["target_field"].float(), 0.), 2)
        target_low = target_low / support.clamp_min(.25)
        # Physical patch boundaries align at both resolutions; partial native
        # no-data blocks retain only the count of genuinely valid target pixels.
        low_region = F.avg_pool2d(region.float(), 2)
        terms["satmae_low_mse"] = masked_per_image(((low - target_low) / scale).square(), low_region)[0].mean()
        terms["satmae_high_l1"] = masked_per_image(error.abs(), region)[0].mean()
    total = sum(config["loss"][key] * value for key, value in terms.items())
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite restoration objective")
    return total, {key: float(value.detach()) for key, value in terms.items()}
