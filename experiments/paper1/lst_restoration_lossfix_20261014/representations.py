"""Fixed-scale, differentiable LST representations for the restoration study.

The input is a scalar field in degrees Celsius, normally ``T - supplied_mean``.
This module never computes a per-image scale or a mean. ``C``, ``C_power`` and
``C_asinh`` borrow only the
scalar-to-colour construction: it does not assert a match to natural images.
The legacy nearest-segment inverse projects onto a bounded RGB path;
the raw inverse is therefore already bounded, unlike R/G/C_affine inverses.
Nonlinear variants use a training-only fixed scale and a signed bounded warp.
The separately registered trilinear_v1 inverse interpolates path vertex
positions continuously in the RGB cube and is exact on encoded path edges.
"""

from __future__ import annotations

import math
from typing import Any

import torch


_VERTICES = (
    (0.0, 0.0, 0.0),  # black
    (1.0, 0.0, 0.0),  # red
    (1.0, 1.0, 0.0),  # yellow
    (0.0, 1.0, 0.0),  # green
    (0.0, 1.0, 1.0),  # cyan
    (0.0, 0.0, 1.0),  # blue
    (1.0, 0.0, 1.0),  # magenta
    (1.0, 1.0, 1.0),  # white
)


class Representation:
    """A fixed physical-to-network mapping with a differentiable inverse.

    ``jacobian_squared`` is the squared local Jacobian norm after removing
    ``1 / A**2``. For C it is a within-segment quantity, not a statement that
    RGB MSE equals physical MSE across distinct segments.

    C's physical zero encodes as (-1, 1, 0). A zero RGB prediction is not a
    zero-residual prediction. Inverse ties select the earliest path segment.
    """

    def __init__(self, config: dict[str, Any]):
        self.kind = str(config.get("kind", "R"))
        if self.kind not in {"R", "G", "G_power", "G_asinh", "C", "C_affine", "C_power", "C_asinh"}:
            raise ValueError(f"Unknown representation: {self.kind!r}")
        self.A_celsius = float(config.get("A_celsius", 120.0))
        if not math.isfinite(self.A_celsius) or self.A_celsius <= 0:
            raise ValueError("A_celsius must be finite and strictly positive")
        self.tau_celsius = float(config.get("tau_celsius", 2.6950881812941048))
        if not math.isfinite(self.tau_celsius) or self.tau_celsius <= 0:
            raise ValueError("Nonlinear temperature scale must be finite and positive")
        self.power_normalizer = 1. - (1. + self.A_celsius / self.tau_celsius) ** -2
        self.asinh_normalizer = math.asinh(self.A_celsius / self.tau_celsius)
        self.inverse_gradient_policy = config.get("inverse_gradient_policy", "native")
        if self.inverse_gradient_policy not in ("native", "saturated_softpath_v1"):
            raise ValueError("Unknown inverse gradient policy")
        self.colour_inverse = config.get("colour_inverse", "nearest_segment")
        if self.colour_inverse not in ("nearest_segment", "trilinear_v1"):
            raise ValueError("Unknown colour inverse")

    def _continuous_colour_position(self, value):
        """Trilinear extension of the eight vertex positions into the RGB cube.

        Exact on every edge used by encode(), continuous throughout the cube,
        and bounded with clamped RGB outside it. The registered saturated
        policy uses endpoint-tangent gradients outside the cube; forward is
        identical in training and evaluation. This is a new inverse protocol,
        not a claim to preserve historical off-path predictions.
        """
        raw = (value + 1.) * .5
        rgb = raw.clamp(0., 1.)
        if self.inverse_gradient_policy == "saturated_softpath_v1" and raw.requires_grad:
            rgb = rgb.detach() + (raw - raw.detach())
        result = torch.zeros_like(rgb[:, :1])
        for index, vertex in enumerate(_VERTICES):
            weight = torch.ones_like(result)
            for channel, bit in enumerate(vertex):
                coordinate = rgb[:, channel:channel+1]
                weight = weight * (coordinate if bit else 1. - coordinate)
            result = result + index * weight
        return result

    @property
    def channels(self) -> int:
        return 1 if self.kind == "R" else 3

    @property
    def jacobian_squared(self) -> float:
        if self.kind in ("G_power", "G_asinh", "C_power", "C_asinh"):
            raise ValueError("Nonlinear colour mappings require local_jacobian_squared(field), not a constant")
        return {"R": 1.0, "G": 3.0, "C": 49.0, "C_affine": 49.0}[self.kind]

    def local_jacobian_squared(self, field_celsius):
        """Squared RGB derivative per Celsius, within a segment; not global MSE equivalence."""
        field = field_celsius.float()
        if self.kind in ("G_power", "C_power"):
            derivative = 2. * self.tau_celsius ** 2 / ((self.tau_celsius + field.abs()) ** 3 * self.power_normalizer)
        elif self.kind in ("G_asinh", "C_asinh"):
            derivative = 1. / (torch.sqrt(field.square() + self.tau_celsius ** 2) * self.asinh_normalizer)
        else:
            return torch.full_like(field, self.jacobian_squared / self.A_celsius ** 2)
        return (3. if self.kind.startswith("G") else 49.) * derivative.square()

    def _warp(self, field):
        if self.kind in ("G_power", "C_power"):
            # Algebraically sign(r)*(1-(1+abs(r)/tau)^-2), with a nonzero
            # derivative at zero and without subtracting nearly equal numbers.
            magnitude = field.abs()
            return field * (2. * self.tau_celsius + magnitude) / (self.tau_celsius + magnitude).square() / self.power_normalizer
        if self.kind in ("G_asinh", "C_asinh"):
            return torch.asinh(field / self.tau_celsius) / self.asinh_normalizer
        return field / self.A_celsius

    def _unwarp(self, signed_position):
        if self.kind in ("G_power", "C_power"):
            bounded = signed_position.clamp(-1., 1.)
            if self.inverse_gradient_policy == "saturated_softpath_v1" and signed_position.requires_grad:
                # Exact bounded forward; endpoint-tangent surrogate outside support.
                bounded = bounded.detach() + (signed_position - signed_position.detach())
            t = bounded * self.power_normalizer
            root = torch.sqrt((1. - t.abs()).clamp_min(1. - self.power_normalizer))
            return self.tau_celsius * t / (root * (1. + root))
        if self.kind in ("G_asinh", "C_asinh"):
            return self.tau_celsius * torch.sinh(signed_position * self.asinh_normalizer)
        return signed_position * self.A_celsius

    @property
    def zero_encoded(self) -> tuple[float, ...]:
        if self.kind == "R":
            return (0.0,)
        if self.kind in ("G", "G_power", "G_asinh"):
            return (0.0, 0.0, 0.0)
        return (-1.0, 1.0, 0.0)

    @staticmethod
    def _check_shape(value: torch.Tensor, channels: int, name: str) -> None:
        if value.ndim != 4 or value.shape[1] != channels:
            raise ValueError(f"{name} must have shape [B,{channels},H,W], got {tuple(value.shape)}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")

    def encode(self, field_celsius: torch.Tensor) -> torch.Tensor:
        self._check_shape(field_celsius, 1, "field_celsius")
        # Retain float32 precision even when the caller is in autocast.
        field = field_celsius.float()
        if not bool(torch.isfinite(field).all()):
            raise ValueError("Non-finite physical values must be masked/fill-zero before encoding")
        scaled = field / self.A_celsius
        if self.kind == "R":
            return scaled
        if self.kind == "G":
            return scaled.expand(-1, 3, -1, -1)
        if self.kind == "C_affine":
            return torch.cat((-torch.ones_like(scaled), torch.ones_like(scaled), 7.0 * scaled), dim=1)

        # Permit only roundoff-sized excursions, never clip physically out of
        # range observations to make the colour conversion silently succeed.
        tolerance = max(1e-4, self.A_celsius * 1e-6)
        if bool((field.abs() > self.A_celsius + tolerance).any()):
            raise ValueError(
                f"{self.kind} input exceeds fixed support +/-{self.A_celsius:g} C; "
                "change/freeze the protocol or diagnose the observation, do not silently clip"
            )
        warped = self._warp(field.clamp(-self.A_celsius, self.A_celsius)).clamp(-1., 1.)
        if self.kind in ("G_power", "G_asinh"):
            return warped.expand(-1, 3, -1, -1)
        position = (warped + 1.0) * 3.5
        segment = position.floor().long().clamp(0, 6)
        fraction = position - segment.to(position.dtype)
        vertices = field.new_tensor(_VERTICES)
        # Gather [B,1,H,W,3], then put channels back in NCHW order.
        start = vertices[segment[:, 0]].permute(0, 3, 1, 2)
        end = vertices[segment[:, 0] + 1].permute(0, 3, 1, 2)
        rgb = start + fraction * (end - start)
        return 2.0 * rgb - 1.0

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Invert values; selected segment is differentiable almost everywhere.

        R/G/C_affine are unbounded here. Their optional common physical-domain
        constraint is ``project_field`` and must be logged separately by the
        caller. C uses closest Euclidean projection to each RGB segment,
        including endpoint projection. It is not globally one affine inverse.
        """
        self._check_shape(encoded, self.channels, "encoded")
        value = encoded.float()
        if self.kind == "R":
            return value * self.A_celsius
        if self.kind == "G":
            return value.mean(dim=1, keepdim=True) * self.A_celsius
        if self.kind in ("G_power", "G_asinh"):
            return self._unwarp(value.mean(dim=1, keepdim=True))
        if self.kind == "C_affine":
            return value[:, 2:3] * (self.A_celsius / 7.0)

        if self.colour_inverse == "trilinear_v1":
            return self._unwarp(self._continuous_colour_position(value) * (2. / 7.) - 1.)

        rgb = (value + 1.0) * 0.5
        vertices = rgb.new_tensor(_VERTICES)
        best_distance = None
        best_position = None
        best_fraction = None
        candidates = []
        surrogate = self.inverse_gradient_policy == "saturated_softpath_v1" and value.requires_grad
        # Seven small passes avoid materialising [B,7,3,H,W]. Strict '<'
        # preserves the first segment on an exact tie deterministically.
        for index in range(7):
            start = vertices[index].view(1, 3, 1, 1)
            direction = (vertices[index + 1] - vertices[index]).view(1, 3, 1, 1)
            fraction = ((rgb - start) * direction).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            closest = start + fraction * direction
            distance = (rgb - closest).square().sum(dim=1, keepdim=True)
            position = fraction + float(index)
            if surrogate:
                candidates.append((distance, position))
            if best_distance is None:
                best_distance = distance
                best_position = position
                best_fraction = fraction
            else:
                choose = distance < best_distance
                best_distance = torch.where(choose, distance, best_distance)
                best_position = torch.where(choose, position, best_position)
                best_fraction = torch.where(choose, fraction, best_fraction)
        assert best_position is not None
        if surrogate:
            # Fixed temperature 0.5 in RGB [0,1] squared-distance units.
            # Only endpoint-saturated inverses use this surrogate backward.
            # No auxiliary loss and no change to hard predictions/metrics.
            denominator = torch.zeros_like(best_position)
            numerator = torch.zeros_like(best_position)
            for distance, position in candidates:
                weight = torch.exp(-2. * (distance - best_distance.detach()))
                denominator = denominator + weight
                numerator = numerator + weight * position
            soft_position = numerator / denominator
            replacement = best_position.detach() + (soft_position - soft_position.detach())
            saturated = (best_fraction <= 0.) | (best_fraction >= 1.)
            best_position = torch.where(saturated, replacement, best_position)
        return self._unwarp(best_position * (2.0 / 7.0) - 1.0)

    def project_field(self, field_celsius: torch.Tensor) -> torch.Tensor:
        """Explicit common field constraint for separately reported metrics."""
        self._check_shape(field_celsius, 1, "field_celsius")
        return field_celsius.clamp(-self.A_celsius, self.A_celsius)
