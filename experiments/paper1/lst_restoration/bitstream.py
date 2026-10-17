"""Versioned fixed-width discrete-index transport using only the standard library.

The body contains only integer tokens, an optional shared float32 mean and
optional reference/observation masks. It contains no observation pixels, target
pixels, floating encoder features or image skip path. Shared decoder weights and
codebook/representation configuration remain external and must be identified in
the experiment manifest. This is an actual fixed-width codec, not entropy coding.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import struct
from collections.abc import Sequence
from typing import Any


MAGIC = b"LSTRIDX\x00"
VERSION = 1
_PREFIX = struct.Struct("<8sI")
_MAX_HEADER_BYTES = 1024 * 1024


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _shape(shape: Sequence[int]) -> tuple[int, ...]:
    try:
        result = tuple(_integer(value, "shape dimension") for value in shape)
    except TypeError as error:
        raise ValueError("shape must be a nonempty integer sequence") from error
    if not result or len(result) > 8 or any(value <= 0 for value in result):
        raise ValueError("shape requires 1 to 8 positive dimensions")
    return result


def _array(value: Any) -> tuple[list[Any], tuple[int, ...]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [value], ()
    if not value:
        raise ValueError("empty arrays are not supported")
    flattened: list[Any] = []
    tail_shape = None
    for child in value:
        child_values, child_shape = _array(child)
        if tail_shape is not None and child_shape != tail_shape:
            raise ValueError("ragged arrays cannot be serialized")
        tail_shape = child_shape
        flattened.extend(child_values)
    return flattened, (len(value),) + (tail_shape or ())


def _pack(values: list[int], bits: int) -> bytes:
    """Pack each integer least-significant bit first, then fill bytes low first."""
    output = bytearray()
    accumulator = 0
    occupied = 0
    for value in values:
        accumulator |= value << occupied
        occupied += bits
        while occupied >= 8:
            output.append(accumulator & 255)
            accumulator >>= 8
            occupied -= 8
    if occupied:
        output.append(accumulator & 255)
    return bytes(output)


def _unpack(payload: bytes, count: int, bits: int) -> list[int]:
    if len(payload) != (count * bits + 7) // 8:
        raise ValueError("packed segment length does not match count and bit width")
    remainder = (count * bits) % 8
    if remainder and payload[-1] >> remainder:
        raise ValueError("nonzero unused padding bits")
    output: list[int] = []
    accumulator = 0
    occupied = 0
    offset = 0
    mask = (1 << bits) - 1
    for _ in range(count):
        while occupied < bits:
            accumulator |= payload[offset] << occupied
            offset += 1
            occupied += 8
        output.append(accumulator & mask)
        accumulator >>= bits
        occupied -= bits
    return output


def _mask_segment(mask: Any) -> tuple[bytes, dict] | None:
    if mask is None:
        return None
    values, shape = _array(mask)
    _shape(shape)
    result = []
    for value in values:
        if isinstance(value, bool):
            result.append(int(value))
        else:
            integer = _integer(value, "mask value")
            if integer not in (0, 1):
                raise ValueError("mask values must be boolean or integer 0/1")
            result.append(integer)
    packed = _pack(result, 1)
    return packed, {"shape": list(shape), "count": len(result), "bytes": len(packed)}


def encode_indices(
    indices: Any,
    shape: Sequence[int],
    codebook_size: int,
    mean_celsius: float | None = None,
    reference_mask: Any = None,
    observed_mask: Any = None,
) -> bytes:
    """Serialize one image's joint integer tokens and declared side information.

    Flat or nested array-like indices are accepted; ``shape`` explicitly defines
    the decoder's grid/scale shape and must contain exactly that many indices.
    K=1000 still uses 10 bits/token. Mixed-radix FSQ must supply its combined joint
    index, not the sum of independent marginal bit estimates. Masks preserve
    their own shape. Omitted masks cost zero here and must not be silently
    assumed free if the deployment decoder needs them.
    """
    dimensions = _shape(shape)
    codebook_size = _integer(codebook_size, "codebook_size")
    if not 2 <= codebook_size <= 2**64:
        raise ValueError("codebook_size must lie in [2, 2**64]")
    bits = (codebook_size - 1).bit_length()
    values, _ = _array(indices)
    if len(values) != math.prod(dimensions):
        raise ValueError("explicit shape does not match the number of indices")
    values = [_integer(value, "index") for value in values]
    if any(value < 0 or value >= codebook_size for value in values):
        raise ValueError("index outside declared codebook")
    token_segment = _pack(values, bits)
    mean_segment = b""
    if mean_celsius is not None:
        if isinstance(mean_celsius, bool) or not math.isfinite(float(mean_celsius)):
            raise ValueError("mean_celsius must be a finite real value")
        try:
            mean_segment = struct.pack("<f", float(mean_celsius))
        except (OverflowError, struct.error) as error:
            raise ValueError("mean_celsius does not fit finite float32") from error
        if not math.isfinite(struct.unpack("<f", mean_segment)[0]):
            raise ValueError("mean_celsius does not fit finite float32")
    reference_segment = _mask_segment(reference_mask)
    observed_segment = _mask_segment(observed_mask)
    if reference_segment is not None and observed_segment is not None:
        if reference_segment[1]["shape"] != observed_segment[1]["shape"]:
            raise ValueError("reference and observed mask shapes must agree")
        rv, _ = _array(reference_mask)
        ov, _ = _array(observed_mask)
        if any(bool(o) and not bool(r) for r, o in zip(rv, ov)):
            raise ValueError("observed_mask must be a subset of reference_mask")
    body = token_segment + mean_segment
    for segment in (reference_segment, observed_segment):
        if segment is not None:
            body += segment[0]
    header = {
        "version": VERSION,
        "packing": "little-bit-first",
        "shape": list(dimensions),
        "count": len(values),
        "codebook_size": codebook_size,
        "bits_per_token": bits,
        "index_bytes": len(token_segment),
        "mean_float32_le": bool(mean_segment),
        "reference_mask": reference_segment[1] if reference_segment else None,
        "observed_mask": observed_segment[1] if observed_segment else None,
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded_header) > _MAX_HEADER_BYTES:
        raise ValueError("header is too large")
    return _PREFIX.pack(MAGIC, len(encoded_header)) + encoded_header + body


def _parse(payload: bytes) -> tuple[dict, bytes, int]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("payload must be bytes-like")
    payload = bytes(payload)
    if len(payload) < _PREFIX.size:
        raise ValueError("truncated prefix")
    magic, length = _PREFIX.unpack_from(payload)
    if magic != MAGIC or length > _MAX_HEADER_BYTES or _PREFIX.size + length > len(payload):
        raise ValueError("invalid magic or header length")
    try:
        header = json.loads(payload[_PREFIX.size:_PREFIX.size + length].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid UTF-8 JSON header") from error
    if not isinstance(header, dict) or header.get("version") != VERSION or header.get("packing") != "little-bit-first":
        raise ValueError("unsupported stream version or packing")
    required = {"version", "packing", "shape", "count", "codebook_size", "bits_per_token", "index_bytes", "mean_float32_le", "reference_mask", "observed_mask", "body_sha256"}
    if set(header) != required or isinstance(header["version"], bool):
        raise ValueError("stream header schema mismatch")
    dimensions = _shape(header.get("shape", ()))
    count = _integer(header.get("count"), "count")
    codebook_size = _integer(header.get("codebook_size"), "codebook_size")
    bits = _integer(header.get("bits_per_token"), "bits_per_token")
    index_bytes = _integer(header.get("index_bytes"), "index_bytes")
    if count != math.prod(dimensions) or not 2 <= codebook_size <= 2**64:
        raise ValueError("invalid count, shape or codebook size")
    if bits != (codebook_size - 1).bit_length() or index_bytes != (count * bits + 7) // 8:
        raise ValueError("invalid token width or byte count")
    if not isinstance(header.get("mean_float32_le"), bool):
        raise ValueError("mean flag must be a boolean")
    expected_bytes = index_bytes + (4 if header["mean_float32_le"] else 0)
    for key in ("reference_mask", "observed_mask"):
        descriptor = header.get(key)
        if descriptor is None:
            continue
        if not isinstance(descriptor, dict):
            raise ValueError("invalid mask descriptor")
        mask_shape = _shape(descriptor.get("shape", ()))
        mask_count = _integer(descriptor.get("count"), "mask count")
        mask_bytes = _integer(descriptor.get("bytes"), "mask bytes")
        if mask_count != math.prod(mask_shape) or mask_bytes != (mask_count + 7) // 8:
            raise ValueError("mask shape and byte count disagree")
        expected_bytes += mask_bytes
    body = payload[_PREFIX.size + length:]
    if len(body) != expected_bytes:
        raise ValueError("truncated stream or unexpected trailing bytes")
    if header.get("body_sha256") != hashlib.sha256(body).hexdigest():
        raise ValueError("body integrity check failed")
    return header, body, _PREFIX.size + length


def decode_indices(payload: bytes) -> dict:
    """Decode tokens and side information; masks return {shape, values} or None.

    The caller reconstructs the integer tensor from ``shape`` and passes only
    these indices and the shared mean into its decoder. Header integrity does
    not authenticate a sender or establish model/checkpoint identity.
    """
    header, body, _ = _parse(payload)
    cursor = header["index_bytes"]
    indices = _unpack(body[:cursor], header["count"], header["bits_per_token"])
    if any(value >= header["codebook_size"] for value in indices):
        raise ValueError("decoded token outside declared codebook")
    mean = None
    if header["mean_float32_le"]:
        mean = struct.unpack("<f", body[cursor:cursor + 4])[0]
        cursor += 4
        if not math.isfinite(mean):
            raise ValueError("nonfinite float32 mean")
    result = {
        "indices": indices, "shape": header["shape"], "codebook_size": header["codebook_size"],
        "bits_per_token": header["bits_per_token"], "mean_celsius": mean,
    }
    for key in ("reference_mask", "observed_mask"):
        descriptor = header[key]
        if descriptor is None:
            result[key] = None
            continue
        segment = body[cursor:cursor + descriptor["bytes"]]
        cursor += descriptor["bytes"]
        values = _unpack(segment, descriptor["count"], 1)
        result[key] = {"shape": descriptor["shape"], "values": [bool(value) for value in values]}
    reference = result["reference_mask"]
    observed = result["observed_mask"]
    if reference is not None and observed is not None:
        if reference["shape"] != observed["shape"] or any(o and not r for r, o in zip(reference["values"], observed["values"])):
            raise ValueError("decoded observed mask must match shape and be a subset of reference mask")
    return result


def rate_audit(payload: bytes, pixels: int) -> dict:
    """Measured payload rate including framing, mean, masks and byte padding.

    ``pixels`` is the explicitly declared image-domain denominator, normally
    H*W, not the number of observed pixels. Learned weights and any deployment
    metadata outside this payload are excluded and must be reported separately.
    """
    pixels = _integer(pixels, "pixels")
    if pixels <= 0:
        raise ValueError("pixels must be positive")
    header, _, header_bytes = _parse(payload)
    reference_bytes = header["reference_mask"]["bytes"] if header["reference_mask"] else 0
    observed_bytes = header["observed_mask"]["bytes"] if header["observed_mask"] else 0
    nominal_token_bits = header["count"] * header["bits_per_token"]
    mean_bytes = 4 if header["mean_float32_le"] else 0
    return {
        "codec": "fixed_width_indices_v1", "pixels": pixels,
        "token_count": header["count"], "codebook_size": header["codebook_size"],
        "bits_per_token": header["bits_per_token"], "nominal_token_bits": nominal_token_bits,
        "nominal_token_bpp": nominal_token_bits / pixels,
        "nominal_token_plus_mean_bits": nominal_token_bits + mean_bytes * 8,
        "actual_bytes": len(payload), "actual_bits": len(payload) * 8,
        "actual_bpp": len(payload) * 8 / pixels,
        "header_and_prefix_bytes": header_bytes,
        "packed_index_bytes": header["index_bytes"],
        "index_padding_bits": header["index_bytes"] * 8 - nominal_token_bits,
        "mean_bytes": mean_bytes, "reference_mask_bytes": reference_bytes,
        "observed_mask_bytes": observed_bytes,
        "excluded": ["shared decoder/codebook weights", "experiment manifest and external metadata"],
        "entropy_coding": False,
    }
