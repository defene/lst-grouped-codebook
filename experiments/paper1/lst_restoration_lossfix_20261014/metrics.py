"""Restoration metrics in degrees Celsius; no model execution or I/O at import.

Rows are per (sample_id, view) within one condition. Bootstrap resamples tiles,
retaining all records and noise views in each sampled tile. It does not interpret
correlated pixels or multiple views of one record as independent observations.
"""

from __future__ import annotations

from collections import Counter
import operator
from typing import Any, Iterable

import numpy as np


def _masked_error(error: np.ndarray, mask: np.ndarray, prefix: str = "") -> dict:
    values = error[mask].astype(np.float64, copy=False)
    count = int(values.size)
    squared_sum = float(np.dot(values, values)) if count else 0.0
    absolute_sum = float(np.abs(values).sum()) if count else 0.0
    signed_sum = float(values.sum()) if count else 0.0
    return {
        prefix + "squared_error_sum": squared_sum,
        prefix + "absolute_error_sum": absolute_sum,
        prefix + "signed_error_sum": signed_sum,
        prefix + "count": count,
        prefix + "mse": squared_sum / count if count else None,
        prefix + "rmse": float(np.sqrt(squared_sum / count)) if count else None,
        prefix + "mae": absolute_sum / count if count else None,
        prefix + "bias": signed_sum / count if count else None,
    }


def index_statistics(indices: np.ndarray, codebook_size: int) -> dict:
    """Joint integer-index statistics, not a sum of FSQ marginal entropies.

    ``neighbor_equal_fraction`` is defined only for a single 2-D index grid.
    Entropy is empirical token usage and is not a measured file bitrate.
    """
    if isinstance(codebook_size, bool) or int(codebook_size) != codebook_size or codebook_size < 2:
        raise ValueError("codebook_size must be an integer >= 2")
    indices = np.asarray(indices)
    if indices.dtype.kind not in "iu" or indices.size == 0:
        raise ValueError("indices must be a nonempty integer array")
    if int(indices.min()) < 0 or int(indices.max()) >= codebook_size:
        raise ValueError("index outside declared codebook")
    unique, counts = np.unique(indices, return_counts=True)
    probability = counts.astype(np.float64) / int(indices.size)
    entropy = float(-(probability * np.log2(probability)).sum())
    neighbor_count = 0
    neighbor_equal = 0
    if indices.ndim == 2:
        for first, second in ((indices[1:, :], indices[:-1, :]), (indices[:, 1:], indices[:, :-1])):
            neighbor_count += int(first.size)
            neighbor_equal += int(np.count_nonzero(first == second))
    return {
        "token_count": int(indices.size),
        "codebook_size": int(codebook_size),
        "used_codes": int(unique.size),
        "utilization": float(unique.size / codebook_size),
        "entropy_bits": entropy,
        "perplexity": float(2.0**entropy),
        "neighbor_pair_count": neighbor_count,
        "neighbor_equal_count": neighbor_equal,
        "neighbor_equal_fraction": neighbor_equal / neighbor_count if neighbor_count else None,
        "index_counts": {str(int(index)): int(count) for index, count in zip(unique, counts)},
    }


def restoration_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    reference_valid: np.ndarray,
    observed_valid: np.ndarray,
    sample_ids: list[str],
    tile_ids: list[str],
    affected_mask: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    codebook_size: int | None = None,
) -> list[dict]:
    """Produce per-image metrics from [B,1,H,W] Celsius predictions and targets.

    The evaluator must add ``view`` and ``condition`` before combining batches.
    Invalid reference pixels are excluded everywhere; missing means reference
    valid and observation invalid. ``affected_mask`` is evaluator-only ground
    truth corruption support, never implied to be supplied to the model.
    Empty regions have count/sums zero and scalar errors None, not perfect error.
    Nonfinite values on valid reference pixels raise instead of being discarded.
    Spatial error removes each image's mean prediction error over valid pixels.
    Gradient RMSE uses horizontal/vertical first differences in Celsius/pixel.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    reference_valid = np.asarray(reference_valid, dtype=bool)
    observed_valid = np.asarray(observed_valid, dtype=bool)
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("prediction must have shape [B,1,H,W]")
    if any(x.shape != prediction.shape for x in (target, reference_valid, observed_valid)):
        raise ValueError("target and masks must exactly match prediction shape")
    if len(sample_ids) != len(prediction) or len(tile_ids) != len(prediction):
        raise ValueError("one sample_id and tile_id are required per batch element")
    if np.any(observed_valid & ~reference_valid):
        raise ValueError("observed_valid must be a subset of reference_valid")
    if not np.all(np.isfinite(prediction[reference_valid])) or not np.all(np.isfinite(target[reference_valid])):
        raise ValueError("nonfinite prediction or target on a valid reference pixel")
    if affected_mask is not None:
        affected_mask = np.asarray(affected_mask, dtype=bool)
        if affected_mask.shape != prediction.shape:
            raise ValueError("affected_mask must exactly match prediction shape")
    if indices is not None:
        indices = np.asarray(indices)
        if indices.ndim < 2 or indices.shape[0] != len(prediction) or codebook_size is None:
            raise ValueError("batched indices require matching B and a codebook_size")
    elif codebook_size is not None:
        raise ValueError("codebook_size without indices is ambiguous")
    rows = []
    for i, (sample_id, tile_id) in enumerate(zip(sample_ids, tile_ids)):
        valid = reference_valid[i, 0]
        observed = observed_valid[i, 0]
        # Invalid pixels may contain NaN; replace them before edge arithmetic.
        error = np.zeros(valid.shape, dtype=np.float64)
        error[valid] = prediction[i, 0][valid].astype(np.float64) - target[i, 0][valid].astype(np.float64)
        row: dict[str, Any] = {"sample_id": str(sample_id), "tile_id": str(tile_id)}
        row.update(_masked_error(error, valid))
        row.update(_masked_error(error, valid & ~observed, "missing_region_"))
        row.update(_masked_error(error, observed, "observed_region_"))
        if affected_mask is not None:
            row.update(_masked_error(error, valid & affected_mask[i, 0], "affected_region_"))
        else:
            row.update(_masked_error(error, np.zeros_like(valid), "affected_region_"))
        row["affected_region_defined"] = affected_mask is not None
        if row["count"]:
            spatial_values = error[valid] - row["bias"]
            spatial_squared_sum = float(np.dot(spatial_values, spatial_values))
            row["target_mean_celsius"] = float(target[i, 0][valid].astype(np.float64).mean())
            row["prediction_mean_celsius"] = float(prediction[i, 0][valid].astype(np.float64).mean())
        else:
            spatial_squared_sum = 0.0
            row["target_mean_celsius"] = None
            row["prediction_mean_celsius"] = None
        row["spatial_squared_error_sum"] = spatial_squared_sum
        row["spatial_mse"] = spatial_squared_sum / row["count"] if row["count"] else None
        row["spatial_rmse"] = float(np.sqrt(row["spatial_mse"])) if row["count"] else None
        gradient_sum = 0.0
        gradient_count = 0
        for difference, edge_valid in (
            (error[1:, :] - error[:-1, :], valid[1:, :] & valid[:-1, :]),
            (error[:, 1:] - error[:, :-1], valid[:, 1:] & valid[:, :-1]),
        ):
            values = difference[edge_valid]
            gradient_sum += float(np.dot(values, values))
            gradient_count += int(values.size)
        row["gradient_squared_error_sum"] = gradient_sum
        row["gradient_count"] = gradient_count
        row["gradient_mse"] = gradient_sum / gradient_count if gradient_count else None
        row["gradient_rmse"] = float(np.sqrt(gradient_sum / gradient_count)) if gradient_count else None
        if indices is not None:
            row["token_statistics"] = index_statistics(indices[i], int(codebook_size))
        rows.append(row)
    return rows


def _rows_by_key(rows: Iterable[dict]) -> dict[tuple[str, Any], dict]:
    indexed = {}
    conditions = set()
    sample_tiles = {}
    for row in rows:
        key = (str(row["sample_id"]), row.get("view", 0))
        if key in indexed:
            raise ValueError(f"duplicate (sample_id, view): {key!r}; summarize one condition at a time")
        if "condition" in row:
            conditions.add(str(row["condition"]))
        tile = str(row["tile_id"])
        if key[0] in sample_tiles and sample_tiles[key[0]] != tile:
            raise ValueError(f"one sample cannot change tile between views: {key[0]!r}")
        sample_tiles[key[0]] = tile
        indexed[key] = row
    if len(conditions) > 1:
        raise ValueError("cannot pool distinct conditions into a single result")
    return indexed


def _cluster_bootstrap(
    values: np.ndarray,
    tiles: list[str],
    seed: int,
    replicates: int,
) -> np.ndarray:
    """Return bootstrap means for columns, with row weighting after tile draws."""
    if isinstance(replicates, bool):
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    try:
        replicates = operator.index(replicates)
    except TypeError as error:
        raise ValueError("bootstrap_replicates must be a nonnegative integer") from error
    if replicates < 0:
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(tiles) or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap requires finite aligned observations")
    unique = sorted(set(tiles))
    if len(unique) < 2 or not len(values) or replicates == 0:
        return np.empty((0, values.shape[1]), dtype=np.float64)
    lookup = {tile: index for index, tile in enumerate(unique)}
    ids = np.asarray([lookup[tile] for tile in tiles], dtype=np.int64)
    sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    counts = np.bincount(ids, minlength=len(unique)).astype(np.float64)
    np.add.at(sums, ids, values)
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, values.shape[1]), dtype=np.float64)
    # Bound temporary memory when a future dataset contains many more tiles.
    for start in range(0, replicates, 256):
        take = min(256, replicates - start)
        weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=take)
        result[start:start + take] = weights @ sums / (weights @ counts)[:, None]
    return result


def _ci(values: np.ndarray) -> list[float] | None:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])] if len(values) else None


def _region_summary(rows: list[dict], prefix: str = "") -> dict:
    count = sum(int(row[prefix + "count"]) for row in rows)
    squared = sum(float(row[prefix + "squared_error_sum"]) for row in rows)
    absolute = sum(float(row[prefix + "absolute_error_sum"]) for row in rows)
    signed = sum(float(row[prefix + "signed_error_sum"]) for row in rows)
    return {
        "count": count,
        "squared_error_sum": squared,
        "pixel_weighted_mse": squared / count if count else None,
        "pixel_weighted_rmse": float(np.sqrt(squared / count)) if count else None,
        "pixel_weighted_mae": absolute / count if count else None,
        "pixel_weighted_bias": signed / count if count else None,
    }


def summarize_rows(
    rows: Iterable[dict], *, bootstrap_seed: int = 1701, bootstrap_replicates: int = 2000,
) -> dict:
    """Aggregate one condition without changing image or pixel weighting silently.

    ``macro_mse`` averages image-view MSE; ``sqrt_macro_mse`` takes its square
    root. ``mean_image_rmse`` averages image-view RMSE, a different statistic.
    Invalid-only images remain in coverage counts but not metric denominators.
    A CI with fewer than two valid tiles is undefined and is returned as None.
    """
    rows = list(_rows_by_key(rows).values())
    usable = [row for row in rows if int(row["count"]) > 0]
    means = np.asarray([row["mse"] for row in usable], dtype=np.float64)
    tiles = [str(row["tile_id"]) for row in usable]
    boot = _cluster_bootstrap(means, tiles, bootstrap_seed, bootstrap_replicates)
    total = _region_summary(rows)
    count = total["count"]
    gradient_count = sum(int(row["gradient_count"]) for row in rows)
    gradient_squared = sum(float(row["gradient_squared_error_sum"]) for row in rows)
    spatial_squared = sum(float(row["spatial_squared_error_sum"]) for row in rows)
    result: dict[str, Any] = {
        "unit": "celsius",
        "gradient_unit": "celsius_per_pixel",
        "row_count": len(rows),
        "unique_records": len({str(row["sample_id"]) for row in rows}),
        "tile_count": len({str(row["tile_id"]) for row in rows}),
        "metric_row_count": len(usable),
        "invalid_only_row_count": len(rows) - len(usable),
        **total,
        "macro_mse": float(means.mean()) if len(means) else None,
        "sqrt_macro_mse": float(np.sqrt(means.mean())) if len(means) else None,
        "mean_image_rmse": float(np.sqrt(means).mean()) if len(means) else None,
        "mean_temperature_rmse": float(np.sqrt(np.mean([row["bias"] ** 2 for row in usable]))) if usable else None,
        "spatial_rmse": float(np.sqrt(spatial_squared / count)) if count else None,
        "spatial_squared_error_sum": spatial_squared,
        "gradient_rmse": float(np.sqrt(gradient_squared / gradient_count)) if gradient_count else None,
        "gradient_squared_error_sum": gradient_squared,
        "gradient_count": gradient_count,
        "missing_region": _region_summary(rows, "missing_region_"),
        "observed_region": _region_summary(rows, "observed_region_"),
        "affected_region": _region_summary(rows, "affected_region_"),
        "affected_region_defined_for_all_rows": bool(rows) and all(row["affected_region_defined"] for row in rows),
        "tile_cluster_bootstrap": {
            "seed": int(bootstrap_seed), "replicates": int(bootstrap_replicates),
            "confidence": 0.95, "method": "percentile; sample tiles with replacement and retain all rows/views",
            "valid_tile_count": len(set(tiles)),
            "macro_mse_ci": _ci(boot[:, 0]),
            "sqrt_macro_mse_ci": _ci(np.sqrt(boot[:, 0])),
        },
    }
    stats = [row["token_statistics"] for row in rows if "token_statistics" in row]
    if stats:
        if len(stats) != len(rows) or len({item["codebook_size"] for item in stats}) != 1:
            raise ValueError("token statistics must cover all rows with one declared codebook size")
        histogram: Counter = Counter()
        for item in stats:
            histogram.update(item["index_counts"])
        token_count = sum(histogram.values())
        probabilities = np.asarray(list(histogram.values()), dtype=np.float64) / token_count
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        neighbor_count = sum(item["neighbor_pair_count"] for item in stats)
        result["token_statistics"] = {
            "codebook_size": stats[0]["codebook_size"], "token_count": token_count,
            "used_codes": len(histogram), "utilization": len(histogram) / stats[0]["codebook_size"],
            "pooled_entropy_bits": entropy, "pooled_perplexity": float(2.0**entropy),
            "mean_within_image_entropy_bits": float(np.mean([item["entropy_bits"] for item in stats])),
            "mean_within_image_perplexity": float(np.mean([item["perplexity"] for item in stats])),
            "neighbor_equal_fraction": sum(item["neighbor_equal_count"] for item in stats) / neighbor_count if neighbor_count else None,
            "index_counts": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
            "warning": "usage entropy is not measured compressed bitrate; continuous KL-VAE has no codebook perplexity",
        }
    return result


def paired_comparison(
    rows_a: Iterable[dict], rows_b: Iterable[dict], *,
    bootstrap_seed: int = 1701, bootstrap_replicates: int = 2000,
) -> dict:
    """Strict paired B minus A; negative means B has lower reconstruction error.

    Identical record/view coverage, tile IDs, reference and region counts are
    mandatory. Evaluators must separately verify physical input/target/mask
    digests; matching metric coverage alone cannot establish equal inputs.
    """
    a, b = _rows_by_key(rows_a), _rows_by_key(rows_b)
    if set(a) != set(b):
        raise ValueError("paired comparison requires identical (sample_id, view) coverage")
    left, right, tiles = [], [], []
    for key in sorted(a, key=lambda value: (value[0], str(value[1]))):
        ra, rb = a[key], b[key]
        if str(ra["tile_id"]) != str(rb["tile_id"]):
            raise ValueError(f"paired tile mismatch: {key!r}")
        if ra.get("condition") != rb.get("condition"):
            raise ValueError(f"paired condition mismatch: {key!r}")
        if ra.get("affected_region_defined") != rb.get("affected_region_defined"):
            raise ValueError(f"paired affected-region definition mismatch: {key!r}")
        for name in ("count", "gradient_count", "missing_region_count", "observed_region_count", "affected_region_count"):
            if int(ra[name]) != int(rb[name]):
                raise ValueError(f"paired mask/region coverage mismatch: {key!r}, {name}")
        if int(ra["count"]) > 0:
            left.append(float(ra["mse"]))
            right.append(float(rb["mse"]))
            tiles.append(str(ra["tile_id"]))
    values = np.column_stack((left, right)) if left else np.empty((0, 2), dtype=np.float64)
    boot = _cluster_bootstrap(values, tiles, bootstrap_seed, bootstrap_replicates)
    delta = values[:, 1] - values[:, 0]
    return {
        "direction": "B minus A; negative favors B", "paired_row_count": len(a),
        "metric_pair_count": len(values), "valid_tile_count": len(set(tiles)),
        "delta_macro_mse": float(delta.mean()) if len(delta) else None,
        "delta_sqrt_macro_mse": float(np.sqrt(values[:, 1].mean()) - np.sqrt(values[:, 0].mean())) if len(values) else None,
        "fraction_image_views_improved": float(np.mean(delta < 0)) if len(delta) else None,
        "fraction_image_views_tied": float(np.mean(delta == 0)) if len(delta) else None,
        "tile_cluster_bootstrap": {
            "seed": int(bootstrap_seed), "replicates": int(bootstrap_replicates), "confidence": 0.95,
            "delta_macro_mse_ci": _ci(boot[:, 1] - boot[:, 0]),
            "delta_sqrt_macro_mse_ci": _ci(np.sqrt(boot[:, 1]) - np.sqrt(boot[:, 0])),
        },
        "input_equivalence": "must be checked using evaluator physical target/observation/mask digests",
    }
