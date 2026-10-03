"""Empirical interventional SHAP for a black-box prediction function.

Missing coalition fields come from whole training-background rows. Small
feature sets use the exact Shapley formula; larger ones average permutation
marginal contributions. No surrogate model or single-field sensitivity is used.
"""
from __future__ import annotations

from math import comb
from typing import Callable

import numpy as np


def interventional_shap(
    predict: Callable[[np.ndarray], np.ndarray], background: np.ndarray,
    row: np.ndarray, *, permutations: int = 16, seed: int = 42,
    batch_size: int = 256,
) -> tuple[np.ndarray, float, float, str]:
    """Return field contributions, reference output, full output, and method.

Exactness refers to the finite background, not the population distribution.
Every permutation telescopes from v(empty) to v(all), so contributions add
up without a post-hoc correction, including for nonlinear interactions.
"""
    background = np.asarray(background, dtype=float)
    row = np.asarray(row, dtype=float)
    if row.ndim != 1 or background.ndim != 2 or background.shape[1] != len(row) or not len(background):
        raise ValueError("A row and a nonempty matching training background are required.")
    if not 1 <= len(row) <= 12 or permutations < 2 or batch_size < len(background):
        raise ValueError("SHAP supports 1–12 fields and a valid evaluation budget.")
    dimensions = len(row)
    full = (1 << dimensions) - 1
    rng = np.random.default_rng(seed)
    orders = []
    if dimensions <= 6:
        masks = list(range(1, full))
        method = "Exact empirical interventional SHAP"
    else:
        coalitions = set()
        for _ in range(permutations // 2):
            order = rng.permutation(dimensions)
            for permutation in (order, order[::-1]):
                orders.append(permutation)
                mask = 0
                for field in permutation[:-1]:
                    mask |= 1 << int(field)
                    coalitions.add(mask)
        masks = sorted(coalitions)
        method = f"Estimated interventional SHAP ({len(orders)} permutations)"

    def checked_predict(values):
        output = np.asarray(predict(values), dtype=float).reshape(-1)
        if len(output) != len(values) or not np.isfinite(output).all():
            raise ValueError("SHAP requires one finite model output per record.")
        return output

    reference = float(checked_predict(background).mean())
    prediction = float(checked_predict(row.reshape(1, -1))[0])
    values = {0: reference, full: prediction}
    masks_per_batch = max(1, batch_size // len(background))
    for start in range(0, len(masks), masks_per_batch):
        group = masks[start:start + masks_per_batch]
        hybrids = []
        for mask in group:
            hybrid = background.copy()
            present = [j for j in range(dimensions) if mask & (1 << j)]
            hybrid[:, present] = row[present]
            hybrids.append(hybrid)
        output = checked_predict(np.vstack(hybrids)).reshape(len(group), len(background)).mean(axis=1)
        values.update(zip(group, output))

    contributions = np.zeros(dimensions)
    if dimensions <= 6:
        for field in range(dimensions):
            bit = 1 << field
            for mask in range(full + 1):
                if mask & bit:
                    continue
                size = mask.bit_count()
                weight = 1 / (dimensions * comb(dimensions - 1, size))
                contributions[field] += weight * (values[mask | bit] - values[mask])
    else:
        for order in orders:
            mask = 0
            for field in order:
                after = mask | (1 << int(field))
                contributions[field] += values[after] - values[mask]
                mask = after
        contributions /= len(orders)
    if not np.isclose(reference + contributions.sum(), prediction, atol=1e-7, rtol=1e-7):
        raise ValueError("SHAP contributions did not reconstruct the model output.")
    return contributions, reference, prediction, method
