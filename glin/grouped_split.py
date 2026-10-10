"""Holdout split that keeps whole groups (state, region, customer, site) on one side."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_GROUPS = 5
MAX_SPLIT_TRIES = 25


@dataclass
class GroupSplit:
    train_idx: np.ndarray
    holdout_idx: np.ndarray
    holdout_groups: int
    train_groups: int


def _group_codes(values: pd.Series) -> tuple[np.ndarray, int]:
    """Integer group id per row; missing values share one extra group.
    Returns (codes, number of distinct non-null groups)."""
    codes, uniques = pd.factorize(values)
    n_real = len(uniques)
    codes = np.where(codes < 0, n_real, codes)
    return codes, n_real


def _pick_holdout_groups(
    sizes: np.ndarray, target_rows: float, rng: np.random.RandomState
) -> np.ndarray:
    """Walk the groups in random order, adding each one that brings the
    holdout row count closer to the target."""
    chosen = np.zeros(len(sizes), dtype=bool)
    held = 0
    for g in rng.permutation(len(sizes)):
        if abs(held + sizes[g] - target_rows) < abs(held - target_rows):
            chosen[g] = True
            held += sizes[g]
    return chosen


def split_by_group(
    groups: pd.Series,
    y: np.ndarray,
    n_classes: int,
    holdout_fraction: float,
    seed: int,
    column: str,
) -> GroupSplit:
    """Hold out whole groups until about ``holdout_fraction`` of the rows are
    held out. Among ``MAX_SPLIT_TRIES`` seeded group orderings, keeps the one
    whose holdout class shares are closest to the overall shares, so the
    result depends only on the data, ``seed`` and ``holdout_fraction``."""
    n = len(y)
    codes, n_real = _group_codes(groups)
    if n_real < MIN_GROUPS:
        raise ValueError(
            f"holdout_group_column '{column}' has {n_real} distinct non-null "
            f"values; need at least {MIN_GROUPS} groups to hold some out"
        )
    sizes = np.bincount(codes)
    biggest = int(sizes.max())
    if biggest > n / 2:
        raise ValueError(
            f"one group in holdout_group_column '{column}' holds {biggest} of "
            f"{n} rows (more than half), so a group holdout cannot be split "
            "sensibly -- pick a finer-grained column or use a random holdout"
        )

    overall = np.bincount(y, minlength=n_classes) / n
    class_counts = np.zeros((len(sizes), n_classes))
    np.add.at(class_counts, (codes, y), 1)
    total_counts = class_counts.sum(axis=0)

    best_gap, best_mask = None, None
    for i in range(MAX_SPLIT_TRIES):
        mask = _pick_holdout_groups(sizes, n * holdout_fraction, np.random.RandomState(seed + i))
        held = class_counts[mask].sum(axis=0)
        rows = held.sum()
        if rows == 0 or (held > 0).sum() < 2 or (total_counts - held).min() <= 0:
            continue
        gap = float(np.abs(held / rows - overall).max())
        if best_gap is None or gap < best_gap:
            best_gap, best_mask = gap, mask
    if best_mask is None:
        raise ValueError(
            f"could not find a holdout of whole '{column}' groups that contains "
            "at least 2 classes while leaving every class in training -- the "
            "classes are concentrated in too few groups; use a random holdout"
        )

    in_holdout = best_mask[codes]
    return GroupSplit(
        train_idx=np.flatnonzero(~in_holdout),
        holdout_idx=np.flatnonzero(in_holdout),
        holdout_groups=int(best_mask.sum()),
        train_groups=int((~best_mask & (sizes > 0)).sum()),
    )
