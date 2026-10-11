"""Holdout split that keeps whole groups (state, region, customer, site) on one side."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from glin.preprocessor import EBMTabularPreprocessor

MIN_GROUPS = 5
MAX_SPLIT_TRIES = 25


@dataclass
class GroupSplit:
    train_idx: np.ndarray
    holdout_idx: np.ndarray
    holdout_groups: int
    train_groups: int
    warnings: list[str] = field(default_factory=list)


def _normalise(values: pd.Series) -> pd.Series:
    """Text groups as the preprocessor sees them (strip, lowercase, blank
    markers as missing), so spelling variants of a group stay on one side."""
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return values
    pre = EBMTabularPreprocessor()
    cleaned = pre._clean_categorical(values)
    return cleaned.mask(cleaned == pre.missing_token)


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
    holdout row count closer to the target.

    Vectorised: each pass takes the longest run of groups that all fit
    (cumsum), then jumps to the next group that fits the new count. Same
    result as the plain loop."""
    order = rng.permutation(len(sizes))
    ordered = sizes[order]
    taken = np.zeros(len(sizes), dtype=bool)
    held = 0
    i = 0
    while i < len(ordered):
        rest = ordered[i:]
        before = held + np.cumsum(rest) - rest
        fits = np.abs(before + rest - target_rows) < np.abs(before - target_rows)
        run = len(rest) if fits.all() else int(np.argmin(fits))
        taken[i : i + run] = True
        if run == len(rest):
            break
        held = before[run]
        i += run + 1
        rest = ordered[i:]
        fits = np.abs(held + rest - target_rows) < np.abs(held - target_rows)
        if not fits.any():
            break
        i += int(np.argmax(fits))
    chosen = np.zeros(len(sizes), dtype=bool)
    chosen[order] = taken
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
    result depends only on the data, ``seed`` and ``holdout_fraction``.
    Text values are grouped after the preprocessor's normalisation."""
    n = len(y)
    normalised = _normalise(groups)
    codes, n_real = _group_codes(normalised)
    warnings = []
    n_spellings = groups[normalised.notna()].nunique()
    if n_spellings > n_real:
        warnings.append(
            f"holdout_group_column '{column}' has {n_spellings} distinct spellings "
            f"that form {n_real} groups after case and whitespace normalisation "
            "(the same values the model sees); the split and the group counts use "
            "the normalised groups"
        )
    if n_real < MIN_GROUPS:
        raise ValueError(
            f"holdout_group_column '{column}' has {n_real} distinct non-null "
            f"values; need at least {MIN_GROUPS} groups to hold some out"
        )
    sizes = np.bincount(codes)
    biggest = int(sizes.max())
    if biggest > n / 2 and sizes.argmax() == n_real:
        raise ValueError(
            f"missing values in holdout_group_column '{column}' form one group of "
            f"{biggest} of {n} rows (more than half), so a group holdout cannot be "
            "split sensibly -- fill the missing values with real groups, pick a "
            "column with fewer gaps or use a random holdout"
        )
    if biggest > n / 2:
        raise ValueError(
            f"one group in holdout_group_column '{column}' holds {biggest} of "
            f"{n} rows (more than half), so a group holdout cannot be split "
            "sensibly -- pick a finer-grained column or use a random holdout"
        )

    overall = np.bincount(y, minlength=n_classes) / n
    class_counts = np.bincount(
        codes * n_classes + y, minlength=len(sizes) * n_classes
    ).reshape(len(sizes), n_classes).astype(float)
    total_counts = class_counts.sum(axis=0)

    target_rows = n * holdout_fraction
    smallest = int(sizes.min())
    if not np.any(np.abs(sizes - target_rows) < target_rows):
        raise ValueError(
            f"every group in holdout_group_column '{column}' is larger than the "
            f"holdout target of {target_rows:.0f} rows (a group must be under "
            "twice the target to be held out); use a larger holdout_fraction "
            f"(smallest group is {smallest} rows = {smallest / n:.1%} of the data, "
            f"so use more than {smallest / (2 * n):.3f}) or a random holdout"
        )

    best_gap, best_mask = None, None
    for i in range(MAX_SPLIT_TRIES):
        mask = _pick_holdout_groups(sizes, target_rows, np.random.RandomState(seed + i))
        held = class_counts[mask].sum(axis=0)
        rows = held.sum()
        if (held > 0).sum() < 2 or (total_counts - held).min() <= 0:
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
        warnings=warnings,
    )
