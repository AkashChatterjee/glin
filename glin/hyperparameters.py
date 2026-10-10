"""The EBM training hyperparameters glin lets a caller (human or agent) tune.

How an EBM learns, in one paragraph, because every knob below maps onto it:
an Explainable Boosting Machine is a generalized additive model,
``logit(P(y)) = intercept + f1(x1) + f2(x2) + ... + f_ij(xi, xj)``, where
each shape function ``f`` is a lookup table over the feature's bins. It is
fit by *cyclic gradient boosting*: tiny trees (``max_leaves=2`` is a
single-split stump) that each look at exactly ONE feature are added
round-robin across features at a very small ``learning_rate``, so no
feature can hog the signal and the order of features doesn't matter. This
is repeated across ``outer_bags`` bootstrap-style resamples and the
per-bag tables are averaged, which smooths the curves. A pairwise
interaction stage (``interactions``) then boosts the best-scoring feature
pairs on the residuals. Because every term is a table lookup, a prediction
decomposes exactly into per-feature contributions -- that is the
glassbox property glin's ``predict`` tool relies on.

Consequently most of the useful tuning is about capacity vs. smoothness:
how finely each feature is binned (``max_bins``), how long boosting runs
before early stopping (``learning_rate``, ``max_rounds``,
``early_stopping_rounds``), how much each tree may fit noise
(``min_samples_leaf``, ``reg_*``), and how many pairwise terms are added
(``interactions``).

The registry below is the single source of truth: engine.train_model
validates against it, the MCP ``list_hyperparameters`` tool and the
``tune_model_hyperparameters`` prompt render it, and defaults are read live
from the installed interpret-core (except the few glin deliberately
overrides) so documentation can't drift from behavior.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Optional

from interpret.glassbox import ExplainableBoostingClassifier

# Where glin's historical defaults differ from interpret-core's. Everything
# not listed here uses the library's own default.
GLIN_DEFAULT_OVERRIDES: dict[str, Any] = {
    "max_bins": 256,
    "interactions": 10,
}


@dataclass(frozen=True)
class Hyperparameter:
    name: str
    kind: type  # int or float
    minimum: float
    maximum: float
    guidance: str

    @property
    def default(self) -> Any:
        if self.name in GLIN_DEFAULT_OVERRIDES:
            return GLIN_DEFAULT_OVERRIDES[self.name]
        param = inspect.signature(ExplainableBoostingClassifier.__init__).parameters[self.name]
        return param.default

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.kind.__name__,
            "default": self.default,
            "min": self.minimum,
            "max": self.maximum,
            "guidance": self.guidance,
        }


_ALL = [
    Hyperparameter(
        "max_bins", int, 2, 1024,
        "Max bins per continuous feature for main effects. More bins = finer, "
        "more flexible shape functions (can capture sharp thresholds, but fits "
        "noise on small data); fewer = smoother and faster. Categoricals are "
        "unaffected. Try 32-128 for small data (<5k rows), 256-1024 for large.",
    ),
    Hyperparameter(
        "max_interaction_bins", int, 2, 256,
        "Same as max_bins but for the 2-D tables of interaction terms. These "
        "tables have bins^2 cells, so keep this small (8-32) unless data is large.",
    ),
    Hyperparameter(
        "interactions", int, 0, 200,
        "Number of pairwise interaction terms to add on top of the main "
        "effects (0 = purely additive model, the simplest and most readable). "
        "Each one is chosen by gain on the residuals. Raise it if the "
        "holdout metric keeps improving; lower it (or 0) if holdout log-loss "
        "is worse than train, or you want maximum interpretability. Pairs only "
        "(never 3-way), capped at the number of feature pairs available.",
    ),
    Hyperparameter(
        "learning_rate", float, 0.0001, 0.5,
        "Step size of each boosting round. Smaller = slower but finer, more "
        "accurate fit (needs a larger max_rounds budget); larger = faster but "
        "coarser. The 0.015 default is already conservative; 0.03-0.1 is a "
        "reasonable way to speed up a trial run, 0.005-0.01 for a final squeeze.",
    ),
    Hyperparameter(
        "max_rounds", int, 1, 100000,
        "Hard cap on boosting rounds. Normally early stopping ends training "
        "first, so this mostly bounds worst-case runtime. Lower it to force a "
        "faster, less-fit model.",
    ),
    Hyperparameter(
        "early_stopping_rounds", int, 0, 1000,
        "Stop a bag after this many rounds without validation improvement. "
        "Smaller = less overfitting and faster, but may stop on a plateau "
        "before the real optimum; larger = trains longer. 0 disables early "
        "stopping (runs exactly max_rounds -- usually a bad idea).",
    ),
    Hyperparameter(
        "validation_size", float, 0.01, 0.5,
        "Fraction of the training rows each bag holds out INTERNALLY to decide "
        "when to stop boosting. Not the same as holdout_fraction, which is a "
        "separate split used only to score the model for you. 0.15 is a good default.",
    ),
    Hyperparameter(
        "outer_bags", int, 1, 50,
        "Number of resampled models averaged into the final one. More bags = "
        "smoother, lower-variance shape functions (and error bars) at roughly "
        "linear cost in training time. Reduce to 4-8 for fast exploratory trials.",
    ),
    Hyperparameter(
        "inner_bags", int, 0, 50,
        "Extra bootstrap bagging inside each boosting step. 0 = off. Adds "
        "smoothing/variance reduction on noisy data at a real runtime cost.",
    ),
    Hyperparameter(
        "min_samples_leaf", int, 1, 1000,
        "Minimum rows a tree leaf must cover. THE main overfitting control for "
        "small/noisy data: raise it (20-100) to stop the model from carving "
        "out tiny bins from a handful of rows. Lower values let it fit sparse regions.",
    ),
    Hyperparameter(
        "max_leaves", int, 2, 10,
        "Leaves per single-feature tree. 2 (a stump) keeps each round's update "
        "a single step function; 3-5 lets one round express more shape at the "
        "cost of smoothness. Rarely the first thing worth changing.",
    ),
    Hyperparameter(
        "reg_alpha", float, 0.0, 100.0,
        "L1 regularization on leaf values. Pushes small contributions to exactly "
        "zero -> sparser, more readable shape functions. 0 = off.",
    ),
    Hyperparameter(
        "reg_lambda", float, 0.0, 1000.0,
        "L2 regularization on leaf values. Shrinks every contribution toward 0; "
        "helps with noisy features and rare categories. 0 = off; try 1-50.",
    ),
    Hyperparameter(
        "smoothing_rounds", int, 0, 2000,
        "Initial rounds of heavily-regularized boosting that establish the broad "
        "shape of each curve before fine detail is fit. More = smoother, "
        "less jagged curves; 0 = off.",
    ),
    Hyperparameter(
        "greedy_ratio", float, 0.0, 50.0,
        "How many greedy (best-feature-first) steps are taken per cyclic "
        "(round-robin) step. 0 = purely cyclic, the classic EBM behavior that "
        "treats features most evenly; higher converges faster on the strongest features.",
    ),
    Hyperparameter(
        "random_state", int, 0, 2**31 - 1,
        "Seed for bagging. A fixed seed makes trials reproducible. To tell "
        "whether a metric difference between two configs is real, re-run the "
        "same config with 2-3 different seeds and compare the spread.",
    ),
]

HYPERPARAMETERS: dict[str, Hyperparameter] = {hp.name: hp for hp in _ALL}


def default_hyperparameters() -> dict[str, Any]:
    return {name: hp.default for name, hp in HYPERPARAMETERS.items()}


def resolve_hyperparameters(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Defaults merged with ``overrides``, every override type- and range-checked.
    Raises ValueError listing every problem at once, so an agent can fix its
    whole request in one retry instead of one error at a time."""
    resolved = default_hyperparameters()
    problems: list[str] = []

    for name, value in (overrides or {}).items():
        hp = HYPERPARAMETERS.get(name)
        if hp is None:
            problems.append(
                f"unknown hyperparameter '{name}' (valid: {', '.join(HYPERPARAMETERS)})"
            )
            continue

        # bool is an int subclass; True/False are never a meaningful value here.
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not is_number or not math.isfinite(value) or (hp.kind is int and value != int(value)):
            problems.append(f"{name} must be {'an integer' if hp.kind is int else 'a number'}, got {value!r}")
            continue
        if not (hp.minimum <= value <= hp.maximum):
            problems.append(f"{name}={value} is outside the allowed range [{hp.minimum}, {hp.maximum}]")
            continue

        resolved[name] = hp.kind(value)

    if problems:
        raise ValueError("invalid hyperparameters: " + "; ".join(problems))
    return resolved
