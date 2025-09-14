from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PipelineConfig:
    noise_levels: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
    binning_methods: Tuple[str, ...] = ("none", "static", "quantile", "kmeans")
    feature_methods: Tuple[str, ...] = ("frfs", "elastic_net", "rf", "xgb")
    classifiers: Tuple[str, ...] = ("random_forest", "xgboost", "svm", "logistic_regression")


NOISE_REPEATS = 3

BINNING_GRID = {
    "none":     [None],
    "static":   [2, 4, 10, 50],
    "quantile": [2, 4, 8, 16],
    "kmeans":   [2, 4, 8, 16],
}
