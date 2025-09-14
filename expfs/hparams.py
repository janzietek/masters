from __future__ import annotations
import os
import yaml
from typing import Any, Dict

DEFAULT_HPARAMS: Dict[str, Any] = {
    "selectors": {
        "frfs": {
        },
        "elastic_net": {
            "l1_ratio": 0.5,
            "C": 0.5,
            "max_iter": 10000,
            "tol": 1e-3,
        },
        "rf_fs": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        },
        "xgb_fs": {
            "n_estimators": 300,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "n_jobs": -1,
        },
    },
    "models": {
        "random_forest": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        },
        "xgboost": {
            "n_estimators": 400,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "n_jobs": -1,
        },
        "svm": {
            "kernel": "linear",
            "C": 1.0,
            "probability": True,
        },
        "logistic_regression": {
            "max_iter": 1000,
            "C": 1.0,
        },
    },
}

def load_hparams(dataset_name: str, base_dir: str = "hparams") -> Dict[str, Any]:
    path = os.path.join(base_dir, f"{dataset_name}.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        merged = DEFAULT_HPARAMS.copy()
        merged_sel = merged.get("selectors", {}).copy()
        merged_sel.update(cfg.get("selectors", {}) or {})
        merged_mod = merged.get("models", {}).copy()
        merged_mod.update(cfg.get("models", {}) or {})
        merged["selectors"] = merged_sel
        merged["models"] = merged_mod
        return merged
    return DEFAULT_HPARAMS
