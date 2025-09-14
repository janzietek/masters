# expfs/selectors.py

from __future__ import annotations
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from frlearn.feature_preprocessors import FRFS  # type: ignore


def _stratified_subsample(
    X: pd.DataFrame, y: pd.Series, max_samples: int, random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Zwraca (X_sub, y_sub) z maks. max_samples próbek (stratyfikacja)."""
    n = len(X)
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return X, y
    test_size = 1.0 - (max_samples / float(n))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_keep, _ = next(splitter.split(X, y))
    return X.iloc[idx_keep], y.iloc[idx_keep]


def _as_dtype(a: np.ndarray, dtype: str | np.dtype | None) -> np.ndarray:
    if dtype is None:
        return a
    dt = np.dtype(dtype)
    if a.dtype != dt:
        return a.astype(dt, copy=False)
    return a


def _frfs_select(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    k_features: int,
    random_state: int,
    params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    FRFS z kontrolą kosztu:
      - strat. subsampling train (max_samples),
      - wymuszenie dtype (domyślnie float32),
      - bez fallbacku.
    """
    max_samples = int(params.get("max_samples", 0) or 0)
    out_dtype = params.get("dtype", "float32")

    X_used, y_used = _stratified_subsample(X_train, y_train, max_samples, random_state)

    X_used_np = _as_dtype(X_used.values, out_dtype)
    X_tr_np = _as_dtype(X_train.values, out_dtype)
    X_te_np = _as_dtype(X_test.values, out_dtype)

    pre = FRFS(n_features=int(k_features))
    frfs_model = pre(X_used_np, y_used.values if hasattr(y_used, "values") else np.asarray(y_used))

    mask = getattr(frfs_model, "selection", None)
    if mask is None:
        raise RuntimeError("FRFS model did not expose 'selection' mask.")
    mask = np.asarray(mask, dtype=bool)
    selected_cols = list(X_used.columns[mask])

    X_tr_sel = pd.DataFrame(frfs_model(X_tr_np), columns=selected_cols, index=X_train.index)
    X_te_sel = pd.DataFrame(frfs_model(X_te_np), columns=selected_cols, index=X_test.index)
    return X_tr_sel, X_te_sel, selected_cols

def _elastic_net_select_with_retries(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    k_features: int,
    random_state: int,
    params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    LR(Elastic Net) z automatycznym „samoleczeniem” konwergencji.
    Przy ConvergenceWarning: max_iter *= 2; C *= 0.5; tol *= 2 (do 3 prób).
    """
    l1_ratio = float(params.get("l1_ratio", 0.5))
    C = float(params.get("C", 0.5))
    max_iter = int(params.get("max_iter", 20000))
    tol = float(params.get("tol", 1e-2))

    tries = 3
    for _ in range(tries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                lr = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=l1_ratio,
                    C=C,
                    max_iter=max_iter,
                    tol=tol,
                    random_state=random_state
                )
                lr.fit(X_train, y_train)
        except ConvergenceWarning:
            max_iter *= 2
            C *= 0.5
            tol *= 2
            continue
        else:
            importance = np.sum(np.abs(lr.coef_), axis=0)
            order = np.argsort(importance)[::-1]
            selected = list(X_train.columns[order][:k_features])
            return X_train[selected], X_test[selected], selected

    lr = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    lr.fit(X_train, y_train)
    importance = np.sum(np.abs(lr.coef_), axis=0)
    order = np.argsort(importance)[::-1]
    selected = list(X_train.columns[order][:k_features])
    return X_train[selected], X_test[selected], selected


def select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    method: str,
    k_features: int,
    random_state: int,
    params: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Zwraca (X_train_selected, X_test_selected, selected_feature_names).
    """
    params = params or {}

    if method == "frfs":
        return _frfs_select(X_train, X_test, y_train, k_features, random_state, params)

    if method == "elastic_net":
        return _elastic_net_select_with_retries(X_train, X_test, y_train, k_features, random_state, params)

    if method == "rf":
        hp = params if params else {}
        model = RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 200)),
            max_depth=hp.get("max_depth", None),
            min_samples_leaf=int(hp.get("min_samples_leaf", 1)),
            n_jobs=int(hp.get("n_jobs", -1)),
            random_state=random_state
        )
        model.fit(X_train, y_train)
        ranked = X_train.columns[np.argsort(model.feature_importances_)[::-1]][:k_features]
        return X_train[ranked], X_test[ranked], list(ranked)

    if method == "xgb":
        hp = params if params else {}
        model = XGBClassifier(
            eval_metric=hp.get("eval_metric", "logloss"),
            random_state=random_state,
            n_estimators=int(hp.get("n_estimators", 300)),
            max_depth=int(hp.get("max_depth", 6)),
            subsample=float(hp.get("subsample", 0.8)),
            colsample_bytree=float(hp.get("colsample_bytree", 0.8)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            n_jobs=int(hp.get("n_jobs", -1)),
        )
        model.fit(X_train, y_train)
        ranked = X_train.columns[np.argsort(model.feature_importances_)[::-1]][:k_features]
        return X_train[ranked], X_test[ranked], list(ranked)

    raise ValueError(f"Unexpected feature selection method: {method}")
