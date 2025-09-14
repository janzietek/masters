# expfs/binning.py

from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _unique_count(a: pd.Series) -> int:
    return int(a.dropna().nunique())


def _static_edges_from_train(col_train: pd.Series, k: int, eps: float) -> np.ndarray:
    x = col_train.dropna().to_numpy()
    if x.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= eps:
        return np.array([lo, lo + eps], dtype=float)
    return np.linspace(lo, hi, num=k + 1, dtype=float)


def _digitize_with_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, x, side="right") - 1
    bins = np.clip(bins, 0, len(edges) - 2)
    return bins


def _quantile_edges_from_train(col_train: pd.Series, k: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, num=k + 1)
    vals = np.quantile(col_train.dropna().to_numpy(), qs, method="linear")
    edges = np.unique(vals.astype(float))
    if edges.size < 2:
        v = float(edges[0]) if edges.size == 1 else 0.0
        edges = np.array([v, v + 1e-12], dtype=float)
    return edges


def _kmeans_bins(
    col_train: pd.Series, col_test: pd.Series, k: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    uniq = _unique_count(col_train)
    k_eff = max(1, min(k, uniq))
    if k_eff == 1:
        return (
            np.zeros(col_train.shape[0], dtype=np.int32),
            np.zeros(col_test.shape[0], dtype=np.int32),
        )

    m = float(col_train.mean()) if col_train.notna().any() else 0.0
    xtr = col_train.fillna(m).to_numpy(dtype=float).reshape(-1, 1)
    xte = col_test.fillna(m).to_numpy(dtype=float).reshape(-1, 1)

    km = KMeans(n_clusters=k_eff, n_init=10, random_state=random_state)
    km.fit(xtr)
    return km.predict(xtr).astype(np.int32), km.predict(xte).astype(np.int32)


def apply_binning(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    bins: int,
    method: str,
    *,
    random_state: int = 42,
    eps: float = 1e-12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if method == "none":
        return X_train.copy(), X_test.copy()

    if method not in {"static", "quantile", "kmeans"}:
        raise ValueError(f"Invalid binning method: {method!r}")

    Xtr_cols: Dict[str, np.ndarray] = {}
    Xte_cols: Dict[str, np.ndarray] = {}

    for col in X_train.columns:
        s_tr = X_train[col]
        s_te = X_test[col]
        nuniq = _unique_count(s_tr)
        k_eff = max(1, min(int(bins), nuniq))

        if method == "static":
            if k_eff == 1:
                tr_bins = np.zeros(s_tr.shape[0], dtype=np.int32)
                te_bins = np.zeros(s_te.shape[0], dtype=np.int32)
            else:
                edges = _static_edges_from_train(s_tr, k_eff, eps)
                tr_bins = _digitize_with_edges(s_tr.to_numpy(dtype=float), edges).astype(np.int32)
                te_bins = _digitize_with_edges(s_te.to_numpy(dtype=float), edges).astype(np.int32)

        elif method == "quantile":
            if k_eff == 1:
                tr_bins = np.zeros(s_tr.shape[0], dtype=np.int32)
                te_bins = np.zeros(s_te.shape[0], dtype=np.int32)
            else:
                edges = _quantile_edges_from_train(s_tr, k_eff)
                tr_bins = _digitize_with_edges(s_tr.to_numpy(dtype=float), edges).astype(np.int32)
                te_bins = _digitize_with_edges(s_te.to_numpy(dtype=float), edges).astype(np.int32)

        else:
            tr_bins, te_bins = _kmeans_bins(s_tr, s_te, k_eff, random_state)

        Xtr_cols[col] = tr_bins
        Xte_cols[col] = te_bins

    Xtr_b = pd.DataFrame(Xtr_cols, index=X_train.index).astype("int32", copy=False)
    Xte_b = pd.DataFrame(Xte_cols, index=X_test.index).astype("int32", copy=False)
    return Xtr_b, Xte_b
