# expfs/experiment.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from .binning import apply_binning
from .selectors import select_features
from .models import make_classifier
from .metrics import compute_metrics, METRIC_COLS
from .constants import NOISE_REPEATS


def _make_seed(base: int, fold_idx: int, repeat_idx: int) -> int:
    return (int(base) + 9973 * int(fold_idx) + 7919 * int(repeat_idx)) % (2**32)


def run_single_configuration(
    X: pd.DataFrame,
    y: pd.Series,
    noise: float,
    binning: str,
    bins_value,
    feature_method: str,
    k_features: int,
    classifiers: Tuple[str, ...],
    k_folds: int,
    random_state: int,
    add_noise_before_scaling: bool,
    hparams: Dict,
) -> List[Dict[str, float]]:
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    sel_params = hparams.get("selectors", {})
    mdl_params = hparams.get("models", {})

    per_clf_metrics: Dict[str, Dict[str, List[float]]] = {
        clf_name: {mc: [] for mc in METRIC_COLS} for clf_name in classifiers
    }

    repeats = 1 if (noise is None or float(noise) == 0.0) else int(NOISE_REPEATS)

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr_raw, X_te_raw = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()

        rep_metrics: Dict[str, Dict[str, List[float]]] = {
            clf_name: {mc: [] for mc in METRIC_COLS} for clf_name in classifiers
        }

        for repeat_idx in range(repeats):
            rng = np.random.default_rng(_make_seed(random_state, fold_idx, repeat_idx))

            X_tr_r, X_te_r = X_tr_raw.copy(), X_te_raw.copy()

            if add_noise_before_scaling and noise and float(noise) > 0.0:
                X_tr_r = X_tr_r + rng.normal(0.0, float(noise), size=X_tr_r.shape)

            scaler = MinMaxScaler()
            X_tr = pd.DataFrame(
                scaler.fit_transform(X_tr_r),
                columns=X_tr_r.columns,
                index=X_tr_r.index,
            )
            X_te = pd.DataFrame(
                scaler.transform(X_te_r),
                columns=X_te_r.columns,
                index=X_te_r.index,
            )

            if (not add_noise_before_scaling) and noise and float(noise) > 0.0:
                X_tr = X_tr + rng.normal(0.0, float(noise), size=X_tr.shape)

            eff_bins = 0 if bins_value is None else int(bins_value)
            X_tr_b, X_te_b = apply_binning(X_tr, X_te, bins=eff_bins, method=binning)

            if feature_method == "elastic_net":
                fs_hp = sel_params.get("elastic_net", {})
            elif feature_method == "rf":
                fs_hp = sel_params.get("rf_fs", {})
            elif feature_method == "xgb":
                fs_hp = sel_params.get("xgb_fs", {})
            elif feature_method == "frfs":
                fs_hp = sel_params.get("frfs", {})
            else:
                fs_hp = {}

            X_tr_fs, X_te_fs, _ = select_features(
                X_tr_b, X_te_b, y_tr, feature_method, k_features, random_state, fs_hp
            )

            avg_type = "binary" if len(np.unique(y)) == 2 else "macro"

            for clf_name in classifiers:
                clf_hp = mdl_params.get(clf_name, {})
                clf = make_classifier(clf_name, random_state, clf_hp)
                clf.fit(X_tr_fs, y_tr)

                y_pred = clf.predict(X_te_fs)
                y_prob = clf.predict_proba(X_te_fs) if hasattr(clf, "predict_proba") else None

                m = compute_metrics(y_te, y_pred, y_prob, average=avg_type)
                for mc in METRIC_COLS:
                    rep_metrics[clf_name][mc].append(m[mc])

        for clf_name in classifiers:
            for mc in METRIC_COLS:
                mean_over_repeats = float(np.nanmean(rep_metrics[clf_name][mc])) if len(rep_metrics[clf_name][mc]) else np.nan
                per_clf_metrics[clf_name][mc].append(mean_over_repeats)

    rows: List[Dict[str, float]] = []
    for clf_name in classifiers:
        means = {f"{mc} mean": float(np.nanmean(per_clf_metrics[clf_name][mc])) for mc in METRIC_COLS}
        stds  = {f"{mc} std":  float(np.nanstd(per_clf_metrics[clf_name][mc], ddof=1)) for mc in METRIC_COLS}

        row = {
            "Classifier": clf_name,
            **means,
            **stds,
        }
        rows.append(row)

    return rows
