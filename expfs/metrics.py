import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def compute_metrics(y_true, y_pred, y_prob: Optional[np.ndarray], average: str) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "roc_auc": np.nan,
    }
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                proba_1 = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                out["roc_auc"] = roc_auc_score(y_true, proba_1)
            else:
                out["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            out["roc_auc"] = np.nan
    return out

def rc_none(summary: pd.DataFrame, main_metric: str) -> pd.DataFrame:
    base = summary[summary["Binning"] == "none"][
        ["Noise", "Feature Selection", "Classifier", "Features Used", f"{main_metric} mean"]
    ].rename(columns={f"{main_metric} mean": "BASE_mean"})
    joined = summary.merge(
        base,
        on=["Noise", "Feature Selection", "Classifier", "Features Used"],
        how="left",
        validate="many_to_one"
    )
    def _rc(row):
        denom = row["BASE_mean"]
        num = row.get(f"{main_metric} mean", np.nan)
        if pd.isna(denom) or denom == 0:
            return np.nan
        return (num - denom) / denom
    joined["RC_none"] = joined.apply(_rc, axis=1)
    return joined
