from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def make_classifier(name: str, random_state: int, params: Optional[Dict] = None):
    """
    Buduje klasyfikator z parametrami przekazanymi z YAML (per-dataset).
    """
    hp = params or {}
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 300)),
            max_depth=hp.get("max_depth", None),
            min_samples_leaf=int(hp.get("min_samples_leaf", 1)),
            n_jobs=int(hp.get("n_jobs", -1)),
            random_state=random_state
        )
    if name == "xgboost":
        return XGBClassifier(
            eval_metric=hp.get("eval_metric", "logloss"),
            random_state=random_state,
            n_estimators=int(hp.get("n_estimators", 400)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            max_depth=int(hp.get("max_depth", 6)),
            subsample=float(hp.get("subsample", 0.9)),
            colsample_bytree=float(hp.get("colsample_bytree", 0.9)),
            n_jobs=int(hp.get("n_jobs", -1)),
        )
    if name == "svm":
        return SVC(
            kernel=str(hp.get("kernel", "linear")),
            C=float(hp.get("C", 1.0)),
            probability=bool(hp.get("probability", True)),
            random_state=random_state
        )
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=int(hp.get("max_iter", 1000)),
            C=float(hp.get("C", 1.0)),
            random_state=random_state
        )
    raise ValueError(f"Invalid classifier: {name}")
