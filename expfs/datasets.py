# expfs/datasets.py

from __future__ import annotations
import pandas as pd
from typing import Tuple
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml
import re


def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usuwa niedozwolone znaki w nazwach kolumn (np. dla XGBoost).
    - zamienia [, ], < na _
    - zamienia spacje na _
    - wymusza unikalność nazw
    """
    new_cols = []
    used = set()
    for c in df.columns:
        s = str(c)
        s = s.replace("[", "_").replace("]", "_").replace("<", "_")
        s = re.sub(r"\s+", "_", s)
        base = s
        i = 1
        while s in used:
            s = f"{base}_{i}"
            i += 1
        used.add(s)
        new_cols.append(s)
    df = df.copy()
    df.columns = new_cols
    return df


def _uci_to_dataframe(uci_obj) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(uci_obj.data.features)
    if isinstance(X.columns, pd.RangeIndex):
        X.columns = [f"f{i}" for i in range(X.shape[1])]
    X = X.apply(pd.to_numeric)

    y_df = pd.DataFrame(uci_obj.data.targets)
    y = y_df.iloc[:, 0]

    if y.dtype == object or str(y.dtype).startswith(("category", "string")):
        classes = sorted(pd.unique(y))
        if len(classes) == 2:
            mapping = {classes[0]: 0, classes[1]: 1}
            y = y.map(mapping)

    return X, y


def _openml_to_dataframe(name: str, version: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loader dla zbiorów z OpenML (tu używany tylko dla 'madelon').
    Zwraca X jako DataFrame i y jako Series (binarne mapowane do {0,1}).
    """
    ds = fetch_openml(name=name, version=version, as_frame=True)
    X = ds.data.copy()
    if isinstance(X.columns, pd.RangeIndex):
        X.columns = [f"f{i}" for i in range(X.shape[1])]
    X = X.apply(pd.to_numeric, errors="coerce")

    y = pd.Series(ds.target, name="target")

    if y.dtype == object or str(y.dtype).startswith(("category", "string")):
        y = y.astype(str).str.strip()
        uniques = set(y.unique())
        if uniques <= {"-1", "1"}:
            y = y.map({"-1": 0, "1": 1}).astype("int8")
        else:
            y = pd.Categorical(y).codes.astype("int32")
    else:
        uniques = set(pd.unique(y))
        if uniques <= {-1, 1}:
            y = (y > 0).astype("int8")
        else:
            y = y.astype("int32")

    return X, y


def load_dataset(name: str) -> pd.DataFrame:
    if name == "breast_cancer_wisconsin":
        uci = fetch_ucirepo(id=17)
        X, y = _uci_to_dataframe(uci)
    elif name == "spambase":
        uci = fetch_ucirepo(id=94)
        X, y = _uci_to_dataframe(uci)
    elif name == "sonar":
        uci = fetch_ucirepo(id=151)
        X, y = _uci_to_dataframe(uci)
    elif name == "musk_v2":
        uci = fetch_ucirepo(id=75)
        X, y = _uci_to_dataframe(uci)
    elif name == "madelon":
        X, y = _openml_to_dataframe(name="madelon", version=1)
    elif name == "isolet":
        uci = fetch_ucirepo(id=75)
        X, y = _uci_to_dataframe(uci)
    else:
        raise ValueError(f"Unsupported dataset: {name!r}")

    df = X.copy()
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="target")
    df["target"] = y.to_numpy()

    df = _sanitize_column_names(df)
    return df
