# expfs/pipeline.py

from __future__ import annotations
import itertools
import os
from typing import List, Optional, Dict
import pandas as pd
import time

from .constants import PipelineConfig, BINNING_GRID
from .experiment import run_single_configuration


class FeatureSelectionPipeline:

    def __init__(self, dataframe: pd.DataFrame, target_column: str, random_state: int = 42, hparams: Optional[Dict] = None):
        self.df = dataframe.copy()
        self.target_column = target_column
        self.rs = random_state
        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]
        self.available_config = PipelineConfig()
        self.hparams = hparams or {}

    @staticmethod
    def _append_df(path: str, df: pd.DataFrame):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = not os.path.exists(path)
        df.to_csv(path, mode="a", header=header, index=False)

    def run_all_configurations(
        self,
        feature_counts: List[int] = (5, 10, 15),
        k_folds: int = 5,
        add_noise_before_scaling: bool = False,
        stream_summary_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        rows = []

        for noise, feat_method, k in itertools.product(
            self.available_config.noise_levels,
            self.available_config.feature_methods,
            feature_counts,
        ):
            for binning in self.available_config.binning_methods:
                for bins in BINNING_GRID[binning]:
                    start_time = time.time()
                    clf_rows = run_single_configuration(
                        self.X, self.y,
                        noise=noise,
                        binning=binning,
                        bins_value=bins,
                        feature_method=feat_method,
                        k_features=k,
                        classifiers=self.available_config.classifiers,
                        k_folds=k_folds,
                        random_state=self.rs,
                        add_noise_before_scaling=add_noise_before_scaling,
                        hparams=self.hparams
                    )
                    elapsed = time.time() - start_time
                    print(
                        f"config: Noise={noise}, Binning={binning}, Bins={bins}, "
                        f"FeatureSel={feat_method}, k={k} → time: {elapsed:.2f}s"
                    )
                    for r in clf_rows:
                        row = {
                            "Noise": noise,
                            "Binning": binning,
                            "Bins": bins if bins is not None else "none",
                            "Feature Selection": feat_method,
                            "Features Used": k,
                            **r,
                        }
                        rows.append(row)
                        if stream_summary_csv:
                            self._append_df(stream_summary_csv, pd.DataFrame([row]))
                            

        summary = pd.DataFrame(rows)
        return summary
