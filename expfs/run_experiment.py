# expfs/run_experiment.py

import sys
import yaml
import pandas as pd
from .datasets import load_dataset
from .pipeline import FeatureSelectionPipeline
from .constants import PipelineConfig
from .hparams import load_hparams


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]
    df = load_dataset(dataset_name)
    target_column = config["target_column"]

    noise_levels = tuple(config.get("noise_levels", [0.0, 0.1, 0.25, 0.5, 1.0]))
    binning_methods = tuple(config["binning_methods"])
    feature_methods = tuple(config["feature_methods"])
    classifiers = tuple(config["classifiers"])

    valid_binning = {"none", "static", "quantile", "kmeans"}
    valid_features = {"frfs", "elastic_net", "rf", "xgb"}
    valid_classifiers = {"random_forest", "xgboost", "svm", "logistic_regression"}
    if not set(binning_methods).issubset(valid_binning):
        raise ValueError(f"Invalid binning_methods: {binning_methods}")
    if not set(feature_methods).issubset(valid_features):
        raise ValueError(f"Invalid feature_methods: {feature_methods}")
    if not set(classifiers).issubset(valid_classifiers):
        raise ValueError(f"Invalid classifiers: {classifiers}")

    feature_counts = list(config.get("feature_counts", [5, 10, 15]))
    k_folds = int(config.get("k_folds", 5))
    add_noise_before_scaling = bool(config.get("add_noise_before_scaling", False))
    stream_summary_csv = config.get("stream_summary_csv")  # opcjonalne
    hparams_dir = config.get("hparams_dir", "hparams")

    hparams = load_hparams(dataset_name, base_dir=hparams_dir)

    pipe = FeatureSelectionPipeline(df, target_column=target_column, hparams=hparams)
    pipe.available_config = PipelineConfig(
        noise_levels=noise_levels,
        binning_methods=binning_methods,
        feature_methods=feature_methods,
        classifiers=classifiers,
    )

    summary = pipe.run_all_configurations(
        feature_counts=feature_counts,
        k_folds=k_folds,
        add_noise_before_scaling=add_noise_before_scaling,
        stream_summary_csv=stream_summary_csv,
    )

    if not isinstance(summary, pd.DataFrame):
        raise RuntimeError("Pipeline nie zwrócił poprawnego DataFrame 'summary'.")

    out_csv = config.get("output_csv", "results_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"✅ Wyniki (summary) zapisane do: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Użycie: python -m expfs.run_experiment configs/bcw.yaml")
        sys.exit(1)
    main(sys.argv[1])
