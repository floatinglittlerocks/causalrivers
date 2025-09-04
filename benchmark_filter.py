import hydra
from omegaconf import DictConfig
import datetime
import pandas as pd

from tools.tools import (
    load_joint_samples,
    benchmarking,
    standard_preprocessing,
    save_run,
    remove_seasonality,  # <- import the seasonality filter
)
from tools.scoring_tools import score


@hydra.main(version_base=None, config_path="config", config_name="benchmark.yaml")
def main(cfg: DictConfig):
    
    if cfg.method.name == "var":
        from tools.baseline_methods import var_baseline as cd_method
    else:
        print("SPECIFY AND LOAD YOUR OWN METHOD HERE")

    start = datetime.datetime.now()
    print(cfg)
    print("Loading data...")

    # Load data with optional standard preprocessing
    test_data, test_labels = load_joint_samples(
        cfg, preprocessing=standard_preprocessing if cfg.dt_preprocess else None
    )

    # Apply seasonality filter if enabled in config
    if getattr(cfg, "apply_seasonality_filter", False):
        print("Applying seasonality filter...")
        test_data = [
            pd.DataFrame(
                remove_seasonality(sample, fs=cfg.seasonality_fs, cutoff_days=cfg.seasonality_cutoff),
                index=sample.index,
                columns=sample.columns
            )
            for sample in test_data
        ]

    # Run causal discovery method
    print("Performing Causal Discovery...")
    preds = benchmarking(test_data, cfg, cd_method)

    # Score predictions
    out = score(
        preds,
        test_labels,
        remove_autoregressive=cfg.remove_diagonal,
        name=cfg.method.name,
    )
    stop_time = datetime.datetime.now() - start
    print(out)

    # Save results if desired
    if cfg.save_full_out:
        print("Saving...")
        save_run(out, stop_time, preds, cfg)

    print("Done", stop_time)


if __name__ == "__main__":
    main()