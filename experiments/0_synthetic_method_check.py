import sys
import pickle
import numpy as np
import hydra
from omegaconf import DictConfig

from tools.baseline_methods import remove_edges_via_mean_values, cross_correlation_for_causal_discovery, combo_baseline, var_baseline
from tools.tools import benchmarking, score, preprocessing_pipeline
import pandas as pd
import os
import pickle
from omegaconf import OmegaConf
import datetime

# Example script to benchmark the baseline methods.

@hydra.main(version_base=None, config_path="config", config_name="predict_single.yaml")
def main(cfg: DictConfig):

    start = datetime.datetime.now()

    print("Loading synth data...")



    X = np.load("resources/linear_base/X.npy")
    Y = np.load("resources/linear_base/Y.npy")

    preds = []
    for n,sample in enumerate(X):
        print(n, len(X))
        if cfg.method_hps.method == "reverse_physical":
            preds.append(remove_edges_via_mean_values(pd.DataFrame(sample.T),cfg.method_hps))
        elif cfg.method_hps.method == "cross_correlation":
            preds.append(cross_correlation_for_causal_discovery(pd.DataFrame(sample.T),cfg.method_hps))
        elif cfg.method_hps.method == "combo":
            preds.append(combo_baseline(pd.DataFrame(sample.T),cfg.method_hps))
        elif cfg.method_hps.method == "var":
            preds.append(var_baseline(pd.DataFrame(sample.T),cfg.method_hps))



        else:
            raise ValueError("Invalid method")

    preds = np.array(preds)
    out = score(preds,Y.sum(axis=-1).astype(bool),cfg)
    print(out)
    if cfg.save_full_out:
        
        # make folder with naming
        p = cfg.save_path  + cfg.method_hps.method + "_" + "synthetic"
        if not os.path.exists(p):
            os.makedirs(p)
        now = datetime.datetime.now()
        inner_p = cfg.save_path  + cfg.method_hps.method + "_" + "synthetic" + "/" + str(now)[:24]
        os.makedirs(inner_p)
        out.to_csv(inner_p + "/scoring.csv")
        stop_time = datetime.datetime.now() -start
        pd.DataFrame([stop_time], columns=["runtime"]).to_csv(inner_p + "/runtime.csv")# dumps to file:
        with open(inner_p + "/config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
        pickle.dump(preds, open(inner_p + "/preds.p", "wb"))
    print(stop_time)
    print(out)

if __name__ == "__main__":
    main()