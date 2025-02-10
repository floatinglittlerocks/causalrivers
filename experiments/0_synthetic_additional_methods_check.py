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

@hydra.main(version_base=None, config_path="config", config_name="additional.yaml")
def main(cfg: DictConfig):

    start = datetime.datetime.now()

    print("Loading synth data...")

    # handle different envs as all of these libraries are poorely maintained we need multiple envs.
    if cfg.method_hps.method == "pcmci":
        from additional_methods.pcmci import pcmci_baseline as baseline_func
    elif cfg.method_hps.method == "cp":
        from additional_methods.causal_pretraining import causal_pretraining_baseline  as baseline_func
    elif cfg.method_hps.method == "dynotears":
        from additional_methods.dynotears import dynotears_baseline  as baseline_func
    elif cfg.method_hps.method == "varlingam":    
        from additional_methods.baselines import varlingam_baseline  as baseline_func
    elif cfg.method_hps.method == "cdmi":    
        from additional_methods.cdmi import cdmi_baseline  as baseline_func
    else:
        raise ValueError("Invalid method")
    

    X = np.load("resources/linear_base/X.npy")
    Y = np.load("resources/linear_base/Y.npy")


    preds = []

    if  cfg.method_hps.method == "cp":
        # quick fix to bring data into river format
        X = pd.DataFrame(X.reshape((-1,600)), index=np.arange(300).astype(str)).T
        Y_names = np.arange(300).reshape(100,3)
        preds = baseline_func(X,Y_names,cfg.method_hps)

    else:
        for n,sample in enumerate(X):
            print(n, len(X))
            preds.append(baseline_func(pd.DataFrame(sample.T),cfg.method_hps))
        

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


if __name__ == "__main__":
    main()