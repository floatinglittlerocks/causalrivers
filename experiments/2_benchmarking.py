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
    print("Loading data...")
    test_data = pickle.load(open(cfg.sample_path + cfg.ds_name + "/" +  cfg.eval_set + ".p", "rb"))


    if cfg.restrict_to:
        test_data = test_data[:cfg.restrict_to]
    # We dont use train as methods do not require finetuning.
    #train = pickle.load(open(cfg.sample_path + cfg.ds_name + "/train.p", "rb"))

    Y,Y_names,X = preprocessing_pipeline(test_data,cfg)

    if cfg.method_hps.method == "reverse_physical":
        preds = benchmarking(Y_names,X,cfg,remove_edges_via_mean_values)
    elif cfg.method_hps.method == "cross_correlation":
        preds = benchmarking(Y_names,X,cfg,cross_correlation_for_causal_discovery)
    elif cfg.method_hps.method == "combo":
        preds =  benchmarking(Y_names,X,cfg,combo_baseline)
    elif cfg.method_hps.method == "var":
        preds =  benchmarking(Y_names,X,cfg,var_baseline)
    else:
        raise ValueError("Invalid method")
    out = score(preds,Y,cfg)

    if cfg.save_full_out:
        # make folder with naming
        p = cfg.save_path  + cfg.method_hps.method + "_" + cfg.ds_name
        if not os.path.exists(p):
            os.makedirs(p)


        now = datetime.datetime.now()
        inner_p = cfg.save_path  + cfg.method_hps.method + "_" + cfg.ds_name + "/" + str(now)[:24]
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