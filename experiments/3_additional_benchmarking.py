import sys
import pickle
import numpy as np
import hydra
from omegaconf import DictConfig
import sys
from tools.tools import benchmarking, score, preprocessing_pipeline
import pandas as pd
import os
import pickle
from omegaconf import OmegaConf
import datetime

# Example script to benchmark the baseline methods.

@hydra.main(version_base=None, config_path="config", config_name="additional.yaml")
def main(cfg: DictConfig):

    
    print(cfg)
    start = datetime.datetime.now()
    print("Loading data...")


    test_data = pickle.load(open(cfg.sample_path + cfg.ds_name + "/" +  cfg.eval_set + ".p", "rb"))


    if cfg.restrict_to >= 0:
        test_data = test_data[cfg.restrict_to :cfg.restrict_to+1]
    # We dont use train as methods do not require finetuning.
    #train = pickle.load(open(cfg.sample_path + cfg.ds_name + "/train.p", "rb"))

    Y,Y_names,X = preprocessing_pipeline(test_data,cfg)
    

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
    


    if cfg.method_hps.method == "cp":
        # We dont want to load the model here for each time so the baseline func handels everything.
        preds = baseline_func(X,Y_names,cfg.method_hps)
    else:    
        preds = benchmarking(Y_names,X,cfg,baseline_func)

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