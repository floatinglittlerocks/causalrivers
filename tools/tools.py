import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
)
from statsmodels.tsa.api import VAR
from scipy import stats



def create_corr_maps_for_all_links(data,ds, ml=250,res = "30min"):
    stack = []
    for n,example in enumerate(ds):
        label = graph_to_label_tensor(example, human_readable=True)
        # Run some standard preprocessing steps
        sample_data = get_ts_data_for_graph(
            data[[str(x) for x in list(example.nodes)]],
            resolution=res,
            interpolate=False,
            normalize=True,
        )
        corr_peak  = [sample_data.shift(x).corrwith(sample_data[sample_data.columns[0]], axis=0).values[1] for x in range(-ml,ml+1)]
        stack.append(corr_peak)
    return stack





def preprocessing_pipeline(test,cfg, train=False):
    # Prep the ds:
    # This is not ram efficient but faster to process.
    Y = [graph_to_label_tensor(sample) for sample in test]
    Y_names = [ sorted(sample.nodes) for sample in test]
    # Select all columns that exist in the current dataset and load them.
    relevant_data_columns = list(set([item for sublist in [list(sample.nodes) for sample in test] for item in sublist]))
    relevant_data_columns = ["datetime"] + [str(x) for x in relevant_data_columns]
    data = pd.read_csv(cfg.train_data_path if train else cfg.data_path, index_col="datetime", usecols=relevant_data_columns)
    print("Prep ds...")
    X  = get_ts_data_for_graph(
        data,
        normalize=cfg.normalize,
        resolution=cfg.resolution_value,
        interpolate=cfg.interpolation,
        subset=(
            False
            if not cfg.window_data_year_value
            else [cfg.window_data_year_value, [cfg.window_data_month_value]] # TODO make this flexible for multiple months
        ),
        subsample=cfg.subsample_value,
    )

    return Y,Y_names,X

def benchmarking(Y_names,X,cfg, method_to_test):

    """
    Performs a full benchmarking of a method specified as method_to_test
    cfg should specify all remaining paramters (hydra config)
    method_to_test should be a function that wraps a specifc causal discovery strategy that takes a single
    sample data (specified as e.g. pandas frame) and a cfg for further parameters and returns a single matrix specifying the prediction (ExC)
    # Can be parallelized if necessary.
    """

    if cfg.parallel:
        preds = []
        print("Starting prediction...")
        executor = Parallel(n_jobs=cfg.parallel-1, backend="multiprocessing")
        tasks = (
            delayed(predict_causal_structure)(X[[str(x) for x in Y_names[sample]]], cfg, method_to_test, sample, len(Y_names))
            for sample in range(len(Y_names))
        )
        preds =  executor(tasks)
    else:
        preds = []
        for sample in range(len(Y_names)):
            preds.append(predict_causal_structure(X[[str(x) for x in Y_names[sample]]], cfg, method_to_test,sample, len(Y_names)))

    return preds

# Subsample only all subgraphs that contain only saxony and thuringia nodes: 
def filter_samples_based_on_properties(ds,G, selection=["T", "S"], prop="origin"):
    sub_ds = []
    for d in ds: 
        origin_check = set([G.nodes[x][prop]for x in d.nodes])
        if np.all([x in selection for x in origin_check]):
            sub_ds.append(d)
    return sub_ds

def graph_to_label_tensor(G_sample, human_readable=False):
    nodes = sorted(G_sample.nodes)
    labels = np.zeros((len(nodes), len(nodes)))

    for n, x in enumerate(nodes):
        for m, y in enumerate(nodes):
            if (x, y) in G_sample.edges:
                labels[m, n] = 1
    if human_readable:
        labels = pd.DataFrame(labels, columns=nodes, index=nodes)
        labels = pd.concat(
            [pd.concat([labels], keys=["Cause"], axis=1)], keys=["Effect"]
        )
        return labels
    else:
        return labels 

def load_sample(which, p = "resources/rivers_ts_east_germany.csv"):
    return pd.read_csv(p, index_col=0, usecols=["datetime"] + [str(x) for x in list(which.nodes)])

def get_ts_data_for_graph(
    data,
    resolution="2H",
    interpolate=True,
    subset=False,
    subsample=1,
    normalize=False,
):
    # Adjust resolution
    sample_data = data.copy() # dont change the original data
    sample_data["dt"] = pd.to_datetime(data.index).round(resolution).values

    sample_data = sample_data.groupby("dt").mean()
    # subsampling
    if subset:
        sample_data = sample_data.loc[
            (sample_data.index.month.isin(subset[1])) &
            (sample_data.index.year == subset[0])
        ]
    sample_data = sample_data.iloc[::subsample, :]
    if normalize:
        sample_data = (sample_data - sample_data.min()) / (
            sample_data.max() - sample_data.min()
        )
    if interpolate:
        sample_data = sample_data.interpolate()
    return sample_data

def predict_causal_structure(sample_prep, cfg,baseline_func, n=1, length=1):
    """"
    Simple wrapper that removes trailing nans if necessary
    """

    print("Sample " + str(n) + "/" + str(length))
    # remove training na

    check_trailing_nans = np.where(sample_prep.isnull().values.any(axis=1) == 0)[0]
    if not len(check_trailing_nans) == 0: # A ts is completely 0:
        sample_prep = sample_prep[check_trailing_nans.min() : check_trailing_nans.max()+1]
    else:
        sample_prep = sample_prep.fillna(value=0)
    assert sample_prep.isnull().sum().max() == 0, "Check nans!"

    # if cfg.subsample_strategy == "model_fit": #TODO FiNISH THIS HERE
    #     resids = []
    #     for x in range (0,len(sample_prep), cfg.var_test_window):
    #         try:
    #             resids.append(VAR(sample_prep.values[x:x+cfg.var_test_window]).fit(maxlags=cfg.method_hps.max_lag).resid)
    #         except:
    #             fail = np.zeros(sample_prep.values[x:x+cfg.var_test_window].shape)
    #             fail = np.inf
    #             resids.append(fail)
    #     lowest_error = np.inf
    #     sample_prep = None
    #     for resid in resids:
    #         print(resid) 
    #         print(resid.shape)
    #         res = stats.normaltest(resid)
    #         print(res.pvalue)


    #     lowest_error = np.argmin([np.mean(np.array(x) **2) for x in resids])
    #     sample_prep = sample_prep[lowest_error*cfg.var_test_window:(lowest_error+1)*cfg.var_test_window]
    # elif cfg.subsample_strategy == "weather":
    #     pass
    # else:
    #     pass

    return baseline_func(sample_prep,cfg.method_hps)


def remove_diagonal(T):
    # Takes in 3 dim tensor and removes diagonal of 2/3 dim.
    out = []
    for x in T:
        out.append(x[~np.eye(x.shape[0],dtype=bool)].reshape(x.shape[0],-1))
    return np.stack(out)


def max_accuracy(labs,preds):
    # ACCURACY MAX
        
    preds = preds.astype(float)
    print(preds.min(), preds.max())
    if preds.min() == preds.max():
        a = []
    else:
        a = list(np.arange(preds.min(), preds.max()+ preds.min(), (preds.max()- preds.min()) / 100)) # 100 steps
    possible_thresholds = [0] + a + [preds.max() + 1e-6]
    acc = [accuracy_score(labs, preds > thresh) for thresh in possible_thresholds]
    acc_thresh = possible_thresholds[np.argmax(acc)]
    acc_score = np.nanmax(acc)
    return acc_thresh,acc_score


def f1_max(labs,preds):
    # F1 MAX
    precision, recall, thresholds = precision_recall_curve(labs, preds)
    f1_scores = 2 * recall * precision / (recall + precision)
    f1_thresh = thresholds[np.argmax(f1_scores)]
    f1_score = np.nanmax(f1_scores)
    return f1_thresh,f1_score


def score(preds,labs,cfg, provided_name = None):

    # Calculates a number of metrics and returns a df holding them
    print("Scoring...")
    # We remove the diagonal as rivers are highly autocorrelated and the causal links here are not relevant.
    if cfg.remove_diagonal:
        labs = remove_diagonal(labs)
        preds = remove_diagonal(preds)
    else:
        labs = np.array(labs)
        preds = np.array(preds)
    # Individual scoring for each sample.

    f1_max_ind = []
    f1_thresh_ind = []
    accuracy_ind =  []
    accuracy_ind_thresh = []
    auroc_ind = []
    for x in range(len(labs)):
        if len(set(labs[x].flatten())) == 1:
            # not defined for empty samples
            # this can sometimes happen if a limited time window is chosen
            continue
        else:
            auroc_ind.append(roc_auc_score(y_true= labs[x].flatten(), y_score=preds[x].flatten()))
        f1_thresh,f1_score = f1_max(labs[x].flatten(), preds[x].flatten())
        f1_max_ind.append(f1_score)

        f1_thresh_ind.append(f1_thresh)
        acc_thresh,acc_score = max_accuracy(labs[x].flatten(), preds[x].flatten())
        accuracy_ind.append(acc_score)
        accuracy_ind_thresh.append(acc_thresh)

    f1_max_ind = np.array(f1_max_ind).mean()
    f1_thresh_ind = np.array(f1_thresh_ind).mean()
    accuracy_ind = np.array(accuracy_ind).mean()
    accuracy_ind_thresh = np.array(accuracy_ind_thresh).mean()
    auroc_ind = np.array(auroc_ind).mean()

    # Joint calculation
    labs = labs.flatten()
    preds = preds.flatten()

    # AUROC
    auroc = roc_auc_score(labs, preds)
    # F1 MAX

    f1_thresh,f1_score = f1_max(labs, preds)
    # ACCURACY MAX
    acc_thresh,acc_score = max_accuracy(labs,preds)

    null_model_auroc = roc_auc_score(labs, np.zeros(preds.shape))

    _, null_model_f1  = f1_max(labs, np.zeros(preds.shape))

    _, null_model_acc  = max_accuracy(labs, np.zeros(preds.shape))

    out = pd.DataFrame(
        [
            acc_thresh,
            acc_score,
            accuracy_ind_thresh,
            accuracy_ind,
            null_model_acc,
            f1_thresh,
            f1_score,
            f1_thresh_ind,
            f1_max_ind,
            null_model_f1,
            auroc,
            auroc_ind,
            null_model_auroc,
        ],
        columns=[provided_name] if provided_name else [cfg.method_hps.method  + "_" + cfg.ds_name] ,
        index=[
            "Max Acc thresh",
            "Max Acc",
            "Max individual Acc thresh",
            "Max individual Acc",
            "Null Acc",
            "Max F1 thresh",
            "Max F1",
            "Max individual F1 thresh",
            "Max individual F1",
            "Null F1",
            "AUROC",
            "Individual AUROC",
            "Null AUROC"
        ],
    )
    out.index.name = "Metric"
    return out

