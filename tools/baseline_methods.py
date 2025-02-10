import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR


def make_human_readable(out,d):
    out = pd.DataFrame(out, columns=d.columns, index=d.columns)
    out = pd.concat(
        [pd.concat([out], keys=["Cause"], axis=1)], keys=["Effect"]
        )
    return out


def summary_transform(pred, cfg):
    if cfg.map_to_summary_graph == "max":
        prediction = pred.max(axis=2)
    elif cfg.map_to_summary_graph == "mean":
        prediction = pred.mean(axis=2)
    return prediction

def calc_lagged_cross_corr(df, max_lag=500, grain=1):
    """
    return Crosscorrelation items for each variables with each variable for each lag. 

    Return format: Stable, moved, lag
    Importantly: If the peak is above the max_lag, 
    the cause likely goes from  stable to moved and the other way arround.
    """
    product = []
    for column in df.columns:
        stack = []
        for x in range(-max_lag,max_lag+1, grain):
            stack.append(df.shift(-x).corrwith(df[column], axis=0).values.T)
        product.append(stack)
    return np.array(product).swapaxes(1,2)


def calc_stepwise_cross_corr(df,interval=2880, max_lag=750):
    product = []
    for step in range(0,len(df), interval):
        product.append(calc_lagged_cross_corr(df[step:step+interval], max_lag=max_lag, grain=1))
    return np.array(product)


def remove_edges_via_mean_values(d, cfg, human_readable=False, return_means=False):
    """
    As rivers cannot flow from big to small (in most cases), 
    a baseline strategy is to simply remove these impossible 
    links (along with the diagonal) and keep the rest as prediction 
    As an additional restriction, we can only keep one link per node. 
    Here the strategy would be either to choose the biggest or the closest one
    input: pd.Dataframe, reverse_physical: Small to big possible
    outtput: Effect, Cause (nxn)
    """
    m = np.expand_dims(d.mean().values,axis=1).repeat(len(d.columns),axis=1)
    out = m > m.T if cfg.reverse_physical else m < m.T

    #for each column we choose the item which is closest to itself or the biggest available.
    if (cfg.filter_mode == "next")  and (cfg.method != "combo"):
        print("using next filter")
        m[~out] = m.max()+1
        out = out * (m == np.expand_dims(m.min(axis=0),axis=0).repeat(out.shape[0],axis=0))
    elif (cfg.filter_mode  == "biggest") and (cfg.method != "combo"):
        print("using biggest filter")
        m[~out] = 0
        out = out * (m == np.expand_dims(m.max(axis=0),axis=0).repeat(out.shape[0],axis=0))

    if human_readable: 
        out = make_human_readable(out,d)
    return  (out, m) if return_means else out


def cross_correlation_for_causal_discovery(d, cfg, human_readable=False, return_corr=False):
    """ 
    Based on a cross correlation map, this decide which direction an error goes.
    """
    # we actually just have to calculate half here but keep it as it is cleaner.
    corr_map = calc_lagged_cross_corr(d,cfg.max_lag,1)
    out =  corr_map.argmax(axis=2).T > int(corr_map.shape[2] / 2)
    # restrict to the river with the higest cross correlation
    if (cfg.filter_mode == "corr") and cfg.method != "combo":
        print("using corr filter")
        peak = corr_map.max(axis=2)
        peak[~out] = 0
        out = out * (peak == np.expand_dims(
            peak.max(axis=0),axis=0).repeat(peak.shape[0],axis=0))

    if human_readable: 
        out = make_human_readable(out,d)
    return (out, corr_map) if return_corr else out


def combo_baseline(d, cfg, human_readable=False):
    """ 
    This combines both baseline rules.
    """

    cc_out, corr_map = cross_correlation_for_causal_discovery(d, cfg, return_corr=True)
    rphy_out, m = remove_edges_via_mean_values(d, cfg, return_means=True)
    out = cc_out * rphy_out

    if (cfg.filter_mode == "next"):
        print("using next filter")
        m[~out] = m.max()+1
        out = out * (m == np.expand_dims(m.min(axis=0),axis=0).repeat(out.shape[0],axis=0))
    elif cfg.filter_mode == "biggest":
        print("using biggest filter")
        m[~out] = 0
        out = out * (m == np.expand_dims(m.max(axis=0),axis=0).repeat(out.shape[0],axis=0))
    elif cfg.filter_mode == "corr":
        print("using biggest corr")
        peak = corr_map.max(axis=2)
        peak[~out] = 0
        out = out * (peak == np.expand_dims(
            peak.max(axis=0),axis=0).repeat(peak.shape[0],axis=0))
    if human_readable: 
        out = make_human_readable(out,d)
    return out


def var_baseline(d, cfg, human_readable=False):
    """ 
    Simple Granger based strategy that selects based on absolute parameter values.
    """
    n_vars = d.values.shape[-1]

    d.to_csv("check.csv")
    
    print(d)
    try: #For some samples, data is bricked to have constant TS. If this happens we predict 0
    # fit var with appropriate max lags
        res = VAR(d).fit(cfg.max_lag)
    # convert to bool and throw away intersection
    # !In the context of rivers, negative correlation do not really make sense.
    # I guess trying both is fair
        pred = res.params[1:]

        if cfg.var_absolute_values:
            pred = np.abs(pred)
        # reformat to original caused causing lag:
        # :) einsum needed i guess
        pred = np.stack(
            [pred.values[:, x].reshape(cfg.max_lag, n_vars).T for x in range(pred.shape[1])]
        )
        out = summary_transform(pred,cfg)
    except:
        out = np.zeros((n_vars,n_vars))

    if human_readable: 
        out = make_human_readable(out,d)
    return out