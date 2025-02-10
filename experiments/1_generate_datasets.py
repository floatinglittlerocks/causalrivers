import hydra

import pickle
from omegaconf import DictConfig
import networkx as nx
import os
import sys
import numpy as np
import random

sys.path.append("..")
from tools.graph_sampling_tools import get_all_subgraphs, get_all_sink_cases,combine_far_apart, get_longest_path,check_corr_character, add_one_random_node

# Here we generate all subsampling strategies that we evaluate and some additional ones.
# Additional ones might be added according to need.

# - Random connected
# - subselection with high elevation (not used) or short distance between nodes (used for paper), .
# - All to one node (Sink) ( Not used for paper).
# - All in a line (Causal Ordering, not used for paper)
# - One random node + connected graph (used for paper)
# - Selection based on correlation peaks.


@hydra.main(
    version_base=None, config_path="config", config_name="data_sampling.yaml"
)
def main(cfg: DictConfig):

    G = pickle.load(open(cfg.test_G_path, "rb"))
    finetune_data = pickle.load(open(cfg.train_G_path, "rb"))
    flood_G = pickle.load(open(cfg.flood_G_path, "rb"))


    print("Nodes to sample from: " + str(len(G.nodes)))
    print("Edges available: " + str(len(G.edges)))
    print("Nodes to finetune with: " + str(len(finetune_data.nodes)))
    print("Edges for finetuning: " + str(len(finetune_data.edges)))
    print("Nodes flood area: " + str(len(flood_G.nodes)))
    print("Edges flood area: " + str(len(flood_G.edges)))

    if cfg.which == "ALL":
        which = ["debug_set", "random","1_random", "root_cause","confounder", "close", "model_fit", "weather",] #"corr_subselection"
    else:
        which = [cfg.which]

    ##### Sampling strategies
    for version in which:
        print(version)


        if not os.path.exists(cfg.save_path + version + "_" + str(cfg.n_vars)):
            os.mkdir(cfg.save_path + version + "_" + str(cfg.n_vars))

        if version == "random":
            print("Generating random samples...")
            candidates_test = get_all_subgraphs(G, n_vars=cfg.n_vars)
            candidates_train = get_all_subgraphs(finetune_data, n_vars=cfg.n_vars)
            candidates_flood = get_all_subgraphs(flood_G, n_vars=cfg.n_vars)


        elif version == "1_random":
            print("Generating random samples with one disconnected...")
            candidates_test = get_all_subgraphs(G, n_vars=cfg.n_vars -1)
            candidates_train = get_all_subgraphs(finetune_data, n_vars=cfg.n_vars -1)
            candidates_test = add_one_random_node(G, candidates_test)
            candidates_train = add_one_random_node(finetune_data, candidates_train)
            
        elif version == "debug_set":
            print("Generating debug samples...")
            candidates_test = get_all_subgraphs(G, n_vars=cfg.n_vars)
            candidates_train = get_all_subgraphs(finetune_data, n_vars=cfg.n_vars)
            candidates_test = candidates_test[:5]
            candidates_train = candidates_train[:5]

        elif version == "root_cause":
            print("Root cause examples...")
            candidates_test = get_all_subgraphs(G, n_vars=cfg.n_vars)
            candidates_train = get_all_subgraphs(finetune_data, n_vars=cfg.n_vars)
            candidates_flood = get_all_subgraphs(flood_G, n_vars=cfg.n_vars)
            candidates_test = [x for x in candidates_test if nx.dag_longest_path_length(G.subgraph(x)) == (cfg.n_vars-1)]
            candidates_train = [x for x in candidates_train if nx.dag_longest_path_length(finetune_data.subgraph(x)) == (cfg.n_vars-1)]
            candidates_flood = [x for x in candidates_flood if nx.dag_longest_path_length(flood_G.subgraph(x)) == (cfg.n_vars-1)]

        elif version == "disjoint":
            print("Disjoint examples...")
            print("subset size:" +  str(int(cfg.n_vars/2)))
            candidates_test = get_all_subgraphs(G, n_vars=int(cfg.n_vars/2))
            candidates_train = get_all_subgraphs(finetune_data, n_vars=int(cfg.n_vars/2))
            candidates_test = combine_far_apart(G, candidates_test)
            candidates_train = combine_far_apart(finetune_data, candidates_train)

        elif version == "sink": # Not used
            print("Selecting sink examples...")
            candidates_test = get_all_sink_cases(G,n_vars=cfg.n_vars,restrict=cfg.g_per_sink)
            candidates_train = get_all_sink_cases(finetune_data,n_vars=cfg.n_vars,restrict=cfg.g_per_sink)

        elif version == "confounder":
            print("Selecting     confounder examples...")
            # TODO Not that elegant...
            conf = [x for x in G.nodes if len(list(G.successors(x))) > 1]
            potential_list = get_all_subgraphs(G,n_vars=cfg.n_vars)
            candidates_test = []
            for con in conf:
                candidates = [x for x in potential_list if con in x]
                for succ in list(G.successors(con)):
                    candidates =  [x for x in candidates if succ in x]
                # remove confounder from set
                candidates_test.append([tuple([y for y in x]) for x in candidates])
            candidates_test = [item for sublist in candidates_test for item in sublist]

            conf = [x for x in finetune_data.nodes if len(list(finetune_data.successors(x))) > 1]
            potential_list = get_all_subgraphs(finetune_data,n_vars=cfg.n_vars)
            candidates_train = []
            for con in conf:
                candidates = [x for x in potential_list if con in x]
                for succ in list(finetune_data.successors(con)):
                    candidates =  [x for x in candidates if succ in x]
                # remove confounder from set
                candidates_train.append([tuple([y for y in x]) for x in candidates])
            candidates_train = [item for sublist in candidates_train for item in sublist]
           

        elif version == "close":
            print("Generating examples with low distance...")
            candidates_train = get_all_subgraphs(finetune_data,n_vars=cfg.n_vars)
            short_enough = []
            # We use coordinates here as the distance all over the place unfortunately. (euclidean is way off often)
            # this strategy return inf if some edges have no specified links. 
            # Therefore it only return graphs which are garantueed to be under the threshold
            for x in candidates_train:
                if get_longest_path(nx.subgraph(finetune_data,x), measure=cfg.dist_measure) < cfg.max_distance:
                    short_enough.append(x)
            candidates_train = short_enough

            candidates_test = get_all_subgraphs(G,n_vars=cfg.n_vars)
            short_enough = []
            for x in candidates_test:
                if get_longest_path(nx.subgraph(G,x),measure=cfg.dist_measure) < cfg.max_distance:
                    short_enough.append(x)
            candidates_test = short_enough


        elif version == "corr_subselection":
            print("Generating examples with specified correlational characteristics...")
            candidates_train = get_all_subgraphs(finetune_data,n_vars=cfg.n_vars)
            short_enough = []
            # this strategy return inf if some edges have no specified links. 
            # Therefore it only return graphs which are garantueed to be under the threshold
            for x in candidates_train:
                if check_corr_character(nx.subgraph(finetune_data,x), cfg.corr_character):
                    short_enough.append(x)
            candidates_train = short_enough

            candidates_test = get_all_subgraphs(G,n_vars=cfg.n_vars)
            for x in candidates_test:
                if check_corr_character(nx.subgraph(G,x), cfg.corr_character):
                    short_enough.append(x)
            candidates_test = short_enough
    
        else:
            print("VERSION UNKNOWN")
            break

        print("Number of subgraphs: " + str(len(candidates_test)))
        print("Number of subgraphs for finetuning: " + str(len(candidates_train)))
        #print("Number of subgraphs for flood area: " + str(len(candidates_flood)))

        if version == "synthetic":
            pickle.dump(
            candidates_test,
            open(cfg.save_path + version + "_" + str(cfg.n_vars) + "/test.p", "wb"),
            )
            pickle.dump(
                candidates_train,
                open(cfg.save_path + version + "_" + str(cfg.n_vars) + "/train.p", "wb"),
            )
        else:
            pickle.dump(
                [nx.subgraph(G, x).copy() for x in candidates_test],
                open(cfg.save_path + version + "_" + str(cfg.n_vars) + "/test.p", "wb"),
            )
            # and for the training data:
            pickle.dump(
                [nx.subgraph(finetune_data, x).copy() for x in candidates_train],
                open(cfg.save_path + version + "_" + str(cfg.n_vars) + "/train.p", "wb"),
            )
            #TODO IMPLEMENT FOR ALL
            print(len(candidates_flood))
            pickle.dump(
               [nx.subgraph(flood_G, x).copy() for x in candidates_flood],
               open(cfg.save_path + version + "_" + str(cfg.n_vars) + "/flood_G.p", "wb"),
           )







if __name__ == "__main__":

    main()
# We use the best performing checkpoints from all std runs here. They can be downloaded here:
