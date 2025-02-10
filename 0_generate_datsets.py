import os
import pickle
import sys

import hydra
import networkx as nx
from omegaconf import DictConfig

sys.path.append("..")
from tools.graph_sampling_tools import (add_one_random_node,
                                        check_corr_character,
                                        combine_far_apart, get_all_sink_cases,
                                        get_all_subgraphs, get_longest_path,
                                        select_confounder_samples)

# Here we generate all subsampling strategies that we evaluate and some additional ones.
# Additional ones might be added according to need.

# - Random connected
# - Confounding
# - subselection with high elevation (not used) or short distance between nodes (used for paper), .
# - All to one node (Sink) ( Not used for paper).
# - All in a line (Causal Ordering
# - One random node + connected graph
# - Disjoint groups


@hydra.main(
    version_base=None, config_path="config", config_name="data_sampling.yaml"
)
def main(cfg: DictConfig):

    east_G = pickle.load(open(cfg.test_G_path, "rb"))
    bav_G = pickle.load(open(cfg.train_G_path, "rb"))
    flood_G = pickle.load(open(cfg.flood_G_path, "rb"))


    print("Nodes in East G: " + str(len(east_G.nodes)))
    print("Edges in East G: " + str(len(east_G.edges)))
    print("Nodes in Bav G: " + str(len(bav_G.nodes)))
    print("Edges in Bav G: " + str(len(bav_G.edges)))
    print("Nodes flood area: " + str(len(flood_G.nodes)))
    print("Edges flood area: " + str(len(flood_G.edges)))


    if cfg.which == "ALL":
        to_generate = []
        for x in [3,5]:
            for y in ["debug_set", "random","1_random", "root_cause","confounder", "close",]:
                to_generate.append((y,x))
        to_generate.append(("disjoint", 10))
    else:
        to_generate = [(cfg.which, cfg.n_vars)]


    print(to_generate)
    ##### Sampling strategies
    for structure, n_vars in to_generate:
        if not os.path.exists(cfg.save_path + structure + "_" + str(n_vars)):
            os.mkdir(cfg.save_path + structure + "_" + str(n_vars))

        if structure == "debug_set":
            print("Generating debug samples...")
            east, bav, flood = (get_all_subgraphs(x, n_vars=n_vars)[:5] for x in [east_G, bav_G, flood_G])

        elif structure == "random":
            print("Generating random samples...")
            east, bav, flood = (get_all_subgraphs(x, n_vars=n_vars) for x in [east_G, bav_G, flood_G])

        elif structure == "1_random":
            print("Generating random samples with one disconnected...")
            east, bav, flood= (get_all_subgraphs(x, n_vars=n_vars-1) for x in [east_G, bav_G, flood_G])
            east = add_one_random_node(east_G, east)
            bav = add_one_random_node(bav_G, bav)
            # There are no disconnected nodes here so we skip it for this set.
            flood = []

        elif structure == "root_cause":
            print("Root cause examples...")
            east, bav, flood= (get_all_subgraphs(x, n_vars=n_vars) for x in [east_G, bav_G, flood_G])
            east = [x for x in east if nx.dag_longest_path_length(east_G.subgraph(x)) == (n_vars-1)]
            bav = [x for x in bav if nx.dag_longest_path_length(bav_G.subgraph(x)) == (n_vars-1)]
            flood = [x for x in flood if nx.dag_longest_path_length(flood_G.subgraph(x)) == (n_vars-1)]

        elif structure == "close":
            print("Generating examples with low distance...")
            east, bav, flood= (get_all_subgraphs(x, n_vars=n_vars) for x in [east_G, bav_G, flood_G])
            east = [x for x in east if get_longest_path(nx.subgraph(east_G,x), measure=cfg.dist_measure) < cfg.max_distance]
            bav = [x for x in bav if get_longest_path(nx.subgraph(bav_G,x), measure=cfg.dist_measure) < cfg.max_distance]
            flood = [x for x in flood if get_longest_path(nx.subgraph(flood_G,x), measure=cfg.dist_measure) < cfg.max_distance]

        elif structure == "confounder":
            print("Generating confounder examples...")
            east = select_confounder_samples(east_G,cfg)
            bav = select_confounder_samples(bav_G,cfg)
            flood = select_confounder_samples(flood_G,cfg)

        elif structure == "disjoint":
            print("Disjoint examples...")
            print("subset size:" +  str(int(n_vars/2)))
            east, bav, flood = (get_all_subgraphs(x, n_vars=int(n_vars/2)) for x in [east_G, bav_G, flood_G])
            # This is slow.
            east = combine_far_apart(east_G, east)
            bav = combine_far_apart(bav_G, bav)
            flood = combine_far_apart(flood_G, flood)

        # NOT USED IN THE PAPER but works.
        elif structure == "sink": # Not used
            print("Selecting sink examples...")
            east, bav, flood= (get_all_sink_cases(x,n_vars=n_vars,restrict=cfg.g_per_sink) for x in [east_G, bav_G, flood_G])

        #TODO Broken because it requires the lag information on the edges. FIX
        # elif version == "corr_subselection":
        #     print("Generating examples with specified correlational characteristics...")
        #     candidates_train = get_all_subgraphs(finetune_data,n_vars=cfg.n_vars)
        #     short_enough = []
        #     # this strategy return inf if some edges have no specified links. 
        #     # Therefore it only return graphs which are garantueed to be under the threshold
        #     for x in candidates_train:
        #         if check_corr_character(nx.subgraph(finetune_data,x), cfg.corr_character):
        #             short_enough.append(x)
        #     candidates_train = short_enough

        #     candidates_test = get_all_subgraphs(G,n_vars=cfg.n_vars)
        #     for x in candidates_test:
        #         if check_corr_character(nx.subgraph(G,x), cfg.corr_character):
        #             short_enough.append(x)
        #     candidates_test = short_enough
    
        else:
            print("VERSION UNKNOWN")
            break

        print("Number of subgraphs: " + str(len(east)))
        print("Number of subgraphs for finetuning: " + str(len(bav)))
        print("Number of subgraphs for flood area: " + str(len(flood)))

        pickle.dump(
            [nx.subgraph(east_G, x).copy() for x in east],
            open(cfg.save_path + structure + "_" + str(n_vars) + "/east.p", "wb"),
        )
        pickle.dump(
            [nx.subgraph(bav_G, x).copy() for x in bav],
            open(cfg.save_path + structure + "_" + str(n_vars) + "/bav.p", "wb"),
        )
        pickle.dump(
            [nx.subgraph(flood_G, x).copy() for x in flood],
            open(cfg.save_path + structure + "_" + str(n_vars) + "/flood.p", "wb"),
        )

if __name__ == "__main__":
    main()