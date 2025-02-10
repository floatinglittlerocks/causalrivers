import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
import matplotlib as mpl
import numpy as np

def plot_current_state_of_graph(
    G,
    dpi=100,
    lim=(50.1, 54.8),
    limx=(9.65, 15.1),
    node_size=50,
    arrowsize=20,
    fs=(10, 10),
    font_size=1,
    save=False,
    river_map=0,
    ger_map=True,
    emphasize=[],
    label=True,
    autozoom=None,
    width=1,
    show_edge_origin=False,
    hardcode_colors = [],
    ger_path = "visualization/east_german_map.shp",
    river_path = "visualization/river_east_german_map.shp",
    extra_points = [],
    river_width=0.5,
    title="Rivers East Germany",
    pos = False,
    ax = False
):
#
    if not pos:
        pos = {x: np.flip(np.array(G.nodes[x]["p"][:2]).astype(float)) for x in G.nodes}

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=fs)

    if ger_map:
        fp = ger_path
        map_df2 = gpd.read_file(fp)
        map_df2.plot(color="green", ax=ax, alpha=0.3, linewidth=5, edgecolor="black")

    if river_map:
        fp = river_path
        map_df = gpd.read_file(fp)
        map_df.plot(color="blue", alpha=0.3, ax=ax, linewidth=river_width, edgecolor='blue')


    if hardcode_colors: 
        colors = hardcode_colors
    else:        
        colors = []
        for x in G.nodes:
            if x in emphasize:
                colors.append("black")
            else:
                colors.append(G.nodes[x]["c"])
    if show_edge_origin:
        cmap = mpl.colormaps['Set1']
        ege_base_colors = cmap(np.linspace(0, 1, 8))
        edge_colors = []
        for x in G.edges:
            edge_colors.append(tuple(ege_base_colors[G.edges[x]["origin"]]))
    nx.draw_networkx(
        G,
        pos= pos,
        with_labels=label,
        font_size=font_size,
        node_size=node_size,
        arrows=True,
        node_color=colors,
        arrowsize=arrowsize,
        edge_color= edge_colors if show_edge_origin else "black",
        width=width,
        ax=ax,
    )
    if autozoom:
        ax.set_xlim(
            min([pos[x][0] for x in pos.keys()]) - autozoom,
            max([pos[x][0] for x in pos.keys()]) + autozoom,
        )
        ax.set_ylim(
            min([pos[x][1] for x in pos.keys()]) - autozoom,
            max([pos[x][1] for x in pos.keys()]) + autozoom,
        )
    else:
        if lim:
            ax.set_ylim(lim[0], lim[1])
            ax.set_xlim(limx[0], limx[1])
        else:
            pass

    if save:

        plt.savefig("resources/" + save + "_G.png", dpi=dpi)  # , dpi= 500
        plt.close()

    ax.set_title(title)
    ax.set_frame_on(True)

    if len(extra_points):
        for ex in extra_points:
            ax.scatter(ex[0],ex[1],color="pink", s=100)
            ax.annotate(ex[2], (ex[0],ex[1]))

    plt.show()



def simple_sample_display(sample_data):
    fig, axs = plt.subplots(len(sample_data.columns),1)
    cmap = mpl.colormaps['plasma']
    # Take colors at regular intervals spanning the colormap.
    for n,s in enumerate(sample_data.columns):
        axs[n].set_ylabel(s, fontstyle="normal", fontsize=8,rotation=45)
        rgba = cmap(1/(n+1))
        axs[n].plot(sample_data[s].values, linewidth=2, color=rgba)
        axs[n].get_xaxis().set_ticks([])
        axs[n].get_yaxis().set_ticks([])
    axs[n].get_yaxis().set_ticks([])
    axs[n].set_xlabel("Timesteps")
    plt.show()