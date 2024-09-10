import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

def calculate_mutual_information_matrix(data):
    mi_matrix = pd.DataFrame(np.zeros((len(data.columns), len(data.columns))), columns=data.columns, index=data.columns)
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != col2:
                mi_matrix.loc[col1, col2] = mutual_info_score(data[col1], data[col2])
    return mi_matrix

def plot_mutual_information_heatmap(mi_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(mi_matrix, annot=True, cmap="viridis")
    plt.title("Mutual Information Heatmap")
    plt.show()

def visualize_network_with_weights(graph, mi_matrix):
    G = nx.DiGraph(graph.edges())
    weights = [mi_matrix.loc[u, v] for u, v in G.edges()]
    max_weight = max(weights)
    min_weight = min(weights)
    norm_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
    cmap = plt.cm.viridis
    edge_colors = [cmap(w) for w in norm_weights]

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, node_color='lightgreen')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), ax=ax, edge_color=edge_colors, width=2, arrows=True, arrowsize=20)

    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='vertical')
    plt.show()
